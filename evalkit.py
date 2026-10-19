"""
Shared evaluation utilities: metrics, per-RNA aggregation, threshold and
minimum-region-length selection, flexible-region calling, and resampling
statistics for comparing models.

Flexible region definition
  a nucleotide is flexible when the model flexibility score exceeds a fixed
  threshold T; consecutive flexible nucleotides are merged into regions and
  regions shorter than L_min are discarded. T and L_min are chosen once, on the
  internal validation split of the *training* chemistry, and then applied
  unchanged to every other dataset, so no test data influence the choice.
"""
import numpy as np
from scipy.stats import beta, norm, spearmanr, pearsonr
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, cohen_kappa_score,
                             brier_score_loss, jaccard_score)


# ------------------------------------------------------------------ metrics ---
def point_metrics(y_true, y_score, thr=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_hat = (y_score >= thr).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    uniq = len(np.unique(y_true)) > 1
    return dict(
        threshold=float(thr),
        accuracy=(tp + tn) / max(tp + tn + fp + fn, 1),
        sensitivity=sens, specificity=spec,
        precision=ppv, negative_predictive_value=npv,
        f1=2 * sens * ppv / max(sens + ppv, 1e-9),
        mcc=float(matthews_corrcoef(y_true, y_hat)) if len(np.unique(y_hat)) > 1 else 0.0,
        kappa=float(cohen_kappa_score(y_true, y_hat)) if len(np.unique(y_hat)) > 1 else 0.0,
        balanced_accuracy=0.5 * (sens + spec),
        tp=tp, tn=tn, fp=fp, fn=fn,
        auc=float(roc_auc_score(y_true, y_score)) if uniq else float("nan"),
        auprc=float(average_precision_score(y_true, y_score)) if uniq else float("nan"),
        brier=float(brier_score_loss(y_true, y_score)),
        prevalence=float(y_true.mean()),
    )


def measured_lengths(valid, lengths):
    """Per-RNA counts of measured positions.

    Per-RNA metrics are computed on a *masked* flat array (unmeasured
    nucleotides removed) but need the per-RNA block sizes of that masked array.
    Passing the un-masked RNA lengths would silently misalign every block after
    the first, so the masked lengths are derived here instead.
    """
    valid = np.asarray(valid)
    off = 0
    out = []
    for n in lengths:
        out.append(int(valid[off:off + n].sum()))
        off += n
    return np.array(out)


def per_rna_point_metrics(y_true, y_score, lengths, thr=0.5):
    """Per-RNA point metrics.

    AUROC needs both classes present, so it is collected only from informative
    hairpins; accuracy, sensitivity, specificity and F1 are collected from every
    hairpin with at least one measured nucleotide.
    """
    aucs, sents, specs, f1s, accs, mccs = [], [], [], [], [], []
    off = 0
    for n in lengths:
        t, s = np.asarray(y_true)[off:off + n], np.asarray(y_score)[off:off + n]
        off += n
        if len(t) == 0:
            continue
        m = point_metrics(t, s, thr)
        if len(np.unique(t)) > 1 and np.isfinite(m["auc"]):
            aucs.append(m["auc"])
        sents.append(m["sensitivity"])
        specs.append(m["specificity"])
        f1s.append(m["f1"])
        accs.append(m["accuracy"])
        mccs.append(m["mcc"])
    if not aucs:
        return dict(median_rna_auroc=float("nan"), n_rnas_auroc=0)
    q = lambda a: [float(np.percentile(a, 25)), float(np.percentile(a, 75))]
    return dict(
        median_rna_auroc=float(np.median(aucs)), iqr_rna_auroc=q(aucs),
        mean_rna_auroc=float(np.mean(aucs)),
        median_rna_sensitivity=float(np.median(sents)),
        median_rna_specificity=float(np.median(specs)),
        median_rna_f1=float(np.median(f1s)),
        median_rna_accuracy=float(np.median(accs)),
        median_rna_mcc=float(np.median(mccs)),
        n_rnas_auroc=len(aucs),
    )


def per_rna_correlations(y_true, y_score, lengths):
    """Within-hairpin Spearman: the per-RNA offset cannot inflate the value."""
    rhos, off = [], 0
    for n in lengths:
        t, s = y_true[off:off + n], y_score[off:off + n]
        off += n
        if len(t) < 10 or np.std(s) == 0:
            continue
        r = spearmanr(t, s).statistic
        if np.isfinite(r):
            rhos.append(float(r))
    return dict(median_rna_spearman=float(np.median(rhos)) if rhos else float("nan"),
                mean_rna_spearman=float(np.mean(rhos)) if rhos else float("nan"),
                frac_rna_rho_gt_half=float(np.mean(np.array(rhos) > 0.5)) if rhos else float("nan"),
                n_rnas=len(rhos))


def correlations(y_cont, y_score):
    y_cont, y_score = np.asarray(y_cont), np.asarray(y_score)
    ok = np.isfinite(y_cont) & np.isfinite(y_score)
    a, b = y_cont[ok], y_score[ok]
    if len(a) < 10 or np.std(b) == 0:
        return dict(spearman=float("nan"), pearson=float("nan"), n=int(len(a)))
    return dict(spearman=float(spearmanr(a, b).statistic),
                pearson=float(pearsonr(a, b).statistic), n=int(len(a)))


# ------------------------------------------------ threshold / length choice ---
def threshold_grid(n=400):
    return np.round(np.linspace(0.02, 0.98, n), 4)


def select_threshold(y_true, y_score, criterion="f1", grid=None):
    """Pick the score cut-off maximising a validation criterion.

    `f1` targets region calls, `youden` maximises sensitivity + specificity - 1,
    and `mcc` maximises the Matthews correlation. Ties resolve to the threshold
    closest to 0.5 so the choice does not drift to a degenerate extreme.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    grid = threshold_grid() if grid is None else grid
    best, best_t = -np.inf, 0.5
    for t in grid:
        m = point_metrics(y_true, y_score, t)
        v = {"f1": m["f1"], "youden": m["sensitivity"] + m["specificity"] - 1.0,
             "mcc": m["mcc"]}[criterion]
        if v > best + 1e-12 or (abs(v - best) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = v, float(t)
    return best_t, float(best)


def select_min_length(y_true, y_score, thr, lengths, grid=(2, 3, 4, 5, 6)):
    """Pick the minimum region length on validation, by mean region F1."""
    best, best_l = -np.inf, grid[0]
    for L in grid:
        m = region_level_scores(y_true, (y_score >= thr), lengths, min_len=L)
        v = m.get("mean_region_f1", 0.0)
        if v > best:
            best, best_l = v, L
    return int(best_l), float(best)


# ------------------------------------------------------------ region calling ---
def call_regions(score, thr, min_len=3, mask=None):
    """Merge consecutive above-threshold nucleotides into (start, end) regions."""
    s = np.asarray(score) >= thr
    if mask is not None:
        s = s & (np.asarray(mask) > 0)
    regions, i, n = [], 0, len(s)
    while i < n:
        if s[i]:
            j = i
            while j + 1 < n and s[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                regions.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return regions


def region_level_scores(y_true, called, lengths, min_len=3):
    """Compare called flexible regions with observed high-reactivity stretches.

    `called` and `y_true` are flat per-nucleotide vectors; observed regions are
    derived with the same merging rule as the calls. Scored per RNA and then
    averaged, so long hairpins do not dominate.
    """
    ious, f1s, n_pred, n_true, matched, prec, rec = [], [], [], [], [], [], []
    off = 0
    for n in lengths:
        p = np.asarray(called[off:off + n]) > 0
        t = np.asarray(y_true[off:off + n]) > 0
        off += n
        if not p.any() and not t.any():
            continue
        iou = float(jaccard_score(t, p, zero_division=0))
        ious.append(iou)
        f1s.append(2 * iou / (1 + iou) if (1 + iou) > 0 else 0.0)
        n_pred.append(int(p.sum()))
        n_true.append(int(t.sum()))
        obs = call_regions(t.astype(float), 0.5, min_len=min_len)
        pred = call_regions(p.astype(float), 0.5, min_len=min_len)
        # recall of observed stretches covered by at least half of a call
        cov = 0
        for a, b in obs:
            lo, hi = max(0, a), min(len(p), b)
            if hi > lo and p[lo:hi].sum() >= 0.5 * (hi - lo):
                cov += 1
        matched.append(cov / max(len(obs), 1))
        rec.append(cov / max(len(obs), 1))
        inter = int((p & t).sum())
        prec.append(inter / max(int(p.sum()), 1))
    if not ious:
        return {}
    return dict(
        median_region_jaccard=float(np.median(ious)),
        mean_region_jaccard=float(np.mean(ious)),
        median_region_f1=float(np.median(f1s)),
        mean_region_f1=float(np.mean(f1s)),
        median_called_nt_per_rna=float(np.median(n_pred)),
        median_observed_nt_per_rna=float(np.median(n_true)),
        median_observed_region_recall=float(np.median(matched)),
        mean_observed_region_recall=float(np.mean(matched)),
        median_region_precision=float(np.median(prec)),
        n_rnas=len(ious),
    )


# ------------------------------------------------------------------ resampling ---
def bootstrap_ci(y_true, y_score, statistic="auc", n_boot=1000, seed=0,
                 groups=None):
    """Percentile bootstrap CI, resampling groups (hairpins) when given."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    fn = {"auc": roc_auc_score, "auprc": average_precision_score}[statistic]
    if groups is None:
        idx = np.arange(len(y_true))
        blocks = [idx]
    else:
        groups = np.asarray(groups)
        order = np.argsort(groups, kind="stable")
        gs = groups[order]
        bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
        blocks = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        sel = np.concatenate([blocks[p] for p in pick])
        yt, ys = y_true[sel], y_score[sel]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(fn(yt, ys))
    if len(vals) < 20:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def delong_paired(y1, s1, y2, s2):
    """Two-sided DeLong test for two correlated ROC AUCs on identical items.

    Uses the structural-components estimator of DeLong et al. (1988). For each
    item i in the positives the component V10_i = mean over positives of
    psi(s_j, s_i) with psi(x, y) = I(x > y) + 0.5*I(x == y); and V01 likewise
    against the negatives (and the mirror images for the negatives). The AUC is
    the mean of V10 over the positives (equivalently of V01 over the
    negatives), and the paired variance is assembled from the sample variances
    of these components scaled by 1/n1 and 1/n0 plus their paired covariances.
    """
    y1, s1, y2, s2 = map(np.asarray, (y1, s1, y2, s2))
    a1, a2 = roc_auc_score(y1, s1), roc_auc_score(y2, s2)
    pos = np.where(y1 == 1)[0]
    neg = np.where(y1 == 0)[0]
    n1, n0 = len(pos), len(neg)
    if n1 < 2 or n0 < 2:
        return dict(auc1=float(a1), auc2=float(a2), z=float("nan"),
                    p=float("nan"), diff=float(a1 - a2))

    def comp(sp, sn, x):
        gt_p = (n1 - np.searchsorted(sp, x, side="right")) / n1
        le_p = np.searchsorted(sp, x, side="right") - np.searchsorted(sp, x, side="left")
        eq_p = le_p / n1
        gt_n = (n0 - np.searchsorted(sn, x, side="right")) / n0
        le_n = np.searchsorted(sn, x, side="right") - np.searchsorted(sn, x, side="left")
        eq_n = le_n / n0
        return gt_p + 0.5 * eq_p, gt_n + 0.5 * eq_n

    sp1, sn1 = np.sort(s1[pos]), np.sort(s1[neg])
    sp2, sn2 = np.sort(s2[pos]), np.sort(s2[neg])
    p10, _ = comp(sp1, sn1, s1[pos])
    p01, _ = comp(sp1, sn1, s1[neg])
    p20, _ = comp(sp2, sn2, s2[pos])
    p02, _ = comp(sp2, sn2, s2[neg])
    _, n01 = comp(sp1, sn1, s1[neg])
    _, n11 = comp(sp1, sn1, s1[pos])
    _, n02 = comp(sp2, sn2, s2[neg])
    _, n12 = comp(sp2, sn2, s2[pos])

    m = lambda v: v.mean()  # noqa: E731
    s10, s01 = np.var(p10, ddof=1), np.var(n01, ddof=1)
    s20, s02 = np.var(p20, ddof=1), np.var(n02, ddof=1)
    var_a1 = s10 / n1 + s01 / n0
    var_a2 = s20 / n1 + s02 / n0
    cov_pos = np.sum((p10 - m(p10)) * (p20 - m(p20))) / (n1 - 1)
    cov_neg = np.sum((n01 - m(n01)) * (n02 - m(n02))) / (n0 - 1)
    cov_a = cov_pos / n1 + cov_neg / n0
    var = var_a1 + var_a2 - 2 * cov_a
    if var <= 0:
        return dict(auc1=float(a1), auc2=float(a2), z=float("nan"),
                    p=float("nan"), diff=float(a1 - a2))
    z = (a1 - a2) / np.sqrt(var)
    return dict(auc1=float(a1), auc2=float(a2), diff=float(a1 - a2),
                var=float(var), z=float(z),
                p=float(2 * (1 - norm.cdf(abs(z)))))


def mcnemar(y1, p1, y2, p2):
    """Exact McNemar test on two classifiers' correctness at a shared threshold."""
    y1, p1, y2, p2 = map(np.asarray, (y1, p1, y2, p2))
    c1, c2 = (p1 == y1), (p2 == y2)
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    if b + c == 0:
        return dict(b=b, c=c, p=1.0)
    from scipy.stats import binom
    # exact two-sided: 2 * Pr(X <= min(b, c)) under X ~ Bin(b + c, 0.5)
    k = min(b, c)
    p = 2 * binom.cdf(k, b + c, 0.5)
    return dict(b=b, c=c, p=float(min(1.0, p)))


def median_ci(x, n_boot=2000, seed=0):
    """Percentile bootstrap CI for a median."""
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 5:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ----------------------------------------------------------------- curve data ---
def roc_pr_curves(y_true, y_score):
    """ROC and PR curve points for plotting."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y, s = y_true[order], y_score[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.concatenate([[0], tps / max(P, 1)])
    fpr = np.concatenate([[0], fps / max(N, 1)])
    thr = np.concatenate([[np.inf], s])
    prec = tps / np.maximum(tps + fps, 1)
    return dict(fpr=fpr, tpr=tpr, precision=prec, recall=tpr[1:], threshold=thr)
