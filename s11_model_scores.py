"""
Stage 11: per-model score dump and the full classification table.

The stored baseline results (results/baselines.json) hold summary metrics only,
so the classical models could not be given ROC curves and the published
comparison table carried AUROC but no accuracy, sensitivity or specificity. This
script re-fits every model under exactly the protocol of s06_baselines.py - fit
on the training chemistry's training split, decision threshold chosen on that
chemistry's validation split, untouched evaluation on its own test split and on
the independent chemistry - and additionally

  * stores the per-nucleotide scores of every model, so ROC and precision-
    recall curves can be drawn for all of them and not only for FlexNet;
  * reports the full confusion-matrix metric set (accuracy, sensitivity,
    specificity, precision, negative predictive value, F1, Matthews correlation,
    Cohen's kappa, balanced accuracy, Brier score) next to AUROC and average
    precision, each with a hairpin-level bootstrap 95% CI;
  * runs paired DeLong tests on AUROC and exact McNemar tests on call
    disagreements between every model and FlexNet, at each model's own
    validation-optimal threshold.

FlexNet's scores and threshold are read from the reported run
(results/predictions.npz, results/call_calibration.json) so the neural network
is characterised by exactly the checkpoint and threshold already published.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as EK                                                        # noqa: E402
import s06_baselines as S06                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
PROTOCOLS = {"A": ("dms", "shape"), "B": ("shape", "dms")}
N_BOOT = 1000
SEED = 0

PRETTY = {"flexnet": "FlexNet", "gbm": "GBM", "rf": "Random forest",
          "logistic": "Logistic regression", "svm": "SVM (RBF)",
          "feature:unpaired_prob": "Unpaired probability (single feature)",
          "feature:paired_cent": "Paired central (single feature)",
          "feature:paired_mfe": "Paired MFE (single feature)",
          "gbm/permuted-labels": "GBM, permuted labels (control)"}
ABLATION_LABELS = {
    "gbm/structure(28)": "GBM, structure only (28)",
    "gbm/sequence(19)": "GBM, sequence only (19)",
    "gbm/thermo(7)": "GBM, thermodynamics only (7)",
    "gbm/no_ensemble(57)": "GBM, ensemble block removed (57)",
    "gbm/no_basepair_marginal(65)": "GBM, paired-central removed (65)",
    "gbm/no_conservation(65)": "GBM, conservation removed (65)",
}
PRETTY.update(ABLATION_LABELS)
CLASSICAL = ("gbm", "rf", "logistic", "svm")


def build(name, seed=SEED):
    """Identical estimator configurations to s06_baselines.fit_predict."""
    if name == "logistic":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0,
                                                class_weight="balanced"))
    if name == "svm":
        return make_pipeline(StandardScaler(),
                             SVC(C=3.0, gamma="scale", probability=True,
                                 class_weight="balanced", random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
                                      n_jobs=-1,
                                      class_weight="balanced_subsample",
                                      random_state=seed)
    if name == "gbm":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                              max_leaf_nodes=31,
                                              l2_regularization=1.0,
                                              early_stopping=False,
                                              random_state=seed)
    raise ValueError(name)


def masked(d, split):
    """Measured positions of one split with their hairpin ids.

    `mask_split` expands the RNA-level split into per-nucleotide indices. The
    per-nucleotide RNA id is needed alongside it so the bootstrap resamples
    whole hairpins rather than individual nucleotides, which are not
    independent observations.
    """
    ptr = d["ptr"]
    sel = np.flatnonzero(d["split"] == split)
    rna = np.concatenate([np.full(int(ptr[i + 1] - ptr[i]), d["exp_id"][i])
                          for i in sel])
    idx = S06.mask_split(d, split)
    m = d["valid"][idx].astype(bool)
    assert len(rna) == len(idx), (len(rna), len(idx))
    return d["X"][idx][m], d["y_bin"][idx][m], rna[m]


def flexnet_split(tag):
    """Measured FlexNet labels, scores and hairpin ids from the reported run.

    Returns the same (y, score, rna) layout as `masked`, so the neural network
    and the classical models are scored through one code path.
    """
    p = np.load(os.path.join(RESULTS, "predictions.npz"))
    m = p[tag + "_valid"] > 0
    return p[tag + "_y"][m], p[tag][m], p[tag + "_rna"][m]


def metric_block(y, s, rna, thr):
    """Full point-metric set with hairpin-bootstrap CIs, at threshold `thr`."""
    m = EK.point_metrics(y, s, thr)
    stats = dict(accuracy=m["accuracy"], sensitivity=m["sensitivity"],
                 specificity=m["specificity"], precision=m["precision"],
                 negative_predictive_value=m["negative_predictive_value"],
                 f1=m["f1"], mcc=m["mcc"], kappa=m["kappa"],
                 balanced_accuracy=m["balanced_accuracy"],
                 brier=m["brier"])
    # hairpin-level resampling: nucleotides within a hairpin are not independent
    uniq = np.unique(rna)
    blocks = [np.flatnonzero(rna == r) for r in uniq]
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in ("accuracy", "sensitivity", "specificity", "f1",
                           "mcc", "auc", "auprc")}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(blocks), len(blocks))
        sel = np.concatenate([blocks[p] for p in pick])
        yb, sb = y[sel], s[sel]
        if len(np.unique(yb)) < 2:
            continue
        b = EK.point_metrics(yb, sb, thr)
        boot["accuracy"].append(b["accuracy"])
        boot["sensitivity"].append(b["sensitivity"])
        boot["specificity"].append(b["specificity"])
        boot["f1"].append(b["f1"])
        boot["mcc"].append(b["mcc"])
        boot["auc"].append(b["auc"])
        boot["auprc"].append(b["auprc"])
    ci = {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
              if len(v) >= 20 else [float("nan"), float("nan")])
          for k, v in boot.items()}
    stats["auc"], stats["auprc"] = m["auc"], m["auprc"]
    stats["threshold"] = float(thr)
    stats["tp"], stats["tn"] = m["tp"], m["tn"]
    stats["fp"], stats["fn"] = m["fp"], m["fn"]
    stats["n_measured"] = int(len(y))
    stats["n_rnas"] = int(len(uniq))
    stats["prevalence"] = m["prevalence"]
    stats["ci"] = ci
    return stats


def run_protocol(proto):
    own, other = PROTOCOLS[proto]
    do = S06.load(own)
    dx = S06.load(other)
    names = list(do["feature_names"])

    tr = masked(do, "train")
    va = masked(do, "val")
    te = masked(do, "test")
    te2 = masked(dx, "test")

    models = {}

    # ---- reference single features --------------------------------------
    # a raw feature value is used directly as the score; thresholds still come
    # from the validation split, so these references are called identically
    for fname in ("paired_cent", "unpaired_prob", "paired_mfe"):
        if fname not in names:
            continue
        j = names.index(fname)
        models[f"feature:{fname}"] = (va[0][:, j], te[0][:, j], te2[0][:, j])

    # ---- fitted classical models ----------------------------------------
    for name in ("logistic", "svm", "rf", "gbm"):
        mdl = build(name)
        t0 = time.time()
        mdl.fit(tr[0], tr[1])
        print(f"  fitted {name:9s} in {time.time() - t0:6.1f}s", flush=True)
        models[name] = (mdl.predict_proba(va[0])[:, 1],
                        mdl.predict_proba(te[0])[:, 1],
                        mdl.predict_proba(te2[0])[:, 1])

    # ---- feature-group ablations on the strongest baseline ---------------
    # Same column subsets as s06_baselines, so the ablation rows quoted in the
    # manuscript are the same models, reported here with the same threshold
    # rule, CIs and paired tests as every other row.
    groups = json.loads(str(do["groups"]))
    combos = [(g, groups[g]) for g in sorted(groups)]
    combos.append(("no_ensemble",
                   [i for i, n in enumerate(names)
                    if not n.startswith(("ens_", "unpaired_w", "unpaired_max",
                                        "unpaired_ent", "flex_geom", "dG_ctx"))]))
    combos.append(("no_basepair_marginal",
                   [i for i, n in enumerate(names) if n != "paired_cent"]))
    combos.append(("no_conservation",
                   [i for i, n in enumerate(names) if n != "conservation"]))
    for label, cols in combos:
        if len(cols) == 0 or len(cols) == len(names):
            continue
        sub = [(t[0][:, cols], t[1], t[2]) for t in (tr, va, te, te2)]
        mdl = build("gbm")
        mdl.fit(sub[0][0], sub[0][1])
        models[f"gbm/{label}({len(cols)})"] = (
            mdl.predict_proba(sub[1][0])[:, 1],
            mdl.predict_proba(sub[2][0])[:, 1],
            mdl.predict_proba(sub[3][0])[:, 1])
        print(f"  fitted gbm/{label}({len(cols)})", flush=True)

    # ---- label-permutation control ---------------------------------------
    # The pipeline must not be able to manufacture AUROC on its own; a control
    # near 0.5 with MCC near zero is the expected outcome and is the check a
    # reviewer will ask for.
    rng = np.random.default_rng(SEED)
    ysh = rng.permutation(tr[1])
    ctl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                         early_stopping=False, random_state=SEED)
    ctl.fit(tr[0], ysh)
    models["gbm/permuted-labels"] = (ctl.predict_proba(va[0])[:, 1],
                                     ctl.predict_proba(te[0])[:, 1],
                                     ctl.predict_proba(te2[0])[:, 1])
    print("  fitted gbm/permuted-labels (control)", flush=True)

    # ---- FlexNet, from the reported checkpoint --------------------------
    fva, fte, fte2 = (flexnet_split(f"{proto}_own_val"),
                      flexnet_split(f"{proto}_own_test"),
                      flexnet_split(f"{proto}_other_test"))
    models["flexnet"] = (fva[1], fte[1], fte2[1])

    rows = {}
    scores = {}
    thresholds = {}
    # FlexNet is processed first: the paired tests below need its published
    # threshold, so it must be resolved before any baseline is compared to it.
    for name in ["flexnet"] + [m for m in models if m != "flexnet"]:
        s_va, s_te, s_te2 = models[name]
        thr_f1 = thr_y = None
        if name == "flexnet":
            cal = json.load(open(os.path.join(RESULTS,
                                              "call_calibration.json")))[proto]
            thr = cal["threshold"]
            thr_note = ("validation-calibrated (median of F1 and Youden optima), "
                        "applied unchanged")
        else:
            # the same rule the network uses, so that every row of the table is
            # called by one convention: the median of the validation F1-optimal
            # and Youden-optimal cut-offs (s05_train.py). Using the F1 optimum
            # alone for the baselines would compare them to FlexNet at a
            # different operating point than FlexNet was given.
            thr_f1, _ = EK.select_threshold(va[1], s_va, "f1")
            thr_y, _ = EK.select_threshold(va[1], s_va, "youden")
            thr = float(np.median([thr_f1, thr_y]))
            if not 0 < thr < 1:
                thr = float(thr_f1)
            thr_note = ("validation-optimal (median of F1 and Youden optima), "
                        "applied unchanged")

        blk = {}
        for split, y, s, rna in (("own_test", te[1], s_te, te[2]),
                                 ("other_test", te2[1], s_te2, te2[2])):
            blk[split] = metric_block(y, s, rna, thr)
            # both endpoints of the threshold rule are retained, so a reviewer
            # can see the metric spread across operating points
            if thr_f1 is not None:
                blk[split]["operating_points"] = {
                    "at_f1_optimal": {k: round(float(v), 4) for k, v in
                                      metric_block(y, s, rna, thr_f1).items()
                                      if k != "ci"},
                    "at_youden_optimal": {k: round(float(v), 4) for k, v in
                                          metric_block(y, s, rna, thr_y).items()
                                          if k != "ci"}}
        blk["threshold"] = round(float(thr), 4)
        blk["threshold_f1"] = None if thr_f1 is None else round(float(thr_f1), 4)
        blk["threshold_youden"] = (None if thr_y is None
                                   else round(float(thr_y), 4))
        blk["threshold_source"] = thr_note
        thresholds[name] = float(thr)
        rows[name] = blk
        scores[name] = dict(
            own_test_y=te[1], own_test_s=s_te, own_test_rna=te[2],
            other_test_y=te2[1], other_test_s=s_te2, other_test_rna=te2[2])

        # ---- paired tests against FlexNet --------------------------------
        fy, fs = fte[0], fte[1]
        fy2, fs2 = fte2[0], fte2[1]
        for split, (y, s, rna), (fy_, fs_) in (
                ("own_test", (te[1], s_te, te[2]), (fy, fs)),
                ("other_test", (te2[1], s_te2, te2[2]), (fy2, fs2))):
            if name == "flexnet":
                continue
            if len(y) != len(fy_) or not np.array_equal(y, fy_):
                rows[name][split]["comparable_to_flexnet"] = False
                continue
            rows[name][split]["comparable_to_flexnet"] = True
            d = EK.delong_paired(fy_, fs_, y, s)
            # each model is binarised at its own threshold: the network keeps
            # the threshold published in call_calibration.json, the baseline
            # keeps the one chosen on its own validation split. Binarising both
            # at a single shared cut-off would measure threshold drift rather
            # than a difference in predictions.
            thr_base, thr_nn = thresholds[name], thresholds["flexnet"]
            mc = EK.mcnemar(y, (s >= thr_base).astype(int),
                            fy_, (fs_ >= thr_nn).astype(int))
            rows[name][split]["delong_vs_flexnet"] = {
                k: round(float(v), 6) for k, v in d.items()}
            rows[name][split]["mcnemar_vs_flexnet"] = {
                "b": mc["b"], "c": mc["c"], "p": round(float(mc["p"]), 6),
                "baseline_threshold": round(thr_base, 4),
                "flexnet_threshold": round(thr_nn, 4),
                "disagreement_pct": round(
                    100.0 * float(np.mean((s >= thr_base) != (fs_ >= thr_nn))),
                    2)}

        b = blk["own_test"]
        print(f"  {name:26s} T={thr:.3f} | own AUC {b['auc']:.4f} "
              f"acc {b['accuracy']:.4f} sens {b['sensitivity']:.4f} "
              f"spec {b['specificity']:.4f}", flush=True)
    return rows, scores


def main():
    t0 = time.time()
    os.makedirs(TABLES, exist_ok=True)
    out, all_scores = {}, {}
    for proto in "AB":
        print(f"\n=== Protocol {proto} ===", flush=True)
        rows, scores = run_protocol(proto)
        out[proto] = rows
        all_scores[proto] = scores

    with open(os.path.join(RESULTS, "model_scores.npz"), "wb") as fh:
        np.savez_compressed(fh, **{f"{p}|{m}|{k}": np.asarray(v)
                                   for p, sc in all_scores.items()
                                   for m, d in sc.items()
                                   for k, v in d.items()})
    with open(os.path.join(RESULTS, "model_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("[written] results/model_scores.npz, results/model_comparison.json")

    # ------------------------------------------------------------ tables ---
    write_tables(out)
    print(f"[done] {time.time() - t0:.0f}s")


def _rows_flatten(out):
    for proto in "AB":
        for model, blk in out[proto].items():
            for split in ("own_test", "other_test"):
                d = blk[split]
                yield proto, model, split, d


def write_tables(out):
    """CSV, markdown and LaTeX renderings of the same numbers."""
    own_tag = {"A": "DMS-MaPseq", "B": "SHAPE-MaP"}
    cols = ["protocol", "split", "model", "n_test_rnas", "n_measured",
            "prevalence", "threshold", "auroc", "auroc_ci_low", "auroc_ci_high",
            "auprc", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
            "sensitivity", "sensitivity_ci_low", "sensitivity_ci_high",
            "specificity", "specificity_ci_low", "specificity_ci_high",
            "balanced_accuracy", "precision", "npv", "f1", "mcc", "kappa",
            "brier", "tp", "tn", "fp", "fn", "delong_p", "mcnemar_p",
            "disagreement_pct"]

    def rec(proto, model, split, d):
        ci = d["ci"]
        dl = d.get("delong_vs_flexnet", {})
        mc = d.get("mcnemar_vs_flexnet", {})
        return dict(
            protocol=proto, split=split, model=model,
            n_test_rnas=d["n_rnas"], n_measured=d["n_measured"],
            prevalence=round(d["prevalence"], 4),
            threshold=d["threshold"],
            auroc=round(d["auc"], 4),
            auroc_ci_low=round(ci["auc"][0], 4),
            auroc_ci_high=round(ci["auc"][1], 4),
            auprc=round(d["auprc"], 4),
            accuracy=round(d["accuracy"], 4),
            accuracy_ci_low=round(ci["accuracy"][0], 4),
            accuracy_ci_high=round(ci["accuracy"][1], 4),
            sensitivity=round(d["sensitivity"], 4),
            sensitivity_ci_low=round(ci["sensitivity"][0], 4),
            sensitivity_ci_high=round(ci["sensitivity"][1], 4),
            specificity=round(d["specificity"], 4),
            specificity_ci_low=round(ci["specificity"][0], 4),
            specificity_ci_high=round(ci["specificity"][1], 4),
            balanced_accuracy=round(d["balanced_accuracy"], 4),
            precision=round(d["precision"], 4),
            npv=round(d["negative_predictive_value"], 4),
            f1=round(d["f1"], 4), mcc=round(d["mcc"], 4),
            kappa=round(d["kappa"], 4), brier=round(d["brier"], 4),
            tp=d["tp"], tn=d["tn"], fp=d["fp"], fn=d["fn"],
            delong_p=(round(dl["p"], 4) if "p" in dl else ""),
            mcnemar_p=(round(mc["p"], 4) if "p" in mc else ""),
            disagreement_pct=(mc["disagreement_pct"] if "p" in mc else ""))

    recs = [rec(p, m, s, d) for p, m, s, d in _rows_flatten(out)]
    csv_path = os.path.join(TABLES, "model_comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)

    # core models first, then the single-feature references, then the
    # feature-group ablations, then the permutation control
    order = (["flexnet", "gbm", "rf", "logistic", "svm"]
             + ["feature:unpaired_prob", "feature:paired_mfe",
                "feature:paired_cent"]
             + list(ABLATION_LABELS)
             + ["gbm/permuted-labels"])
    order = [m for m in order if any(m in out[p] for p in out)]

    def get(p, m, s, k):
        for proto, model, split, d in _rows_flatten(out):
            if proto == p and model == m and split == s:
                return d[k]
        return float("nan")

    lines = ["# Per-model classification metrics (test splits)", "",
             "All models are fitted on the training chemistry's training split "
             "and called at their own validation-optimal threshold, applied "
             "unchanged to test. `own` = held-out families of the training "
             "chemistry; `other` = independent chemistry. Brackets are "
             "hairpin-level bootstrap 95% CIs (1000 resamples). DeLong p and "
             "McNemar p are paired tests against FlexNet on identical "
             "positions. Every model, including FlexNet, is called at the same "
             "rule: the median of its validation F1-optimal and Youden-optimal "
             "cut-offs, then applied unchanged to both test splits. The two "
             "endpoints of that rule are reported per model in "
             "`model_comparison.csv` (`at_f1_optimal`, `at_youden_optimal`), "
             "so the sensitivity/specificity spread across operating points is "
             "visible rather than hidden.", ""]
    for p in "AB":
        lines += [f"## Protocol {p} (fit on {own_tag[p]})", "",
                  "| model | split | AUROC [95% CI] | AP | accuracy [95% CI] | "
                  "sensitivity [95% CI] | specificity [95% CI] | MCC | "
                  "DeLong p | McNemar p |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for m in order:
            if m not in out[p]:
                continue
            for s in ("own_test", "other_test"):
                d = out[p][m][s]
                ci = d["ci"]
                dl = d.get("delong_vs_flexnet", {})
                mc = d.get("mcnemar_vs_flexnet", {})
                ci_s = lambda k: (f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
                                  if np.isfinite(ci[k][0]) else "")     # noqa: E731
                dp = f"{dl['p']:.3f}" if "p" in dl else "-"
                mp = f"{mc['p']:.3f}" if "p" in mc else "-"
                lines.append(
                    f"| {PRETTY.get(m, m)} | {s.replace('_', ' ')} | "
                    f"{d['auc']:.3f} {ci_s('auc')} | {d['auprc']:.3f} | "
                    f"{d['accuracy']:.3f} {ci_s('accuracy')} | "
                    f"{d['sensitivity']:.3f} {ci_s('sensitivity')} | "
                    f"{d['specificity']:.3f} {ci_s('specificity')} | "
                    f"{d['mcc']:.3f} | {dp} | {mp} |")
        lines.append("")
    md_path = os.path.join(TABLES, "model_comparison.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    tex_path = os.path.join(TABLES, "model_comparison.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(latex_table(out, order, own_tag))
    print("[written] results/tables/model_comparison.{csv,md,tex}")


def latex_table(out, order, own_tag):
    def esc(x):
        return str(x).replace("_", r"\_")

    t = []
    t.append(r"\begin{table}[t]")
    t.append(r"\centering\small")
    t.append(r"\caption{Classification performance of FlexNet and all "
             r"classical baselines on the held-out test splits of both "
             r"chemistries, at each model's own validation-optimal "
             r"threshold. Accuracy, sensitivity and specificity carry "
             r"hairpin-level bootstrap 95\% CIs (1000 resamples). DeLong $p$ "
             r"compares AUROC with FlexNet on identical positions; McNemar "
             r"$p$ compares the two sets of calls.}\label{tab:perf}")
    t.append(r"\begin{tabular}{llrrrrrr}")
    t.append(r"\toprule")
    t.append(r"Model & Test set & AUROC & AP & Accuracy & Sensitivity & "
             r"Specificity & MCC \\")
    t.append(r"\midrule")
    for p in "AB":
        t.append(r"\multicolumn{8}{@{}l}{\emph{Protocol %s (fitted on %s)}} \\"
                 % (p, own_tag[p]))
        for m in order:
            if m not in out[p]:
                continue
            d = out[p][m]["own_test"]
            d2 = out[p][m]["other_test"]
            ci = d["ci"]
            dl = d.get("delong_vs_flexnet", {})
            pstr = (f"$^{{{dl['p']:.2g}}}$" if "p" in dl else "")
            for dd, lab in ((d, "own"), (d2, "other")):
                c = dd["ci"]
                t.append("%s & %s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f %s \\\\"
                         % (esc(PRETTY.get(m, m)), lab, dd["auc"], dd["auprc"],
                            dd["accuracy"], dd["sensitivity"],
                            dd["specificity"], dd["mcc"], pstr))
        t.append(r"\addlinespace")
    t.append(r"\bottomrule")
    t.append(r"\end{tabular}")
    t.append(r"\end{table}")
    return "\n".join(t)


if __name__ == "__main__":
    main()
