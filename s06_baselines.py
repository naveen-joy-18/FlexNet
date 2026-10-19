"""
Stage 6: classical baselines and feature-group ablations.

Every model sees exactly the same family-aware split as the neural network, is
fitted on the training split of the training chemistry, has its decision
threshold chosen on that chemistry's validation split, and is then evaluated
untouched on the held-out test split of its own chemistry and on the fully
independent chemistry. A label-permutation control re-runs the best baseline on
shuffled training labels to confirm the pipeline cannot manufacture AUROC.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as EK                                                        # noqa: E402

FEAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "features")
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
PROTOCOLS = {"A": ("dms", "shape"), "B": ("shape", "dms")}


def load(tag):
    d = np.load(os.path.join(FEAT, f"{tag}_modelling.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def mask_split(d, split):
    """RNA-level split mask expanded to per-nucleotide indices."""
    if split is None:
        return np.arange(len(d["valid"]))
    sel = np.nonzero(d["split"] == split)[0]
    ptr = d["ptr"]
    return np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in sel])


def flatten(d, split, cols=None):
    """Return (X, y_bin, y_rank, valid, measured_lengths) for one split.

    `y_bin`, `y_rank` and `valid` are returned at full hairpin length so that
    per-RNA aggregation lines up with `measured_lengths`; callers mask with
    `valid.astype(bool)` when they need measured positions only.
    """
    idx = mask_split(d, split)
    X = d["X"][idx]
    if cols is not None:
        X = X[:, cols]
    lv = d["valid"][idx]
    return (X, d["y_bin"][idx], d["y_rank"][idx], lv,
            EK.measured_lengths(lv, d["lengths"][d["split"] == split]))


def measured(t):
    """Drop unmeasured positions from (X, valid) or any (array, valid) pair."""
    return t[0][t[1].astype(bool)]



def fit_predict(name, tr, va, te, te2, seed=0):
    """Fit on tr, threshold on va, return scores for te and te2."""
    if name == "logistic":
        mdl = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=3000, C=1.0,
                                              class_weight="balanced"))
    elif name == "svm":
        mdl = make_pipeline(StandardScaler(),
                           SVC(C=3.0, gamma="scale", probability=True,
                               class_weight="balanced", random_state=seed))
    elif name == "rf":
        mdl = RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
                                     n_jobs=-1, class_weight="balanced_subsample",
                                     random_state=seed)
    elif name == "gbm":
        mdl = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                             max_leaf_nodes=31,
                                             l2_regularization=1.0,
                                             early_stopping=False,
                                             random_state=seed)
    else:
        raise ValueError(name)

    t0 = time.time()
    tv = tr[3].astype(bool)
    mdl.fit(tr[0][tv], tr[1][tv])
    s_va = mdl.predict_proba(va[0])[:, 1]
    s_te = mdl.predict_proba(te[0])[:, 1]
    s_te2 = mdl.predict_proba(te2[0])[:, 1]
    return s_va, s_te, s_te2, time.time() - t0


def score(s_va, s_te, s_te2, va, te, te2, label, out, fit_s=None):
    """Scores arrive at full hairpin length and are masked to measured sites."""
    mv = va[3].astype(bool)
    s_va = np.asarray(s_va)[mv]
    thr, _ = EK.select_threshold(va[1][mv], s_va, "f1")
    L, _ = EK.select_min_length(va[1][mv], s_va, thr, va[4])
    row = {"model": label, "threshold": round(thr, 4), "min_len": L}
    if fit_s is not None:
        row["fit_seconds"] = round(fit_s, 1)
    for nm, sc, ds in (("own_test", s_te, te), ("other_test", s_te2, te2)):
        mm = ds[3].astype(bool)
        sc = np.asarray(sc)[mm]
        m = EK.point_metrics(ds[1][mm], sc, thr)

        rho = EK.per_rna_correlations(ds[2][mm], sc, ds[4])["median_rna_spearman"]
        r = EK.region_level_scores(ds[1][mm], sc >= thr, ds[4], min_len=L)
        row[nm] = dict(auc=round(m["auc"], 4), auprc=round(m["auprc"], 4),
                       f1=round(m["f1"], 4), mcc=round(m["mcc"], 4),
                       accuracy=round(m["accuracy"], 4),
                       sensitivity=round(m["sensitivity"], 4),
                       specificity=round(m["specificity"], 4),
                       median_rna_spearman=round(rho, 3),
                       mean_region_f1=round(r.get("mean_region_f1", np.nan), 3),
                       median_region_jaccard=round(
                           r.get("median_region_jaccard", np.nan), 3))
    out.append(row)
    a, b = row["own_test"], row["other_test"]
    print(f"  {label:34s} T={thr:.3f} L={L} | own AUC {a['auc']:.4f} "
          f"AP {a['auprc']:.4f} rho {a['median_rna_spearman']:+.3f} | "
          f"other AUC {b['auc']:.4f} AP {b['auprc']:.4f}", flush=True)
    return row


def main():
    os.makedirs(RES, exist_ok=True)
    results = {}
    for proto, (own, other) in PROTOCOLS.items():
        print(f"\n=== Protocol {proto}: fit on {own.upper()}, transfer to "
              f"{other.upper()} ===", flush=True)
        do, dx = load(own), load(other)
        names = list(do["feature_names"])
        groups = json.loads(str(do["groups"]))
        print("  feature groups: "
              + ", ".join(f"{g}={len(v)}" for g, v in sorted(groups.items())))

        tr = flatten(do, "train")
        va = flatten(do, "val")
        te = flatten(do, "test")
        te2 = flatten(dx, "test")
        out = []

        # reference single features, to show how far the structure alone goes
        for fname in ("paired_cent", "unpaired_prob", "paired_mfe"):
            if fname not in names:
                continue
            j = names.index(fname)
            score(va[0][:, j], te[0][:, j], te2[0][:, j],
                  va, te, te2, f"feature:{fname}", out)

        for name in ("logistic", "svm", "rf", "gbm"):
            s_va, s_te, s_te2, fs = fit_predict(name, tr, va, te, te2)
            score(s_va, s_te, s_te2, va, te, te2, name, out, fit_s=fs)

        # feature-group ablations on the strongest baseline
        combos = [(g, groups[g]) for g in sorted(groups)]
        # define ensemble block precisely: use the full ensemble group (12 features)
        combos.append(("no_ensemble_group",
                       [i for i in range(len(names)) if i not in groups["ensemble"]]))
        combos.append(("ensemble_only_group", groups["ensemble"]))
        combos.append(("no_basepair_marginal",
                       [i for i, n in enumerate(names) if n != "paired_cent"]))
        combos.append(("no_conservation",
                       [i for i, n in enumerate(names) if n != "conservation"]))
        for label, cols in combos:
            if len(cols) == 0 or len(cols) == len(names):
                continue
            sv, st, st2, fs = fit_predict(
                "gbm",
                flatten(do, "train", cols), flatten(do, "val", cols),
                flatten(do, "test", cols), flatten(dx, "test", cols))
            score(sv, st, st2, va, te, te2, f"gbm/{label}({len(cols)})", out, fit_s=fs)

        # label-permutation control: the pipeline must not manufacture signal
        rng = np.random.default_rng(0)
        ysh = rng.permutation(measured((tr[1], tr[3])))
        mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                             early_stopping=False, random_state=0)
        mdl.fit(measured((tr[0], tr[3])), ysh)

        sv = mdl.predict_proba(va[0])[:, 1]
        st = mdl.predict_proba(te[0])[:, 1]
        st2 = mdl.predict_proba(te2[0])[:, 1]
        score(sv, st, st2, va, te, te2, "gbm/permuted-labels", out)

        results[proto] = out
        with open(os.path.join(RES, "baselines.json"), "w") as fh:
            json.dump(results, fh, indent=2)
    print("\n[written] results/baselines.json")


if __name__ == "__main__":
    main()
