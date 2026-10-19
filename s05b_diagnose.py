"""
Diagnostic: how predictive is each feature, and what can a strong tabular model
achieve? This bounds what the neural network should be able to do and reveals
whether the ceiling comes from the features/labels or from the architecture.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FEAT                                               # noqa: E402
import evalkit as EK                                                # noqa: E402


def load(tag):
    d = np.load(os.path.join(FEAT, f"{tag}_modelling.npz"), allow_pickle=True)
    return d


def flat(d, split):
    idx = np.where(d["split"] == split)[0]
    ptr = d["ptr"]
    out = {k: [] for k in ("X", "y_bin", "y", "valid")}
    lens = []
    for i in idx:
        a, b = int(ptr[i]), int(ptr[i + 1])
        out["X"].append(d["X"][a:b])
        out["y_bin"].append(d["y_bin"][a:b])
        out["y"].append(d["y"][a:b])
        out["valid"].append(d["valid"][a:b])
        lens.append(b - a)
    return (np.concatenate(out["X"]), np.concatenate(out["y_bin"]),
            np.concatenate(out["y"]), np.concatenate(out["valid"]),
            np.array(lens), idx)


def main():
    dms = load("dms")
    shape = load("shape")
    names = [str(s) for s in dms["feature_names"]]

    for tag, d in (("DMS", dms), ("SHAPE", shape)):
        Xtr, ytr, rtr, vtr, ltr, _ = flat(d, "train")
        Xte, yte, rte, vte, lte, _ = flat(d, "test")
        mtr, mte = vtr > 0, vte > 0
        print(f"\n===== {tag} =====")
        print(f"train {Xtr[mtr].shape} pos={ytr[mtr].mean():.3f} | "
              f"test {Xte[mte].shape} pos={yte[mte].mean():.3f}")

        # single-feature Spearman against continuous reactivity
        rows = []
        for j, nm in enumerate(names):
            a, b = Xtr[mtr][:, j], rtr[mtr]
            if np.std(a) == 0:
                rows.append((nm, 0.0))
                continue
            rows.append((nm, float(spearmanr(a, b).statistic)))
        rows.sort(key=lambda x: -abs(x[1]))
        print("top 12 features by |Spearman| with reactivity:")
        for nm, r in rows[:12]:
            print(f"   {nm:22s} {r:+.3f}")

        # class-balance check
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            yy = (rtr[mtr] >= thr).astype(int)
            if 0 < yy.sum() < len(yy):
                print(f"   reactivity>={thr:.1f} -> prevalence {yy.mean():.3f}")

        gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             random_state=0)
        gbm.fit(Xtr[mtr], ytr[mtr])
        p = gbm.predict_proba(Xte[mte])[:, 1]
        m = EK.point_metrics(yte[mte], p)
        print(f"   GBM  test AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
              f"rho={EK.correlations(rte[mte], p)['spearman']:+.3f}")
        log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        log.fit(Xtr[mtr], ytr[mtr])
        p2 = log.predict_proba(Xte[mte])[:, 1]
        m2 = EK.point_metrics(yte[mte], p2)
        print(f"   Logit test AUC={m2['auc']:.4f} AUPRC={m2['auprc']:.4f} "
              f"rho={EK.correlations(rte[mte], p2)['spearman']:+.3f}")

    # cross-chemistry transfer of a single strong feature
    print("\n===== cross-chemistry single-feature check =====")
    unl = names.index("unpaired_prob")
    pairp = names.index("pair_prob")
    for tag, d in (("DMS", dms), ("SHAPE", shape)):
        X, yb, r, v, l, _ = flat(d, "test")
        m = v > 0
        for j, nm in ((unl, "unpaired_prob"), (pairp, "-pair_prob")):
            s = X[m][:, j] * (-1 if nm.startswith("-") else 1)
            print(f"  {tag:6s} {nm:16s} rho(reactivity)="
                  f"{EK.correlations(r[m], s)['spearman']:+.3f} "
                  f"AUC={roc_auc_score(yb[m], s):.4f}")


if __name__ == "__main__":
    main()
