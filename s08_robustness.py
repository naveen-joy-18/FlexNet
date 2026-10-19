"""
Stage 8: robustness, model comparison and heterogeneity.

Four questions:
  1. is the neural net significantly better than the best classical model on the
     same held-out nucleotides (DeLong for AUROC, McNemar for the calls);
  2. how stable are the predictions across the three random seeds;
  3. does performance vary by miRNA family or hairpin length;
  4. how much does the model depend on the ensemble (structure-dynamics) block,
     measured by permuting that block within each hairpin.

Family-level and length-level breakdowns resample hairpins, not nucleotides, so
a few long hairpins cannot dominate.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as EK                                                        # noqa: E402
import s05_train as T                                                       # noqa: E402
from model import FlexNet, RNADataset, FEAT                              # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
PROTOCOLS = {"A": ("dms", "shape"), "B": ("shape", "dms")}
SEEDS = (0, 1, 2)


def load_scaled(path, proto, split=None):
    sc = np.load(os.path.join(RESULTS, f"scaler_{proto}.npy.npz"))
    return RNADataset(path, split=split, feature_scale=(sc["mu"], sc["sd"]))


def gbm_scores(own, other, split, cols=None):
    d = {t: np.load(os.path.join(FEAT, f"{t}_modelling.npz"), allow_pickle=True)
         for t in (own, other)}
    X = d[own]["X"]
    if cols is not None:
        X = X[:, cols]

    def flat(tag, sp):
        dd = d[tag]
        sel = np.nonzero(dd["split"] == sp)[0] if sp else np.arange(len(dd["ptr"]) - 1)
        ptr = dd["ptr"]
        idx = np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in sel])
        xx = d[tag]["X"][idx]
        if cols is not None:
            xx = xx[:, cols]
        m = dd["valid"][idx] > 0
        return xx[m], dd["y_bin"][idx][m], idx[m]

    Xtr, ytr, _ = flat(own, "train")
    mdl = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=False, random_state=0)
    mdl.fit(Xtr, ytr)
    s_te = mdl.predict_proba(flat(own, "test")[0])[:, 1]
    s_ot = mdl.predict_proba(flat(other, "test")[0])[:, 1]
    # own-protocol F1-optimal threshold matched to the network's convention
    # (median of the F1 and Youden optima on the validation split)
    y_va, s_va = flat(own, "val")[1], mdl.predict_proba(flat(own, "val")[0])[:, 1]
    thr_f1, _ = EK.select_threshold(y_va, s_va, "f1")
    thr_y, _ = EK.select_threshold(y_va, s_va, "youden")
    thr = float(np.median([thr_f1, thr_y]))
    thr = thr if 0 < thr < 1 else float(thr_f1)
    return (s_te, s_ot, flat(own, "test")[1], flat(other, "test")[1],
            flat(own, "test")[2], thr)


def score_with_checkpoint(proto, tag, split, seed, ensemble_perm=None):
    path = os.path.join(FEAT, f"{tag}_modelling.npz")
    sc = np.load(os.path.join(RESULTS, f"scaler_{proto}.npy.npz"))
    ds = RNADataset(path, split=split, feature_scale=(sc["mu"], sc["sd"]))
    if ensemble_perm is not None:
        # permute the ensemble block inside each hairpin: keeps the marginal
        # distribution but destroys the position-structure coupling
        d = np.load(path, allow_pickle=True)
        names = list(d["feature_names"])
        ens = [i for i, n in enumerate(names)
               if n.startswith(("ens_", "unpaired_", "flex_geom", "pair_dens",
                                "dG_ctx", "ens_diversity", "ens_defect"))]
        rng = np.random.default_rng(123)
        X = d["X"].copy()
        ptr = d["ptr"]
        for i in range(len(ptr) - 1):
            lo, hi = ptr[i], ptr[i + 1]
            X[lo:hi, ens] = X[lo:hi, ens][rng.permutation(hi - lo)]
        tmp = os.path.join(RESULTS, f"_perm_{tag}.npz")
        d2 = {k: d[k] for k in d.files}
        d2["X"] = X
        np.savez_compressed(tmp, **d2)
        ds = RNADataset(tmp, split=split, feature_scale=(sc["mu"], sc["sd"]))
    model = FlexNet(n_features=ds.n_features)
    model.load_state_dict(torch.load(
        os.path.join(RESULTS, f"model_{proto}_s{seed}.pt"), map_location="cpu"))
    s, _, _, _ = T.predict(model, T.make_loader(ds))
    y, _, valid, lengths, _, _, per = T.flat_eval(ds)
    return s, y, valid, lengths, per


def main():
    cal = json.load(open(os.path.join(RESULTS, "call_calibration.json")))
    metrics = json.load(open(os.path.join(RESULTS, "metrics_main.json")))
    preds = np.load(os.path.join(RESULTS, "predictions.npz"))
    out = {}

    for proto, (own, other) in PROTOCOLS.items():
        thr = cal[proto]["threshold"]
        seed_nn = int(metrics[f"{proto}/chosen_seed"])
        res = {}

        # ---- 1. neural net vs gradient boosting on identical nucleotides ----
        s_te_g, s_ot_g, y_te_g, y_ot_g, idx_te_g, thr_g = gbm_scores(
            own, other, "test")
        for split_name, s_g, y_g, pkey in (
                ("own_test", s_te_g, y_te_g, f"{proto}_own_test"),
                ("other_test", s_ot_g, y_ot_g, f"{proto}_other_test")):
            nn = preds[pkey][preds[pkey + "_valid"] > 0]
            yy = preds[pkey + "_y"][preds[pkey + "_valid"] > 0]
            if len(nn) != len(s_g):
                print(f"[skip] {split_name}: NN {len(nn)} nt vs GBM {len(s_g)} nt")
                continue
            auc_nn = EK.point_metrics(yy, nn)["auc"]
            auc_gb = EK.point_metrics(y_g, s_g)["auc"]
            dl = EK.delong_paired(yy, nn, yy, s_g)
            mc = EK.mcnemar(yy, (nn >= thr).astype(int),
                            yy, (s_g >= thr_g).astype(int))
            res[f"nn_vs_gbm/{split_name}"] = dict(
                nn_auc=round(auc_nn, 4), gbm_auc=round(auc_gb, 4),
                delta_auc=round(auc_nn - auc_gb, 4),
                delong_z=round(dl["z"], 3), delong_p=float(dl["p"]),
                delong_diff=round(dl["diff"], 4),
                mcnemar_b=mc["b"], mcnemar_c=mc["c"], mcnemar_p=float(mc["p"]),
                nn_threshold=thr, gbm_threshold=thr_g)
            print(f"  {proto}/{split_name}: NN {auc_nn:.4f} vs GBM {auc_gb:.4f} "
                  f"(delta {auc_nn - auc_gb:+.4f}, DeLong p={dl['p']:.2e}, "
                  f"McNemar p={mc['p']:.2e})", flush=True)

        # ---- 2. seed stability ------------------------------------------------
        all_seeds = {}
        for sd in SEEDS:
            s, y, valid, lengths, per = score_with_checkpoint(
                proto, own, "test", sd)
            mm = valid > 0
            all_seeds[sd] = (s[mm], per[mm], y[mm], mm, lengths)
        rhos, aucs = [], []
        for a in SEEDS:
            for b in SEEDS:
                if a < b:
                    r = EK.correlations(all_seeds[a][0], all_seeds[b][0])
                    rhos.append(round(r["spearman"], 4))
            aucs.append(round(EK.point_metrics(all_seeds[a][2],
                                              all_seeds[a][0])["auc"], 4))
        res["seed_stability"] = dict(pairwise_score_spearman=rhos,
                                     per_seed_auc=aucs,
                                     reported_seed=seed_nn,
                                     auc_range=[min(aucs), max(aucs)])
        print(f"  {proto} seed AUCs {aucs}, pairwise score rho mean "
              f"{np.mean(rhos):.4f}", flush=True)

        # ---- 3. per-family and per-length heterogeneity ------------------------
        s, y, valid, lengths, per = score_with_checkpoint(proto, own, "test",
                                                         seed_nn)
        d = np.load(os.path.join(FEAT, f"{own}_modelling.npz"), allow_pickle=True)
        ptr = d["ptr"]
        sel = np.nonzero(d["split"] == "test")[0]
        mlen = EK.measured_lengths(valid, lengths)
        off, fam_auc, len_auc = 0, [], []
        for i in sel:
            n = int(ptr[i + 1] - ptr[i])
            blk = slice(off, off + n)
            mm = valid[blk] > 0
            off += n
            if mm.sum() < 20 or len(np.unique(y[blk][mm])) < 2:
                continue
            m = EK.point_metrics(y[blk][mm], s[blk][mm])
            if np.isfinite(m["auc"]):
                fam_auc.append((str(d["family"][i]), m["auc"]))
                len_auc.append((n, m["auc"]))
        fa = np.array([a for _, a in fam_auc])
        res["per_family_auc"] = dict(
            n_families=len(fa), median=round(float(np.median(fa)), 4),
            iqr=[round(float(np.percentile(fa, 25)), 4),
                 round(float(np.percentile(fa, 75)), 4)],
            frac_above_0_8=round(float(np.mean(fa > 0.8)), 4),
            frac_below_0_6=round(float(np.mean(fa < 0.6)), 4))
        la = np.array([a for _, a in len_auc])
        ln = np.array([n for n, _ in len_auc])
        bins = [(50, 65), (65, 80), (80, 120), (120, 10000)]
        res["per_length_auc"] = {
            f"{lo}-{hi if hi < 1000 else '+'}": round(
                float(np.median(la[(ln >= lo) & (ln < hi)])), 4)
            for lo, hi in bins if ((ln >= lo) & (ln < hi)).any()}
        print(f"  {proto} per-family AUC median {np.median(fa):.4f} "
              f"(n={len(fa)}), by length {res['per_length_auc']}", flush=True)

        # ---- 4. dependence on the structure-dynamics block --------------------
        s_p, y_p, valid_p, lengths_p, _ = score_with_checkpoint(
            proto, own, "test", seed_nn, ensemble_perm=True)
        mmp = valid_p > 0
        base = EK.point_metrics(y[valid > 0], s[valid > 0])
        perm = EK.point_metrics(y_p[mmp], s_p[mmp])
        rho_y = EK.correlations(s[valid > 0], s_p[mmp])["spearman"]
        res["ensemble_block_permuted"] = dict(
            auc_before=round(base["auc"], 4), auc_after=round(perm["auc"], 4),
            delta_auc=round(perm["auc"] - base["auc"], 4),
            score_spearman_before_after=round(rho_y, 4))
        print(f"  {proto} within-hairpin ensemble permutation: AUC "
              f"{base['auc']:.4f} -> {perm['auc']:.4f}, score rho {rho_y:.4f}",
              flush=True)
        os.remove(os.path.join(RESULTS, f"_perm_{own}.npz"))

        out[proto] = res

    with open(os.path.join(RESULTS, "robustness.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[written] results/robustness.json")


if __name__ == "__main__":
    main()
