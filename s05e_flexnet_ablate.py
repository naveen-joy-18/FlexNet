def main():
    import os, json
    import numpy as np
    from model import FlexNet, RNADataset, collate, multitask_loss, FEAT, RESULTS
    import evalkit as EK
    SEEDS = (0,1,2)
    import numpy as np
    from model import FlexNet, RNADataset, collate, multitask_loss, FEAT, RESULTS
    import evalkit as EK
    SEEDS = (0,1,2)
    paths = {k: os.path.join(FEAT, f"{k}_modelling.npz") for k in ("dms", "shape")}
    d = np.load(paths["dms"], allow_pickle=True)
    names = list(d["feature_names"])
    groups = json.loads(str(d["groups"]))

    out = {}
    for proto, (own_tag, _) in [("A", ("dms", "shape")), ("B", ("shape", "dms"))]:
        path = paths[own_tag]
        base_tr = RNADataset(path, split="train", feature_scale=None)
        scaler_full = base_tr.fit_scaler()
        tr0 = RNADataset(path, split="train", feature_scale=scaler_full)
        va0 = RNADataset(path, split="val", feature_scale=scaler_full)
        te0 = RNADataset(path, split="test", feature_scale=scaler_full)
        yv = int(np.nansum(tr0.y_bin[tr0.valid]))
        nv = int(tr0.valid.sum())
        pos_weight = float(max(1.0, (nv - yv) / max(yv, 1)))

        combos = []
        combos.append(("ensemble_only_group", groups["ensemble"]))
        combos.append(("no_ensemble_group", [i for i in range(len(names)) if i not in groups["ensemble"]]))
        combos.append(("sequence_only", groups["sequence"]))
        combos.append(("structure_only", groups["structure"]))
        combos.append(("thermo_only", groups["thermo"]))
        combos.append(("no_ensemble_prefix", [i for i,n in enumerate(names) if not n.startswith(('ens_', 'unpaired_w', 'unpaired_max', 'unpaired_ent', 'flex_geom', 'dG_ctx'))]))
        combos.append(("ensemble_prefix_only", [i for i,n in enumerate(names) if n.startswith(('ens_', 'unpaired_w', 'unpaired_max', 'unpaired_ent', 'flex_geom', 'dG_ctx'))]))

        res = {}
        for label, cols in combos:
            if len(cols) == 0 or len(cols) == len(names):
                continue
            scaler_ab = RNADataset(path, split="train", feature_scale=None).fit_scaler(cols=cols)
            tr = RNADataset(path, split="train", feature_scale=scaler_ab, cols=cols)
            va = RNADataset(path, split="val", feature_scale=scaler_ab, cols=cols)
            te = RNADataset(path, split="test", feature_scale=scaler_ab, cols=cols)
            aucs = []
            for seed in SEEDS:
                model = train_one(tr, va, seed, pos_weight)
                m = eval_model(model, te)
                aucs.append(m["auc"])
            res[label] = {"n_feats": int(len(cols)), "aucs": [float(a) for a in aucs], "mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=0)), "median": float(np.median(aucs))}
        out[proto] = res

    with open(os.path.join(RESULTS, "flexnet_ablations.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
