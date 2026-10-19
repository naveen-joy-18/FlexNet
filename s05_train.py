"""
Stage 5: training and evaluation of FlexNet.

Two complementary protocols, neither of which lets the model see the reactivity
it is scored against:

  Protocol A  train on DMS-MaPseq, report on the held-out DMS families and on
              the independent SHAPE-MaP atlas (cross-chemistry).
  Protocol B  train on SHAPE-MaP,   report on the held-out SHAPE families and on
              the independent DMS-MaPseq atlas (cross-chemistry, mirrored).

Within each protocol miRNA *families* - not nucleotides and not individual
hairpins - are partitioned once, over the union of both chemistries, so
hsa-let-7a-1/-2/-3 and every paralogue sit on the same side of the split and no
near-identical hairpin can straddle it.

The decision threshold T and the minimum region length L_min are chosen once, on
the validation split of the training chemistry, and are then applied unchanged
to every other dataset. Each protocol is repeated over three seeds; the seed
with the median own-test AUROC is reported and the spread is stored.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (FlexNet, RNADataset, collate, multitask_loss,               # noqa: E402
                   ROOT, FEAT, RESULTS)
import evalkit as EK                                                        # noqa: E402

DEV = torch.device("cpu")

EPOCHS = 40          # early stopping typically fires around epoch 11-15
BATCH = 32
LR = 3e-3            # selected on the DMS validation split (s05c_tune.py)
WD = 1e-4
PATIENCE = 8

SEEDS = (0, 1, 2)


def unpack(padded, lens):
    off, out = 0, []
    for n in lens:
        out.append(padded[off:off + n])
        off += n
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def predict(model, loader):
    """Score a loader and return the flat per-nucleotide layout."""
    model.eval()
    S, R, G, lens = [], [], [], []
    for b in loader:
        point, reg, region = model(b.X, b.mask)
        keep = b.mask.bool()
        S.append(torch.sigmoid(point)[keep].numpy())
        R.append(reg[keep].numpy())
        G.append(torch.sigmoid(region)[keep].numpy())
        lens.append(b.lengths.numpy())
    return (np.concatenate(S), np.concatenate(R), np.concatenate(G),
            np.concatenate(lens))


def make_loader(ds, batch=BATCH, shuffle=False, seed=0):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed) if shuffle else None)


def flat_eval(ds):
    """Unpadded per-nucleotide targets for a dataset, in loader order."""
    idx = ds.index
    ptr = ds.ptr
    seg = lambda a: np.concatenate([a[int(ptr[i]):int(ptr[i + 1])] for i in idx])
    y = seg(np.nan_to_num(ds.y_bin))
    yv = seg(ds.valid.astype(np.float32))
    yr = seg(np.nan_to_num(ds.y_rank, nan=0.0))
    lengths = np.array([int(ptr[i + 1] - ptr[i]) for i in idx])
    ids = np.array([ds.exp_id[i] for i in idx])
    fam = np.array([ds.family[i] for i in idx])
    per = np.concatenate([np.full(int(ptr[i + 1] - ptr[i]), ds.exp_id[i]) for i in idx])
    return y, yr, yv, lengths, ids, fam, per


def train_one(tr, va, seed, label, pos_weight):
    torch.manual_seed(seed)
    model = FlexNet(n_features=tr.n_features).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, EPOCHS * (len(tr) // BATCH + 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.3)
    tl = make_loader(tr, shuffle=True, seed=seed)
    vl = make_loader(va)
    va_y, va_yr, va_v, va_len, _, _, _ = flat_eval(va)
    va_mlen = EK.measured_lengths(va_v, va_len)

    hist = dict(train_loss=[], val_loss=[], val_auc=[], val_spearman=[],
                lr=[], train_point=[], train_reg=[], train_region=[],
                train_rank=[], early_stop=None, best_epoch=None)
    best_auc, best_state, bad = -np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        agg = []
        for b in tl:
            opt.zero_grad()
            loss, sub = multitask_loss(model(b.X, b.mask), b, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            agg.append(sub)
        s, reg, _, lens = predict(model, vl)
        s_t, reg_t = unpack(s, lens), unpack(reg, lens)
        mm = va_v > 0
        m = EK.point_metrics(va_y[mm], s_t[mm])
        tr_mean = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
        hist["train_loss"].append(tr_mean["total"])
        for k in ("point", "reg", "region", "rank"):
            hist[f"train_{k}"].append(tr_mean[k])
        hist["val_loss"].append(float(np.mean((reg_t[mm] - va_yr[mm]) ** 2)))
        hist["val_auc"].append(m["auc"])
        hist["val_spearman"].append(
            EK.per_rna_correlations(va_yr[mm], s_t[mm], va_mlen)[
                "median_rna_spearman"])
        hist["lr"].append(float(opt.param_groups[0]["lr"]))
        if np.isfinite(m["auc"]) and m["auc"] > best_auc:
            best_auc, bad = m["auc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = ep + 1
        else:
            bad += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  {label} ep{ep + 1:3d} loss={tr_mean['total']:.4f} "
                  f"val_auc={m['auc']:.4f}", flush=True)
        if bad >= PATIENCE:
            hist["early_stop"] = ep + 1
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def score_split(model, dpath, scaler, split=None):
    ds = RNADataset(dpath, split=split, feature_scale=scaler)
    s, reg, region, lens = predict(model, make_loader(ds))
    y, yr, valid, lengths, ids, fam, per = flat_eval(ds)
    assert len(s) == len(y), (len(s), len(y))
    return dict(score=s, reg=reg, region=region, y=y, y_rank=yr, valid=valid,
                lengths=lengths, ids=ids, family=fam, rna=per,
                struct=ds.struct)


def evaluate(pred, name, thr, min_len, with_ci=False):
    mm = pred["valid"] > 0
    mlen = EK.measured_lengths(pred["valid"], pred["lengths"])
    m = EK.point_metrics(pred["y"][mm], pred["score"][mm], thr)
    m.update(EK.per_rna_point_metrics(pred["y"][mm], pred["score"][mm], mlen, thr))
    m.update(EK.per_rna_correlations(pred["y_rank"][mm], pred["score"][mm], mlen))
    aux = EK.point_metrics(pred["y"][mm], pred["region"][mm], thr)
    m["aux_region_auc"] = aux["auc"]
    m["aux_region_auprc"] = aux["auprc"]
    auxr = EK.correlations(pred["y_rank"][mm], pred["reg"][mm])
    m["aux_reg_corr"] = auxr["pearson"]
    m["corr_raw"] = EK.correlations(pred["y_rank"][mm], pred["score"][mm])
    m["corr_rank"] = EK.correlations(pred["y_rank"][mm], pred["reg"][mm])
    m["corr_region"] = EK.correlations(pred["y_rank"][mm], pred["region"][mm])
    if with_ci:
        # 1000 hairpin-level resamples; only run for the seed actually reported
        m["auc_ci"] = EK.bootstrap_ci(pred["y"][mm], pred["score"][mm], "auc", 1000,
                                      groups=pred["rna"][mm])
        m["auprc_ci"] = EK.bootstrap_ci(pred["y"][mm], pred["score"][mm], "auprc",
                                        1000, groups=pred["rna"][mm])
    called = (pred["score"] >= thr) & pred["valid"].astype(bool)
    m["regions"] = EK.region_level_scores(pred["y"], called, pred["lengths"], min_len)
    m["n_rnas"] = int(len(pred["lengths"]))
    m["n_measured"] = int(mm.sum())
    m["name"] = name
    print(f"    {name:28s} n_rna={m['n_rnas']:4d} auc={m['auc']:.4f} "
          f"auprc={m['auprc']:.4f} acc={m['accuracy']:.4f} sen={m['sensitivity']:.3f} "
          f"spec={m['specificity']:.3f} mcc={m['mcc']:.3f} f1={m['f1']:.3f} "
          f"rho={m['corr_raw']['spearman']:+.3f} "
          f"medRho={m['median_rna_spearman']:+.3f}", flush=True)
    return m


def main():
    t0 = time.time()
    paths = {k: os.path.join(FEAT, f"{k}_modelling.npz") for k in ("dms", "shape")}
    train_all = RNADataset(paths["dms"], split="train")
    scaler = train_all.fit_scaler()          # fitted on the DMS training split only

    metrics, histories, predictions, cal = {}, {}, {}, {}

    protocols = (
        ("A", "dms", "shape", "DMS-MaPseq", "SHAPE-MaP"),
        ("B", "shape", "dms", "SHAPE-MaP", "DMS-MaPseq"),
    )

    for proto, own_tag, other_tag, own_name, other_name in protocols:
        path = paths[own_tag]
        tr = RNADataset(path, split="train", feature_scale=scaler)
        va = RNADataset(path, split="val", feature_scale=scaler)
        yv = int(np.nansum(tr.y_bin[tr.valid]))
        nv = int(tr.valid.sum())
        pos_weight = float(max(1.0, (nv - yv) / max(yv, 1)))
        print(f"\n=== Protocol {proto}: train on {own_name} ({len(tr)} RNAs) "
              f"-> validate against {other_name} ===", flush=True)
        print(f"  train prevalence {yv / nv:.3f}, pos_weight {pos_weight:.2f}",
              flush=True)

        seed_metrics, seed_preds, seed_models, seed_cal = [], [], [], []
        for seed in SEEDS:
            model, hist = train_one(tr, va, seed, f"[{proto} s{seed}]", pos_weight)
            histories[f"{proto}_seed{seed}"] = hist

            # ---- threshold and minimum region length: validation split only ----
            val_pred = score_split(model, path, scaler, "val")
            mmv = val_pred["valid"] > 0
            thr_f1, _ = EK.select_threshold(val_pred["y"][mmv],
                                            val_pred["score"][mmv], "f1")
            thr_y, _ = EK.select_threshold(val_pred["y"][mmv],
                                           val_pred["score"][mmv], "youden")
            thr = float(np.median([thr_f1, thr_y]))
            min_len, _ = EK.select_min_length(val_pred["y"], val_pred["score"],
                                              thr, val_pred["lengths"])
            print(f"  seed {seed}: T(f1)={thr_f1:.3f} T(youden)={thr_y:.3f} "
                  f"-> T={thr:.3f}, L_min={min_len}", flush=True)

            preds = {"own_test": score_split(model, path, scaler, "test"),
                     "own_val": score_split(model, path, scaler, "val")}
            for sp in ("val", "test", "all"):
                preds[f"other_{sp}"] = score_split(model, paths[other_tag], scaler,
                                                   None if sp == "all" else sp)
            ev = {k: evaluate(v, f"{proto}/{k}", thr, min_len)
                  for k, v in preds.items()}
            seed_metrics.append(ev)
            seed_preds.append(preds)
            seed_models.append(model)
            seed_cal.append((thr, min_len))
            torch.save(model.state_dict(),
                       os.path.join(RESULTS, f"model_{proto}_s{seed}.pt"))
            np.savez_compressed(os.path.join(RESULTS, f"scaler_{proto}.npy"),
                                mu=scaler[0], sd=scaler[1])

        # Select representative seed using validation AUROC (not test)
        # Report test performance across seeds; choose seed by validation median
        val_aucs = [s["own_val"]["auc"] for s in seed_metrics]
        test_aucs = [s["own_test"]["auc"] for s in seed_metrics]
        med = int(np.argsort(val_aucs)[len(val_aucs) // 2])
        # the reported numbers must use *this* seed's threshold and minimum
        # region length, chosen on validation
        thr, min_len = seed_cal[med]
        chosen = {k: evaluate(seed_preds[med][k], f"{proto}/{k}", thr, min_len,
                              with_ci=True)
                  for k in seed_preds[med]}
        for k, v in chosen.items():
            metrics[f"{proto}/{k}"] = v
        metrics[f"{proto}/seed_aucs_test"] = [float(a) for a in test_aucs]
        metrics[f"{proto}/seed_aucs_val"] = [float(a) for a in val_aucs]
        metrics[f"{proto}/chosen_seed"] = int(SEEDS[med])
        metrics[f"{proto}/auc_spread_test"] = [float(np.min(test_aucs)), float(np.max(test_aucs))]
        metrics[f"{proto}/auc_spread_val"] = [float(np.min(val_aucs)), float(np.max(val_aucs))]
        metrics[f"{proto}/seed_stats_test"] = {
            "mean": float(np.mean(test_aucs)),
            "std": float(np.std(test_aucs, ddof=0)),
            "median": float(np.median(test_aucs)),
            "min": float(np.min(test_aucs)),
            "max": float(np.max(test_aucs)),
        }
        cal[proto] = dict(threshold=thr, min_region_len=int(min_len),
                          own_dataset=own_name, other_dataset=other_name,
                          n_train=int(len(tr)), pos_weight=pos_weight)
        for k, v in seed_preds[med].items():
            predictions[f"{proto}_{k}"] = {kk: vv for kk, vv in v.items()
                                          if kk != "struct"}
        print(f"  protocol {proto}: val AUC across seeds = "
              f"{[round(a, 4) for a in val_aucs]}; test AUC across seeds = "
              f"{[round(a, 4) for a in test_aucs]}; reporting seed {SEEDS[med]} (by val median)",
              flush=True)

    with open(os.path.join(RESULTS, "metrics_main.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    with open(os.path.join(RESULTS, "training_history.json"), "w") as fh:
        json.dump(histories, fh, indent=2, default=float)
    with open(os.path.join(RESULTS, "call_calibration.json"), "w") as fh:
        json.dump(cal, fh, indent=2, default=float)
    np.savez_compressed(os.path.join(RESULTS, "predictions.npz"),
                        **{k: v["score"] for k, v in predictions.items()},
                        **{k + "_reg": v["reg"] for k, v in predictions.items()},
                        **{k + "_region": v["region"] for k, v in predictions.items()},
                        **{k + "_y": v["y"] for k, v in predictions.items()},
                        **{k + "_yrank": v["y_rank"] for k, v in predictions.items()},
                        **{k + "_valid": v["valid"] for k, v in predictions.items()},
                        **{k + "_len": v["lengths"] for k, v in predictions.items()},
                        **{k + "_rna": v["rna"] for k, v in predictions.items()},
                        )
    print(f"\n[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
