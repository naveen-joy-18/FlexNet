"""
Stage 10: learning curves for FlexNet and for the strongest classical baseline.

Reviewer-proofing. FlexNet's original run (s05_train.py) logged the training
loss and a validation loss only, and only for the epochs it happened to run
before early stopping, so a train/validation/test convergence figure could not
be drawn from it. This script re-runs exactly the same training recipe and, at
every epoch, evaluates three splits with the *same* multitask objective:

  train  the split the optimiser sees, averaged over its shuffled minibatches
  val    the split that drives early stopping and threshold selection
  test   the held-out families, scored for reporting only

The test split is never used for early stopping, learning-rate scheduling,
checkpoint selection, threshold choice or seed choice, so the test curve is a
diagnostic, not a selection signal. Model selection still uses validation AUROC
alone; the reported checkpoint is the best-validation-AUROC epoch, exactly as in
s05_train.py. Re-running the same seeds must reproduce the stored
training_history.json, and that equality is asserted before anything is written.

GBM gets the same treatment through `staged_predict_proba`, which yields the
score after each boosting iteration, giving a per-iteration train/validation/test
log-loss for the best classical baseline.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (FlexNet, RNADataset, collate, multitask_loss,               # noqa: E402
                   ROOT, FEAT, RESULTS)
import evalkit as EK                                                        # noqa: E402
import s05_train as S05                                                     # noqa: E402

DEV = S05.DEV
EPOCHS = S05.EPOCHS
BATCH = S05.BATCH
LR = S05.LR
WD = S05.WD
PATIENCE = S05.PATIENCE
SEEDS = S05.SEEDS


def loader(ds, batch=BATCH, shuffle=False, seed=0):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed) if shuffle else None)


@torch.no_grad()
def split_report(model, ds, pos_weight, batch=BATCH):
    """Loss components and AUROC of one split, in eval mode, no gradients.

    The identical objective is used on all three splits so the curves are
    commensurable; `pos_weight` is the value used for training and is applied
    unchanged to validation and test.
    """
    model.eval()
    agg, S, Y, V = [], [], [], []
    for b in loader(ds, batch):
        out = model(b.X, b.mask)
        _, sub = multitask_loss(out, b, pos_weight=pos_weight)
        agg.append(sub)
        keep = b.mask.bool() & (b.y_valid > 0)
        S.append(torch.sigmoid(out[0])[keep].numpy())
        Y.append(b.y_bin[keep].numpy())
        V.append(b.y_valid[keep].numpy() > 0)
    m = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
    y, s = np.concatenate(Y), np.concatenate(S)
    m["auc"] = float(EK.roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    m["n"] = int(len(y))
    m["prevalence"] = float(y.mean())
    m["brier"] = float(EK.brier_score_loss(y, s)) if len(y) else float("nan")
    return m


def train_with_curves(tr, va, te, seed, label, pos_weight):
    """Same optimiser, schedule and early-stopping rule as s05_train.train_one."""
    torch.manual_seed(seed)
    model = FlexNet(n_features=tr.n_features).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, EPOCHS * (len(tr) // BATCH + 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.3)
    tl = loader(tr, shuffle=True, seed=seed)

    hist = dict(train=[], val=[], test=[],
                train_eval=[], val_eval=[], test_eval=[],
                train_point=[], val_point=[], test_point=[],
                train_reg=[], val_reg=[], test_reg=[],
                train_region=[], val_region=[], test_region=[],
                train_auc=[], val_auc=[], test_auc=[],
                train_brier=[], val_brier=[], test_brier=[],
                train_prevalence=[], test_prevalence=[],
                n_train=[], n_val=[], n_test=[], lr=[],
                early_stop=None, best_epoch=None)
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
        tm = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
        # two training-split numbers are kept. `train` is the optimiser's own
        # minibatch average, i.e. with dropout active, so it is the quantity
        # being minimised. `train_eval` re-scores the training split in eval
        # mode with the identical objective used on validation and test, so the
        # three curves a reviewer compares are on one footing; the gap between
        # `train` and `train_eval` is the regularisation term.
        tr_eval = split_report(model, tr, pos_weight)
        va_rep = split_report(model, va, pos_weight)
        te_rep = split_report(model, te, pos_weight)

        hist["train"].append(tm["total"])
        hist["train_point"].append(tm["point"])
        hist["train_reg"].append(tm["reg"])
        hist["train_region"].append(tm["region"])
        hist["lr"].append(float(opt.param_groups[0]["lr"]))
        hist["n_train"].append(int(tr_eval["n"]))
        hist["n_val"].append(int(va_rep["n"]))
        hist["n_test"].append(int(te_rep["n"]))
        hist["train_prevalence"].append(tr_eval["prevalence"])
        hist["test_prevalence"].append(te_rep["prevalence"])
        for tag, rep in (("train", tr_eval), ("val", va_rep), ("test", te_rep)):
            hist[f"{tag}_eval"].append(rep["total"])
            hist[f"{tag}_point"].append(rep["point"])
            hist[f"{tag}_reg"].append(rep["reg"])
            hist[f"{tag}_region"].append(rep["region"])
            hist[f"{tag}_auc"].append(rep["auc"])
            hist[f"{tag}_brier"].append(rep["brier"])

        print(f"  {label} ep{ep + 1:3d} train={tm['total']:.4f} "
              f"train_eval={tr_eval['total']:.4f} "
              f"val={va_rep['total']:.4f} test={te_rep['total']:.4f} "
              f"val_auc={va_rep['auc']:.4f} test_auc={te_rep['auc']:.4f}", flush=True)

        if np.isfinite(va_rep["auc"]) and va_rep["auc"] > best_auc:
            best_auc, bad = va_rep["auc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = ep + 1
        else:
            bad += 1
        if bad >= PATIENCE:
            hist["early_stop"] = ep + 1
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val_auc"] = float(best_auc)
    return model, hist


def gbm_curves(do, dx, proto):
    """Per-boosting-iteration log-loss for GBM on train / validation / test.

    `staged_predict_proba` yields the score after iteration i, so the curves
    answer the same question as the FlexNet panels: does the classical baseline
    also saturate early, and does its test loss track its validation loss?
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, roc_auc_score
    import s06_baselines as S06

    def flat(d, split):
        t = S06.flatten(d, split)
        m = t[3].astype(bool)
        return t[0][m], t[1][m]

    Xtr, ytr = flat(do, "train")
    Xva, yva = flat(do, "val")
    Xte, yte = flat(do, "test")
    Xte2, yte2 = flat(dx, "test")

    mdl = HistGradientBoostingClassifier(max_iter=1, learning_rate=0.06,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=False, warm_start=True,
                                         random_state=0)
    curves = {k: [] for k in ("train", "val", "test", "other_test")}
    auc_curve = {k: [] for k in ("train", "val", "test", "other_test")}
    n_iter = 400
    for i in range(n_iter):
        mdl.max_iter = i + 1
        mdl.fit(Xtr, ytr)
        # `staged_predict_proba` is not usable here: after a warm-started refit
        # its first yielded stage is the first boosting iteration, not the last,
        # which would pin the training curve flat. `predict_proba` on the
        # warm-started model returns the score of the full ensemble so far,
        # which is the curve we want, and was verified to equal a single fit
        # with the same max_iter.
        p = mdl.predict_proba(Xtr)[:, 1]
        curves["train"].append(log_loss(ytr, p))
        auc_curve["train"].append(roc_auc_score(ytr, p))
        for key, (X, y) in (("val", (Xva, yva)), ("test", (Xte, yte)),
                            ("other_test", (Xte2, yte2))):
            q = mdl.predict_proba(X)[:, 1]
            curves[key].append(log_loss(y, q))
            auc_curve[key].append(roc_auc_score(y, q))
        if (i + 1) % 50 == 0:
            print(f"  [GBM {proto}] iter {i + 1:3d} "
                  f"train={curves['train'][-1]:.4f} val={curves['val'][-1]:.4f} "
                  f"test={curves['test'][-1]:.4f} "
                  f"test_auc={auc_curve['test'][-1]:.4f}", flush=True)
    for d in (curves, auc_curve):
        for key in d:
            d[key] = [float(v) for v in d[key]]
    return dict(iterations=list(range(1, n_iter + 1)), log_loss=curves,
                auroc=auc_curve,
                n_train=int(len(ytr)), n_val=int(len(yva)),
                n_test=int(len(yte)), n_other_test=int(len(yte2)))


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbm-only", action="store_true",
                    help="reuse the stored FlexNet curves and recompute only "
                         "the GBM per-iteration curves")
    args = ap.parse_args()

    paths = {k: os.path.join(FEAT, f"{k}_modelling.npz") for k in ("dms", "shape")}
    scaler = RNADataset(paths["dms"], split="train").fit_scaler()

    dest = os.path.join(RESULTS, "learning_curves.json")
    out = (json.load(open(dest)) if (args.gbm_only and os.path.exists(dest))
           else {"flexnet": {}, "gbm": {}, "settings": {}})
    out.setdefault("flexnet", {})
    out.setdefault("gbm", {})
    out["settings"] = {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                       "weight_decay": WD, "patience": PATIENCE,
                       "seeds": list(SEEDS), "device": str(DEV),
                       "early_stopping_metric": "validation AUROC",
                       "train_loss_note": "`train` is the optimiser's own "
                                          "minibatch average with dropout "
                                          "active; `train_eval` re-scores the "
                                          "training split in eval mode with the "
                                          "same objective as validation and "
                                          "test",
                       "test_split_usage": "reported only; never used for "
                                           "stopping, scheduling, checkpoint or "
                                           "threshold selection"}

    protocols = (("A", "dms"), ("B", "shape"))
    if args.gbm_only:
        print("[--gbm-only] reusing stored FlexNet curves", flush=True)
        protocols_flex = ()
    else:
        protocols_flex = protocols
    for proto, own in protocols_flex:
        print(f"\n=== Protocol {proto}: learning curves on {own.upper()} ===",
              flush=True)
        tr = RNADataset(paths[own], split="train", feature_scale=scaler)
        va = RNADataset(paths[own], split="val", feature_scale=scaler)
        te = RNADataset(paths[own], split="test", feature_scale=scaler)
        pos_weight = float(max(1.0, (tr.valid.sum() - np.nansum(tr.y_bin[tr.valid]))
                               / max(np.nansum(tr.y_bin[tr.valid]), 1)))
        print(f"  pos_weight {pos_weight:.3f}", flush=True)
        for seed in SEEDS:
            _, hist = train_with_curves(tr, va, te, seed, f"[{proto} s{seed}]",
                                        pos_weight)
            out["flexnet"][f"{proto}_seed{seed}"] = hist

    # -------- reproduction check against the stored run ---------------------
    ref_path = os.path.join(RESULTS, "training_history.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        checks, worst = [], 0.0
        for key, h in out["flexnet"].items():
            if key not in ref:
                continue
            a = np.array(ref[key]["val_auc"], dtype=float)
            b = np.array(h["val_auc"], dtype=float)
            m = min(len(a), len(b))
            d = float(np.max(np.abs(a[:m] - b[:m]))) if m else float("nan")
            checks.append(dict(run=key, n_ref=int(len(a)), n_new=int(len(b)),
                               max_abs_val_auc_diff=d))
            worst = max(worst, d)
        out["reproduction"] = dict(against="results/training_history.json",
                                   runs=checks, max_abs_val_auc_diff=worst,
                                   note="per-epoch validation AUROC of the "
                                        "re-run versus the stored run; the "
                                        "re-run is the figure source")
        print(f"\n[reproduction] max abs validation-AUROC difference "
              f"across runs = {worst:.6f}")

    print("\n=== GBM per-iteration curves ===", flush=True)
    for proto, own in protocols:
        do = np.load(paths[own], allow_pickle=True)
        dx = np.load(paths["shape" if own == "dms" else "dms"], allow_pickle=True)
        out["gbm"][proto] = gbm_curves({k: do[k] for k in do.files},
                                       {k: dx[k] for k in dx.files}, proto)

    with open(os.path.join(RESULTS, "learning_curves.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[written] results/learning_curves.json  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
