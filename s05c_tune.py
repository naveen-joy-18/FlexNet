"""
Stage 5c: quick architecture/objective check on the validation split only.

The neural model was scoring well below a gradient-boosted tree on the same
features, so this sweeps the objective weighting and optimiser settings to find
out whether the gap comes from the multi-task auxiliary losses or from the
optimiser. The test split is never read here; selection is on the validation
split of the training chemistry alone.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (FlexNet, RNADataset, collate, multitask_loss, FEAT)     # noqa: E402
import evalkit as EK                                                        # noqa: E402
import s05_train as T                                                       # noqa: E402

DEV = torch.device("cpu")


def run(tr, va, seed, pw, epochs, lr, wd, weights, label, patience=6):
    torch.manual_seed(seed)
    model = FlexNet(n_features=tr.n_features).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = max(1, epochs * (len(tr) // 32 + 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=steps, pct_start=0.3)
    tl = T.make_loader(tr, shuffle=True, seed=seed)
    vl = T.make_loader(va)
    vy, vr, vv, vlen, _, _, _ = T.flat_eval(va)
    vmlen = EK.measured_lengths(vv, vlen)
    mm = vv > 0
    best, best_auc = -np.inf, -np.inf
    bad = 0
    for ep in range(epochs):
        model.train()
        for b in tl:
            opt.zero_grad()
            loss, _ = multitask_loss(model(b.X, b.mask), b, pos_weight=pw, **weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
        s, reg, _, lens = T.predict(model, vl)
        st, rt = T.unpack(s, lens), T.unpack(reg, lens)
        m = EK.point_metrics(vy[mm], st[mm])
        rho = EK.per_rna_correlations(vr[mm], st[mm], vmlen)["median_rna_spearman"]
        if np.isfinite(m["auc"]) and m["auc"] > best_auc:
            best_auc, best = m["auc"], (st.copy(), rt.copy())
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    st, rt = best
    m = EK.point_metrics(vy[mm], st[mm])
    rho = EK.per_rna_correlations(vr[mm], st[mm], vmlen)["median_rna_spearman"]
    print(f"  {label:52s} val_auc={m['auc']:.4f} val_ap={m['auprc']:.4f} "
          f"medRho={rho:+.3f} ep={ep + 1}", flush=True)
    return m["auc"], rho


def main():
    dms = os.path.join(FEAT, "dms_modelling.npz")
    tr0 = RNADataset(dms, split="train")
    sc = tr0.fit_scaler()
    tr = RNADataset(dms, split="train", feature_scale=sc)
    va = RNADataset(dms, split="val", feature_scale=sc)
    yv = int(np.nansum(tr.y_bin[tr.valid]))
    nv = int(tr.valid.sum())
    pw = float(max(1.0, (nv - yv) / yv))
    print(f"train {len(tr)} RNAs, prevalence {yv / nv:.3f}, pos_weight {pw:.2f}\n")

    full = dict(w_point=1.0, w_reg=0.3, w_region=0.5, w_rank=0.0)
    point = dict(w_point=1.0, w_reg=0.0, w_region=0.0, w_rank=0.0)
    heavy = dict(w_point=1.0, w_reg=0.5, w_region=1.0, w_rank=0.3)

    t0 = time.time()
    configs = [
        ("decoupled aux .3/.5", dict(full), 2e-3, 1e-4, 30),
        ("point-only (reference)", dict(point), 2e-3, 1e-4, 30),
        ("decoupled heavy .5/1/.3", dict(heavy), 2e-3, 1e-4, 30),
        ("decoupled aux lr=3e-3", dict(full), 3e-3, 1e-4, 30),
        ("decoupled aux lr=1e-3 ep=40", dict(full), 1e-3, 1e-4, 40),
    ]
    out = {}
    for label, w, lr, wd, ep in configs:
        auc, rho = run(tr, va, 0, pw, ep, lr, wd, w, label)
        out[label] = dict(auc=auc, rho=rho, lr=lr, wd=wd, epochs=ep,
                          weights={k: v for k, v in w.items()})
    print(f"\n[total {time.time() - t0:.0f}s]")
    with open(os.path.join(FEAT, "..", "..", "results", "tuning.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
