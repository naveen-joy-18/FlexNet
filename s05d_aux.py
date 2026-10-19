"""
Stage 5d: auxiliary branch comparisons (Issue 2).
Compare (a) pointwise-only (aux_mode='none'), (b) fully-coupled (aux_mode='coupled'),
(c) detached (aux_mode='detached').
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FlexNet, RNADataset, collate, multitask_loss, FEAT, RESULTS
import evalkit as EK

DEV = torch.device("cpu")
EPOCHS = 40
BATCH = 32
LR = 3e-3
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
    idx = ds.index
    ptr = ds.ptr
    seg = lambda a: np.concatenate([a[int(ptr[i]):int(ptr[i + 1])] for i in idx])
    y = seg(np.nan_to_num(ds.y_bin))
    yv = seg(ds.valid.astype(np.float32))
    yr = seg(np.nan_to_num(ds.y_rank, nan=0.0))
    lengths = np.array([int(ptr[i + 1] - ptr[i]) for i in idx])
    return y, yr, yv, lengths


def train_one(tr, va, seed, aux_mode, pos_weight):
    torch.manual_seed(seed)
    model = FlexNet(n_features=tr.n_features, aux_mode=aux_mode).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, EPOCHS * (len(tr) // BATCH + 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.3)
    tl = make_loader(tr, shuffle=True, seed=seed)
    vl = make_loader(va)
    va_y, va_yr, va_v, va_len = flat_eval(va)

    best_auc, best_state, bad = -np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            opt.zero_grad()
            loss, _ = multitask_loss(model(b.X, b.mask), b, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
        s, _, _, lens = predict(model, vl)
        s_t = unpack(s, lens)
        mm = va_v > 0
        m = EK.point_metrics(va_y[mm], s_t[mm])
        if np.isfinite(m["auc"]) and m["auc"] > best_auc:
            best_auc, bad = m["auc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_model(model, ds):
    s, _, _, lens = predict(model, make_loader(ds))
    y, yr, v, lengths = flat_eval(ds)
    mm = v > 0
    m = EK.point_metrics(y[mm], s[mm])
    return m


def main():
    paths = {k: os.path.join(FEAT, f"{k}_modelling.npz") for k in ("dms", "shape")}
    scaler = RNADataset(paths["dms"], split="train").fit_scaler()

    out = {}
    for proto, (own_tag, _) in [("A", ("dms", "shape")), ("B", ("shape", "dms"))]:
        path = paths[own_tag]
        tr = RNADataset(path, split="train", feature_scale=scaler)
        va = RNADataset(path, split="val", feature_scale=scaler)
        te = RNADataset(path, split="test", feature_scale=scaler)
        yv = int(np.nansum(tr.y_bin[tr.valid]))
        nv = int(tr.valid.sum())
        pos_weight = float(max(1.0, (nv - yv) / max(yv, 1)))

        res = {}
        for mode in ["none", "coupled", "detached"]:
            aucs = []
            for seed in SEEDS:
                model = train_one(tr, va, seed, mode, pos_weight)
                m = eval_model(model, te)
                aucs.append(m["auc"])
            res[mode] = {
                "aucs": [float(a) for a in aucs],
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs, ddof=0)),
                "median": float(np.median(aucs)),
            }
        out[proto] = res

    with open(os.path.join(RESULTS, "auxiliary_comparisons.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
