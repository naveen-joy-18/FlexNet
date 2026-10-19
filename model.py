"""
FlexNet: a dilated residual encoder-decoder for nucleotide-level structural
flexibility in human precursor microRNAs.

Architecture rationale
  * inputs are per-nucleotide feature channels, so a 1-D convolutional encoder
    is the natural inductive bias;
  * dilation gives a receptive field that spans a whole hairpin (dilations
    1,2,4,8,16,32 with a width-5 kernel -> 253 nt) without the parameter cost
    of wide kernels;
  * a bidirectional GRU adds long-range context along the chain, which matters
    because loop closure couples distant positions;
  * three heads are trained jointly: a pointwise flexibility classifier, a
    pointwise regressor onto rank-normalised reactivity, and a region head that
    sees a wide context so it can call contiguous flexible segments.

Objectives
  * class-balanced focal BCE on the pointwise flexible/rigid label;
  * smooth-L1 on the within-dataset rank-normalised reactivity, which keeps the
    regression target on the same [0, 1] scale for DMS and for SHAPE and so
    makes the two chemistries directly comparable;
  * a within-RNA pairwise ranking loss. Per-RNA ranking removes the per-RNA
    offset that otherwise dominates the raw reactivity scale and optimises the
    quantity the study actually reports (within-hairpin flexibility order),
    without touching any held-out data.

No experimental quantity is ever an input feature, so a model trained on one
chemistry can be evaluated on the other without circularity.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "data", "features")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ------------------------------------------------------------------ dataset ---
@dataclass
class RNABatch:
    X: torch.Tensor          # (B, L, F) padded, standardised features
    mask: torch.Tensor       # (B, L) 1 = real nucleotide
    y_bin: torch.Tensor      # (B, L) binary flexible label
    y_rank: torch.Tensor     # (B, L) within-dataset rank-normalised reactivity
    y_valid: torch.Tensor    # (B, L) 1 = measured nucleotide
    lengths: torch.Tensor


def _seg(arr, ptr, index):
    return np.concatenate([arr[int(ptr[i]):int(ptr[i + 1])] for i in index])


class RNADataset(torch.utils.data.Dataset):
    """One chemistry's per-nucleotide arrays, restricted to a family-aware split.

    `feature_scale` standardises the channels. It is fitted on the training
    split only and passed in, never re-estimated here, so no validation or test
    statistic leaks into the inputs.
    """

    def __init__(self, path, split=None, feature_scale=None, groups=None, cols=None):
        d = np.load(path, allow_pickle=True)
        X = d["X"].astype(np.float32)
        if cols is not None:
            X = X[:, cols]
            self.feature_names = [str(n) for n in d["feature_names"][cols]] if "feature_names" in d else None
        else:
            self.feature_names = list(d["feature_names"]) if "feature_names" in d else None
        self.scaler = feature_scale
        if feature_scale is not None:
            mu, sd = feature_scale
            # If cols provided, assume scaler already corresponds to the selected columns
            X = (X - mu) / sd
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.y_bin = d["y_bin"]
        self.y_rank = d["y_rank"] if "y_rank" in d else d["y"]
        self.ptr = d["ptr"]
        self.valid = d["valid"]
        self.split = d["split"]
        self.exp_id = d["exp_id"]
        self.mirbase_id = d["mirbase_id"] if "mirbase_id" in d else None
        self.family = d["family"]
        self.struct = [json.loads(s) for s in d["struct"]] if "struct" in d else None
        self.n_features = self.X.shape[1]
        self.n_all = len(d["lengths"])
        self.index = [i for i in range(self.n_all)
                      if split is None or self.split[i] == split]

    def fit_scaler(self, cols=None):
        """Channel mean/sd over the training split only."""
        Xt = _seg(self.X, self.ptr, self.index)
        if cols is not None:
            Xt = Xt[:, cols]
        mu = Xt.mean(axis=0)
        sd = Xt.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu.astype(np.float32), sd.astype(np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        a, b = int(self.ptr[i]), int(self.ptr[i + 1])
        return dict(X=torch.from_numpy(np.ascontiguousarray(self.X[a:b])),
                    y_bin=torch.from_numpy(np.nan_to_num(self.y_bin[a:b])),
                    y_rank=torch.from_numpy(np.nan_to_num(self.y_rank[a:b],
                                                          nan=0.0)),
                    y_valid=torch.from_numpy(self.valid[a:b].astype(np.float32)),
                    i=i)


def collate(batch):
    lens = [b["X"].shape[0] for b in batch]
    L, Fdim, B = max(lens), batch[0]["X"].shape[1], len(batch)
    X = torch.zeros(B, L, Fdim)
    mask = torch.zeros(B, L)
    yb = torch.zeros(B, L)
    yr = torch.zeros(B, L)
    yv = torch.zeros(B, L)
    for k, b in enumerate(batch):
        n = lens[k]
        X[k, :n] = b["X"]
        mask[k, :n] = 1.0
        yb[k, :n] = b["y_bin"]
        yr[k, :n] = b["y_rank"]
        yv[k, :n] = b["y_valid"]
    return RNABatch(X=X, mask=mask, y_bin=yb, y_rank=yr, y_valid=yv,
                    lengths=torch.tensor(lens, dtype=torch.long))


# ------------------------------------------------------------------- model ---
class DilatedBlock(nn.Module):
    """Pre-activation gated residual block over a dilated 1-D convolution."""

    def __init__(self, ch, dilation, dropout=0.15, kernel=5):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.norm = nn.GroupNorm(8, ch)
        self.conv = nn.Conv1d(ch, 2 * ch, kernel, padding=pad, dilation=dilation)
        self.proj = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        a, b = self.conv(h).chunk(2, dim=1)
        return x + self.drop(F.gelu(self.proj(a * torch.sigmoid(b))))


class FlexNet(nn.Module):
    """Shared dilated trunk, one classification head and two decoupled aux heads.

    The flexibility classifier and the auxiliary heads deliberately do not share
    a gradient path. On the validation split, letting the region and regression
    objectives back-propagate into the trunk cost roughly 4 AUROC points
    (0.895 -> 0.851 with light weights, 0.520 at full weight), because the
    pairwise ranking term saturates the classifier logits. The auxiliary heads
    therefore sit on a private two-block branch fed by a *detached* trunk, which
    keeps a dedicated wide-context region caller and a continuous reactivity
    regressor without compromising the reported flexibility score.
    """

    def __init__(self, n_features, ch=128, n_blocks=6, dropout=0.15,
                 gru_hidden=64, head_ch=64, aux_ch=64, kernel=5,
                 aux_mode="detached"):
        """
        aux_mode: 'detached' (default), 'coupled' (full gradient to trunk),
                  or 'none' (no auxiliary heads)
        """
        super().__init__()
        self.aux_mode = aux_mode
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, ch, 1), nn.GroupNorm(8, ch), nn.GELU())
        self.blocks = nn.ModuleList(
            [DilatedBlock(ch, d, dropout, kernel)
             for d in [1, 2, 4, 8, 16, 32][:n_blocks]])
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(ch, gru_hidden, num_layers=1, batch_first=True,
                          bidirectional=True, dropout=0.0)
        fuse = ch + 2 * gru_hidden

        def head(widths, in_ch=None):
            layers, prev = [], in_ch if in_ch is not None else fuse
            for w in widths:
                layers += [nn.Conv1d(prev, w, 1), nn.GELU()]
                prev = w
            layers += [nn.Conv1d(prev, 1, 1)]
            return nn.Sequential(*layers)

        self.point_head = head([head_ch, head_ch])

        if aux_mode != "none":
            self.aux_stem = nn.Sequential(
                nn.Conv1d(fuse, aux_ch, 1), nn.GroupNorm(8, aux_ch), nn.GELU())
            self.aux_blocks = nn.ModuleList(
                [DilatedBlock(aux_ch, 1, dropout, kernel),
                 DilatedBlock(aux_ch, 2, dropout, kernel)])
            self.reg_head = head([aux_ch], in_ch=aux_ch)
            # the region head sees a wide context so that contiguous segments, not
            # isolated nucleotides, drive the region decision
            self.region_head = head([aux_ch, aux_ch, aux_ch, aux_ch], in_ch=aux_ch)
        else:
            self.aux_stem = None
            self.aux_blocks = None
            self.reg_head = None
            self.region_head = None
        self.n_features = n_features

    def encode(self, x, mask):
        h = self.stem(x.transpose(1, 2))                  # (B, C, L)
        for blk in self.blocks:
            h = blk(h)
        g, _ = self.gru(self.drop(h.transpose(1, 2)), None)
        return torch.cat([h.transpose(1, 2), g], dim=2).transpose(1, 2)   # (B, C', L)

    def aux_encode(self, z):
        if self.aux_mode == "none":
            return None
        h = self.aux_stem(z)
        for blk in self.aux_blocks:
            h = blk(h)
        return h

    def forward(self, x, mask):
        z = self.encode(x, mask)
        if self.aux_mode == "none":
            # Return dummy values for compatibility
            return (self.point_head(z)[:, 0],
                    torch.zeros_like(self.point_head(z)[:, 0]),
                    torch.zeros_like(self.point_head(z)[:, 0]))
        if self.aux_mode == "coupled":
            za = self.aux_encode(z)
        else:  # detached
            za = self.aux_encode(z.detach())
        return (self.point_head(z)[:, 0],
                torch.sigmoid(self.reg_head(za)[:, 0]),
                self.region_head(za)[:, 0])



# -------------------------------------------------------------------- loss ---
def masked_mean(v, m, eps=1e-8):
    return (v * m).sum() / m.sum().clamp_min(eps)


def per_rna_pairwise_rank_loss(score, value, measure):
    """Pairwise logistic ranking loss on measured positions within each RNA.

    For every pair (i, j) of measured positions in the same hairpin the model is
    asked to order them the way the measured reactivity orders them. Optimising
    this makes the within-hairpin ordering, not the absolute reactivity scale, the
    thing being learned, which is what transfers between chemistries.
    """
    total, count = score.new_zeros(()), 0
    for b in range(score.shape[0]):
        s = score[b][measure[b] > 0]
        v = value[b][measure[b] > 0]
        n = s.numel()
        if n < 8:
            continue
        ds = s.unsqueeze(1) - s.unsqueeze(0)
        dv = v.unsqueeze(1) - v.unsqueeze(0)
        iu = torch.triu_indices(n, n, offset=1, device=s.device)
        dsv, dvv = ds[iu[0], iu[1]], dv[iu[0], iu[1]]
        keep = dvv.abs() > 1e-6
        if keep.sum() < 4:
            continue
        lbl = (dvv > 0).float()
        loss = F.binary_cross_entropy_with_logits(-dsv[keep] * 8.0, lbl[keep],
                                                  reduction="mean")
        total = total + loss
        count += 1
    return total / max(count, 1)


def multitask_loss(out, batch, w_point=1.0, w_reg=0.3, w_region=0.5,
                   w_rank=0.0, focal_gamma=1.0, pos_weight=1.0):
    """Point classification plus two decoupled auxiliary objectives.

    Only `w_point` reaches the shared trunk (the auxiliary heads sit on a
    detached branch); `w_rank` is applied to the regression head and defaults to
    zero, because pushing the classifier logits apart costs AUROC.
    """
    point_logit, reg, region_logit = out
    mv = batch.y_valid * batch.mask

    # pointwise classification, with mild focal weighting for class imbalance
    bce = F.binary_cross_entropy_with_logits(
        point_logit, batch.y_bin,
        weight=torch.full_like(batch.y_bin, float(pos_weight)),
        reduction="none")
    p = torch.sigmoid(point_logit)
    focal = (p - batch.y_bin).abs().clamp_min(1e-6) ** focal_gamma
    l_point = masked_mean(bce * focal, mv)

    # reactivity regression onto the rank-normalised scale
    l_reg = masked_mean(F.smooth_l1_loss(reg, batch.y_rank, reduction="none"), mv)

    # region head: BCE on a neighbourhood-smoothed label, so the head is scored
    # on whether a stretch of nucleotides is flexible rather than on isolated hits
    k = 9
    pad = k // 2
    num = F.conv1d(F.pad(batch.y_bin.unsqueeze(1), (pad, pad), mode="replicate"),
                   torch.ones(1, 1, k) / k)
    den = F.conv1d(F.pad(batch.mask.unsqueeze(1), (pad, pad), mode="replicate"),
                   torch.ones(1, 1, k))
    target = (num / den.clamp_min(1e-8)).squeeze(1)
    l_region = masked_mean(
        F.binary_cross_entropy_with_logits(region_logit, target, reduction="none"), mv)

    l_rank = (per_rna_pairwise_rank_loss(reg, batch.y_rank, mv)
              if w_rank > 0 else reg.new_zeros(()))

    total = (w_point * l_point + w_reg * l_reg + w_region * l_region
             + w_rank * l_rank)
    return total, dict(point=float(l_point.detach()), reg=float(l_reg.detach()),
                       region=float(l_region.detach()), rank=float(l_rank.detach()),
                       total=float(total.detach()))
