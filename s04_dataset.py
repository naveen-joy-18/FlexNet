"""
Stage 4: dataset assembly, family-aware splitting and label definition.

Central design decision: SHAPE and DMS are *labels*, never predictors. The
classifier is trained on one chemistry and evaluated on the other, so no model
ever sees the reactivity it is scored against.

Labels
  binary   a nucleotide is "flexible" if its normalised reactivity exceeds a
           fixed, chemistry-specific cut-off. Thresholds are fixed a priori:
           DMS  >= 0.50 (normalised 0-1 DMS), SHAPE >= 1.00 (the canonical
           Weeks/Vandreyden protected/unprotected boundary for normalised
           SHAPE reactivity).
  continuous normalised reactivity, rank-transformed within each dataset, is
           kept for Spearman/Pearson correlation analyses.

Splits
  Group = miRNA family stem, so hsa-let-7a-1/-2/-3 and every paralogue stay in
  one partition and no near-identical hairpin can straddle the split.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "data", "features")
PROC = os.path.join(ROOT, "data", "processed")

DMS_THRESHOLD = 0.50
SHAPE_THRESHOLD = 1.00
SEED = 20240617
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15      # remainder is the independent test set


def family_of(mirbase_id):
    """hsa-let-7a-2 -> let-7a ; hsa-mir-16-2 -> mir-16 ; hsa-mir-10226 -> mir-10226.

    miRBase copy suffixes (let-7a-1/-2/-3, mir-16-1/-2) mark near-identical
    paralogous loci and must stay in the same partition, so a trailing single
    digit is removed. Gene numbers are multi-digit (mir-21, mir-10226), so only
    a one-digit suffix is stripped; a blanket trailing ``-\\d+`` strip would
    collapse all 1061 hairpins onto 24 groups.
    """
    stem = mirbase_id.split("-", 1)[1]
    return re.sub(r"-\d$", "", stem)


def load(tag):
    d = np.load(os.path.join(FEAT, f"{tag}_features.npz"), allow_pickle=True)
    ids = [json.loads(s) for s in d["ids"]]
    struct = [json.loads(s) for s in d["struct"]]
    groups = json.loads(str(d["groups"]))
    return dict(X=d["X"], y=d["y"], ptr=d["ptr"], lengths=d["lengths"],
                ids=ids, struct=struct, names=[str(s) for s in d["feature_names"]],
                groups=groups)


def per_rna(dataset):
    """Split the flat arrays into per-RNA lists."""
    ptr = dataset["ptr"]
    out = []
    for i, rec in enumerate(dataset["ids"]):
        a, b = int(ptr[i]), int(ptr[i + 1])
        out.append({
            "exp_id": rec["exp_id"],
            "mirbase_id": rec["mirbase_id"],
            "family": family_of(rec["mirbase_id"]),
            "length": int(b - a),
            "X": dataset["X"][a:b],
            "y": dataset["y"][a:b],
            "struct": dataset["struct"][i],
        })
    return out


def labelise(rnas, threshold):
    for r in rnas:
        y = r["y"]
        valid = np.isfinite(y)
        lab = np.full(len(y), np.nan)
        lab[valid] = (y[valid] >= threshold).astype(float)
        r["y_bin"] = lab
        r["valid"] = valid
    return rnas


def rank_transform(rnas, mask_name="valid"):
    """Within-dataset rank-normalised reactivity in [0, 1] for correlation work.

    ``order`` is the permutation that sorts the pooled measured reactivities, so
    ``order[k]`` is the flat position of the k-th smallest value and the value at
    that position must receive percentile ``(k + 0.5) / n``. Applying argsort
    twice would make ``order`` a second permutation rather than the ranks, which
    shuffles the target and leaves it uncorrelated with the binary label.
    """
    allv = np.concatenate([r["y"][r[mask_name]] for r in rnas])
    order = np.argsort(allv)
    pct = (np.arange(len(allv)) + 0.5) / len(allv)
    ranks = np.empty(len(allv))
    ranks[order] = pct
    off = 0
    for r in rnas:
        n = int(r[mask_name].sum())
        r["y_rank"] = np.full(len(r["y"]), np.nan)
        r["y_rank"][r[mask_name]] = ranks[off:off + n]
        off += n
    return rnas


def group_split(rnas, seed=SEED):
    """Assign whole miRNA families to train / val / test."""
    fams = sorted({r["family"] for r in rnas})
    rng = np.random.default_rng(seed)
    rng.shuffle(fams)
    n = len(fams)
    n_tr = int(round(TRAIN_FRAC * n))
    n_va = int(round(VAL_FRAC * n))
    assign = {}
    for f in fams[:n_tr]:
        assign[f] = "train"
    for f in fams[n_tr:n_tr + n_va]:
        assign[f] = "val"
    for f in fams[n_tr + n_va:]:
        assign[f] = "test"
    for r in rnas:
        r["split"] = assign[r["family"]]
    return rnas, assign


def main():
    os.makedirs(PROC, exist_ok=True)
    summary = {}

    dms = per_rna(load("dms"))
    dms = labelise(dms, DMS_THRESHOLD)
    dms = rank_transform(dms)
    shape = per_rna(load("shape"))
    shape = labelise(shape, SHAPE_THRESHOLD)
    shape = rank_transform(shape)

    # Group-level split is defined once on the union of families so that the
    # SHAPE test set is genuinely unseen, and so DMS and SHAPE never share a
    # family across the train/test boundary.
    all_rnas = dms + shape
    all_rnas, assign = group_split(all_rnas)

    def stats(rnas, tag):
        n_pos = sum(int(np.nansum(r["y_bin"])) for r in rnas)
        n_valid = sum(int(r["valid"].sum()) for r in rnas)
        c = defaultdict(int)
        for r in rnas:
            c[r["split"]] += 1
        return dict(dataset=tag, n_rnas=len(rnas), n_families=len({r["family"] for r in rnas}),
                    n_positions=int(sum(r["length"] for r in rnas)),
                    n_valid=n_valid, n_positive=n_pos,
                    pos_rate=round(n_pos / max(n_valid, 1), 4),
                    n_train=c["train"], n_val=c["val"], n_test=c["test"],
                    median_length=float(np.median([r["length"] for r in rnas])))

    summary["dms"] = stats(dms, "DMS-MaPseq")
    summary["shape"] = stats(shape, "SHAPE-MaP")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(PROC, "family_splits.json"), "w") as fh:
        json.dump(assign, fh, indent=1)

    with open(os.path.join(PROC, "dataset_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    for tag, rnas in (("dms", dms), ("shape", shape)):
        np.savez_compressed(
            os.path.join(FEAT, f"{tag}_modelling.npz"),
            n_rnas=len(rnas),
            lengths=np.array([r["length"] for r in rnas]),
            X=np.concatenate([r["X"] for r in rnas], axis=0),
            y=np.concatenate([r["y"] for r in rnas], axis=0),
            y_bin=np.concatenate([r["y_bin"] for r in rnas], axis=0),
            y_rank=np.concatenate([r["y_rank"] for r in rnas], axis=0),
            valid=np.concatenate([r["valid"] for r in rnas], axis=0),
            ptr=np.concatenate([[0], np.cumsum([r["length"] for r in rnas])]),
            mirbase_id=np.array([r["mirbase_id"] for r in rnas]),
            exp_id=np.array([r["exp_id"] for r in rnas]),
            family=np.array([r["family"] for r in rnas]),
            split=np.array([r["split"] for r in rnas]),
            feature_names=load("dms")["names"],
            groups=np.array(json.dumps(load("dms")["groups"])),
            struct=np.array([json.dumps(r["struct"]) for r in rnas], dtype=object),
        )
        print(f"[out] {tag} modelling arrays written")

    # family overlap between the two chemistries, to document the independence
    fd = defaultdict(set)
    for r in dms:
        fd[r["split"]].add(r["family"])
    fs = defaultdict(set)
    for r in shape:
        fs[r["split"]].add(r["family"])
    for s in ("train", "val", "test"):
        inter = len(fd[s] & fs[s])
        print(f"[overlap] split={s} families shared between chemistries: {inter}")
    with open(os.path.join(PROC, "split_overlap.json"), "w") as fh:
        json.dump({s: len(fd[s] & fs[s]) for s in ("train", "val", "test")}, fh, indent=2)


if __name__ == "__main__":
    main()
