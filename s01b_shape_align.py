"""
Stage 1b: put the SHAPE-MaP reactivity on hairpin coordinates.

GSE254361 probed synthetic 125-nt pri-miRNA minigenes, so the published
reactivity vector is indexed on the *construct*, not on the miRBase hairpin, and
the hairpin interval has to be recovered per RNA. The companion
``SHAPE-based_pri-miRNA_structures`` file gives the experimentally determined
pair table on the same 125-nt construct, which is an independent structural
measurement rather than the reactivity quantity this study predicts.

For every candidate offset o the McCaskill pairing marginal of the hairpin alone,
P(i paired), is compared with the experimental pair table restricted to
[o, o+L) by the log-likelihood

    LL(o) = sum_i [ log P(i paired)   if tab[o+i] is paired
                    log (1 - P(i paired)) otherwise ]

which uses all L positions instead of only the MFE pairs and is therefore far
less noisy. The argmax is accepted when it beats the runner-up by a margin in
nats; ambiguous placements are dropped rather than guessed. Reactivity is never
read during alignment, so the coordinate frame does not depend on the quantity
being predicted; the resulting reactivity/unpaired-agreement correlation is
reported only as an audit statistic.

Outputs
  data/processed/shape_reactivity.json   sequence/reactivity on hairpin coords
  data/processed/shape_alignment_qc.tsv  per-RNA offset and agreement audit
"""
import csv
import gzip
import json
import os

import numpy as np
import RNA
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw")
MIR = os.path.join(ROOT, "data", "mirbase")

CONSTRUCT_LEN = 125
MIN_MARGIN_NATS = 5.0        # argmax must beat the runner-up offset by this much
MIN_HAIRPIN = 25


def read_fasta(path):
    recs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    recs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        recs[name] = "".join(buf)
    return recs


def read_wide(path):
    """GSE wide CSV -> (row name -> list of floats)."""
    out = {}
    with gzip.open(path, "rt", errors="replace") as fh:
        fh.readline()
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            name = parts[0].strip().strip('"')
            vals = []
            for p in parts[1:]:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(float("nan"))
            out[name] = vals
    return out


def pair_marginal(seq, md):
    """P(position i is paired) from the McCaskill ensemble of seq alone.

    ViennaRNA returns a strictly upper-triangular base-pair matrix, so it is
    symmetrised before the marginal is taken; summing rows alone would assign
    every position zero pairing probability whenever its partner lies 5' of it.
    """
    n = len(seq)
    fc = RNA.fold_compound(seq, md)
    fc.mfe()
    fc.pf()
    b = np.array(fc.bpp(), dtype=float)
    n_eff = min(n, b.shape[0])
    b = b[:n_eff, :n_eff]
    m = (b + np.triu(b, 1).T).sum(axis=1)
    if len(m) < n:
        m = np.pad(m, (0, n - len(m)))
    return m[:n]


def main():
    hairpin = read_fasta(os.path.join(MIR, "hairpin.fa"))
    hb = {h.split()[0]: s for h, s in hairpin.items() if h.startswith("hsa-")}

    react = read_wide(os.path.join(RAW, "GSE254361_SHAPE_reactivity.csv.gz"))
    struct = read_wide(os.path.join(RAW,
                                    "GSE254361_SHAPE-based_pri-miRNA_structures.csv.gz"))

    md = RNA.md()
    md.uniq_ML = True

    out, qc = [], []
    drop = {"no_hairpin": 0, "no_struct": 0, "length": 0, "ambiguous": 0,
            "no_reactivity": 0}

    for name in sorted(react):
        if not name.startswith("hsa-"):
            continue
        seq = hb.get(name)
        tab = struct.get(name)
        if seq is None or "N" in seq:
            drop["no_hairpin"] += 1
            continue
        if tab is None:
            drop["no_struct"] += 1
            continue
        L = len(seq)
        if not (MIN_HAIRPIN <= L <= CONSTRUCT_LEN):
            drop["length"] += 1
            continue

        pp = np.clip(pair_marginal(seq, md), 1e-4, 1.0 - 1e-4)
        exp_paired = np.zeros(CONSTRUCT_LEN)
        for i, v in enumerate(tab[:CONSTRUCT_LEN]):
            j = int(v) - 1
            if 0 <= j < CONSTRUCT_LEN:
                exp_paired[i] = 1.0

        scored = []
        for off in range(0, CONSTRUCT_LEN - L + 1):
            e = exp_paired[off:off + L]
            ll = float(np.where(e > 0.5, np.log(pp), np.log(1.0 - pp)).sum())
            scored.append((ll, off))
        scored.sort(reverse=True)
        best_ll, best_off = scored[0]
        second_ll = scored[1][0] if len(scored) > 1 else -np.inf
        margin = best_ll - second_ll
        if margin < MIN_MARGIN_NATS:
            drop["ambiguous"] += 1
            continue

        R = list(react[name])
        R += [float("nan")] * (CONSTRUCT_LEN - len(R))
        rh = R[best_off:best_off + L]
        fin = np.isfinite(rh)
        if fin.sum() < 20:
            drop["no_reactivity"] += 1
            continue
        # Exact zeros are genuine "protected, not modified" 1M7 readouts and are
        # strongly enriched in predicted base-paired positions, so they are kept
        # as real low-reactivity measurements rather than treated as missing.
        # Only genuine non-numeric cells become None.
        react_json = [None if not np.isfinite(v) else float(v) for v in rh]

        out.append({
            "exp_id": name, "mirbase_id": name, "dataset": "SHAPE",
            "sequence": seq, "reactivity": react_json,
            "construct_length": CONSTRUCT_LEN, "offset": int(best_off),
            "hairpin_length": L, "length": L,
        })
        # audit only: reactivity is not used to choose the offset
        rho = float(spearmanr(np.asarray(rh)[fin], 1.0 - pp[fin]).statistic)
        qc.append({"mirbase_id": name, "hairpin_length": L, "offset": best_off,
                   "loglik": round(best_ll, 2), "margin_nats": round(margin, 2),
                   "n_measured": int(fin.sum()),
                   "rho_react_vs_unpaired_qc": round(rho, 4)})

    with open(os.path.join(PROC, "shape_reactivity.json"), "w") as fh:
        json.dump(out, fh)
    with open(os.path.join(PROC, "shape_alignment_qc.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qc[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(qc)

    off = np.array([r["offset"] for r in qc])
    rho = np.array([r["rho_react_vs_unpaired_qc"] for r in qc])
    mg = np.array([r["margin_nats"] for r in qc])
    print(f"[shape-align] retained {len(out)} hsa RNAs; dropped {drop}")
    print(f"[shape-align] hairpin offset in the 125-nt construct: "
          f"median={int(np.median(off))} IQR=[{np.percentile(off, 25):.0f}, "
          f"{np.percentile(off, 75):.0f}] range=[{off.min()}, {off.max()}]")
    print(f"[shape-align] argmax margin over runner-up: median={np.median(mg):.1f} nats")
    print(f"[shape-align] QC rho(reactivity, predicted unpaired): "
          f"median={np.median(rho):.3f} mean={rho.mean():.3f} "
          f"frac>0.3={(rho > 0.3).mean():.3f}")


if __name__ == "__main__":
    main()
