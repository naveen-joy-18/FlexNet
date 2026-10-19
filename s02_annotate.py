"""
Stage 2a: per-hairpin annotation tracks (phylogenetic conservation, mature-arm
labels) and localisation of each miRBase hairpin inside its 230-nt DMS-MaPseq
window.

Conservation is computed from exact miRBase hairpin-ID orthologues spanning a
broad vertebrate panel, so the score integrates phylogenetic depth. For each
human hairpin every orthologue is globally aligned to it and a per-position
column entropy is accumulated; the conservation score is the normalised
information content  1 - H(col)/log(4), which is a phastCons-style measure.

Outputs
  data/processed/conservation.tsv     per-position orthologue conservation
  data/processed/arm_annotations.tsv  per-position mature-arm label
  data/processed/dms_window_loc.tsv   hairpin interval inside each DMS window
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
MIRBASE = os.path.join(ROOT, "data", "mirbase")
BASES = "ACGU"
B2I = {c: i for i, c in enumerate(BASES)}

# Vertebrate reference panel, ordered by increasing distance from human.
# Two non-human primates, two rodents, and five more distant vertebrates.
ORTHO_SPECIES = ["ptr", "mml", "rno", "cfa", "gga", "xla", "dre", "lfe"]


def read_fasta(path):
    recs, name, seq = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs[name] = "".join(seq)
                name, seq = line[1:], []
            else:
                seq.append(line)
    if name is not None:
        recs[name] = "".join(seq)
    return recs


def split_header(h):
    p = h.split()
    return {"id": p[0], "acc": p[1], "name": p[-2], "desc": " ".join(p[2:])}


def ortholog_key(mirbase_id):
    """hsa-let-7a-1 -> let-7a-1 (the part used for exact orthology matching)."""
    return mirbase_id.split("-", 1)[1].lower()


# ------------------------------------------------- orthologue conservation ---
print("[cons] indexing vertebrate orthologues", flush=True)
all_recs = read_fasta(os.path.join(MIRBASE, "hairpin.fa"))
meta = {h: split_header(h) for h in all_recs}

human = {h: s for h, s in all_recs.items()
         if meta[h]["id"].startswith("hsa-") and "N" not in s and 20 <= len(s) <= 200}
hairpin_len = {meta[h]["id"]: len(s) for h, s in human.items()}

by_key = defaultdict(list)
for h, s in all_recs.items():
    mid = meta[h]["id"]
    sp = mid.split("-")[0]
    if sp in ORTHO_SPECIES and "N" not in s and 20 <= len(s) <= 200:
        by_key[ortholog_key(mid)].append(s)
print(f"[cons] human hairpins={len(human)}  "
      f"reference orthologue groups={len(by_key)}", flush=True)

aligner = Align.PairwiseAligner()
aligner.mode = "global"
# NUC.4.4 is a nucleotide matrix, but its Biopython alphabet is DNA-only, so
# RNA is aligned in T-space and mapped back to U coordinates. BLOSUM62 cannot
# be used at all: it rejects U outright.
_nt = substitution_matrices.load("NUC.4.4")
aligner.substitution_matrix = _nt
aligner.open_gap_score = -6
aligner.extend_gap_score = -1
aligner.end_insertion_score = 0.0
aligner.end_deletion_score = 0.0


def aligned_column_indices(query, target):
    """Yield (query_index, target_index) for columns where both hold a base."""
    aln = aligner.align(query.replace("U", "T"), target.replace("U", "T"))[0]
    for (q0, q1), (t0, t1) in zip(*aln.aligned):
        for q, t in zip(range(q0, q1), range(t0, t1)):
            yield q, t


rows = []
n_orth_total = 0
for i, (h, hs) in enumerate(human.items()):
    mid = meta[h]["id"]
    orth = by_key.get(ortholog_key(mid), [])
    n = len(hs)
    if not orth:
        rows.append({"mirbase_id": mid, "n_orth": 0, "cons_score": "",
                     "cons_mean": np.nan, "cons_std": np.nan,
                     "cons_detectable": 0})
        continue
    counts = np.zeros((n, 4), dtype=np.float32)
    for os_ in orth:
        try:
            pairs = list(aligned_column_indices(os_, hs))
        except Exception:
            continue
        if not pairs:
            continue
        qi = np.fromiter((q for q, _ in pairs), dtype=np.int64, count=len(pairs))
        ti = np.fromiter((t for _, t in pairs), dtype=np.int64, count=len(pairs))
        qb = np.frombuffer(os_.encode(), dtype=np.uint8)
        hb = np.frombuffer(hs.encode(), dtype=np.uint8)
        lut = np.full(256, -1, dtype=np.int64)
        for c, k in B2I.items():
            lut[ord(c)] = k
        ti_v, qb_v = ti, qb[qi]
        keep = lut[hb[ti_v]] >= 0
        np.add.at(counts, (ti_v[keep], lut[qb_v[keep]]), 1.0)
    col = counts.sum(axis=1)
    seen = col > 0
    if not seen.any():
        rows.append({"mirbase_id": mid, "n_orth": len(orth), "cons_score": "",
                     "cons_mean": np.nan, "cons_std": np.nan, "cons_detectable": 0})
        continue
    frac = np.divide(counts, col[:, None], out=np.zeros_like(counts), where=seen[:, None])
    ent = -(frac * np.log2(np.clip(frac, 1e-12, None))).sum(axis=1)
    cons = np.clip(1.0 - ent / 2.0, 0.0, 1.0)
    cons[~seen] = np.nan
    enc = np.nan_to_num(cons, nan=0.0)
    rows.append({
        "mirbase_id": mid, "n_orth": len(orth),
        "cons_score": "".join(
            "-" if not seen[k] else str(int(round(enc[k] * 9))) for k in range(n)),
        "cons_mean": float(np.nanmean(cons)),
        "cons_std": float(np.nanstd(cons)),
        "cons_detectable": int(seen.sum()),
    })
    n_orth_total += len(orth)
    if (i + 1) % 400 == 0:
        print(f"  ...{i + 1}/{len(human)}", flush=True)

cons = pd.DataFrame(rows)
cons.to_csv(os.path.join(PROC, "conservation.tsv"), sep="\t", index=False)
det = cons[cons.n_orth > 0].copy()
det["len"] = det["mirbase_id"].map(hairpin_len)
det["frac"] = det["cons_detectable"] / det["len"]
print(f"[cons] mean orthologues/hairpin={n_orth_total / max(len(human), 1):.2f} | "
      f"hairpins with >=1 orthologue={len(det)}/{len(human)} | "
      f"mean aligned-column fraction={det['frac'].mean():.3f}")

# ------------------------------------------------------- mature arm tracks ---
print("[arms] locating mature miRNAs inside hairpins", flush=True)
mrecs = read_fasta(os.path.join(MIRBASE, "mature.fa"))
mature_by_name = defaultdict(list)
for h, s in mrecs.items():
    if not h.startswith("hsa-"):
        continue
    p = h.split()
    tag = p[-1]                       # e.g. "miR-21-5p"
    arm = tag[-2:] if tag[-2:] in ("5p", "3p") else "."
    name = (tag[:-3] if arm != "." else tag).lower()
    mature_by_name[name].append((arm, s.replace("T", "U").upper()))
print(f"[arms] mature records for {len(mature_by_name)} names", flush=True)


def best_arm_offset(hs, ms, max_mismatch=2):
    """Hamming-scan every window of the hairpin for the mature sequence.
    miRBase arm calls occasionally differ from the hairpin by a 3'-terminal
    base, so a small number of mismatches is tolerated."""
    m = len(ms)
    if m == 0 or m > len(hs):
        return None
    hb = np.frombuffer(hs.encode(), dtype=np.uint8)
    mb = np.frombuffer(ms.encode(), dtype=np.uint8)
    best, best_mis = None, m + 1
    for i in range(len(hs) - m + 1):
        mis = int((hb[i:i + m] != mb).sum())
        if mis < best_mis:
            best, best_mis = i, mis
    if best is None or best_mis > max_mismatch:
        return None
    return best


arm_rows = []
for h, hs in human.items():
    labels = np.array(["."] * len(hs), dtype="<U2")
    n_loc = 0
    for arm, ms in mature_by_name.get(meta[h]["id"].split("-", 1)[1].lower(), []):
        idx = best_arm_offset(hs, ms)
        if idx is not None:
            labels[idx:idx + len(ms)] = arm
            n_loc += 1
    arm_rows.append({"mirbase_id": meta[h]["id"], "arms": "".join(labels),
                     "n_arms_located": n_loc,
                     "n_5p": int((labels == "5p").sum()),
                     "n_3p": int((labels == "3p").sum())})
arms = pd.DataFrame(arm_rows)
arms.to_csv(os.path.join(PROC, "arm_annotations.tsv"), sep="\t", index=False)
print(f"[arms] hairpins with a located mature arm: "
      f"{(arms['n_5p'] + arms['n_3p'] > 0).sum()}/{len(human)}")

# ------------------------------------- hairpin position inside DMS window ---
print("[loc] localising hairpins in DMS windows", flush=True)
with open(os.path.join(PROC, "dms_reactivity.json")) as fh:
    dms_tab = json.load(fh)
hairpin_seq = {meta[h]["id"]: human[h] for h in human}

loc = []
for rec in dms_tab:
    win = rec["sequence"]
    hs = hairpin_seq.get(rec["mirbase_id"])
    if hs is None or hs not in win:
        loc.append({"exp_id": rec["exp_id"], "mirbase_id": rec["mirbase_id"],
                    "start": -1, "end": -1, "identity": np.nan})
        continue
    s = win.find(hs)
    loc.append({"exp_id": rec["exp_id"], "mirbase_id": rec["mirbase_id"],
                "start": int(s), "end": int(s + len(hs)), "identity": 1.0})
locdf = pd.DataFrame(loc)
locdf.to_csv(os.path.join(PROC, "dms_window_loc.tsv"), sep="\t", index=False)
ok = locdf["start"] >= 0
print(f"[loc] hairpin located in {ok.sum()}/{len(locdf)} DMS windows")
