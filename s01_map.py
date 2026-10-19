"""
Stage 1-4: Data acquisition, parsing and sequence->experiment mapping.

Inputs
  data/mirbase/hairpin.fa                    miRBase hairpin (all species)
  data/raw/GSE254361_SHAPE_reactivity.csv.gz  SHAPE-MaP nucleotide reactivity
  data/raw/GSE262014_pri-miRNA.json.gz       DMS-MaPseq nucleotide reactivity

Outputs
  data/processed/human_hairpins.fa
  data/processed/human_hairpins.tsv
  data/processed/sequence_mapping.tsv
  data/processed/shape_reactivity.tsv
  data/processed/dms_reactivity.tsv
"""
import gzip
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
MIRBASE = os.path.join(ROOT, "data", "mirbase")
os.makedirs(PROC, exist_ok=True)

MIN_IDENTITY = 0.90
MIN_COVERAGE = 0.90


# ---------------------------------------------------------------- miRBase ---
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


recs = read_fasta(os.path.join(MIRBASE, "hairpin.fa"))
# header layout: ><mir_id> <accession> <species genus> <species epithet> <name> ...
meta = {}
for h, s in recs.items():
    parts = h.split()
    meta[h] = {"id": parts[0], "acc": parts[1], "name": parts[-2],
               "desc": " ".join(parts[2:])}

human = {h: s for h, s in recs.items() if meta[h]["id"].startswith("hsa-") and "N" not in s}
print(f"[mirbase] total={len(recs)}  human={len(human)}")

with open(os.path.join(PROC, "human_hairpins.fa"), "w") as out:
    for h, s in sorted(human.items(), key=lambda x: meta[x[0]]["id"]):
        out.write(f">{h}\n{s}\n")

hb = pd.DataFrame(
    [
        {
            "mirbase_id": meta[h]["id"],
            "accession": meta[h]["acc"],
            "mir_name": meta[h]["name"],
            "length": len(s),
            "sequence": s,
        }
        for h, s in human.items()
    ]
).sort_values("mirbase_id")
hb.to_csv(os.path.join(PROC, "human_hairpins.tsv"), sep="\t", index=False)

# mature miRNA arm annotation from the free energy / naming is unavailable in
# hairpin.fa, so arms are annotated from mature.fa when present.
mature_path = os.path.join(MIRBASE, "mature.fa")
if os.path.exists(mature_path):
    mrecs = read_fasta(mature_path)
    arms = defaultdict(list)
    for h, s in mrecs.items():
        if h.startswith("hsa-"):
            arms[meta.get(h, {}).get("name", h.split()[0])].append(s.replace("T", "U").upper())
    hb["n_mature_arms"] = hb["mir_name"].map(lambda n: len(arms.get(n, [])))
else:
    hb["n_mature_arms"] = 0
print(f"[mirbase] human hairpins written; arms annotated for "
      f"{(hb['n_mature_arms'] > 0).sum()} hairpins")


# ------------------------------------------------------------ SHAPE-MaP ---
def read_shape(path):
    with gzip.open(path, "rt", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split(",")
        rows = []
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            rows.append(parts)
    return header[1:], rows


hdr, rows = read_shape(os.path.join(RAW, "GSE254361_SHAPE_reactivity.csv.gz"))
out = []
for parts in rows:
    name = parts[0].strip().strip('"')
    vals = [p for p in parts[1:] if p not in ("", "nan")]
    react = []
    for p in vals:
        try:
            v = float(p)
        except ValueError:
            react.append(np.nan)
            continue
        react.append(v)
    out.append({"exp_id": name, "reactivity": react})

shape = pd.DataFrame(out)
shape["n_pos"] = shape["reactivity"].map(len)
shape_hsa = shape[shape["exp_id"].str.startswith("hsa-")].copy()
print(f"[shape] rows={len(shape)}  hsa rows={len(shape_hsa)}  "
      f"median len={shape_hsa['n_pos'].median():.0f}")

# structure (pair table) from the companion file
_, srows = read_shape(os.path.join(RAW, "GSE254361_SHAPE-based_pri-miRNA_structures.csv.gz"))
shape_struct = {}
for parts in srows:
    name = parts[0].strip().strip('"')
    tab = []
    for p in parts[1:]:
        try:
            tab.append(int(p))
        except ValueError:
            tab.append(0)
    shape_struct[name] = tab

# ------------------------------------------------------------- DMS-MaP ---
with gzip.open(os.path.join(RAW, "GSE262014_pri-miRNA.json.gz"), "rt") as fh:
    dms_raw = json.load(fh)

DMS_SENTINEL = -1000.0
dms_rows = []
for name, rec in dms_raw.items():
    seq = rec["sequence"].replace("T", "U").upper()
    d = rec["dms"]
    react = [np.nan if (v is None or v <= DMS_SENTINEL) else float(v) for v in d]
    if len(react) != len(seq):
        n = min(len(react), len(seq))
        seq, react = seq[:n], react[:n]
    dms_rows.append({"exp_id": name, "sequence": seq, "reactivity": react,
                     "pairs": rec.get("structure", [])})

dms = pd.DataFrame(dms_rows)
dms["n_pos"] = dms["reactivity"].map(len)
print(f"[dms] records={len(dms)}  all_len_230={bool((dms['n_pos'] == 230).all())}  "
      f"valid frac={np.mean([np.mean([~np.isnan(r)]) for r in dms['reactivity']]):.3f}")


# ------------------------------------------------------------- mapping ---
def best_match(exp_id, exp_seq, human_by_id):
    """Exact-identifier match, then full-sequence identity, then local alignment."""
    rec = {"exp_id": exp_id, "map_method": "unmapped", "identity": np.nan,
           "coverage": 0.0, "mirbase_id": "", "n_mismatch": np.nan}

    if exp_id in human_by_id:
        h = human_by_id[exp_id]
        rec.update(mirbase_id=exp_id, map_method="identifier",
                   identity=1.0, coverage=1.0)
        return rec

    if exp_seq is None:
        return rec

    e = exp_seq.replace("T", "U").upper()
    # exact full-sequence identity against any human hairpin
    for h, s in human.items():
        if len(s) == len(e) and s == e:
            rec.update(mirbase_id=meta[h]["id"], map_method="exact_sequence",
                       identity=1.0, coverage=1.0)
            return rec

    # anchored local alignment: find the longest common substring anchor then
    # score by global identity on the overlapping window.
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    best = None
    targets = [(meta[h]["id"], s) for h, s in human.items()
               if abs(len(s) - len(e)) <= 40]
    for mid, s in targets:
        try:
            aln = aligner.align(e, s)[0]
        except Exception:
            continue
        if aln.score < 0.30 * min(len(e), len(s)):
            continue
        a_idx, b_idx = aln.aligned
        total = 0
        matches = 0
        for (a0, a1), (b0, b1) in zip(a_idx, b_idx):
            n = a1 - a0
            total += n
            matches += sum(1 for x, y in zip(e[a0:a1], s[b0:b1]) if x == y)
        if total == 0:
            continue
        ident = matches / total
        cov = total / min(len(e), len(s))
        if best is None or ident * cov > best[0]:
            best = (ident * cov, mid, ident, cov)
    if best is not None and best[2] >= MIN_IDENTITY and best[3] >= MIN_COVERAGE:
        rec.update(mirbase_id=best[1], map_method="local_alignment",
                   identity=round(best[2], 4), coverage=round(best[3], 4))
    return rec


human_by_id = {meta[h]["id"]: h for h in human}
mir2id = {}
for h in human:
    mir2id.setdefault(meta[h]["name"], []).append(meta[h]["id"])

# --- SHAPE mapping (no explicit sequence in the reactivity file: the row name
#     is the miRBase identifier for hsa rows) ---
map_rows = []
for _, r in shape.iterrows():
    eid = r["exp_id"]
    if eid in human_by_id:
        map_rows.append({"exp_id": eid, "dataset": "SHAPE", "mirbase_id": eid,
                         "map_method": "identifier", "identity": 1.0,
                         "coverage": 1.0, "n_pos": r["n_pos"]})
    else:
        cand = mir2id.get(eid, [])
        if cand:
            map_rows.append({"exp_id": eid, "dataset": "SHAPE", "mirbase_id": cand[0],
                             "map_method": "identifier_name", "identity": 1.0,
                             "coverage": 1.0, "n_pos": r["n_pos"]})
        else:
            map_rows.append({"exp_id": eid, "dataset": "SHAPE", "mirbase_id": "",
                             "map_method": "unmapped", "identity": np.nan,
                             "coverage": 0.0, "n_pos": r["n_pos"]})

# --- DMS mapping (explicit sequences available) ---
seq_by_id = {r["exp_id"]: r["sequence"] for _, r in dms.iterrows()}
dms_ids = [e for e in dms["exp_id"].tolist() if e not in human_by_id]
resolved = {e: best_match(e, seq_by_id.get(e), human_by_id) for e in dms_ids}
for _, r in dms.iterrows():
    eid = r["exp_id"]
    if eid in human_by_id:
        map_rows.append({"exp_id": eid, "dataset": "DMS", "mirbase_id": eid,
                         "map_method": "identifier", "identity": 1.0,
                         "coverage": 1.0, "n_pos": r["n_pos"]})
    else:
        res = resolved.get(eid)
        if res and res["mirbase_id"]:
            map_rows.append({"exp_id": eid, "dataset": "DMS", "mirbase_id": res["mirbase_id"],
                             "map_method": res["map_method"], "identity": res["identity"],
                             "coverage": res["coverage"], "n_pos": r["n_pos"]})
        else:
            map_rows.append({"exp_id": eid, "dataset": "DMS", "mirbase_id": "",
                             "map_method": "unmapped", "identity": np.nan,
                             "coverage": 0.0, "n_pos": r["n_pos"]})

mapping = pd.DataFrame(map_rows)
mapping.to_csv(os.path.join(PROC, "sequence_mapping.tsv"), sep="\t", index=False)
print("[mapping] " + str(mapping.groupby(["dataset", "map_method"]).size()))


# ------------------------------------------------- aligned reactivity tables ---
hb_seq = dict(zip(hb["mirbase_id"], hb["sequence"]))

shape_ok = mapping[(mapping.dataset == "SHAPE") & (mapping.mirbase_id != "")]
rows = []
for _, m in shape_ok.iterrows():
    react = shape.loc[shape["exp_id"] == m["exp_id"], "reactivity"].iloc[0]
    seq = hb_seq.get(m["mirbase_id"])
    n = min(len(react), len(seq)) if seq else len(react)
    struct = shape_struct.get(m["exp_id"], [])[:n]
    rows.append({
        "exp_id": m["exp_id"], "mirbase_id": m["mirbase_id"],
        "dataset": "SHAPE", "sequence": seq[:n] if seq else None,
        "reactivity": react[:n], "pair_table": struct,
        "length": n,
    })
shape_tab = pd.DataFrame(rows)
shape_tab.to_json(os.path.join(PROC, "shape_reactivity.json"), orient="records")

dms_ok = mapping[(mapping.dataset == "DMS") & (mapping.mirbase_id != "")]
rows = []
for _, m in dms_ok.iterrows():
    r = dms.loc[dms["exp_id"] == m["exp_id"]].iloc[0]
    rows.append({
        "exp_id": m["exp_id"], "mirbase_id": m["mirbase_id"], "dataset": "DMS",
        "sequence": r["sequence"], "reactivity": list(r["reactivity"]),
        "pairs": r["pairs"], "length": r["n_pos"],
    })
dms_tab = pd.DataFrame(rows)
dms_tab.to_json(os.path.join(PROC, "dms_reactivity.json"), orient="records")

print(f"[out] shape records={len(shape_tab)}  dms records={len(dms_tab)}")
print(f"[out] unique mapped miRNAs: shape={shape_tab['mirbase_id'].nunique()} "
      f"dms={dms_tab['mirbase_id'].nunique()} "
      f"union={len(set(shape_tab['mirbase_id']) | set(dms_tab['mirbase_id']))}")
print(f"[out] both chemistries: {len(set(shape_tab['mirbase_id']) & set(dms_tab['mirbase_id']))}")
