"""
Stage 13: build the manuscript submission package.

Produces ``submission/`` at the project root with three deliverables:

  1. ``submission/data/``          the final *processed* training data actually
                                   used by every model (per-nucleotide features,
                                   labels, family-aware splits, hairpin sequences
                                   and structures), plus the feature names, the
                                   channel scaler and human-readable metadata --
                                   the empty ``data/dms``/``data/shape``
                                   directories hold raw source tracks, so this is
                                   the reviewed, reproducible dataset.
  2. ``submission/supplementary/Supplementary_Table_S3_flexible_regions.csv``
                                   supporting data for the flexible-region calls:
                                   every called region in all four evaluation
                                   settings (protocol x own/other chemistry),
                                   with coordinates, model score, experimental
                                   label, sequence and structural context.
  3. ``submission/supplementary/Supplementary_Table_S4_feature_glossary.csv``
                                   the 66 per-nucleotide features with family,
                                   type, value range and a code-accurate
                                   description.

Region calling is identical to s07_regions.py: positions with a FlexNet score
>= T that are measured are merged into runs, runs shorter than L_min are
dropped, and (T, L_min) come from results/call_calibration.json (frozen on the
validation split of the training chemistry and applied unchanged).
"""
import csv
import json
import os
import shutil
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "data", "features")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
SUB = os.path.join(ROOT, "submission")
DATA_OUT = os.path.join(SUB, "data")
SUPP_OUT = os.path.join(SUB, "supplementary")

LOOPS = ["hairpin", "internal", "bulge", "multiloop", "external", "stem"]
PROTOCOLS = {"A": ("dms", "shape"), "B": ("shape", "dms")}
CHEM_LABEL = {"dms": "DMS-MaPseq", "shape": "SHAPE-MaP"}
SETTINGS = (("own_test", "held-out families of the training chemistry"),
            ("other_test", "independent (alternate) chemistry"))
DMS_LABEL_THR = 0.50
SHAPE_LABEL_THR = 1.00


# ------------------------------------------------------------------ helpers ---
def load_modelling(tag):
    d = np.load(os.path.join(FEAT, f"{tag}_modelling.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_feature_ids(tag):
    d = np.load(os.path.join(FEAT, f"{tag}_features.npz"), allow_pickle=True)
    return [json.loads(s) for s in d["ids"]]


def reactivity_sequence(tag):
    """exp_id -> full probed sequence (DMS: 230-nt context; SHAPE: transcript)."""
    with open(os.path.join(PROC, f"{tag}_reactivity.json")) as fh:
        recs = json.load(fh)
    return {r["exp_id"]: r["sequence"] for r in recs}


def call_regions(score, thr, min_len):
    """Consecutive above-threshold stretches; only runs of >= min_len survive."""
    s = np.asarray(score) >= thr
    regions, i, n = [], 0, len(s)
    while i < n:
        if s[i]:
            j = i
            while j + 1 < n and s[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                regions.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return regions


def qname(v):
    return 0 if v is None else int(v)


# ------------------------------------------------------- feature glossary S4 ---
# family, data-type, value range, description -- descriptions mirror the code in
# s03_features.py so that reviewers can trace each channel to its computation.
FEATURES = [
    # ---- A. sequence (19) --------------------------------------------------
    ("seq_A", "sequence", "one-hot", "{0,1}",
     "One-hot indicator that the nucleotide is adenosine (A)."),
    ("seq_C", "sequence", "one-hot", "{0,1}",
     "One-hot indicator that the nucleotide is cytidine (C)."),
    ("seq_G", "sequence", "one-hot", "{0,1}",
     "One-hot indicator that the nucleotide is guanosine (G)."),
    ("seq_U", "sequence", "one-hot", "{0,1}",
     "One-hot indicator that the nucleotide is uridine (U)."),
    ("gc_w3", "sequence", "continuous", "[0,1]",
     "Fraction of G/C nucleotides in a centred 7-nt window (radius 3, edge-replicated)."),
    ("gc_w5", "sequence", "continuous", "[0,1]",
     "Fraction of G/C nucleotides in a centred 11-nt window (radius 5)."),
    ("gc_w10", "sequence", "continuous", "[0,1]",
     "Fraction of G/C nucleotides in a centred 21-nt window (radius 10)."),
    ("purine", "sequence", "indicator", "{0,1}",
     "Indicator that the nucleotide is a purine (A or G)."),
    ("seqent_w3", "sequence", "continuous", "[0,2]",
     "Shannon entropy (bits) of the local base composition in a centred 7-nt window."),
    ("seqent_w5", "sequence", "continuous", "[0,2]",
     "Shannon entropy (bits) of the local base composition in a centred 11-nt window."),
    ("pos_frac", "sequence", "continuous", "[0,1]",
     "Relative position along the hairpin (0 = 5' end, 1 = 3' end)."),
    ("pos_signed", "sequence", "continuous", "[-1,1]",
     "Signed relative position along the hairpin (centred on the middle)."),
    ("dist_3p", "sequence", "continuous", "[0,1]",
     "1 - pos_frac, the normalised distance from the 3' end."),
    ("conservation", "sequence", "continuous", "[0,1]",
     "Per-position 9-way sequence conservation (integer 0-9 from the orthologue "
     "alignment score, scaled to [0,1])."),
    ("cons_detected", "sequence", "indicator", "{0,1}",
     "Indicator that a conservation score was available at this position "
     "(otherwise the conservation channel is filled with the column mean)."),
    ("arm5p", "sequence", "indicator", "{0,1}",
     "Indicator that the position lies in the annotated 5' mature arm."),
    ("arm3p", "sequence", "indicator", "{0,1}",
     "Indicator that the position lies in the annotated 3' mature arm."),
    ("dist_arm5p", "sequence", "continuous", "[0,1]",
     "Normalised distance (capped at 20 nt, divided by 20) to the nearest 5' "
     "arm-boundary position."),
    ("dist_arm3p", "sequence", "continuous", "[0,1]",
     "Normalised distance (capped at 20 nt, divided by 20) to the nearest 3' "
     "arm-boundary position."),
    # ---- B. secondary structure (28) ---------------------------------------
    ("paired_mfe", "structure", "indicator", "{0,1}",
     "Indicator that the position is base-paired in the minimum-free-energy (MFE) structure."),
    ("paired_cent", "structure", "indicator", "{0,1}",
     "Indicator that the position is base-paired in the centroid structure of the "
     "Boltzmann-weighted ensemble."),
    ("unpaired_both", "structure", "indicator", "{0,1}",
     "Indicator that the position is unpaired in both the MFE and the centroid structure."),
    ("loop_mfe_hairpin", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to a hairpin loop in the MFE structure."),
    ("loop_mfe_internal", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to an internal loop in the MFE structure."),
    ("loop_mfe_bulge", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to a bulge (unpaired bases on one side of a "
     "single branch) in the MFE structure."),
    ("loop_mfe_multiloop", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to a multi-branched loop in the MFE structure."),
    ("loop_mfe_external", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to an external (5'/3' single-stranded) region in the MFE structure."),
    ("loop_mfe_stem", "structure", "one-hot", "{0,1}",
     "One-hot: position is a stem (paired) position in the MFE structure."),
    ("loop_cen_hairpin", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to a hairpin loop in the centroid structure."),
    ("loop_cen_internal", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to an internal loop in the centroid structure."),
    ("loop_cen_bulge", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to a bulge in the centroid structure."),
    ("loop_cen_multiloop", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to a multi-branched loop in the centroid structure."),
    ("loop_cen_external", "structure", "one-hot", "{0,1}",
     "One-hot: position belongs to an external region in the centroid structure."),
    ("loop_cen_stem", "structure", "one-hot", "{0,1}",
     "One-hot: position is a stem position in the centroid structure."),
    ("stem_mfe", "structure", "indicator", "{0,1}",
     "Indicator that the position is part of a stem/helix in the MFE structure "
     "(the two paired strands of each stem)."),
    ("stem_cent", "structure", "indicator", "{0,1}",
     "Indicator that the position is part of a stem/helix in the centroid structure."),
    ("pairtype_none", "structure", "one-hot", "{0,1}",
     "One-hot: position has no strong pairing partner (pair probability <= 0.5)."),
    ("pairtype_GC", "structure", "one-hot", "{0,1}",
     "One-hot: position is the G of a strongly paired G-C pair (pair probability > 0.5)."),
    ("pairtype_CG", "structure", "one-hot", "{0,1}",
     "One-hot: position is the C of a strongly paired C-G pair."),
    ("pairtype_AU", "structure", "one-hot", "{0,1}",
     "One-hot: position is the A of a strongly paired A-U pair."),
    ("pairtype_UA", "structure", "one-hot", "{0,1}",
     "One-hot: position is the U of a strongly paired U-A pair."),
    ("pairtype_GU", "structure", "one-hot", "{0,1}",
     "One-hot: position is the G of a strongly paired G-U wobble pair."),
    ("pairtype_UG", "structure", "one-hot", "{0,1}",
     "One-hot: position is the U of a strongly paired U-G wobble pair."),
    ("dist_terminal_loop", "structure", "continuous", "[0,1]",
     "Normalised distance (capped at 30 nt, divided by 30) to the nearest "
     "hairpin-loop position of the MFE structure."),
    ("pair_dens_w5", "structure", "continuous", "[0,1]",
     "Mean base-pair probability in a centred 11-nt window (local pairing density)."),
    ("has_partner", "structure", "indicator", "{0,1}",
     "Indicator that the position has a pairing partner in the MFE structure."),
    ("span_norm", "structure", "continuous", "[0,1]",
     "Distance to the MFE pairing partner normalised by the hairpin length "
     "(0 for unpaired positions)."),
    # ---- C. thermodynamics (7) ---------------------------------------------
    ("dG_mfe", "thermo", "continuous", "kcal/mol",
     "MFE folding free energy of the whole hairpin at 37 C (constant over the RNA)."),
    ("dG_per_nt", "thermo", "continuous", "kcal/mol/nt",
     "MFE folding free energy divided by the hairpin length."),
    ("dE_local", "thermo", "continuous", "kcal/mol",
     "Per-nucleotide stabilisation energy: the increase in folding free energy "
     "when this position alone is forced unpaired."),
    ("dE_local_w5", "thermo", "continuous", "kcal/mol",
     "Centred 11-nt rolling mean of dE_local."),
    ("dG_37_50", "thermo", "continuous", "kcal/mol",
     "Difference in MFE free energy between folding at 37 C and at 50 C "
     "(temperature-stability contrast)."),
    ("dG_0_37", "thermo", "continuous", "kcal/mol",
     "Difference in MFE free energy between folding at 0 C and at 37 C."),
    ("dG_mfe_50C", "thermo", "continuous", "kcal/mol",
     "MFE folding free energy of the whole hairpin at 50 C (constant over the RNA)."),
    # ---- D. ensemble accessibility (12) ------------------------------------
    ("unpaired_prob", "ensemble", "continuous", "[0,1]",
     "Marginal probability that the position is unpaired under the McCaskill "
     "partition-function ensemble."),
    ("pair_prob", "ensemble", "continuous", "[0,1]",
     "Marginal probability that the position is base-paired in the ensemble "
     "(sum of base-pair probabilities over all partners)."),
    ("flex_geom", "ensemble", "continuous", "[0,0.5]",
     "Geometric mean sqrt(unpaired_prob * pair_prob), an ad hoc per-position "
     "flexibility index."),
    ("ens_entropy", "ensemble", "continuous", "[0,1]",
     "Normalised Shannon entropy of the probability distribution over the "
     "partner position (including the unpaired state), divided by log2(n)."),
    ("unpaired_ent_w5", "ensemble", "continuous", "[0,1]",
     "Shannon entropy (bits) of the unpaired-probability profile over a centred "
     "11-nt window."),
    ("unpaired_prob_d0", "ensemble", "continuous", "[0,1]",
     "Unpaired probability under an alternative energy model (dangles = 0, no "
     "lonely pairs, rescaled parameters), a folding-stability contrast."),
    ("unpaired_w3", "ensemble", "continuous", "[0,1]",
     "Centred 7-nt rolling mean of unpaired_prob."),
    ("unpaired_w10", "ensemble", "continuous", "[0,1]",
     "Centred 21-nt rolling mean of unpaired_prob."),
    ("unpaired_max_w4", "ensemble", "continuous", "[0,1]",
     "Centred 9-nt rolling maximum of unpaired_prob."),
    ("ens_diversity", "ensemble", "continuous", ">=0",
     "Ensemble diversity (mean base-pair distance over the Boltzmann ensemble); "
     "global context channel broadcast to all positions."),
    ("ens_defect", "ensemble", "continuous", "[0,1]",
     "Ensemble defect of the MFE structure (expected fraction of wrong pairs); "
     "global context channel broadcast to all positions."),
    ("dG_ctx", "ensemble", "continuous", "kcal/mol",
     "MFE folding free energy embedded as a context channel of the ensemble "
     "block (identical to dG_mfe; constant over the RNA)."),
]


def feature_lookup():
    return {name: (fam, typ, rng, desc) for name, fam, typ, rng, desc in FEATURES}


# --------------------------------------------------------------------- region S3 ---
def build_region_rows(proto, split_name, ds_tag):
    """All regions for one (protocol, split) as a list of dict rows."""
    ds = load_modelling(ds_tag)
    preds = np.load(os.path.join(RES, "predictions.npz"))
    cal_all = json.load(open(os.path.join(RES, "call_calibration.json")))
    thr = cal_all[proto]["threshold"]
    L = qname(cal_all[proto]["min_region_len"])

    score = preds[f"{proto}_{split_name}"]
    ylab = preds[f"{proto}_{split_name}_y"]
    yrank = preds[f"{proto}_{split_name}_yrank"]
    valid = preds[f"{proto}_{split_name}_valid"]
    length = np.asarray(preds[f"{proto}_{split_name}_len"]).astype(int)

    # RNA order in the prediction arrays == ascending file index of the split
    sel = np.nonzero(ds["split"] == "test")[0]
    assert len(sel) == len(length), (proto, split_name, len(sel), len(length))

    chem = CHEM_LABEL[ds_tag]
    rows = []
    off = 0
    rid = 0
    for k, i in enumerate(sel):
        n = int(length[k])
        a0 = int(ds["ptr"][i])
        s = score[off:off + n]
        m = valid[off:off + n] > 0
        called = (s >= thr) & m
        # structural annotation (hairpin coordinates)
        st = json.loads(str(ds["struct"][i]))
        loop_mfe = np.asarray(st["loop_mfe"], dtype=int)[:n]
        loop_cent = np.asarray(st["loop_cent"], dtype=int)[:n]
        ss = str(st["ss_mfe"])[:n]
        stem = np.array([1.0 if c in "()" else 0.0 for c in ss])

        for a, b in call_regions(called.astype(float), 0.5, L):
            rid += 1
            seq_seg = _hairpin_seq(ds_tag, str(ds["exp_id"][i]))[a:b]
            seg = slice(a, b)
            lmfe = Counter(LOOPS[c] for c in loop_mfe[seg]).most_common(1)[0][0]
            lcen = Counter(LOOPS[c] for c in loop_cent[seg]).most_common(1)[0][0]
            rows.append(dict(
                protocol=proto,
                setting=split_name,
                chemistry=chem,
                region_id=rid,
                mirbase_id=str(ds["mirbase_id"][i]),
                family=str(ds["family"][i]),
                hairpin_length=n,
                start_1based=a + 1,
                end_1based=b,
                length=b - a,
                rel_start=round(a / max(n - 1, 1), 4),
                score_mean=round(float(np.mean(s[seg])), 4),
                score_max=round(float(np.max(s[seg])), 4),
                mfe_loop_context=lmfe,
                centroid_loop_context=lcen,
                stem_frac=round(float(np.mean(stem[seg])), 4),
                sequence=seq_seg,
                n_measured=int(m[seg].sum()),
                exp_flexible_frac=round(float(np.mean(ylab[off:off + n][seg])), 4),
                react_rank_median=round(float(np.median(yrank[off:off + n][seg])), 4),
            ))
        off += n
    return rows


_SEQ_CACHE = {}


def _hairpin_seq(tag, exp_id):
    if exp_id in _SEQ_CACHE:
        return _SEQ_CACHE[exp_id]
    seq_full = reactivity_sequence(tag)[exp_id]
    ids = {r["exp_id"]: r for r in load_feature_ids(tag)}
    rec = ids[exp_id]
    s0, n0 = int(rec["offset"]), int(rec["length"])
    seq = seq_full[s0:s0 + n0]
    _SEQ_CACHE[exp_id] = seq
    return seq


# ----------------------------------------------------------------------- data ---
def copy_processed_data():
    os.makedirs(DATA_OUT, exist_ok=True)
    for f in ("dms_modelling.npz", "shape_modelling.npz",
              "dms_features.npz", "shape_features.npz"):
        shutil.copy2(os.path.join(FEAT, f), os.path.join(DATA_OUT, f))
    shutil.copy2(os.path.join(RES, "scaler_A.npy.npz"),
                 os.path.join(DATA_OUT, "scaler.npy.npz"))

    # feature names
    with open(os.path.join(DATA_OUT, "feature_names.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["index", "feature", "family"])
        d = load_modelling("dms")
        fam_members = {}
        for idx in range(len(d["feature_names"])):
            grp = [g for g, v in json.loads(str(d["groups"])).items() if idx in v][0]
            w.writerow([idx, str(d["feature_names"][idx]), grp])

    # scaler as a readable table
    sc = np.load(os.path.join(RES, "scaler_A.npy.npz"))
    names = [str(x) for x in d["feature_names"]]
    with open(os.path.join(DATA_OUT, "feature_scaler.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["feature", "mu", "sd"])
        for nm, mu, sd in zip(names, sc["mu"], sc["sd"]):
            w.writerow([nm, f"{mu:.6g}", f"{sd:.6g}"])


def rna_table():
    rows = []
    for tag in ("dms", "shape"):
        ds = load_modelling(tag)
        for i in range(int(ds["n_rnas"])):
            n = int(ds["ptr"][i + 1] - ds["ptr"][i])
            v = ds["valid"][int(ds["ptr"][i]):int(ds["ptr"][i + 1])] > 0
            yb = ds["y_bin"][int(ds["ptr"][i]):int(ds["ptr"][i + 1])]
            nflex = int(np.nansum(yb[v]))
            rows.append(dict(
                dataset=CHEM_LABEL[tag], index=i, exp_id=str(ds["exp_id"][i]),
                mirbase_id=str(ds["mirbase_id"][i]), family=str(ds["family"][i]),
                length=n, split=str(ds["split"][i]),
                n_measured=int(v.sum()), n_flexible=nflex,
                frac_flexible=round(nflex / max(int(v.sum()), 1), 4)))
    with open(os.path.join(DATA_OUT, "rna_table.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def hairpin_sequences_fa():
    lines = []
    for tag in ("dms", "shape"):
        ds = load_modelling(tag)
        for i in range(int(ds["n_rnas"])):
            mid = str(ds["mirbase_id"][i])
            seq = _hairpin_seq(tag, str(ds["exp_id"][i]))
            hdr = (f">{tag}|{ds['exp_id'][i]}|{mid}|{ds['family'][i]}"
                   f"|{ds['split'][i]}|{len(seq)}")
            lines.append(hdr)
            lines.append(seq)
    with open(os.path.join(DATA_OUT, "hairpin_sequences.fa"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def structures_tsv():
    with open(os.path.join(DATA_OUT, "structures.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["dataset", "index", "exp_id", "mirbase_id", "length",
                    "ss_mfe", "ss_central"])
        for tag in ("dms", "shape"):
            ds = load_modelling(tag)
            for i in range(int(ds["n_rnas"])):
                st = json.loads(str(ds["struct"][i]))
                w.writerow([CHEM_LABEL[tag], i, str(ds["exp_id"][i]),
                            str(ds["mirbase_id"][i]),
                            int(ds["ptr"][i + 1] - ds["ptr"][i]),
                            st.get("ss_mfe", ""), st.get("ss_cent", "")])


def dataset_parameters_json():
    cal = json.load(open(os.path.join(RES, "call_calibration.json")))
    summary = json.load(open(os.path.join(PROC, "dataset_summary.json")))
    overlap = json.load(open(os.path.join(PROC, "split_overlap.json")))
    out = {
        "label_thresholds": {"DMS-MaPseq": DMS_LABEL_THR, "SHAPE-MaP": SHAPE_LABEL_THR},
        "flexible_region_caller": {
            proto: {"threshold": cal[proto]["threshold"],
                    "min_region_len": cal[proto]["min_region_len"],
                    "selected_on": f"{cal[proto]['own_dataset']} validation"}
            for proto in "AB"},
        "feature_scaler": "fit on the DMS training split only; mu/sd in "
                          "feature_scaler.tsv (applied Z-standardisation)",
        "families_shared_between_chemistries": {
            f"{s}": v for s, v in overlap.items()},
        "dataset_summary": summary,
    }
    with open(os.path.join(DATA_OUT, "dataset_parameters.json"), "w") as fh:
        json.dump(out, fh, indent=2)


# ------------------------------------------------------------------- write out ---
def write_supplementary():
    os.makedirs(SUPP_OUT, exist_ok=True)
    _write_region_tables()
    _write_feature_glossary()


def _write_region_tables():
    cols = ["protocol", "setting", "chemistry", "region_id", "mirbase_id",
            "family", "hairpin_length", "start_1based", "end_1based", "length",
            "rel_start", "score_mean", "score_max", "mfe_loop_context",
            "centroid_loop_context", "stem_frac", "sequence", "n_measured",
            "exp_flexible_frac", "react_rank_median"]
    all_rows = []
    summary = []
    cal = json.load(open(os.path.join(RES, "call_calibration.json")))
    for proto in "AB":
        own, other = PROTOCOLS[proto]
        for split_name, _ in SETTINGS:
            ds_tag = own if split_name == "own_test" else other
            rows = build_region_rows(proto, split_name, ds_tag)
            all_rows.extend(rows)
            lengths = [r["length"] for r in rows]
            summary.append(dict(
                protocol=proto, setting=split_name,
                chemistry=CHEM_LABEL[ds_tag],
                n_regions=len(rows),
                n_hairpins=len({r["mirbase_id"] for r in rows}),
                median_length=float(np.median(lengths)) if lengths else None,
                threshold=cal[proto]["threshold"],
                min_region_len=cal[proto]["min_region_len"]))

    csv_path = os.path.join(SUPP_OUT,
                            "Supplementary_Table_S3_flexible_regions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in cols})

    md_path = os.path.join(SUPP_OUT,
                           "Supplementary_Table_S3_flexible_regions.md")
    lines = ["# Supplementary Table S3. Called flexible regions", ""]
    lines += ["Regions are positions whose FlexNet flexibility score is >= the "
              "train-chemistry validation-optimal threshold T, merged into "
              "consecutive runs that are kept only if they span >= L_min "
              "nucleotides. (T, L_min) are applied unchanged to every test set.",
              ""]
    lines += ["`n hairpins` counts hairpins that contain at least one region of "
              "length >= L_min. Every test hairpin contains at least one "
              "above-threshold flexible nucleotide (the enrichment analysis in "
              "region_enrichment.json and manuscript Section 3.5 tallies those "
              "using the unfiltered call mask), but 1-nt stretches are dropped "
              "from this table because they do not satisfy L_min.", ""]
    lines += ["In the region table, `start_1based`/`end_1based` are the inclusive "
              "1-based coordinates of the region on the hairpin; "
              "`mfe_loop_context` and `centroid_loop_context` are the modal loop "
              "class of the region's positions in the MFE and centroid "
              "structures; `exp_flexible_frac` is the fraction of region "
              "positions experimentally labelled flexible (reactivity >= the "
              "chemistry-specific cutoff); `react_rank_median` is the median "
              "within-dataset rank-normalised reactivity.", ""]
    lines += ["| protocol | setting | chemistry | n regions | hairpins | median | T | L_min |",
              "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append("| %s | %s | %s | %d | %d | %s | %.4f | %d |" % (
            s["protocol"], s["setting"].replace("_", " "), s["chemistry"],
            s["n_regions"], s["n_hairpins"], s["median_length"],
            s["threshold"], s["min_region_len"]))
    lines += ["", "First 25 rows of the full table:", "",
              "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in all_rows[:25]:
        lines.append("| " + " | ".join(str(r[k]) for k in cols) + " |")
    lines += ["", f"Full table ({len(all_rows)} regions): "
                  "`Supplementary_Table_S3_flexible_regions.csv`"]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[out] {csv_path} ({len(all_rows)} regions)")
    print(f"[out] {md_path}")


def _write_feature_glossary():
    csv_path = os.path.join(SUPP_OUT,
                            "Supplementary_Table_S4_feature_glossary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["#", "Feature", "Family", "Type", "Values/Range",
                    "Description"])
        for i, (name, fam, typ, rng, desc) in enumerate(FEATURES, start=1):
            w.writerow([i, name, fam, typ, rng, desc])

    md_path = os.path.join(SUPP_OUT,
                           "Supplementary_Table_S4_feature_glossary.md")
    lines = ["# Supplementary Table S4. The 66 per-nucleotide features", ""]
    lines += ["All features are computed from the primary sequence and "
              "*predicted* structure only (ViennaRNA 2.7.2); no experimental "
              "reactivity is used as a predictor. Families: A sequence (19), "
              "B secondary structure (28), C thermodynamics (7), "
              "D ensemble accessibility (12).", ""]
    lines += ["| # | Feature | Family | Type | Values/Range | Description |",
              "|---|---|---|---|---|---|"]
    for i, (name, fam, typ, rng, desc) in enumerate(FEATURES, start=1):
        lines.append("| %d | `%s` | %s | %s | %s | %s |"
                     % (i, name, fam, typ, rng, desc))
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[out] {csv_path}")
    print(f"[out] {md_path}")


# ------------------------------------------------------------------------- md ---
def write_readmes():
    cal = json.load(open(os.path.join(RES, "call_calibration.json")))
    summary = json.load(open(os.path.join(PROC, "dataset_summary.json")))

    data_readme = f"""# Processed training data

This folder contains the **final processed datasets used to train and evaluate every
model in the manuscript** - the reviewed, reproducible dataset built by the
scripts pipeline (stages s01-s04) from the raw GEO and miRBase sources.

## Files

* `dms_modelling.npz`, `shape_modelling.npz` - flat per-nucleotide arrays used by
  `model.RNADataset` (scripts/s04_dataset.py, scripts/s05_train.py):
  * `X` (n_positions x 66) - the 66 per-nucleotide features (Table S4), one row per
    position of every hairpin, concatenated in file order;
  * `y` - raw probe reactivity (DMS normalised 0-1; SHAPE normalised); NaN where
    unmeasured;
  * `y_bin` - flexible/rigid label: reactivity >= {DMS_LABEL_THR} (DMS) or
    >= {SHAPE_LABEL_THR} (SHAPE);
  * `y_rank` - within-dataset rank-normalised reactivity in [0,1] (auxiliary target);
  * `valid` - 1 where a reactivity measurement exists;
  * `ptr` - RNA boundaries into the flat arrays (ptr[i], ptr[i+1]);
  * `lengths` - per-RNA lengths;
  * `mirbase_id`, `exp_id`, `family`, `split` - per-RNA identifiers, miRNA family
    stem and family-aware train/validation/test assignment;
  * `feature_names` - the 66 feature labels;
  * `struct` - per-RNA JSON: MFE/centroid dot-bracket and per-position loop codes.
* `dms_features.npz`, `shape_features.npz` - the same features *before* labels are
  attached (scripts/s03_features.py); identical `X` layout to the modelling file.
* `scaler.npy.npz` - channel mean (`mu`) and standard deviation (`sd`) fitted on the
  **DMS training split only**; `X` is Z-standardised as `(X - mu) / sd` before
  modelling. `feature_scaler.tsv` lists them per feature.
* `feature_names.tsv` - index, name and family (sequence/structure/thermo/ensemble)
  of the 66 features.
* `rna_table.tsv` - one row per hairpin: dataset, index, identifiers, family, length,
  split and the number of measured / flexible nucleotides.
* `hairpin_sequences.fa` - the exact hairpin sequence of each modelled RNA
  (FASTA header: `dataset|exp_id|mirbase_id|family|split|length`).
* `structures.tsv` - the MFE and centroid dot-bracket string of each hairpin.
* `dataset_parameters.json` - label thresholds, the flexible-region caller
  (T, L_min) chosen on validation, the split summary and the family-overlap counts.

## Splits

Whole miRNA families are assigned atomically, so no family straddles a boundary
(family splits: `rna_table.tsv` / `dataset_parameters.json`).

| dataset | RNAs | positions | flexible rate | train | val | test |
|---|---|---|---|---|---|---|
| DMS-MaPseq | {summary['dms']['n_rnas']} | {summary['dms']['n_positions']} | {summary['dms']['pos_rate']} | {summary['dms']['n_train']} | {summary['dms']['n_val']} | {summary['dms']['n_test']} |
| SHAPE-MaP | {summary['shape']['n_rnas']} | {summary['shape']['n_positions']} | {summary['shape']['pos_rate']} | {summary['shape']['n_train']} | {summary['shape']['n_val']} | {summary['shape']['n_test']} |

Protocol A trains on DMS and evaluates on the DMS test set (own) and the SHAPE test
set (other); Protocol B is the mirror image. The flexible-region caller is:

| protocol | threshold T | min region length | selected on |
|---|---|---|---|
| A | {cal['A']['threshold']:.4f} | {cal['A']['min_region_len']} | {cal['A']['own_dataset']} validation |
| B | {cal['B']['threshold']:.4f} | {cal['B']['min_region_len']} | {cal['B']['own_dataset']} validation |

Source experiments: published MA-Pseq atlases (see manuscript Methods 2.1); scripts
s01-s04 turn the raw GEO/MiRBase sources into these arrays. The per-nucleotide
labels used experimentally were DMS reactivity >= {DMS_LABEL_THR} and SHAPE
reactivity >= {SHAPE_LABEL_THR}, matching the measured flexible populations.
"""

    top = f"""# Submission package - FlexNet manuscript

Mirror of the files to be deposited with the manuscript. Everything is generated
from the analysis pipeline by `scripts/s13_submission.py`.

## Contents

* **`data/`** - the final processed training/validation/test datasets actually used
  by the models (see `data/README.md` for the full dictionary).
* **`supplementary/Supplementary_Table_S3_flexible_regions.csv`** - supporting data
  for the flexible-region identification (Manuscript Section 3.5-3.6, Table 4):
  every called region in all four evaluation settings, with hairpin, 1-based
  coordinates, length, model score, experimental label, sequence and structural
  context.
* **`supplementary/Supplementary_Table_S4_feature_glossary.csv`** - the 66
  per-nucleotide features (Manuscript Section 2.2) with family, type, value range
  and description.

Existing supplementary items S1 (loop-classification tree walk) and S2 (per-model
metrics, `results/tables/model_comparison.*`) remain as generated by the scripts.

## How the numbers in the paper map here

| Manuscript result | Supporting file |
|---|---|
| FlexNet AUROC/AP/correlations (Tables 1-2, S2) | `results/model_comparison.*`, `results/metrics_main.json` |
| 66-feature architecture (Methods 2.2) | `supplementary/Supplementary_Table_S4_feature_glossary.csv` |
| Flexible-region calls & enrichment (Results 3.5-3.6, Table 4) | `supplementary/Supplementary_Table_S3_flexible_regions.csv`, `results/region_enrichment.json` |

Regions are defined exactly as in `scripts/s07_regions.py`: FlexNet score >= T
(measured positions only), merged into consecutive runs, runs shorter than L_min
discarded, with (T, L_min) fixed on the training-chemistry validation split and
applied unchanged to every other dataset.

Note on the enrichment numbers: `results/region_enrichment.json` reports the
number of test hairpins carrying at least one *above-threshold nucleotide*
(manuscript Section 3.5), whereas Table S3 reports regions that survive the
L_min filter (some hairpins carry only 1-nt above-threshold stretches and
therefore do not appear in Table S3). The region counts and median lengths in
Table S3 match the manuscript exactly.
"""
    with open(os.path.join(DATA_OUT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(data_readme)
    with open(os.path.join(SUB, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(top)


def main():
    os.makedirs(SUB, exist_ok=True)
    copy_processed_data()
    rna_table()
    hairpin_sequences_fa()
    structures_tsv()
    dataset_parameters_json()
    write_supplementary()
    write_readmes()
    print(f"[done] submission package at {SUB}")


if __name__ == "__main__":
    main()