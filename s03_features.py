"""
Stage 3: nucleotide-level feature extraction.

Four feature families, all computed from sequence and *predicted* structure only.
No experimental reactivity is ever used as a predictor, which is what keeps the
later SHAPE/DMS evaluation non-circular.

  A  sequence          identity, local composition/GC, entropy, position, arms
  B  secondary structure MFE/centroid pairing state, motif, pair type, geometry
  C  thermodynamics     MFE, per-nucleotide energy, ensemble dG, diversity
  D  ensemble access    unpaired probability, pair probability, ensemble entropy

Writes one .npz per dataset to data/features/.
"""
import json
import os
import sys

import numpy as np
import RNA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FEAT = os.path.join(ROOT, "data", "features")
os.makedirs(FEAT, exist_ok=True)

TEMP_C = 37.0
TEMP_ALT = 50.0            # second temperature for a folding-stability contrast
BASES = "ACGU"
B2I = {c: i for i, c in enumerate(BASES)}


# ------------------------------------------------------------------ helpers ---
def model(temperature=TEMP_C, dangles=2, noLP=False):
    md = RNA.md()
    md.temperature = float(temperature)
    md.dangles = int(dangles)
    md.noLP = bool(noLP)
    md.uniq_ML = True
    return md


def rolling(arr, radius, func=np.mean):
    """Centred moving average of a 1-D array with edge replication."""
    n = len(arr)
    if n == 0:
        return arr.astype(float)
    pad = np.pad(arr.astype(float), (radius, radius), mode="edge")
    ker = np.ones(2 * radius + 1)
    return np.convolve(pad, ker, mode="valid") * func(ker)


def rolling_max(arr, radius):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    if n == 0:
        return np.asarray(arr, dtype=float)
    pad = np.pad(np.asarray(arr, dtype=float), (radius, radius), mode="edge")
    return sliding_window_view(pad, 2 * radius + 1).max(axis=1)


def sliding_entropy(arr, radius):
    """Shannon entropy (bits) of every centred window of a 1-D signal.

    For a 1-D input the result is a single scalar per position; for an (n, k)
    input the entropy is taken over the window axis of each column, so the
    output stays (n, k) and can be concatenated with other per-base channels.
    """
    a = np.asarray(arr, dtype=float)
    one_d = a.ndim == 1
    a = a[:, None] if one_d else a
    win = 2 * radius + 1
    padded = np.pad(a, ((radius, radius), (0, 0)), mode="edge")
    # explicit window stack -> (n, win, k)
    w = np.stack([padded[i:i + win] for i in range(a.shape[0])], axis=0)
    tot = w.sum(axis=1, keepdims=True)
    p = np.divide(w, np.where(tot == 0, 1.0, tot), out=np.zeros_like(w), where=tot > 0)
    ent = -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=1)
    ent = np.where(tot[:, 0] > 0, ent, 0.0)
    return ent[:, 0] if one_d else ent


def base_entropy(onehot, radius):
    """Local base-composition entropy (bits) at every position.

    onehot is the (n, 4) indicator matrix; the window spans the radius on each
    side with edge replication, and the entropy is taken over the 4 bases.
    """
    n = onehot.shape[0]
    win = 2 * radius + 1
    padded = np.pad(onehot.astype(float), ((radius, radius), (0, 0)), mode="edge")
    w = np.stack([padded[i:i + win].sum(axis=0) for i in range(n)], axis=0)
    tot = w.sum(axis=1, keepdims=True)
    p = np.divide(w, np.where(tot == 0, 1.0, tot), out=np.zeros_like(w), where=tot > 0)
    ent = -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=1)
    return np.where(tot[:, 0] > 0, ent, 0.0)


def parse_structure(ss, n):
    """Return partner index per position (-1 if unpaired)."""
    partner = [-1] * n
    stack = []
    for i, c in enumerate(ss[:n]):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                partner[i], partner[j] = j, i
    return np.array(partner)


MOTIFS = ["hairpin", "internal", "bulge", "multiloop", "external", "stem"]


def parse_loops(ss):
    """Classify every position of a dot-bracket string.

    Returns (helix_label, loop_label) as integer codes into MOTIFS, where
    helix_label marks the two strands of each stem pair and loop_label marks
    the unpaired nucleotides with the type of the element enclosing them
    (hairpin loop, internal loop, bulge, multiloop or external).
    """
    n = len(ss)
    stem = np.full(n, MOTIFS.index("external"), dtype=np.int8)
    loop = np.full(n, MOTIFS.index("external"), dtype=np.int8)

    pairs = []          # (open_idx, close_idx) in order
    stack = []
    for i, c in enumerate(ss):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            pairs.append((j, i))
    for j, i in pairs:
        stem[j] = stem[i] = MOTIFS.index("stem")

    # Walk the nesting tree. elements[0] is the virtual exterior.
    elements = [[]]
    cur = elements[0]
    stack = [elements[0]]
    for i, c in enumerate(ss):
        if c == "(":
            new = []
            cur.append((i, new))
            elements.append(new)
            cur = new
            stack.append(new)
        elif c == ")":
            if len(stack) > 1:
                stack.pop()
                cur = stack[-1]
        else:
            cur.append(i)

    def element_kind(node):
        """hairpin / internal / bulge / multiloop for one element's children.

        `n_child` must count only the nested helices. Counting the unpaired
        positions as well makes every element look multi-branched, which labels
        the whole hairpin as a multiloop.
        """
        kids = [x for x in node if isinstance(x, tuple)]
        ups = [x for x in node if not isinstance(x, tuple)]
        n_child, n_unpaired = len(kids), len(ups)
        if n_child == 0:
            return "hairpin" if n_unpaired > 0 else None
        if n_child == 1:
            # a single branch: a bulge when all the unpaired bases sit on one
            # side of it, an internal loop when they are split across both
            i0 = kids[0][0]
            if n_unpaired == 0:
                return "internal"
            side = [i < i0 for i in ups]
            return "bulge" if (all(side) or not any(side)) else "internal"
        return "multiloop"

    # the virtual exterior keeps its initial "external" label
    for node in elements[1:]:
        kind = element_kind(node)
        if kind is None:
            continue
        for x in node:
            if not isinstance(x, tuple):
                loop[x] = MOTIFS.index(kind)
    return stem, loop


def count_children(inner):
    """Number of immediately nested helices in a bracket string."""
    depth = 0
    n_child = 0
    for c in inner:
        if c == "(":
            if depth == 0:
                n_child += 1
            depth += 1
        elif c == ")":
            depth -= 1
    return n_child


PAIR_TYPES = ["none", "GC", "CG", "AU", "UA", "GU", "UG"]
PT2I = {p: i for i, p in enumerate(PAIR_TYPES)}


def pair_type_matrix(seq, bpp):
    n = len(seq)
    pt = np.zeros(n, dtype=np.int8)
    i, j = np.triu_indices(n, 1)
    strong = (bpp > 0.5)[i, j]
    keys = np.array([seq[a] + seq[b] for a, b in zip(i, j)], dtype="<U2")
    for code, idx in PT2I.items():
        sel = strong & (keys == code)
        if sel.any():
            pt[i[sel]] = idx
            pt[j[sel]] = idx
    return pt


def ensemble_profiles(seq, bpp, ss_mfe, ss_centroid):
    """D-family profiles: unpaired probability, pair probability, entropies.

    ``bpp`` must already be symmetrised; see :func:`pair_prob_matrix`.
    """
    n = len(seq)
    pair_prob = bpp.sum(axis=1)                 # P(position i is paired)
    unpaired = 1.0 - pair_prob
    # normalised per-position ensemble entropy over all partner assignments
    p = bpp / np.maximum(pair_prob[:, None], 1e-12)
    ent = -(p * np.log2(np.clip(p, 1e-12, None))).sum(axis=1)
    max_ent = np.log2(np.maximum(n, 2))
    ent_n = ent / max_ent
    # probability of being unpaired in the centroid / MFE structure
    in_mfe = np.array([1.0 if c != "." else 0.0 for c in ss_mfe[:n]])
    in_cent = np.array([1.0 if c != "." else 0.0 for c in ss_centroid[:n]])
    return pair_prob, unpaired, ent_n, in_mfe, in_cent


def pair_prob_matrix(fc, n):
    """Symmetric McCaskill base-pair probability matrix, truncated to n.

    The ViennaRNA Python binding returns ``fc.bpp()`` as a strictly
    *upper-triangular* matrix: every entry below the diagonal is exactly zero.
    Summing rows to obtain P(position i is paired) therefore reports ~0 for any
    nucleotide whose partner lies 5' of it, which silently destroys the whole
    ensemble-accessibility feature family. Reflecting the upper triangle gives
    the correct marginal.
    """
    b = np.array(fc.bpp(), dtype=float)
    n_eff = min(n, b.shape[0])
    b = b[:n_eff, :n_eff]
    sym = b + np.triu(b, 1).T
    if n_eff < n:                       # guard against short bpp matrices
        sym = np.pad(sym, ((0, n - n_eff), (0, n - n_eff)))
    return sym



def new_fc(seq, md=None):
    """ViennaRNA's Python binding only accepts the model positionally."""
    return RNA.fold_compound(seq) if md is None else RNA.fold_compound(seq, md)


def energy_profile(seq, md):
    """Per-nucleotide stabilisation energy (kcal/mol) in the MFE structure.

    Each nucleotide is constrained unpaired in turn and the rise in the optimal
    folding energy is recorded, which is the total free energy the structure
    gains from pairing that nucleotide. This is the per-position analogue of a
    Turner decomposition and is computed with the same energy model.
    """
    n = len(seq)
    out = np.zeros(n)
    for i in range(n):
        fc = new_fc(seq, md)
        fc.hc_add_up(i, i)
        out[i] = fc.mfe()[1]
    return out


# ------------------------------------------------------------------- main ---
def featurise(seq, cons, arms):
    """Return (X, names, groups) for one RNA of length n."""
    n = len(seq)
    onehot = np.zeros((n, 4), dtype=np.float32)
    for i, c in enumerate(seq):
        j = B2I.get(c)
        if j is not None:
            onehot[i, j] = 1.0
    gc = (onehot[:, 1] + onehot[:, 2]).astype(float)
    purine = (onehot[:, 0] + onehot[:, 2]).astype(float)

    # ---- A: sequence -------------------------------------------------------
    A = np.concatenate([
        onehot,
        rolling(gc, 3)[:, None], rolling(gc, 5)[:, None], rolling(gc, 10)[:, None],
        purine[:, None],
        base_entropy(onehot, 3)[:, None], base_entropy(onehot, 5)[:, None],
        (np.arange(n) / max(n - 1, 1))[:, None],
        (np.arange(n) / max(n - 1, 1) * 2 - 1)[:, None],
        (1.0 - np.arange(n) / max(n - 1, 1))[:, None],
    ], axis=1)
    A_names = ["seq_A", "seq_C", "seq_G", "seq_U", "gc_w3", "gc_w5", "gc_w10",
               "purine", "seqent_w3", "seqent_w5", "pos_frac", "pos_signed",
               "dist_3p"]
    if len(cons) != n:            # absent or mismatched track -> neutral values
        cons = "." * n
    cons_v = np.array([int(c) / 9.0 if c.isdigit() else np.nan for c in cons], dtype=float)
    cons_has = ~np.isnan(cons_v)
    cons_fill = np.where(cons_has, cons_v, np.nanmean(cons_v) if cons_has.any() else 0.5)
    A = np.concatenate([A, cons_fill[:, None], cons_has.astype(float)[:, None]], axis=1)
    A_names += ["conservation", "cons_detected"]

    if len(arms) != n:
        arms = "." * n
    arm_arr = np.array([1.0 if a == "5p" else 0.0 for a in arms])
    arm3_arr = np.array([1.0 if a == "3p" else 0.0 for a in arms])
    # distance (nt) to the nearest mature-arm boundary, capped at 20
    def dist_to_arm(flags):
        idx = np.where(flags > 0)[0]
        if len(idx) == 0:
            return np.full(n, 20.0)
        d = np.abs(np.arange(n)[:, None] - idx[None, :]).min(axis=1)
        return np.clip(d, 0, 20) / 20.0
    A = np.concatenate([A, arm_arr[:, None], arm3_arr[:, None],
                        dist_to_arm(arm_arr)[:, None], dist_to_arm(arm3_arr)[:, None]],
                       axis=1)
    A_names += ["arm5p", "arm3p", "dist_arm5p", "dist_arm3p"]

    # ---- structure / ensemble ---------------------------------------------
    fc = new_fc(seq, model())
    ss_mfe, mfe = fc.mfe()
    fc.pf()
    bpp = pair_prob_matrix(fc, len(seq))
    # centroid() returns (dot-bracket, ensemble distance)
    ss_cent = fc.centroid()[0]

    partner_mfe = parse_structure(ss_mfe, n)
    stem_mfe, loop_mfe = parse_loops(ss_mfe)
    stem_cent, loop_cent = parse_loops(ss_cent)
    pt_mfe = pair_type_matrix(seq, bpp)
    pair_prob, unpaired, ent_n, in_mfe, in_cent = ensemble_profiles(seq, bpp, ss_mfe, ss_cent)

    # distance to the nearest terminal (hairpin) loop of the MFE structure
    hairpin_pos = np.where(loop_mfe == MOTIFS.index("hairpin"))[0]
    if len(hairpin_pos):
        dh = np.abs(np.arange(n)[:, None] - hairpin_pos[None, :]).min(axis=1)
        dh = np.clip(dh, 0, 30) / 30.0
    else:
        dh = np.ones(n)
    # local pairing density
    dens = rolling(pair_prob, 5)

    B = np.concatenate([
        in_mfe[:, None], in_cent[:, None],
        ((1 - in_mfe) * (1 - in_cent))[:, None],
        np.eye(len(MOTIFS), dtype=np.float32)[loop_mfe],
        np.eye(len(MOTIFS), dtype=np.float32)[loop_cent],
        np.eye(len(MOTIFS), dtype=np.float32)[stem_mfe][:, [MOTIFS.index("stem")]],
        np.eye(len(MOTIFS), dtype=np.float32)[stem_cent][:, [MOTIFS.index("stem")]],
        np.eye(len(PAIR_TYPES), dtype=np.float32)[pt_mfe],
        dh[:, None], dens[:, None],
        (partner_mfe != -1).astype(float)[:, None],
        np.array([0.0 if p == -1 else abs(p - i) for i, p in enumerate(partner_mfe)])[:, None]
        / max(n, 1),
    ], axis=1)
    B_names = (["paired_mfe", "paired_cent", "unpaired_both",
                "loop_mfe_hairpin", "loop_mfe_internal", "loop_mfe_bulge",
                "loop_mfe_multiloop", "loop_mfe_external", "loop_mfe_stem",
                "loop_cen_hairpin", "loop_cen_internal", "loop_cen_bulge",
                "loop_cen_multiloop", "loop_cen_external", "loop_cen_stem",
                "stem_mfe", "stem_cent"]
               + [f"pairtype_{p}" for p in PAIR_TYPES]
               + ["dist_terminal_loop", "pair_dens_w5", "has_partner", "span_norm"])

    # ---- C: thermodynamics -------------------------------------------------
    md50 = model(TEMP_ALT)
    ss50, mfe50 = new_fc(seq, md50).mfe()
    md_cold = model(0.0)
    _, mfe_cold = new_fc(seq, md_cold).mfe()
    md_tc = model()
    _, mfe_tc = new_fc(seq, md_tc).mfe()
    eprof = energy_profile(seq, md_tc) - mfe_tc
    dG = float(mfe) / 10.0
    dG50 = float(mfe50) / 10.0
    dG0 = float(mfe_cold) / 10.0
    C = np.column_stack([
        np.full(n, dG),
        np.full(n, dG / max(n, 1)),
        eprof,
        rolling(eprof, 5),
        np.full(n, (dG - dG50)),
        np.full(n, (dG0 - dG)),
        np.full(n, dG50),
    ])
    C_names = ["dG_mfe", "dG_per_nt", "dE_local", "dE_local_w5",
               "dG_37_50", "dG_0_37", "dG_mfe_50C"]

    # ---- D: ensemble accessibility ----------------------------------------
    md2 = model(dangles=0, noLP=True)
    fc2 = new_fc(seq, md2)
    ss2, mfe2 = fc2.mfe()
    fc2.exp_params_rescale(mfe2)
    fc2.pf()
    bpp2 = pair_prob_matrix(fc2, len(seq))
    unl2 = 1.0 - bpp2.sum(axis=1)
    # ensemble defect and diversity are global; broadcast as context channels
    try:
        defect = float(fc.ensemble_defect(ss_mfe))
    except Exception:
        defect = np.nan
    try:
        diversity = float(fc.mean_bp_distance())
    except Exception:
        diversity = np.nan
    ent_full = sliding_entropy(unpaired, 5)
    D = np.column_stack([
        unpaired, pair_prob, np.sqrt(np.clip(unpaired * pair_prob, 0, None)),
        ent_n, ent_full, unl2,
        rolling(unpaired, 3), rolling(unpaired, 10),
        rolling_max(unpaired, 4),
        np.full(n, diversity if np.isfinite(diversity) else 0.0),
        np.full(n, defect if np.isfinite(defect) else 0.0),
        np.full(n, dG),
    ])
    D_names = ["unpaired_prob", "pair_prob", "flex_geom", "ens_entropy",
               "unpaired_ent_w5", "unpaired_prob_d0", "unpaired_w3",
               "unpaired_w10", "unpaired_max_w4", "ens_diversity", "ens_defect",
               "dG_ctx"]

    X = np.concatenate([A, B, C, D], axis=1).astype(np.float32)
    names = A_names + B_names + C_names + D_names
    groups = {
        "sequence": list(range(len(A_names))),
        "structure": list(range(len(A_names), len(A_names) + len(B_names))),
        "thermo": list(range(len(A_names) + len(B_names),
                             len(A_names) + len(B_names) + len(C_names))),
        "ensemble": list(range(len(A_names) + len(B_names) + len(C_names),
                               len(A_names) + len(B_names) + len(C_names) + len(D_names))),
    }
    return X, names, groups, dict(ss_mfe=ss_mfe, ss_cent=ss_cent,
                                  loop_mfe=loop_mfe, loop_cent=loop_cent,
                                  stem_mfe=stem_mfe, stem_cent=stem_cent,
                                  mfe=mfe, unpaired=unpaired, pair_prob=pair_prob,
                                  ent=ent_n)


def main():
    cons_df = {}
    with open(os.path.join(PROC, "conservation.tsv")) as fh:
        import csv
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            cons_df[r["mirbase_id"]] = r["cons_score"]
    arm_df = {}
    with open(os.path.join(PROC, "arm_annotations.tsv")) as fh:
        import csv
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            arm_df[r["mirbase_id"]] = r["arms"]

    with open(os.path.join(PROC, "dms_reactivity.json")) as fh:
        dms = json.load(fh)
    with open(os.path.join(PROC, "shape_reactivity.json")) as fh:
        shape = json.load(fh)
    loc = {}
    import csv
    with open(os.path.join(PROC, "dms_window_loc.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            loc[r["exp_id"]] = r

    for tag, recs in (("dms", dms), ("shape", shape)):
        print(f"[feat] {tag}: {len(recs)} records", flush=True)
        out_seq, out_lab, out_id, out_struct = [], [], [], []
        n_ok = 0
        for i, rec in enumerate(recs):
            mid = rec["mirbase_id"]
            seq_full = rec["sequence"]
            react = np.array([np.nan if v is None else float(v) for v in rec["reactivity"]])

            if tag == "dms":
                # restrict to the hairpin itself, so both chemistries are
                # evaluated on the same coordinate system
                L = loc.get(rec["exp_id"])
                if L is None or int(L["start"]) < 0:
                    continue
                s, e = int(L["start"]), int(L["end"])
            else:
                s, e = 0, len(seq_full)
            seq = seq_full[s:e]
            if len(seq) < 25 or len(seq) > 400:
                continue
            if "N" in seq:
                continue
            react = react[s:e]
            if len(react) != len(seq):
                continue

            # conservation and arm tracks are indexed on the hairpin itself.
            # For DMS the analysis window is a 230-nt genomic context, so the
            # hairpin interval is [s, e) there but the tracks start at 0.
            cons = cons_df.get(mid, "")
            arms = arm_df.get(mid, "")
            if len(cons) == len(seq):
                pass
            elif s == 0 and len(cons) == len(seq_full):
                cons = cons[s:e]
            else:
                cons = ""
            if len(arms) == len(seq):
                pass
            elif s == 0 and len(arms) == len(seq_full):
                arms = arms[s:e]
            else:
                arms = ""

            try:
                X, names, groups, aux = featurise(seq, cons, arms)
            except Exception as exc:
                print(f"  skip {rec['exp_id']}: {exc}", flush=True)
                continue
            out_seq.append(X)
            out_lab.append(react)
            out_id.append({"exp_id": rec["exp_id"], "mirbase_id": mid,
                           "offset": s, "length": len(seq)})
            out_struct.append({"ss_mfe": aux["ss_mfe"], "ss_cent": aux["ss_cent"],
                               "loop_mfe": aux["loop_mfe"].tolist(),
                               "loop_cent": aux["loop_cent"].tolist()})
            n_ok += 1
            if (i + 1) % 100 == 0:
                print(f"  ...{i + 1}/{len(recs)}", flush=True)

        if not out_seq:
            raise SystemExit(f"no records featurised for {tag}")
        lens = np.array([x.shape[0] for x in out_seq])
        p = os.path.join(FEAT, f"{tag}_features.npz")
        np.savez_compressed(
            p,
            ids=np.array([json.dumps(o) for o in out_id]),
            struct=np.array([json.dumps(o) for o in out_struct], dtype=object),
            feature_names=np.array(names),
            groups=np.array(json.dumps(groups)),
            lengths=lens,
            X=np.concatenate(out_seq, axis=0),
            y=np.concatenate(out_lab, axis=0),
            ptr=np.concatenate([[0], np.cumsum(lens)]),
        )
        print(f"[feat] {tag}: wrote {n_ok} RNAs, {len(names)} features, "
              f"{sum(lens)} positions -> {p}", flush=True)
        print(f"[feat] {tag} group sizes: "
              f"{ {k: len(v) for k, v in groups.items()} }", flush=True)


if __name__ == "__main__":
    main()
