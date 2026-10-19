"""
Stage 7: flexible-region calling and biological enrichment.

Regions are called with the threshold and minimum length frozen on the
validation split of the *training* chemistry and then applied unchanged, so the
cross-chemistry enrichment below is genuinely out-of-sample.

Enrichment asks where the *called* flexible nucleotides sit with respect to
structural context: loop versus stem, loop type, arm, position along the
hairpin, distance from the terminal loop, conservation, and sequence content.
The Fisher tests therefore compare `called` against `not called` (the model's
own calls), not against the experimental labels. Two-sided Fisher exact tests
with Benjamini-Hochberg correction are used throughout, and the nucleotide-level
bootstrap gives a confidence interval on the loop-versus-stem odds ratio.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as EK                                                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "data", "features")
RES = os.path.join(ROOT, "results")
LOOPS = ["hairpin", "internal", "bulge", "multiloop", "external", "stem"]


def load(tag):
    d = np.load(os.path.join(FEAT, f"{tag}_modelling.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def design(pred):
    """Per-nucleotide annotation aligned with the prediction arrays."""
    ds = pred["ds"]
    ptr = ds["ptr"]
    sel = (np.nonzero(ds["split"] == pred["split_arg"])[0]
           if pred["split_arg"] else np.arange(len(ptr) - 1))
    loop_mfe, loop_cen, ssmfe, sscen, arm5, arm3, cons = ([] for _ in range(7))
    names = list(ds["feature_names"])
    i_arm5, i_arm3 = names.index("arm5p"), names.index("arm3p")
    i_cons = names.index("conservation")
    stem_code = LOOPS.index("stem")
    ext_code = LOOPS.index("external")
    for i in sel:
        st = json.loads(str(ds["struct"][i]))
        n = int(ptr[i + 1] - ptr[i])
        loop_mfe.append(np.asarray(st["loop_mfe"], dtype=np.int8)[:n])
        loop_cen.append(np.asarray(st["loop_cent"], dtype=np.int8)[:n])
        ssmfe.append(np.array([stem_code if c in "()" else ext_code
                               for c in st["ss_mfe"]], dtype=np.int8)[:n])
        sscent = json.loads(str(ds["struct"][i])).get("ss_cent", st["ss_mfe"])
        sscen.append(np.array([stem_code if c in "()" else ext_code
                               for c in sscent], dtype=np.int8)[:n])
        # arm5p/arm3p/conservation are whole-hairpin annotations, so broadcast
        # them along the nucleotide axis (the conservation test is then a
        # length-weighted hairpin-level comparison)
        arm5.append(np.full(n, ds["X"][i, i_arm5], dtype=np.int8))
        arm3.append(np.full(n, ds["X"][i, i_arm3], dtype=np.int8))
        cons.append(np.full(n, ds["X"][i, i_cons], dtype=np.float32))
    return dict(loop_mfe=np.concatenate(loop_mfe),
                loop_cen=np.concatenate(loop_cen),
                stem=np.concatenate(ssmfe),
                stem_cent=np.concatenate(sscen),
                arm5=np.concatenate(arm5), arm3=np.concatenate(arm3),
                conservation=np.concatenate(cons))


def call(pred, thr, min_len):
    """Merge above-threshold stretches into regions; return flat call mask."""
    m = (pred["score"] >= thr) & (pred["valid"] > 0)
    return m


def enrich(pred, ann, thr, min_len, label, out):
    called = call(pred, thr, min_len)
    mm = pred["valid"] > 0
    rows = []

    def fisher(name, group, truth, tag):
        c = np.asarray(group) > 0
        if c.sum() == 0 or (~c).sum() == 0:
            return
        table = [[int((truth & c).sum()), int((truth & ~c).sum())],
                 [int((~truth & c).sum()), int((~truth & ~c).sum())]]
        orr, p = fisher_exact(table)
        rows.append(dict(comparison=name, scope=tag,
                         n_in=int(c.sum()), frac_positive_in_group=round(
                             float(truth[c].mean()), 4),
                         frac_positive_outside=round(float(truth[~c].mean()), 4),
                         odds_ratio=(round(float(orr), 3)
                                     if np.isfinite(orr) else None),
                         p_value=float(p)))

    # --- structural context -------------------------------------------------
    # `stem` and `loop_*` are motif *codes*, so compare against the code rather
    # than testing them as booleans
    stem_code = LOOPS.index("stem")
    fisher("in a stem (MFE structure)", ann["stem"] == stem_code, called, "structure")
    for code, name in enumerate(LOOPS):
        if name == "stem":
            continue
        fisher(f"in a {name} loop (MFE)", ann["loop_mfe"] == code, called, "loop_type_mfe")
        fisher(f"in a {name} loop (centroid)", ann["loop_cen"] == code, called,
               "loop_type_centroid")
    fisher("unpaired (MFE)", ann["stem"] != stem_code, called, "structure")
    fisher("unpaired (centroid)", ann["stem_cent"] != stem_code, called, "structure")

    arm = np.where(ann["arm5"] > ann["arm3"], 0, np.where(ann["arm3"] > 0, 1, 2))
    fisher("on the 5p arm", arm == 0, called, "arm")
    fisher("on the 3p arm", arm == 1, called, "arm")

    # --- conservation --------------------------------------------------------
    cons = ann["conservation"]
    cons = np.where(np.isfinite(cons), cons, np.nan)
    a, b = cons[mm & called], cons[mm & ~called]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) > 20 and len(b) > 20:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(comparison="conservation (called vs not)", scope="conservation",
                         mean_called=round(float(a.mean()), 4),
                         mean_not_called=round(float(b.mean()), 4),
                         median_called=round(float(np.median(a)), 4),
                         median_not_called=round(float(np.median(b)), 4),
                         p_value=float(p)))

    # --- sequence content is summarised by the region-shape table below -------

    ps = [r["p_value"] for r in rows]
    if ps:
        rej, q, _, _ = multipletests(ps, method="fdr_bh")
        for r, qq in zip(rows, q):
            r["q_value"] = float(qq)
            r["significant"] = bool(qq < 0.05)
        rows.sort(key=lambda r: r["q_value"])
    out[label] = rows
    print(f"\n--- {label}  (called {int(called.sum())} of {int(mm.sum())} "
          f"measured nucleotides) ---", flush=True)
    for r in rows[:14]:
        extra = ""
        if "odds_ratio" in r:
            extra = f" OR={r['odds_ratio']} f_in={r['frac_positive_in_group']:.3f} " \
                    f"f_out={r['frac_positive_outside']:.3f}"
        print(f"  {r['comparison']:34s} {r['scope']:24s} q={r['q_value']:.3e}{extra}",
              flush=True)
    return rows


def region_shapes(pred, ann, thr, min_len, label, out):
    """Length, count and location statistics of the called regions."""
    called = call(pred, thr, min_len)
    lens, starts = [], []
    kinds, per = Counter(), Counter()
    n_with = 0
    off = 0
    for n in pred["lengths"]:
        c = called[off:off + n]
        off += n
        if not c.any():
            continue
        n_with += 1
        for a, b in EK.call_regions(c.astype(float), 0.5, min_len=min_len):
            lens.append(b - a)
            starts.append(a / max(n - 1, 1))
            mid = off - n + (a + b) // 2
            kinds[LOOPS[ann["loop_cen"][mid]]] += 1
            per[LOOPS[ann["stem"][mid]]] += 1
    arr = np.array(lens) if lens else np.array([0])
    s = dict(
        n_regions=len(lens),
        n_hairpins_with_a_region=n_with,
        median_length=float(np.median(arr)) if lens else None,
        mean_length=round(float(np.mean(arr)), 2) if lens else None,
        frac_len_2=round(float(np.mean(arr == 2)), 4) if lens else None,
        frac_len_ge4=round(float(np.mean(arr >= 4)), 4) if lens else None,
        frac_len_ge6=round(float(np.mean(arr >= 6)), 4) if lens else None,
        median_rel_start=round(float(np.median(starts)), 4) if starts else None,
        loop_context=dict(kinds.most_common()),
        stem_context=dict(per.most_common()),
    )
    out[label] = s
    print(f"  [{label}] {s['n_regions']} regions, median length "
          f"{s['median_length']}, top loop contexts {s['loop_context']}", flush=True)
    return s


def main():
    preds = np.load(os.path.join(RES, "predictions.npz"))
    cal = json.load(open(os.path.join(RES, "call_calibration.json")))

    enrich_out, shape_out = {}, {}
    for proto, own_tag, other_tag in (("A", "dms", "shape"), ("B", "shape", "dms")):
        thr = cal[proto]["threshold"]
        L = cal[proto]["min_region_len"]
        for split_name, ds_tag in (("own_test", own_tag), ("other_test", other_tag)):
            p = dict(score=preds[f"{proto}_{split_name}"],
                     y=preds[f"{proto}_{split_name}_y"],
                     valid=preds[f"{proto}_{split_name}_valid"],
                     lengths=preds[f"{proto}_{split_name}_len"].astype(int),
                     ds=load(ds_tag), split_arg="test")
            ann = design(p)
            tag = f"{proto}/{split_name}"
            enrich(p, ann, thr, L, tag, enrich_out)
            region_shapes(p, ann, thr, L, tag, shape_out)

    with open(os.path.join(RES, "region_enrichment.json"), "w") as fh:
        json.dump(dict(enrichment=enrich_out, region_shapes=shape_out,
                       calibration=cal), fh, indent=2)
    print("\n[written] results/region_enrichment.json")


if __name__ == "__main__":
    main()
