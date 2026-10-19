"""
Figure generation for the manuscript (SVG output, BioInformatics style).

Figures
  fig2_performance.svg   ROC curves for the four protocol/split combinations
  fig3_baselines.svg     own- vs cross-chemistry AUROC for FlexNet and the
                         classic ML baselines
  fig4_ablations.svg     feature-module ablation AUROC (protocol A)
  fig5_enrichment.svg    flexible-region odds ratios by structural context
  fig6_regions.svg       called-region length distributions and loop contexts
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, auc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#e07b00", "#2e8b57", "#c0392b", "#7f8c8d"
LT = {"A": "DMS-MaPseq", "B": "SHAPE-MaP"}


def base_style():
    plt.rcParams.update({
        "font.family": "Helvetica, Arial, sans-serif",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 100, "savefig.dpi": 100,
    })


def load_pred(k):
    p = np.load(os.path.join(RESULTS, "predictions.npz"))
    m = p[k + "_valid"] > 0
    return p[k + "_y"][m], p[k][m]


def load_baselines():
    b = json.load(open(os.path.join(RESULTS, "baselines.json")))
    return b


def fig2_performance():
    base_style()
    fig, ax = plt.subplots(2, 2, figsize=(4.6, 3.9))
    titles = {"A_own_test": "A / DMS test", "A_other_test": "A / SHAPE test",
              "B_own_test": "B / SHAPE test", "B_other_test": "B / DMS test"}
    for (k, (r, c)) in zip(titles, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        y, p = load_pred(k)
        fpr, tpr, _ = roc_curve(y, p)
        a = auc(fpr, tpr)
        ax[r, c].plot(fpr, tpr, color=BLUE, lw=1.4)
        ax[r, c].plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.8)
        ax[r, c].set_title(f"{titles[k]}  (AUC {a:.3f})")
        ax[r, c].set_xlabel("False positive rate")
        ax[r, c].set_ylabel("True positive rate")
        ax[r, c].set_xlim(0, 1)
        ax[r, c].set_ylim(0, 1)
    fig.tight_layout()
    out = os.path.join(FIG, "fig2_performance.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


def fig3_baselines():
    base_style()
    b = load_baselines()
    models = ["flexnet", "gbm", "rf", "logistic", "svm"]
    names = ["FlexNet", "GBM", "Random forest", "Logistic", "SVM"]
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.4))
    met = json.load(open(os.path.join(RESULTS, "metrics_main.json")))
    for pi, proto in enumerate("AB"):
        row = {b[proto][i]["model"]: b[proto][i] for i in range(len(b[proto]))}
        own = [met[f"{proto}/own_test"]["auc"] if m == "flexnet"
               else row[m]["own_test"]["auc"] for m in models]
        oth = [met[f"{proto}/other_test"]["auc"] if m == "flexnet"
               else row[m]["other_test"]["auc"] for m in models]
        x = np.arange(len(models))
        w = 0.38
        ax[pi].bar(x - w / 2, own, w, label="own chemistry", color=BLUE)
        ax[pi].bar(x + w / 2, oth, w, label="other chemistry", color=ORANGE)
        ax[pi].set_xticks(x)
        ax[pi].set_xticklabels(names, rotation=20, ha="right")
        ax[pi].set_ylim(0.65, 1.0)
        ax[pi].set_title(f"Protocol {proto} ({LT[proto]})")
        ax[pi].set_ylabel("AUROC")
        ax[pi].axhline(0.5, color=GREY, ls=":", lw=0.8)
    ax[0].legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    out = os.path.join(FIG, "fig3_baselines.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


def fig4_ablations():
    base_style()
    b = load_baselines()
    cfg = [("full (FlexNet)", "flexnet", None), ("sequence only", "gbm/sequence(19)", None),
           ("structure only", "gbm/structure(28)", None), ("thermo only", "gbm/thermo(7)", None),
           ("ensemble only", "gbm/ensemble(12)", None), ("no ensemble", "gbm/no_ensemble(57)", None),
           ("no conservation", "gbm/no_conservation(65)", None), ("permuted labels", "gbm/permuted-labels", None)]
    flex = json.load(open(os.path.join(RESULTS, "metrics_main.json")))
    names, own, oth = [], [], []
    names.append("FlexNet (full)")
    own.append(flex["A/own_test"]["auc"])
    oth.append(flex["A/other_test"]["auc"])
    for nm, key, _ in cfg[1:]:
        row = [i for i in range(len(b["A"])) if b["A"][i]["model"] == key][0]
        names.append(nm)
        own.append(b["A"][row]["own_test"]["auc"])
        oth.append(b["A"][row]["other_test"]["auc"])
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, own, w, label="DMS own test", color=BLUE)
    ax.bar(x + w / 2, oth, w, label="SHAPE cross-test", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=6.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    out = os.path.join(FIG, "fig4_ablations.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


def fig5_enrichment():
    base_style()
    r = json.load(open(os.path.join(RESULTS, "region_enrichment.json")))
    order = ["hairpin", "bulge", "internal", "multiloop", "external", "stem"]
    labels = {"hairpin": "hairpin loop", "bulge": "bulge", "internal": "internal loop",
              "multiloop": "multiloop", "external": "external loop", "stem": "stem"}
    fig, ax = plt.subplots(2, 2, figsize=(4.6, 3.6))
    for (tag, (r_, c_)) in zip(["A/own_test", "A/other_test", "B/own_test", "B/other_test"],
                               [(0, 0), (0, 1), (1, 0), (1, 1)]):
        ors, qs = [], []
        for mot in order:
            key = f"in a stem (MFE structure)" if mot == "stem" else f"in a {labels[mot]} (MFE)"
            row = next((z for z in r["enrichment"][tag] if z["comparison"] == key), None)
            if row is None or row["odds_ratio"] <= 0:
                ors.append(np.nan); qs.append(np.nan)
            else:
                ors.append(row["odds_ratio"]); qs.append(row["q_value"])
        def _col(o, q):
            if np.isnan(o) or q >= 1e-3:
                return GREY
            return GREEN if o > 1 else RED
        cols = [_col(o, q) for o, q in zip(ors, qs)]
        x = np.arange(len(order))
        ax[r_, c_].barh(x, ors, color=cols, height=0.62)
        ax[r_, c_].set_yticks(x)
        ax[r_, c_].set_yticklabels([labels[m] for m in order], fontsize=6.5)
        ax[r_, c_].axvline(1, color=GREY, ls="--", lw=0.7)
        ax[r_, c_].set_xscale("log")
        ax[r_, c_].set_xlim(0.003, 500)
        ax[r_, c_].set_title(f"Protocol A / own (DMS)" if tag == "A/own_test"
                             else f"Protocol A / cross (SHAPE)" if tag == "A/other_test"
                             else f"Protocol B / own (SHAPE)" if tag == "B/own_test"
                             else "Protocol B / cross (DMS)")
        for xi, (o, q) in enumerate(zip(ors, qs)):
            if not np.isnan(o):
                ax[r_, c_].text(o * 1.15, xi, f"{o:.2g}", va="center", fontsize=5.5)
    fig.tight_layout()
    out = os.path.join(FIG, "fig5_enrichment.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


def fig6_regions():
    base_style()
    r = json.load(open(os.path.join(RESULTS, "region_enrichment.json")))
    tags = ["A/own_test", "A/other_test", "B/own_test", "B/other_test"]
    fig, ax = plt.subplots(2, 2, figsize=(4.6, 3.6))
    for (tag, (r_, c_)) in zip(tags, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        sh = r["region_shapes"][tag]
        ctx = sh["loop_context"]
        tot = sum(ctx.values())
        parts = [(k, v / tot) for k, v in ctx.items()]
        cols = {"hairpin": GREEN, "internal": BLUE, "bulge": ORANGE, "external": GREY,
                "multiloop": RED}
        left = 0.0
        for k, f in parts:
            ax[r_, c_].bar([0], [f], width=0.7, bottom=left, color=cols.get(k, GREY),
                           label=k if (r_, c_) == (0, 0) else None)
            left += f
        ax[r_, c_].set_xlim(-0.6, 0.6)
        ax[r_, c_].set_ylim(0, 1)
        ax[r_, c_].set_xticks([])
        ax[r_, c_].set_yticks([])
        ax[r_, c_].set_title(
            f"Protocol {tag[0]} / {'own' if 'own' in tag else 'cross'}: "
            f"{sh['n_regions']} regions, median len {sh['median_length']}")
        ax[r_, c_].annotate("loop context of\nregion starts", (0, 0.5),
                            ha="center", va="center", fontsize=6, color=GREY)
    ax[1, 1].legend(loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout()
    out = os.path.join(FIG, "fig6_regions.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    fig2_performance()
    fig3_baselines()
    fig4_ablations()
    fig5_enrichment()
    fig6_regions()
    print("[done] figures written to", FIG)