"""
Stage 12: the reviewer-proofing figures.

Everything here is drawn from stored result files only; no model is re-fitted.

  fig7_roc_all_models.svg     ROC curves for FlexNet and all four classical
                              baselines, in all four evaluation settings
  fig8_pr_all_models.svg      the same comparison as precision-recall curves,
                              which is the informative view at ~10-21% prevalence
  fig9_loss_curves.svg        FlexNet training / validation / test loss per
                              epoch (protocols A and B, mean of three seeds with
                              the seed range shaded), plus the per-boosting-
                              iteration equivalent for GBM
  fig10_metrics_all.svg      accuracy, sensitivity and specificity per model
                              in all four settings
  fig11_calibration.svg      reliability of FlexNet and the baselines
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (auc, average_precision_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = ("#1f77b4", "#e07b00", "#2e8b57",
                                          "#c0392b", "#7f8c8d", "#7b3294")
LT = {"A": "DMS-MaPseq", "B": "SHAPE-MaP"}
ORDER = [("flexnet", "FlexNet", BLUE, 2.0),
         ("gbm", "GBM", ORANGE, 1.2),
         ("rf", "Random forest", GREEN, 1.2),
         ("logistic", "Logistic", RED, 1.2),
         ("svm", "SVM", PURPLE, 1.2)]
PANELS = [("A", "own_test", "A / DMS test", "own"),
          ("A", "other_test", "A / SHAPE test", "cross"),
          ("B", "own_test", "B / SHAPE test", "own"),
          ("B", "other_test", "B / DMS test", "cross")]


def base_style():
    plt.rcParams.update({
        "font.family": "Helvetica, Arial, sans-serif",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 100, "savefig.dpi": 100,
        "svg.fonttype": "none",
    })


def scores():
    return np.load(os.path.join(RESULTS, "model_scores.npz"))


def get(d, proto, model, split):
    return (d[f"{proto}|{model}|{split}_y"],
            d[f"{proto}|{model}|{split}_s"])


def _panels(figsize):
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    for (proto, split, title, kind), a in zip(PANELS, ax.ravel()):
        a.plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.7, zorder=0)
        a.set_xlim(-0.01, 1.01)
        a.set_ylim(-0.01, 1.01)
        a.set_xlabel("False positive rate")
        a.set_ylabel("True positive rate")
        a.set_title(f"{title}\n({kind}-chemistry)", fontsize=8)
    return fig, ax


def _legend(fig, ax, models, split):
    handles = [plt.Line2D([], [], color=c, lw=w, label=n)
               for k, n, c, w in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models),
               frameon=False, bbox_to_anchor=(0.5, -0.02))


def fig_roc():
    base_style()
    d = scores()
    models = [(k, n, c, w) for k, n, c, w in ORDER
              if f"A|{k}|own_test_y" in d.files]
    fig, ax = _panels((5.0, 4.6))
    for (proto, split, title, kind), a in zip(PANELS, ax.ravel()):
        for key, name, col, lw in models:
            y, s = get(d, proto, key, split)
            fpr, tpr, _ = roc_curve(y, s)
            a.plot(fpr, tpr, color=col, lw=lw,
                   label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    _legend(fig, ax, [(n, n, c, w) for k, n, c, w in models],
            "own_test")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(FIG, "fig7_roc_all_models.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


def fig_pr():
    base_style()
    d = scores()
    models = [(k, n, c, w) for k, n, c, w in ORDER
              if f"A|{k}|own_test_y" in d.files]
    fig, ax = _panels((5.0, 4.6))
    for (proto, split, title, kind), a in zip(PANELS, ax.ravel()):
        for key, name, col, lw in models:
            y, s = get(d, proto, key, split)
            prec, rec, _ = precision_recall_curve(y, s)
            a.plot(rec, prec, color=col, lw=lw,
                   label=f"{name} (AP {average_precision_score(y, s):.3f})")
        a.axhline(y.mean(), color=GREY, ls=":", lw=0.8)
        a.text(0.55, y.mean() + 0.02, f"prevalence {y.mean():.2f}",
               fontsize=6, color=GREY)
        a.set_xlabel("Recall (sensitivity)")
        a.set_ylabel("Precision (PPV)")
    handles = [plt.Line2D([], [], color=c, lw=w, label=n)
               for k, n, c, w in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(FIG, "fig8_pr_all_models.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


def fig_loss():
    base_style()
    lc = json.load(open(os.path.join(RESULTS, "learning_curves.json")))
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.2))

    for pi, proto in enumerate("AB"):
        runs = [lc["flexnet"][f"{proto}_seed{s}"] for s in (0, 1, 2)]
        n_ep = max(len(r["train"]) for r in runs)
        ep = np.arange(1, n_ep + 1)

        def stack(key):
            arr = np.array([r[key] + [np.nan] * (n_ep - len(r[key]))
                            for r in runs], dtype=float)
            return arr

        for col, (key, label) in enumerate((
                ("train_eval", "train (eval mode)"),
                ("val_eval", "validation"),
                ("test_eval", "test"))):
            arr = stack(key)
            m = np.nanmean(arr, axis=0)
            lo, hi = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
            a = ax[pi, col]
            a.plot(ep, m, color=BLUE, lw=1.4, marker="o", ms=2.4)
            a.fill_between(ep, lo, hi, color=BLUE, alpha=0.18, lw=0)
            if col == 0:
                # the optimiser's own dropout-active average, for reference
                a.plot(ep, np.nanmean(stack("train"), axis=0), color=BLUE,
                       lw=0.9, ls=":", alpha=0.8)
                a.set_ylabel(f"Protocol {proto} ({LT[proto]})\n"
                             f"multitask loss", fontsize=7.5)
                a.axvline(runs[0]["best_epoch"], color=RED, ls="--", lw=0.8)
                a.text(runs[0]["best_epoch"] + 0.15, 0.95, "selected epoch",
                       transform=a.get_yaxis_transform(), fontsize=5.8,
                       color=RED, va="top")
            if pi == 1:
                a.set_xlabel("epoch")
            a.set_title(label, fontsize=8)

    # third column: validation and test AUROC per epoch
    for pi, proto in enumerate("AB"):
        a = ax[pi, 2]
        runs = [lc["flexnet"][f"{proto}_seed{s}"] for s in (0, 1, 2)]
        n_ep = max(len(r["train"]) for r in runs)
        ep = np.arange(1, n_ep + 1)
        for key, col, name in (("val_auc", GREEN, "validation AUROC"),
                               ("test_auc", ORANGE, "test AUROC")):
            arr = np.array([r[key] + [np.nan] * (n_ep - len(r[key]))
                            for r in runs], dtype=float)
            m = np.nanmean(arr, axis=0)
            lo, hi = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
            a.plot(ep, m, color=col, lw=1.4, marker="s", ms=2.2, label=name)
            a.fill_between(ep, lo, hi, color=col, alpha=0.16, lw=0)
        best = int(np.nanargmax(np.nanmean(stack("val_auc"), axis=0))) + 1
        be = int(np.nanargmax(runs[0]["val_auc"])) + 1
        a.axvline(be, color=GREY, ls="--", lw=0.8)
        a.text(be + 0.2, 0.55, f"seed-0 selection\n(epoch {be})",
               fontsize=6, color=GREY)
        a.set_xlabel("epoch")
        if pi == 0:
            a.legend(frameon=False, loc="lower right", fontsize=6.5)
            a.set_ylabel("AUROC")
            a.set_title("AUROC per epoch", fontsize=8)

    fig.suptitle("FlexNet learning curves (mean of three seeds, shaded = seed "
                 "range); test split never used for stopping", fontsize=8, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIG, "fig9a_loss_curves.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)

    # ---- GBM per-boosting-iteration equivalent ---------------------------
    base_style()
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.3))
    for pi, proto in enumerate("AB"):
        g = lc["gbm"][proto]
        it = np.array(g["iterations"])
        a = ax[pi]
        ymax = max(float(np.max(g["log_loss"][k]))
                   for k in ("train", "val", "test"))
        for key, col, name in (("train", BLUE, "train"),
                               ("val", GREEN, "validation"),
                               ("test", ORANGE, "test")):
            a.plot(it, g["log_loss"][key], color=col, lw=1.3, label=name)
        best = int(np.argmin(g["log_loss"]["val"])) + 1
        a.axvline(best, color=GREY, ls="--", lw=0.8)
        a.annotate(f"best validation\nloss: iter {best}",
                   xy=(best, g["log_loss"]["val"][best - 1]),
                   xytext=(0.98, 0.30), textcoords="axes fraction", ha="right",
                   fontsize=6, color=GREY,
                   arrowprops=dict(arrowstyle="->", lw=0.6, color=GREY))
        a.set_xlabel("boosting iteration")
        a.set_title(f"GBM, protocol {proto} ({LT[proto]})", fontsize=8)
        a.set_ylim(0, ymax * 1.08)
        if pi == 0:
            a.set_ylabel("binary log loss")
            a.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig9b_gbm_loss_curves.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


def fig_metrics():
    base_style()
    cmp = json.load(open(os.path.join(RESULTS, "model_comparison.json")))
    models = [("flexnet", "FlexNet"), ("gbm", "GBM"), ("rf", "RF"),
              ("logistic", "Logistic"), ("svm", "SVM")]
    keys = ("accuracy", "sensitivity", "specificity")
    klab = ("Accuracy", "Sensitivity", "Specificity")
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for pi, proto in enumerate("AB"):
        for ci, (key, kl) in enumerate(zip(keys, klab)):
            a = ax[pi, ci]
            x = np.arange(len(models))
            w = 0.38
            own = [cmp[proto][m]["own_test"][key] for m, _ in models]
            oth = [cmp[proto][m]["other_test"][key] for m, _ in models]
            ci_o = [cmp[proto][m]["own_test"]["ci"][key] for m, _ in models]
            ci_t = [cmp[proto][m]["other_test"]["ci"][key] for m, _ in models]
            eo = [(o[1] - o[0]) / 2 for o in ci_o]
            et = [(o[1] - o[0]) / 2 for o in ci_t]
            a.bar(x - w / 2, own, w, color=BLUE, label="own chemistry",
                  yerr=eo, ecolor="black", capsize=1.5, error_kw=dict(lw=0.5))
            a.bar(x + w / 2, oth, w, color=ORANGE, label="other chemistry",
                  yerr=et, ecolor="black", capsize=1.5, error_kw=dict(lw=0.5))
            a.set_xticks(x)
            a.set_xticklabels([n for _, n in models], rotation=30, ha="right",
                             fontsize=6.5)
            a.set_ylim(0, 1.02)
            a.set_title(f"{kl} - protocol {proto} ({LT[proto]})", fontsize=8)
            if ci == 0:
                a.set_ylabel("rate")
            if pi == 0 and ci == 0:
                a.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig10_acc_sens_spec.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


def fig_calibration():
    base_style()
    d = scores()
    models = [(k, n, c, w) for k, n, c, w in ORDER
              if f"A|{k}|own_test_y" in d.files]
    fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.1), sharey=True)
    for pi, (proto, split, title, kind) in enumerate(PANELS):
        a = ax[pi]
        a.plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.7)
        for key, name, col, lw in models:
            y, s = get(d, proto, key, split)
            q = np.clip((np.floor(s * 10) + 0.5) / 10, 0, 1)
            bins = np.linspace(0, 1, 11)
            bi = np.digitize(s, bins[1:-1])
            obs, pred = [], []
            for k in range(10):
                m = bi == k
                if m.sum() > 20:
                    obs.append(y[m].mean())
                    pred.append(q[m].mean())
            a.plot(pred, obs, color=col, lw=lw, marker="o", ms=2.2, label=name)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_xlabel("mean predicted score")
        a.set_title(f"{title}\n({kind}-chemistry)", fontsize=7.5)
        if pi == 0:
            a.set_ylabel("observed flexibility rate")
            a.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG, "fig11_calibration.svg")
    fig.savefig(out)
    plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    fig_roc()
    fig_pr()
    fig_loss()
    fig_metrics()
    fig_calibration()
    print("[done] figures written to", FIG)
