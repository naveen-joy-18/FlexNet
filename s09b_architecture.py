"""
FlexNet architecture figure (publication quality, visually polished).

Left-to-right pipeline: per-nucleotide feature families -> convolutional stem ->
six dilated residual blocks -> bidirectional GRU -> pointwise flexibility head,
with an auxiliary (gradient-detached) branch feeding a continuous-reactivity
regression head and a region-level classification head.

Styling: gradient boxes, soft drop shadows, stage banners, colour-coded feature
families and an accent-highlighted output. Vector text (editable in Illustrator).

Outputs (in figures/):
  fig_architecture.svg   editable vector
  fig_architecture.png   600 dpi raster

Run:  python scripts/s09b_architecture.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "svg.fonttype": "none"})

INK = "#16233A"
GREY = "#5A6B82"

# (light, dark, edge)
MAIN = ("#ECF2FC", "#AFC8EE", "#2E4E86")
ACCENT = ("#FFF3D6", "#FBC766", "#D9821B")
AUX = ("#FFF4EA", "#F6CDA6", "#B96D2E")
FAM = {
    "seq": ("#E4EEFD", "#A9CAF2", "#3C6FC0"),
    "str": ("#DEF5EC", "#93D9C4", "#1F8A70"),
    "thr": ("#FFEFD9", "#F6C888", "#D67E27"),
    "ens": ("#E7F5D8", "#B4DE8D", "#4E9631"),
}
BAND = ("#26325A", "#4E74C4")

fig, ax = plt.subplots(figsize=(14.1, 5.1))
ax.set_xlim(-0.5, 140.5)
ax.set_ylim(-0.4, 50.6)
ax.axis("off")

CHECKS = []


def _reg(t, rect):
    CHECKS.append((t, rect))
    return t


def _rbox(x, y, w, h, fc, ec, lw=1.1, rounding=0.9, ls="solid", z=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def _shadow(x, y, w, h, rounding=0.9, off=0.34):
    _rbox(x + off, y - off, w, h, "#22314F", "none", rounding=rounding, z=1)
    ax.patches[-1].set_alpha(0.16)


def gradbox(x, y, w, h, pair, edge, rounding=0.9, lw=1.2, shadow=True):
    if shadow:
        _shadow(x, y, w, h, rounding)
    p = _rbox(x, y, w, h, "none", edge, lw=lw, rounding=rounding, z=3)
    cmap = LinearSegmentedColormap.from_list("", [pair[0], pair[1]])
    arr = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(arr, extent=[x, x + w, y, y + h], origin="lower",
                   cmap=cmap, aspect="auto", zorder=2,
                   interpolation="bilinear")
    im.set_clip_path(p)
    return p


def node(x, y, w, h, title, body, pair, title_fs=8.2, body_fs=6.6):
    gradbox(x, y, w, h, pair, pair[2], lw=1.3)
    ax.plot([x + 1.0, x + w - 1.0], [y + h - 4.1, y + h - 4.1],
            color=pair[2], lw=0.9, alpha=0.6, zorder=4)
    cx = x + w / 2
    if title:
        t = ax.text(cx, y + h - 2.4, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=INK, zorder=4)
        _reg(t, (x, y, w, h))
    if body:
        t = ax.text(cx, y + (h - 4.6) / 2 + 0.4, body, ha="center", va="center",
                    fontsize=body_fs, color=INK, zorder=4)
        _reg(t, (x, y, w, h))


def leaf(x, y, w, h, text, pair, fs=7.2, lw=1.3):
    gradbox(x, y, w, h, pair, pair[2], lw=lw)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=4)
    _reg(t, (x, y, w, h))


def pill(cx, y, text, color, w, fs=7.0):
    h = 2.7
    _rbox(cx - w / 2, y, w, h, color, "none", rounding=1.3, z=6)
    ax.patches[-1].set_alpha(0.96)
    t = ax.text(cx, y + h / 2 + 0.05, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=7)
    _reg(t, (cx - w / 2, y, w, h))


def arrow(x0, y0, x1, y1, color="#42597F", lw=1.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=11, linewidth=lw, color=color, shrinkA=0,
                 shrinkB=0, zorder=5))


def dline(x0, y0, x1, y1, color="#A9702F", lw=1.1):
    ax.plot([x0, x1], [y0, y1], linestyle=(0, (2.5, 2)), color=color, lw=lw,
            zorder=4)


# =============================================================== title band
band = gradbox(1.0, 45.2, 138.0, 4.2, BAND, "#243256", rounding=1.1, lw=0,
               shadow=False)
ax.text(-10, -10, "", alpha=0)  # keep band gradient clean
_rbox(2.5, 46.0, 2.7, 2.7, "#FFFFFF", "none", rounding=0.7, z=6)
ax.text(3.85, 47.35, "Fx", ha="center", va="center", fontsize=8.5,
        fontweight="bold", color=BAND[0], zorder=7)
ax.text(6.4, 47.95, "FlexNet", ha="left", va="center", fontsize=16,
        fontweight="bold", color="white", zorder=7)
ax.text(6.7, 46.05,
        "Nucleotide-resolution prediction of RNA flexible regions",
        ha="left", va="center", fontsize=7.2, color="#D3E0F8", zorder=7)

# RNA double-helix motif on the right of the title band
_hx = np.linspace(100.0, 137.0, 260)
_hy = 47.35
_hamp = 1.05
_k = 2 * np.pi * 3.0 / 37.0
_s1 = np.sin(_k * (_hx - 100.0))
ax.plot(_hx, _hy + _hamp * _s1, color="#FFFFFF", lw=1.3, alpha=0.55, zorder=7,
        solid_capstyle="round")
ax.plot(_hx, _hy - _hamp * _s1, color="#9FBEF2", lw=1.3, alpha=0.6, zorder=7,
        solid_capstyle="round")
for _xx in np.linspace(101.0, 136.0, 22):
    _s = np.sin(_k * (_xx - 100.0))
    ax.plot([_xx, _xx], [_hy - _hamp * _s, _hy + _hamp * _s], color="#FFFFFF",
            lw=0.6, alpha=0.32, zorder=6)

# =============================================================== stage pills
pill(13.0, 41.5, "1   INPUT FEATURES", "#4A78C8", 23.4)
pill(47.5, 41.5, "2   CONVOLUTIONAL ENCODER", "#3B5E9E", 38.0)
pill(86.0, 41.5, "3   FUSION & POINTWISE HEAD", "#356F63", 34.0)
pill(122.3, 41.5, "4   FLEXIBLE-REGION CALLING", "#D08A1E", 33.4)

# =============================================================== input panel
gradbox(0.8, 23.6, 24.6, 17.0, ("#FFFFFF", "#F3F7FE"), "#9FB0CC",
        rounding=1.3, lw=1.2, shadow=False)
ax.text(13.1, 39.2, "per-nucleotide features   (L \u00d7 66)", ha="center",
        va="center", fontsize=7.0, color=GREY, style="italic", zorder=4)
rows = [("Sequence  \u00b7  19", "seq"), ("Secondary structure  \u00b7  28", "str"),
        ("Thermodynamics  \u00b7  7", "thr"),
        ("Ensemble accessibility  \u00b7  12", "ens")]
for i, (lbl, key) in enumerate(rows):
    yy = 35.4 - i * 3.9
    pair = FAM[key]
    gradbox(1.9, yy, 22.4, 3.3, pair, pair[2], rounding=0.7, lw=0.9,
            shadow=False)
    _rbox(2.35, yy + 0.7, 1.9, 1.9, pair[2], "none", rounding=0.5, z=5)
    t = ax.text(5.1, yy + 1.65, lbl, ha="left", va="center", fontsize=7.0,
                color=INK, zorder=4)
    _reg(t, (1.9, yy, 22.4, 3.3))

# =============================================================== trunk
node(27.0, 25.0, 12.0, 15.5, "Conv stem",
     "Conv1d  k = 1\n66 \u2192 128\nGroupNorm \u00b7 GELU", MAIN)

# dilated residual container
gradbox(40.8, 25.4, 26.4, 15.1, ("#FBFCFE", "#EEF3FB"), "#93A8C8",
        rounding=1.1, lw=1.1, shadow=False)
ax.text(54.0, 39.4, "Dilated residual blocks  (\u00d76)", ha="center",
        va="center", fontsize=8.2, fontweight="bold", color=INK, zorder=4)
ax.text(54.0, 37.6, "GN \u2192 Conv1d k=5 (dilated) \u2192 GLU \u2192 + residual",
        ha="center", va="center", fontsize=6.0, color=GREY, style="italic",
        zorder=4)
for i, d in enumerate([1, 2, 4, 8, 16, 32]):
    r, c = divmod(i, 3)
    gradbox(41.8 + c * 8.1, 28.4 + (1 - r) * 4.8, 7.4, 3.6, MAIN, MAIN[2],
            rounding=0.7, lw=0.9, shadow=False)
    ax.text(41.8 + c * 8.1 + 3.7, 28.4 + (1 - r) * 4.8 + 1.8, f"d = {d}",
            ha="center", va="center", fontsize=7.4, color=INK, zorder=4)

node(69.0, 25.0, 15.0, 15.5, "BiGRU",
     "64 units / direction\nbidirectional\n(L \u00d7 128)", MAIN, body_fs=6.4)

# merge node
_shadow(85.65, 31.9, 1.5, 1.5, rounding=3.0)
ax.add_patch(Circle((86.4, 32.75), 0.78, facecolor="#FFFFFF",
                    edgecolor="#3B5E9E", lw=1.5, zorder=5))

node(88.0, 25.0, 15.0, 15.5, "Pointwise head",
     "input (L \u00d7 256)\n2 \u00d7 [Conv1d 64\n+ GELU]\nConv1d \u2192 1\nsigmoid",
     MAIN, body_fs=6.3)

leaf(105.0, 27.6, 14.5, 10.3, "Per-nucleotide\nflexibility score\n(L)", MAIN)

# accent-highlighted output
_rbox(120.4, 26.4, 19.8, 12.7, ACCENT[2], "none", rounding=1.6, z=1)
ax.patches[-1].set_alpha(0.22)
leaf(121.5, 27.6, 17.5, 10.3, "Flexible-region\ncaller\n(T, L\u2098\u1d62\u2099)",
     ACCENT)

# arrows
yb = 32.75
for x0, x1 in [(25.4, 27.0), (39.0, 40.8), (67.2, 69.0), (84.0, 85.6),
               (87.2, 88.0), (103.0, 105.0), (119.5, 121.5)]:
    arrow(x0, yb, x1, yb)

# =============================================================== aux branch
gradbox(40.8, 4.0, 98.0, 18.6, ("#FFFCF9", "#FDF0E6"), "#D8A882",
        rounding=1.2, lw=1.2, shadow=False)
ax.text(100.0, 20.9, "AUXILIARY BRANCH  \u00b7  GRADIENT DETACHED", ha="left",
        va="center", fontsize=7.0, fontweight="bold", color="#B4661F",
        zorder=4)

node(44.0, 8.0, 22.0, 8.0, "Aux stem",
     "Conv1d  k = 1,  256 \u2192 64\nGroupNorm \u00b7 GELU", AUX, body_fs=6.4)
leaf(69.0, 8.0, 19.0, 8.0, "2\u00d7 dilated residual\n(d = 1, 2)", AUX, fs=7.0)
leaf(90.0, 11.8, 46.0, 5.6,
     "Regression head \u2192 continuous reactivity   (smooth L1,  w = 0.3)",
     AUX, fs=6.8)
leaf(90.0, 5.2, 46.0, 5.6,
     "Region head \u2192 region labels   (BCE,  w = 0.5)", AUX, fs=6.8)

arrow(66.0, 12.0, 69.0, 12.0, color="#C4783A", lw=1.3)
arrow(88.0, 12.0, 90.0, 14.6, color="#C4783A", lw=1.3)
arrow(88.0, 12.0, 90.0, 8.0, color="#C4783A", lw=1.3)

dline(86.4, 31.97, 86.4, 20.0)
dline(86.4, 20.0, 55.0, 20.0)
dline(55.0, 20.0, 55.0, 16.0)
ax.text(70.0, 20.4, "detach  (stop-gradient)", ha="center", va="bottom",
        fontsize=6.6, color="#B4661F", style="italic", zorder=4)

# =============================================================== loss note
ax.text(1.8, 2.1,
        r"Total loss  $\mathcal{L} = 1.0\,\mathcal{L}_{point} + 0.3\,"
        r"\mathcal{L}_{reg} + 0.5\,\mathcal{L}_{region}$   "
        r"($\mathcal{L}_{point}$: focal BCE)",
        ha="left", va="center", fontsize=7.4, color=GREY, zorder=4)

# =============================================================== finalise
ax.set_xlim(-0.5, 140.5)
ax.set_ylim(-0.4, 50.6)
fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)

fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()
warn = 0
for t, (x, y, w, h) in CHECKS:
    bb = t.get_window_extent(rend)
    (x0, y0) = inv.transform((bb.x0, bb.y0))
    (x1, y1) = inv.transform((bb.x1, bb.y1))
    xlo, xhi = min(x0, x1), max(x0, x1)
    ylo, yhi = min(y0, y1), max(y0, y1)
    if xlo < x - 0.3 or xhi > x + w + 0.3 or ylo < y - 0.3 or yhi > y + h + 0.3:
        warn += 1
        print(f"  OVERFLOW {t.get_text()!r}: box=({x:.1f},{y:.1f},{w:.1f},{h:.1f})"
              f" text=({xlo:.1f},{ylo:.1f},{xhi:.1f},{yhi:.1f})")
print("fit warnings:", warn)

exts = []
for t in ax.texts:
    if not t.get_text().strip():
        continue
    bb = t.get_window_extent(rend)
    (x0, y0) = inv.transform((bb.x0, bb.y0))
    (x1, y1) = inv.transform((bb.x1, bb.y1))
    exts.append((t.get_text(), min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
coll = 0
for i in range(len(exts)):
    for j in range(i + 1, len(exts)):
        _, ax0, ay0, ax1, ay1 = exts[i]
        _, bx0, by0, bx1, by1 = exts[j]
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        if ix > 0 and iy > 0:
            ov = ix * iy
            amin = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
            if amin > 0 and ov / amin > 0.04:
                coll += 1
                print("  TEXT OVERLAP:", repr(exts[i][0]), "<->", repr(exts[j][0]))
print("text-overlap collisions:", coll)

svg = os.path.join(OUT, "fig_architecture.svg")
png = os.path.join(OUT, "fig_architecture.png")
fig.savefig(svg, format="svg")
fig.savefig(png, dpi=600, facecolor="white")
plt.close(fig)
print("written", svg)
print("written", png)
