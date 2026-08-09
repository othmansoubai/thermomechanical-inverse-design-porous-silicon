#!/usr/bin/env python3
"""
make_inv1_planview.py - Supplementary Figure S1 for Paper 3
============================================================
x-y plan view (looking down z) of INV1's two pore footprints.

Shows why the neck convention is ill-defined for INV1: pore A spans
x[0,5] and pore B spans x[3,8], so their union covers the full 8-cell
box width. No unblocked strip of solid remains along x, and the rule
"box width minus pore B's far edge" returns 8 - 8 = 0.

Run on MARWAN:
    python3 make_inv1_planview.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = 5.431                      # lattice constant, Angstrom
NX = NY = 8                    # box is 8 x 8 unit cells in x and y
FIGURES_DIR = os.path.expanduser(
    os.environ.get("FIGURES_DIR", "figures"))

# INV1 pore footprints, in lattice cells
PORE_A = dict(x0=0, x1=5, y0=0, y1=2, label="Pore A", color="#C8792E")
PORE_B = dict(x0=3, x1=8, y0=3, y1=5, label="Pore B", color="#2E7D8C")

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                     "savefig.bbox": "tight", "figure.dpi": 120})

fig, ax = plt.subplots(figsize=(6.2, 5.6))

# simulation cell
ax.add_patch(Rectangle((0, 0), NX*A, NY*A, fill=False,
                       edgecolor="black", linewidth=2, zorder=5))

# pore footprints
for p in (PORE_A, PORE_B):
    ax.add_patch(Rectangle((p["x0"]*A, p["y0"]*A),
                           (p["x1"]-p["x0"])*A, (p["y1"]-p["y0"])*A,
                           facecolor=p["color"], alpha=0.55,
                           edgecolor=p["color"], linewidth=1.8, zorder=3))
    ax.text((p["x0"]+p["x1"])/2*A, (p["y0"]+p["y1"])/2*A,
            f'{p["label"]}\nx[{p["x0"]},{p["x1"]}]',
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", zorder=4)

# label the uninterrupted solid columns
for cx, cy, w, h in [(5, 0, 3, 2), (0, 3, 3, 2)]:
    ax.text((cx + w/2)*A, (cy + h/2)*A, "solid\ncolumn",
            ha="center", va="center", fontsize=8.5, color="#5a5a52",
            style="italic", zorder=4)

# guide lines at the pore x-edges
for xv in (3, 5):
    ax.plot([xv*A, xv*A], [0, NY*A], ":", color="#7a7a72", lw=1.2, zorder=2)

ax.set_xlim(-1.5, NX*A + 1.5)
ax.set_ylim(-1.5, NY*A + 1.5)
ax.set_aspect("equal")
ax.set_xlabel(r"$x$ ($\mathrm{\AA}$)")
ax.set_ylabel(r"$y$ ($\mathrm{\AA}$)")
ax.set_xticks([i*A for i in range(0, NX+1, 2)])
ax.set_yticks([i*A for i in range(0, NY+1, 2)])
ax.set_xticklabels([f"{i*A:.0f}" for i in range(0, NX+1, 2)])
ax.set_yticklabels([f"{i*A:.0f}" for i in range(0, NY+1, 2)])
ax.grid(alpha=0.25, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

os.makedirs(FIGURES_DIR, exist_ok=True)
for ext in ("pdf", "png"):
    out = os.path.join(FIGURES_DIR, f"figS1_inv1_planview.{ext}")
    plt.savefig(out, dpi=300 if ext == "png" else None, facecolor="white")
    print(f"wrote {out}")
plt.close(fig)
