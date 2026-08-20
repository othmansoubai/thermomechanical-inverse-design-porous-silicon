#!/usr/bin/env python3
"""
make_fig1.py — Figure 1 as a two-row panel.

Top row    : x–z cross-sections through pore A (the thermal story: a continuous
             solid channel in the aligned case, interrupted once the second pore
             channel is offset).
Bottom row : x–y plan views of both pore footprints (the geometric story: the
             footprint shape and the offset between the two channels).

The second row exists because a single x–z slice cannot distinguish
D_AR2.5_vwide from INV1: both share the pore-A footprint x[0,5] y[0,2], and
their pore-B channels lie outside the slice plane, so their top panels
coincide exactly. The plan view shows the offset that separates them.

Run from ~/phase3_mech (needs render_structures.py alongside):

    module purge && module load GCCcore/12.3.0 Python/3.11.3-GCCcore-12.3.0
    source ~/phase2_ml/analysis_env/bin/activate
    python3 make_fig1.py
    deactivate

Writes figures/fig1_4panel_slices.pdf and .png — the same filenames the
manuscript already uses, so no LaTeX change is needed beyond the caption.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_structures as rs          # reuse the reader and the slice renderer

A = 5.431                                # lattice parameter, Angstrom
NX = NY = 8                              # transverse cell size, unit cells
FIGURES_DIR = rs.FIGURES_DIR
STRUCT_DIR  = rs.STRUCTURES_DIR

# label, letter, pore A (x0,x1,y0,y1), pore B (x0,x1,y0,y1), porosity
PANELS = [
    ("C_d0_aligned",   "(a)", (0, 3, 0, 3), (0, 3, 0, 3), 16.6),
    ("C_d3_37pct",     "(b)", (0, 3, 0, 3), (3, 6, 3, 6), 17.7),
    ("D_AR2.5_vwide",  "(c)", (0, 5, 0, 2), (2, 7, 2, 4), 19.6),
    ("INV1_AR2.5_s37", "(d)", (0, 5, 0, 2), (3, 8, 3, 5), 19.3),
]

C_A = "#C8792E"     # pore A
C_B = "#2E7D8C"     # pore B

plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight",
                     "figure.dpi": 120})


def plan_view(ax, pA, pB):
    """Draw the two pore footprints in the x–y plane."""
    ax.add_patch(Rectangle((0, 0), NX * A, NY * A, fill=False,
                           edgecolor="black", lw=1.8, zorder=5))
    # When the offset is zero the two footprints coincide exactly; draw once.
    same = (tuple(pA) == tuple(pB))
    items = [(pA, C_A, "A = B" if same else "A")] if same else \
            [(pA, C_A, "A"), (pB, C_B, "B")]
    for p, c, lab in items:
        x0, x1, y0, y1 = p
        ax.add_patch(Rectangle((x0 * A, y0 * A), (x1 - x0) * A, (y1 - y0) * A,
                               facecolor=c, alpha=0.55, edgecolor=c,
                               lw=1.4, zorder=3))
        ax.text((x0 + x1) / 2 * A, (y0 + y1) / 2 * A, lab, ha="center",
                va="center", fontsize=9.5, fontweight="bold",
                color="white", zorder=4)
    ax.set_xlim(-1.5, NX * A + 1.5)
    ax.set_ylim(-1.5, NY * A + 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 2 * A, 4 * A, 6 * A, 8 * A])
    ax.set_yticks([0, 2 * A, 4 * A, 6 * A, 8 * A])
    ax.set_xticklabels(["0", "11", "22", "33", "43"])
    ax.set_yticklabels(["0", "11", "22", "33", "43"])
    ax.grid(alpha=0.22, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(13, 9.5),
                             gridspec_kw={"height_ratios": [2.05, 1.0]})

    missing = []
    for i, (label, letter, pA, pB, phi) in enumerate(PANELS):
        # ---- top: x–z slice through pore A -------------------------------
        f = os.path.join(STRUCT_DIR, f"structure_{label}.data")
        top = axes[0][i]
        if os.path.exists(f):
            atoms, box = rs.read_lammps_data(f)
            rs.render_2d_slice(top, atoms, box, "", "",
                               slice_axis="y", slice_pos=5.0,
                               slice_thickness=1.5)
        else:
            missing.append(label)
            top.text(0.5, 0.5, "missing", ha="center", va="center")
        top.set_title(letter, fontsize=12, fontweight="bold", loc="left")
        if i:
            top.set_ylabel("")

        # ---- bottom: x–y plan view ---------------------------------------
        bot = axes[1][i]
        plan_view(bot, pA, pB)
        bot.set_title(f"({chr(101+i)})", fontsize=12, fontweight="bold", loc="left")
        bot.set_xlabel(r"$x$ ($\mathrm{\AA}$)")
        if i == 0:
            bot.set_ylabel(r"$y$ ($\mathrm{\AA}$)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"fig1_4panel_slices.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)

    if missing:
        print("\nMISSING structure files (top row blank for these):")
        for m in missing:
            print(f"  structure_{m}.data")
        print("Build them with build_structures.py and make_inv1.py first.")
    else:
        print("\nAll four structures rendered.")
        print("Note: the top panels of (c) and (d) are identical by")
        print("construction — same pore-A footprint, pore B outside the slice.")
        print("The bottom row is what distinguishes them.")


if __name__ == "__main__":
    main()
