#!/usr/bin/env python3
"""
render_structures.py - Publication-quality figures of porous Si structures
============================================================================
Reads LAMMPS .data files and renders 3D + 2D views with matplotlib.
Fully reproducible (same code = same figure every time).

Usage on MARWAN:
    python3 render_structures.py
Then download the PNG files to your laptop:

Outputs:
    figures/fig1_panel_<label>.png     - 3D perspective views
    figures/fig1_4panel_composite.png  - 2x2 composite for Paper 3 Fig 1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---- Configuration ---------------------------------------------------------
STRUCTURES_DIR = os.path.expanduser(
    os.environ.get("STRUCTURES_DIR", "structures_for_ovito"))
FIGURES_DIR    = os.path.expanduser(
    os.environ.get("FIGURES_DIR", "figures"))

# Structures to render (label, panel_title, panel_letter)
# Paper 3 Fig 1: design-logic progression at ~constant porosity (phi 16-20%),
# walking the reader through the two orthogonal knobs and their combination.
PANELS = [
    ("C_d0_aligned",    "Aligned pores (\u03c6=16.6%)",                 "(a)"),
    ("C_d3_37pct",      "+ Stagger 37.5%  \u2014 thermal knob (\u03c6=17.7%)", "(b)"),
    ("D_AR2.5_vwide",   "+ Aspect ratio 2.5  \u2014 mechanical knob (\u03c6=19.6%)", "(c)"),
    ("INV1_AR2.5_s37",  "INV1: stagger + AR combined (\u03c6=19.3%)",   "(d)"),
]

ATOM_COLOR  = "#D4A574"   # Sand color
ATOM_EDGE   = "#7D5A33"   # Darker outline
CELL_COLOR  = "black"
ATOM_SIZE   = 3           # Marker size for 3D scatter

# ---- LAMMPS data file reader ------------------------------------------------
def read_lammps_data(filepath):
    """Parse LAMMPS atomic-style .data file. Returns (atoms, box) where
    atoms is (N, 3) array of positions and box is (xlo, xhi, ylo, yhi, zlo, zhi)."""
    with open(filepath) as f:
        lines = f.readlines()

    box = [0, 0, 0, 0, 0, 0]
    in_atoms = False
    atoms = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "xlo xhi" in line:
            parts = line.split()
            box[0], box[1] = float(parts[0]), float(parts[1])
        elif "ylo yhi" in line:
            parts = line.split()
            box[2], box[3] = float(parts[0]), float(parts[1])
        elif "zlo zhi" in line:
            parts = line.split()
            box[4], box[5] = float(parts[0]), float(parts[1])
        elif line.startswith("Atoms"):
            in_atoms = True
            continue
        elif line.startswith("Masses") or line.startswith("Velocities"):
            in_atoms = False
        elif in_atoms:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    atoms.append([float(parts[2]), float(parts[3]), float(parts[4])])
                except ValueError:
                    pass

    return np.array(atoms), box

# ---- Plot a 3D box (simulation cell) ----------------------------------------
def draw_box_3d(ax, box, color="black", lw=1.5):
    xlo, xhi, ylo, yhi, zlo, zhi = box
    # 8 corners
    corners = np.array([
        [xlo, ylo, zlo], [xhi, ylo, zlo], [xhi, yhi, zlo], [xlo, yhi, zlo],
        [xlo, ylo, zhi], [xhi, ylo, zhi], [xhi, yhi, zhi], [xlo, yhi, zhi],
    ])
    # 12 edges
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color=color, linewidth=lw)

# ---- Single 3D panel --------------------------------------------------------
def render_3d_panel(ax, atoms, box, title, letter, view_elev=22, view_azim=-72, zcrop=0.42):
    """Render one 3D structure panel with fixed view angle."""
    # Subsample atoms for speed (3D scatter is slow with 24k points)
    # Show every 3rd atom - still looks dense, ~10x faster
    if zcrop:
        zlo0, zhi0 = box[4], box[5]
        zcut = zlo0 + zcrop*(zhi0-zlo0)
        atoms = atoms[atoms[:,2] <= zcut]
        box = list(box); box[5] = zcut
    if len(atoms) > 5000:
        atoms_show = atoms[::2]
    else:
        atoms_show = atoms

    ax.scatter(atoms_show[:, 0], atoms_show[:, 1], atoms_show[:, 2],
               c=ATOM_COLOR, edgecolors=ATOM_EDGE, linewidths=0.2,
               s=ATOM_SIZE, alpha=0.85, depthshade=True)

    draw_box_3d(ax, box, color=CELL_COLOR, lw=1.5)

    # Equal aspect ratio
    xlo, xhi, ylo, yhi, zlo, zhi = box
    ax.set_box_aspect([(xhi-xlo), (yhi-ylo), (zhi-zlo)])

    # Fixed viewing angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Clean appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)

    # Title above
    ax.set_title(f"{letter} {title}", fontsize=11, pad=10, fontweight='bold')

# ---- Single 2D slice panel --------------------------------------------------
def render_2d_slice(ax, atoms, box, title, letter, slice_axis='y', slice_pos=None, slice_thickness=2.5):
    """2D projection: show atoms in a slab perpendicular to slice_axis."""
    if slice_pos is None:
        # Middle of box
        if slice_axis == 'x':
            slice_pos = 0.5*(box[0]+box[1])
        elif slice_axis == 'y':
            slice_pos = 0.5*(box[2]+box[3])
        else:
            slice_pos = 0.5*(box[4]+box[5])

    if slice_axis == 'y':
        mask = np.abs(atoms[:,1] - slice_pos) < slice_thickness
        x_data, y_data = atoms[mask, 0], atoms[mask, 2]
        x_lo, x_hi = box[0], box[1]
        y_lo, y_hi = box[4], box[5]
        ax.set_xlabel("x (\u00c5)", fontsize=10)
        ax.set_ylabel("z (\u00c5)", fontsize=10)
    elif slice_axis == 'x':
        mask = np.abs(atoms[:,0] - slice_pos) < slice_thickness
        x_data, y_data = atoms[mask, 1], atoms[mask, 2]
        x_lo, x_hi = box[2], box[3]
        y_lo, y_hi = box[4], box[5]
        ax.set_xlabel("y (\u00c5)", fontsize=10)
        ax.set_ylabel("z (\u00c5)", fontsize=10)
    else:  # z
        mask = np.abs(atoms[:,2] - slice_pos) < slice_thickness
        x_data, y_data = atoms[mask, 0], atoms[mask, 1]
        x_lo, x_hi = box[0], box[1]
        y_lo, y_hi = box[2], box[3]
        ax.set_xlabel("x (\u00c5)", fontsize=10)
        ax.set_ylabel("y (\u00c5)", fontsize=10)

    ax.scatter(x_data, y_data, c=ATOM_COLOR, edgecolors=ATOM_EDGE, linewidths=0.3,
               s=18, alpha=0.9)
    ax.add_patch(Rectangle((x_lo, y_lo), x_hi-x_lo, y_hi-y_lo,
                           fill=False, edgecolor=CELL_COLOR, linewidth=1.5))
    ax.set_xlim(x_lo - 2, x_hi + 2)
    ax.set_ylim(y_lo - 2, y_hi + 2)
    ax.set_aspect('equal')
    ax.set_title(f"{letter} {title}", fontsize=11, fontweight='bold')

# ---- Main -------------------------------------------------------------------
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # === Individual 3D panels ===
    print("Rendering individual 3D panels...")
    for label, title, letter in PANELS:
        datafile = os.path.join(STRUCTURES_DIR, f"structure_{label}.data")
        if not os.path.exists(datafile):
            print(f"  SKIP: {datafile} not found")
            continue

        atoms, box = read_lammps_data(datafile)

        fig = plt.figure(figsize=(5, 7))
        ax = fig.add_subplot(111, projection='3d')
        render_3d_panel(ax, atoms, box, title, letter)

        outpath = os.path.join(FIGURES_DIR, f"fig1_panel_{label}.png")
        plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  OK: {outpath}  ({len(atoms)} atoms)")

    # === 2x2 composite ===
    print("\nRendering 2x2 composite (Paper 3 Fig 1 candidate)...")
    fig, axes = plt.subplots(1, 4, figsize=(13, 8))
    for i, (label, title, letter) in enumerate(PANELS):
        datafile = os.path.join(STRUCTURES_DIR, f"structure_{label}.data")
        if not os.path.exists(datafile):
            continue
        atoms, box = read_lammps_data(datafile)
        # slice through Pore A (y ~ 5 A) -> pores appear as clear voids
        short = {0:"Aligned", 1:"Stagger 37.5%", 2:"AR 2.5", 3:"INV1"}[i]
        render_2d_slice(axes[i], atoms, box, short, letter,
                        slice_axis='y', slice_pos=5.0, slice_thickness=1.5)
        axes[i].set_title(f"{letter} {short}", fontsize=11, fontweight='bold')
        if i:
            axes[i].set_ylabel("")
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_4panel_slices.png"),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_4panel_slices.pdf"),
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  OK: fig1_4panel_slices.png (+ .pdf)  <-- 2D slice version")

    fig = plt.figure(figsize=(12, 14))
    for i, (label, title, letter) in enumerate(PANELS):
        datafile = os.path.join(STRUCTURES_DIR, f"structure_{label}.data")
        if not os.path.exists(datafile):
            continue
        atoms, box = read_lammps_data(datafile)
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        render_3d_panel(ax, atoms, box, title, letter)

    # No embedded suptitle: the manuscript figure caption provides the title.
    # (Uncomment if you want a title baked into the PNG for slides/preview.)
    # fig.suptitle("Porous Si architecture: two design knobs and their combination",
    #              fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "fig1_4panel_composite.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_4panel_composite.pdf"),
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  OK: {outpath}  (+ .pdf)")

    # === 2D slice through INV1 (shows the designed pore layering along z) ===
    print("\nRendering 2D x-z slice of INV1 (designed structure)...")
    datafile = os.path.join(STRUCTURES_DIR, "structure_INV1_AR2.5_s37.data")
    if os.path.exists(datafile):
        atoms, box = read_lammps_data(datafile)
        fig, ax = plt.subplots(figsize=(6, 10))
        # Slice through y within Pore-A (y in [0,2]*a = [0, 10.9] A) -> shows the
        # 6 pore-A z-slices as gaps along the heat-flow (z) direction.
        render_2d_slice(ax, atoms, box,
                        "INV1 \u2014 x\u2013z slice through Pore A (y \u2248 5 \u00c5)", "",
                        slice_axis='y', slice_pos=5.0, slice_thickness=1.5)
        outpath = os.path.join(FIGURES_DIR, "fig1_slice_INV1.png")
        plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(FIGURES_DIR, "fig1_slice_INV1.pdf"),
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  OK: {outpath}")

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Download to your laptop with:")


if __name__ == "__main__":
    main()
