#!/usr/bin/env python3
"""
build_structures.py - Pure-Python porous Si structure generator
================================================================
Generates LAMMPS .data files for all 18 Paper 3 geometries.
No LAMMPS, no MPI, no module loads required.

Mirrors in.nemd_porous_parametric exactly:
- 8 x 8 x 48 unit cell diamond cubic Si box (a = 5.431 A)
- 6 Pore-A z-slices: z = 0-4, 8-12, 16-20, 24-28, 32-36, 40-44
- 6 Pore-B z-slices: z = 4-8, 12-16, 20-24, 28-32, 36-40, 44-48
- LAMMPS-style INCLUSIVE box boundaries (matches actual Phase 2 atom counts)

Usage on MARWAN (login node is fine):
    python3 build_structures.py
Output: ./structures_for_ovito/structure_<label>.data (override with STRUCTURES_DIR)
        18 files, OVITO opens them directly.
"""

import os
import numpy as np

A = 5.431          # Si lattice constant (A)
NX, NY, NZ = 8, 8, 48

# Diamond cubic basis - 8 atoms per unit cell, fractional coords
BASIS = np.array([
    [0.0,  0.0,  0.0 ],
    [0.0,  0.5,  0.5 ],
    [0.5,  0.0,  0.5 ],
    [0.5,  0.5,  0.0 ],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
    [0.75, 0.75, 0.25],
])

# Pore z-slice ranges (lattice units) - exactly as in in.nemd_porous_parametric
PORE_A_Z_SLICES = [(0, 4), (8, 12), (16, 20), (24, 28), (32, 36), (40, 44)]
PORE_B_Z_SLICES = [(4, 8), (12, 16), (20, 24), (28, 32), (36, 40), (44, 48)]

# 18 geometries (same as build_all_structures.sh, from Phase 2 CSVs)
#   (label, pAxlo, pAxhi, pAylo, pAyhi, pBxlo, pBxhi, pBylo, pByhi)
GEOMETRIES = [
    ("bulk_si",            0,    0,    0,    0,    0,    0,    0,    0  ),
    ("P1_low_50pct",       1,    4,    1,    4,    4,    7,    4,    7  ),
    ("P1_high_50pct",      0,    4,    0,    4,    4,    8,    4,    8  ),
    ("A_S2_d2",            0,    2,    0,    2,    2,    4,    2,    4  ),
    ("A_S2.5_d2",          0,    2.5,  0,    2.5,  2,    4.5,  2,    4.5),
    ("A_S3_d2",            0,    3,    0,    3,    2,    5,    2,    5  ),
    ("A_S4.5_d2",          0,    4.5,  0,    4.5,  2,    6.5,  2,    6.5),
    ("B_S4_aligned_low",   0,    2,    0,    2,    0,    2,    0,    2  ),
    ("B_S4_quarter_low",   0,    2,    0,    2,    4,    6,    4,    6  ),
    ("B_S4_aligned_high",  0,    4,    0,    4,    0,    4,    0,    4  ),
    ("B_S4_quarter_high",  0,    4,    0,    4,    2,    6,    2,    6  ),
    ("C_d0_aligned",       0,    3,    0,    3,    0,    3,    0,    3  ),
    ("C_d1_12pct",         0,    3,    0,    3,    1,    4,    1,    4  ),
    ("C_d3_37pct",         0,    3,    0,    3,    3,    6,    3,    6  ),
    ("C_d4_50pct",         0,    3,    0,    3,    4,    7,    4,    7  ),
    ("D_AR0.5_tall",       0,    2,    0,    4,    2,    4,    2,    6  ),
    ("D_AR2.0_wide",       0,    4,    0,    2,    2,    6,    2,    4  ),
    ("D_AR2.5_vwide",      0,    5,    0,    2,    2,    7,    2,    4  ),
]

# Phase 2 reference porosities (from ml_dataset_paper2.csv) for QC
PHASE2_PHI = {
    "bulk_si":           0.00,
    "P1_low_50pct":      6.84,
    "P1_high_50pct":    28.27,
    "A_S2_d2":           8.54,
    "A_S2.5_d2":        12.63,
    "A_S3_d2":          17.50,
    "A_S4.5_d2":        36.78,
    "B_S4_aligned_low":  6.25,
    "B_S4_quarter_low":  7.29,
    "B_S4_aligned_high":28.32,
    "B_S4_quarter_high":29.57,
    "C_d0_aligned":     17.50,
    "C_d1_12pct":       17.50,
    "C_d3_37pct":       17.50,
    "C_d4_50pct":       17.50,
    "D_AR0.5_tall":     17.50,
    "D_AR2.0_wide":     17.50,
    "D_AR2.5_vwide":    17.50,
}


def build_bulk_lattice():
    """Diamond cubic Si: NX x NY x NZ unit cells, 8 atoms/cell."""
    ix, iy, iz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
    cells = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)  # (N_cells, 3)
    atoms = (cells[:, None, :] + BASIS[None, :, :]).reshape(-1, 3) * A
    return atoms  # (24576, 3) for 8x8x48


def in_pore_inclusive(atoms, xlo, xhi, ylo, yhi, zlo, zhi, tol=1e-9):
    """LAMMPS-style inclusive boundary box region. Bounds in lattice units."""
    x, y, z = atoms[:, 0], atoms[:, 1], atoms[:, 2]
    return ((x >= xlo*A - tol) & (x <= xhi*A + tol) &
            (y >= ylo*A - tol) & (y <= yhi*A + tol) &
            (z >= zlo*A - tol) & (z <= zhi*A + tol))


def carve_pores(bulk_atoms, pAxlo, pAxhi, pAylo, pAyhi, pBxlo, pBxhi, pBylo, pByhi):
    """Apply 6 A-slices + 6 B-slices. Returns remaining atoms."""
    keep = np.ones(len(bulk_atoms), dtype=bool)
    if pAxhi > pAxlo:
        for zlo, zhi in PORE_A_Z_SLICES:
            keep &= ~in_pore_inclusive(bulk_atoms, pAxlo, pAxhi, pAylo, pAyhi, zlo, zhi)
    if pBxhi > pBxlo:
        for zlo, zhi in PORE_B_Z_SLICES:
            keep &= ~in_pore_inclusive(bulk_atoms, pBxlo, pBxhi, pBylo, pByhi, zlo, zhi)
    return bulk_atoms[keep]


def write_lammps_data(atoms, filepath, label):
    """Write LAMMPS data file (atom_style atomic). OVITO reads this directly."""
    Lx, Ly, Lz = NX*A, NY*A, NZ*A
    n = len(atoms)
    with open(filepath, 'w') as f:
        f.write(f"LAMMPS data file - {label} - generated by build_structures.py\n\n")
        f.write(f"{n} atoms\n")
        f.write("1 atom types\n\n")
        f.write(f"0.0 {Lx:.6f} xlo xhi\n")
        f.write(f"0.0 {Ly:.6f} ylo yhi\n")
        f.write(f"0.0 {Lz:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 28.0855\n\n")
        f.write("Atoms  # atomic\n\n")
        for i, (x, y, z) in enumerate(atoms, 1):
            f.write(f"{i} 1 {x:.6f} {y:.6f} {z:.6f}\n")


def main():
    # Output directory: override with STRUCTURES_DIR, else ./structures_for_ovito
    out_dir = os.path.expanduser(
        os.environ.get("STRUCTURES_DIR", "structures_for_ovito"))
    os.makedirs(out_dir, exist_ok=True)

    print("Building 18 structures (pure Python, no LAMMPS)...")
    print("=" * 78)

    bulk = build_bulk_lattice()
    n_bulk = len(bulk)
    assert n_bulk == NX * NY * NZ * 8 == 24576, f"Bulk count wrong: {n_bulk}"
    print(f"Bulk lattice: {n_bulk} atoms (expected 24576)\n")

    header = f"  {'Label':<22} {'Atoms':>7} {'phi (%)':>8} {'Phase 2 phi':>12} {'match':>7}"
    print(header)
    print("-" * 78)

    all_match = True
    for label, pAxlo, pAxhi, pAylo, pAyhi, pBxlo, pBxhi, pBylo, pByhi in GEOMETRIES:
        atoms = carve_pores(bulk, pAxlo, pAxhi, pAylo, pAyhi,
                                  pBxlo, pBxhi, pBylo, pByhi)
        n = len(atoms)
        phi = 100.0 * (n_bulk - n) / n_bulk
        phi_ref = PHASE2_PHI.get(label, np.nan)
        diff = abs(phi - phi_ref)
        ok = "OK" if diff < 0.1 else f"OFF {diff:.2f}"
        if diff >= 0.1:
            all_match = False

        write_lammps_data(atoms, os.path.join(out_dir, f"structure_{label}.data"), label)
        print(f"  {label:<22} {n:>7} {phi:>8.2f} {phi_ref:>12.2f} {ok:>7}")

    print("=" * 78)
    print(f"Output: {out_dir}/")
    print(f"Total files: {len(GEOMETRIES)}")
    if all_match:
        print("ALL 18 POROSITIES MATCH PHASE 2 - GREEN LIGHT")
    else:
        print("[!] Some porosities differ from Phase 2 - check geometry definitions")
    print()
    print("Then drag any .data file into OVITO.")


if __name__ == "__main__":
    main()
