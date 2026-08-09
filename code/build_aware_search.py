#!/usr/bin/env python3
"""
build_aware_search.py — the second pass of the inverse design.

paper3_multiobj_ml.py ranks aspect-ratio candidates while holding porosity at
the parent tiling's value. That assumption is false on a discrete lattice:
a pore footprint is an integer number of unit cells, so changing the aspect
ratio changes the footprint area and drags porosity with it.

This script constructs each candidate, counts its atoms, re-predicts kappa and
E at the porosity the structure actually has, and re-ranks. The ranking
reverses: the candidate selected under the inherited porosity is not the one
selected under the built porosity.

Run from the repository root:

    python3 code/build_aware_search.py

Reproduces the candidate table of the manuscript (Table 3a).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_structures as bs

RANDOM_STATE = 42
FEATURES = ["phi", "S", "stagger", "neck_uc", "AR"]
TARGET_KAPPA, TARGET_E = 2.4, 70.0

DATA = os.path.expanduser(
    os.environ.get("DATA_DIR", "data")) + "/paper3_multiobj_dataset.csv"

# Candidates: aspect-ratio variants of the C_d3 tiling (37.5% stagger, S=3).
# Pore A footprint, pore B footprint, both as (x0, x1, y0, y1) in unit cells.
# The parent is 3x3 at offset 3; the variants keep the offset and reshape.
CANDIDATES = [
    ("AR = 2.0", "4x2", 2.0, (0, 4, 0, 2), (3, 7, 3, 5)),
    ("AR = 2.5", "5x2", 2.5, (0, 5, 0, 2), (3, 8, 3, 5)),
]
PARENT_PHI = 17.68     # atom-counted porosity of the 3x3 parent tiling


def make_gp():
    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(1e-2, (1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=10, alpha=1e-10,
                                    random_state=RANDOM_STATE)


def main():
    if not os.path.exists(DATA):
        sys.exit(f"dataset not found: {DATA}\n"
                 f"Run from the repository root, or set DATA_DIR.")
    d = pd.read_csv(DATA)
    d.columns = [c.strip() for c in d.columns]
    X = d[FEATURES].values
    sc = StandardScaler().fit(X)

    gk = make_gp().fit(sc.transform(X), np.log(d["kappa"].values))
    gE = make_gp().fit(sc.transform(X), d["E"].values)
    sig_k, sig_E = np.std(d["kappa"].values), np.std(d["E"].values)

    def predict(phi, ar):
        x = sc.transform([[phi, 3, 37.5, 2, ar]])
        return float(np.exp(gk.predict(x)[0])), float(gE.predict(x)[0])

    def distance(k, e):
        return float(np.hypot((k - TARGET_KAPPA) / sig_k,
                              (e - TARGET_E) / sig_E))

    bulk = bs.build_bulk_lattice()
    n_bulk = len(bulk)

    print("=" * 74)
    print("  BUILD-AWARE INVERSE DESIGN")
    print(f"  target: kappa = {TARGET_KAPPA} W/m.K, E = {TARGET_E} GPa")
    print(f"  scalarisation: normalised Euclidean distance, "
          f"sigma_kappa = {sig_k:.4f}, sigma_E = {sig_E:.4f}")
    print("=" * 74)

    rows = []
    for name, footprint, ar, pa, pb in CANDIDATES:
        atoms = bs.carve_pores(bulk, *pa, *pb)
        phi_true = 100.0 * (1.0 - len(atoms) / n_bulk)
        cells = (pa[1] - pa[0]) * (pa[3] - pa[2])

        k_h, e_h = predict(PARENT_PHI, ar)
        k_t, e_t = predict(phi_true, ar)
        rows.append((name, footprint, cells, PARENT_PHI, k_h, e_h,
                     distance(k_h, e_h), phi_true, k_t, e_t,
                     distance(k_t, e_t)))

    print(f"\n  {'porosity used':<15}{'candidate':<11}{'footprint':>10}"
          f"{'cells':>7}{'phi (%)':>9}{'kappa':>8}{'E':>8}{'d':>8}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {'inherited':<15}{r[0]:<11}{r[1]:>10}{r[2]:>7}"
              f"{r[3]:>9.2f}{r[4]:>8.3f}{r[5]:>8.2f}{r[6]:>8.3f}")
    print("  " + "-" * 74)
    best = min(range(len(rows)), key=lambda i: rows[i][10])
    for i, r in enumerate(rows):
        mark = "  <-- selected" if i == best else ""
        print(f"  {'built':<15}{r[0]:<11}{r[1]:>10}{r[2]:>7}"
              f"{r[7]:>9.2f}{r[8]:>8.3f}{r[9]:>8.2f}{r[10]:>8.3f}{mark}")

    h_best = min(range(len(rows)), key=lambda i: rows[i][6])
    print("\n" + "=" * 74)
    if h_best != best:
        print(f"  The ranking REVERSES. Under the inherited porosity "
              f"{rows[h_best][0]} ranks first;")
        print(f"  at the porosity each structure actually has, "
              f"{rows[best][0]} does.")
    ratio_h = max(r[6] for r in rows) / min(r[6] for r in rows)
    ratio_t = max(r[10] for r in rows) / min(r[10] for r in rows)
    print(f"  Separation: a factor of {ratio_h:.1f} under the inherited "
          f"porosity,")
    print(f"              a factor of {ratio_t:.0f} at the built porosity.")
    print(f"\n  Selected geometry: {rows[best][0]}, "
          f"{rows[best][1]} footprint, phi = {rows[best][7]:.2f}%")
    print(f"  Predicted kappa = {rows[best][8]:.3f} W/m.K, "
          f"E = {rows[best][9]:.2f} GPa")
    print(f"  Build it with:  python3 code/make_inv1.py")
    print("=" * 74)
    print("\n  Porosity is a dependent output of the tiling, not a free")
    print("  variable. Candidates must be constructed before they are ranked.")


if __name__ == "__main__":
    main()
