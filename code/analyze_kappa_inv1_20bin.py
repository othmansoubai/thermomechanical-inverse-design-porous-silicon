#!/usr/bin/env python3
"""
analyze_kappa_inv1_20bin.py — kappa for INV1 from the corrected 20-bin profile.

Same reduction as analyze_kappa_inv1.py (split the profile at the midpoint,
fit each branch linearly, average the two magnitudes), applied to the runs
made with the corrected binning. Reports the linear-fit quality per branch,
which the four-bin profile could not provide, and compares the result with
the published four-bin value.

Run from ~/phase3_mech:
    module purge && module load GCCcore/12.3.0 Python/3.11.3-GCCcore-12.3.0
    source ~/phase2_ml/analysis_env/bin/activate
    python3 analyze_kappa_inv1_20bin.py
    deactivate
"""

import os
import re
import sys
import numpy as np
from scipy.stats import linregress

RESULTS = os.path.expanduser(os.environ.get("RESULTS_DIR", "results_thermal"))
LABEL   = "INV1_AR2.5_s37_20bin"
SEEDS   = [12345, 45678, 78901]

AREA_A2 = (8 * 5.431) * (8 * 5.431)     # cross-section normal to z, A^2
EV_TO_J = 1.602176634e-19
A_TO_M  = 1e-10
PS_TO_S = 1e-12
PROD_PS = 500.0                          # production time

# published four-bin result, for comparison
K4_MEAN, K4_STD = 2.362, 0.178


def read_profile(path):
    """Return (z_A, T_K) averaged over the second half of the production run."""
    blocks, cur, header = [], [], None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) == 3 and "." not in p[1]:      # timestep header
                if cur:
                    blocks.append(np.array(cur))
                cur, header = [], p
            elif len(p) == 4:
                cur.append([float(p[1]), float(p[3])])
    if cur:
        blocks.append(np.array(cur))
    if not blocks:
        raise RuntimeError(f"no data parsed from {path}")
    half = blocks[len(blocks) // 2:]                  # steady state only
    z = half[0][:, 0]
    T = np.mean([b[:, 1] for b in half], axis=0)
    return z, T, len(blocks), len(half)


def read_eswap(path):
    """Cumulative swapped energy, eV — last value in the file."""
    last = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                try:
                    last = float(p[-1])
                except ValueError:
                    pass
    if last is None:
        raise RuntimeError(f"no swap energy in {path}")
    return last


def main():
    print("=" * 70)
    print(f"  kappa from the CORRECTED 20-bin profile — {LABEL}")
    print("=" * 70)

    kappas = []
    for seed in SEEDS:
        pf = os.path.join(RESULTS, f"temp_profile_{LABEL}_seed{seed}.dat")
        ef = os.path.join(RESULTS, f"e_transfer_{LABEL}_seed{seed}.dat")
        if not (os.path.exists(pf) and os.path.exists(ef)):
            print(f"  seed {seed}: files not found, skipping")
            continue

        z, T, nblk, nused = read_profile(pf)
        e_swap = read_eswap(ef)

        mid = len(z) // 2
        f1 = linregress(z[:mid], T[:mid])
        f2 = linregress(z[mid:], T[mid:])
        # slopes in K/A -> K/m
        s1, s2 = f1.slope / A_TO_M, f2.slope / A_TO_M
        grad = 0.5 * (abs(s1) + abs(s2))

        kappa = (e_swap * EV_TO_J) / (
            2.0 * AREA_A2 * A_TO_M**2 * PROD_PS * PS_TO_S * grad)
        kappas.append(kappa)

        print(f"\n  seed {seed}   ({nused} of {nblk} blocks averaged, "
              f"{len(z)} bins)")
        print(f"    branch 1: slope = {s1:+.3e} K/m   R^2 = {f1.rvalue**2:.4f}")
        print(f"    branch 2: slope = {s2:+.3e} K/m   R^2 = {f2.rvalue**2:.4f}")
        print(f"    |dT/dz| = {grad:.3e} K/m   E_swap = {e_swap:.0f} eV"
              f"   ->  kappa = {kappa:.3f} W/m.K")

    if not kappas:
        sys.exit("\n  no seeds reduced — check that the runs completed.")

    k = np.array(kappas)
    m, s = k.mean(), (k.std(ddof=1) if len(k) > 1 else 0.0)
    print("\n" + "-" * 70)
    print(f"  20-bin :  kappa = {m:.3f} +/- {s:.3f} W/m.K  (n={len(k)})")
    print(f"   4-bin :  kappa = {K4_MEAN:.3f} +/- {K4_STD:.3f} W/m.K  "
          f"(published)")

    diff = m - K4_MEAN
    sem = np.hypot(s / max(np.sqrt(len(k)), 1), K4_STD / np.sqrt(3))
    print(f"\n  difference = {diff:+.3f} W/m.K "
          f"({100*diff/K4_MEAN:+.1f}%), "
          f"{abs(diff)/sem:.2f} standard errors of the seed means")
    if abs(diff) < sem * 2:
        print("  -> within seed scatter: the coarse binning does not bias "
              "the reported value.")
    else:
        print("  -> OUTSIDE seed scatter: the binning does affect the result "
              "and must be reported.")
    print("=" * 70)
    print("\n  R^2 per branch above is the check the four-bin profile could")
    print("  not provide: values near 1 indicate a linear profile, lower")
    print("  values indicate curvature near the thermostatted slabs.")


if __name__ == "__main__":
    main()
