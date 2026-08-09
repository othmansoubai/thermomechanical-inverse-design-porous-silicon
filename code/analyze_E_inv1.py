#!/usr/bin/env python3
"""
analyze_E_inv1.py - Young's modulus (E) + Poisson (nu) for the INV1 candidate
=============================================================================
analyze_elastic.py is hard-coded to the original 17 geometries, so this does a
focused fit for INV1_AR2.5_s37 from its stress_strain_*.dat files (3 seeds).

stress_strain columns (from in.uniaxial_tensile_parametric):
    step temp exx eyy ezz sxx_GPa syy_GPa szz_GPa evol

    E  = d(sxx)/d(exx)   linear-fit over exx in [EXX_LO, EXX_HI]
    nu = mean(-d(eyy)/d(exx), -d(ezz)/d(exx))   over the same window

>>> CONSISTENCY: if your analyze_elastic.py uses a different linear window than
    [0.0005, 0.005], tell me the bounds and I'll match them so INV1's E sits on
    the same footing as the other 17. R^2 is printed so you can see the fit is
    clean (expect > 0.99 in the elastic regime).

Usage (analysis venv):
    python3 analyze_E_inv1.py
"""

import os
import glob
import numpy as np

LABEL   = "INV1_AR2.5_s37"
SEEDS   = [12345, 45678, 78901]
RESULTS = os.path.expanduser("./results")
EXX_LO, EXX_HI = 0.0005, 0.005     # linear elastic window (matches input comment)


def fit_seed(path):
    d = np.loadtxt(path, comments="#")
    exx, eyy, ezz = d[:, 2], d[:, 3], d[:, 4]
    sxx = d[:, 5]
    m = (exx >= EXX_LO) & (exx <= EXX_HI)
    if m.sum() < 3:
        raise ValueError(f"only {m.sum()} points in [{EXX_LO},{EXX_HI}] for {path}")
    # E from sxx vs exx
    pE, covE = np.polyfit(exx[m], sxx[m], 1, cov=True)
    E = pE[0]
    yhat = np.polyval(pE, exx[m])
    ss_res = np.sum((sxx[m] - yhat) ** 2)
    ss_tot = np.sum((sxx[m] - sxx[m].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # nu from lateral contraction
    nu_y = -np.polyfit(exx[m], eyy[m], 1)[0]
    nu_z = -np.polyfit(exx[m], ezz[m], 1)[0]
    nu = 0.5 * (nu_y + nu_z)
    sigma_max = float(np.max(sxx))
    return E, nu, r2, sigma_max


def main():
    print("=" * 64)
    print(f"  E / nu extraction - {LABEL}  (3 seeds, exx in [{EXX_LO},{EXX_HI}])")
    print("=" * 64)
    Es, nus = [], []
    for seed in SEEDS:
        hits = glob.glob(os.path.join(RESULTS, f"stress_strain_{LABEL}_seed{seed}.dat"))
        if not hits:
            print(f"  seed {seed}: stress_strain file missing -- skipped")
            continue
        try:
            E, nu, r2, smax = fit_seed(hits[0])
        except Exception as e:
            print(f"  seed {seed}: fit failed ({e})")
            continue
        Es.append(E); nus.append(nu)
        print(f"  seed {seed}:  E = {E:6.2f} GPa   nu = {nu:5.3f}   "
              f"R^2 = {r2:.4f}   sigma_max = {smax:.2f} GPa")

    if Es:
        E = np.array(Es); nu = np.array(nus)
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
        print("-" * 64)
        print(f"  E  = {E.mean():6.2f} +/- {sd(E):.2f} GPa   (n={len(E)})")
        print(f"  nu = {nu.mean():.3f} +/- {sd(nu):.3f}")
        print(f"  GP prediction: E ~ 70-71 GPa   |   target: 70 GPa")
        print("=" * 64)
    else:
        print("  no E values computed -- check results/ for stress_strain_*.dat")


if __name__ == "__main__":
    main()
