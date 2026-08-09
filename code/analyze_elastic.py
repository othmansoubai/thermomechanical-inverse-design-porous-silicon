#!/usr/bin/env python3
"""
analyze_elastic.py - Paper 3 Phase 1 stress-strain analysis
============================================================
Reads stress_strain_<label>_seed<seed>.dat from results/ and
extracts elastic constants per (geometry, seed), then aggregates
mean +/- std across seeds.

Quantities computed:
    E      [GPa]  Young's modulus (slope of sigma_xx vs eps_xx)
    nu_y          Poisson's ratio from -d(eps_yy)/d(eps_xx)
    nu_z          Poisson's ratio from -d(eps_zz)/d(eps_xx)
    nu            mean of nu_y, nu_z (anisotropy = |nu_y - nu_z|)
    K      [GPa]  Bulk modulus  K = E / [3(1 - 2 nu)]   (isotropic est.)
    sigma_max [GPa]  Peak tensile stress reached
    eps_y         Yield strain (5% deviation below linear extrapolation)

Linear fit window: eps_xx in [0.0005, 0.005].
The lower bound avoids thermal-noise-dominated near-zero strains;
the upper bound stays safely inside the elastic regime for Si.

Usage on MARWAN:
    python3 analyze_elastic.py
Outputs:
    elastic_perseed.csv     - 54 rows (one per simulation)
    elastic_dataset.csv     - 18 rows (mean +/- std per geometry)
                              joinable to ml_dataset_paper2.csv on `label`
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---- Configuration ---------------------------------------------
ELASTIC_WINDOW = (0.0005, 0.005)   # linear-fit strain range
YIELD_DEVIATION = 0.05             # 5% drop below linear extrapolation = yield


# ---- Helpers ----------------------------------------------------
def fit_linear(x, y, window):
    """Linear regression y = a*x + b in window. Returns (slope, R^2, n_points)."""
    mask = (x >= window[0]) & (x <= window[1])
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    p = np.polyfit(x[mask], y[mask], 1)
    slope, intercept = p
    y_pred = np.polyval(p, x[mask])
    ss_res = np.sum((y[mask] - y_pred) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(r2), n


def find_yield(exx, sxx, E, deviation=YIELD_DEVIATION):
    """First strain (after the elastic window) where sigma drops `deviation`
    below the linear-elastic extrapolation E*eps_xx. Returns NaN if not found.
    """
    if not np.isfinite(E) or E <= 0:
        return np.nan
    sigma_lin = E * exx
    rel_drop = (sigma_lin - sxx) / np.maximum(sigma_lin, 1e-9)
    mask = (exx > ELASTIC_WINDOW[1]) & (rel_drop > deviation)
    if not mask.any():
        return np.nan
    idx = int(np.argmax(mask))
    return float(exx[idx])


def analyze_one(filepath):
    """Read one stress_strain_*.dat file and extract elastic constants."""
    data = np.loadtxt(filepath, comments='#')
    # columns: step temp exx eyy ezz sxx syy szz evol
    exx, eyy, ezz = data[:, 2], data[:, 3], data[:, 4]
    sxx          = data[:, 5]

    # Sort by exx in case timesteps are non-monotonic (they should be monotonic
    # under fix deform, but be defensive).
    order = np.argsort(exx)
    exx, eyy, ezz, sxx = exx[order], eyy[order], ezz[order], sxx[order]

    E,    R2_E,    n_E    = fit_linear(exx, sxx, ELASTIC_WINDOW)
    sl_y, R2_nu_y, _      = fit_linear(exx, eyy, ELASTIC_WINDOW)
    sl_z, R2_nu_z, _      = fit_linear(exx, ezz, ELASTIC_WINDOW)
    nu_y, nu_z = -sl_y, -sl_z
    nu = 0.5 * (nu_y + nu_z) if np.isfinite(nu_y) and np.isfinite(nu_z) else np.nan
    K  = E / (3.0 * (1.0 - 2.0 * nu)) if np.isfinite(E) and np.isfinite(nu) and (1.0 - 2.0 * nu) > 1e-3 else np.nan

    return {
        'E_GPa':            E,
        'R2_E':             R2_E,
        'n_fit':            n_E,
        'nu_y':             nu_y,
        'nu_z':             nu_z,
        'nu':               nu,
        'anisotropy':       abs(nu_y - nu_z) if (np.isfinite(nu_y) and np.isfinite(nu_z)) else np.nan,
        'K_GPa':            K,
        'sigma_max_GPa':    float(np.max(sxx)),
        'eps_y':            find_yield(exx, sxx, E),
    }


# ---- Main ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results',        default='results',           help='Directory with stress_strain_*.dat files')
    ap.add_argument('--jobs',           default='jobs_phase3.csv',   help='Job table')
    ap.add_argument('--out-perseed',    default='elastic_perseed.csv')
    ap.add_argument('--out-aggregated', default='elastic_dataset.csv')
    args = ap.parse_args()

    jobs = pd.read_csv(args.jobs, comment='#')
    results_dir = Path(args.results)

    rows = []
    missing = []
    for _, job in jobs.iterrows():
        label = str(job['label'])
        seed  = int(job['seed'])
        fname = f"stress_strain_{label}_seed{seed}.dat"
        path  = results_dir / fname
        if not path.exists():
            missing.append(fname)
            continue
        try:
            r = analyze_one(path)
        except Exception as exc:
            print(f"FAILED  {fname}: {exc}")
            continue
        r['label'] = label
        r['group'] = job['group']
        r['seed']  = seed
        rows.append(r)

    if missing:
        print(f"\n[!] Missing {len(missing)} of {len(jobs)} files. First few:")
        for m in missing[:5]:
            print(f"    {m}")
        print()

    if not rows:
        print("No files analyzed. Aborting.")
        return

    perseed = pd.DataFrame(rows)
    cols = ['label', 'group', 'seed',
            'E_GPa', 'R2_E', 'n_fit',
            'nu', 'nu_y', 'nu_z', 'anisotropy',
            'K_GPa', 'sigma_max_GPa', 'eps_y']
    perseed = perseed[cols]
    perseed.to_csv(args.out_perseed, index=False)
    print(f"[OK] Per-seed: {args.out_perseed} ({len(perseed)} rows)")

    # Aggregate per geometry
    metrics = ['E_GPa', 'nu', 'K_GPa', 'sigma_max_GPa', 'eps_y', 'anisotropy']
    agg = perseed.groupby(['label', 'group'])[metrics].agg(['mean', 'std']).reset_index()
    agg.columns = [f"{a}_{b}" if b else a for a, b in agg.columns]
    agg.to_csv(args.out_aggregated, index=False)
    print(f"[OK] Aggregated: {args.out_aggregated} ({len(agg)} rows)")

    # Pretty-print a summary table
    print("\n" + "=" * 92)
    print(f"{'Geometry':<22} {'group':<6} {'E (GPa)':>14} {'nu':>10} {'K (GPa)':>14} {'sigma_max':>10} {'eps_y':>8}")
    print("=" * 92)
    for _, r in agg.iterrows():
        E_str  = f"{r['E_GPa_mean']:6.1f} +/- {r['E_GPa_std']:4.1f}"  if np.isfinite(r['E_GPa_mean'])  else "       -"
        K_str  = f"{r['K_GPa_mean']:6.1f} +/- {r['K_GPa_std']:4.1f}"  if np.isfinite(r['K_GPa_mean'])  else "       -"
        nu_str = f"{r['nu_mean']:7.3f}"                                if np.isfinite(r['nu_mean'])     else "      -"
        sg_str = f"{r['sigma_max_GPa_mean']:7.2f}"                     if np.isfinite(r['sigma_max_GPa_mean']) else "      -"
        ey_str = f"{r['eps_y_mean']:6.3f}"                             if np.isfinite(r['eps_y_mean'])  else "     -"
        print(f"{r['label']:<22} {r['group']:<6} {E_str:>14} {nu_str:>10} {K_str:>14} {sg_str:>10} {ey_str:>8}")
    print("=" * 92)


if __name__ == '__main__':
    main()
