#!/usr/bin/env python3
"""
analyze_kappa_inv1.py - extract kappa for the INV1 candidate (3 seeds)
======================================================================
Reproduces the Phase 2 analyze_kappa.py reduction so INV1's kappa is directly
comparable to the 17-geometry dataset:

    kappa = E_swapped / ( 2 * A * t * |dT/dz| )

  E_swapped : cumulative KE swapped by fix thermal/conductivity (eV -> J)
  A         : cross-section Lx*Ly (m^2)            } from the log, with
  t         : production time (s)                  } fixed-box fallbacks
  dT/dz     : split the z profile at its midpoint, linear-fit each half using
              all points (Müller-Plathe hot slab at mid z, cold at both ends),
              mean |slope|.  IDENTICAL to Paper 2 compute_dtdz.
  factor 2  : heat flows both ways from the hot slab (periodic box).

phi is computed from ATOM COUNTS (24576 -> remaining), NOT the logged
"ACTUAL POROSITY", which is 0 due to the equal-style count(all) re-eval bug.

NOTE: the Phase 2 NEMD input records only 4 z-bins ('$(lz/20)' missing
'units box'); the whole dataset was reduced from 4-bin profiles, so this
matches that -- do not switch INV1 to 20 bins or it stops being comparable.

Usage (analysis venv):
    python3 analyze_kappa_inv1.py
"""

import os
import re
import glob
import numpy as np

LABEL  = "INV1_AR2.5_s37"
SEEDS  = [12345, 45678, 78901]
RESULTS = os.path.expanduser(os.environ.get("RESULTS_DIR", "results_thermal"))
LOGDIR  = os.path.expanduser("./logs")
EV_J   = 1.602176634e-19
NSLABS = 20            # matches Nslabs in the NEMD input

# fixed-box fallbacks (8x8x48, a=5.431 A; tnemd=500 ps) if a log value is missing
A_FALLBACK = (8 * 5.431e-10) ** 2          # m^2
T_FALLBACK = 500e-12                        # s


def grab(logtext, pattern):
    m = re.search(pattern, logtext)
    return float(m.group(1)) if m else None


def read_log(seed):
    """Pull E_swapped (eV), A (m^2), t (s), phi (%) from the LAMMPS log."""
    hits = (glob.glob(os.path.join(LOGDIR, f"lammps_nemd_{LABEL}_seed{seed}.log"))
            + glob.glob(os.path.join(LOGDIR, f"*nemd*{LABEL}*seed{seed}*.log"))
            + glob.glob(os.path.join(LOGDIR, f"*{LABEL}*seed{seed}*.log")))
    if not hits:
        return None
    txt = open(hits[0], errors="replace").read()
    return {
        "E_eV": grab(txt, r"FINAL E_TRANSFERRED \(eV\):\s*([0-9.eE+-]+)"),
        "A":    grab(txt, r"CROSS-SECTION A \(m\^2\):\s*([0-9.eE+-]+)"),
        "t":    grab(txt, r"PRODUCTION TIME \(s\):\s*([0-9.eE+-]+)"),
        # NOTE: the logged "ACTUAL POROSITY" is buggy (equal-style count(all)
        # re-evaluates after delete_atoms -> 0). Compute phi from atom counts:
        # ATOMS_INIT is printed BEFORE deletion (correct); 'remaining=' after.
        "n_init": grab(txt, r"ATOMS_INIT:\s*([0-9]+)"),
        "n_rem":  grab(txt, r"remaining=\s*([0-9]+)"),
    }


def last_profile_block(path):
    """Return (z[Å], T[K]) arrays from the final ave/chunk block."""
    blocks, cur = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            f = s.split()
            if len(f) == 3:                      # block header: step nchunks totalcount
                if cur:
                    blocks.append(cur); cur = []
            elif len(f) >= 4:                    # chunk row: id coord ncount temp ...
                cur.append((float(f[1]), float(f[3])))
        if cur:
            blocks.append(cur)
    if not blocks:
        raise ValueError(f"no data blocks in {path}")
    arr = np.array(blocks[-1])
    return arr[:, 0], arr[:, 1]


def gradient_K_per_m(z_ang, T):
    """dT/dz reproducing Paper 2 analyze_kappa.py compute_dtdz EXACTLY, so INV1's
    kappa sits on the same footing as the 17-geometry dataset:

      split the profile at n//2, linear-fit EACH half using all its points
      (Müller-Plathe hot slab at mid z, cold at both ends), return mean |slope|.

    NOTE: the Phase 2 NEMD input records only 4 z-bins -- '$(lz/20)' set the bin
    WIDTH but was read in lattice units (missing 'units box'), giving 4 fat bins
    instead of 20. Paper 2's whole dataset was reduced this way, so we match it
    rather than 'fix' to 20 bins (which would make INV1 non-comparable). With 4
    bins each half has 2 points -> an exact line, no conditioning issue.
    """
    from scipy.stats import linregress
    z = z_ang * 1e-10                            # Å -> m
    n = len(T)
    mid = n // 2
    sL = linregress(z[:mid], T[:mid]).slope      # cold-end -> hot-mid  (+)
    sR = linregress(z[mid:], T[mid:]).slope      # hot-mid  -> cold-end (-)
    return 0.5 * (abs(sL) + abs(sR)), sL, sR


def main():
    print("=" * 64)
    print(f"  kappa extraction - {LABEL}  (Müller-Plathe, 3 seeds)")
    print("=" * 64)
    kappas, phis = [], []
    for seed in SEEDS:
        prof = glob.glob(os.path.join(RESULTS, f"temp_profile_{LABEL}_seed{seed}.dat"))
        etr  = glob.glob(os.path.join(RESULTS, f"e_transfer_{LABEL}_seed{seed}.dat"))
        log  = read_log(seed)
        if not prof:
            print(f"  seed {seed}: temp_profile missing -- skipped")
            continue

        z, T = last_profile_block(prof[0])
        dTdz, s1, s2 = gradient_K_per_m(z, T)

        # energy swapped: prefer log print, fall back to last row of e_transfer file
        E_eV = log["E_eV"] if (log and log["E_eV"]) else None
        if E_eV is None and etr:
            try:
                E_eV = float(np.loadtxt(etr[0], comments="#")[-1, 1])
            except Exception:
                E_eV = None
        A = (log["A"] if (log and log["A"]) else A_FALLBACK)
        t = (log["t"] if (log and log["t"]) else T_FALLBACK)
        # phi from atom counts (NOT the buggy logged value); bulk=24576 fallback
        if log and log.get("n_rem"):
            n0 = log["n_init"] if log.get("n_init") else 24576.0
            phi = 100.0 * (n0 - log["n_rem"]) / n0
        else:
            phi = float("nan")

        if E_eV is None:
            print(f"  seed {seed}: energy-swapped not found in log or e_transfer -- skipped")
            continue

        kappa = (E_eV * EV_J) / (2.0 * A * t * dTdz)
        kappas.append(kappa); phis.append(phi)
        phi_txt = f"phi={phi:5.2f}%  " if np.isfinite(phi) else ""
        print(f"  seed {seed}:  {phi_txt}|dT/dz|={dTdz:.3e} K/m  "
              f"E_swap={E_eV:.4g} eV  ->  kappa = {kappa:.3f} W/m.K")
        print(f"            (branch slopes {s1:+.3e} / {s2:+.3e} K/m)")

    if kappas:
        k = np.array(kappas)
        print("-" * 64)
        print(f"  kappa = {k.mean():.3f} +/- {k.std(ddof=1) if len(k)>1 else 0:.3f} W/m.K "
              f"(n={len(k)})" + ("" if not np.any(np.isfinite(phis))
                        else f"   phi ~ {np.nanmean(phis):.2f}%"))
        print(f"  GP prediction: kappa 2.380 W/m.K (1-sigma 2.048-2.765)   |   target: 2.4 W/m.K")
        print("=" * 64)
    else:
        print("  no kappa values computed -- check that results/ has the .dat files.")


if __name__ == "__main__":
    main()
