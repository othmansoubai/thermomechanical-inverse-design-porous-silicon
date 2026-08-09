#!/usr/bin/env python3
"""
analyze_kappa.py — Phase 2 Group B post-processing
====================================================
Computes thermal conductivity κ (W/m·K) from NEMD Müller-Plathe output.

For each geometry (label × 3 seeds):
  1. Reads temp_profile_<label>_seed<seed>.dat
  2. Fits linear dT/dz to the two quasi-linear segments
     (cold→hot half and hot→cold half), averages them
  3. Reads E_transferred from LAMMPS log (f_MP final value)
  4. Computes κ = (E_total × e) / (2 × t × A × |dT/dz|)
  5. Reports per-seed values and mean ± std across seeds

Usage:
  python3 analyze_kappa.py                  # process all in results/
  python3 analyze_kappa.py --label grpB_aligned_28pct

Output:
  kappa_results_grpB.csv  — all results
  kappa_summary_grpB.png  — bar chart with error bars

Paper 1 reference values (for sanity check):
  full_stagger_7pct  : κ = 7.63 ± 0.99 W/m·K  (φ_actual = 6.84%)
  full_stagger_28pct : κ = 2.59 ± 0.02 W/m·K  (φ_actual = 28.27%)
  bulk               : κ = 12.03 ± 0.29 W/m·K (φ = 0%)
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

# ── Physical constants ────────────────────────────────────
eV_to_J = 1.602176634e-19   # J/eV
ps_to_s  = 1e-12             # s/ps
AA_to_m  = 1e-10             # m/Angstrom

# ── Simulation parameters (must match LAMMPS script) ─────
N_PROD     = 500_000
DT_PS      = 0.001           # ps
T_PROD_S   = N_PROD * DT_PS * ps_to_s
LX_AA      = 8 * 5.431       # 43.448 Å
LY_AA      = 8 * 5.431       # 43.448 Å
A_CROSS_M2 = (LX_AA * AA_to_m) * (LY_AA * AA_to_m)

SEEDS = [12345, 34567, 56789]

LABELS = [
    "grpB_aligned_28pct",
    "grpB_quarter_stagger_28pct",
    "grpB_aligned_7pct",
    "grpB_quarter_stagger_7pct",
]

# Paper 1 reference points (to add to plots)
PAPER1_REF = {
    "bulk":                 {"phi": 0.00,  "kappa": 12.03, "kappa_std": 0.29},
    "full_stagger_7pct":    {"phi": 6.84,  "kappa":  7.63, "kappa_std": 0.99},
    "full_stagger_28pct":   {"phi": 28.27, "kappa":  2.59, "kappa_std": 0.02},
}


# ── Helper: read temperature profile ──────────────────────
def read_temp_profile(filepath):
    """
    Read LAMMPS ave/chunk output.
    Returns arrays: z_coords (Å), T_mean (K)
    Only the LAST block is used (fully averaged production run).
    """
    blocks = []
    current = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:          # new block header: timestep Nchunk Natoms
                if current:
                    blocks.append(np.array(current, dtype=float))
                current = []
            else:
                try:
                    current.append([float(x) for x in parts])
                except ValueError:
                    pass
    if current:
        blocks.append(np.array(current, dtype=float))

    if not blocks:
        raise ValueError(f"No data blocks found in {filepath}")

    last = blocks[-1]
    z = last[:, 1]   # coordinate (Å)
    T = last[:, 3]   # temperature (K)
    return z, T


# ── Helper: compute dT/dz from temperature profile ────────
def compute_dtdz(z_AA, T_K):
    """
    NEMD Müller-Plathe profile: hot slab at z=Lz/2, cold at z=0 and z=Lz.
    Fit two quasi-linear halves, return mean |dT/dz| in K/m.
    """
    z_m = z_AA * AA_to_m
    n   = len(z_m)
    mid = n // 2

    slope_L, _, _, _, _ = linregress(z_m[:mid], T_K[:mid])
    slope_R, _, _, _, _ = linregress(z_m[mid:], T_K[mid:])

    dtdz_abs = (abs(slope_L) + abs(slope_R)) / 2.0  # K/m

    T_hot  = T_K[mid]
    T_cold = (T_K[0] + T_K[-1]) / 2.0
    delta_T = T_hot - T_cold

    return dtdz_abs, T_hot, T_cold, delta_T


# ── Helper: extract E_transferred from LAMMPS log ─────────
def extract_e_transfer_from_log(log_path):
    e_final = None
    col_idx = None
    with open(log_path, "r", encoding="latin-1") as f:
        for line in f:
            if "f_mp" in line and "Step" in line:
                headers = line.split()
                if "f_mp" in headers:
                    col_idx = headers.index("f_mp")
            elif col_idx is not None:
                parts = line.split()
                if len(parts) > col_idx:
                    try:
                        e_final = float(parts[col_idx])
                    except ValueError:
                        pass
    if e_final is None:
        raise ValueError(f"Could not extract f_MP from {log_path}")
    return e_final


# ── Helper: read E_transferred from e_transfer file ───────
def extract_e_transfer_from_file(filepath):
    e_last = None
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    e_last = float(parts[1])
                except ValueError:
                    pass
    if e_last is None:
        raise ValueError(f"No data in {filepath}")
    return e_last


# ── Main κ calculation ─────────────────────────────────────
def compute_kappa(label, seed, results_dir="results", logs_dir="logs"):
    tprof_file = Path(results_dir) / f"temp_profile_{label}_seed{seed}.dat"
    etrans_file = Path(results_dir) / f"e_transfer_{label}_seed{seed}.dat"
    log_file    = Path(logs_dir)    / f"lammps_{label}_seed{seed}.log"

    z, T = read_temp_profile(tprof_file)
    dtdz, T_hot, T_cold, delta_T = compute_dtdz(z, T)

    if etrans_file.exists():
        E_eV = extract_e_transfer_from_file(etrans_file)
    elif log_file.exists():
        E_eV = extract_e_transfer_from_log(log_file)
    else:
        raise FileNotFoundError(
            f"Cannot find energy file for {label} seed {seed}.\n"
            f"  Looked for: {etrans_file}\n"
            f"  and log:    {log_file}"
        )

    E_J   = E_eV * eV_to_J
    kappa = E_J / (2.0 * T_PROD_S * A_CROSS_M2 * dtdz)

    phi_actual = None
    if log_file.exists():
        with open(log_file, "r", encoding="latin-1") as f:
            for line in f:
                m = re.search(r"ACTUAL POROSITY:\s*([\d.]+)\s*%", line)
                if m:
                    phi_actual = float(m.group(1))

    return {
        "label":      label,
        "seed":       seed,
        "kappa":      kappa,
        "E_eV":       E_eV,
        "dtdz_Km":    dtdz,
        "T_hot":      T_hot,
        "T_cold":     T_cold,
        "delta_T":    delta_T,
        "phi_actual": phi_actual,
    }


# ── Aggregate over seeds ───────────────────────────────────
def aggregate_label(label, results_dir="results", logs_dir="logs"):
    rows = []
    for seed in SEEDS:
        try:
            r = compute_kappa(label, seed, results_dir, logs_dir)
            rows.append(r)
            print(f"  {label} seed={seed}: κ = {r['kappa']:.2f} W/m·K  "
                  f"ΔT = {r['delta_T']:.1f} K  φ = {r['phi_actual']}%")
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: skipping {label} seed={seed}: {e}")
    return rows


# ── Plot summary ───────────────────────────────────────────
def plot_summary(summary_df, outfile="kappa_summary_grpB.png"):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "grpB_aligned_28pct":           "#E8706A",
        "grpB_quarter_stagger_28pct":   "#5BA3CF",
        "grpB_aligned_7pct":            "#F4A261",
        "grpB_quarter_stagger_7pct":    "#3B9B6D",
    }
    nice_labels = {
        "grpB_aligned_28pct":           "Aligned\n(~28%)",
        "grpB_quarter_stagger_28pct":   "Quarter stagger\n(~28%)",
        "grpB_aligned_7pct":            "Aligned\n(~7%)",
        "grpB_quarter_stagger_7pct":    "Quarter stagger\n(~7%)",
    }

    xs = list(range(len(summary_df)))
    bar_labels = [nice_labels.get(l, l) for l in summary_df["label"]]

    bars = ax.bar(
        xs,
        summary_df["kappa_mean"],
        yerr=summary_df["kappa_std"],
        color=[colors.get(l, "gray") for l in summary_df["label"]],
        edgecolor="black", linewidth=1.2, capsize=5, width=0.6,
    )

    ax.axhline(PAPER1_REF["bulk"]["kappa"],               color="gray",    ls="--", lw=1.5, label="Bulk (0%)")
    ax.axhline(PAPER1_REF["full_stagger_7pct"]["kappa"],  color="#3B9B6D", ls=":",  lw=1.5, label="Full stagger 7%  (Paper 1)")
    ax.axhline(PAPER1_REF["full_stagger_28pct"]["kappa"], color="#E8706A", ls=":",  lw=1.5, label="Full stagger 28% (Paper 1)")

    for bar, val, err in zip(bars, summary_df["kappa_mean"], summary_df["kappa_std"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel("κ [W/(m·K)]", fontsize=12)
    ax.set_title("Phase 2 Group B — Stagger Offset Effect on Thermal Conductivity\n"
                 "(NEMD Müller-Plathe, Nz=48, Lz=26 nm, 3 seeds each)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 14)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"\n✅ Summary plot saved: {outfile}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",   default=None, help="Process single label only")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--logs",    default="logs",    help="Logs directory")
    args = parser.parse_args()

    labels_to_run = [args.label] if args.label else LABELS

    all_rows = []
    for label in labels_to_run:
        print(f"\nProcessing: {label}")
        rows = aggregate_label(label, args.results, args.logs)
        all_rows.extend(rows)

    if not all_rows:
        print("No results found. Have the NEMD runs completed?")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv("kappa_results_grpB_perseed.csv", index=False)
    print(f"\nPer-seed results saved: kappa_results_grpB_perseed.csv")

    summary = (
        df.groupby("label")
          .agg(
              kappa_mean=("kappa",      "mean"),
              kappa_std=("kappa",       "std"),
              delta_T_mean=("delta_T",  "mean"),
              phi_actual=("phi_actual", "first"),
          )
          .reset_index()
    )
    summary["label"] = pd.Categorical(summary["label"], categories=LABELS, ordered=True)
    summary = summary.sort_values("label").reset_index(drop=True)

    print("\n" + "="*60)
    print("SUMMARY (mean ± std over 3 seeds)")
    print("="*60)
    for _, row in summary.iterrows():
        print(f"  {row['label']:<35}  κ = {row['kappa_mean']:.2f} ± {row['kappa_std']:.2f} W/m·K"
              f"  φ_actual = {row['phi_actual']}%")

    print("\nPaper 1 references:")
    for name, ref in PAPER1_REF.items():
        print(f"  {name:<35}  κ = {ref['kappa']:.2f} ± {ref['kappa_std']:.2f} W/m·K")

    summary.to_csv("kappa_results_grpB.csv", index=False)
    print(f"\nSummary saved: kappa_results_grpB.csv")

    plot_summary(summary)


if __name__ == "__main__":
    main()
