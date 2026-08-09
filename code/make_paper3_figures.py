#!/usr/bin/env python3
"""
make_paper3_figures.py - publication figures for Paper 3 (Figs 2-6)
===================================================================
Reads paper3_multiobj_dataset.csv (17 geometries) and produces:
  fig_pareto.{pdf,png}            - kappa-E Pareto front
  fig_parity.{pdf,png}           - LOOCV predicted vs actual (kappa, E)
  fig_importance.{pdf,png}       - permutation importance, kappa vs E
  fig_inverse_selection.{pdf,png}- AR<->phi coupling; why AR=2.5 beats AR=2.0
  fig_validation.{pdf,png}       - INV1 measured vs target vs neighbors

Fig 1 (structure schematic) is an OVITO render (use render_structures.py) - not here.

Usage (analysis venv, from ~/phase3_mech):
    python3 make_paper3_figures.py                 # uses paper3_multiobj_dataset.csv
    python3 make_paper3_figures.py mydataset.csv   # or a custom path
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.inspection import permutation_importance

RS = 42
np.random.seed(RS)
DATA = sys.argv[1] if len(sys.argv) > 1 else "paper3_multiobj_dataset.csv"
FIGURES_DIR = os.path.expanduser(
    os.environ.get("FIGURES_DIR", "figures"))   # override with FIGURES_DIR
FEATURES = ["phi", "S", "stagger", "neck_uc", "AR"]

# ---- shared style ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})
TEAL, GRAY, CORAL, AMBER = "#0F6E56", "#8A897F", "#D0521B", "#B07A10"

# INV1 measured (from inv1_validation_point.csv) - held-out validation point
INV1 = dict(kappa=2.362, kappa_std=0.178, E=68.40, E_std=0.15, phi=19.26)
TARGET = dict(kappa=2.4, E=70.0)


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, name)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf / .png")


def load():
    d = pd.read_csv(DATA)
    d.columns = [c.strip() for c in d.columns]
    need = FEATURES + ["kappa", "E", "label"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        sys.exit(f"dataset missing columns {miss}; has {list(d.columns)}")
    return d


def make_gp(log=False):
    k = (ConstantKernel(1.0, (1e-2, 1e3)) *
         Matern(1.0, (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=10, random_state=RS)


def pareto_mask(kappa, E):
    """non-dominated for (min kappa, max E)."""
    nd = np.ones(len(kappa), bool)
    for i in range(len(kappa)):
        for j in range(len(kappa)):
            if j != i and kappa[j] <= kappa[i] and E[j] >= E[i] and (
               kappa[j] < kappa[i] or E[j] > E[i]):
                nd[i] = False
                break
    return nd


# ---- Fig 2: Pareto front --------------------------------------------------
def fig_pareto(d):
    k, E = d["kappa"].values, d["E"].values
    nd = pareto_mask(k, E)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ks = next((c for c in ["kappa_std","kappa_sd","kappa_err"] if c in d.columns), None)
    Es = next((c for c in ["E_std","E_sd","E_err"] if c in d.columns), None)
    if ks and Es:
        ax.errorbar(k, E, xerr=d[ks].values, yerr=d[Es].values, fmt="none",
                    ecolor="#999999", elinewidth=0.8, capsize=2, zorder=1)
        print(f"   pareto: error bars drawn from {ks}/{Es}")
    else:
        print("   pareto: NO std columns -> DROP the error-bar clause from the caption")
    ax.scatter(k[~nd], E[~nd], s=42, c=GRAY, edgecolor="k", lw=0.5,
               label="dominated", zorder=2)
    ax.scatter(k[nd], E[nd], s=70, c=TEAL, edgecolor="k", lw=0.6,
               label="Pareto-optimal", zorder=3)
    order = np.argsort(k[nd])
    ax.plot(k[nd][order], E[nd][order], "-", c=TEAL, lw=1.5, alpha=0.7, zorder=1)
    nudge = {"P1_high_50pct": (4, 8), "B_S4_quarter_high": (4, -10),
             "D_AR2.0_wide": (6, 4), "D_AR2.5_vwide": (4, -10),
             "B_S4_quarter_low": (6, -2)}
    for lbl, kk, ee in zip(d["label"][nd], k[nd], E[nd]):
        dx, dy = nudge.get(lbl, (4, 3))
        ax.annotate(lbl.replace("_", " "), (kk, ee), fontsize=7,
                    xytext=(dx, dy), textcoords="offset points")
    ax.set_xlabel(r"thermal conductivity $\kappa$ (W m$^{-1}$K$^{-1}$)")
    ax.set_ylabel(r"Young's modulus $E$ (GPa)")
    ax.set_title("Thermal-mechanical trade-off:\n1 bulk reference + 16 porous architectures")
    ax.legend(loc="lower right")
    save(fig, "fig_pareto")


# ---- Fig 3: LOOCV parity --------------------------------------------------
def loocv_pred(X, y, log=False):
    pred = np.zeros(len(y))
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        sc = StandardScaler().fit(X[tr])
        gp = make_gp()
        yt = np.log(y[tr]) if log else y[tr]
        gp.fit(sc.transform(X[tr]), yt)
        p = gp.predict(sc.transform(X[te]))
        pred[te] = np.exp(p) if log else p
    return pred


def fig_parity(d):
    X = d[FEATURES].values
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, tgt, log, unit, col in [
        (axes[0], "kappa", True,  r"$\kappa$ (W m$^{-1}$K$^{-1}$)", TEAL),
        (axes[1], "E",     False, r"$E$ (GPa)", CORAL)]:
        y = d[tgt].values
        p = loocv_pred(X, y, log=log)
        mae = np.mean(np.abs(p - y))
        r2 = 1 - np.sum((p - y)**2) / np.sum((y - y.mean())**2)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        pad = 0.06 * (hi - lo)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", c="k", lw=1, alpha=0.6)
        ax.scatter(y, p, s=55, c=col, edgecolor="k", lw=0.5, zorder=3)
        ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(f"MD {unit}"); ax.set_ylabel(f"GP-LOOCV predicted {unit}")
        ax.set_title(f"{'GP(log)' if log else 'GP'} : MAE={mae:.2f}, "
                     rf"$R^2$={r2:.2f}")
    fig.suptitle("Leave-one-out forward-model parity", y=1.02)
    save(fig, "fig_parity")


# ---- Fig 4: permutation importance ----------------------------------------
def fig_importance(d):
    X = d[FEATURES].values
    imp = {}
    for tgt in ["kappa", "E"]:
        rf = RandomForestRegressor(n_estimators=300, random_state=RS).fit(X, d[tgt].values)
        r = permutation_importance(rf, X, d[tgt].values, n_repeats=30, random_state=RS)
        tot = r.importances_mean.sum()
        imp[tgt] = (100*r.importances_mean/tot, 100*r.importances_std/tot)
        print(f"   {tgt} normalised %: " +
              ", ".join(f"{f}={v:.1f}" for f, v in zip(FEATURES, 100*r.importances_mean/tot)))
    order = np.argsort(imp["kappa"][0])
    feats = [FEATURES[i] for i in order]
    yk = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    h = 0.38
    ax.barh(yk + h/2, imp["kappa"][0][order], height=h, xerr=imp["kappa"][1][order],
            color=TEAL, edgecolor="k", lw=0.5, capsize=3, label=r"$\kappa$")
    ax.barh(yk - h/2, imp["E"][0][order], height=h, xerr=imp["E"][1][order],
            color=CORAL, edgecolor="k", lw=0.5, capsize=3, label=r"$E$")
    ax.set_yticks(yk); ax.set_yticklabels([f.replace("_uc", " width").replace("phi", "porosity") for f in feats])
    ax.set_xlabel("permutation importance (% of target total)")
    ax.set_title("Feature importance: neck = thermal knob, AR = mechanical knob")
    ax.legend()
    save(fig, "fig_importance")


# ---- Fig 5: inverse-design selection (AR<->phi coupling) -------------------
def fig_inverse_selection(d):
    # geometric facts: C_d3 stagger, integer footprints -> area -> atom-counted phi
    AR   = [1.0, 2.0, 2.5]
    area = [9, 8, 10]            # 3x3, 4x2, 5x2
    phi  = [17.68, 16.07, 19.26] # atom-counted (built)
    # GP kappa at TRUE phi (build-aware) and at HELD phi (=17.68 for all)
    X = d[FEATURES].values
    sc = StandardScaler().fit(X)
    gk = make_gp(); gk.fit(sc.transform(X), np.log(d["kappa"].values))
    def kap(phi_, ar): return float(np.exp(gk.predict(sc.transform([[phi_, 3, 37.5, 2, ar]]))[0]))
    k_true = [kap(p, a) for p, a in zip(phi, AR)]
    k_held = [kap(17.68, a) for a in AR]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.3))
    # panel a: phi vs AR (coupling)
    a1.plot(AR, phi, "o-", c=TEAL, lw=1.5, ms=8, mec="k")
    for x, y, ar_area in zip(AR, phi, area):
        dxy = {9: (8, -4), 8: (8, -4), 10: (-10, -18)}[ar_area]
        ha = "right" if ar_area == 10 else "left"
        a1.annotate(f"{['3x3','4x2','5x2'][area.index(ar_area)]}\n(area {ar_area})",
                    (x, y), fontsize=8, xytext=dxy, textcoords="offset points", ha=ha)
    a1.set_xlabel("aspect ratio AR"); a1.set_ylabel(r"atom-counted $\phi$ (%)")
    a1.margins(x=0.16, y=0.20)
    a1.set_title("(a) AR is not area-preserving on the lattice")
    # panel b: predicted kappa vs AR, held vs true phi, target line
    a2.plot(AR, k_held, "s--", c=GRAY, lw=1.5, ms=8, mec="k", label=r"held $\phi$=17.7%")
    a2.plot(AR, k_true, "o-", c=CORAL, lw=1.6, ms=9, mec="k", label=r"true (built) $\phi$")
    a2.axhline(TARGET["kappa"], color=AMBER, ls=":", lw=1.5, label=r"target $\kappa$=2.4")
    a2.annotate("held-$\\phi$ pick", (2.0, k_held[1]), fontsize=8, color="#6a6a63",
                xytext=(0, 14), textcoords="offset points", ha="center", fontweight="bold")
    a2.annotate("build-aware pick", (2.5, k_true[2]), fontsize=8, color=CORAL,
                xytext=(-12, 10), textcoords="offset points", ha="right", fontweight="bold")
    a2.set_xlabel("aspect ratio AR"); a2.set_ylabel(r"GP-predicted $\kappa$ (W m$^{-1}$K$^{-1}$)")
    a2.margins(x=0.16, y=0.18)
    a2.set_title("(b) True porosity reverses the selection")
    a2.legend(loc="center left", fontsize=8, framealpha=0.9)
    save(fig, "fig_inverse_selection")


# ---- Fig 6: INV1 validation -----------------------------------------------
def fig_validation(d):
    # neighbors in the design region
    reg = d[(d["kappa"] < 3.6) & (d["E"] > 45)]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(reg["kappa"], reg["E"], s=55, c=TEAL, edgecolor="k", lw=0.5,
               zorder=2, label="existing geometries")
    for lbl, kk, ee in zip(reg["label"], reg["kappa"], reg["E"]):
        ax.annotate(lbl.replace("_", " "), (kk, ee), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points", color="#555")
    # target crosshair
    ax.errorbar(TARGET["kappa"], TARGET["E"], fmt="+", ms=16, mew=2, c=AMBER, zorder=3)
    ax.annotate("target (2.4, 70)", (TARGET["kappa"], TARGET["E"]), color=AMBER,
                fontsize=9, xytext=(0, 14), textcoords="offset points", ha="center")
    # INV1 measured w/ error bars
    ax.errorbar(INV1["kappa"], INV1["E"], xerr=INV1["kappa_std"], yerr=INV1["E_std"],
                fmt="D", ms=11, c=CORAL, mec="k", mew=0.6, capsize=4, zorder=4,
                label="INV1 (MD-measured)")
    ax.annotate("INV1 measured\n(2.36, 68.4)", (INV1["kappa"], INV1["E"]), color=CORAL,
                fontsize=9.5, xytext=(0, -32), textcoords="offset points", ha="center")
    ax.set_xlabel(r"thermal conductivity $\kappa$ (W m$^{-1}$K$^{-1}$)")
    ax.set_ylabel(r"Young's modulus $E$ (GPa)")
    ax.set_title("Inverse-design validation: INV1 vs target and neighbors")
    ax.legend(loc="lower right")
    save(fig, "fig_validation")


def main():
    print(f"Reading {DATA} ...")
    d = load()
    print(f"  {len(d)} geometries. Building figures:")
    fig_pareto(d)
    fig_parity(d)
    fig_importance(d)
    fig_inverse_selection(d)
    fig_validation(d)
    print("Done. Fig 1 (structure schematic) is a separate OVITO render.")


if __name__ == "__main__":
    main()
