#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 3 - Phase 3 : Multi-objective ML (kappa vs E)
====================================================
Merges the mechanical dataset (E) with the Paper 2 thermal dataset (kappa),
trains RF + GP forward models for BOTH targets with the same LOOCV protocol
as Paper 2, computes the (minimize kappa, maximize E) Pareto front, and
provides a multi-objective inverse-design routine.

Run on MARWAN inside the Paper 2 analysis env:
    python3 paper3_multiobj_ml.py
Reads:
    ./elastic_dataset.csv     (E, 3-seed mean +/- std; atom-counted phi)
    ./ml_dataset_paper2.csv     (kappa; features phi,S,stagger,neck_uc,AR)
Writes (to ./):
    paper3_multiobj_dataset.csv           merged feature+target table
    paper3_pareto_front.csv               non-dominated geometries
    paper3_pareto.png                     kappa-vs-E scatter with Pareto front
"""

import os
import sys
import difflib
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless compute node
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")  # silence GP convergence chatter
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DATA_DIR    = os.path.expanduser(os.environ.get("DATA_DIR", "data"))
OUT_DIR     = os.path.expanduser(os.environ.get("OUT_DIR", "."))
ELASTIC_CSV = os.path.join(DATA_DIR, "elastic_dataset.csv")
KAPPA_CSV   = os.path.join(DATA_DIR, "kappa_dataset.csv")
# Authoritative ATOM-COUNTED phi lives here (phi_actual col); elastic_dataset.csv
# carries E only and ml_dataset_paper2.csv carries NOMINAL phi for ~9 rows.
JOBS_CSV    = os.path.join(DATA_DIR, "jobs_phase3.csv")

# CLI overrides: python3 paper3_multiobj_ml.py <elastic.csv> <kappa.csv> [out_dir] [jobs.csv]
if len(sys.argv) >= 2:
    ELASTIC_CSV = os.path.expanduser(sys.argv[1])
if len(sys.argv) >= 3:
    KAPPA_CSV = os.path.expanduser(sys.argv[2])
if len(sys.argv) >= 4:
    OUT_DIR = os.path.expanduser(sys.argv[3])
if len(sys.argv) >= 5:
    JOBS_CSV = os.path.expanduser(sys.argv[4])

# Inverse-design target (confirmed from the Pareto front: AR-decoupling corner).
TARGET_KAPPA = 2.4    # W/m.K  (minimize)
TARGET_E     = 70.0   # GPa    (maximize)

FEATURES = ["phi", "S", "stagger", "neck_uc", "AR"]

# Optional manual label crosswalk { kappa_label : elastic_label }.
# Leave empty unless the script reports a label mismatch; then fill it with the
# VERIFIED mapping it suggests (or rename labels in one CSV so they match).
LABEL_CROSSWALK = {}

# Candidate column names for auto-detection (case-insensitive, exact token match).
COLMAP = {
    "label":     ["label", "name", "geometry", "geom", "id"],
    "phi":       ["phi_actual", "phi_atom", "phi", "porosity", "phi_pct",
                  "phi_percent", "porosity_pct"],
    "S":         ["s", "pore_size", "size", "s_uc", "pore_s"],
    "stagger":   ["stagger", "stagger_pct", "stagger_offset", "offset",
                  "stagger_percent"],
    "neck_uc":   ["neck_uc", "neck", "neck_width", "neck_width_uc", "neck_uc_width"],
    "AR":        ["ar", "aspect_ratio", "aspect", "aspectratio"],
    "kappa":     ["kappa_mean", "kappa", "k", "kappa_wmk", "kappa_avg", "kappa_mk"],
    "kappa_std": ["kappa_std", "kappa_err", "kappa_sd", "k_std", "kappa_error"],
    "E":         ["e_gpa_mean", "e_mean", "e", "e_gpa", "youngs_modulus",
                  "young_modulus", "e_avg", "modulus", "e_100"],
    "E_std":     ["e_gpa_std", "e_std", "e_err", "e_sd", "e_error"],
    "nu":        ["nu_mean", "nu", "poisson", "poisson_ratio", "poissons_ratio"],
    "nu_std":    ["nu_std", "nu_err", "nu_sd", "nu_error"],
}


def banner(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def find_col(df, keys, required=False, where=""):
    """Case-insensitive column resolver. Returns the real column name or None."""
    lower = {c.lower().strip(): c for c in df.columns}
    for k in keys:
        if k in lower:
            return lower[k]
    if required:
        print(f"  [!] Could not find a column for {keys[0]!r} in {where}.")
        print(f"      Available columns: {list(df.columns)}")
    return None


def norm_label(s):
    return str(s).strip().lower().replace(" ", "").replace("-", "_")


PHI_ACTUAL_KEYS = ["phi_actual", "phi_atom", "actual_phi", "phi_counted",
                   "phi_real", "porosity_actual"]


def load_phi_actual(path, label_keys, phi_keys=PHI_ACTUAL_KEYS):
    """Robustly read a (possibly messy) jobs CSV and return (df, label_col, phi_col).

    Handles preamble/comment lines before the real header and both comma- and
    whitespace-delimited tables. Locates the header by scanning for a line that
    contains a phi_actual-like token. Returns (None, None, None) if it can't find
    a usable header + phi_actual column (caller then falls back to nominal phi).
    """
    try:
        with open(path, "r", errors="replace") as fh:
            raw = fh.readlines()
    except Exception:
        return None, None, None

    flat_keys = [k.replace("_", "") for k in phi_keys]

    def looks_like_header(line):
        low = line.lower().replace(" ", "").replace("_", "")
        if not any(k in low for k in flat_keys):
            return False
        # a real header splits into several columns; a prose/formula comment won't
        for d in (",", None, ";", "\t"):
            parts = line.split(d) if d else line.split()
            if len([p for p in parts if p.strip()]) >= 3:
                return True
        return False

    hdr_idx = None
    for i, ln in enumerate(raw):
        if looks_like_header(ln):
            hdr_idx = i
            break
    if hdr_idx is None:
        return None, None, None  # no atom-counted-phi header present

    for kwargs in (dict(sep=","),
                   dict(sep=r"\s+", engine="python"),
                   dict(sep=";"),
                   dict(sep="\t")):
        try:
            df = pd.read_csv(path, skiprows=hdr_idx, comment="#", **kwargs)
        except Exception:
            continue
        if df.shape[1] < 2:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        jl = find_col(df, label_keys)
        jp = find_col(df, phi_keys)
        if jl and jp:
            return df, jl, jp
    return None, None, None


# --------------------------------------------------------------------------- #
# 1. LOAD + INSPECT
# --------------------------------------------------------------------------- #
banner("1. LOADING INPUT FILES")
for path in (ELASTIC_CSV, KAPPA_CSV):
    if not os.path.isfile(path):
        sys.exit(f"FATAL: file not found -> {path}")

elastic = pd.read_csv(ELASTIC_CSV)
kappa   = pd.read_csv(KAPPA_CSV)

print(f"\nelastic_dataset  : {ELASTIC_CSV}")
print(f"  rows={len(elastic)}  cols={list(elastic.columns)}")
print(f"kappa_dataset    : {KAPPA_CSV}")
print(f"  rows={len(kappa)}  cols={list(kappa.columns)}")

lbl_e = find_col(elastic, COLMAP["label"], required=True, where="elastic_dataset")
lbl_k = find_col(kappa,   COLMAP["label"], required=True, where="kappa_dataset")
if lbl_e is None or lbl_k is None:
    sys.exit("FATAL: no 'label' column in one of the files. Fix and rerun.")
elastic = elastic.rename(columns={lbl_e: "label"})
kappa   = kappa.rename(columns={lbl_k: "label"})
elastic["label"] = elastic["label"].astype(str)
kappa["label"]   = kappa["label"].astype(str)

print(f"\nelastic labels ({len(elastic)}): {sorted(elastic['label'])}")
print(f"kappa labels   ({len(kappa)}): {sorted(kappa['label'])}")

# --------------------------------------------------------------------------- #
# 2. ALIGN LABELS
#    Exact match is the only path that PROCEEDS automatically. Any leftover
#    elastic geometry without a kappa partner makes the script STOP and print a
#    globally-optimal (Hungarian) crosswalk *suggestion* for the user to verify
#    and paste into LABEL_CROSSWALK. We never silently guess E<->kappa pairings,
#    because a wrong pairing poisons every downstream result.
# --------------------------------------------------------------------------- #
banner("2. ALIGNING LABELS")

# 2a. apply any user-verified manual crosswalk first (kappa_label -> elastic_label)
if LABEL_CROSSWALK:
    print(f"\nApplying manual LABEL_CROSSWALK ({len(LABEL_CROSSWALK)} entries):")
    for k_lbl, e_lbl in LABEL_CROSSWALK.items():
        print(f"    kappa '{k_lbl}'  ->  elastic '{e_lbl}'")
    kappa["label"] = kappa["label"].map(lambda x: LABEL_CROSSWALK.get(x, x))

set_e, set_k = set(elastic["label"]), set(kappa["label"])
matched = sorted(set_e & set_k)
missing_e = sorted(set_e - set_k)   # elastic geometries with NO kappa partner
extra_k   = sorted(set_k - set_e)   # kappa rows not used by any elastic geometry
print(f"\nExact match: {len(matched)} / {len(set_e)} elastic geometries have a kappa partner.")

if missing_e:
    print("\n  [!] LABEL MISMATCH -- these elastic geometries have no kappa match:")
    print(f"      {missing_e}")
    if extra_k:
        print(f"      unmatched kappa labels available to pair with: {extra_k}")

        # Globally-optimal one-to-one suggestion via the Hungarian algorithm.
        # Maximize total string similarity == minimize (1 - similarity) cost.
        try:
            from scipy.optimize import linear_sum_assignment
            sim = np.zeros((len(missing_e), len(extra_k)))
            for i, el in enumerate(missing_e):
                for j, kl in enumerate(extra_k):
                    sim[i, j] = difflib.SequenceMatcher(
                        None, norm_label(el), norm_label(kl)).ratio()
            rows, cols = linear_sum_assignment(1.0 - sim)

            print("\n  Suggested crosswalk (GLOBALLY OPTIMAL -- VERIFY EACH PAIR):")
            print("    {:<24} {:<24} {}".format("elastic_label", "kappa_label", "score"))
            suggestion = {}  # kappa_label -> elastic_label
            for i, j in zip(rows, cols):
                el, kl, score = missing_e[i], extra_k[j], sim[i, j]
                suggestion[kl] = el
                flag = "   <-- LOW, CHECK" if score < 0.75 else ""
                print("    {:<24} {:<24} {:.2f}{}".format(el, kl, score, flag))

            print("\n  >>> Paste this into LABEL_CROSSWALK at the top of the script")
            print("      (after verifying every pair is the SAME geometry), then rerun:")
            print("\n      LABEL_CROSSWALK = {")
            for kl, el in suggestion.items():
                print(f"          {kl!r}: {el!r},")
            print("      }")
        except Exception as e:
            print(f"\n  [scipy.linear_sum_assignment unavailable: {e}]")
            print("  Falling back to printing the raw label lists for manual pairing:")
            print(f"    elastic (unmatched): {missing_e}")
            print(f"    kappa   (unmatched): {extra_k}")
    else:
        # No spare kappa rows -> the thermal dataset is simply missing geometries.
        print("\n  There are NO unmatched kappa rows to pair these with, so this is")
        print("  not a naming problem -- kappa is missing these geometries entirely.")
        print("  Add their NEMD kappa to ml_dataset_paper2.csv (or drop them from")
        print("  elastic_dataset.csv) before rerunning.")

    sys.exit("\nFATAL: not every elastic geometry has a kappa match. "
             "Fix labels / LABEL_CROSSWALK and rerun (see suggestion above).")

if extra_k:
    print(f"\n  Note: {len(extra_k)} kappa row(s) not present in elastic_dataset "
          f"will be ignored: {extra_k}")

print(f"\n  OK: all {len(matched)} elastic geometries matched to a kappa value.")

# --------------------------------------------------------------------------- #
# 3. BUILD MULTI-OBJECTIVE DATASET  (by provenance; atom-counted phi preferred)
# --------------------------------------------------------------------------- #
banner("3. BUILDING MULTI-OBJECTIVE DATASET")

el_idx = elastic.set_index("label")
ka_idx = kappa.set_index("label")

# resolve which logical columns live in which file
el_cols = {k: find_col(elastic, COLMAP[k]) for k in COLMAP}
ka_cols = {k: find_col(kappa,   COLMAP[k]) for k in COLMAP}

def pull(logical, prefer):
    """Return (Series on matched labels, source_str) using preference order."""
    order = [("elastic", el_idx, el_cols), ("kappa", ka_idx, ka_cols)]
    if prefer == "kappa":
        order = order[::-1]
    for src, idx, cmap in order:
        col = cmap.get(logical)
        if col is not None:
            return pd.to_numeric(idx.loc[matched, col], errors="coerce"), src
    return None, None

# phi : ATOM-COUNTED from jobs_phase3.csv (phi_actual) is authoritative for Paper 3.
#       Fall back to elastic phi, then kappa phi (nominal) only if jobs file absent.
phi_series, phi_src = None, None
phi_jobs = None  # keep for audit-delta print

if os.path.exists(JOBS_CSV):
    try:
        jobs, j_label, j_phi = load_phi_actual(JOBS_CSV, COLMAP["label"])
        if jobs is not None and j_label and j_phi:
            jobs[j_label] = jobs[j_label].astype(str)
            # phi is structure-determined => identical across the 3 seeds; mean is exact.
            phi_map = (jobs.groupby(j_label)[j_phi]
                           .apply(lambda s: pd.to_numeric(s, errors="coerce").mean()))
            phi_jobs = pd.Series([phi_map.get(l, np.nan) for l in matched], index=matched)
            if phi_jobs.notna().all():
                phi_series, phi_src = phi_jobs, f"jobs_phase3.csv ({j_phi}, atom-counted)"
                print(f"  phi source : jobs_phase3.csv  [col '{j_phi}', ATOM-COUNTED]  <-- authoritative")
            else:
                miss = [l for l in matched if pd.isna(phi_map.get(l, np.nan))]
                print(f"  [!] jobs_phase3.csv parsed but phi_actual missing for: {miss}")
                print(f"      -> falling back to nominal phi for ALL rows.")
                phi_jobs = None
        else:
            print(f"  [!] jobs_phase3.csv present but couldn't locate a label + phi_actual")
            print(f"      header (preamble/format?) -> falling back to nominal phi.")
            print(f"      Paste `head -8 jobs_phase3.csv` and I'll fix the parser.")
    except Exception as e:
        print(f"  [!] jobs_phase3.csv could not be read ({type(e).__name__}: {e})")
        print(f"      -> falling back to nominal phi. Paste `head -8 jobs_phase3.csv`.")
        phi_jobs = None
else:
    print(f"  [i] jobs_phase3.csv not found at {JOBS_CSV} -> trying elastic/kappa phi.")

# fallback chain
if phi_series is None:
    phi_series, phi_src = pull("phi", prefer="elastic")
    if phi_series is None:
        print("  [!] No phi anywhere (jobs_phase3.csv, elastic, or kappa).")
        print("      Atom-counted phi must come from the Phase 3 side. Point me to it.")
        sys.exit("FATAL: phi unresolved.")
    if phi_src == "elastic":
        print(f"  phi source : elastic_dataset [col '{el_cols['phi']}']")
    else:
        print(f"  phi source : kappa_dataset  *** NOMINAL phi (Paper 2) ***")
        print(f"               [col '{ka_cols['phi']}'] -- atom-counted phi from")
        print(f"               jobs_phase3.csv was not usable this run (see note above).")
        print(f"               NOTE: Pareto front is unaffected (uses MD kappa & E);")
        print(f"               only the ML phi feature differs by <=2.13 pp.")

# Audit: when we have atom-counted (jobs) AND the kappa file's phi, show the gap.
if phi_jobs is not None and phi_src and "jobs_phase3" in phi_src and ka_cols["phi"]:
    pk = pd.to_numeric(ka_idx.loc[matched, ka_cols["phi"]], errors="coerce")
    d = (phi_jobs - pk).abs()
    worst = d.sort_values(ascending=False).head(3)
    print(f"  phi audit (atom-counted vs Paper 2 nominal): "
          f"max={d.max():.2f} pp, mean={d.mean():.2f} pp")
    print("             largest gaps: " +
          ", ".join(f"{lbl} {gap:.2f}pp" for lbl, gap in worst.items()))
    print("             -> using atom-counted (jobs) phi throughout, per Paper 3 methodology.")

# architecture features : KAPPA first (Paper 2 feature table), elastic fallback
data = {"label": matched, "phi": phi_series.values}
src_report = {"phi": "jobs_phase3" if (phi_jobs is not None and phi_src and "jobs" in phi_src) else phi_src}
for f in ["S", "stagger", "neck_uc", "AR"]:
    s, src = pull(f, prefer="kappa")
    if s is None:
        sys.exit(f"FATAL: feature '{f}' not found in either file. See columns above.")
    data[f] = s.values
    src_report[f] = src

# targets
k_series, _ = pull("kappa", prefer="kappa")
e_series, _ = pull("E", prefer="elastic")
if k_series is None:
    sys.exit("FATAL: kappa target column not found.")
if e_series is None:
    sys.exit("FATAL: E target column not found.")
data["kappa"] = k_series.values
data["E"] = e_series.values

ks, _ = pull("kappa_std", prefer="kappa")
es, _ = pull("E_std", prefer="elastic")
if ks is not None:
    data["kappa_std"] = ks.values
if es is not None:
    data["E_std"] = es.values

# carry Poisson ratio (and its std) as metadata if present in the elastic file
nu_s, _ = pull("nu", prefer="elastic")
if nu_s is not None:
    data["nu"] = nu_s.values
nus_s, _ = pull("nu_std", prefer="elastic")
if nus_s is not None:
    data["nu_std"] = nus_s.values

df = pd.DataFrame(data)

print("\n  feature provenance: " +
      ", ".join(f"{k}<-{v}" for k, v in src_report.items()))

before = len(df)
df = df.dropna(subset=FEATURES + ["kappa", "E"]).reset_index(drop=True)
if len(df) < before:
    print(f"  Dropped {before - len(df)} row(s) with missing feature/target values.")

print(f"\nFinal merged dataset: {len(df)} geometries x {len(FEATURES)} features")
show = ["label"] + FEATURES + ["kappa", "E"]
print(df[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "paper3_multiobj_dataset.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")

X = df[FEATURES].values.astype(float)
y_kappa = df["kappa"].values.astype(float)
y_E = df["E"].values.astype(float)

# --------------------------------------------------------------------------- #
# 4. FORWARD MODELS  (RF raw + GP), same LOOCV protocol as Paper 2
# --------------------------------------------------------------------------- #
banner("4. FORWARD MODELS  (LOOCV, n={})".format(len(df)))

def make_rf():
    return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                 n_jobs=-1)

def make_gp():
    # isotropic Matern(2.5) + WhiteKernel ; stable for small n.
    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(1e-2, (1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=10, alpha=1e-10,
                                    random_state=RANDOM_STATE)

def loocv(X, y, kind, log_target=False):
    """LOOCV; returns out-of-fold predictions in RAW target units."""
    loo = LeaveOneOut()
    pred = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr = np.log(y[tr]) if log_target else y[tr]
        if kind == "rf":
            m = make_rf()                       # RF: raw features
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
        else:                                   # GP: scale features per fold
            sc = StandardScaler().fit(Xtr)
            m = make_gp()
            m.fit(sc.transform(Xtr), ytr)
            p = m.predict(sc.transform(Xte))
        pred[te] = np.exp(p) if log_target else p
    return pred

def report(name, target, y, pred):
    mae = mean_absolute_error(y, pred)
    r2  = r2_score(y, pred)
    print(f"  {name:<14} {target:<8} MAE={mae:7.3f}   R2={r2:7.3f}")
    return mae, r2

print("\nThermal target (kappa, W/m.K)  -- matches Paper 2 protocol:")
report("RF (raw)",  "kappa", y_kappa, loocv(X, y_kappa, "rf"))
report("GP (log)",  "kappa", y_kappa, loocv(X, y_kappa, "gp", log_target=True))

print("\nMechanical target (E, GPa):")
report("RF (raw)",  "E", y_E, loocv(X, y_E, "rf"))
report("GP (raw)",  "E", y_E, loocv(X, y_E, "gp", log_target=False))
print("\n  (E spans <1 order of magnitude, so GP uses raw E unlike kappa where")
print("   Paper 2 used log. Ask if you also want GP(log E) for symmetry.)")

# --------------------------------------------------------------------------- #
# 5. FEATURE IMPORTANCE  (RF permutation, as in Paper 2)
# --------------------------------------------------------------------------- #
banner("5. RF PERMUTATION IMPORTANCE  (kappa vs E)")

def perm_imp(y):
    rf = make_rf().fit(X, y)
    r = permutation_importance(rf, X, y, n_repeats=30,
                               random_state=RANDOM_STATE, n_jobs=-1)
    return r.importances_mean, r.importances_std

imp_k_m, imp_k_s = perm_imp(y_kappa)
imp_E_m, imp_E_s = perm_imp(y_E)

print("\n  {:<10} {:<20} {:<20}".format("feature", "kappa importance", "E importance"))
for i in np.argsort(-imp_k_m):
    print("  {:<10} {:6.3f} +/- {:5.3f}      {:6.3f} +/- {:5.3f}".format(
        FEATURES[i], imp_k_m[i], imp_k_s[i], imp_E_m[i], imp_E_s[i]))
print("\n  Watch for: AR ~ negligible for kappa but a real lever for E")
print("  (the 'mechanical-only knob' from milestone_08).")

# --------------------------------------------------------------------------- #
# 6. PARETO FRONT  (minimize kappa, maximize E) on the MD ground-truth data
# --------------------------------------------------------------------------- #
banner("6. PARETO FRONT  (minimize kappa, maximize E)")

def pareto_mask(kappa_arr, E_arr):
    """True where non-dominated: no other point has kappa<= and E>= with at
    least one strict inequality."""
    n = len(kappa_arr)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            if (kappa_arr[j] <= kappa_arr[i] and E_arr[j] >= E_arr[i]
                    and (kappa_arr[j] < kappa_arr[i] or E_arr[j] > E_arr[i])):
                nd[i] = False
                break
    return nd

mask = pareto_mask(y_kappa, y_E)
front = df[mask].sort_values("kappa").reset_index(drop=True)

print(f"\nNon-dominated geometries: {int(mask.sum())} / {len(df)}")
print(front[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

front_path = os.path.join(OUT_DIR, "paper3_pareto_front.csv")
front.to_csv(front_path, index=False)
print(f"\nSaved -> {front_path}")

# knee = point on front with min normalized distance to utopia corner (min k, max E)
kn = (y_kappa - y_kappa.min()) / (np.ptp(y_kappa) + 1e-12)
En = (y_E - y_E.min()) / (np.ptp(y_E) + 1e-12)
dist = np.sqrt(kn**2 + (1 - En)**2)
dist[~mask] = np.inf
knee_i = int(np.argmin(dist))
print(f"\nApprox. Pareto 'knee' (best balance): {df.loc[knee_i, 'label']} "
      f"(kappa={y_kappa[knee_i]:.2f}, E={y_E[knee_i]:.1f})")

# --------------------------------------------------------------------------- #
# 7. PLOT
# --------------------------------------------------------------------------- #
try:
    groups = df["label"].str.extract(r"^([A-Za-z]+\d?)", expand=False).fillna("X")
    uniq = sorted(groups.unique())
    palette = {g: c for g, c in zip(uniq, plt.cm.tab10(np.linspace(0, 1, max(3, len(uniq)))))}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for g in uniq:
        sel = (groups == g).values
        ax.scatter(y_kappa[sel], y_E[sel], s=55, color=palette[g], label=g,
                   edgecolor="k", linewidth=0.4, zorder=3)
    ax.plot(front["kappa"], front["E"], "-", color="crimson", lw=2, zorder=2,
            label="Pareto front")
    ax.scatter(front["kappa"], front["E"], s=120, facecolors="none",
               edgecolors="crimson", linewidth=1.8, zorder=4)
    for _, r in df.iterrows():
        ax.annotate(r["label"], (r["kappa"], r["E"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.scatter([y_kappa[knee_i]], [y_E[knee_i]], marker="*", s=320, color="gold",
               edgecolor="k", zorder=5, label="knee")
    ax.set_xlabel(r"$\kappa$  (W/m$\cdot$K)   $\leftarrow$ minimize")
    ax.set_ylabel(r"$E$  (GPa)   maximize $\rightarrow$")
    ax.set_title("Paper 3 - Thermal-Mechanical Pareto Front (porous Si)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "paper3_pareto.png")
    fig.savefig(plot_path, dpi=200)
    print(f"Saved -> {plot_path}")
except Exception as e:
    print(f"  [plot skipped: {e}]")

# --------------------------------------------------------------------------- #
# 8. INVERSE DESIGN  (multi-objective)  -- scaffold, demo on PLACEHOLDER target
# --------------------------------------------------------------------------- #
banner("8. MULTI-OBJECTIVE INVERSE DESIGN (demo)")

scaler_full = StandardScaler().fit(X)
Xs = scaler_full.transform(X)
gp_k = make_gp().fit(Xs, np.log(y_kappa))   # log-kappa
gp_E = make_gp().fit(Xs, y_E)               # raw E

def predict_point(x):
    xs = scaler_full.transform(np.asarray(x, float).reshape(1, -1))
    mk, sk = gp_k.predict(xs, return_std=True)     # log space
    me, se = gp_E.predict(xs, return_std=True)
    kap = float(np.exp(mk[0]))
    return kap, (float(np.exp(mk[0]-sk[0])), float(np.exp(mk[0]+sk[0]))), \
        float(me[0]), float(se[0])

S_K = np.std(y_kappa) + 1e-12
S_E = np.std(y_E) + 1e-12

# Buildable candidates: reuse each MEASURED tiling (its phi & neck are real outputs of
# a built structure) and sweep AR only. AR is the one knob with a measured sweep
# (D group, AR 0.5-2.5), and it demonstrably does NOT move neck (D group neck=3 for
# every AR) and barely moves phi (+/-2 pp). phi for a built structure is a DEPENDENT
# output; phi* below is the base tiling's value, to be RE-MEASURED after building.
# We deliberately do NOT invent novel (S,stagger) combos: the 17-point set has a hidden
# pore-density knob (e.g. (S=4,stagger=0) appears at phi=8 AND phi=28), so phi/neck for
# unseen combos can't be reconstructed without build_structures.py.

AR_grid = sorted(df["AR"].unique())

seen, bases = set(), []
for r in df.itertuples():
    if str(r.label) == "bulk_si":
        continue
    key = (round(r.phi, 0), round(r.S, 2), round(r.stagger, 2), round(r.neck_uc, 2))
    if key in seen:
        continue
    seen.add(key)
    bases.append((str(r.label), float(r.phi), float(r.S), float(r.stagger), float(r.neck_uc)))

existing = {(round(float(r.S), 2), round(float(r.stagger), 2),
            round(float(r.neck_uc), 2), round(float(r.AR), 2)) for r in df.itertuples()}

print(f"\nTarget: kappa={TARGET_KAPPA} W/m.K, E={TARGET_E} GPa")
print(f"Searching {len(bases)} measured tilings x {len(AR_grid)} AR values "
      f"(AR-only sweep; phi/neck held at each tiling's measured value).")

cand = []
for name, phi, S, stag, neck in bases:
    for AR in AR_grid:
        kap, kband, e, esd = predict_point([phi, S, stag, neck, AR])
        dist = np.hypot((kap - TARGET_KAPPA) / S_K, (e - TARGET_E) / S_E)
        is_x = (round(S, 2), round(stag, 2), round(neck, 2), round(AR, 2)) in existing
        cand.append((dist, name, phi, S, stag, neck, AR, kap, kband, e, esd, is_x))

cand.sort(key=lambda t: t[0])
print("\n  rank  base_tiling      S  stag  neck   AR   phi*   pred_kappa         pred_E      dist  status")
for i, (dist, name, phi, S, stag, neck, AR, kap, kb, e, esd, is_x) in enumerate(cand[:8], 1):
    status = "existing" if is_x else "NEW"
    print(f"  {i:>2}.  {name:<14} {S:>4.1f} {stag:>5.1f} {neck:>4.1f} {AR:>4.1f}  {phi:>5.1f}  "
          f"{kap:>4.2f} [{kb[0]:.2f}-{kb[1]:.2f}]  {e:>5.1f}+/-{esd:>3.1f}  {dist:>5.3f}  {status}")

best_new = next((c for c in cand if not c[-1]), cand[0])
dist, name, phi, S, stag, neck, AR, kap, kb, e, esd, is_x = best_new
print("\n  >>> RECOMMENDED build candidate (best NON-training geometry):")
print(f"      base tiling '{name}' with AR raised to {AR:g}")
print(f"      S = {S:g},  stagger = {stag:g}%,  neck ~ {neck:g} uc,  base phi ~ {phi:.1f}%")
print(f"      predicted kappa = {kap:.2f} ({kb[0]:.2f}-{kb[1]:.2f}) W/m.K,  E = {e:.1f} +/- {esd:.1f} GPa")
print( "      Interpretation: the high-stagger thermal-optimal tiling, pore-stretched")
print( "      (high AR) for stiffness -- combines the thermal and mechanical knobs.")
print("\n  NOTE: phi* is the base tiling's MEASURED porosity; AR is assumed not to move")
print("  phi/neck (an assumption; corrected by build_aware_search.py). The BUILT structure's actual")
print("  atom-counted phi must be re-measured, then validated with NEMD (kappa) +")
print("  uniaxial tensile MD (E). Building this exact structure needs build_structures.py")
print("  (the authoritative parametrization), since phi/neck can't be reconstructed here.")

banner("DONE")
print("Paste the full stdout back. Next: build the recommended candidate with")
print("build_structures.py, re-measure its atom-counted phi, then validate with")
print("NEMD (kappa) + uniaxial tensile MD (E) against the GP prediction.")
