# Machine-learning-guided multi-objective inverse design of porous silicon

Code and data for a study that treats thermal conductivity (κ) and
Young's modulus (E) as coupled objectives for porous silicon, maps the κ–E
Pareto front over 17 geometries, and uses a build-aware inverse search to
design a new structure at a targeted point on that plane.

**Headline result.** Targeting κ = 2.4 W m⁻¹ K⁻¹ and E = 70 GPa — a point no
measured geometry occupied — the search proposed a novel architecture
(**INV1**: 37.5 % stagger, aspect ratio 2.5). Independent molecular dynamics
returned **κ = 2.36 ± 0.18 W m⁻¹ K⁻¹** and **E = 68.4 ± 0.15 GPa**. INV1 was
excluded from all model fitting, so this is a held-out test of the design
procedure rather than of a fit.

A manuscript describing this work, by Othman Soubai and Younes Abouelhanoune
(LSA Laboratory, ENSAH, Abdelmalek Essaadi University, Al-Hoceima, Morocco), is
under review. This README will be updated with the full reference on
publication.

---

## What is here

```
code/        pipeline, structure builders, analysis and figure scripts
data/        the coupled dataset and the held-out validation point
lammps/      LAMMPS input decks and the Stillinger–Weber parameter file
slurm/       job scripts as run on the MARWAN HPC cluster
structures/  the designed geometry INV1 as a LAMMPS data file
results_thermal/  raw NEMD output for INV1 (three seeds)
```

### `data/`

| file | contents |
|---|---|
| `paper3_multiobj_dataset.csv` | the 17-geometry training set: descriptors, κ, E |
| `inv1_validation_point.csv` | **held out.** INV1's measured κ, E, ν — not training data |
| `paper3_pareto_front.csv` | the nine non-dominated geometries |
| `elastic_dataset.csv` | per-geometry E, ν, transverse asymmetry (3-seed means) |
| `elastic_perseed.csv` | the same, per seed, before averaging |
| `jobs_phase3.csv` | job manifest with atom-counted porosity per structure |

Porosity throughout is **atom-counted** from the constructed configuration,
not taken from the nominal footprint area.

### `results_thermal/`

Raw reverse-NEMD output for the inverse-designed geometry INV1, three
independent velocity seeds, at **two profile resolutions**. In each,
`temp_profile_*.dat` holds the steady-state temperature profile along the
transport axis and `e_transfer_*.dat` the cumulative kinetic energy exchanged
by the swap algorithm.

| files | profile | reduce with | result |
|---|---|---|---|
| `*_seed*.dat` | four bins, as used for the whole dataset | `code/analyze_kappa_inv1.py` | κ = 2.36 ± 0.18 W m⁻¹ K⁻¹ |
| `*_20bin_seed*.dat` | twenty bins, the intended resolution | `code/analyze_kappa_inv1_20bin.py` | κ = 2.323 ± 0.017 W m⁻¹ K⁻¹ |

The four-bin set is what the manuscript reports, because it is the reduction
applied to every geometry and used to train the models, and so the only value
commensurable with the prediction. The twenty-bin set is a re-derivation of the
same geometry with the binning corrected, run to test whether the coarse
profile biased the result: it does not — the two agree to within 0.38 standard
errors of the seed means. The finer profile also allows the linearity of the
temperature gradient to be checked, which two points per branch cannot: the
per-branch fits return R² = 0.989–0.995, so the profile is linear to better
than 1 % with no detectable curvature near the thermostatted slabs.

Both reductions can therefore be run from the shipped data, and the headline
thermal result derived rather than taken on trust.

Raw output for the seventeen training geometries is not included; it is
available from the corresponding author on request.

### `code/`

| script | role |
|---|---|
| `build_structures.py` | builds the 18 parametric structures from pore footprints |
| `make_inv1.py` | builds INV1 (reuses the carve logic above) |
| `paper3_multiobj_ml.py` | merge → RF/GP forward models → Pareto front → first-pass inverse search |
| `build_aware_search.py` | second pass: builds each candidate, counts its atoms, re-ranks at true porosity |
| `analyze_kappa.py` | NEMD reduction: κ from the temperature profile and swap energy |
| `analyze_kappa_inv1.py` | the same reduction applied to INV1 (four-bin profile) |
| `analyze_kappa_inv1_20bin.py` | INV1 at the corrected twenty-bin resolution, with per-branch fit quality |
| `analyze_elastic.py` | E, ν and transverse asymmetry from stress–strain data |
| `analyze_E_inv1.py` | the same fit applied to INV1 |
| `make_paper3_figures.py` | Figures 2–6 |
| `render_structures.py` | Figure 1 (structure cross-sections) |
| `make_inv1_planview.py` | Supplementary Figure S1 (INV1 plan view) |

---

## Reproducing the work

Requires Python 3.11 with numpy, pandas, scikit-learn and matplotlib, and
LAMMPS with the MANYBODY package for the molecular dynamics.

**1. Build the structures.** Only the designed geometry INV1 is shipped, in
`structures/`, since it is the result the study reports. Every other structure
regenerates exactly — the carve is deterministic — as do all figures.

```bash
python3 code/build_structures.py     # the 18 parametric geometries
python3 code/make_inv1.py            # INV1 (19,843 of 24,576 atoms, φ = 19.26 %)
```

**2. Run the molecular dynamics** (HPC; see `slurm/` for the job scripts as
submitted). Thermal: reverse NEMD after Müller-Plathe, 500 ps production,
3 seeds. Mechanical: uniaxial tension along *x* at 10⁻⁴ ps⁻¹, 3 seeds.

**3. Reduce and model.**

```bash
python3 code/analyze_kappa.py            # κ per geometry
python3 code/analyze_kappa_inv1.py       # κ for INV1, from the shipped raw output
python3 code/analyze_kappa_inv1_20bin.py # the same at twenty-bin resolution
python3 code/analyze_elastic.py          # E, ν per geometry
python3 code/paper3_multiobj_ml.py       # models, Pareto front, first-pass search
python3 code/build_aware_search.py       # build-aware correction; final candidate
```

The two search steps disagree by design, and that disagreement is the
methodological point of the study. The first ranks aspect-ratio candidates at
the parent tiling's porosity and selects AR = 2.0. The second constructs each
candidate, counts its atoms, and re-ranks at the porosity the structure
actually has — which reverses the selection to AR = 2.5, the geometry that was
then built and measured. Porosity is a dependent output of the tiling, not a
free variable.

**4. Generate the figures.** These are not shipped; the scripts below produce
them from the data in this repository.

```bash
python3 code/make_paper3_figures.py  # Figures 2–6
python3 code/render_structures.py    # Figure 1
python3 code/make_inv1_planview.py   # Figure S1
```

Scripts use fixed random seeds (42), so model outputs are reproducible.
Molecular-dynamics results carry seed-to-seed scatter; the reported values are
means over three seeds with standard deviations.

---

## Notes for anyone reusing the data

**INV1 is held out.** It was designed from models fitted to the 17-geometry set
and only then simulated. Folding it into that set before quoting the
cross-validation metrics would make the reported agreement circular. It is kept
in a separate file for this reason.

**Two descriptors are nominal for some rows.** Pore size *S* and neck width are
defined on the square parent tiling. Where the aspect ratio departs from unity
the footprint is not square, and those columns then describe the parent rather
than the realised structure — this affects the three aspect-ratio geometries and
INV1. Aspect ratio and atom-counted porosity always describe the structure as
built. The neck convention (box width less pore size less offset) also fails to
reproduce the recorded value for the two maximally staggered geometries. The
manuscript discusses this and proposes a configuration-based alternative.

**Absolute conductivities are size-limited.** The cell is 48 unit cells
(≈ 26 nm) along the transport direction, well below silicon's room-temperature
phonon mean free paths, so absolute κ should not be compared with experiment.
Comparisons between architectures at fixed cell size are unaffected, and those
carry the conclusions.

**Potential.** Stillinger–Weber, from the `Si.sw` file distributed with LAMMPS
(ε = 2.1683 eV). Benchmarks in the manuscript are against the potential's own
elastic constants, not against measured silicon.

---

## Citation

Please cite the manuscript once published. Until then, cite this repository.

## Contact

Othman Soubai — othman.soubai@etu.uae.ac.ma
LSA Laboratory, ENSAH, Abdelmalek Essaadi University, Al-Hoceima, Morocco

## Licence

Code released under the MIT Licence; data under CC BY 4.0.
