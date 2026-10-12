# QSM and R2* differentiation of hydroxyapatite and calcium oxalate calcifications

Code accompanying *Quantitative Susceptibility Mapping for Differentiating Hydroxyapatite and Calcium Oxalate Breast Calcifications at 3T: A Phantom Study*, Klara Mišak, Enrico De Vita,  Chris A Clark,  Matt T Cashmore,  Simon Walker-Samuel. DOI: doi: https://doi.org/10.64898/2026.07.15.738683

This repository contains the analysis pipeline, the digital twin used to
validate it, and the derived results tables from which the figures and tables
in the paper are produced.

## What is and is not included

**Included:** all analysis code, and the derived per-tube results as CSV
files. Every number in the paper's tables can be traced to one of these CSVs,
and every CSV can be regenerated from the code given the source images.

**Not included:** the MRI data. The phantom images are not distributed. The
scripts that read images therefore cannot be run as they stand; the scripts
that read the CSVs can.

**External dependencies, separately licensed and not redistributed here:**

- [ROMEO](https://github.com/korbinian90/ROMEO) for phase unwrapping, called
  as an external binary.
- The MEDI toolbox from Cornell MRI Research Lab, for PDF and LBV background
  field removal, used from MATLAB.

Install both yourself and set the paths in the configuration block at the top
of each script.

## Requirements

Python 3.9 or later, with:

```
numpy
scipy
nibabel
matplotlib
```

MATLAB is needed only for the background field removal step.

## Running order

Every script has a configuration block at the top. Nothing below that block
hardcodes a path. Set `RESULTS_ROOT` and `RESULTS_SUFFIX` identically in
`qsm_pipeline.py`, `roi_analysis.py` and `r2star_analysis.py`, so that the
later scripts find what the pipeline wrote.

### Measured data

1. **`qsm_pipeline.py`** with `MODE = "prep"` — masking, ROMEO phase
   unwrapping, field map. Writes `in_<tag>.mat`.
2. **`run_medi_batch_PDF_LBV.m`** in MATLAB — background field removal by PDF
   and LBV. Writes `out_<tag>.mat`.
3. **`qsm_pipeline.py`** with `MODE = "invert"` — TV-regularised dipole
   inversion. Writes the susceptibility map.
4. Repeat 1 to 3 for each formulation and resolution.
5. **`roi_analysis.py`** — susceptibility per region of interest, with the
   detection criterion. Writes `roi_results_per_tube.csv`.
6. **`r2star_analysis.py`** — mono-exponential R2* fitting over the same
   regions. Writes `r2star_results.csv`, and
   `r2star_results_allechoes.csv` when run with `USE_ALL_ECHOES = True`.

### Digital twin

7. **`verify_twin.py`** — verification of the forward model against analytic
   solutions. Self-contained: no data, no paths, runs anywhere. Run this
   first if you want to confirm the simulation is sound before trusting
   anything built on it.
8. **`twin_recovery.py`** — recovery of a known susceptibility across
   particle realisations and noise seeds, in two modes either side of the
   same MATLAB step. Writes `v4_recovery.csv`.
9. **`twin_method_comparison.py`** — V-SHARP, PDF and LBV compared against
   ground truth, in two modes either side of the MATLAB step.
10. **`caox_bound.py`** — sweeps simulated calcium oxalate susceptibility to
    establish the largest value that would still go undetected, in two modes
    either side of the MATLAB step.

## What produces what

| Script | Produces |
|---|---|
| `qsm_pipeline.py` | susceptibility maps, local field maps, masks |
| `roi_analysis.py` | `roi_results_per_tube.csv`: per-tube Δχ mean and peak, gel reference, focal dip, detection outcome |
| `r2star_analysis.py` | `r2star_results.csv`: per-tube R2* and ΔR2*, with and without echo truncation |
| `twin_recovery.py` | `v4_recovery.csv`: the split of the loss between partial volume and the reconstruction pipeline |
| `twin_method_comparison.py` | background removal methods scored against ground truth |
| `caox_bound.py` | the upper bound on calcium oxalate susceptibility |
| `verify_twin.py` | pass or fail for six analytic checks of the forward model |

Map these to the table and figure numbers of the published version before
release, since those can change in proof.

## Derived data

| File | Rows | Contents |
|---|---|---|
| `roi_results_per_tube.csv` | 24 | one row per tube, six conditions × four tubes |
| `r2star_results.csv` | 24 | the same tubes, R2* with echo truncation |
| `r2star_results_allechoes.csv` | 24 | the same, fitting every echo |
| `v4_recovery.csv` | 80 | five particle realisations × four noise seeds × four tubes |

The two R2* files are a pair. Fitting a mono-exponential through the Rician
noise floor biases R2* downward, and more so for fast-decaying samples, so
the difference between the two files is that bias.

## Conventions worth knowing before reading the code

**Field strength** is 2.895 T, from the Larmor frequency, not the nominal
3 T.

**B0 direction** is given in array coordinates, not scanner coordinates. For
these data the tubes were scanned with their long axes perpendicular to B0.
Check it for your own geometry from the sign of the dipole field around an
isolated inclusion: negative lobes along B0.

**Regions of interest are centred on the magnitude images**, not on the
susceptibility maps, so that the region is defined independently of the
quantity being measured.

**Detection requires two criteria together**, a peak susceptibility
exceeding three times the gel standard deviation and a focal dip at the
centre relative to its surroundings. Either alone is not sufficient.

**Regularisation weight** was selected by L-curve on these data. It is
scale-dependent, so re-run that selection for a different object, field
strength or resolution.

## Citation

Mišak K, Vita E De, Clark CA, et al (2026) Quantitative Susceptibility Mapping for Differentiating Hydroxyapatite and Calcium Oxalate Breast Calcifications at 3T: A Phantom Study. bioRxiv 2026.07.15.738683. https://doi.org/10.64898/2026.07.15.738683
