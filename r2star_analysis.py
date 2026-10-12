#!/usr/bin/env python3
"""
R2* fitting and region-of-interest extraction
=============================================

Fits a voxelwise mono-exponential to the multi-echo magnitude signal and
measures R2* in the same regions as the susceptibility analysis: the
hydroxyapatite inclusion, the calcium oxalate inclusion, and a gel reference.

ECHO TRUNCATION
---------------
In a Rician magnitude image the signal does not decay to zero but plateaus at
the noise floor. Fitting a mono-exponential through that plateau biases R2*
downward, and the bias grows with the decay rate, so it is larger for
fast-decaying formulations than for slow ones. Left uncorrected it would
distort a comparison between formulations rather than adding a constant
offset.

Echoes are therefore truncated per voxel where the signal falls below
NOISE_FACTOR times the measured noise floor, subject to a minimum of
MIN_ECHOES. Setting USE_ALL_ECHOES = True fits every echo instead and writes
to a separate CSV, so the size of the bias can be measured rather than
assumed. Run it both ways and compare.

INPUTS AND OUTPUTS
------------------
Needs the multi-echo magnitude images and mask.nii.gz, which qsm_pipeline.py
writes, so run that first for every dataset and resolution.

Writes, per condition, into the pipeline's results folder:

    r2star_map.nii.gz          fitted R2* in s^-1
    r2star_n_echoes.nii.gz     echoes used per voxel

and one table into OUT_DIR:

    r2star_results.csv             truncated fit (default)
    r2star_results_allechoes.csv   all echoes, for the comparison
"""

import os
import csv
import numpy as np
import nibabel as nib
from scipy.optimize import curve_fit

# =============================================================================
# CONFIGURATION
# =============================================================================
#
# USING THIS ON YOUR OWN DATA
# ---------------------------
# The imaging data for the published analysis is not distributed, so none of
# the paths below will exist on your machine, and the coordinates refer to
# those specific phantoms.
#
# RESULTS_ROOT and RESULTS_SUFFIX must match qsm_pipeline.py, so that this
# script finds the masks it wrote. DATA_ROOT points at the raw magnitude
# images. ECHO_TIMES must come from your own protocol.
# =============================================================================

# ---- PATHS: must match qsm_pipeline.py ----
RESULTS_ROOT = r"C:\QSM_PDF_results"
RESULTS_SUFFIX = "RESULTS_PDF"
DATA_ROOT = r"D:\QSM Study"
OUT_DIR = os.path.join(RESULTS_ROOT, "R2STAR_ANALYSIS")

DATASET_DIRS = {
    "Alginate": os.path.join(DATA_ROOT, "0704ALGINATE_123456"),
    "FGT":      os.path.join(DATA_ROOT, "0704FGT_123456"),
    "Adipose":  os.path.join(DATA_ROOT, "0704ADIPOSE_123456"),
}

# ---- ECHO TRUNCATION, see the docstring ----
USE_ALL_ECHOES = False
MIN_ECHOES = 4              
NOISE_FACTOR = 3.0          # require signal above 3x the noise floor

# ---- REGIONS ----
ROI_RADIUS_VOXELS = 2
MIN_COVERAGE = 0.60         # fraction of the region inside the mask

ECHO_TIMES = {"070mm": [2.89, 7.14, 11.39, 15.64, 19.89, 24.14, 28.39],
              "086mm": [2.70, 6.95, 11.20, 15.45, 19.70, 23.95, 28.20]}

# Voxel coordinates in array index order.
#
#
# They are duplicated across two files with nothing enforcing agreement. If
# either set changes, change both.
COORDINATES = {
    ("Alginate", "070mm"): {
        "Tube1": {"HA": (17, 35, 212), "Gel": (28, 57, 217), "CaOx": (13, 77, 217)},
        "Tube2": {"HA": (24, 33, 165), "Gel": (12, 86, 169), "CaOx": (20, 79, 159)},
        "Tube3": {"HA": (19, 35, 95),  "Gel": (35, 74, 97),  "CaOx": (27, 65, 96)},
        "Tube4": {"HA": (15, 28, 46),  "Gel": (13, 73, 49),  "CaOx": (29, 55, 48)},
    },
    ("Alginate", "086mm"): {
        "Tube1": {"HA": (21, 48, 196), "Gel": (26, 59, 203), "CaOx": (20, 72, 196)},
        "Tube2": {"HA": (27, 45, 157), "Gel": (17, 68, 161), "CaOx": (18, 81, 156)},
        "Tube3": {"HA": (23, 47, 100), "Gel": (20, 84, 104), "CaOx": (26, 71, 107)},
        "Tube4": {"HA": (19, 41, 60),  "Gel": (33, 69, 68),  "CaOx": (26, 64, 64)},
    },
    ("FGT", "070mm"): {
        "Tube1": {"HA": (31, 49, 207), "Gel": (35, 55, 204), "CaOx": (25, 72, 205)},
        "Tube2": {"HA": (27, 41, 155), "Gel": (15, 57, 159), "CaOx": (23, 72, 161)},
        "Tube3": {"HA": (22, 37, 90),  "Gel": (15, 55, 82),  "CaOx": (20, 57, 92)},
        "Tube4": {"HA": (14, 30, 40),  "Gel": (31, 40, 45),  "CaOx": (12, 53, 32)},
    },
    ("FGT", "086mm"): {
        "Tube1": {"HA": (30, 53, 200), "Gel": (19, 70, 205), "CaOx": (31, 71, 198)},
        "Tube2": {"HA": (27, 46, 156), "Gel": (16, 50, 159), "CaOx": (24, 72, 161)},
        "Tube3": {"HA": (22, 43, 105), "Gel": (17, 51, 99),  "CaOx": (22, 60, 99)},
        "Tube4": {"HA": (18, 37, 61),  "Gel": (12, 48, 61),  "CaOx": (22, 60, 107)},
    },
    ("Adipose", "070mm"): {
        "Tube1": {"HA": (9, 34, 222),  "Gel": (9, 79, 221),  "CaOx": (26, 84, 221)},
        "Tube2": {"HA": (24, 37, 166), "Gel": (33, 58, 164), "CaOx": (20, 48, 162)},
        "Tube3": {"HA": (13, 13, 91),  "Gel": (12, 58, 92),  "CaOx": (17, 66, 85)},
        "Tube4": {"HA": (28, 23, 44),  "Gel": (31, 56, 40),  "CaOx": (19, 67, 42)},
    },
    ("Adipose", "086mm"): {
        "Tube1": {"HA": (14, 46, 208), "Gel": (27, 58, 208), "CaOx": (16, 84, 193)},
        "Tube2": {"HA": (27, 45, 156), "Gel": (28, 62, 156), "CaOx": (18, 71, 152)},
        "Tube3": {"HA": (19, 25, 100), "Gel": (33, 66, 100), "CaOx": (23, 67, 92)},
        "Tube4": {"HA": (26, 31, 55),  "Gel": (27, 75, 55),  "CaOx": (19, 72, 56)},
    },
}


# =============================================================================
# R2* FITTING
# =============================================================================

def mono_exponential(te, s0, r2star):
    return s0 * np.exp(-te * r2star)


def estimate_noise_floor(mag_4d, mask):
    """Noise floor from voxels outside the sample at the last echo.

    In a Rician magnitude image the background does not reach zero: it
    plateaus at roughly 1.25 sigma. That plateau is what biases the fit.
    """
    bg = mag_4d[..., -1][~(mask > 0)]
    if bg.size < 1000:
        return 0.0
    return float(np.median(bg))


def fit_r2star(mag_4d, TEs_ms, mask, noise_floor):
    """Voxelwise mono-exponential fit, with optional echo truncation.

    Echoes are truncated per voxel where the signal falls below
    NOISE_FACTOR x the noise floor, unless USE_ALL_ECHOES is set. See the
    module docstring for why this matters more for fast-decaying samples.
    """
    TEs_s = np.array(TEs_ms) / 1000.0
    shape = mag_4d.shape[:3]
    r2 = np.zeros(shape)
    s0m = np.zeros(shape)
    nech = np.zeros(shape, dtype=np.int8)
    err = np.zeros(shape)

    thresh = NOISE_FACTOR * noise_floor
    idx = np.argwhere(mask > 0)
    n_ok = n_bad = 0

    for ix, iy, iz in idx:
        sig = mag_4d[ix, iy, iz, :]
        if sig[0] < 1e-6:
            continue

        if USE_ALL_ECHOES:
            use = np.ones(len(TEs_s), bool)
        else:
            use = sig > thresh
            # keep a contiguous run from the first echo
            if not use[0]:
                use[:] = True
            else:
                first_bad = np.argmax(~use) if (~use).any() else len(use)
                use = np.zeros(len(use), bool)
                use[:max(first_bad, MIN_ECHOES)] = True
            if use.sum() < MIN_ECHOES:
                use[:MIN_ECHOES] = True

        te_f, sig_f = TEs_s[use], sig[use]
        try:
            r0 = -np.log(max(sig_f[-1], 1e-9)/sig_f[0]) / (te_f[-1]-te_f[0])
            r0 = float(np.clip(r0, 1.0, 400.0))
            popt, pcov = curve_fit(mono_exponential, te_f, sig_f,
                                   p0=[sig_f[0], r0],
                                   bounds=([0, 0], [sig_f[0]*2, 500]),
                                   maxfev=2000)
            r2[ix, iy, iz] = popt[1]
            s0m[ix, iy, iz] = popt[0]
            err[ix, iy, iz] = np.sqrt(np.diag(pcov))[1]
            nech[ix, iy, iz] = int(use.sum())
            n_ok += 1
        except (RuntimeError, ValueError):
            n_bad += 1

    m = mask > 0
    print(f"    fitted {n_ok}, failed {n_bad}")
    print(f"    echoes used: median {np.median(nech[m & (nech > 0)]):.0f} "
          f"of {len(TEs_s)}")
    print(f"    R2* {r2[m].min():.1f} to {r2[m].max():.1f}, "
          f"mean {r2[m].mean():.1f} s^-1")
    return r2, s0m, err, nech


# =============================================================================
# REGION EXTRACTION, mask-aware
# =============================================================================

def sphere_idx(c, shape, R=ROI_RADIUS_VOXELS):
    r = int(R)
    x, y, z = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    keep = (x**2 + y**2 + z**2) <= R**2
    idx = np.array(c)[None, :] + np.stack([x[keep], y[keep], z[keep]], 1)
    ok = np.all((idx >= 0) & (idx < np.array(shape)[None, :]), 1)
    idx = idx[ok]
    return idx[:, 0], idx[:, 1], idx[:, 2]


def roi_values(vol, mask, c):
    """Values inside the mask only, plus the coverage fraction."""
    i, j, k = sphere_idx(c, vol.shape)
    inside = mask[i, j, k] > 0
    cov = inside.mean() if len(inside) else 0.0
    if inside.sum() < 3:
        return None, cov
    return vol[i[inside], j[inside], k[inside]], cov


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    print("=" * 78)
    print("  R2* ANALYSIS")
    print(f"  echo truncation: "
          f"{'OFF (all echoes)' if USE_ALL_ECHOES else 'ON'}"
          f"   mask from: {RESULTS_SUFFIX}")
    print("=" * 78)

    for (gel, res), tubes in sorted(COORDINATES.items()):
        base = DATASET_DIRS[gel]
        magp = os.path.join(base, f"QSM_{res}_FINAL", f"QSM_{res}_mag_4d.nii")
        rdir = os.path.join(RESULTS_ROOT, f"{gel}_{res}_{RESULTS_SUFFIX}")
        maskp = os.path.join(rdir, "mask.nii.gz")

        if not os.path.exists(magp):
            print(f"\n  SKIP {gel} {res}: no magnitude at {magp}")
            continue
        if not os.path.exists(maskp):
            print(f"\n  SKIP {gel} {res}: no mask at {maskp}")
            print(f"       Run qsm_pipeline.py for {gel} {res} first.")
            continue

        print(f"\n{'='*78}\n  {gel} ({res})\n{'='*78}")
        nii = nib.load(magp)
        mag4 = nii.get_fdata().astype(np.float64)
        affine = nii.affine
        mask = nib.load(maskp).get_fdata()

        if mag4.shape[3] != len(ECHO_TIMES[res]):
            print(f"  ** {mag4.shape[3]} echoes in the data but "
                  f"{len(ECHO_TIMES[res])} echo times configured. "
                  f"Fix ECHO_TIMES before continuing.")
            continue

        nf = estimate_noise_floor(mag4, mask)
        print(f"  noise floor (background, last echo): {nf:.1f}")
        r2, s0m, err, nech = fit_r2star(mag4, ECHO_TIMES[res], mask, nf)

        nib.save(nib.Nifti1Image(r2.astype(np.float32), affine),
                 os.path.join(rdir, "r2star_map.nii.gz"))
        nib.save(nib.Nifti1Image(nech.astype(np.float32), affine),
                 os.path.join(rdir, "r2star_n_echoes.nii.gz"))

        print(f"\n  {'tube':<7}{'cov':>6}{'HA p95':>9}{'CaOx p95':>10}"
              f"{'gel':>8}{'dR2* HA':>10}{'dR2* CaOx':>11}")
        print("  " + "-" * 62)

        for tube in sorted(tubes):
            c = tubes[tube]
            gv, gcov = roi_values(r2, mask, c["Gel"])
            hv, hcov = roi_values(r2, mask, c["HA"])
            xv, xcov = roi_values(r2, mask, c["CaOx"])
            if gv is None or hv is None:
                print(f"  {tube:<7}gel or HA region has too few valid "
                      f"voxels, skipped")
                continue
            gm, gsd = gv.mean(), gv.std()
            ha_p95 = np.percentile(hv, 95)
            if xv is None:
                cx_p95 = cx_mean = np.nan
            else:
                cx_p95 = np.percentile(xv, 95)
                cx_mean = xv.mean()
            flag = "" if min(hcov, gcov) >= MIN_COVERAGE else "  <-- LOW COV"
            print(f"  {tube:<7}{100*hcov:>5.0f}%{ha_p95:>9.1f}{cx_p95:>10.1f}"
                  f"{gm:>8.1f}{ha_p95-gm:>10.1f}{cx_p95-gm:>11.1f}{flag}")

            rows.append(dict(
                gel=gel, resolution=res, tube=tube,
                HA_coverage=round(hcov, 3), Gel_coverage=round(gcov, 3),
                CaOx_coverage=round(xcov, 3),
                HA_p95=round(ha_p95, 2), HA_mean=round(hv.mean(), 2),
                CaOx_p95=round(cx_p95, 2), CaOx_mean=round(cx_mean, 2),
                Gel_mean=round(gm, 2), Gel_sd=round(gsd, 2),
                dR2s_HA=round(ha_p95-gm, 2), dR2s_CaOx=round(cx_p95-gm, 2),
                n_echoes_median=int(np.median(nech[mask > 0])),
                noise_floor=round(nf, 1),
                all_echoes=USE_ALL_ECHOES))

    if not rows:
        print("\n  Nothing processed. Check RESULTS_ROOT and RESULTS_SUFFIX "
              "match qsm_pipeline.py.")
        return

    # ---- summary ----
    print("\n\n" + "=" * 78)
    print("  SUMMARY: dR2* (particle - gel), s^-1")
    print("=" * 78)
    print(f"  {'condition':<20}{'HA':>18}{'CaOx':>18}{'ratio':>9}")
    print("  " + "-" * 62)
    for gel in ("Alginate", "Adipose", "FGT"):
        for res in ("070mm", "086mm"):
            sel = [r for r in rows
                   if r['gel'] == gel and r['resolution'] == res]
            if not sel:
                continue
            ha = np.array([r['dR2s_HA'] for r in sel])
            cx = np.array([r['dR2s_CaOx'] for r in sel])
            ratio = ha.mean()/cx.mean() if cx.mean() != 0 else np.nan
            print(f"  {gel+' ('+res+')':<20}"
                  f"{ha.mean():>9.1f}+-"
                  f"{ha.std(ddof=1) if len(ha) > 1 else 0:<8.1f}"
                  f"{cx.mean():>9.1f}+-"
                  f"{cx.std(ddof=1) if len(cx) > 1 else 0:<8.1f}"
                  f"{ratio:>9.1f}")

    p = os.path.join(
        OUT_DIR,
        f"r2star_results{'_allechoes' if USE_ALL_ECHOES else ''}.csv")
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {p}")

    print("""
  RUN IT BOTH WAYS
  ----------------
  Once with USE_ALL_ECHOES = False (the default) and once with True, then
  compare the two CSVs. The difference is the bias from fitting through the
  Rician noise floor, and it should be larger for formulations with a higher
  baseline R2*, since their signal reaches the floor at an earlier echo.
""")


if __name__ == "__main__":
    main()