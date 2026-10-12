#!/usr/bin/env python3
"""
Region-of-interest extraction and detection calls from susceptibility maps
=========================================================================

Reads the susceptibility maps written by qsm_pipeline.py and, for each tube,
measures susceptibility in three regions: the hydroxyapatite inclusion, the
calcium oxalate inclusion, and a gel reference. 

Detection requires both of two criteria:

    |delta_chi_peak| > DETECT_SD_MULTIPLE x gel SD    (limit of detection)
    focal dip       >= FOCAL_DIP_SD_MULTIPLE x gel SD (centre more
                                                       diamagnetic than its
                                                       surrounds)

delta_chi_peak is the 5th percentile within the region minus the gel mean.
delta_chi_mean is the region mean minus the gel mean. Both are reported.

Run qsm_pipeline.py for every dataset and resolution first.
"""

import os
import csv
import warnings
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
#
# USING THIS ON YOUR OWN DATA
#
# RESULTS_ROOT must match qsm_pipeline.py's RESULTS_ROOT, and RESULTS_SUFFIX
# its RESULTS_SUFFIX, so that this script finds the folders that pipeline
# wrote. DATA_ROOT is needed only for the magnitude images used to centre the
# particle regions.
#
# COORDINATES holds one voxel coordinate per region per tube, in array index
# order, read off the magnitude images. Replace them with your own. The
# refinement step moves a particle centre by at most MAG_SEARCH_RADIUS_VOX
# voxels, so the listed coordinate has to be within a couple of voxels of the
# real void to begin with.
# =============================================================================

# ---- PATHS: must match qsm_pipeline.py ----
RESULTS_ROOT = r"C:\QSM_PDF_results"      # where the pipeline wrote its output
RESULTS_SUFFIX = "RESULTS_PDF"            # folders are <Gel>_<res>_<SUFFIX>
DATA_ROOT = r"D:\QSM Study"               # raw data, for the magnitude images
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "ROI_ANALYSIS")

DATASET_DIRS = {
    "Alginate": os.path.join(DATA_ROOT, "0704ALGINATE_123456"),
    "FGT":      os.path.join(DATA_ROOT, "0704FGT_123456"),
    "Adipose":  os.path.join(DATA_ROOT, "0704ADIPOSE_123456"),
}

# ---- REGION GEOMETRY ----
ROI_RADIUS_VOXELS = 2            # 33 voxels in a radius-2 sphere
FOCAL_DIP_SPACING_VOX = 2        # shell distance for the focal dip test

# ---- COVERAGE REQUIREMENTS ----
MIN_COVERAGE = 0.60              # particle region, fraction inside valid region
GEL_MIN_COVERAGE = 1.00          # gel reference must be fully valid
GEL_MIN_DIST_FROM_PARTICLE_MM = 8.0
GEL_SEARCH_RADIUS_VOX = 25       # how far to look when relocating a gel region

# ---- MAGNITUDE-BASED CENTRING ----
REFINE_FROM_MAGNITUDE = True
MAG_SEARCH_RADIUS_VOX = 2        # how far refinement may move a centre

# ---- DETECTION CRITERIA ----
DETECT_SD_MULTIPLE = 3.0
FOCAL_DIP_SD_MULTIPLE = 1.0


# Voxel coordinates in array index order, read from the magnitude images.
COORDINATES = {
    ("Alginate", "070mm"): {
        "Tube1": {"HA": (17, 35, 212), "Gel": (28, 57, 217), "CaOx": (13, 77, 217)},
        "Tube2": {"HA": (24, 33, 165), "Gel": (10, 86, 171), "CaOx": (20, 79, 159)},
        "Tube3": {"HA": (19, 35, 95),  "Gel": (36, 71, 98),  "CaOx": (27, 65, 96)},
        "Tube4": {"HA": (15, 28, 46),  "Gel": (13, 73, 49),  "CaOx": (29, 55, 48)},
    },
    ("Alginate", "086mm"): {
        "Tube1": {"HA": (21, 48, 196), "Gel": (26, 59, 203), "CaOx": (20, 72, 196)},
        "Tube2": {"HA": (27, 45, 156), "Gel": (17, 68, 161), "CaOx": (18, 81, 156)},
        "Tube3": {"HA": (23, 47, 100), "Gel": (20, 84, 104), "CaOx": (26, 71, 107)},
        "Tube4": {"HA": (19, 41, 60),  "Gel": (33, 69, 68),  "CaOx": (26, 64, 64)},
    },
    ("FGT", "070mm"): {
        "Tube1": {"HA": (31, 48, 206), "Gel": (35, 55, 204), "CaOx": (25, 72, 205)},
        "Tube2": {"HA": (27, 41, 155), "Gel": (15, 57, 159), "CaOx": (23, 72, 161)},
        "Tube3": {"HA": (21, 37, 90),  "Gel": (15, 55, 82),  "CaOx": (20, 57, 92)},
        "Tube4": {"HA": (15, 31, 40),  "Gel": (31, 40, 45),  "CaOx": (12, 53, 32)},
    },
    ("FGT", "086mm"): {
        "Tube1": {"HA": (30, 53, 200), "Gel": (19, 70, 205), "CaOx": (26, 72, 198)},
        "Tube2": {"HA": (27, 46, 156), "Gel": (16, 50, 159), "CaOx": (24, 72, 161)},
        "Tube3": {"HA": (22, 43, 105), "Gel": (17, 51, 99),  "CaOx": (22, 60, 99)},
        "Tube4": {"HA": (18, 37, 61),  "Gel": (12, 48, 61),  "CaOx": (22, 60, 107)},
    },
    ("Adipose", "070mm"): {
        "Tube1": {"HA": (9, 33, 224),  "Gel": (9, 79, 221),  "CaOx": (26, 84, 221)},
        "Tube2": {"HA": (24, 37, 166), "Gel": (33, 58, 164), "CaOx": (20, 48, 162)},
        "Tube3": {"HA": (13, 13, 91),  "Gel": (12, 58, 92),  "CaOx": (17, 66, 85)},
        "Tube4": {"HA": (27, 25, 44),  "Gel": (31, 56, 40),  "CaOx": (19, 67, 42)},
    },
    ("Adipose", "086mm"): {
        "Tube1": {"HA": (14, 46, 208), "Gel": (27, 58, 208), "CaOx": (16, 84, 193)},
        "Tube2": {"HA": (27, 45, 156), "Gel": (28, 62, 156), "CaOx": (18, 71, 152)},
        "Tube3": {"HA": (19, 25, 100), "Gel": (33, 66, 100), "CaOx": (23, 67, 92)},
        "Tube4": {"HA": (26, 31, 55),  "Gel": (27, 75, 55),  "CaOx": (19, 72, 56)},
    },
}


# =============================================================================
# REGION GEOMETRY
# =============================================================================

def sphere_offsets(radius):
    r = int(radius)
    x, y, z = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    keep = (x**2 + y**2 + z**2) <= radius**2
    return np.stack([x[keep], y[keep], z[keep]], axis=1)


OFFS = sphere_offsets(ROI_RADIUS_VOXELS)


def roi_indices(centre, shape):
    idx = np.array(centre)[None, :] + OFFS
    ok = np.all((idx >= 0) & (idx < np.array(shape)[None, :]), axis=1)
    idx = idx[ok]
    return idx[:, 0], idx[:, 1], idx[:, 2]


def extract_roi_stats(chi, centre, valid):
    """Statistics over valid voxels only.

    Coverage is returned so that a partially invalid region can never be used
    without the caller knowing.
    """
    i, j, k = roi_indices(centre, chi.shape)
    inside = valid[i, j, k]
    n_box = len(inside)
    n_valid = int(inside.sum())
    coverage = n_valid / n_box if n_box else 0.0

    if n_valid == 0:
        nan = float('nan')
        return dict(n_box=n_box, n_valid=0, coverage=0.0, mean=nan, sd=nan,
                    median=nan, p5=nan, p95=nan, min=nan, max=nan)

    v = chi[i[inside], j[inside], k[inside]]
    return dict(
        n_box=n_box, n_valid=n_valid, coverage=coverage,
        mean=float(v.mean()), sd=float(v.std()),
        median=float(np.median(v)),
        p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
        min=float(v.min()), max=float(v.max()))


def refine_centre_on_magnitude(mag_avg, centre, valid,
                               search=MAG_SEARCH_RADIUS_VOX):
    """Move the region centre to the darkest magnitude voxel nearby.
    """
    c = np.array(centre)
    best, best_val = tuple(c), np.inf
    for dx in range(-search, search+1):
        for dy in range(-search, search+1):
            for dz in range(-search, search+1):
                p = c + np.array([dx, dy, dz])
                if np.any(p < 0) or np.any(p >= np.array(mag_avg.shape)):
                    continue
                if not valid[tuple(p)]:
                    continue
                val = mag_avg[tuple(p)]
                if val < best_val:
                    best_val, best = val, tuple(int(q) for q in p)
    return best


def relocate_gel(centre, valid, particle_centres, voxel_size):
    """Nearest fully valid gel position at least
    """
    part = np.zeros(valid.shape, bool)
    for p in particle_centres:
        part[tuple(p)] = True
    dist_mm = distance_transform_edt(~part, sampling=voxel_size)

    c = np.array(centre)
    R = GEL_SEARCH_RADIUS_VOX
    best, best_d = None, np.inf
    for dx in range(-R, R+1, 1):
        for dy in range(-R, R+1, 1):
            for dz in range(-R, R+1, 1):
                p = c + np.array([dx, dy, dz])
                if np.any(p < 0) or np.any(p >= np.array(valid.shape)):
                    continue
                if dist_mm[tuple(p)] < GEL_MIN_DIST_FROM_PARTICLE_MM:
                    continue
                i, j, k = roi_indices(tuple(p), valid.shape)
                if len(i) < len(OFFS) or not valid[i, j, k].all():
                    continue
                d = dx*dx + dy*dy + dz*dz
                if d < best_d:
                    best_d, best = d, tuple(int(q) for q in p)
    return best


def check_focal_dip(chi, centre, valid, gel_sd,
                    spacing=FOCAL_DIP_SPACING_VOX):
    """Centre voxel against the shell `spacing` voxels away, valid voxels only.
    """
    shape = chi.shape
    if not valid[tuple(centre)]:
        return False, float('nan')
    centre_val = chi[tuple(centre)]
    vals = []
    for dx in (-spacing, 0, spacing):
        for dy in (-spacing, 0, spacing):
            for dz in (-spacing, 0, spacing):
                if dx == dy == dz == 0:
                    continue
                p = (centre[0]+dx, centre[1]+dy, centre[2]+dz)
                if all(0 <= p[a] < shape[a] for a in range(3)) and valid[p]:
                    vals.append(chi[p])
    if not vals:
        return False, float('nan')
    dip = centre_val - np.mean(vals)      # negative: centre more diamagnetic
    return bool(dip < -FOCAL_DIP_SD_MULTIPLE * gel_sd), float(dip)


# =============================================================================
# STATISTICS
# =============================================================================

def compute_cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    s = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)
                / (len(a)+len(b)-2))
    return float('nan') if s == 0 else float((a.mean()-b.mean())/s)


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_results = []
    cond = defaultdict(lambda: defaultdict(list))
    warnings_log = []

    print("=" * 104)
    print("  REGION-OF-INTEREST ANALYSIS")
    print(f"  valid voxels only | particle centres from magnitude: "
          f"{REFINE_FROM_MAGNITUDE}"
          f" | detection: {DETECT_SD_MULTIPLE}x gel SD AND focal dip")
    print("=" * 104)

    for (gel_type, resolution), tubes in sorted(COORDINATES.items()):
        base = DATASET_DIRS[gel_type]
        rdir = os.path.join(RESULTS_ROOT,
                            f"{gel_type}_{resolution}_{RESULTS_SUFFIX}")
        qsm_p = os.path.join(rdir, "qsm_susceptibility_ppm.nii.gz")
        val_p = os.path.join(rdir, "vsharp_valid_mask.nii.gz")
        mag_p = os.path.join(base, f"QSM_{resolution}_FINAL",
                             f"QSM_{resolution}_mag_4d.nii")

        if not os.path.exists(qsm_p):
            print(f"\n  MISSING: {qsm_p}")
            print(f"  Run qsm_pipeline.py for {gel_type} {resolution} first.")
            continue
        if not os.path.exists(val_p):
            print(f"\n  MISSING valid mask: {val_p}")
            print("  This is written by qsm_pipeline.py with MODE = \"invert\".")
            continue

        nii = nib.load(qsm_p)
        chi = nii.get_fdata().astype(np.float64)
        valid = nib.load(val_p).get_fdata() > 0
        voxel_size = np.array(nii.header.get_zooms()[:3], dtype=float)
        if not os.path.exists(mag_p):
            print(f"\n  NOTE magnitude not found: {mag_p}")
            print("  Particle centres will not be refined for this condition.")
            mag_avg = None
        else:
            mag_avg = nib.load(mag_p).get_fdata().mean(axis=3)

        print(f"\n{'='*92}")
        print(f"  {gel_type} ({resolution})   valid voxels {valid.sum()}")
        print(f"{'='*92}")

        for tube in sorted(tubes):
            co = dict(tubes[tube])

            # --- centre particle regions on the magnitude void ---
            moved = {}
            if REFINE_FROM_MAGNITUDE and mag_avg is not None:
                for part in ("HA", "CaOx"):
                    new = refine_centre_on_magnitude(mag_avg, co[part], valid)
                    if new != tuple(co[part]):
                        moved[part] = (tuple(co[part]), new)
                        co[part] = new

            # --- relocate the gel reference if coverage is insufficient ---
            gel_stats = extract_roi_stats(chi, co["Gel"], valid)
            gel_moved = None
            if gel_stats["coverage"] < GEL_MIN_COVERAGE:
                new = relocate_gel(co["Gel"], valid,
                                   [co["HA"], co["CaOx"]], voxel_size)
                if new is not None:
                    gel_moved = (tuple(tubes[tube]["Gel"]), new)
                    co["Gel"] = new
                    gel_stats = extract_roi_stats(chi, co["Gel"], valid)
                else:
                    warnings_log.append(
                        f"{gel_type} {resolution} {tube}: no fully valid gel "
                        f"region found within {GEL_SEARCH_RADIUS_VOX} voxels")

            ha_stats = extract_roi_stats(chi, co["HA"], valid)
            cax_stats = extract_roi_stats(chi, co["CaOx"], valid)

            gel_mean, gel_sd = gel_stats["mean"], gel_stats["sd"]
            thr = DETECT_SD_MULTIPLE * gel_sd

            ha_dm = ha_stats["mean"] - gel_mean
            ha_dp = ha_stats["p5"] - gel_mean
            cx_dm = cax_stats["mean"] - gel_mean
            cx_dp = cax_stats["p5"] - gel_mean

            ha_focal, ha_dip = check_focal_dip(chi, co["HA"], valid, gel_sd)
            cx_focal, cx_dip = check_focal_dip(chi, co["CaOx"], valid, gel_sd)

            ha_ok = ha_stats["coverage"] >= MIN_COVERAGE
            cx_ok = cax_stats["coverage"] >= MIN_COVERAGE
            gel_ok = gel_stats["coverage"] >= GEL_MIN_COVERAGE

            ha_det = bool(ha_ok and gel_ok and abs(ha_dp) > thr and ha_focal)
            cx_det = bool(cx_ok and gel_ok and abs(cx_dp) > thr and cx_focal)

            ratio_p = (abs(ha_dp)/abs(cx_dp)) if abs(cx_dp) > 1e-6 else np.inf
            ratio_m = (abs(ha_dm)/abs(cx_dm)) if abs(cx_dm) > 1e-6 else np.inf

            print(f"\n  {tube}")
            for nm, st, dm, dp, det, ok in [
                ("HA", ha_stats, ha_dm, ha_dp, ha_det, ha_ok),
                ("CaOx", cax_stats, cx_dm, cx_dp, cx_det, cx_ok)]:
                cov = f"{100*st['coverage']:.0f}%"
                flag = "" if ok else "  <-- LOW COVERAGE, EXCLUDED"
                print(f"    {nm:<5} n={st['n_valid']:>3}/{st['n_box']:<3} "
                      f"cov {cov:>5}  mean {st['mean']:+.4f}  "
                      f"p5 {st['p5']:+.4f}  "
                      f"dchi_mean {dm:+.4f}  dchi_peak {dp:+.4f}  "
                      f"{'DETECTED' if det else 'not det'}{flag}")
            gflag = "" if gel_ok else "  <-- GEL REFERENCE UNRELIABLE"
            print(f"    {'Gel':<5} n={gel_stats['n_valid']:>3}/"
                  f"{gel_stats['n_box']:<3} "
                  f"cov {100*gel_stats['coverage']:.0f}%  "
                  f"mean {gel_mean:+.4f}  "
                  f"sd {gel_sd:.4f}  -> threshold {thr:.4f} ppm{gflag}")
            print(f"    focal dip: HA {ha_dip:+.4f} "
                  f"({'yes' if ha_focal else 'no'})   "
                  f"CaOx {cx_dip:+.4f} ({'yes' if cx_focal else 'no'})")
            for part, (old, new) in moved.items():
                print(f"    NOTE {part} centre moved on magnitude: "
                      f"{old} -> {new}")
            if gel_moved:
                print(f"    NOTE gel region relocated for coverage: "
                      f"{gel_moved[0]} -> {gel_moved[1]}")

            rec = dict(gel_type=gel_type, resolution=resolution, tube=tube,
                       HA_coord=str(co["HA"]), HA_coverage=ha_stats["coverage"],
                       HA_mean=ha_stats["mean"], HA_p5=ha_stats["p5"],
                       HA_delta_mean=ha_dm, HA_delta_peak=ha_dp,
                       HA_focal_dip=ha_dip, HA_detected=ha_det,
                       CaOx_coord=str(co["CaOx"]),
                       CaOx_coverage=cax_stats["coverage"],
                       CaOx_mean=cax_stats["mean"], CaOx_p5=cax_stats["p5"],
                       CaOx_delta_mean=cx_dm, CaOx_delta_peak=cx_dp,
                       CaOx_focal_dip=cx_dip, CaOx_detected=cx_det,
                       Gel_coord=str(co["Gel"]),
                       Gel_coverage=gel_stats["coverage"],
                       Gel_mean=gel_mean, Gel_sd=gel_sd, threshold=thr,
                       ratio_mean=ratio_m, ratio_peak=ratio_p,
                       gel_relocated=bool(gel_moved),
                       particles_refined=";".join(moved.keys()))
            all_results.append(rec)

            key = (gel_type, resolution)
            if ha_ok and gel_ok:
                cond[key]["HA_mean"].append(ha_dm)
                cond[key]["HA_peak"].append(ha_dp)
            if cx_ok and gel_ok:
                cond[key]["CaOx_mean"].append(cx_dm)
                cond[key]["CaOx_peak"].append(cx_dp)
            cond[key]["HA_detected"].append(ha_det)
            cond[key]["CaOx_detected"].append(cx_det)
            cond[key]["ratio_peak"].append(ratio_p)

    if not all_results:
        print("\n  No data processed. Check RESULTS_ROOT and RESULTS_SUFFIX "
              "match qsm_pipeline.py.")
        return

    # ---- summary ----
    print("\n" + "=" * 104)
    print("  SUMMARY PER CONDITION")
    print("=" * 104)
    print(f"\n  {'Condition':<22}{'HA dchi_peak':>20}{'CaOx dchi_peak':>20}"
          f"{'Cohen d':>10}{'HA det':>9}{'CaOx det':>10}")
    print("  " + "-" * 90)
    for key in sorted(cond):
        g, r = key
        d = cond[key]
        ha, cx = np.array(d["HA_peak"]), np.array(d["CaOx_peak"])
        dcoh = compute_cohen_d(ha, cx)
        print(f"  {g+' ('+r+')':<22}"
              f"{ha.mean():>9.4f}+-"
              f"{ha.std(ddof=1) if len(ha) > 1 else 0:<9.4f}"
              f"{cx.mean():>9.4f}+-"
              f"{cx.std(ddof=1) if len(cx) > 1 else 0:<9.4f}"
              f"{dcoh:>10.2f}"
              f"{sum(d['HA_detected'])}/{len(d['HA_detected']):<7}"
              f"{sum(d['CaOx_detected'])}/{len(d['CaOx_detected'])}")

    ha_tot = sum(r["HA_detected"] for r in all_results)
    cx_tot = sum(r["CaOx_detected"] for r in all_results)
    n = len(all_results)
    print("\n" + "=" * 104)
    print(f"  TOTAL DETECTION:  HA {ha_tot}/{n}   CaOx {cx_tot}/{n}")
    print("=" * 104)

    excluded = [r for r in all_results
                if r["HA_coverage"] < MIN_COVERAGE
                or r["CaOx_coverage"] < MIN_COVERAGE
                or r["Gel_coverage"] < GEL_MIN_COVERAGE]
    if excluded:
        print("\n  Regions below the coverage requirement, excluded from the")
        print("  per-condition statistics above:")
        for r in excluded:
            print(f"    {r['gel_type']} {r['resolution']} {r['tube']}: "
                  f"HA {100*r['HA_coverage']:.0f}%  "
                  f"CaOx {100*r['CaOx_coverage']:.0f}%  "
                  f"Gel {100*r['Gel_coverage']:.0f}%")

    relocated = [r for r in all_results if r["gel_relocated"]]
    if relocated:
        print(f"\n  Gel regions relocated for coverage: {len(relocated)} of "
              f"{n}. Recorded per tube in the CSV as gel_relocated.")

    for w in warnings_log:
        print(f"  WARNING: {w}")

    # ---- CSV ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p1 = os.path.join(OUTPUT_DIR, "roi_results_per_tube.csv")
    with open(p1, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\n  Saved: {p1}")


if __name__ == "__main__":
    main()