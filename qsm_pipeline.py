#!/usr/bin/env python3
"""
QSM pipeline for microcalcification phantoms
============================================

This is the pipeline used for the results reported in the paper.

Background field removal is performed by Projection onto Dipole Fields (PDF),
which is implemented in the MEDI MATLAB toolbox rather than in Python, so the
pipeline runs in two halves with a MATLAB step between them:

    MODE = "prep"                 masking, ROMEO unwrapping, field map
                                  -> writes in_<tag>.mat
    run_medi_batch_PDF_LBV.m      background removal by PDF
                                  -> writes out_<tag>.mat
    MODE = "invert"               TV-ADMM dipole inversion
                                  -> writes qsm_susceptibility_ppm.nii.gz

Outputs, per dataset and resolution, into <RESULTS_ROOT>/<tag>_RESULTS_PDF:

    field_map_hz.nii.gz              total field, after unwrapping
    mask.nii.gz                      sample mask used for prep
    local_field_hz.nii.gz            after PDF background removal
    pdf_valid_mask.nii.gz            where PDF returned a value
    qsm_susceptibility_ppm.nii.gz    susceptibility map

Diagnostics are printed at each stage. Read them: a corrupted field map
propagates silently through everything downstream.

Companion scripts: roi_analysis.py (susceptibility per region of interest)
and r2star_analysis.py (relaxometry).
"""

import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import nibabel as nib
from scipy.io import savemat, loadmat
from scipy.fft import fftn, ifftn
from scipy.ndimage import (binary_erosion, generate_binary_structure,
                           binary_fill_holes, label, find_objects)

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
#
# USING THIS ON YOUR OWN DATA
# ---------------------------
# The imaging data for the published analysis is not distributed, so none of
# the paths below will exist on your machine. Everything you need to change is
# in this block; nothing further down the file hardcodes a path.
#
# 1. INPUT DATA. Point DATA_ROOT at a folder holding one subfolder per
#    condition, and name those subfolders in DATASET_DIRS. The pipeline
#    expects, inside each:
#
#        <condition>/QSM_<RES>_FINAL/QSM_<RES>_mag_4d.nii
#        <condition>/QSM_<RES>_FINAL/QSM_<RES>_phase_4d.nii
#
#    Both 4D, with echoes along the fourth dimension. Magnitude and phase from
#    a multi-echo 3D GRE, in that layout, is the only requirement: the
#    condition names are labels, so a single dataset of your own can be
#    registered under one key and run.
#
#    Phase units are detected at load time: Siemens integer (range above 100),
#    degrees (above 10), otherwise radians. Check the printed line says what
#    you expect.
#
# 2. ECHO_TIMES and VOXEL_SIZES. Read the echo times from your own protocol,
#    not from the values below. They are the setting most likely to be wrong on
#    a new dataset, and wrong values corrupt the field map while leaving every
#    intermediate looking plausible. The script stops if the echo count
#    disagrees with the data, which catches the gross case but not a shifted
#    train. RESOLUTION is just a dictionary key: rename it, or add entries, to
#    match your own protocol.
#
# 3. B0 and B0_DIRECTION. B0 is the true field from the Larmor frequency, not
#    the nominal value. B0_DIRECTION is in array coordinates and depends on how
#    the sample was oriented in the scanner, so it must be checked rather than
#    copied. See the note beside it.
#
# 4. ROMEO_PATH. ROMEO is a separate binary, not a Python dependency. Download
#    it from https://github.com/korbinian90/ROMEO/releases and point at the
#    executable. There is deliberately no Laplacian fallback: the inverse
#    Laplacian low-pass filters the field at exactly the sharp susceptibility
#    boundaries this work is about.
#
# 5. The MATLAB step needs the MEDI toolbox on the MATLAB path, obtained from
#    Cornell's MRI Research Lab under their own licence. See
#    run_medi_batch_PDF_LBV.m.
#
# RECONSTRUCTION PARAMETERS. TV_LAMBDA was selected by L-curve on this data.
# It is scale-dependent and not a universal value, so re-run the sweep for a
# different object, field strength or resolution.
# =============================================================================

MODE = "prep"              # "prep" -> run_medi_batch_PDF_LBV.m -> "invert"

DATASET = "Alginate"       # a key of DATASET_DIRS
RESOLUTION = "070mm"       # a key of ECHO_TIMES and VOXEL_SIZES

# ---- PATHS: the only machine-specific settings ----
DATA_ROOT    = r"D:\QSM Study"            # holds the per-condition folders
RESULTS_ROOT = r"C:\QSM_PDF_results"      # outputs, created if absent
EXCHANGE     = os.path.join(RESULTS_ROOT, "MEDI_EXCHANGE")  # .mat to/from MATLAB
ROMEO_PATH   = r"D:\romeo\romeo_win_3.2.8\bin\romeo.exe"

DATASET_DIRS = {
    "Alginate": os.path.join(DATA_ROOT, "0704ALGINATE_123456"),
    "FGT":      os.path.join(DATA_ROOT, "0704FGT_123456"),
    "Adipose":  os.path.join(DATA_ROOT, "0704ADIPOSE_123456"),
}

RESULTS_SUFFIX = "RESULTS_PDF"

# ---- ACQUISITION: from your own protocol, see note 2 above ----
ECHO_TIMES = {
    "070mm": [2.89, 7.14, 11.39, 15.64, 19.89, 24.14, 28.39],
    "086mm": [2.70, 6.95, 11.20, 15.45, 19.70, 23.95, 28.20],
}
VOXEL_SIZES = {"070mm": [0.70, 0.70, 0.70], "086mm": [0.86, 0.86, 0.86]}

B0 = 2.895                 # from the Larmor frequency, 123.244417 MHz

# B0 direction in ARRAY coordinates, not scanner coordinates.
#
# For the published data the tubes were scanned with their long axes
# perpendicular to B0, which in this array convention puts B0 along axis 2 and
# the tube long axis along axis 1.
#
# Verify this for your own geometry from the sign of the dipole field around an
# isolated inclusion: negative lobes along B0, positive perpendicular to it. A
# wrong direction here produces a plausible-looking map with the wrong sign
# structure around every particle.
B0_DIRECTION = [0, 0, 1]

# ---- MASKING ----
MASK_THRESHOLD = 0.1            # fraction of maximum mean magnitude
MASK_EROSION = 2                # voxels
PHASE_QUALITY_THRESHOLD = 0.3   # inter-echo phase coherence, 0 to 1

# ---- PHASE UNWRAPPING ----
# With several disconnected samples in one field of view, whole-volume
# unwrapping has to bridge empty air between them and can fail when the
# inter-sample field offsets exceed a wrap. Unwrapping each connected region
# separately removes that failure mode. "auto" measures the residual error
# first and only switches if it exceeds WRAP_JUMP_LIMIT.
PER_TUBE_UNWRAP = "auto"        # "auto", "always", or "never"
WRAP_JUMP_LIMIT = 100
MIN_TUBE_VOXELS = 5000          # connected regions smaller than this ignored
CROP_MARGIN_VOX = 10

# ---- DIPOLE INVERSION (TV-ADMM, preconditioned CG chi-update) ----
TV_LAMBDA = 0.0005
TV_MU = 0.05                    # ADMM penalty: affects convergence rate only
TV_MAX_ITER = 60
TV_CG_MAX_ITER = 10
TV_CG_TOL = 1e-3

SKIP_IF_EXISTS = False          # True reuses an existing .mat without saying
                                # much, which is convenient for resuming a
                                # batch and dangerous after a settings change


def tag():
    return f"{DATASET}_{RESOLUTION}"


def results_dir():
    return os.path.join(RESULTS_ROOT, f"{tag()}_{RESULTS_SUFFIX}")


# =============================================================================
# STAGES
# =============================================================================

def phase_quality(phase, mag):
    """Inter-echo phase coherence, 0 to 1. Low where the signal dephases."""
    n = phase.shape[3]
    cs = mag*np.exp(1j*phase)
    acc = np.zeros(phase.shape[:3], np.complex128)
    for e in range(1, n):
        d = cs[..., e]*np.conj(cs[..., e-1])
        acc += d/(np.abs(d)+1e-10)
    return np.abs(acc)/(n-1)


def make_mask(mag, phase):
    avg = np.mean(mag, axis=3)
    m = avg > MASK_THRESHOLD*avg.max()
    n0 = m.sum()
    q = phase_quality(phase, mag)
    m = m & (q > PHASE_QUALITY_THRESHOLD)
    n1 = m.sum()
    # Fill holes AFTER the quality refinement, not before. Dephasing around a
    # dense particle lowers coherence at exactly the location of interest, so
    # the quality step punches a hole there; filling afterwards restores it.
    # Filling first would leave the particles outside the mask.
    m = binary_fill_holes(m)
    n2 = m.sum()
    st = generate_binary_structure(3, 1)
    for _ in range(MASK_EROSION):
        m = binary_erosion(m, structure=st)
    print(f"    threshold {n0}, quality -{n0-n1}, fill +{n2-n1}, "
          f"erosion -{n2-m.sum()}  ->  {m.sum()} voxels "
          f"({100*m.sum()/m.size:.1f}% of FOV)")
    return m.astype(np.float64)


def romeo(phase, mag, TEs, affine):
    if not os.path.exists(ROMEO_PATH):
        raise FileNotFoundError(
            f"ROMEO not found at {ROMEO_PATH}. See note 4 in the "
            f"configuration block. There is no Laplacian fallback by design.")
    tmp = tempfile.mkdtemp(prefix="romeo_")
    try:
        pp, mp = os.path.join(tmp, "p.nii"), os.path.join(tmp, "m.nii")
        nib.save(nib.Nifti1Image(phase.astype(np.float32), affine), pp)
        nib.save(nib.Nifti1Image(mag.astype(np.float32), affine), mp)
        out = os.path.join(tmp, "o")
        os.makedirs(out, exist_ok=True)
        te = "["+",".join(f"{t:.2f}" for t in TEs)+"]"
        cmd = [ROMEO_PATH, "-p", pp, "-m", mp, "-t", te, "-k", "nomask",
               "-q", "--template", "2", "-o", out]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            r = subprocess.run(cmd+["--individual-unwrapping"],
                               capture_output=True, text=True, timeout=1200)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-800:])
        up = os.path.join(out, "unwrapped.nii")
        if not os.path.exists(up):
            up = os.path.join(out, "unwrapped.nii.gz")
        return nib.load(up).get_fdata().astype(np.float64)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def romeo_per_tube(phase, mag, TEs, affine, mask):
    """Unwrap each connected region on its own cropped volume, then
    reassemble. Nothing has to be bridged across empty air."""
    lab, n = label(mask > 0, structure=generate_binary_structure(3, 1))
    sizes = [(lab == i).sum() for i in range(1, n+1)]
    keep = [i+1 for i in range(n) if sizes[i] >= MIN_TUBE_VOXELS]
    print(f"    per-region unwrapping: {len(keep)} regions")
    un = np.zeros_like(phase)
    sl = find_objects(lab)
    for l in keep:
        s0 = sl[l-1]
        w = tuple(slice(max(0, x.start-CROP_MARGIN_VOX),
                        min(phase.shape[d], x.stop+CROP_MARGIN_VOX))
                  for d, x in enumerate(s0))
        sub = romeo(phase[w], mag[w], TEs, affine)
        this = (lab == l)[w]
        for e in range(phase.shape[3]):
            t = un[w][..., e]
            t[this] = sub[..., e][this]
            un[w+(e,)] = t
    return un


def field_map(un, TEs, mag, mask):
    """Magnitude-squared-weighted linear regression of phase against echo
    time, in Hz."""
    ts = np.array(TEs)/1000.0
    w = mag**2
    te = ts[None, None, None, :]
    sw = np.sum(w, 3)
    swt = np.sum(w*te, 3)
    swt2 = np.sum(w*te**2, 3)
    swp = np.sum(w*un, 3)
    swtp = np.sum(w*te*un, 3)
    den = sw*swt2-swt**2
    den[den == 0] = 1e-10
    return ((sw*swtp-swt*swp)/den/(2*np.pi))*mask


def count_wraps(f, mask, TEs):
    """A residual 2-pi error appears in the field map as a step of 1/dTE Hz.
    Counting those steps measures unwrapping failure directly."""
    wrap = 1.0/((TEs[1]-TEs[0])/1000.0)
    m = mask > 0
    tot = 0
    for ax in range(3):
        d = np.abs(np.diff(f, axis=ax))
        a = [slice(None)]*3
        a[ax] = slice(1, None)
        b = [slice(None)]*3
        b[ax] = slice(0, -1)
        tot += int(np.sum(d[m[tuple(a)] & m[tuple(b)]] > 0.85*wrap))
    return tot


# =============================================================================
# INVERSION
# =============================================================================

def kernels(sh, v):
    kx = np.fft.fftfreq(sh[0], v[0])
    ky = np.fft.fftfreq(sh[1], v[1])
    kz = np.fft.fftfreq(sh[2], v[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2+KY**2+KZ**2
    k2[0, 0, 0] = 1e-10
    b = np.array(B0_DIRECTION)/np.linalg.norm(B0_DIRECTION)
    D = 1/3-(KX*b[0]+KY*b[1]+KZ*b[2])**2/k2
    D[0, 0, 0] = 0
    lap = ((2-2*np.cos(2*np.pi*KX*v[0]))/v[0]**2 +
           (2-2*np.cos(2*np.pi*KY*v[1]))/v[1]**2 +
           (2-2*np.cos(2*np.pi*KZ*v[2]))/v[2]**2)
    return D, lap


def _g(c, v):
    return ((np.roll(c, -1, 0)-c)/v[0], (np.roll(c, -1, 1)-c)/v[1],
            (np.roll(c, -1, 2)-c)/v[2])


def _d(gx, gy, gz, v):
    return ((gx-np.roll(gx, 1, 0))/v[0] + (gy-np.roll(gy, 1, 1))/v[1]
            + (gz-np.roll(gz, 1, 2))/v[2])


def _GtG(x, v):
    """G^H G x = -div(grad(x)), whose Fourier symbol is +lap, matching the
    preconditioner denominator. The minus sign is the adjoint, not a
    convention: forward differences have backward-difference adjoints."""
    gx, gy, gz = _g(x, v)
    return -_d(gx, gy, gz, v)


def cg(rhs, chi0, mask, D, lap, mu, v):
    """Chi-subproblem by preconditioned conjugate gradient.

    The data-consistency operator D^H M D is not diagonal in k-space when a
    mask is present, so solving by k-space division solves the unmasked
    problem instead, and the result then depends on mu. CG solves the masked
    problem, which makes the solution mu-independent at convergence.
    """
    den = np.abs(D)**2+mu*lap
    den[den < 1e-12] = 1e-12

    def A(x):
        xm = x*mask
        Dx = np.real(ifftn(D*fftn(xm)))
        return mask*(np.real(ifftn(D*fftn(mask*Dx)))+mu*_GtG(xm, v))

    def M(r):
        return mask*np.real(ifftn(fftn(r)/den))

    x = chi0*mask
    r = rhs-A(x)
    z = M(r)
    p = z.copy()
    rz = np.sum(r*z)
    r0 = np.linalg.norm(r)
    if r0 < 1e-30:
        return x
    for _ in range(TV_CG_MAX_ITER):
        Ap = A(p)
        pAp = np.sum(p*Ap)
        if abs(pAp) < 1e-30:
            break
        a = rz/pAp
        x += a*p
        r -= a*Ap
        if np.linalg.norm(r)/r0 < TV_CG_TOL:
            break
        z = M(r)
        rzn = np.sum(r*z)
        p = z+(rzn/rz)*p
        rz = rzn
    return x*mask


def invert(lf, mask, v):
    """Total-variation-regularised dipole inversion by ADMM.

    Warm-started from a thresholded k-space division (TKD).
    """
    f = lf/(42.576e6*B0)*1e6
    D, lap = kernels(lf.shape, v)
    rhs0 = mask*np.real(ifftn(D*fftn(f*mask)))

    Di = np.zeros_like(D)
    k = np.abs(D) > 0.2
    Di[k] = 1/D[k]
    Di[~k] = np.sign(D[~k])/0.2
    chi = np.real(ifftn(fftn(f*mask)*Di))*mask

    gx, gy, gz = _g(chi, v)
    zx, zy, zz = gx.copy(), gy.copy(), gz.copy()
    ux, uy, uz = np.zeros_like(gx), np.zeros_like(gy), np.zeros_like(gz)
    lm = TV_LAMBDA/TV_MU

    for _ in range(TV_MAX_ITER):
        # Minus, because the adjoint of the forward-difference gradient is
        # minus the divergence. See _GtG.
        rhs = rhs0 - TV_MU*mask*_d(zx-ux, zy-uy, zz-uz, v)
        chi = cg(rhs, chi, mask, D, lap, TV_MU, v)
        gx, gy, gz = _g(chi, v)
        vx, vy, vz = gx+ux, gy+uy, gz+uz
        gm = np.sqrt(vx**2+vy**2+vz**2)
        sh = np.maximum(1-lm/(gm+1e-10), 0)
        zx, zy, zz = vx*sh, vy*sh, vz*sh
        ux += gx-zx
        uy += gy-zy
        uz += gz-zz
    return chi*mask


# =============================================================================
# MODES
# =============================================================================

def prep():
    base = DATASET_DIRS[DATASET]
    magp = os.path.join(base, f"QSM_{RESOLUTION}_FINAL",
                        f"QSM_{RESOLUTION}_mag_4d.nii")
    phap = os.path.join(base, f"QSM_{RESOLUTION}_FINAL",
                        f"QSM_{RESOLUTION}_phase_4d.nii")
    for p in (magp, phap):
        if not os.path.exists(p):
            print(f"  MISSING: {p}")
            print("  Expected layout: "
                  "<condition>/QSM_<RES>_FINAL/QSM_<RES>_mag_4d.nii")
            print("  See note 1 in the configuration block.")
            return

    matp = os.path.join(EXCHANGE, f"in_{tag()}.mat")
    if SKIP_IF_EXISTS and os.path.exists(matp):
        print(f"  {tag()}: in_{tag()}.mat exists, skipping. Set "
              f"SKIP_IF_EXISTS = False to reprocess.")
        return

    out = results_dir()
    os.makedirs(out, exist_ok=True)
    os.makedirs(EXCHANGE, exist_ok=True)
    TEs = ECHO_TIMES[RESOLUTION]
    v = VOXEL_SIZES[RESOLUTION]

    print("=" * 72)
    print(f"  {DATASET} {RESOLUTION}, PDF route, prep")
    print(f"  B0 {B0} T, direction {B0_DIRECTION}")
    print(f"  echo times {TEs}")
    print("=" * 72)

    nii = nib.load(magp)
    mag = nii.get_fdata().astype(np.float64)
    pha = nib.load(phap).get_fdata().astype(np.float64)
    affine = nii.affine
    print(f"\n  shape {mag.shape}")
    if mag.shape[3] != len(TEs):
        print(f"  ** {mag.shape[3]} echoes in the data but {len(TEs)} echo "
              f"times configured. Fix ECHO_TIMES before continuing.")
        return

    pr = pha.max()-pha.min()
    if pr > 100:
        pha = pha*np.pi/4096
        print(f"  phase range {pr:.0f}, scaled from Siemens units to radians")
    elif pr > 10:
        pha = np.deg2rad(pha)
        print(f"  phase range {pr:.1f}, converted from degrees")
    else:
        print(f"  phase range {pr:.2f}, assuming radians")

    print("\n  [1] masking")
    mask = make_mask(mag, pha)

    print("\n  [2] unwrapping")
    un = romeo(pha, mag, TEs, affine)
    fld = field_map(un, TEs, mag, mask)
    w = count_wraps(fld, mask, TEs)
    print(f"    whole-volume: {w} wrap-sized jumps")
    if PER_TUBE_UNWRAP == "always" or (PER_TUBE_UNWRAP == "auto"
                                       and w > WRAP_JUMP_LIMIT):
        un = romeo_per_tube(pha, mag, TEs, affine, mask)
        fld = field_map(un, TEs, mag, mask)
        w = count_wraps(fld, mask, TEs)
        print(f"    after per-region: {w}")
    if w > WRAP_JUMP_LIMIT:
        print(f"""
    ** {w} wrap-sized jumps remain, above the limit of {WRAP_JUMP_LIMIT}.
       Stop here. The field map is corrupted and nothing downstream will be
       meaningful. The usual cause is echo times that do not match the data;
       check them against the protocol. If they are right, the wraps are
       within a region rather than between regions, so inspect phase quality
       and magnitude where the jumps occur.""")

    fv = fld[mask > 0]
    print(f"\n  [3] field map: {fv.min():.0f} to {fv.max():.0f} Hz, "
          f"SD {fv.std():.1f} Hz")

    nib.save(nib.Nifti1Image(fld.astype(np.float32), affine),
             os.path.join(out, "field_map_hz.nii.gz"))
    nib.save(nib.Nifti1Image(mask.astype(np.float32), affine),
             os.path.join(out, "mask.nii.gz"))
    savemat(matp, {"iFreq": fld*2*np.pi, "Mask": mask, "N_std": mask,
                   "matrix_size": np.array(fld.shape, float),
                   "voxel_size": np.array(v),
                   "B0_dir": np.array(B0_DIRECTION, float)},
            do_compression=True)
    print(f"\n  wrote {matp}")
    print(f"""
  NEXT
  ----
  1. Run run_medi_batch_PDF_LBV.m in MATLAB, with its EXCHANGE set to
         {EXCHANGE}
     Watch the PDF conjugate-gradient residual it reports: it should fall
     several orders of magnitude below its starting value. A residual that
     stalls means PDF has not converged and the local field is unreliable.
  2. Return here with MODE = "invert".
""")


def invert_mode():
    out = results_dir()
    matp = os.path.join(EXCHANGE, f"out_{tag()}.mat")
    if not os.path.exists(matp):
        print(f"  {matp} not found. Run run_medi_batch_PDF_LBV.m first.")
        return

    fmp = os.path.join(out, "field_map_hz.nii.gz")
    if not os.path.exists(fmp):
        print(f"  {fmp} not found. Run MODE = \"prep\" first.")
        return

    affine = nib.load(fmp).affine
    v = VOXEL_SIZES[RESOLUTION]

    d = loadmat(matp)
    lf = np.asarray(d["RDF_PDF"], float)/(2*np.pi)
    valid = np.asarray(d["Mask_PDF"], float)
    vm = valid > 0
    mask = nib.load(os.path.join(out, "mask.nii.gz")).get_fdata() > 0

    print("=" * 72)
    print(f"  {DATASET} {RESOLUTION}, inversion")
    print("=" * 72)
    print(f"\n  coverage    {100*vm.sum()/mask.sum():.1f}% of mask")
    print(f"  local field {lf[vm].min():.1f} to {lf[vm].max():.1f} Hz, "
          f"SD {lf[vm].std():.2f} Hz")
    print("    A local field SD much larger than a few Hz indicates residual "
          "background field")
    print("    or an unwrapping error rather than sample structure. For "
          "reference, the")
    print("    phantoms in this study gave about 1.2 Hz.")

    print("\n  inverting ...")
    chi = invert(lf, valid, v)

    nib.save(nib.Nifti1Image(chi.astype(np.float32), affine),
             os.path.join(out, "qsm_susceptibility_ppm.nii.gz"))
    nib.save(nib.Nifti1Image(lf.astype(np.float32), affine),
             os.path.join(out, "local_field_hz.nii.gz"))
    # Filename kept as vsharp_valid_mask.nii.gz for compatibility with
    # roi_analysis.py, which reads that name. The mask comes from
    # PDF, not V-SHARP. Rename in both files together if you prefer.
    nib.save(nib.Nifti1Image(valid.astype(np.float32), affine),
             os.path.join(out, "vsharp_valid_mask.nii.gz"))

    print(f"  chi {chi[vm].min():.3f} to {chi[vm].max():.3f} ppm")
    print("    A range spanning several ppm means unwrapping or background "
          "removal failed.")
    print("    For reference, the alginate phantoms gave about -1.6 to "
          "+0.5 ppm.")
    print(f"\n  saved to {out}")
    print("""
  NEXT
  ----
  Locate the inclusions on the MAGNITUDE images, not on the susceptibility
  map, so that the region of interest is defined independently of the
  reconstruction being measured.

  Then check the sign of the local field around each magnitude void before
  treating it as a mineral: a diamagnetic inclusion gives negative dipole
  lobes along B0, whereas trapped air gives positive lobes. Air and mineral
  both appear as voids in magnitude.

  Add the coordinates to roi_analysis.py.
""")


if __name__ == "__main__":
    if MODE == "prep":
        prep()
    elif MODE == "invert":
        invert_mode()
    else:
        raise ValueError(f'MODE must be "prep" or "invert", not {MODE!r}')