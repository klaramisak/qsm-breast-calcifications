#!/usr/bin/env python3
"""
Digital twin: susceptibility recovery with uncertainty
======================================================

Puts a known susceptibility into a simulated phantom, runs it through the same
pipeline as the measured data, and compares what comes back with what went in.
The difference is what the measurement loses.

A single run gives one number with no uncertainty. Two things make it vary,
and both are sampled here:

  PARTICLE SHAPE   the real particles are irregular crushed granules that
                   differ from one another, so each realisation draws new
                   random ellipsoids from the measured size distribution
  NOISE            the same phantom with different random noise, as thermal
                   noise differs between acquisitions

THE LOSS SPLITS IN TWO, AND THE SPLIT IS THE POINT

    input susceptibility        what was specified
    ground truth in the ROI     after partial volume
    recovered susceptibility    after the whole pipeline

Recovery is reported two ways, over the particle mask and over a radius-2
sphere at the particle centre, because only the second matches how the
measured data is analysed. Both the mean and the 5th percentile are given,
with the coefficient of variation across realisations, since that is what
decides which is the more reproducible statistic.

HOW TO RUN

    MODE = "generate"           builds every phantom and preps it
    run_medi_batch_PDF_LBV.m    ONCE over the whole batch
    MODE = "invert"             inverts every one and prints the table

N_PARTICLE x N_NOISE sets the number of runs. Generation and inversion are
both slow, of the order of an hour each for twenty runs.
"""

import os
import csv
import shutil
import subprocess
import tempfile
import numpy as np
import nibabel as nib
from scipy.io import savemat, loadmat
from scipy.fft import fftn, ifftn
from scipy import ndimage
from scipy.ndimage import (binary_erosion, generate_binary_structure,
                           binary_fill_holes, gaussian_filter, label,
                           find_objects)

# =============================================================================
# CONFIGURATION
# =============================================================================
#
# USING THIS ON YOUR OWN DATA
# ---------------------------
# This script is self-contained: it synthesises its own phantom and needs no
# imaging data, so it runs anywhere once the paths and the ROMEO binary are
# set. It does need the MEDI toolbox for the background removal step, through
# run_medi_batch_PDF_LBV.m.
#
# COIL_PROFILE_FILE is an optional measured coil sensitivity map. If the file
# is absent the simulation runs without it, with uniform sensitivity.
#
# The constants below the paths describe the phantom: tube geometry, gel
# susceptibility, baseline relaxation rates and signal-to-noise per
# formulation, all measured from the phantoms of this study. Changing GEL and
# RESOLUTION is sufficient to switch between conditions.
# =============================================================================

MODE = "generate"          # "generate" -> run_medi_batch_PDF_LBV.m -> "invert"

# ---- PATHS ----
OUT_ROOT = r"C:\QSM_twin\v4_runs"
EXCHANGE = r"C:\QSM_twin\MEDI_EXCHANGE_V4"
ROMEO_PATH = r"D:\romeo\romeo_win_3.2.8\bin\romeo.exe"
COIL_PROFILE_FILE = r"C:\QSM_twin\coil_profile\coil_sensitivity_measured.nii.gz"

# ---- CONDITION ----
GEL = "Alginate"           # "Alginate", "FGT", "Adipose"
RESOLUTION = "070mm"       # "070mm", "086mm"

CHI_HA_PPM = -2.0          # hydroxyapatite, relative to gel
CHI_CAOX_PPM = -1.0        # calcium oxalate, relative to gel

# ---- SAMPLING ----
N_PARTICLE = 5             # particle realisations (different ellipsoids)
N_NOISE = 4                # noise seeds per realisation
SUPERSAMPLE = 2            # 2 simulates signal sub-voxel; 1 disables it

# ---- AIR BUBBLE VARIANT ----
# An air bubble placed beside a particle at a controlled gap, to quantify the
# effect of an adjacent paramagnetic inclusion. Off for recovery runs.
# BUBBLE_PLAN gives the gap in mm per tube; None makes that tube a control.
ADD_BUBBLES = False
BUBBLE_BESIDE = "HA"       # "HA" or "CaOx"
BUBBLE_DIAMETER_MM = 2.5
BUBBLE_PLAN = {0: 2.0, 1: 4.0, 2: None, 3: 0.5}
CHI_AIR_BUBBLE_PPM = 9.4

# ---- SURROUNDING MEDIUM ----
# "air" reproduces how the phantoms were scanned. "water" simulates
# submerging them, which nearly removes the susceptibility step at the tube
# wall. Water also gives signal where air does not, so the variant changes
# the mask as well as the field.
SURROUND = "air"           # "air" or "water"

SKIP_IF_EXISTS = True      # safe to re-run after an interruption


# =============================================================================
# PHANTOM AND ACQUISITION CONSTANTS
# Identical to the measured pipeline where they correspond.
# =============================================================================

B0 = 2.895                 # from the Larmor frequency, 123.244417 MHz
GAMMA = 2.675e8
PAD = 48                   # zero padding for the forward dipole convolution
B0_DIR = [0, 0, 1]         # array coordinates; see qsm_pipeline.py

TUBE_ID_MM, TUBE_WALL_MM = 27.0, 1.5
CHI_TUBE_WALL_PPM = -0.75
SEMI_LONG_MM, SEMI_LONG_SD = 1.10, 0.30
SEMI_SHORT_MM, SEMI_SHORT_SD = 0.70, 0.10
CHI_GEL_PPM = {"Alginate": -9.030, "FGT": -8.794, "Adipose": -8.649}
CHI_AIR_PPM = 0.36
CHI_WATER_PPM = -9.05

# Gel inhomogeneity is set to zero deliberately. The susceptibility variation
# measured in the real gel is entirely short-range: smoothing at 25 mm leaves
# about 1% of it, whereas material inhomogeneity would vary over centimetres
# and survive. What is present is reconstruction noise, correlated over a few
# millimetres by the regularisation and the dipole inversion. Adding a smooth
# susceptibility term gave the twin more large-scale structure than the
# measured data has.
GEL_INHOMOGENEITY_PCT, GEL_INHOMO_SCALE_MM = 0.0, 15.0

# Noise amplitude is scaled so that the gel standard deviation in the twin
# matches the measured one. The detection threshold is three times the gel
# standard deviation, so a twin with too little noise yields a threshold that
# is too low and makes every detection claim derived from it optimistic.
# Susceptibility noise scales roughly linearly with phase noise, so this is a
# single multiplier. Recalibrate it if the acquisition or the pipeline
# changes.
NOISE_SCALE = 1.96

ECHO_TIMES = {"070mm": [2.89, 7.14, 11.39, 15.64, 19.89, 24.14, 28.39],
              "086mm": [2.70, 6.95, 11.20, 15.45, 19.70, 23.95, 28.20]}
VOXEL_MM = {"070mm": 0.70, "086mm": 0.86}
SHAPE = {"070mm": (48, 152, 256), "086mm": (52, 152, 256)}
BASE_R2S = {("Alginate", "070mm"): 7.5, ("Alginate", "086mm"): 9.6,
            ("Adipose", "070mm"): 84.2, ("Adipose", "086mm"): 86.2,
            ("FGT", "070mm"): 5.9, ("FGT", "086mm"): 9.8}
SNR = {("Alginate", "070mm"): 27.3, ("Alginate", "086mm"): 33.9,
       ("Adipose", "070mm"): 44.0, ("Adipose", "086mm"): 64.0,
       ("FGT", "070mm"): 31.1, ("FGT", "086mm"): 52.4}
DR2S_HA, DR2S_CAOX = 175.5, 43.5
TUBE_Z = {"070mm": [43, 93, 161, 212], "086mm": [59, 100, 156, 197]}
HA_Y, CAOX_Y = 34, 72
ROI_RADIUS = 2

# ---- RUNTIME CHECK SETTINGS ----
CHECK_PARTICLE_FIELD = True
PARTICLE_FIELD_FACTOR = 1.5
R2S_CHECK_TOLERANCE = 0.8

MASK_THR, MASK_ERODE, PQ_THR = 0.1, 2, 0.3
WRAP_LIMIT, MIN_TUBE, CROP_MARGIN = 100, 5000, 10
TV_LAM, TV_MU, ADMM_IT, CG_IT, CG_TOL = 0.0005, 0.05, 60, 10, 1e-3


def surround_chi():
    """Susceptibility outside the tubes. Water is close to the gel value, so
    the interface almost vanishes; air is over 9 ppm away from it."""
    return CHI_WATER_PPM if SURROUND == "water" else CHI_AIR_PPM


def surround_signal():
    """Water gives signal, air does not."""
    return 900.0 if SURROUND == "water" else 5.0


def tag_for(p, n):
    b = f"_bub{BUBBLE_BESIDE}" if ADD_BUBBLES else ""
    w = "_water" if SURROUND == "water" else ""
    return f"V4_{GEL}_{RESOLUTION}_p{p}_n{n}{b}{w}"


# =============================================================================
# PHANTOM
# =============================================================================

def random_ellipsoid(rng):
    a = max(0.3, rng.normal(SEMI_LONG_MM, SEMI_LONG_SD))
    b = max(0.2, rng.normal(SEMI_SHORT_MM, SEMI_SHORT_SD))
    c = max(0.2, rng.normal(SEMI_SHORT_MM, SEMI_SHORT_SD))
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q*np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return np.array([a, b, c]), Q


def stamp(vol, value, centre, semi, rot, v_fine):
    rmax = int(np.ceil(semi.max()/v_fine)) + 2
    lo = [max(0, int(centre[d]) - rmax) for d in range(3)]
    hi = [min(vol.shape[d], int(centre[d]) + rmax + 1) for d in range(3)]
    if any(hi[d] <= lo[d] for d in range(3)):
        return np.zeros(0, bool), tuple()
    g = [np.arange(lo[d], hi[d], dtype=np.float32) - centre[d]
         for d in range(3)]
    X, Y, Z = np.meshgrid(*g, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()])*v_fine
    loc = rot.T @ pts
    q = ((loc[0]/semi[0])**2 + (loc[1]/semi[1])**2 + (loc[2]/semi[2])**2)
    inside = (q < 1.0).reshape(X.shape)
    box = tuple(slice(lo[d], hi[d]) for d in range(3))
    vol[box][inside] = value
    return inside, box


def block_mean(a, ss):
    if ss == 1:
        return a
    s = a.shape
    return a.reshape(s[0]//ss, ss, s[1]//ss, ss, s[2]//ss, ss).mean((1, 3, 5))


def build_phantom(p_seed):
    """Particle shapes depend only on p_seed, so a realisation is repeatable
    across noise seeds."""
    ss = SUPERSAMPLE
    v = VOXEL_MM[RESOLUTION]
    vf = v/ss
    sh = SHAPE[RESOLUTION]
    fsh = tuple(s*ss for s in sh)
    rng = np.random.default_rng(1000 + p_seed)

    chi = np.full(fsh, surround_chi(), dtype=np.float32)
    tube = np.zeros(fsh, bool)
    ha = np.zeros(fsh, bool)
    cx = np.zeros(fsh, bool)

    c0 = fsh[0]/2.0
    r_in, r_out = (TUBE_ID_MM/2)/vf, (TUBE_ID_MM/2 + TUBE_WALL_MM)/vf
    X = np.arange(fsh[0], dtype=np.float32)[:, None] - c0
    for cz in TUBE_Z[RESOLUTION]:
        czf = cz*ss + ss/2.0
        Z = np.arange(fsh[2], dtype=np.float32)[None, :] - czf
        r = np.sqrt(X**2 + Z**2)
        inner, wall = r <= r_in, (r > r_in) & (r <= r_out)
        for iy in range(fsh[1]):
            tube[:, iy, :] |= inner
            chi[:, iy, :][inner] = CHI_GEL_PPM[GEL]
            chi[:, iy, :][wall] = CHI_TUBE_WALL_PPM

    inhomo = np.ones(fsh, np.float32)
    if GEL_INHOMOGENEITY_PCT > 0:
        n = rng.normal(0, 1, fsh).astype(np.float32)
        sm = gaussian_filter(n, sigma=GEL_INHOMO_SCALE_MM/vf)
        inhomo = (1 + (GEL_INHOMOGENEITY_PCT/100)*sm
                  / (sm.std()+1e-9)).astype(np.float32)
        chi[tube] += (inhomo[tube]-1)*abs(CHI_GEL_PPM[GEL])*0.1

    bub = np.zeros(fsh, bool)
    for i, cz in enumerate(TUBE_Z[RESOLUTION]):
        czf = cz*ss + ss/2.0
        for y0, cp, m in [(HA_Y, CHI_HA_PPM, ha), (CAOX_Y, CHI_CAOX_PPM, cx)]:
            semi, rot = random_ellipsoid(rng)
            ins, box = stamp(chi, CHI_GEL_PPM[GEL]+cp,
                             (c0, y0*ss + ss/2.0, czf), semi, rot, vf)
            if ins.size:
                m[box][ins] = True
        if ADD_BUBBLES:
            gap = BUBBLE_PLAN.get(i)
            if gap is not None:
                y_t = HA_Y if BUBBLE_BESIDE == "HA" else CAOX_Y
                sep = (SEMI_LONG_MM + BUBBLE_DIAMETER_MM/2 + gap)/vf
                ins, box = stamp(chi, CHI_GEL_PPM[GEL] + CHI_AIR_BUBBLE_PPM,
                                 (c0 + sep, y_t*ss + ss/2.0, czf),
                                 np.full(3, BUBBLE_DIAMETER_MM/2),
                                 np.eye(3), vf)
                if ins.size:
                    bub[box][ins] = True
    return chi, tube, ha, cx, bub, inhomo, vf


def dipole(sh, v):
    kx = np.fft.fftfreq(sh[0], d=v).astype(np.float32)
    ky = np.fft.fftfreq(sh[1], d=v).astype(np.float32)
    kz = np.fft.fftfreq(sh[2], d=v).astype(np.float32)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2+KY**2+KZ**2
    K2[0, 0, 0] = 1.0
    D = (1/3 - KZ**2/K2).astype(np.float32)
    D[0, 0, 0] = 0.0
    return D


def signal(chi_f, tube_f, ha_f, cx_f, bub_f, inhomo_f, vf, n_seed):
    ss = SUPERSAMPLE
    sh = SHAPE[RESOLUTION]
    TEs = np.array(ECHO_TIMES[RESOLUTION])/1000.0
    rng = np.random.default_rng(50000 + n_seed)

    pad = PAD*ss
    cp = np.pad(chi_f, pad, constant_values=surround_chi())
    f = np.real(ifftn(fftn(cp)*dipole(cp.shape, vf)))[pad:-pad, pad:-pad,
                                                      pad:-pad]
    del cp
    fhz = (f*1e-6*GAMMA*B0/(2*np.pi)).astype(np.float32)
    del f
    # Linear shim residual, approximating the measured background field.
    zz = np.linspace(-1, 1, fhz.shape[2], dtype=np.float32)[None, None, :]
    yy = np.linspace(-1, 1, fhz.shape[1], dtype=np.float32)[None, :, None]
    fhz = fhz + 300.0*zz + 80.0*yy

    r2s = np.full(fhz.shape, 1.0, np.float32)
    r2s[tube_f] = BASE_R2S[(GEL, RESOLUTION)]
    r2s[ha_f] = BASE_R2S[(GEL, RESOLUTION)] + DR2S_HA
    r2s[cx_f] = BASE_R2S[(GEL, RESOLUTION)] + DR2S_CAOX
    if ADD_BUBBLES:
        r2s[bub_f] = 300.0
    S0 = np.where(tube_f, np.float32(1000.0),
                  np.float32(surround_signal()))*inhomo_f
    if ADD_BUBBLES:
        S0[bub_f] = 5.0

    coil = None
    if os.path.exists(COIL_PROFILE_FILE):
        coil = nib.load(COIL_PROFILE_FILE).get_fdata().astype(np.float32)

    # ---------------------------------------------------------------
    # CHECKS on the field, before any signal is synthesised.
    # ---------------------------------------------------------------
    # The field must not be identically zero inside the tubes, which is what
    # a kernel normalisation error or a uniform susceptibility map produces.
    assert np.abs(fhz[tube_f]).std() > 1e-3, (
        "field inside the tubes has no variation: check the dipole kernel "
        "and that the susceptibility map is not uniform")

    # The particle must produce a LOCAL field perturbation, compared against
    # gel in its own neighbourhood rather than the whole volume. The shim
    # gradient spans hundreds of hertz across the phantom, so gel at opposite
    # ends differs far more than any particle perturbs its surroundings, and
    # a whole-volume comparison fails for a reason unrelated to the particle.
    gel_only = tube_f & ~ha_f & ~cx_f
    if ha_f.sum() > 10:
        near_gel = ndimage.binary_dilation(ha_f, iterations=6) & gel_only
        if near_gel.sum() < 50:
            near_gel = gel_only
        local_mean = fhz[near_gel].mean()
        near = np.abs(fhz[ha_f] - local_mean).mean()
        typ = np.abs(fhz[near_gel] - local_mean).mean()
        ok = near > PARTICLE_FIELD_FACTOR*typ
        msg = (f"field at the particle ({near:.1f} Hz from the local gel "
               f"mean) is not distinguishable from local gel variation "
               f"({typ:.1f} Hz): the particle may not be producing a field")
        if CHECK_PARTICLE_FIELD:
            assert ok, msg
        elif not ok:
            print(f"    WARNING: {msg}")

    sig_n = NOISE_SCALE*1000.0/SNR[(GEL, RESOLUTION)]
    mag = np.zeros(sh+(len(TEs),), np.float32)
    pha = np.zeros(sh+(len(TEs),), np.float32)
    for e, te in enumerate(TEs):
        amp = S0*np.exp(-r2s*np.float32(te))
        ph = np.float32(2*np.pi*te)*fhz
        sf = amp*np.exp(1j*ph)
        s = block_mean(sf.real, ss) + 1j*block_mean(sf.imag, ss)
        del sf, amp, ph
        if coil is not None:
            s = s*coil
        s = (s.real + rng.normal(0, sig_n, sh)
             + 1j*(s.imag + rng.normal(0, sig_n, sh)))
        mag[..., e] = np.abs(s)
        pha[..., e] = np.angle(s)

    gel_c = block_mean(gel_only.astype(np.float32), ss) > 0.99
    if gel_c.sum() > 500:
        prof = np.array([mag[..., e][gel_c].mean() for e in range(len(TEs))])
        A = np.stack([np.ones(len(TEs)), -TEs], axis=1)
        coef, *_ = np.linalg.lstsq(A, np.log(np.maximum(prof, 1e-9)),
                                   rcond=None)
        r2_fit, r2_set = float(coef[1]), BASE_R2S[(GEL, RESOLUTION)]
        assert abs(r2_fit - r2_set) < max(R2S_CHECK_TOLERANCE*r2_set, 3.0), (
            f"gel R2* recovered as {r2_fit:.1f} s^-1 from the synthesised "
            f"signal but {r2_set:.1f} was specified: check the echo time "
            f"units in the signal synthesis")
        print(f"    check: gel R2* {r2_fit:.1f} s^-1 recovered "
              f"({r2_set:.1f} specified)")

    return mag, (pha/np.pi*4096).astype(np.float32)


# =============================================================================
# PIPELINE, identical to the measured one
# =============================================================================

def phase_quality(phase, mag):
    n = phase.shape[3]
    cs = mag*np.exp(1j*phase)
    acc = np.zeros(phase.shape[:3], np.complex128)
    for e in range(1, n):
        d = cs[..., e]*np.conj(cs[..., e-1])
        acc += d/(np.abs(d)+1e-10)
    return np.abs(acc)/(n-1)


def make_mask(mag, phase):
    avg = np.mean(mag, axis=3)
    m = binary_fill_holes(avg > MASK_THR*avg.max())
    m = m & (phase_quality(phase, mag) > PQ_THR)
    m = binary_fill_holes(m)
    st = generate_binary_structure(3, 1)
    for _ in range(MASK_ERODE):
        m = binary_erosion(m, structure=st)
    return m.astype(np.float64)


def romeo(phase, mag, TEs, affine):
    if not os.path.exists(ROMEO_PATH):
        raise FileNotFoundError(
            f"ROMEO not found at {ROMEO_PATH}. Download it from "
            f"https://github.com/korbinian90/ROMEO/releases")
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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            r = subprocess.run(cmd+["--individual-unwrapping"],
                               capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                raise RuntimeError(r.stderr)
        up = os.path.join(out, "unwrapped.nii")
        if not os.path.exists(up):
            up = os.path.join(out, "unwrapped.nii.gz")
        return nib.load(up).get_fdata().astype(np.float64)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def romeo_per_tube(phase, mag, TEs, affine, mask):
    lab, n = label(mask > 0, structure=generate_binary_structure(3, 1))
    sizes = [(lab == i).sum() for i in range(1, n+1)]
    keep = [i+1 for i in range(n) if sizes[i] >= MIN_TUBE]
    un = np.zeros_like(phase)
    sl = find_objects(lab)
    for l in keep:
        s0 = sl[l-1]
        w = tuple(slice(max(0, x.start-CROP_MARGIN),
                        min(phase.shape[d], x.stop+CROP_MARGIN))
                  for d, x in enumerate(s0))
        sub = romeo(phase[w], mag[w], TEs, affine)
        this = (lab == l)[w]
        for e in range(phase.shape[3]):
            t = un[w][..., e]
            t[this] = sub[..., e][this]
            un[w+(e,)] = t
    return un


def field_map(un, TEs, mag, mask):
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


def inv_kernels(sh, v):
    kx = np.fft.fftfreq(sh[0], v)
    ky = np.fft.fftfreq(sh[1], v)
    kz = np.fft.fftfreq(sh[2], v)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2+KY**2+KZ**2
    k2[0, 0, 0] = 1e-10
    b = np.array(B0_DIR)/np.linalg.norm(B0_DIR)
    D = 1/3-(KX*b[0]+KY*b[1]+KZ*b[2])**2/k2
    D[0, 0, 0] = 0
    lap = ((2-2*np.cos(2*np.pi*KX*v))/v**2 + (2-2*np.cos(2*np.pi*KY*v))/v**2
           + (2-2*np.cos(2*np.pi*KZ*v))/v**2)
    return D, lap


def _g(c, v):
    return ((np.roll(c, -1, 0)-c)/v, (np.roll(c, -1, 1)-c)/v,
            (np.roll(c, -1, 2)-c)/v)


def _d(gx, gy, gz, v):
    return ((gx-np.roll(gx, 1, 0))/v + (gy-np.roll(gy, 1, 1))/v
            + (gz-np.roll(gz, 1, 2))/v)


def _GtG(x, v):
    """Adjoint of the forward-difference gradient is minus the divergence."""
    gx, gy, gz = _g(x, v)
    return -_d(gx, gy, gz, v)


def cg(rhs, chi0, mask, D, lap, mu, v):
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
    for _ in range(CG_IT):
        Ap = A(p)
        pAp = np.sum(p*Ap)
        if abs(pAp) < 1e-30:
            break
        a = rz/pAp
        x += a*p
        r -= a*Ap
        if np.linalg.norm(r)/r0 < CG_TOL:
            break
        z = M(r)
        rzn = np.sum(r*z)
        p = z+(rzn/rz)*p
        rz = rzn
    return x*mask


def invert(lf, mask, v):
    f = lf/(42.576e6*B0)*1e6
    D, lap = inv_kernels(lf.shape, v)
    rhs0 = mask*np.real(ifftn(D*fftn(f*mask)))
    Di = np.zeros_like(D)
    k = np.abs(D) > 0.2
    Di[k] = 1/D[k]
    Di[~k] = np.sign(D[~k])/0.2
    chi = np.real(ifftn(fftn(f*mask)*Di))*mask
    gx, gy, gz = _g(chi, v)
    zx, zy, zz = gx.copy(), gy.copy(), gz.copy()
    ux, uy, uz = np.zeros_like(gx), np.zeros_like(gy), np.zeros_like(gz)
    lm = TV_LAM/TV_MU
    for _ in range(ADMM_IT):
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


def sphere_idx(c, shape, R=ROI_RADIUS):
    r = int(R)
    x, y, z = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    keep = (x**2+y**2+z**2) <= R**2
    idx = np.array(c)[None, :] + np.stack([x[keep], y[keep], z[keep]], 1)
    ok = np.all((idx >= 0) & (idx < np.array(shape)[None, :]), 1)
    idx = idx[ok]
    return idx[:, 0], idx[:, 1], idx[:, 2]


# =============================================================================
# MODES
# =============================================================================

def generate_all():
    os.makedirs(EXCHANGE, exist_ok=True)
    TEs = ECHO_TIMES[RESOLUTION]
    v = VOXEL_MM[RESOLUTION]
    affine = np.diag([v, v, v, 1.0])
    total = N_PARTICLE*N_NOISE

    print("=" * 74)
    print(f"  TWIN RECOVERY BATCH: {N_PARTICLE} particle realisations x "
          f"{N_NOISE} noise seeds = {total} runs")
    print(f"  supersample {SUPERSAMPLE}x, chi_HA {CHI_HA_PPM:+.2f} ppm")
    print(f"  surround: {SURROUND} ({surround_chi():+.2f} ppm)")
    print("=" * 74)

    k = 0
    for p in range(N_PARTICLE):
        print(f"\n  --- particle realisation {p} ---")
        chi_f, tube_f, ha_f, cx_f, bub_f, inh_f, vf = build_phantom(p)
        # ---------------------------------------------------------------
        # CHECKS on the phantom. Each guards against a build that looks
        # plausible but has the wrong contents, which is far harder to spot
        # downstream than an outright failure.
        # ---------------------------------------------------------------
        vox_mm = VOXEL_MM[RESOLUTION]
        n_tubes = len(TUBE_Z[RESOLUTION])

        # Particle occupancy against analytic geometry. A binary particle
        # mask gives roughly 1.6 times the fractional value and removes
        # partial volume by construction, so this catches that substitution.
        ha_frac = block_mean(ha_f.astype(np.float32), SUPERSAMPLE)
        occ = ha_frac.sum()/n_tubes
        expect = (4/3*np.pi*SEMI_LONG_MM*SEMI_SHORT_MM**2)/vox_mm**3
        assert 0.4*expect < occ < 2.5*expect, (
            f"HA occupancy {occ:.2f} voxels per particle is implausible "
            f"against {expect:.2f} expected from the mean semi-axes")

        # Partial volume must be present. If every particle voxel is fully
        # occupied the build has gone binary, and partial volume cannot then
        # be measured because the phantom no longer contains any.
        partial = ((ha_frac > 0.01) & (ha_frac < 0.99)).sum()
        full = (ha_frac >= 0.99).sum()
        assert partial > full, (
            f"only {partial} partial-occupancy voxels against {full} full: "
            f"the particle build is effectively binary and partial volume "
            f"is absent by construction")

        # Gel susceptibility must be the value for this formulation.
        gel_only = tube_f & ~ha_f & ~cx_f
        assert abs(float(chi_f[gel_only].mean()) - CHI_GEL_PPM[GEL]) < 0.05, (
            f"gel susceptibility {float(chi_f[gel_only].mean()):+.3f} does "
            f"not match {CHI_GEL_PPM[GEL]:+.3f} for {GEL}")

        # Particle susceptibility relative to gel.
        assert abs(float(chi_f[ha_f].mean()) - CHI_GEL_PPM[GEL]
                   - CHI_HA_PPM) < 0.05, "HA susceptibility wrong"

        # Ground truth is the same for every noise seed of this realisation.
        gt = block_mean(
            np.where(tube_f | ha_f | cx_f,
                     chi_f - CHI_GEL_PPM[GEL], 0.0).astype(np.float32),
            SUPERSAMPLE)

        # Ground truth must carry the fractional mixture, not a binary mask.
        # If it takes only a few distinct values, partial volume has been
        # lost between the phantom and the ground truth even though the
        # phantom itself had it.
        gt_ha = gt[ha_frac > 0.01]
        assert len(np.unique(np.round(gt_ha, 3))) > 3, (
            "ground truth over the particle takes too few distinct values: "
            "the fractional mixture has been lost")
        f_ha = block_mean(ha_f.astype(np.float32), SUPERSAMPLE)
        f_cx = block_mean(cx_f.astype(np.float32), SUPERSAMPLE)
        f_bb = block_mean(bub_f.astype(np.float32), SUPERSAMPLE)

        for n in range(N_NOISE):
            k += 1
            tag = tag_for(p, n)
            out = os.path.join(OUT_ROOT, tag)
            matp = os.path.join(EXCHANGE, f"in_{tag}.mat")
            if SKIP_IF_EXISTS and os.path.exists(matp):
                print(f"    [{k}/{total}] {tag}: exists, skipping")
                continue
            os.makedirs(out, exist_ok=True)

            mag, pha = signal(chi_f, tube_f, ha_f, cx_f, bub_f, inh_f,
                              vf, p*100+n)
            pr = pha.astype(np.float64)*np.pi/4096
            mf = mag.astype(np.float64)
            mask = make_mask(mf, pr)
            un = romeo(pr, mf, TEs, affine)
            fld = field_map(un, TEs, mf, mask)
            w = count_wraps(fld, mask, TEs)
            if w > WRAP_LIMIT:
                un = romeo_per_tube(pr, mf, TEs, affine, mask)
                fld = field_map(un, TEs, mf, mask)
                w = count_wraps(fld, mask, TEs)

            nib.save(nib.Nifti1Image(gt, affine),
                     os.path.join(out, "ground_truth_chi_ppm.nii.gz"))
            nib.save(nib.Nifti1Image(((f_ha > 0.5).astype(np.float32)
                                      + 2*(f_cx > 0.5).astype(np.float32)
                                      + 3*(f_bb > 0.5).astype(np.float32)),
                                     affine),
                     os.path.join(out, "particle_mask.nii.gz"))
            nib.save(nib.Nifti1Image(mask.astype(np.float32), affine),
                     os.path.join(out, "mask.nii.gz"))
            nib.save(nib.Nifti1Image(fld.astype(np.float32), affine),
                     os.path.join(out, "field_map_hz.nii.gz"))
            # The synthesised magnitude is saved because MEDI's
            # morphological prior needs it. Substituting a synthetic image
            # would make that prior meaningless and any comparison against
            # MEDI unfair.
            nib.save(nib.Nifti1Image(mag.astype(np.float32), affine),
                     os.path.join(out, "magnitude.nii.gz"))
            savemat(matp, {"iFreq": fld*2*np.pi, "Mask": mask,
                           "N_std": mask.astype(np.float64),
                           "matrix_size": np.array(fld.shape, float),
                           "voxel_size": np.array([v, v, v]),
                           "B0_dir": np.array(B0_DIR, float)},
                    do_compression=True)
            print(f"    [{k}/{total}] {tag}: mask {mask.sum():.0f}, "
                  f"wraps {w}")

        del chi_f, tube_f, ha_f, cx_f, bub_f, inh_f

    print(f"""
  NEXT
  ----
  1. Run run_medi_batch_PDF_LBV.m in MATLAB, with its EXCHANGE set to
         {EXCHANGE}
  2. Return here with MODE = "invert".
""")


def _analyse(chi, gt, pm, mask, valid, gel_mean, p, n, rows):
    """Per-tube recovery, measured as the measured-data analysis measures it.

    Both statistics are taken over the same radius-2 sphere so that they can
    be compared with each other, and so that they match what the results
    tables report. Values over the particle mask are recorded alongside for
    reference, but the mask is not available for measured data.
    """
    vm = valid > 0
    gel = vm & (pm == 0) & mask
    tubes, nt = ndimage.label(
        mask, structure=ndimage.generate_binary_structure(3, 1))
    sizes = ndimage.sum(mask, tubes, range(1, nt+1))
    keep = sorted([i+1 for i in np.argsort(sizes)[::-1][:4]
                   if sizes[i] > 5000],
                  key=lambda l: np.argwhere(tubes == l)[:, 2].mean())
    for ti, l in enumerate(keep):
        tm = tubes == l
        ha = (pm == 1) & tm & vm
        if ha.sum() == 0:
            continue
        cen = np.array(ndimage.center_of_mass((pm == 1) & tm)).round()
        i_, j_, k_ = sphere_idx(tuple(int(x) for x in cen), chi.shape)
        sel = vm[i_, j_, k_]
        roi_chi = chi[i_[sel], j_[sel], k_[sel]] - gel_mean
        roi_gt = gt[i_[sel], j_[sel], k_[sel]]
        rows.append(dict(
            particle=p, noise=n, tube=ti+1,
            input_chi=CHI_HA_PPM,
            gt_mask=gt[(pm == 1) & tm].mean(),
            gt_roi=roi_gt.mean(),
            gt_roi_p5=np.percentile(roi_gt, 5),
            rec_mask=chi[ha].mean() - gel_mean,
            rec_mask_p5=np.percentile(chi[ha], 5) - gel_mean,
            rec_roi=roi_chi.mean(),
            rec_roi_p5=np.percentile(roi_chi, 5),
            gel_sd=chi[gel].std()))


def invert_all():
    vox = VOXEL_MM[RESOLUTION]
    rows = []
    total = N_PARTICLE*N_NOISE
    k = 0
    print("=" * 74)
    print("  TWIN RECOVERY BATCH: inverting")
    print("=" * 74)

    for p in range(N_PARTICLE):
        for n in range(N_NOISE):
            k += 1
            tag = tag_for(p, n)
            out = os.path.join(OUT_ROOT, tag)
            matp = os.path.join(EXCHANGE, f"out_{tag}.mat")
            if not os.path.exists(matp):
                print(f"  [{k}/{total}] {tag}: no MATLAB output, skipping")
                continue

            done_p = os.path.join(out, "qsm_susceptibility_ppm.nii.gz")
            if SKIP_IF_EXISTS and os.path.exists(done_p):
                chi = nib.load(done_p).get_fdata()
                valid = nib.load(os.path.join(
                    out, "vsharp_valid_mask.nii.gz")).get_fdata()
                gt = nib.load(os.path.join(
                    out, "ground_truth_chi_ppm.nii.gz")).get_fdata()
                pm = nib.load(os.path.join(
                    out, "particle_mask.nii.gz")).get_fdata()
                mask = nib.load(os.path.join(out,
                                             "mask.nii.gz")).get_fdata() > 0
                vm = valid > 0
                gel = vm & (pm == 0) & mask
                gel_mean = chi[gel].mean()
                _analyse(chi, gt, pm, mask, valid, gel_mean, p, n, rows)
                print(f"  [{k}/{total}] {tag}: reused saved inversion")
                continue

            d = loadmat(matp)
            lf = np.asarray(d["RDF_PDF"], float)/(2*np.pi)
            valid = np.asarray(d["Mask_PDF"], float)
            chi = invert(lf, valid, vox)
            # The inversion is the slow step, so it is saved: a later change
            # to the region analysis then needs no re-inversion.
            nib.save(nib.Nifti1Image(chi.astype(np.float32),
                                     np.diag([vox, vox, vox, 1.0])),
                     os.path.join(out, "qsm_susceptibility_ppm.nii.gz"))
            nib.save(nib.Nifti1Image(valid.astype(np.float32),
                                     np.diag([vox, vox, vox, 1.0])),
                     os.path.join(out, "vsharp_valid_mask.nii.gz"))

            gt = nib.load(
                os.path.join(out, "ground_truth_chi_ppm.nii.gz")).get_fdata()
            pm = nib.load(
                os.path.join(out, "particle_mask.nii.gz")).get_fdata()
            mask = nib.load(os.path.join(out, "mask.nii.gz")).get_fdata() > 0
            vm = valid > 0
            # Gel is inside a tube and not a particle. Selecting on gt == 0
            # would be wrong whenever gel susceptibility varies, since it is
            # then never exactly zero and the selection comes out empty.
            gel = vm & (pm == 0) & mask
            if gel.sum() < 100:
                print(f"  [{k}/{total}] {tag}: gel region empty, skipping")
                continue
            gel_mean = chi[gel].mean()

            _analyse(chi, gt, pm, mask, valid, gel_mean, p, n, rows)
            print(f"  [{k}/{total}] {tag}: done")

    if not rows:
        print("\n  Nothing to report. Run MODE = \"generate\" and the MATLAB "
              "step first.")
        return

    a = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    n_meas = len(rows)

    print("\n\n" + "=" * 78)
    print(f"  RECOVERY, {n_meas} measurements "
          f"({N_PARTICLE} particle realisations x {N_NOISE} noise seeds "
          f"x 4 tubes)")
    print("=" * 78)

    print("\n  MEASURED OVER THE PARTICLE MASK")
    print(f"    input chi                {CHI_HA_PPM:+.4f} ppm")
    print(f"    ground truth in mask     {a['gt_mask'].mean():+.4f} "
          f"+- {a['gt_mask'].std(ddof=1):.4f}   "
          f"({100*a['gt_mask'].mean()/CHI_HA_PPM:.1f}% of input)")
    print(f"    recovered                {a['rec_mask'].mean():+.4f} "
          f"+- {a['rec_mask'].std(ddof=1):.4f}   "
          f"({100*a['rec_mask'].mean()/CHI_HA_PPM:.1f}% of input)")

    print("\n  MEASURED OVER A RADIUS-2 ROI  (as the measured analysis does)")
    print(f"    {'':<22}{'value (ppm)':>16}{'% of input':>13}{'CV':>8}")
    print("    " + "-" * 59)
    for name, key in [("ground truth, mean", "gt_roi"),
                      ("ground truth, 5th pct", "gt_roi_p5"),
                      ("RECOVERED, mean", "rec_roi"),
                      ("RECOVERED, 5th pct", "rec_roi_p5")]:
        vals = a[key]
        cv = (100*vals.std(ddof=1)/abs(vals.mean())
              if vals.mean() else np.nan)
        print(f"    {name:<22}{vals.mean():>+10.4f} "
              f"+-{vals.std(ddof=1):>5.4f}"
              f"{100*vals.mean()/CHI_HA_PPM:>12.1f}%{cv:>7.1f}%")

    print(f"""
  The CV column is the scatter of each statistic relative to its own size,
  across {n_meas} measurements of an identical phantom.

  The mean is heavily diluted by partial volume but should be stable. The 5th
  percentile sits closer to the particle's own susceptibility but is a tail
  statistic and may not be. Whichever has the lower CV is the more
  reproducible measurement, and is the one to quote as a value; the other may
  still suit the detection decision, where a threshold survives scatter that
  a quoted number does not.""")

    pv = 100*a['gt_roi'].mean()/CHI_HA_PPM
    pipe = 100*a['rec_roi'].mean()/a['gt_roi'].mean()
    tot = 100*a['rec_roi'].mean()/CHI_HA_PPM

    print("\n" + "=" * 78)
    print("  THE LOSS, SPLIT  (radius-2 ROI, as reported in the results)")
    print("=" * 78)
    print(f"    partial volume keeps     {pv:.1f}% of the input")
    print(f"    the pipeline then keeps  {pipe:.1f}% of what remains")
    print(f"    end to end               {tot:.1f}%")

    print("\n  WHERE THE VARIABILITY COMES FROM (recovered, radius-2 ROI)")
    by_p = [a['rec_roi'][a['particle'] == p].mean()
            for p in range(N_PARTICLE)]
    by_n = [a['rec_roi'][a['noise'] == n].mean() for n in range(N_NOISE)]
    by_t = [a['rec_roi'][a['tube'] == t].mean() for t in range(1, 5)]
    print(f"    between particle realisations  SD "
          f"{np.std(by_p, ddof=1) if len(by_p) > 1 else 0:.4f} ppm")
    print(f"    between noise seeds            SD "
          f"{np.std(by_n, ddof=1) if len(by_n) > 1 else 0:.4f} ppm")
    print(f"    between tubes                  SD "
          f"{np.std(by_t, ddof=1):.4f} ppm")
    print("""
    If the between-particle SD dominates, the uncertainty comes from not
    knowing the exact particle shape and more noise repeats will not reduce
    it. If the between-noise SD dominates, the reverse holds.""")

    os.makedirs(OUT_ROOT, exist_ok=True)
    p_csv = os.path.join(OUT_ROOT, "v4_recovery.csv")
    with open(p_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {p_csv}")

    print(f"""
  The two figures to report separately:

    partial volume alone reduces the measurable susceptibility to {pv:.0f}%
    of the true value for a {2*SEMI_LONG_MM:.1f} mm particle in
    {VOXEL_MM[RESOLUTION]} mm voxels;

    the reconstruction pipeline then recovers {pipe:.0f}% of what partial
    volume leaves, giving {tot:.0f}% end to end.
""")


if __name__ == "__main__":
    if MODE == "generate":
        generate_all()
    elif MODE == "invert":
        invert_all()
    else:
        raise ValueError(f'MODE must be "generate" or "invert", not {MODE!r}')