#!/usr/bin/env python3
"""
Background field removal compared against twin ground truth
===========================================================

V-SHARP, PDF and LBV applied to the same simulated field map, each inverted
and compared against a known susceptibility. Self-contained apart from one
MATLAB step: it builds the phantom, preps it, and does the comparison.

WHY GROUND TRUTH IS NEEDED FOR THIS

On measured data the methods can only be compared on proxy metrics: coverage,
striping, and the standard deviation of a gel reference region. A lower gel
standard deviation is ambiguous, because it can mean either better background
removal or more smoothing, and the two are indistinguishable without a known
answer. Smoothing also reduces the recovered particle susceptibility, so
ground truth separates them: a method with the lowest gel standard deviation
but not the highest recovery is smoothing rather than removing background.

This is a different claim from the three methods agreeing on measured data.
That agreement says the detection outcome is not an artefact of one
pipeline; it says nothing about which method is most accurate.

THE BACKGROUND FIELD

By default the phantom carries a smooth synthetic shim, and that is the
configuration the published comparison used. An option exists to substitute a
polynomial fit to a measured background field instead, which is a harder and
more realistic problem: a measured background is several times stronger than
the synthetic shim and roughly a third of it is non-linear, whereas the
synthetic shim is linear by construction. PDF models the background as dipole
sources outside the mask, which is close to what a smooth gradient is, so a
purely linear background hands PDF the problem it is built for.

Running with the measured background therefore tests the methods on the
harder problem, but it needs a background field file derived from measured
data, and it carries a caveat worth stating wherever it is reported: a
measured background is obtained as total field minus local field, so it is
partly defined by the method that removed it. Two points mitigate that. The
field outside the tubes arises from physical sources whichever method removes
it, and all three methods are given the same background, so the comparison
between them is fair even if the background is not perfectly method-neutral.

HOW TO RUN

    MODE = "prep"               build the phantom, unwrap, field map, write
                                one input .mat per particle realisation
    run_medi_batch_PDF_LBV.m    ONCE over the exchange folder; it must
                                produce both PDF and LBV
    MODE = "compare"            V-SHARP here in Python, PDF and LBV from the
                                MATLAB output, all three inverted and
                                compared against ground truth

Every realisation in PARTICLE_SEEDS is processed in one pass, each into its
own subfolder, since particle geometry is the dominant source of variability
and a single realisation is not representative of the mean.
"""

import os
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
# EDIT THESE
# =============================================================================


MODE = "prep"
 
# One subfolder per seed. prep() writes field_map_hz.nii.gz, mask.nii.gz and
# the ground truth under fixed names, so without this each run would
# overwrite the last and compare() would read the wrong phantom.
OUT_DIR_BASE = r"C:\QSM_twin\method_comparison"
OUT_DIR = OUT_DIR_BASE          # set per seed by the loop at the bottom
EXCHANGE = r"C:\QSM_twin\MEDI_EXCHANGE_COMPARE"
ROMEO_PATH = r"D:\romeo\romeo_win_3.2.8\bin\romeo.exe"
COIL_PROFILE_FILE = r"C:\QSM_twin\coil_profile\coil_sensitivity_measured.nii.gz"
 
GEL = "Alginate"
RESOLUTION = "070mm"
CHI_HA_PPM = -2.0
CHI_CAOX_PPM = -1.0
# Particle realisations. All of them are processed in one pass, each into its
# own subfolder, because particle geometry is the dominant source of
# variability and one realisation is not the mean. Use the same number of
# realisations here as the recovery table averages, so the two are comparable.
PARTICLE_SEEDS = [0, 1, 2, 3, 4]
PARTICLE_SEED = PARTICLE_SEEDS[0]     # set per iteration by the loop
NOISE_SEED = 0
SUPERSAMPLE = 2
NOISE_SCALE = 1.96
 
# The seed is in the tag so the five runs do not overwrite one another.
TAG = f"COMPARE_{GEL}_{RESOLUTION}_p{PARTICLE_SEED}"
 
# V-SHARP settings, matching the implementation it is compared against
VSHARP_RADII_MM = [4, 6, 8, 10]
VSHARP_DECONV_THRESHOLD = 0.1
 
# =============================================================================
# CONSTANTS
# =============================================================================
 
B0 = 2.895
GAMMA = 2.675e8
PAD = 48
B0_DIR = [0, 0, 1]
TUBE_ID_MM, TUBE_WALL_MM = 27.0, 1.5
CHI_TUBE_WALL_PPM = -0.75
SEMI_LONG_MM, SEMI_LONG_SD = 1.10, 0.30
SEMI_SHORT_MM, SEMI_SHORT_SD = 0.70, 0.10
CHI_GEL_PPM = {"Alginate": -9.030, "FGT": -8.794, "Adipose": -8.649}
CHI_AIR_PPM = 0.36
ECHO_TIMES = {"070mm": [2.89, 7.14, 11.39, 15.64, 19.89, 24.14, 28.39],
              "086mm": [2.70, 6.95, 11.20, 15.45, 19.70, 23.95, 28.20]}
VOXEL_MM = {"070mm": 0.70, "086mm": 0.86}
SHAPE = {"070mm": (48, 152, 256), "086mm": (52, 152, 256)}
BASE_R2S = {("Alginate", "070mm"): 7.5, ("Alginate", "086mm"): 9.6}
SNR = {("Alginate", "070mm"): 27.3, ("Alginate", "086mm"): 33.9}
DR2S_HA, DR2S_CAOX = 175.5, 43.5
TUBE_Z = {"070mm": [43, 93, 161, 212], "086mm": [59, 100, 156, 197]}
HA_Y, CAOX_Y = 34, 72
MASK_THR, MASK_ERODE, PQ_THR = 0.1, 2, 0.3
WRAP_LIMIT, MIN_TUBE, CROP_MARGIN = 100, 5000, 10
TV_LAM, TV_MU, ADMM_IT, CG_IT, CG_TOL = 0.0005, 0.05, 60, 10, 1e-3
ROI_RADIUS = 2
 
 
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
 
 
def stamp(vol, value, centre, semi, rot, vf):
    rmax = int(np.ceil(semi.max()/vf)) + 2
    lo = [max(0, int(centre[d]) - rmax) for d in range(3)]
    hi = [min(vol.shape[d], int(centre[d]) + rmax + 1) for d in range(3)]
    if any(hi[d] <= lo[d] for d in range(3)):
        return np.zeros(0, bool), tuple()
    g = [np.arange(lo[d], hi[d], dtype=np.float32) - centre[d]
         for d in range(3)]
    X, Y, Z = np.meshgrid(*g, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()])*vf
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
 
 
def build_phantom():
    ss = SUPERSAMPLE
    v = VOXEL_MM[RESOLUTION]
    vf = v/ss
    sh = SHAPE[RESOLUTION]
    fsh = tuple(x*ss for x in sh)
    rng = np.random.default_rng(1000 + PARTICLE_SEED)
 
    chi = np.full(fsh, CHI_AIR_PPM, dtype=np.float32)
    tube = np.zeros(fsh, bool); ha = np.zeros(fsh, bool)
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
 
    for cz in TUBE_Z[RESOLUTION]:
        czf = cz*ss + ss/2.0
        for y0, cp, m in [(HA_Y, CHI_HA_PPM, ha), (CAOX_Y, CHI_CAOX_PPM, cx)]:
            semi, rot = random_ellipsoid(rng)
            ins, box = stamp(chi, CHI_GEL_PPM[GEL]+cp,
                             (c0, y0*ss + ss/2.0, czf), semi, rot, vf)
            if ins.size:
                m[box][ins] = True
    return chi, tube, ha, cx, vf
 
 
def dipole(sh, v):
    kx = np.fft.fftfreq(sh[0], d=v).astype(np.float32)
    ky = np.fft.fftfreq(sh[1], d=v).astype(np.float32)
    kz = np.fft.fftfreq(sh[2], d=v).astype(np.float32)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2+KY**2+KZ**2; K2[0, 0, 0] = 1.0
    D = (1/3 - KZ**2/K2).astype(np.float32); D[0, 0, 0] = 0.0
    return D
 
 
def signal(chi_f, tube_f, ha_f, cx_f, vf):
    ss = SUPERSAMPLE
    sh = SHAPE[RESOLUTION]
    TEs = np.array(ECHO_TIMES[RESOLUTION])/1000.0
    rng = np.random.default_rng(50000 + NOISE_SEED)
 
    pad = PAD*ss
    cp = np.pad(chi_f, pad, constant_values=CHI_AIR_PPM)
    f = np.real(ifftn(fftn(cp)*dipole(cp.shape, vf)))[pad:-pad, pad:-pad,
                                                      pad:-pad]
    del cp
    fhz = (f*1e-6*GAMMA*B0/(2*np.pi)).astype(np.float32)
    del f
 
    # ---- BACKGROUND ----
    # Synthetic shim: linear along the two in-plane directions. See the
    # docstring for how this differs from a measured background field.
    zz = np.linspace(-1, 1, fhz.shape[2], dtype=np.float32)[None, None, :]
    yy = np.linspace(-1, 1, fhz.shape[1], dtype=np.float32)[None, :, None]
    fhz = fhz + 300.0*zz + 80.0*yy
    print("    synthetic linear shim added")
 
    r2s = np.full(fhz.shape, 1.0, np.float32)
    r2s[tube_f] = BASE_R2S[(GEL, RESOLUTION)]
    r2s[ha_f] = BASE_R2S[(GEL, RESOLUTION)] + DR2S_HA
    r2s[cx_f] = BASE_R2S[(GEL, RESOLUTION)] + DR2S_CAOX
    S0 = np.where(tube_f, np.float32(1000.0), np.float32(5.0))
 
    coil = None
    if os.path.exists(COIL_PROFILE_FILE):
        coil = nib.load(COIL_PROFILE_FILE).get_fdata().astype(np.float32)
 
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
        mag[..., e] = np.abs(s); pha[..., e] = np.angle(s)
    return mag, (pha/np.pi*4096).astype(np.float32)
 
 
# =============================================================================
# PIPELINE
# =============================================================================
 
def phase_quality(phase, mag):
    n = phase.shape[3]; cs = mag*np.exp(1j*phase)
    acc = np.zeros(phase.shape[:3], np.complex128)
    for e in range(1, n):
        d = cs[..., e]*np.conj(cs[..., e-1]); acc += d/(np.abs(d)+1e-10)
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
    tmp = tempfile.mkdtemp(prefix="romeo_")
    try:
        pp, mp = os.path.join(tmp, "p.nii"), os.path.join(tmp, "m.nii")
        nib.save(nib.Nifti1Image(phase.astype(np.float32), affine), pp)
        nib.save(nib.Nifti1Image(mag.astype(np.float32), affine), mp)
        out = os.path.join(tmp, "o"); os.makedirs(out, exist_ok=True)
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
    un = np.zeros_like(phase); sl = find_objects(lab)
    for l in keep:
        s0 = sl[l-1]
        w = tuple(slice(max(0, x.start-CROP_MARGIN),
                        min(phase.shape[d], x.stop+CROP_MARGIN))
                  for d, x in enumerate(s0))
        sub = romeo(phase[w], mag[w], TEs, affine)
        this = (lab == l)[w]
        for e in range(phase.shape[3]):
            t = un[w][..., e]; t[this] = sub[..., e][this]; un[w+(e,)] = t
    return un
 
 
def field_map(un, TEs, mag, mask):
    ts = np.array(TEs)/1000.0; w = mag**2
    te = ts[None, None, None, :]
    sw = np.sum(w, 3); swt = np.sum(w*te, 3); swt2 = np.sum(w*te**2, 3)
    swp = np.sum(w*un, 3); swtp = np.sum(w*te*un, 3)
    den = sw*swt2-swt**2; den[den == 0] = 1e-10
    return ((sw*swtp-swt*swp)/den/(2*np.pi))*mask
 
 
def count_wraps(f, mask, TEs):
    wrap = 1.0/((TEs[1]-TEs[0])/1000.0); m = mask > 0; tot = 0
    for ax in range(3):
        d = np.abs(np.diff(f, axis=ax))
        a = [slice(None)]*3; a[ax] = slice(1, None)
        b = [slice(None)]*3; b[ax] = slice(0, -1)
        tot += int(np.sum(d[m[tuple(a)] & m[tuple(b)]] > 0.85*wrap))
    return tot
 
 
# =============================================================================
# V-SHARP, in Python, as the original pipeline had it
# =============================================================================
 
def sphere_kernel(shape, v, radius_mm):
    n = [int(np.ceil(radius_mm/v))*2 + 1]*3
    g = [np.arange(x) - (x-1)/2 for x in n]
    X, Y, Z = np.meshgrid(*g, indexing='ij')
    s = (np.sqrt(X**2 + Y**2 + Z**2)*v <= radius_mm).astype(np.float64)
    s /= s.sum()
    k = np.zeros(shape)
    sl = tuple(slice(0, x) for x in s.shape)
    k[sl] = s
    k = np.roll(k, [-(x//2) for x in s.shape], axis=(0, 1, 2))
    return fftn(k)
 
 
def vsharp(field, mask, v):
    """Multi-radius V-SHARP with per-radius deconvolution."""
    sh = field.shape
    st = generate_binary_structure(3, 1)
    out = np.zeros(sh)
    valid = np.zeros(sh, bool)
    for radius in sorted(VSHARP_RADII_MM, reverse=True):
        er = mask.copy().astype(bool)
        n_er = int(np.ceil(radius/v))
        for _ in range(n_er):
            er = binary_erosion(er, structure=st)
        if er.sum() < 100:
            continue
        S = sphere_kernel(sh, v, radius)
        filt = np.real(ifftn(fftn(field*mask)*(1 - S)))
        # deconvolution, thresholded
        Sd = 1 - S
        inv = np.where(np.abs(Sd) > VSHARP_DECONV_THRESHOLD, 1/Sd, 0)
        dec = np.real(ifftn(fftn(filt*er)*inv))
        new = er & ~valid
        out[new] = dec[new]
        valid |= new
    return out*valid, valid.astype(float)
 
 
# =============================================================================
# INVERSION
# =============================================================================
 
def kernels(sh, v):
    kx = np.fft.fftfreq(sh[0], v); ky = np.fft.fftfreq(sh[1], v)
    kz = np.fft.fftfreq(sh[2], v)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2+KY**2+KZ**2; k2[0, 0, 0] = 1e-10
    b = np.array(B0_DIR)/np.linalg.norm(B0_DIR)
    D = 1/3-(KX*b[0]+KY*b[1]+KZ*b[2])**2/k2; D[0, 0, 0] = 0
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
    gx, gy, gz = _g(x, v); return -_d(gx, gy, gz, v)
 
 
def cg(rhs, chi0, mask, D, lap, mu, v):
    den = np.abs(D)**2+mu*lap; den[den < 1e-12] = 1e-12
 
    def A(x):
        xm = x*mask; Dx = np.real(ifftn(D*fftn(xm)))
        return mask*(np.real(ifftn(D*fftn(mask*Dx)))+mu*_GtG(xm, v))
 
    def M(r):
        return mask*np.real(ifftn(fftn(r)/den))
    x = chi0*mask; r = rhs-A(x); z = M(r); p = z.copy(); rz = np.sum(r*z)
    r0 = np.linalg.norm(r)
    if r0 < 1e-30:
        return x
    for _ in range(CG_IT):
        Ap = A(p); pAp = np.sum(p*Ap)
        if abs(pAp) < 1e-30:
            break
        a = rz/pAp; x += a*p; r -= a*Ap
        if np.linalg.norm(r)/r0 < CG_TOL:
            break
        z = M(r); rzn = np.sum(r*z); p = z+(rzn/rz)*p; rz = rzn
    return x*mask
 
 
def invert(lf, mask, v):
    f = lf/(42.576e6*B0)*1e6
    D, lap = kernels(lf.shape, v)
    rhs0 = mask*np.real(ifftn(D*fftn(f*mask)))
    Di = np.zeros_like(D); k = np.abs(D) > 0.2
    Di[k] = 1/D[k]; Di[~k] = np.sign(D[~k])/0.2
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
        ux += gx-zx; uy += gy-zy; uz += gz-zz
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
 
def prep():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXCHANGE, exist_ok=True)
    v = VOXEL_MM[RESOLUTION]
    affine = np.diag([v, v, v, 1.0])
    TEs = ECHO_TIMES[RESOLUTION]
 
    print("=" * 74)
    print(f"  THREE-METHOD COMPARISON, prep: {TAG}")
    print("=" * 74)
 
    print("\n  building phantom ...")
    chi_f, tube_f, ha_f, cx_f, vf = build_phantom()
    mag, pha = signal(chi_f, tube_f, ha_f, cx_f, vf)
 
    gt = block_mean(np.where(tube_f | ha_f | cx_f,
                             chi_f - CHI_GEL_PPM[GEL], 0.0
                             ).astype(np.float32), SUPERSAMPLE)
    f_ha = block_mean(ha_f.astype(np.float32), SUPERSAMPLE)
    f_cx = block_mean(cx_f.astype(np.float32), SUPERSAMPLE)
    tube = block_mean(tube_f.astype(np.float32), SUPERSAMPLE)
    del chi_f, tube_f, ha_f, cx_f
 
    pr = pha.astype(np.float64)*np.pi/4096
    mf = mag.astype(np.float64)
    mask = make_mask(mf, pr)
    print(f"  mask {mask.sum():.0f} voxels")
 
    print("  unwrapping ...")
    un = romeo(pr, mf, TEs, affine)
    field = field_map(un, TEs, mf, mask)
    w = count_wraps(field, mask, TEs)
    if w > WRAP_LIMIT:
        print(f"  {w} wrap jumps, redoing per tube")
        un = romeo_per_tube(pr, mf, TEs, affine, mask)
        field = field_map(un, TEs, mf, mask)
        w = count_wraps(field, mask, TEs)
    print(f"  wrap jumps {w}")
    print(f"  field map SD {field[mask>0].std():.1f} Hz")
 
    nib.save(nib.Nifti1Image(gt, affine),
             os.path.join(OUT_DIR, "ground_truth_chi_ppm.nii.gz"))
    nib.save(nib.Nifti1Image(((f_ha > 0.5).astype(np.float32)
                              + 2*(f_cx > 0.5).astype(np.float32)), affine),
             os.path.join(OUT_DIR, "particle_mask.nii.gz"))
    nib.save(nib.Nifti1Image((tube > 0.5).astype(np.float32), affine),
             os.path.join(OUT_DIR, "tube_mask.nii.gz"))
    nib.save(nib.Nifti1Image(mask.astype(np.float32), affine),
             os.path.join(OUT_DIR, "mask.nii.gz"))
    nib.save(nib.Nifti1Image(field.astype(np.float32), affine),
             os.path.join(OUT_DIR, "field_map_hz.nii.gz"))
    savemat(os.path.join(EXCHANGE, f"in_{TAG}.mat"),
            {"iFreq": field*2*np.pi, "Mask": mask, "N_std": mask,
             "matrix_size": np.array(field.shape, float),
             "voxel_size": np.array([v, v, v]),
             "B0_dir": np.array(B0_DIR, float)}, do_compression=True)
 
    print(f"""
  NEXT
  ----
  1. Run run_medi_batch_PDF_LBV.m with EXCHANGE = '{EXCHANGE}'.
     It must produce both RDF_PDF and RDF_LBV: the comparison needs all
     three methods.
  2. Return here with MODE = "compare".
""")
 
 
def compare():
    v = VOXEL_MM[RESOLUTION]
    matp = os.path.join(EXCHANGE, f"out_{TAG}.mat")
    if not os.path.exists(matp):
        print(f"  {matp} not found. Run run_medi_batch_PDF_LBV.m first.")
        return
 
    field = nib.load(os.path.join(
        OUT_DIR, "field_map_hz.nii.gz")).get_fdata()
    mask = nib.load(os.path.join(OUT_DIR, "mask.nii.gz")).get_fdata()
    gt = nib.load(os.path.join(
        OUT_DIR, "ground_truth_chi_ppm.nii.gz")).get_fdata()
    pm = nib.load(os.path.join(
        OUT_DIR, "particle_mask.nii.gz")).get_fdata()
    tube = nib.load(os.path.join(OUT_DIR, "tube_mask.nii.gz")).get_fdata() > 0
    affine = np.diag([v, v, v, 1.0])
 
    d = loadmat(matp)
    methods = {}
 
    print("=" * 84)
    print(f"  THREE-METHOD COMPARISON AGAINST GROUND TRUTH: {TAG}")
    print("=" * 84)
 
    print("\n  running V-SHARP here ...")
    lf_v, vm_v = vsharp(field, mask, v)
    methods["V-SHARP"] = (lf_v, vm_v)
 
    for name, key_lf, key_m in [("PDF", "RDF_PDF", "Mask_PDF"),
                                ("LBV", "RDF_LBV", "Mask_LBV")]:
        if key_lf in d:
            methods[name] = (np.asarray(d[key_lf], float)/(2*np.pi),
                             np.asarray(d[key_m], float))
        else:
            print(f"  {name} not in the MATLAB output, skipping")
 
    dist = ndimage.distance_transform_edt(mask > 0, sampling=(v, v, v))
    rows = []
 
    for name, (lf, vmask) in methods.items():
        vm = vmask > 0
        if vm.sum() < 1000:
            print(f"\n  {name}: valid region too small, skipping")
            continue
        print(f"\n  {name}: inverting ...")
        chi = invert(lf, vmask, v)
        nib.save(nib.Nifti1Image(chi.astype(np.float32), affine),
                 os.path.join(OUT_DIR, f"qsm_{name.replace('-','')}.nii.gz"))
 
        cov = 100*vm.sum()/(mask > 0).sum()
        band = vm & (dist >= 4) & (dist < 6)
        strip = lf[band].std() if band.sum() > 500 else np.nan
        gel = vm & tube & (pm == 0)
        gel_sd = chi[gel].std()
        gel_mean = chi[gel].mean()
 
        # recovery over a radius-2 ROI, as the real analysis measures it
        recs, truths = [], []
        lab, nt = ndimage.label(tube,
                                structure=generate_binary_structure(3, 1))
        for l in range(1, nt+1):
            tm = lab == l
            sel = (pm == 1) & tm
            if sel.sum() == 0:
                continue
            cen = tuple(int(x) for x in
                        np.array(ndimage.center_of_mass(sel)).round())
            i_, j_, k_ = sphere_idx(cen, chi.shape)
            ok = vm[i_, j_, k_]
            if ok.sum() < 10:
                continue
            # Ground truth and recovery are averaged over the SAME voxels:
            # only those where background removal produced valid output.
            # Averaging the ground truth over the whole sphere while
            # averaging the recovery over the valid subset compares two
            # different denominators and shifts the apparent ground truth.
            # It matters most for the method with the smallest valid region,
            # where the dropped voxels are a large and biased subset.
            recs.append((chi[i_[ok], j_[ok], k_[ok]] - gel_mean).mean())
            truths.append(gt[i_[ok], j_[ok], k_[ok]].mean())
 
        rec = np.mean(recs) if recs else np.nan
        tru = np.mean(truths) if truths else np.nan
        rows.append(dict(method=name, coverage=cov, striping=strip,
                         gel_sd=gel_sd, recovered=rec, truth=tru,
                         recovery=100*rec/tru if tru else np.nan))
        print(f"    coverage {cov:.1f}%, striping {strip:.2f} Hz, "
              f"gel SD {gel_sd:.5f} ppm")
        print(f"    recovered {rec:+.4f} ppm against truth {tru:+.4f} "
              f"({100*rec/tru:.1f}%)")
 
    if not rows:
        return
 
    print("\n\n" + "=" * 84)
    print("  RESULT")
    print("=" * 84)
    print(f"  {'method':<10}{'coverage':>10}{'striping':>11}{'gel SD':>11}"
          f"{'recovered':>12}{'% of truth':>12}")
    print("  " + "-" * 66)
    for r in rows:
        print(f"  {r['method']:<10}{r['coverage']:>9.1f}%"
              f"{r['striping']:>10.2f}H{r['gel_sd']:>11.5f}"
              f"{r['recovered']:>12.4f}{r['recovery']:>11.1f}%")
 
    best_acc = max(rows, key=lambda r: r['recovery']
                   if not np.isnan(r['recovery']) else -1e9)
    best_noise = min(rows, key=lambda r: r['gel_sd']
                     if not np.isnan(r['gel_sd']) else 1e9)
 
    print(f"""
  HIGHEST RECOVERY: {best_acc['method']} ({best_acc['recovery']:.1f}%)
  LOWEST GEL SD:    {best_noise['method']} ({best_noise['gel_sd']:.5f} ppm)
""")
 
 
if __name__ == "__main__":
    for _seed in PARTICLE_SEEDS:
        PARTICLE_SEED = _seed
        TAG = f"COMPARE_{GEL}_{RESOLUTION}_p{_seed}"
        OUT_DIR = os.path.join(OUT_DIR_BASE, f"p{_seed}")
        print(f"\n{'#'*74}\n#  particle realisation {_seed}\n{'#'*74}")
        if MODE == "prep":
            prep()
        else:
            compare()
 
    if MODE == "prep":
        print(f"""
  All {len(PARTICLE_SEEDS)} realisations prepped.
 
  NEXT
  ----
  1. Run run_medi_batch_PDF_LBV.m with EXCHANGE = '{EXCHANGE}'. It processes
     every in_*.mat in the folder, so one MATLAB run covers them all.
  2. Return here with MODE = "compare".
""")
    else:
        print(f"""
  {len(PARTICLE_SEEDS)} realisations compared.
""")