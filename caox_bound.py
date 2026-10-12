#!/usr/bin/env python3
"""
Bounding the calcium oxalate susceptibility
===========================================

Converts a non-detection into a measurement: instead of reporting that
calcium oxalate was not detected, it establishes how large its susceptibility
could be and still have gone undetected.

WHAT THIS SCRIPT DOES

    MODE = "generate"           builds a phantom at each susceptibility in
                                CAOX_SWEEP and preps it
    run_medi_batch_PDF_LBV.m    ONCE over the exchange folder
    MODE = "invert"             inverts each, applies the same detection
                                criterion as the measured analysis, and
                                reports the smallest susceptibility at which
                                detection still occurs

The output is an upper bound: the susceptibility of the real particles must
be smaller in magnitude than the smallest simulated value that was still
detected.

Hydroxyapatite is held at CHI_HA_PPM throughout as an internal control. It
should be detected in every run; if it is not, the run itself is at fault
rather than the calcium oxalate result.
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
# EDIT THESE
# =============================================================================

MODE = "generate"

OUT_ROOT = r"C:\QSM_twin\caox_bound"
EXCHANGE = r"C:\QSM_twin\MEDI_EXCHANGE_CAOX"
ROMEO_PATH = r"D:\romeo\romeo_win_3.2.8\bin\romeo.exe"
COIL_PROFILE_FILE = r"C:\QSM_twin\coil_profile\coil_sensitivity_measured.nii.gz"

GEL = "Alginate"
RESOLUTION = "070mm"
CHI_HA_PPM = -2.0          # held fixed, as an internal control

# The sweep. Spans the pure-mineral estimate down to essentially nothing.
CAOX_SWEEP = [-1.82, -1.41, -1.00, -0.50, -0.25, -0.10, -0.05, -0.02]

N_PARTICLE = 1             # shape variation is not the question here
SUPERSAMPLE = 2
SKIP_IF_EXISTS = True

# Detection criterion, identical to the measured analysis
DETECT_SD_MULTIPLE = 3.0
FOCAL_DIP_SD_MULTIPLE = 1.0
ROI_RADIUS = 2

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
GEL_INHOMOGENEITY_PCT, GEL_INHOMO_SCALE_MM = 0.0, 15.0
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
MASK_THR, MASK_ERODE, PQ_THR = 0.1, 2, 0.3
WRAP_LIMIT, MIN_TUBE, CROP_MARGIN = 100, 5000, 10
TV_LAM, TV_MU, ADMM_IT, CG_IT, CG_TOL = 0.0005, 0.05, 60, 10, 1e-3


def tag_for(chi_caox):
    return f"CAOX_{GEL}_{RESOLUTION}_chi{chi_caox:g}"


# =============================================================================
# PHANTOM AND SIGNAL
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


def build_phantom(chi_caox, p_seed=0):
    ss = SUPERSAMPLE
    v = VOXEL_MM[RESOLUTION]
    vf = v/ss
    sh = SHAPE[RESOLUTION]
    fsh = tuple(x*ss for x in sh)
    rng = np.random.default_rng(1000 + p_seed)

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

    inhomo = np.ones(fsh, np.float32)
    if GEL_INHOMOGENEITY_PCT > 0:
        n = rng.normal(0, 1, fsh).astype(np.float32)
        sm = gaussian_filter(n, sigma=GEL_INHOMO_SCALE_MM/vf)
        inhomo = (1 + (GEL_INHOMOGENEITY_PCT/100)*sm/(sm.std()+1e-9)
                  ).astype(np.float32)
        chi[tube] += (inhomo[tube]-1)*abs(CHI_GEL_PPM[GEL])*0.1

    for cz in TUBE_Z[RESOLUTION]:
        czf = cz*ss + ss/2.0
        for y0, cp, m in [(HA_Y, CHI_HA_PPM, ha), (CAOX_Y, chi_caox, cx)]:
            semi, rot = random_ellipsoid(rng)
            ins, box = stamp(chi, CHI_GEL_PPM[GEL]+cp,
                             (c0, y0*ss + ss/2.0, czf), semi, rot, vf)
            if ins.size:
                m[box][ins] = True
    return chi, tube, ha, cx, inhomo, vf


def dipole(sh, v):
    kx = np.fft.fftfreq(sh[0], d=v).astype(np.float32)
    ky = np.fft.fftfreq(sh[1], d=v).astype(np.float32)
    kz = np.fft.fftfreq(sh[2], d=v).astype(np.float32)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2+KY**2+KZ**2; K2[0, 0, 0] = 1.0
    D = (1/3 - KZ**2/K2).astype(np.float32); D[0, 0, 0] = 0.0
    return D


def signal(chi_f, tube_f, ha_f, cx_f, inhomo_f, vf, n_seed=0):
    ss = SUPERSAMPLE
    sh = SHAPE[RESOLUTION]
    TEs = np.array(ECHO_TIMES[RESOLUTION])/1000.0
    rng = np.random.default_rng(50000 + n_seed)
    pad = PAD*ss
    cp = np.pad(chi_f, pad, constant_values=CHI_AIR_PPM)
    f = np.real(ifftn(fftn(cp)*dipole(cp.shape, vf)))[pad:-pad, pad:-pad,
                                                      pad:-pad]
    del cp
    fhz = (f*1e-6*GAMMA*B0/(2*np.pi)).astype(np.float32)
    del f
    zz = np.linspace(-1, 1, fhz.shape[2], dtype=np.float32)[None, None, :]
    yy = np.linspace(-1, 1, fhz.shape[1], dtype=np.float32)[None, :, None]
    fhz = fhz + 300.0*zz + 80.0*yy
    r2s = np.full(fhz.shape, 1.0, np.float32)
    r2s[tube_f] = BASE_R2S[(GEL, RESOLUTION)]
    r2s[ha_f] = BASE_R2S[(GEL, RESOLUTION)] + DR2S_HA
    r2s[cx_f] = BASE_R2S[(GEL, RESOLUTION)] + DR2S_CAOX
    S0 = np.where(tube_f, np.float32(1000.0), np.float32(5.0))*inhomo_f
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


def inv_kernels(sh, v):
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
    D, lap = inv_kernels(lf.shape, v)
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


def focal_dip(chi, valid, c, spacing=2):
    if not valid[c]:
        return np.nan
    vals = []
    for dx in (-spacing, 0, spacing):
        for dy in (-spacing, 0, spacing):
            for dz in (-spacing, 0, spacing):
                if dx == dy == dz == 0:
                    continue
                p = (c[0]+dx, c[1]+dy, c[2]+dz)
                if all(0 <= p[a] < chi.shape[a] for a in range(3)) \
                        and valid[p]:
                    vals.append(chi[p])
    return float(chi[c] - np.mean(vals)) if vals else np.nan


# =============================================================================
# MODES
# =============================================================================

def generate_all():
    os.makedirs(EXCHANGE, exist_ok=True)
    TEs = ECHO_TIMES[RESOLUTION]
    v = VOXEL_MM[RESOLUTION]
    affine = np.diag([v, v, v, 1.0])
    print("=" * 74)
    print(f"  CaOx BOUND: {len(CAOX_SWEEP)} susceptibilities")
    print(f"  HA held at {CHI_HA_PPM:+.2f} ppm as an internal control")
    print("=" * 74)

    for chi_cx in CAOX_SWEEP:
        tag = tag_for(chi_cx)
        out = os.path.join(OUT_ROOT, tag)
        matp = os.path.join(EXCHANGE, f"in_{tag}.mat")
        if SKIP_IF_EXISTS and os.path.exists(matp):
            print(f"\n  {tag}: exists, skipping"); continue
        print(f"\n  {tag}   chi_CaOx {chi_cx:+.2f} ppm")
        os.makedirs(out, exist_ok=True)

        chi_f, tube_f, ha_f, cx_f, inh_f, vf = build_phantom(chi_cx)
        mag, pha = signal(chi_f, tube_f, ha_f, cx_f, inh_f, vf)
        gt = block_mean(np.where(tube_f | ha_f | cx_f,
                                 chi_f - CHI_GEL_PPM[GEL], 0.0
                                 ).astype(np.float32), SUPERSAMPLE)
        f_ha = block_mean(ha_f.astype(np.float32), SUPERSAMPLE)
        f_cx = block_mean(cx_f.astype(np.float32), SUPERSAMPLE)
        del chi_f, tube_f, ha_f, cx_f, inh_f

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
                                  + 2*(f_cx > 0.5).astype(np.float32)),
                                 affine),
                 os.path.join(out, "particle_mask.nii.gz"))
        nib.save(nib.Nifti1Image(mask.astype(np.float32), affine),
                 os.path.join(out, "mask.nii.gz"))
        nib.save(nib.Nifti1Image(fld.astype(np.float32), affine),
                 os.path.join(out, "field_map_hz.nii.gz"))
        savemat(matp, {"iFreq": fld*2*np.pi, "Mask": mask,
                       "N_std": mask.astype(np.float64),
                       "matrix_size": np.array(fld.shape, float),
                       "voxel_size": np.array([v, v, v]),
                       "B0_dir": np.array(B0_DIR, float)}, do_compression=True)
        print(f"    mask {mask.sum():.0f}, wraps {w}, wrote in_{tag}.mat")

    print(f"""
  NEXT
  ----
  1. Run run_medi_batch_PDF_LBV.m with EXCHANGE = '{EXCHANGE}'.
  2. Come back with MODE = "invert".
""")


def invert_all():
    v = VOXEL_MM[RESOLUTION]
    rows = []
    print("=" * 82)
    print("  CaOx BOUND: applying the measured detection criterion")
    print("  (|Delta-chi_peak| > 3x gel SD AND a focal dip)")
    print("=" * 82)

    for chi_cx in CAOX_SWEEP:
        tag = tag_for(chi_cx)
        out = os.path.join(OUT_ROOT, tag)
        matp = os.path.join(EXCHANGE, f"out_{tag}.mat")
        if not os.path.exists(matp):
            print(f"\n  {tag}: no output, skipping"); continue

        d = loadmat(matp)
        lf = np.asarray(d["RDF_PDF"], float)/(2*np.pi)
        valid = np.asarray(d["Mask_PDF"], float)
        chi = invert(lf, valid, v)
        nib.save(nib.Nifti1Image(chi.astype(np.float32),
                                 np.diag([v, v, v, 1.0])),
                 os.path.join(out, "qsm_susceptibility_ppm.nii.gz"))

        gt = nib.load(os.path.join(out,
                                   "ground_truth_chi_ppm.nii.gz")).get_fdata()
        pm = nib.load(os.path.join(out, "particle_mask.nii.gz")).get_fdata()
        mask = nib.load(os.path.join(out, "mask.nii.gz")).get_fdata() > 0
        vm = valid > 0
        gel = vm & (pm == 0) & mask

        tubes, nt = ndimage.label(
            mask, structure=ndimage.generate_binary_structure(3, 1))
        sizes = ndimage.sum(mask, tubes, range(1, nt+1))
        keep = sorted([i+1 for i in np.argsort(sizes)[::-1][:4]
                       if sizes[i] > 5000],
                      key=lambda l: np.argwhere(tubes == l)[:, 2].mean())

        ha_det = cx_det = 0
        peaks = {"HA": [], "CaOx": []}
        for l in keep:
            tm = tubes == l
            g = gel & tm
            if g.sum() < 100:
                continue
            gmean, gsd = chi[g].mean(), chi[g].std()
            thr, dthr = DETECT_SD_MULTIPLE*gsd, FOCAL_DIP_SD_MULTIPLE*gsd
            for code, name in [(1, "HA"), (2, "CaOx")]:
                sel = (pm == code) & tm & vm
                if sel.sum() == 0:
                    continue
                cen = tuple(int(x) for x in
                            np.array(ndimage.center_of_mass((pm == code) & tm)
                                     ).round())
                i_, j_, k_ = sphere_idx(cen, chi.shape)
                s2 = vm[i_, j_, k_]
                vals = chi[i_[s2], j_[s2], k_[s2]]
                peak = np.percentile(vals, 5) - gmean
                dip = focal_dip(chi, vm, cen)
                det = (abs(peak) > thr) and (not np.isnan(dip)) \
                    and (dip < -dthr)
                peaks[name].append(peak)
                if det:
                    if code == 1:
                        ha_det += 1
                    else:
                        cx_det += 1

        rows.append(dict(chi_caox=chi_cx,
                         HA_det=ha_det, CaOx_det=cx_det,
                         HA_peak=np.mean(peaks["HA"]) if peaks["HA"] else np.nan,
                         CaOx_peak=(np.mean(peaks["CaOx"])
                                    if peaks["CaOx"] else np.nan)))
        print(f"\n  chi_CaOx {chi_cx:+.2f}: HA {ha_det}/4, CaOx {cx_det}/4  "
              f"(CaOx peak {rows[-1]['CaOx_peak']:+.4f} ppm)")

    if not rows:
        return

    print("\n\n" + "=" * 82)
    print("  RESULT")
    print("=" * 82)
    print(f"  {'chi_CaOx':>10}{'CaOx peak':>13}{'CaOx det':>11}"
          f"{'HA det':>9}   (HA is the control)")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['chi_caox']:>10.2f}{r['CaOx_peak']:>13.4f}"
              f"{r['CaOx_det']:>9}/4{r['HA_det']:>7}/4")

    undet = [r['chi_caox'] for r in rows if r['CaOx_det'] == 0]
    det = [r['chi_caox'] for r in rows if r['CaOx_det'] > 0]

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "caox_bound.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if undet and det:
        bound = min(det, key=abs)
        print(f"""
  CaOx was DETECTED down to {bound:+.2f} ppm and NOT detected below it.

  The measured particles were not detected in any tube, so their
  susceptibility relative to gel must be smaller in magnitude than
  {abs(bound):.2f} ppm.

  That is a measurement rather than an absence. It converts

      "calcium oxalate was not detected"

  into

      "the susceptibility of calcium oxalate is smaller in magnitude than
       {abs(bound):.2f} ppm, bounded by simulation at matched
       signal-to-noise, particle size and gel reference."
""")
    elif undet:
        print(f"\n  CaOx was not detected at ANY value tested, down to "
              f"{max(undet, key=abs):+.2f} ppm.")
        print("  Extend the sweep upward to find where detection begins.")
    else:
        print("\n  CaOx was detected at every value tested. Extend the sweep")
        print("  downward.")


if __name__ == "__main__":
    if MODE == "generate":
        generate_all()
    else:
        invert_all()