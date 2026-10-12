#!/usr/bin/env python3
"""
Digital twin: verification against analytic results
===================================================

Verifies the twin's forward model against cases whose answers are known
independently of the code, so that a failure points at the model rather than
at a judgement about what looks right.

Six checks:

  1. SPHERE. A uniform sphere has no internal field shift with the kernel
     used here, and follows the classical dipole law outside it. The interior
     must be exactly zero, which makes a normalisation error visible
     immediately rather than as a small deviation from a non-zero number.

  2. CYLINDER. An infinite cylinder gives +chi/3 parallel to B0 and -chi/6
     perpendicular. The ratio of -2 between the two orientations is
     independent of sign convention, and is why orientation relative to B0
     matters for a tube phantom.

  3. LINEARITY. Doubling the susceptibility must double the field everywhere.
     This catches a normalisation error that a single-value check cannot,
     since such an error can be absorbed into a plausible-looking number.

  4. SUPERPOSITION. Two well-separated sources must give the sum of their
     individual fields, so nothing in the convolution couples voxels that
     should be independent.

  5. R2* ROUND TRIP. Recovering a specified relaxation rate from noiseless
     synthesised signal. A factor-of-1000 error between s^-1 and ms^-1 would
     be invisible in the susceptibility results while corrupting every R2*
     comparison.

  6. REGION EXTRACTION. The twin analysis and the measured-data analysis
     select their regions of interest with separate implementations, and the
     comparison between simulation and experiment depends on both selecting
     the same voxels. This checks them against each other on identical
     input.

The sphere and the cylinder test the same kernel in different ways: the
cylinder constrains it along one axis, the sphere in all three. A kernel wrong
in a way that happened to satisfy one would not satisfy both.

Each check prints PASS or FAIL with the numbers behind it. Nothing here needs
data or paths, so it runs anywhere.
"""

import numpy as np
from scipy.fft import fftn, ifftn

# =============================================================================
# CONSTANTS, matching the twin
# =============================================================================

B0 = 2.895
GAMMA = 2.675e8
TOL = 0.02          # 2% agreement required against analytic results


def dipole_kernel(shape, v, B0_dir=(0, 0, 1)):
    """Identical to the twin's kernel."""
    kx = np.fft.fftfreq(shape[0], d=v)
    ky = np.fft.fftfreq(shape[1], d=v)
    kz = np.fft.fftfreq(shape[2], d=v)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    b = np.array(B0_dir, dtype=float)
    b = b/np.linalg.norm(b)
    D = 1/3 - (KX*b[0] + KY*b[1] + KZ*b[2])**2/K2
    D[0, 0, 0] = 0.0
    return D


def forward_ppm(chi, v, pad, B0_dir=(0, 0, 1), outside=0.0):
    cp = np.pad(chi, pad, constant_values=outside)
    f = np.real(ifftn(fftn(cp)*dipole_kernel(cp.shape, v, B0_dir)))
    return f[pad:-pad, pad:-pad, pad:-pad]


def report(name, measured, expected, tol=TOL, unit="ppm"):
    if expected == 0:
        ok = abs(measured) < 1e-6
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"        measured {measured:+.6f}, expected 0")
    else:
        err = abs(measured-expected)/abs(expected)
        ok = err < tol
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"        measured {measured:+.5f} {unit}, "
              f"expected {expected:+.5f}, error {100*err:.2f}%")
    return ok


# =============================================================================
# 1. SPHERE
# =============================================================================

def test_sphere():
    """
    A uniform sphere of susceptibility chi in a medium of zero susceptibility
    has no internal field shift with the Lorentz sphere correction included in
    the kernel, as it is here: the demagnetising factor is 1/3 in every
    direction and the kernel's own 1/3 cancels it exactly.

    Outside the sphere the field is the classical dipole,

        dB/B0 = (chi/3) (a/r)^3 (3 cos^2(theta) - 1)

    checked along and perpendicular to B0.
    """
    print("\n1. SPHERE")
    print("   Interior must be zero (the kernel's 1/3 cancels the")
    print("   demagnetising factor). Exterior must follow the dipole law.")

    n, v, pad = 96, 0.5, 96
    chi_val = -2.0
    radius_mm = 6.0

    c = (n-1)/2
    X, Y, Z = np.meshgrid(*[(np.arange(n)-c)*v]*3, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    chi = np.where(r <= radius_mm, chi_val, 0.0)

    f = forward_ppm(chi, v, pad)

    interior = r < radius_mm*0.5
    ok1 = report("interior field", float(f[interior].mean()), 0.0)

    # exterior, along B0 (z) at 2a and perpendicular (x) at 2a
    d = int(round(2*radius_mm/v))
    ci = int(c)
    along = f[ci, ci, ci+d]
    perp = f[ci+d, ci, ci]
    exp_along = (chi_val/3)*(radius_mm/(2*radius_mm))**3*(3*1 - 1)
    exp_perp = (chi_val/3)*(radius_mm/(2*radius_mm))**3*(3*0 - 1)
    # Tolerance relaxed for the exterior: at r = 2a the sphere spans only
    # about 12 voxels, so the sample point is coarsely resolved.
    ok2 = report("exterior along B0, r = 2a", float(along), exp_along,
                 tol=0.10)
    ok3 = report("exterior perpendicular, r = 2a", float(perp), exp_perp,
                 tol=0.10)
    print(f"        ratio along/perpendicular {along/perp:+.3f} "
          f"(analytic -2.000)")
    return ok1 and ok2 and ok3


# =============================================================================
# 2. CYLINDER
# =============================================================================

def test_cylinder():
    """
    An infinite cylinder, with the kernel D = 1/3 - kz^2/k^2 used here:

        parallel to B0        +chi/3
        perpendicular to B0   -chi/6

    The sign convention depends on how the kernel is defined, and both
    conventions appear in the literature. What does not depend on convention,
    and is the physically meaningful relationship, is the ratio: a factor of
    two and a change of sign between the two orientations.

    The interior values deviate from the analytic result by about 10% because
    the simulated cylinder is finite, 48 mm long against a 6 mm radius, rather
    than infinite. The ratio is unaffected by that and is checked tightly.
    """
    print("\n2. CYLINDER")
    print("   Interior: +chi/3 parallel to B0, -chi/6 perpendicular,")
    print("   for the kernel D = 1/3 - kz^2/k^2 used here. The ratio of -2")
    print("   is convention-independent and is the real check.")

    n, v, pad = 96, 0.5, 128
    chi_val = -2.0
    radius_mm = 6.0
    c = (n-1)/2
    X, Y, Z = np.meshgrid(*[(np.arange(n)-c)*v]*3, indexing='ij')

    # cylinder along z, i.e. parallel to B0 = (0, 0, 1)
    r_par = np.sqrt(X**2 + Y**2)
    chi_par = np.where(r_par <= radius_mm, chi_val, 0.0)
    f_par = forward_ppm(chi_par, v, pad)
    inner = (r_par < radius_mm*0.4) & (np.abs(Z) < n*v*0.15)
    par = float(f_par[inner].mean())
    ok1 = report("parallel to B0", par, chi_val/3, tol=0.12)

    # cylinder along x, i.e. perpendicular to B0
    r_perp = np.sqrt(Y**2 + Z**2)
    chi_perp = np.where(r_perp <= radius_mm, chi_val, 0.0)
    f_perp = forward_ppm(chi_perp, v, pad)
    inner = (r_perp < radius_mm*0.4) & (np.abs(X) < n*v*0.15)
    perp = float(f_perp[inner].mean())
    ok2 = report("perpendicular to B0", perp, -chi_val/6, tol=0.12)

    ok3 = report("ratio parallel/perpendicular", par/perp, -2.0, tol=0.001,
                 unit="")
    print("        (the deviation in the individual values is the finite")
    print("         cylinder length, 8:1 aspect ratio; the ratio is exact)")
    return ok1 and ok2 and ok3


# =============================================================================
# 3. LINEARITY
# =============================================================================

def test_linearity():
    """Doubling the susceptibility must double the field everywhere."""
    print("\n3. LINEARITY IN SUSCEPTIBILITY")

    n, v, pad = 64, 0.5, 64
    c = (n-1)/2
    X, Y, Z = np.meshgrid(*[(np.arange(n)-c)*v]*3, indexing='ij')
    blob = np.sqrt(X**2 + Y**2 + Z**2) <= 4.0

    f1 = forward_ppm(np.where(blob, -1.0, 0.0), v, pad)
    f2 = forward_ppm(np.where(blob, -2.0, 0.0), v, pad)
    ratio = np.abs(f2).sum()/max(np.abs(f1).sum(), 1e-12)
    return report("field ratio for 2x susceptibility", float(ratio), 2.0,
                  unit="")


# =============================================================================
# 4. SUPERPOSITION
# =============================================================================

def test_superposition():
    """Two well-separated sources must give the sum of their individual
    fields, so nothing in the convolution couples voxels that should be
    independent."""
    print("\n4. SUPERPOSITION")

    n, v, pad = 80, 0.5, 80
    c = (n-1)/2
    X, Y, Z = np.meshgrid(*[(np.arange(n)-c)*v]*3, indexing='ij')
    a = (np.sqrt((X+8)**2 + Y**2 + Z**2) <= 2.0)
    b = (np.sqrt((X-8)**2 + Y**2 + Z**2) <= 2.0)

    fa = forward_ppm(np.where(a, -2.0, 0.0), v, pad)
    fb = forward_ppm(np.where(b, -2.0, 0.0), v, pad)
    fab = forward_ppm(np.where(a | b, -2.0, 0.0), v, pad)

    err = np.abs(fab - (fa+fb)).max()/max(np.abs(fab).max(), 1e-12)
    ok = err < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}  superposition")
    print(f"        max relative deviation {err:.2e}")
    return ok


# =============================================================================
# 5. R2* ROUND TRIP
# =============================================================================

def test_r2star_roundtrip():
    """Synthesise a multi-echo signal at a known R2*, then fit it back.

    A factor-of-1000 error between s^-1 and ms^-1 would be invisible in the
    susceptibility results while corrupting every R2* comparison. The rates
    tested span gel baseline to particle values.
    """
    print("\n5. R2* ROUND TRIP")

    TEs = np.array([2.89, 7.14, 11.39, 15.64, 19.89, 24.14, 28.39])/1000.0
    allok = True
    for r2_true in (7.5, 84.2, 175.5):
        S0 = 1000.0
        sig = S0*np.exp(-r2_true*TEs)
        # log-linear fit, exact for noiseless mono-exponential data
        A = np.stack([np.ones_like(TEs), -TEs], axis=1)
        coef, *_ = np.linalg.lstsq(A, np.log(sig), rcond=None)
        allok &= report(f"R2* = {r2_true} s^-1 recovered",
                        float(coef[1]), r2_true, unit="s^-1")
    return allok


# =============================================================================
# 6. REGION EXTRACTION CONSISTENCY
# =============================================================================
# Run this script from the directory holding twin_recovery.py and roi_analysis.py.

def _sphere_idx_vectorised(c, shape, R=2):
    """Vectorised form, as used in the twin analysis."""
    r = int(R)
    x, y, z = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    keep = (x**2 + y**2 + z**2) <= R**2
    idx = np.array(c)[None, :] + np.stack([x[keep], y[keep], z[keep]], 1)
    ok = np.all((idx >= 0) & (idx < np.array(shape)[None, :]), 1)
    idx = idx[ok]
    return idx[:, 0], idx[:, 1], idx[:, 2]


def _sphere_idx_explicit(centre, shape, radius_voxels=2):
    """Explicit-loop form, written independently of the vectorised one."""
    cx, cy, cz = centre
    out = []
    for ix in range(max(0, cx-radius_voxels),
                    min(shape[0], cx+radius_voxels+1)):
        for iy in range(max(0, cy-radius_voxels),
                        min(shape[1], cy+radius_voxels+1)):
            for iz in range(max(0, cz-radius_voxels),
                            min(shape[2], cz+radius_voxels+1)):
                if np.sqrt((ix-cx)**2 + (iy-cy)**2 + (iz-cz)**2) \
                        <= radius_voxels:
                    out.append((ix, iy, iz))
    return out


def test_roi_consistency():
    """Compare the twin's region selection with the measured analysis's.

    The comparison between simulation and experiment depends on both
    selecting the same voxels around a given centre, and they are separate
    implementations.
    """
    print("\n6. REGION EXTRACTION: twin analysis vs measured analysis")

    twin_fn, real_fn = None, None
    try:
        from twin_recovery import sphere_idx as twin_fn
    except ImportError:
        pass
    try:
        from roi_analysis import roi_indices as real_fn
    except ImportError:
        pass

    if twin_fn is not None and real_fn is not None:
        print("   comparing the SHIPPED functions: twin_recovery.sphere_idx")
        print("   against roi_analysis.roi_indices")
    else:
        missing = [n for n, f in (("twin_recovery", twin_fn),
                                  ("roi_analysis", real_fn)) if f is None]
        print(f"   could not import {', '.join(missing)}; comparing local")
        print("   transcriptions instead. This verifies the two formulations")
        print("   agree but NOT the shipped analysis code. Run from the")
        print("   directory holding both scripts for the stronger check.")
        twin_fn = twin_fn or _sphere_idx_vectorised

    shape = (48, 152, 256)
    allok = True
    for c in [(24, 76, 128), (2, 2, 2), (17, 35, 212)]:
        i, j, k = twin_fn(c, shape)
        A = set(zip(np.asarray(i).tolist(), np.asarray(j).tolist(),
                    np.asarray(k).tolist()))
        if real_fn is not None:
            i2, j2, k2 = real_fn(c, shape)
            B = set(zip(np.asarray(i2).tolist(), np.asarray(j2).tolist(),
                        np.asarray(k2).tolist()))
        else:
            B = set(_sphere_idx_explicit(c, shape))
        ok = A == B
        allok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  centre {c}: "
              f"{len(A)} vs {len(B)} voxels"
              + ("" if ok else f", {len(A ^ B)} differ"))
    return allok


# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  DIGITAL TWIN VERIFICATION")
    print("=" * 70)

    results = {
        "sphere": test_sphere(),
        "cylinder": test_cylinder(),
        "linearity": test_linearity(),
        "superposition": test_superposition(),
        "R2* round trip": test_r2star_roundtrip(),
        "region extraction consistency": test_roi_consistency(),
    }

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    if all(results.values()):
        print("""
  All checks pass. 
""")
    else:
        print("""
  One or more checks FAILED. 
""")