"""Nondipole multipole coupling (Multipole = 2, 3) of a plane wave E(t - n.r/c).

Multipolar gauge, charge q, mass m, C = n x E:
  1/c   : + q/(2c) (n.r)(r.E')            - q/(2mc) L.C
  1/c^2 : - q/(6c^2) (n.r)^2 (r.E'')      + q/(3mc^2) {n.r, L}/2 . C'   + q^2/(8mc^2) |C x r|^2

* Multipole = 2 keeps H quadratic for an oscillator, so <r>(t) obeys the
  classical Hamilton equations exactly (Ehrenfest): checked against SciPy, with
  a small speed of light so the nondipole terms are large. <y> (along n) is
  driven by the nondipole terms alone.
* The electric terms equal hand-written FieldTerm lines with analytic E', E''.
* Rotation covariance (3D, full Multipole = 3): (n, E) = (z, x) and (x, y) in an
  isotropic potential give the same <r>, rotated; the norm stays 1 (Hermitian).
"""

import numpy as np
import pytest
import zkit
from scipy.integrate import solve_ivp
from zkit.layouts import dipole_layout

from tests.test_tdsez import REPO, _expect_fatal, _run, _step, _write_tmp

E0, W, TP = 0.3, 1.0, 20.0   # field amplitude, frequency, sin^2 pulse length
LASER = f"t < {TP} ? {E0}*sin(pi*t/{TP})^2*sin({W}*t) : 0"


def _field(t, tp=TP):
    """E, E', E'' of the sin^2 pulse (analytic)."""
    a = np.pi / tp
    s, c = np.sin(a * t) ** 2, np.sin(2 * a * t) * a          # env, env'
    c2 = 2 * a * a * np.cos(2 * a * t)                          # env''
    on = t < tp
    e = E0 * s * np.sin(W * t)
    e1 = E0 * (c * np.sin(W * t) + s * W * np.cos(W * t))
    e2 = E0 * (c2 * np.sin(W * t) + 2 * c * W * np.cos(W * t) - s * W * W * np.sin(W * t))
    return np.where(on, e, 0), np.where(on, e1, 0), np.where(on, e2, 0)


HO2 = """Verbose = 0
Dimension = 2
Domain = -12, 12, -12, 12
SplineDegree = 6
NQuadratures = 9
KnotSequence = uniform
NSplines = 40
Charge = -1.0
Mass = 1.5
Potential = 0.5*1.5*0.25*(x*x + y*y)
PotentialDerivativeX = 1.5*0.25*x
PotentialDerivativeY = 1.5*0.25*y
TargetEigenvalue = 0
NBoundStates = 1
InitialState = ground
EnablePropagation = 1
EnableCAP = 0
Propagator = magnus4
TimeStep = 0.01
FinalTime = 24
OutputStrideWFS = 0
Polarization = x
Omegax = 1.0
"""


def _evolution(name, text, nproc=2):
    path = _write_tmp(name, text)
    r, _, _ = _run(path, zkit.find_binary(), nproc)
    _step(f"{name} runs", r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-500:])
    return zkit.Run(REPO, path).evolution


def test_multipole2_oscillator_follows_classical_equations():
    c, q, m, w0 = 2.0, -1.0, 1.5, 0.5
    ev = _evolution("mp2_ho.inp", HO2 + f"LaserX = {LASER}\nMultipole = 2\nPropagationDirection = 0, 1, 0\nSpeedOfLight = {c}\n")
    t = ev.time

    def rhs(tt, s):
        x, y, px, py = s
        e, e1, _ = _field(np.array(tt))
        cz = -e                                   # C = n x E = y x x E = -E z
        k = q / (2 * m * c) * cz
        return [px / m + k * y, py / m - k * x,
                -m * w0**2 * x + q * e - q / (2 * c) * y * e1 + k * py,
                -m * w0**2 * y - q / (2 * c) * x * e1 - k * px]

    sol = solve_ivp(rhs, (0, t[-1]), [0, 0, 0, 0], t_eval=t, rtol=1e-11, atol=1e-13, max_step=0.02)
    L2 = dipole_layout(2)
    dx, dy = ev.dipoles[:, L2.dipole_x], ev.dipoles[:, L2.dipole_y]     # q<x>, q<y>
    rx, ry = q * sol.y[0], q * sol.y[1]
    ex = np.max(np.abs(dx - rx)) / np.max(np.abs(rx))
    ey = np.max(np.abs(dy - ry)) / np.max(np.abs(ry))
    _step(f"nondipole drift along n is present (max |<y>| = {np.max(np.abs(ry)):.3f})", np.max(np.abs(ry)) > 1e-2)
    _step(f"q<x>(t) = classical ({ex:.1e})", ex < 1e-6)
    _step(f"q<y>(t) = classical, nondipole only ({ey:.1e})", ey < 1e-6)
    _step("norm = 1 (Hermitian terms)", np.max(np.abs(ev.energies[:, 6] - 1)) < 1e-10)


def test_electric_multipoles_equal_hand_written_field_terms():
    c = 2.0
    a = np.pi / TP
    # analytic E' and E'' of the pulse as muParser expressions
    env, d1, d2 = f"sin({a}*t)^2", f"{a}*sin(2*{a}*t)", f"2*{a}^2*cos(2*{a}*t)"
    e1 = f"(t < {TP} ? {E0}*({d1}*sin({W}*t) + {env}*{W}*cos({W}*t)) : 0)"
    e2 = f"(t < {TP} ? {E0}*({d2}*sin({W}*t) + 2*{d1}*{W}*cos({W}*t) - {env}*{W}^2*sin({W}*t)) : 0)"
    base = HO2 + f"LaserX = {LASER}\n"
    auto = _evolution("mp3_electric.inp", base + f"Multipole = 3\nMultipoleMagnetic = 0\nPropagationDirection = 0, 1, 0\nSpeedOfLight = {c}\n")
    # q = -1, s = 1: + q/(2c) x y E' = s * (-E'/(2c)) x y ;  - q/(6c^2) x y^2 E'' = s * E''/(6c^2) x y^2
    path = _write_tmp("mp3_hand.inp", base + f"FieldTerm = x*y ; -{e1}/(2*{c})\nFieldTerm = x*y^2 ; {e2}/(6*{c}^2)\n", raw=True)
    r, _, _ = _run(path, zkit.find_binary(), 2)
    _step("hand-written FieldTerms run", r.returncode == 0, r.stdout.decode()[-1500:])
    hand = zkit.Run(REPO, path).evolution
    L2 = dipole_layout(2)
    cols = [L2.dipole_x, L2.dipole_y]
    err = np.max(np.abs(auto.dipoles[:, cols] - hand.dipoles[:, cols])) / np.max(np.abs(hand.dipoles[:, cols]))
    # E'' of the sin^2 pulse jumps at t = TP: the stencil straddles it at that one sample
    off = np.abs(hand.time - TP) > 0.05
    eerr = np.max(np.abs(auto.energies[off, 1:5] - hand.energies[off, 1:5])) / np.max(np.abs(hand.energies[:, 1:5]))
    _step(f"<r>: Multipole = 3 (electric) = FieldTerms ({err:.1e})", err < 1e-8)
    _step(f"energies: Multipole = 3 (electric) = FieldTerms ({eerr:.1e})", eerr < 1e-8)


HO3 = """Verbose = 0
Dimension = 3
Domain = -9, 9, -9, 9, -9, 9
SplineDegree = 4
NQuadratures = 6
KnotSequence = uniform
NSplines = 18
Charge = -1.0
Mass = 1.0
Potential = 0.5*0.36*(x*x + y*y + z*z)
PotentialDerivativeX = 0.36*x
PotentialDerivativeY = 0.36*y
PotentialDerivativeZ = 0.36*z
TargetEigenvalue = 0
NBoundStates = 1
InitialState = ground
EnablePropagation = 1
EnableCAP = 0
Propagator = pade22
TimeStep = 0.02
FinalTime = 12
OutputStrideWFS = 0
Multipole = 3
SpeedOfLight = 1.5
"""


def test_multipole3_rotation_covariance_3d():
    laser = LASER.replace(str(TP), "10")
    a = _evolution("mp3_rot_a.inp", HO3 + f"Polarization = x\nOmegax = 1\nLaserX = {laser}\nPropagationDirection = 0, 0, 1\n", 4)
    b = _evolution("mp3_rot_b.inp", HO3 + f"Polarization = y\nOmegay = 1\nLaserY = {laser}\nPropagationDirection = 1, 0, 0\n", 4)
    # rotation x -> y, y -> z, z -> x maps (n, E) = (z, x) onto (x, y)
    L3 = dipole_layout(3)
    cols = [L3.dipole_x, L3.dipole_y, L3.dipole_z]
    da, db = a.dipoles[:, cols], b.dipoles[:, cols]
    rot = np.stack([db[:, 1], db[:, 2], db[:, 0]], 1)   # b's (y, z, x) = a's (x, y, z)
    err = np.max(np.abs(da - rot)) / np.max(np.abs(da))
    _step(f"<r> along n is driven ({np.max(np.abs(da[:, 2])):.2e})", np.max(np.abs(da[:, 2])) > 1e-4)
    _step(f"<r> covariant under the rotation ({err:.1e})", err < 1e-9)
    for lab, ev in (("a", a), ("b", b)):
        _step(f"{lab}: norm = 1 (Hermitian terms)", np.max(np.abs(ev.energies[:, 6] - 1)) < 1e-10)


@pytest.mark.parametrize(
    "extra,label",
    [
        ("LaserX = 0.1*sin(t)\nMultipole = 4\n", "mp_order"),
        ("LaserX = 0.1*sin(t)\nMultipole = 2\n", "mp_no_direction"),
        ("LaserX = 0.1*sin(t)\nMultipole = 2\nPropagationDirection = 1, 0, 0\n", "mp_not_transverse"),
        ("LaserX = 0.1*sin(t)\nMultipole = 2\nPropagationDirection = 0, 0, 1\n", "mp_2d_out_of_plane"),
        ("LaserX = 0.1*sin(t)\nMultipole = 2\nPropagationDirection = 0, 1, 0\nAccelerationMethod = operator\n", "mp_operator_acc"),
    ],
)
def test_multipole_guards(extra, label):
    _expect_fatal(HO2 + extra, label)


# ---------------------------------------------------------------------------
# Full Multipole = 3 (all five nondipole terms, magnetic and diamagnetic
# included) against an independent reference: an anisotropic oscillator in its
# own Hermite basis. Every term is a polynomial in r and p, so its matrix follows
# exactly from the ladder operators (x = (a + a+)/sqrt(2 m w), p = i sqrt(m w/2)(a+ - a)),
# with no grid; the TDSE in that basis is integrated by SciPy (DOP853, 1e-11).
# Also checked: the 1/c^2 part alone, <r>(Multipole = 3) - <r>(Multipole = 2).
# ---------------------------------------------------------------------------
import scipy.sparse as sp


def _hermite_reference(ws, m, q, c, mp, n, pol, ts, nb, tp):
    """q<r_a>(t) for the oscillator frequencies ws, field along axis pol, direction n."""
    dim = len(ws)
    lad = sp.diags(np.sqrt(np.arange(1, nb)), 1, format="csr")
    eye = sp.identity(nb, format="csr")

    def embed(op, a):
        mats = [eye] * dim
        mats[a] = op
        out = mats[0]
        for k in mats[1:]:
            out = sp.kron(out, k, format="csr")
        return out

    R = [embed((lad + lad.T) / np.sqrt(2 * m * w), a) for a, w in enumerate(ws)]
    P = [embed(1j * np.sqrt(m * w / 2) * (lad.T - lad), a) for a, w in enumerate(ws)]
    zero = sp.csr_matrix((nb**dim, nb**dim))
    R3, P3 = R + [zero] * (3 - dim), P + [zero] * (3 - dim)
    L = [R3[1] @ P3[2] - R3[2] @ P3[1], R3[2] @ P3[0] - R3[0] @ P3[2], R3[0] @ P3[1] - R3[1] @ P3[0]]
    levels = sum(np.meshgrid(*[w * (np.arange(nb) + 0.5) for w in ws], indexing="ij"))
    H0 = sp.diags(levels.ravel())
    nr = sum(n[a] * R[a] for a in range(dim))
    ex = np.zeros(3); ex[pol] = 1.0
    cvec = np.cross(n, ex)                                   # C = n x E = cvec E(t)
    r2 = sum(Ra @ Ra for Ra in R)
    Ep = R[pol]
    quad = nr @ Ep                                           # (n.r)(r.e)
    octu = nr @ nr @ Ep                                      # (n.r)^2 (r.e)
    mag = sum(cvec[b] * L[b] for b in range(3))              # L.C / E
    magq = 0.5 * (nr @ mag + mag @ nr)                       # {n.r, L}/2 . C / E
    cr = sum(cvec[b] * R3[b] for b in range(3))
    dia = r2 - cr @ cr                                       # |c x r|^2 = |c|^2 r^2 - (c.r)^2, |c| = 1
    s = -q

    def H(t):
        e, e1, e2 = (float(v) for v in _field(np.array(t), tp))
        h = H0 + s * e * Ep
        if mp >= 2:
            h = h + q / (2 * c) * e1 * quad - q / (2 * m * c) * e * mag
        if mp >= 3:
            h = h - q / (6 * c * c) * e2 * octu + q / (3 * m * c * c) * e1 * magq + q * q / (8 * m * c * c) * e * e * dia
        return h

    psi0 = np.zeros(nb**dim, complex)
    psi0[0] = 1.0
    sol = solve_ivp(lambda t, y: -1j * (H(t) @ y), (0, ts[-1]), psi0, t_eval=ts, method="DOP853", rtol=1e-11, atol=1e-13)
    psi = sol.y
    occ = (np.abs(psi[:, -1]) ** 2).reshape((nb,) * dim)
    edge = sum(np.take(occ, range(nb - 3, nb), axis=a).sum() for a in range(dim))
    return np.stack([q * np.real(np.einsum("ik,ik->k", psi.conj(), Ra @ psi)) for Ra in R], 1), edge


HO2A = HO2.replace("Potential = 0.5*1.5*0.25*(x*x + y*y)", "Potential = 0.5*1.5*(0.25*x*x + 0.49*y*y)").replace(
    "PotentialDerivativeY = 1.5*0.25*y", "PotentialDerivativeY = 1.5*0.49*y")


@pytest.mark.parametrize("dim", [2, 3])
def test_multipole3_against_hermite_basis_reference(dim):
    if dim == 2:
        ws, m, nb, n, deck, npr = (0.5, 0.7), 1.5, 36, (0.0, 1.0, 0.0), HO2A, 2
        base = deck + f"LaserX = {LASER}\nPropagationDirection = 0, 1, 0\nSpeedOfLight = 2.0\n"
    else:
        ws, m, nb, n, npr = (0.6, 0.65, 0.7), 1.0, 18, (0.0, 0.0, 1.0), 4
        base = HO3.replace("Multipole = 3\nSpeedOfLight = 1.5\n", "").replace(
            "Potential = 0.5*0.36*(x*x + y*y + z*z)", "Potential = 0.5*(0.36*x*x + 0.4225*y*y + 0.49*z*z)").replace(
            "PotentialDerivativeY = 0.36*y", "PotentialDerivativeY = 0.4225*y").replace(
            "PotentialDerivativeZ = 0.36*z", "PotentialDerivativeZ = 0.49*z").replace(
            "SplineDegree = 4", "SplineDegree = 6").replace("NQuadratures = 6", "NQuadratures = 8").replace(
            "NSplines = 18", "NSplines = 22").replace("Domain = -9, 9, -9, 9, -9, 9", "Domain = -8, 8, -8, 8, -8, 8") + (
            f"Polarization = x\nOmegax = 1\nLaserX = {LASER.replace(str(TP), '10')}\nPropagationDirection = 0, 0, 1\nSpeedOfLight = 2.0\n")
    q, c = -1.0, 2.0
    got = {mp: _evolution(f"mp_herm{dim}d_{mp}.inp", base + f"Multipole = {mp}\n", npr) for mp in (2, 3)}
    ts = got[3].time
    tp = TP if dim == 2 else 10.0                   # the 3D deck uses a 10 a.u. pulse
    ref = {mp: _hermite_reference(ws, m, q, c, mp, np.array(n), 0, ts, nb, tp) for mp in (2, 3)}
    # grid error of the dipole dynamics alone: 4e-7 (2D), 8e-6 (3D, degree 6, h = 0.73)
    tol = 1e-6 if dim == 2 else 5e-5
    lay = dipole_layout(dim)
    cols = [lay.dipole_x, lay.dipole_y, lay.dipole_z][:dim]
    for mp in (2, 3):
        r, edge = ref[mp]
        _step(f"{dim}D Multipole = {mp}: reference converged (top-state population {edge:.0e})", edge < 1e-12)
        d = got[mp].dipoles[:, cols]
        err = np.max(np.abs(d - r)) / np.max(np.abs(r))
        _step(f"{dim}D Multipole = {mp}: q<r>(t) = Hermite reference ({err:.1e})", err < tol)
    # the 1/c^2 terms alone: octupole, magnetic quadrupole, diamagnetic
    a = n.index(1.0)
    d23 = got[3].dipoles[:, cols[a]] - got[2].dipoles[:, cols[a]]
    r23 = ref[3][0][:, a] - ref[2][0][:, a]
    err = np.max(np.abs(d23 - r23)) / np.max(np.abs(r23))
    _step(f"{dim}D 1/c^2 part of q<r_n> (max {np.max(np.abs(r23)):.2e}) = reference ({err:.1e})", err < 50 * tol)
