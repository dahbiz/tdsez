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
from scipy.integrate import solve_ivp

import zkit
from tests.test_tdsez import REPO, _expect_fatal, _run, _step, _write_tmp

E0, W, TP = 0.3, 1.0, 20.0   # field amplitude, frequency, sin^2 pulse length
LASER = f"t < {TP} ? {E0}*sin(pi*t/{TP})^2*sin({W}*t) : 0"


def _field(t):
    """E, E', E'' of the sin^2 pulse (analytic)."""
    a = np.pi / TP
    s, c = np.sin(a * t) ** 2, np.sin(2 * a * t) * a          # env, env'
    c2 = 2 * a * a * np.cos(2 * a * t)                          # env''
    on = t < TP
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
    dx, dy = ev.dipoles[:, 3], ev.dipoles[:, 4]     # q<x>, q<y>
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
    err = np.max(np.abs(auto.dipoles[:, 3:5] - hand.dipoles[:, 3:5])) / np.max(np.abs(hand.dipoles[:, 3:5]))
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
    da, db = a.dipoles[:, 4:7], b.dipoles[:, 4:7]
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
