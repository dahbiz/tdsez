"""Gauge = velocity: H = (p - q A(t - n.r/c))^2 / 2m + V, against analytics.

The pulses are defined by their vector potential, A = A0 sin^2(pi t/T) sin(w t)
(E = -A'), so A(T) = 0 and the final populations are gauge invariant too.

* Dipole: velocity = length gauge for q<r>(t), the acceleration and the
  final populations (1D atom, 2D circular field).
* Multipole = 2 keeps H quadratic for an oscillator: <r>(t) follows the
  classical Hamilton equations of the velocity-gauge Hamiltonian.
* Multipole = 3 against the Hermite-basis reference (exact ladder-operator
  matrices of every term).
* Velocity and multipolar gauges agree to the truncation order: at
  Multipole = 3 they differ by O(1/c^3), so doubling c divides the gap by ~8.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import sympy as sy
import zkit
from scipy.integrate import solve_ivp
from zkit.layouts import dipole_layout

from tests.test_tdsez import REPO, _expect_fatal, _run, _step, _write_tmp

_t = sy.Symbol("t")


def _pulse(A0, w, T, phase=0.0, power=2):
    """A(t) and E(t) = -A'(t) of a sin^power pulse: muParser text and numpy callables."""
    A = A0 * sy.sin(sy.pi * _t / T) ** power * sy.sin(w * _t + phase)
    E = -sy.diff(A, _t)
    text = str(E).replace("**", "^")
    on = lambda f: (lambda tt: np.where(np.asarray(tt) < T, f(tt), 0.0))
    funcs = [on(sy.lambdify(_t, e, "numpy")) for e in (A, E, sy.diff(E, _t))]
    return f"t < {T} ? {text} : 0", funcs


def _evolution(name, text, nproc=2):
    path = _write_tmp(name, text)
    r, _, _ = _run(path, zkit.find_binary(), nproc)
    _step(f"{name} runs", r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-500:])
    return zkit.Run(REPO, path).evolution


ATOM1D = """Verbose = 0
Dimension = 1
Domain = -100, 100
SplineDegree = 7
NQuadratures = 9
NSplines = 900
Potential = -1/sqrt(x*x + 2)
PotentialDerivativeX = x/(x*x + 2)^1.5
NBoundStates = 4
TargetEigenvalue = -0.6
InitialState = ground
EnablePropagation = 1
EnableCAP = 0
Propagator = magnus4
TimeStep = 0.01
FinalTime = 40
OutputStrideWFS = 0
PhysicsOutput = dipole, population, energy
Polarization = x
Omegax = 0.6
"""

ATOM2D = """Verbose = 0
Dimension = 2
Domain = -25, 25, -25, 25
SplineDegree = 6
NQuadratures = 8
NSplines = 130
Potential = -1/sqrt(x*x + y*y + 0.6363)
PotentialDerivativeX = x/(x*x + y*y + 0.6363)^1.5
PotentialDerivativeY = y/(x*x + y*y + 0.6363)^1.5
NBoundStates = 4
TargetEigenvalue = -0.6
InitialState = ground
EnablePropagation = 1
EnableCAP = 0
Propagator = magnus4
TimeStep = 0.01
FinalTime = 20
OutputStrideWFS = 0
PhysicsOutput = dipole, population, energy
Polarization = xy
Omegax = 1.0
"""


@pytest.mark.parametrize("dim", [1, 2])
def test_dipole_velocity_equals_length_gauge(dim):
    if dim == 1:
        ex, _ = _pulse(0.25, 0.6, 40.0)
        deck = ATOM1D + f"LaserX = {ex}\n"
        tol = 1e-8
    else:  # circular
        ex, _ = _pulse(0.2, 1.0, 20.0)
        ey, _ = _pulse(0.2, 1.0, 20.0, np.pi / 2)
        deck = ATOM2D + f"LaserX = {ex}\nLaserY = {ey}\n"
        tol = 1e-7                    # 1.8e-6 with 90 splines, 1.4e-8 with 130: discretisation
    ln = _evolution(f"vg_len{dim}d.inp", deck)
    ve = _evolution(f"vg_vel{dim}d.inp", deck + "Gauge = velocity\n")
    lay = dipole_layout(dim)
    d = [lay.dipole_x, lay.dipole_y][:dim]
    a = [lay.accel_x, lay.accel_y][:dim]
    # The length-gauge field term of the discrete acceleration, E.<[D, Vel]_M>, equals
    # E only up to basis truncation near the walls (x large): 1.8e-5 in this +-25
    # box, 4e-8 in +-40 at the same spacing; in the velocity gauge it is exact.
    for lab, cols, tl in (("q<r>(t)", d, tol), ("acceleration", a, tol if dim == 1 else 3e-5)):
        ref = ln.dipoles[:, cols]
        err = np.max(np.abs(ve.dipoles[:, cols] - ref)) / np.max(np.abs(ref))
        _step(f"{dim}D {lab}: velocity = length gauge ({err:.1e})", err < tl)
    pl, pv = ln.populations[-1, 1:], ve.populations[-1, 1:]
    _step(f"{dim}D final bound populations (A(T) = 0): {np.max(np.abs(pl - pv)):.1e}",
          np.max(np.abs(pl - pv)) < tol * 10)


HO2 = """Verbose = 0
Dimension = 2
Domain = -12, 12, -12, 12
SplineDegree = 6
NQuadratures = 9
KnotSequence = uniform
NSplines = 40
Charge = -1.0
Mass = 1.5
Potential = 0.5*1.5*(0.25*x*x + 0.49*y*y)
PotentialDerivativeX = 1.5*0.25*x
PotentialDerivativeY = 1.5*0.49*y
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
PropagationDirection = 0, 1, 0
"""
A0, W, TP = 0.3, 1.0, 20.0
Q, M, WX, WY = -1.0, 1.5, 0.5, 0.7


def test_multipole2_velocity_gauge_oscillator_is_classical():
    """H = p^2/2m + V - (q/m) A p_x - (q/mc) y E p_x + (q^2/mc) y A E is quadratic."""
    c = 2.0
    ex, (Af, Ef, _) = _pulse(A0, W, TP)
    ev = _evolution("vg_mp2_ho.inp", HO2 + f"LaserX = {ex}\nGauge = velocity\nMultipole = 2\nSpeedOfLight = {c}\n")

    def rhs(t, s):
        x, y, px, py = s
        A, E = float(Af(t)), float(Ef(t))
        return [px / M - Q / M * A - Q / (M * c) * y * E, py / M,
                -M * WX**2 * x, -M * WY**2 * y + Q / (M * c) * E * px - Q * Q / (M * c) * A * E]

    t = ev.time
    sol = solve_ivp(rhs, (0, t[-1]), [0, 0, 0, 0], t_eval=t, rtol=1e-11, atol=1e-13, max_step=0.02)
    lay = dipole_layout(2)
    scale = np.max(np.abs(Q * sol.y[:2]))
    for k, (lab, col) in enumerate((("x", lay.dipole_x), ("y", lay.dipole_y))):
        ref = Q * sol.y[k]
        err = np.max(np.abs(ev.dipoles[:, col] - ref)) / scale
        _step(f"q<{lab}>(t) = classical, velocity gauge 1/c ({err:.1e}; max {np.max(np.abs(ref)):.2e})", err < 1e-6)


def _hermite_velocity(ws, m, q, c, order, ts, nb, Af, Ef, dEf):
    """q<x>, q<y> of the 2D oscillator (n = y, E along x) in its Hermite basis."""
    lad = sp.diags(np.sqrt(np.arange(1, nb)), 1, format="csr")
    eye = sp.identity(nb, format="csr")
    X = sp.kron((lad + lad.T) / np.sqrt(2 * m * ws[0]), eye, format="csr")
    Y = sp.kron(eye, (lad + lad.T) / np.sqrt(2 * m * ws[1]), format="csr")
    PX = sp.kron(1j * np.sqrt(m * ws[0] / 2) * (lad.T - lad), eye, format="csr")
    H0 = sp.diags(np.add.outer(ws[0] * (np.arange(nb) + 0.5), ws[1] * (np.arange(nb) + 0.5)).ravel())
    YPX, Y2, Y2PX = Y @ PX, Y @ Y, Y @ Y @ PX

    def H(t):
        A, E, dE = float(Af(t)), float(Ef(t)), float(dEf(t))
        h = H0 - q / m * A * PX
        if order >= 2:
            h = h - q / (m * c) * E * YPX + q * q / (m * c) * A * E * Y
        if order >= 3:
            h = h + q / (2 * m * c * c) * dE * Y2PX + q * q / (2 * m * c * c) * (E * E - A * dE) * Y2
        return h

    psi0 = np.zeros(nb * nb, complex)
    psi0[0] = 1.0
    sol = solve_ivp(lambda t, y: -1j * (H(t) @ y), (0, ts[-1]), psi0, t_eval=ts, method="DOP853", rtol=1e-11, atol=1e-13)
    psi = sol.y
    occ = (np.abs(psi[:, -1]) ** 2).reshape(nb, nb)
    edge = occ[-3:].sum() + occ[:, -3:].sum()
    return np.stack([q * np.real(np.einsum("ik,ik->k", psi.conj(), R @ psi)) for R in (X, Y)], 1), edge


def test_multipole3_velocity_gauge_against_hermite_reference():
    c = 2.0
    ex, (Af, Ef, dEf) = _pulse(A0, W, TP)
    got = {o: _evolution(f"vg_herm_{o}.inp", HO2 + f"LaserX = {ex}\nGauge = velocity\nMultipole = {o}\nSpeedOfLight = {c}\n")
           for o in (2, 3)}
    lay = dipole_layout(2)
    cols = [lay.dipole_x, lay.dipole_y]
    ts = got[3].time
    ref = {o: _hermite_velocity((WX, WY), M, Q, c, o, ts, 36, Af, Ef, dEf) for o in (2, 3)}
    for o in (2, 3):
        r, edge = ref[o]
        _step(f"Multipole = {o}: reference converged (top states {edge:.0e})", edge < 1e-12)
        err = np.max(np.abs(got[o].dipoles[:, cols] - r)) / np.max(np.abs(r))
        _step(f"Multipole = {o}: q<r>(t) = Hermite reference ({err:.1e})", err < 1e-6)
    d23 = got[3].dipoles[:, cols[1]] - got[2].dipoles[:, cols[1]]
    r23 = ref[3][0][:, 1] - ref[2][0][:, 1]
    err = np.max(np.abs(d23 - r23)) / np.max(np.abs(r23))
    _step(f"1/c^2 part of q<y> (max {np.max(np.abs(r23)):.2e}) = reference ({err:.1e})", err < 1e-4)


def test_velocity_and_multipolar_gauges_agree_to_truncation_order():
    """Both keep the fields to O(1/c^2); they differ at O(1/c^3). (Both also converge
    to the untruncated A(t - y/c) as 1/c^3.) A sin^4 envelope keeps E'' continuous:
    with a jump in E' the octupole term has an impulse (TDSEZWarnFieldKinks)."""
    ex, _ = _pulse(A0, W, TP, power=4)
    lay = dipole_layout(2)
    cols = [lay.dipole_x, lay.dipole_y]
    gap = {}
    for c in (4.0, 8.0):
        base = HO2 + f"LaserX = {ex}\nMultipole = 3\nSpeedOfLight = {c}\n"
        vel = _evolution(f"vg_ord_v{int(c)}.inp", base + "Gauge = velocity\n")
        mpl = _evolution(f"vg_ord_m{int(c)}.inp", base)
        gap[c] = np.max(np.abs(vel.dipoles[:, cols] - mpl.dipoles[:, cols]))
    ratio = gap[4.0] / gap[8.0]
    _step(f"velocity - multipolar gap: {gap[4.0]:.2e} (c = 4), {gap[8.0]:.2e} (c = 8); ratio {ratio:.2f} ~ 8 (1/c^3)",
          6.5 < ratio < 10)


@pytest.mark.parametrize(
    "extra,label",
    [
        ("Gauge = velocity\nMode = kh\n", "vg_kh"),
        ("Gauge = velocity\nTSurff = 1\n", "vg_tsurff"),
        ("Gauge = velocity\nAccelerationMethod = operator\n", "vg_operator_acc"),
        ("Gauge = velocity\nMass = 1 + 0.01*x*x\n", "vg_varying_mass"),
        ("Gauge = coulomb\n", "vg_bad_gauge"),
    ],
)
def test_velocity_gauge_guards(extra, label):
    deck = """Verbose = 0
Dimension = 1
Domain = -20, 20
NSplines = 60
Potential = 0.5*x*x
NBoundStates = 2
InitialState = ground
EnablePropagation = 1
TimeStep = 0.05
FinalTime = 2
Polarization = x
LaserX = 0.1*sin(t)
"""
    _expect_fatal(deck + extra, label)


@pytest.mark.parametrize("gauge,order,warn", [("length", 3, True), ("length", 2, False), ("velocity", 3, False)])
def test_field_kink_warning(gauge, order, warn):
    """A-defined sin^2 pulse: E continuous, E' jumps at T. The length-gauge octupole
    (E'') then has an impulse and TDSE-Z warns; the other terms need E or E' only."""
    ex, _ = _pulse(A0, W, TP)
    deck = HO2.replace("FinalTime = 24", "FinalTime = 21").replace("TimeStep = 0.01", "TimeStep = 0.1")
    deck = deck.replace("Verbose = 0", "Verbose = 1") + f"LaserX = {ex}\nGauge = {gauge}\nMultipole = {order}\nSpeedOfLight = 4\n"
    path = _write_tmp(f"vg_kink_{gauge}{order}.inp", deck)
    r, _, _ = _run(path, zkit.find_binary(), 1)
    out = " ".join(r.stdout.decode().split())
    _step(f"{gauge} gauge, Multipole = {order}: warning {'present' if warn else 'absent'}",
          r.returncode == 0 and (("jumps near t = 20" in out) == warn), out[-800:])
