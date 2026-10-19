"""Gauge = velocity: H = (p - q A(t))^2 / 2m + V, against analytics.

The pulses are defined by their vector potential, A = A0 sin^2(pi t/T) sin(w t)
(E = -A'), so A(T) = 0 and the final populations are gauge invariant too.

* Dipole: velocity = length gauge for q<r>(t), the acceleration and the
  final populations (1D atom, 2D circular field).
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


