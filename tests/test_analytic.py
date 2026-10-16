"""Analytic regression tests for TDSE-Z.

Every test compares the solver with an exact result -- a closed form or an
independent high-precision solution computed here -- with tolerances close to
the accuracy the discretisation actually reaches, so that a regression of a
few digits is caught. All cases are small (1D, or 2D/3D with a handful of
elements) and run in a few seconds each.

    pytest tests/test_analytic.py -v
"""

import os
import re
import subprocess

import h5py
import numpy as np
import pytest
from scipy.optimize import brentq

import zkit

NM = 18.8973          # bohr per nm
HA_MEV = 27211.386    # meV per hartree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(tmp_path, name, text, nproc=2, args=()):
    """Run tdsez on an input text inside tmp_path; return (log, static h5, td h5)."""
    inp = tmp_path / f"{name}.inp"
    inp.write_text(text)
    cmd = ["mpirun", "--bind-to", "none", "-np", str(nproc), zkit.find_binary(),
           "-inp", inp.name, *args]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    log = " ".join(log.split())  # wrapped log lines
    assert proc.returncode == 0, f"tdsez failed (rc={proc.returncode}):\n{log[-2000:]}"
    return (log, tmp_path / "static" / f"EigenData_{inp.name}.h5",
            tmp_path / "td" / f"TimeEvolutionData_{inp.name}.h5")


def spectrum(h5):
    with h5py.File(h5) as f:
        return np.asarray(f["spectrum"])[:, 0]


def series(h5, name, width):
    with h5py.File(h5) as f:
        return np.real(np.asarray(f[name])).reshape(-1, width)


def log_value(log, pattern):
    m = re.findall(pattern, log)
    assert m, f"pattern {pattern!r} not found in log"
    return float(m[-1])


def layered_neumann(widths, masses, pots, nmin, emax):
    """Exact BDD eigenvalues of a piecewise-constant layered system with
    homogeneous Neumann walls: psi and psi'/m continuous at every interface.
    Returns the first `nmin` roots below emax (hartree)."""
    def f(E):
        u = np.array([1.0, 0.0])                      # (psi, psi'/m) at the left wall
        for w, m, V in zip(widths, masses, pots):
            q2 = 2.0 * m * (E - V)
            if q2 > 0:
                k = np.sqrt(q2)
                c, s = np.cos(k * w), np.sin(k * w)
                T = np.array([[c, m * s / k], [-k * s / m, c]])
            else:
                k = np.sqrt(-q2)
                c, s = np.cosh(k * w), np.sinh(k * w)
                T = np.array([[c, m * s / k], [k * s / m, c]])
            u = T @ u
            u /= np.max(np.abs(u))
        return u[1]                                     # psi'/m at the right wall
    Es = np.linspace(1e-6, emax, 20000)
    fs = np.array([f(E) for E in Es])
    roots = [brentq(f, Es[i], Es[i + 1], xtol=1e-16, rtol=1e-15)
             for i in range(len(Es) - 1) if fs[i] * fs[i + 1] < 0]
    return np.array(roots[:nmin])


def mass_step_dirichlet(m1, m2, nmin):
    """Exact energies of V=0 in [-1,1] with Dirichlet walls and m=m1 (x<0), m2 (x>0)."""
    def f(E):
        k1, k2 = np.sqrt(2 * m1 * E), np.sqrt(2 * m2 * E)
        return (k1 / m1) * np.cos(k1) * np.sin(k2) + (k2 / m2) * np.sin(k1) * np.cos(k2)
    Es = np.linspace(1e-6, 40.0, 40000)
    fs = f(Es)
    roots = [brentq(f, Es[i], Es[i + 1], xtol=1e-16, rtol=1e-15)
             for i in range(len(Es) - 1) if fs[i] * fs[i + 1] < 0]
    return np.array(roots[:nmin])


HEAD_1D = """\
Verbose           = 0
Dimension         = 1
Charge            = -1.0
UseDirectSolve    = 1
EnableCAP         = 0
Polarization      = x
"""


# ---------------------------------------------------------------------------
# Static spectra
# ---------------------------------------------------------------------------

def test_infinite_well(tmp_path):
    """Particle in a box [0, 10], Dirichlet walls: E_n = n^2 pi^2 / (2 L^2)."""
    text = HEAD_1D + """\
Domain = 0.0, 10.0
BoundaryType = Dirichlet
SplineDegree = 7
NQuadratures = 10
KnotSequence = uniform
NSplines = 30
Mass = 1.0
MassIsConstant = 1
Potential = 0.0
PotentialDerivativeX = 0.0
TargetEigenvalue = 0.0
NBoundStates = 4
EnablePropagation = 0
"""
    _, h5, _ = run(tmp_path, "well", text)
    E = spectrum(h5)[:4]
    exact = np.array([n * n * np.pi ** 2 / 200.0 for n in range(1, 5)])
    assert np.max(np.abs(E - exact) / exact) < 1e-11, (E, exact)


def test_quesne_position_dependent_mass(tmp_path):
    """Quesne oscillator, m(x) = (1 + x/a)^-1 with a = 2: E_n = (n + 1/2)/2."""
    text = HEAD_1D + """\
Domain = -1.999999999, 50.0
BoundaryType = Dirichlet
SplineDegree = 7
NQuadratures = 10
KnotSequence = uniform
NSplines = 1000
Mass = (1.0 + x/2.0)^(-1.0)
MassIsConstant = 0
Potential = x*x/(4.0*(x + 2.0))
PotentialDerivativeX = x*(x + 4.0)/(4.0*(x + 2.0)*(x + 2.0))
TargetEigenvalue = 0.25
NBoundStates = 5
EnablePropagation = 0
"""
    _, h5, _ = run(tmp_path, "quesne", text)
    E = spectrum(h5)[:5]
    exact = 0.5 * (np.arange(5) + 0.5)
    assert np.max(np.abs(E - exact)) < 1e-9, (E, exact)


@pytest.mark.parametrize("bc", ["Dirichlet", "Neumann"])
def test_mass_step_interface_knots(tmp_path, bc):
    """Abrupt mass step m = 1 | 4 at x = 0: with C0 interface knots the BDD
    spectrum is exact to round-off already on a very coarse mesh (this also
    guards the symmetric Dirichlet wall treatment)."""
    text = HEAD_1D + f"""\
Domain = -1.0, 1.0
BoundaryType = {bc}
SplineDegree = 7
NQuadratures = 10
KnotSequence = interfaces
NSplines = 12
Mass = (x < 0.0) ? 1.0 : 4.0
MassIsConstant = 0
Potential = 0.0
PotentialDerivativeX = 0.0
TargetEigenvalue = 0.0
NBoundStates = 3
EnablePropagation = 0
"""
    log, h5, _ = run(tmp_path, f"mstep_{bc}", text)
    assert "C0 interface" in log
    E = np.sort(spectrum(h5)[:3])
    if bc == "Dirichlet":
        exact = mass_step_dirichlet(1.0, 4.0, 3)
        assert np.allclose(exact[:2], [0.6615513245573471, 1.9826989028677189], atol=1e-12)
    else:
        # Neumann walls: the constant function is the E = 0 ground state.
        assert abs(E[0]) < 1e-11, E
        exact = np.concatenate([[0.0], layered_neumann([1.0, 1.0], [1.0, 4.0], [0.0, 0.0], 2, 10.0)])
    assert np.max(np.abs(E[:2] - exact[:2])) < 1e-12, (E, exact)
    assert abs(E[2] - exact[2]) < 1e-9, (E, exact)


def test_double_quantum_well_exact(tmp_path):
    """GaAs/Al0.3Ga0.7As double quantum well (w_b = 4 nm, Neumann walls) against
    the exact transfer-matrix solution of the BDD problem."""
    a, b, L = 2.0 * NM, 10.0 * NM, 15.0 * NM
    mw, mb, V0 = 0.067, 0.0879, 0.00893
    m = f"(abs(x)<{a})?{mb}:((abs(x)<{b})?{mw}:{mb})"
    v = f"(abs(x)<{a})?{V0}:((abs(x)<{b})?0.0:{V0})"
    text = HEAD_1D + f"""\
Domain = {-L}, {L}
SplineDegree = 7
NQuadratures = 9
KnotSequence = interfaces
NSplines = 200
Mass = {m}
MassIsConstant = 0
Potential = {v}
PotentialDerivativeX = 0.0
TargetEigenvalue = 0.0
NBoundStates = 2
EnablePropagation = 0
"""
    _, h5, _ = run(tmp_path, "dqw", text)
    E = spectrum(h5)[:2] * HA_MEV
    widths = [L - b, b - a, 2 * a, b - a, L - b]
    exact = layered_neumann(widths, [mb, mw, mb, mw, mb], [V0, 0, V0, 0, V0], 2, 0.0025) * HA_MEV
    assert np.allclose(exact, [40.6074468017, 43.4686725099], atol=1e-8)
    assert np.max(np.abs(E - exact)) < 1e-8, (E, exact)          # meV


# ---------------------------------------------------------------------------
# Time propagation
# ---------------------------------------------------------------------------

DRIVEN_HO = HEAD_1D + """\
Domain = -20.0, 20.0
SplineDegree = 7
NQuadratures = 10
KnotSequence = uniform
NSplines = 200
Mass = 1.0
MassIsConstant = 1
Potential = 0.5*x*x
PotentialDerivativeX = x
TargetEigenvalue = 0.0
NBoundStates = 4
InitialState = ground
LaserX = 0.5*cos(0.6*t)
EnablePropagation = 1
FinalTime = 20.0
PhysicsOutput = dipole,energy
OutputStrideTS = 1
OutputStrideWFS = 1000000
"""


def x_exact(t, E0=0.5, w=0.6):
    """Driven oscillator from the ground state: <x>(t) = -E0/(1-w^2) (cos wt - cos t)."""
    return -E0 / (1.0 - w * w) * (np.cos(w * t) - np.cos(t))


def driven_error(tmp_path, prop, dt):
    _, _, td = run(tmp_path, f"ho_{prop}_{dt}", DRIVEN_HO + f"Propagator = {prop}\nTimeStep = {dt}\n")
    d = series(td, "dipoles", 4)
    e = series(td, "energies", 7)
    x = -d[:, 2]   # dipole column = q<x> with q = Charge = -1 (electron)
    return np.max(np.abs(x - x_exact(d[:, 0]))), np.max(np.abs(e[:, 6] - e[0, 6]))


@pytest.mark.parametrize("prop,dt,order,tol", [
    ("cn", 0.01, 2.0, 2e-3),
    ("magnus4", 0.1, 4.0, 2e-4),
])
def test_driven_oscillator_accuracy_and_order(tmp_path, prop, dt, order, tol):
    """Coherent-state solution of the driven oscillator: absolute accuracy,
    observed order of accuracy (step halving) and norm conservation."""
    err1, norm1 = driven_error(tmp_path, prop, dt)
    err2, norm2 = driven_error(tmp_path, prop, dt / 2)
    assert err1 < tol, err1
    observed = np.log2(err1 / err2)
    assert abs(observed - order) < 0.3, (prop, observed)
    assert max(norm1, norm2) < 1e-9, (norm1, norm2)


def test_hole_inverts_potential_and_coupling(tmp_path):
    """Charge = +1 (hole): H = K - (V + x E). With Potential = -x^2/2 the hole
    sees +x^2/2 (oscillator spectrum) and is pushed by +E: exact
    <x>(t) = +E0/(1-w^2)(cos wt - cos t); the dipole column is q<x> = +<x>
    and the acceleration column q d<v>/dt."""
    text = (DRIVEN_HO.replace("Charge            = -1.0", "Charge            = 1.0")
            .replace("Potential = 0.5*x*x", "Potential = -0.5*x*x")
            .replace("PotentialDerivativeX = x", "PotentialDerivativeX = -x")
            + "Propagator = magnus4\nTimeStep = 0.05\n")
    _, h5, td = run(tmp_path, "hole", text)
    E = spectrum(h5)
    assert np.allclose(E[:3], [0.5, 1.5, 2.5], atol=1e-8), E
    d = series(td, "dipoles", 4)
    t = d[:, 0]
    x = -x_exact(t)                      # hole: opposite displacement
    a = -x + 0.5 * np.cos(0.6 * t)       # x'' = -x + E
    assert np.max(np.abs(d[:, 2] - x)) < 1e-4
    assert np.max(np.abs(d[:, 3] - a)) < 1e-4


CROSS_2D = """\
Verbose = 0
Dimension = 2
Domain = -8.0, 8.0
SplineDegree = 5
NQuadratures = 6
KnotSequence = uniform
NSplines = 24
Charge = -1.0
Mass = 1.0
MassIsConstant = 1
UseDirectSolve = 1
EnableCAP = 0
Potential = 0.5*(x*x + y*y) + 0.2*x*y
PotentialDerivativeX = x + 0.2*y
PotentialDerivativeY = y + 0.2*x
TargetEigenvalue = 0.0
NBoundStates = 1
InitialState = ground
LaserX = 0.05*sin(0.5*t)
EnablePropagation = 1
Propagator = magnus4
TimeStep = 0.05
FinalTime = 10.0
PhysicsOutput = dipole,energy
OutputStrideTS = 1
OutputStrideWFS = 1000000
"""


def test_cross_polarization_is_recorded(tmp_path):
    """x-polarised drive of an anisotropic 2D oscillator (x-y coupling): the
    induced <y>(t) is recorded although y carries no field, equals a run that
    couples y with a zero field, and obeys Ehrenfest (A_y = D_y'')."""
    _, _, td = run(tmp_path, "cross_x", CROSS_2D + "Polarization = x\n")
    _, _, td2 = run(tmp_path, "cross_xy", CROSS_2D + "Polarization = xy\nLaserY = 0\n")
    d = series(td, "dipoles", 7)
    d2 = series(td2, "dipoles", 7)
    t, Dy, Ay = d[:, 0], d[:, 4], d[:, 6]
    assert np.max(np.abs(Dy)) > 1e-3                    # cross-polarized response present
    assert np.max(np.abs(d - d2)) < 1e-10
    h = t[1] - t[0]
    Dyy = (np.roll(Dy, -1) - 2 * Dy + np.roll(Dy, 1)) / h**2
    s = slice(3, -3)
    assert np.max(np.abs(Ay - Dyy)[s]) < 1e-3 * np.max(np.abs(Ay))


DRIVEN_STEP = HEAD_1D + """\
Domain = -1.0, 1.0
BoundaryType = Dirichlet
SplineDegree = 7
NQuadratures = 10
KnotSequence = interfaces
NSplines = 100
Mass = (x < 0.0) ? 1.0 : 4.0
MassIsConstant = 0
Potential = 0.0
PotentialDerivativeX = 0.0
TargetEigenvalue = 0.6
NBoundStates = 4
InitialState = sup: 0.7071067811865476*0 + 0.7071067811865476*1
LaserX = 0.012*sin(0.6615513246*t)
EnablePropagation = 1
TimeStep = 0.001
FinalTime = 2.0
AccelerationMethod = discrete
PhysicsOutput = dipole,energy,current
OutputStrideTS = 1
OutputStrideWFS = 1000000
"""


def test_driven_mass_step_ehrenfest_and_gauge(tmp_path):
    """Driven abrupt mass step: discrete Ehrenfest acceleration and velocity
    agree with the derivatives of <x>(t), the length-velocity relation holds
    for the eigenpairs, and the norm is conserved."""
    log, _, td = run(tmp_path, "mstep_driven", DRIVEN_STEP)
    d = series(td, "dipoles", 4)
    c = series(td, "currents", 11)
    e = series(td, "energies", 7)
    t, x, acc = d[:, 0], d[:, 2], d[:, 3]
    h = t[1] - t[0]
    s = slice(3, -3)
    xd = np.gradient(x, h)
    xdd = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / h ** 2
    assert np.max(np.abs(acc - xdd)[s]) / np.max(np.abs(xdd[s])) < 1e-4
    J = c[:, 3]
    assert np.max(np.abs(J - xd)[s]) / np.max(np.abs(J)) < 1e-4
    assert np.max(np.abs(J - c[:, 5] - c[:, 7] - c[:, 9])) < 1e-12   # decomposition closes
    assert log_value(log, r"Max length-vel residual\s*=\s*([0-9.eE+-]+)") < 1e-9
    assert np.max(np.abs(e[:, 6] - e[0, 6])) < 1e-11


# ---------------------------------------------------------------------------
# Assembly consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,pol", [(3, "x"), (3, "z"), (2, "xy")])
def test_sum_factorised_assembly_matches_pointwise(tmp_path, dim, pol):
    """Every element matrix of the sum-factorised assembly equals the
    per-point kernels to rounding (H, M and the time-dependent operators,
    including the batched y/z/multi-axis path)."""
    dom = ", ".join(["-6.0, 6.0" if dim == 2 else "-5.0, 5.0"] * dim)
    r2 = "+".join(["x*x", "y*y", "z*z"][:dim])
    fields = {"x": "LaserX = 0.05*sin(t)\n", "z": "LaserZ = 0.05*sin(t)\n",
              "xy": "LaserX = 0.05*sin(t)\nLaserY = 0.03*cos(t)\n"}[pol]
    text = f"""\
Verbose = 0
Dimension = {dim}
Domain = {dom}
SplineDegree = 4
NQuadratures = 7
KnotSequence = uniform
NSplines = {8 if dim == 2 else 5}
BoundaryType = Dirichlet
Mass = 1.0
MassIsConstant = 1
Potential = 0.5*({r2})
PotentialDerivativeX = x
PotentialDerivativeY = y
PotentialDerivativeZ = z
TargetEigenvalue = 0.0
UseDirectSolve = 1
NBoundStates = 1
Polarization = {pol}
{fields}TimeStep = 0.05
FinalTime = 0.1
EnablePropagation = 1
PhysicsOutput = dipole,energy
OutputStrideTS = 1
OutputStrideWFS = 1000000
"""
    log, h5, _ = run(tmp_path, f"sf_{dim}{pol}", text, args=("-tdsez_sumfact_check",))
    diffs = [float(v) for v in re.findall(r"max relative difference\s*=?\s*([0-9.eE+-]+)", log)]
    assert diffs, "sum-factorisation check did not run"
    assert max(diffs) < 1e-12, diffs
    # Deliberately coarse mesh (h = 1.5-2): only a sanity check of the energy.
    assert abs(spectrum(h5)[0] - 0.5 * dim) < 0.1 * dim
