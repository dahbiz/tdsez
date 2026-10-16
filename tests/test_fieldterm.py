"""Structured (separable) field terms: FieldTerm = w(x,y,z) ; f(t).

Each term adds s f(t) w(r) to H (s = -Charge), so FieldTerm = x ; E(t) is the
uniform field along x: it must reproduce Polarization = x, LaserX = E(t) to
rounding with every propagator, matrix-free and on several ranks.

A genuinely structured field is checked against an exact solution: for the
1D oscillator with FieldTerm = x^2 ; f(t) the frequency is time dependent,
Omega^2(t) = w^2 + 2 s f(t), and the moments X = <x^2>, P = <p^2>,
C = <xp + px> obey the closed linear system X' = C, C' = 2P - 2 Omega^2 X,
P' = -Omega^2 C. TDSE-Z's potential and kinetic energies are w^2 X / 2 and P / 2.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import zkit
from tests.test_tdsez import REPO, _expect_fatal, _run, _step, _write_tmp

PULSE = "t<12 ? 0.08*sin(pi*t/12)^2*sin(1.1*t) : 0"

ATOM2D = """\
Verbose           = 0
Dimension         = 2
Domain            = -24.0, 24.0, -24.0, 24.0
SplineDegree      = 5
NQuadratures      = 8
KnotSequence      = uniform
NSplines          = 40
Charge            = -1.0
Mass              = 1.0
MassIsConstant    = 1
Potential         = -1/sqrt(x*x + y*y + 0.5)
PotentialDerivativeX = x/(x*x + y*y + 0.5)^1.5
PotentialDerivativeY = y/(x*x + y*y + 0.5)^1.5
TargetEigenvalue  = -1.5
NBoundStates      = 3
InitialState      = ground
EnablePropagation = 1
EnableCAP         = 1
CAPKmin           = 1.0
AccelerationMethod = discrete
TimeStep          = 0.05
FinalTime         = 14.0
OutputStrideWFS   = 100000
"""

UNIFORM = f"Polarization = x\nLaserX = {PULSE}\n"
TERM = f"FieldTerm = x ; {PULSE}\n"


def _evolution(name, text, nproc=1, flags=()):
    path = _write_tmp(name, text)
    r, _, _ = _run(path, zkit.find_binary(), nproc) if not flags else _run_flags(path, nproc, flags)
    _step(f"{name} runs", r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-800:])
    return zkit.Run(REPO, path).evolution


def _run_flags(path, nproc, flags):
    import os
    import subprocess

    cmd = ["mpirun", "--bind-to", "none", "-np", str(nproc), zkit.find_binary(), "-inp", path, *flags]
    return subprocess.run(cmd, capture_output=True, env={**os.environ}), None, None


def _rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


@pytest.mark.parametrize("prop", ["cn", "pade22", "magnus4"])
def test_uniform_field_term_reproduces_polarization(prop):
    """FieldTerm = x ; E(t) is the dipole coupling: same dipole, acceleration, energies and populations."""
    deck = ATOM2D + f"Propagator = {prop}\n"
    ref = _evolution(f"ft_ref_{prop}.inp", deck + UNIFORM)
    new = _evolution(f"ft_term_{prop}.inp", deck + TERM)
    dim = 2
    resp = slice(1 + dim, 1 + 3 * dim)   # dipole and acceleration columns
    for label, a, b, tol in (
        ("dipole, acceleration", new.dipoles[:, resp], ref.dipoles[:, resp], 1e-9),
        ("energies (kinetic, potential, interaction, total)", new.energies[:, 1:5], ref.energies[:, 1:5], 1e-9),
        ("populations", new.populations[:, 1:], ref.populations[:, 1:], 1e-9),
    ):
        err = _rel(a, b)
        _step(f"{prop}: {label} relative difference {err:.1e} < {tol:g}", err < tol)
    _step(f"{prop}: the field drives the atom", np.abs(ref.dipoles[:, 1 + dim]).max() > 1e-3)


def test_uniform_field_term_matrix_free_and_mpi():
    """The term list in the matrix-free stage update and on 2 ranks."""
    deck = ATOM2D + "Propagator = pade22\n"
    ref = _evolution("ft_ref_mf.inp", deck + UNIFORM)
    mf = _evolution("ft_term_mf.inp", deck + TERM, flags=("-tdsez_mat_free",))
    mpi = _evolution("ft_term_mpi.inp", deck + TERM, nproc=2)
    resp = slice(3, 7)
    for label, run in (("matrix-free", mf), ("2 ranks", mpi)):
        err = _rel(run.dipoles[:, resp], ref.dipoles[:, resp])
        _step(f"{label}: dipole/acceleration difference {err:.1e} < 1e-9", err < 1e-9)


HO1D = """\
Verbose           = 0
Dimension         = 1
Domain            = -16.0, 16.0
SplineDegree      = 6
NQuadratures      = 10
KnotSequence      = uniform
NSplines          = 160
Charge            = -1.0
Mass              = 1.0
MassIsConstant    = 1
Potential         = 0.5*0.25*x*x
PotentialDerivativeX = 0.25*x
TargetEigenvalue  = 0.0
NBoundStates      = 1
InitialState      = ground
EnablePropagation = 1
EnableCAP         = 0
Propagator        = magnus4
TimeStep          = 0.02
FinalTime         = 30.0
OutputStrideWFS   = 100000
"""
W2 = 0.25
F_EXPR = "t<25 ? 0.04*sin(pi*t/25)^2*sin(1.3*t) : 0"


def _f(t):
    return np.where(t < 25, 0.04 * np.sin(np.pi * t / 25) ** 2 * np.sin(1.3 * t), 0.0)


def test_quadrupole_like_term_against_exact_moments():
    """FieldTerm = x^2 ; f(t): the parametrically driven oscillator, exact second moments."""
    ev = _evolution("ft_ho_x2.inp", HO1D + f"FieldTerm = x^2 ; {F_EXPR}\n")
    t = ev.time
    s = 1.0   # -Charge

    def rhs(tt, y):
        om2 = W2 + 2 * s * _f(tt)
        X, C, P = y
        return [C, 2 * P - 2 * om2 * X, -om2 * C]

    w = np.sqrt(W2)
    sol = solve_ivp(rhs, (0, t[-1]), [1 / (2 * w), 0.0, w / 2], t_eval=t, rtol=1e-12, atol=1e-14, max_step=0.05)
    X, P = sol.y[0], sol.y[2]
    norm = ev.energies[:, 6]
    pot = ev.energies[:, 2] / norm
    kin = ev.energies[:, 1] / norm
    inter = ev.energies[:, 3] / norm
    _step("the drive changes <x^2> by more than 5 %", np.ptp(X) / X[0] > 0.05)
    for label, a, b in (
        ("potential energy w^2 <x^2>/2", pot, 0.5 * W2 * X),
        ("kinetic energy <p^2>/2", kin, 0.5 * P),
        ("interaction energy s f(t) <x^2>", inter, s * _f(t) * X),
    ):
        err = np.abs(a - b).max() / np.abs(b).max() if np.abs(b).max() > 0 else np.abs(a).max()
        _step(f"{label}: max relative error {err:.1e} < 1e-6", err < 1e-6)


def test_two_terms_add():
    """Two FieldTerm lines add: x ; f and x ; f equals x ; 2 f (raw file, the key repeats)."""
    from tests.test_tdsez import _write_tmp as write

    deck = ATOM2D + "Propagator = pade22\n"
    path = write("ft_two.inp", deck + TERM + TERM, raw=True)
    r, _, _ = _run(path, zkit.find_binary(), 1)
    _step("two FieldTerm lines run", r.returncode == 0, r.stdout.decode()[-1500:])
    two = zkit.Run(REPO, path).evolution
    one = _evolution("ft_double.inp", deck + f"FieldTerm = 2*x ; {PULSE}\n")
    err = _rel(two.dipoles[:, 3:7], one.dipoles[:, 3:7])
    _step(f"x;f + x;f = 2x;f, difference {err:.1e} < 1e-10", err < 1e-10)


@pytest.mark.parametrize(
    "line,label",
    [
        ("FieldTerm = x*t ; 1", "profile_uses_t"),
        ("FieldTerm = x ; x*t", "time_uses_x"),
        ("FieldTerm = x", "no_semicolon"),
        ("FieldTerm = x ; 1 ; 2", "two_semicolons"),
        ("FieldTerm = z ; 1", "profile_axis_missing_in_2d"),
        ("FieldTerm = x ; 1\nAccelerationMethod = operator", "operator_acceleration"),
    ],
)
def test_field_term_guards(line, label):
    _expect_fatal(ATOM2D + line + "\n", f"fieldterm_{label}")


def test_field_term_must_vanish_outside_tsurff_surface():
    deck = ATOM2D + TERM + "TSurff = 1\nSurffRadius = 10\nSurffKmax = 2, 2\nSurffNk = 11, 11\n"
    _expect_fatal(deck, "fieldterm_tsurff")
