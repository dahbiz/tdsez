"""Mode = kh: eigenstates of the static Kramers-Henneberger potential, 1D/2D/3D.

V_KH(r) = < V(r + alpha(t)) > over one optical cycle of the free-carrier quiver
m alpha'' = q E(t), integrated by TDSE-Z from the laser expressions.

* Oscillator V = m w0^2 r^2 / 2: V_KH = V + m w0^2 <|alpha|^2> / 2, a constant,
  so every level of Mode = kh equals the level of Mode = td on the same grid
  plus the exact shift (the discretisation cancels):
    linear field E0 sin(w t):   alpha0 = |q| E0 / (m w^2),  shift = m w0^2 alpha0^2 / 4
    circular field (2D):        |alpha| = alpha0 constant,  shift = m w0^2 alpha0^2 / 2
* Gaussian well and a sin^2 pulse (period, peak window and quiver found
  automatically) against an independent reference: the quiver integrated with
  scipy and V_KH diagonalised by 4th-order finite differences.
"""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

import zkit
from zkit.validation import KH_CASES, Validator, kramers_henneberger
import re

from tests.test_tdsez import REPO, _expect_fatal, _run, _step, _write_tmp


def _levels(name, text):
    path = _write_tmp(name, text)
    r, _, _ = _run(path, zkit.find_binary(), 2)
    _step(f"{name} runs", r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-500:])
    run = zkit.Run(REPO, path)
    return np.sort(run.spectrum.values), r.stdout.decode()


@pytest.mark.parametrize("case", list(KH_CASES))
def test_kh_oscillator_against_exact_levels(case, tmp_path):
    """Shift vs Mode = td on the same grid, and absolute levels vs the closed form
    (cases and tolerances shared with zkit validate)."""
    v = Validator(zkit.find_binary(), nproc=2, workdir=tmp_path)
    kramers_henneberger(v, case)
    for c in v.report.checks:
        _step(f"{c.case}: {c.quantity}: {c.error:.1e} <= {c.tol:g}", c.ok)


GAUSS = """Verbose = 0
Mode = kh
Dimension = 1
Domain = -25, 25
SplineDegree = 7
NQuadratures = 10
NSplines = 300
Potential = -1.0*exp(-x*x/2)
NBoundStates = 2
TargetEigenvalue = -0.5
Polarization = x
LaserX = 0.8*sin(pi*t/40)^2*sin(1.2*t)
FinalTime = 40
"""


def _reference_gauss(n_levels=2):
    E = lambda t: 0.8 * np.sin(np.pi * t / 40) ** 2 * np.sin(1.2 * t)  # noqa: E731
    tg = np.linspace(0, 40, 40001)                     # peak of |E| on the solver's grid
    tpk = tg[np.argmax(np.abs(E(tg)))]
    P = 2 * np.pi / 1.2
    t = np.linspace(tpk - P / 2, tpk + P / 2, 400001)
    v = cumulative_trapezoid(E(t), t, initial=0)       # q/m = -1
    v -= np.trapezoid(v, t) / P
    a = cumulative_trapezoid(v, t, initial=0)
    a = -(a - np.trapezoid(a, t) / P)
    g, wg = np.polynomial.legendre.leggauss(64)          # cycle average on 64 Gauss nodes
    ts = tpk - P / 2 + 0.5 * P * (g + 1)
    alpha, wts = np.interp(ts, t, a), 0.5 * wg
    x = np.linspace(-25, 25, 5001)
    h = x[1] - x[0]
    vkh = sum(w * -np.exp(-((x + al) ** 2) / 2) for al, w in zip(alpha, wts))
    n = len(x) - 2
    main = 0.5 * (5 / 2) / h**2 + vkh[1:-1]            # -1/2 d2/dx2, 4th order, Dirichlet
    H = np.diag(main) + np.diag(np.full(n - 1, -0.5 * (4 / 3) / h**2), 1) + np.diag(np.full(n - 2, 0.5 * (1 / 12) / h**2), 2)
    H = np.triu(H) + np.triu(H, 1).T
    return np.sort(np.linalg.eigvalsh(H))[:n_levels], np.max(np.abs(a))


def test_kh_gaussian_well_pulse_against_independent_reference():
    got, out = _levels("kh_gauss.inp", GAUSS)
    ref, amp = _reference_gauss()
    m = re.search(r"Quiver x\s*=\s*amplitude ([0-9.eE+-]+)", " ".join(out.split()))
    # the reported amplitude is the largest |alpha| at the averaging nodes
    _step(f"quiver amplitude reported ({m.group(1) if m else None}) = max |alpha(t)| {amp:.6f}",
          m is not None and abs(float(m.group(1)) - amp) < 2e-3 * amp)
    bound = ref[ref < 0]
    err = np.max(np.abs(got[: len(bound)] - bound))
    _step(f"bound KH levels {got[:len(bound)]} = reference {bound} ({err:.1e})", err < 1e-6)
    # the dressed well is shallower than the bare one
    _step("KH ground state above the bare ground state", got[0] > -0.6)


@pytest.mark.parametrize(
    "extra,label",
    [
        ("Mode = kh\nPolarization = none\n", "kh_no_laser"),
        ("Mode = kh\nMass = 1 + 0.1*x*x\n", "kh_varying_mass"),
        ("Mode = kh\nFieldTerm = x ; sin(t)\n", "kh_field_term"),
        ("Mode = kh\nKHWindow = 5, 2\n", "kh_bad_window"),
        ("Mode = static\n", "bad_mode"),
        ("NQuadratures = 12\n", "quadrature_above_10"),
    ],
)
def test_kh_guards(extra, label):
    deck = """Verbose = 0
Dimension = 1
Domain = -20, 20
NSplines = 60
Potential = 0.5*x*x
NBoundStates = 2
Polarization = x
LaserX = 0.1*sin(t)
FinalTime = 20
"""
    _expect_fatal(deck + extra, label)
