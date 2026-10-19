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

import re

import numpy as np
import pytest
import zkit
from scipy.integrate import cumulative_trapezoid
from zkit.validation import KH_CASES, Validator, kramers_henneberger

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
    E = lambda t: 0.8 * np.sin(np.pi * t / 40) ** 2 * np.sin(1.2 * t)
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


# ---------------------------------------------------------------------------
# The quiver itself against closed forms. A field that is a sum of sines,
# E_a = sum_j c_j sin(k_j t + phi_j), has the particular double integral
# a_p = -sum_j c_j sin(k_j t + phi_j) / k_j^2; TDSE-Z removes the mean velocity
# and the mean position over the window [t0, t1], which fixes the added line:
#   alpha = (q/m) (a_p + b t + c),  b = -(a_p(t1) - a_p(t0)) / P,
#   c = -(<a_p> + b (t0 + t1) / 2),  <a_p> = sum_j c_j (cos(k_j t1+phi_j) - cos(k_j t0+phi_j)) / (k_j^3 P).
# sin^2 pulse: E0 sin^2(pi t/T) sin(w t) = E0/2 sin(w t) - E0/4 [sin((w+W) t) + sin((w-W) t)], W = 2 pi / T.
# ---------------------------------------------------------------------------
import h5py


def _exact_quiver(t, terms, t0, t1, qm):
    P = t1 - t0
    ap = lambda tt: -sum(c * np.sin(k * tt + p) / k**2 for c, k, p in terms)
    mean_ap = sum(c * (np.cos(k * t1 + p) - np.cos(k * t0 + p)) / k**3 for c, k, p in terms) / P
    b = -(ap(t1) - ap(t0)) / P
    c = -(mean_ap + b * 0.5 * (t0 + t1))
    return qm * (ap(t) + b * t + c)


def _pulse(E0, w, T, phase=0.0):
    W = 2 * np.pi / T
    return [(E0 / 2, w, phase), (-E0 / 4, w + W, phase), (-E0 / 4, w - W, phase)]


QUIVER_CASES = {
    # name: (deck, {axis: terms}, q/m)
    "CW 1D, m = 2, phase": ("""Dimension = 1
Domain = -10, 10
NSplines = 24
Mass = 2.0
Polarization = x
LaserX = 0.3*sin(1.5*t + 0.4)
FinalTime = 20
""", {0: [(0.3, 1.5, 0.4)]}, -1.0 / 2.0),
    "circular 2D": ("""Dimension = 2
Domain = -10, 10, -10, 10
NSplines = 12
Polarization = xy
LaserX = 0.4*sin(1.3*t)
LaserY = 0.4*cos(1.3*t)
FinalTime = 30
""", {0: [(0.4, 1.3, 0.0)], 1: [(0.4, 1.3, np.pi / 2)]}, -1.0),
    "3D elliptical + second harmonic, hole": ("""Dimension = 3
Domain = -10, 10, -10, 10, -10, 10
NSplines = 8
Charge = 1.0
Polarization = xyz
Omegax = 1.1
LaserX = 0.5*sin(1.1*t)
LaserY = 0.2*cos(1.1*t)
LaserZ = 0.1*sin(2.2*t + 0.3)
FinalTime = 30
""", {0: [(0.5, 1.1, 0.0)], 1: [(0.2, 1.1, np.pi / 2)], 2: [(0.1, 2.2, 0.3)]}, 1.0),
    "CW high frequency, long run": ("""Dimension = 1
Domain = -10, 10
NSplines = 24
Polarization = x
LaserX = 2.0*sin(20.0*t)
FinalTime = 2000
""", {0: [(2.0, 20.0, 0.0)]}, -1.0),
    "sin^2 pulse 1D, automatic window": ("""Dimension = 1
Domain = -10, 10
NSplines = 24
Polarization = x
LaserX = 0.8*sin(pi*t/40)^2*sin(1.2*t)
FinalTime = 40
""", {0: _pulse(0.8, 1.2, 40)}, -1.0),
    "sin^2 pulse 1D, KHWindow not a cycle": ("""Dimension = 1
Domain = -10, 10
NSplines = 24
Polarization = x
LaserX = 0.8*sin(pi*t/40)^2*sin(1.2*t)
FinalTime = 40
KHWindow = 13.0, 21.7
""", {0: _pulse(0.8, 1.2, 40)}, -1.0),
}


@pytest.mark.parametrize("case", list(QUIVER_CASES))
def test_kh_quiver_against_closed_form(case):
    deck, terms, qm = QUIVER_CASES[case]
    head = "Verbose = 0\nMode = kh\nSplineDegree = 3\nNQuadratures = 4\nPotential = 0.5*(x*x+y*y+z*z)\nNBoundStates = 1\n"
    tag = "".join(ch for ch in case if ch.isalnum())[:24]
    path = _write_tmp(f"khq_{tag}.inp", head + deck)
    r, _, _ = _run(path, zkit.find_binary(), 1)
    _step(f"{case}: runs", r.returncode == 0, r.stdout.decode()[-1500:])
    with h5py.File(zkit.Run(REPO, path).path("eigen")) as f:
        g = f["run_metadata"]
        t, w, alpha = g["kh_times"][:], g["kh_weights"][:], g["kh_alpha"][:]
        t0, t1 = float(g.attrs["KH_window_start"]), float(g.attrs["KH_window_end"])
    _step(f"{case}: weights sum to 1", abs(w.sum() - 1) < 1e-14)
    _step(f"{case}: nodes inside the window", t.min() > t0 and t.max() < t1)
    for a in range(3):
        if a in terms:
            ex = _exact_quiver(t, terms[a], t0, t1, qm)
            err = np.max(np.abs(alpha[:, a] - ex)) / np.max(np.abs(ex))
            # rounding of the field itself: sin(k t) at large k t carries ~eps k t
            tol = max(1e-12, 20 * np.finfo(float).eps * max(k for _, k, _ in terms[a]) * t1)
            _step(f"{case}: alpha_{'xyz'[a]} = closed form ({err:.1e} <= {tol:.0e})", err < tol)
            _step(f"{case}: <alpha_{'xyz'[a]}> = 0", abs(w @ alpha[:, a]) < 1e-13 * np.max(np.abs(ex)))
        else:
            _step(f"{case}: alpha_{'xyz'[a]} = 0 (not polarised)", not np.any(alpha[:, a]))
    if "KHWindow" not in deck and "pulse" not in case:
        k = terms[0][0][1]                       # one full period of the fundamental
        _step(f"{case}: window = one period", abs((t1 - t0) - 2 * np.pi / k) < 1e-12 * (t1 - t0))


def test_nondipole_not_in_this_release():
    """Multipole = 2, 3 (nondipole coupling) stop with a clear message."""
    deck = """Verbose = 0
Dimension = 2
Domain = -10, 10, -10, 10
NSplines = 10
Potential = 0.5*(x^2 + y^2)
PotentialDerivativeX = x
PotentialDerivativeY = y
NBoundStates = 1
Polarization = x
LaserX = 0.1*sin(t)
FinalTime = 1
Multipole = 2
"""
    _expect_fatal(deck, "nondipole_soon")
