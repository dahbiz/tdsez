"""TDSEZ validation harness.

Runs the tdsez binary (via mpirun) on small analytic inputs with
closed-form solutions and validates the computed results against
hard-coded reference values.

Test categories:
  1. Analytic correctness — harmonic oscillator spectra (1D, 2D, 3D)
  2. HDF5 provenance — run metadata, eigenvalue residuals, knot PoU
  3. Input validation — guard tests for invalid/missing parameters
  4. Initial state — ground state, excited state, superposition

Each individual check prints [PASS] or [FAIL] as it runs, giving
immediate per-step visibility during the test run.

Usage:
    pytest tests/test_tdsez.py -v -s
    make test    (from the build directory)
"""

import os
import csv
import subprocess
import tempfile

import h5py
import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

import zkit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(REPO, "tests", "inputs")
_TEMP_INPUTS = tempfile.TemporaryDirectory(prefix="tdsez-tests-")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(label, condition, detail=""):
    """Print [PASS] or [FAIL] for a single check and assert on failure.

    Provides per-step visibility during the test run: each individual
    assertion prints its result immediately rather than only showing
    a traceback at the end.

    Args:
        label: Short description of the check.
        condition: Boolean result of the check.
        detail: Extra context printed on failure.
    """
    if condition:
        print(f"    [PASS] {label}")
    else:
        print(f"    [FAIL] {label}  {detail}")
        assert condition, f"{label}: {detail}"


def _run(input_file, binary, nproc):
    """Run the tdsez binary via mpirun and return the result + output paths.

    Removes stale HDF5 outputs before running so each test starts fresh.
    Uses ``--bind-to none`` to avoid OpenMPI 5.x CPU-pinning slowdowns.

    Args:
        input_file: Path to the .inp input file.
        binary: Path to the tdsez executable.
        nproc: Number of MPI ranks.

    Returns:
        Tuple (result, out_h5_path, td_h5_path) where result is the
        subprocess.CompletedProcess instance.
    """
    base = os.path.basename(input_file)
    out_h5 = os.path.join(REPO, "static", f"EigenData_{base}.h5")
    td_h5 = os.path.join(REPO, "td", f"TimeEvolutionData_{base}.h5")
    for stale in (out_h5, td_h5):
        if os.path.exists(stale):
            os.remove(stale)
    cmd = ["mpirun", "--bind-to", "none", "-np", str(nproc), binary, "-inp", input_file]
    r = subprocess.run(cmd, capture_output=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return r, out_h5, td_h5


def _run_guard(input_file, binary):
    """Run an input expected to trigger a FATAL error.

    Uses -np 1 so the FATAL message is not lost to cross-rank MPI_ABORT
    output races that swallow the line on capture.

    Args:
        input_file: Path to the .inp input file.
        binary: Path to the tdsez executable.

    Returns:
        subprocess.CompletedProcess instance.
    """
    r, out_h5, td_h5 = _run(input_file, binary, 1)
    return r


def _write_tmp(name, text):
    """Write a generated input outside the source tree and return its path."""
    path = os.path.join(_TEMP_INPUTS.name, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# Hard-coded expected results
# ---------------------------------------------------------------------------
# All expected values are literal constants, not computed from formulas at
# runtime.  This makes the test a true independent check: a bug in the
# formula cannot mask a bug in the solver.
#
# TDSEZ units (hbar^2/2m = 1/2, hbar=1, m=1):
#   1D HO, w=0.2:  E_n = (n + 1/2) * w  ->  E0=0.1, E1=0.3, E2=0.5
#   2D HO, w=0.2:  E    = (nx+ny + 1) * w  ->  0.2, 0.4, 0.4, 0.6, 0.6, 0.6
#   3D HO, w=0.2:  E    = (nx+ny+nz + 1.5) * w  ->  0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7

# 1D HO spectrum (NBoundStates=3)
EXPECTED_HO1D = [0.1, 0.3, 0.5]

# 2D HO spectrum (NBoundStates=6): degeneracy pattern 1, 2, 3
EXPECTED_HO2D = [0.2, 0.4, 0.4, 0.6, 0.6, 0.6]

# 3D HO spectrum (NBoundStates=7): degeneracy pattern 1, 3, 3 (of 6)
EXPECTED_HO3D = [0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7]

# Tolerances
TOL_1D = 5e-3   # 1D: NSplines=80, very accurate
TOL_2D = 5e-3   # 2D: NSplines=32
TOL_3D = 1e-2   # 3D: NSplines=12, coarser grid

# Provenance keys that must be present in run_metadata
EXPECTED_META_KEYS = [
    "code_version", "input_file", "units",
    "Dimension", "SplineDegree",
    "LMinX", "LMaxX",
    "Hbar", "Charge",
]

# Expected residual bound for eigenvalue convergence
EXPECTED_RESID_TOL = 1e-6

# Expected partition-of-unity tolerance
EXPECTED_POU_TOL = 1e-9

# Expected superposition population split (normalized 0.7*0+0.7*1 -> 0.5/0.5)
EXPECTED_POP_SPLIT = 0.5
EXPECTED_POP_TOL = 1e-3


# 1-D harmonic oscillator input template, m=1, omega=0.2.
# TDSEZ units (hbar^2/2m = 1/2, hbar=1, m=1):
#   H = -(1/2) d^2/dx^2 + (1/2) m w^2 x^2
#   E_n = (n + 1/2) * w   ->  E0=0.1, E1=0.3, E2=0.5  (a.u.)
BOX1D = """\
Verbose           = 0
Dimension         = 1
Domain            = -20.0, 20.0
SplineDegree      = 5
NQuadratures      = 8
KnotSequence      = uniform
NSplines          = 80
Planck            = 1.0
Charge            = 1.0
Mass              = 1.0
MassIsConstant    = 1
Potential         = 0.5 * 1.0 * 0.2 * 0.2 * x * x
PotentialDerivativeX = 0.2 * 0.2 * x
UseDirectSolve    = 0
TargetEigenvalue  = 0.0
NBoundStates      = 3
NBoundStatesSave  = 0
EnablePropagation = 0
EnableCAP         = 0
Polarization      = x
"""


# ---------------------------------------------------------------------------
# Analytic correctness + provenance + knot PoU
# ---------------------------------------------------------------------------

def test_box1d_energies():
    """Verify the 1D harmonic oscillator spectrum matches the closed form."""
    print("\n  --- test_box1d_energies ---")
    binary = zkit.find_binary()
    path = _write_tmp("box1d_test.inp", BOX1D)
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    d = zkit.open_eig(out_h5)
    E = d["spectrum"][:3]
    for n, (En, ref) in enumerate(zip(E, EXPECTED_HO1D)):
        rel = abs(En - ref) / ref
        _step(f"E{n} = {En:.6f} (ref {ref})", rel < TOL_1D,
              f"rel err {rel:.2e} >= {TOL_1D}")
    print(f"  RESULT: E = {E}")


def test_variable_mass_spectrum_against_finite_difference():
    """Compare spatially varying mass with an independent flux-form discretization."""
    binary = zkit.find_binary()
    text = BOX1D.replace("Mass              = 1.0", "Mass              = 1.0 + 0.02*x*x")
    text = text.replace("MassIsConstant    = 1", "MassIsConstant    = 0")
    path = _write_tmp("variable_mass_reference.inp", text)
    r, out_h5, _ = _run(path, binary, 2)
    _step("variable-mass solve succeeds", r.returncode == 0,
          r.stdout.decode()[-800:] + r.stderr.decode()[-800:])
    actual = zkit.open_eig(out_h5)["spectrum"][:3]

    # Independent centered finite-volume approximation of
    # H = -1/2 d/dx[(1/m(x)) d/dx] + 0.02 x^2, with Dirichlet endpoints.
    h = 0.02
    x = np.arange(-20.0 + h, 20.0, h)
    fplus = 1.0 / (1.0 + 0.02 * (x + h / 2.0) ** 2)
    fminus = 1.0 / (1.0 + 0.02 * (x - h / 2.0) ** 2)
    diag = (fplus + fminus) / (2.0 * h * h) + 0.02 * x * x
    offdiag = -fplus[:-1] / (2.0 * h * h)
    reference = eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, 2),
                                 eigvals_only=True)
    _step("variable-mass low spectrum agrees with finite differences",
          np.allclose(actual, reference, rtol=0.01, atol=0.001),
          f"solver={actual}, reference={reference}")


def test_variable_mass_2d_mpi_consistency():
    """Exercise the multidimensional variable-mass assembly on two MPI layouts."""
    binary = zkit.find_binary()
    text = open(os.path.join(INP, "ho2d.inp")).read()
    text = text.replace("NSplines          = 32", "NSplines          = 12")
    text = text.replace("NBoundStates      = 6", "NBoundStates      = 3")
    text = text.replace("Mass              = 1.0", "Mass              = 1.0 + 0.005*(x*x + y*y)")
    text = text.replace("MassIsConstant    = 1", "MassIsConstant    = 0")
    energies = []
    for nproc in (1, 2):
        path = _write_tmp(f"variable_mass_2d_{nproc}.inp", text)
        r, out_h5, _ = _run(path, binary, nproc)
        _step(f"2D variable-mass {nproc}-rank solve succeeds", r.returncode == 0,
              r.stdout.decode()[-800:] + r.stderr.decode()[-800:])
        E = np.asarray(zkit.open_eig(out_h5)["spectrum"][:3])
        _step("2D variable-mass spectrum is finite", np.all(np.isfinite(E)), str(E))
        energies.append(E)
    _step("2D variable-mass spectrum agrees across MPI ranks",
          np.allclose(*energies, rtol=2e-3, atol=1e-6), str(energies))


def test_cap_absorbs_without_norm_growth():
    """Check a near-boundary state loses norm under the assembled CAP."""
    binary = zkit.find_binary()
    text = BOX1D.replace("Domain            = -20.0, 20.0", "Domain            = -6.0, 6.0")
    text += "\nEnablePropagation = 1\nEnableCAP = 1\nCAPKmin = 2.0\n"
    text += "LaserX = 0.0\nTimeStep = 0.1\nFinalTime = 1.0\nOutputStrideWFS = 1000\n"
    path = _write_tmp("cap_norm_reference.inp", text)
    r, _, td_h5 = _run(path, binary, 2)
    _step("CAP propagation succeeds", r.returncode == 0,
          r.stdout.decode()[-800:] + r.stderr.decode()[-800:])
    with h5py.File(td_h5) as f:
        norm = np.asarray(f["energies"])[:, 6]
    _step("CAP norms finite", np.all(np.isfinite(norm)), str(norm))
    _step("CAP norm decreases", norm[-1] < norm[0] - 1e-5, str(norm))
    _step("CAP has no norm growth", np.all(np.diff(norm) <= 1e-6), str(norm))


@pytest.mark.parametrize("mode", ["interface", "adaptive", "adaptive_wf"])
def test_pending_knot_modes_are_uniform_fallbacks(mode):
    """Lock down the current fallback so specialized placement cannot be claimed."""
    binary = zkit.find_binary()
    text = BOX1D.replace("KnotSequence      = uniform", f"KnotSequence      = {mode}")
    path = _write_tmp(f"knot_fallback_{mode}.inp", text)
    r, out_h5, _ = _run(path, binary, 1)
    _step(f"{mode} fallback run succeeds", r.returncode == 0,
          r.stdout.decode()[-800:] + r.stderr.decode()[-800:])
    with h5py.File(out_h5) as f:
        knots = np.asarray(f["knots_x"])
    interior = knots[(knots > -20.0) & (knots < 20.0)]
    expected = np.linspace(-20.0, 20.0, len(interior) + 2)[1:-1]
    _step(f"{mode} uses exactly uniform interior knots",
          np.allclose(interior, expected, atol=1e-12), str(knots))


def test_postprocessor_harmonic_oscillator_dipoles():
    """Check all three CPU postprocessors against the oscillator x selection rule."""
    binary = zkit.find_binary()
    stem = "postprocessor_ho_reference.inp"
    text = BOX1D + "\nNBoundStatesSave = 1\nSaveDipoleMatrix = 1\nSaveDipoleAxes = x\n"
    path = _write_tmp(stem, text)
    r, _, _ = _run(path, binary, 1)
    _step("postprocessor fixture solve succeeds", r.returncode == 0,
          r.stdout.decode()[-800:] + r.stderr.decode()[-800:])
    bindir = os.path.dirname(binary)
    for name, args in (
        ("tdmprocessor", []),
        ("tdmselect", ["-i", "0", "-Ethr", "0.2"]),
        ("drecprocessor", ["-i", "0", "-Ethr", "0.2"]),
    ):
        proc = subprocess.run([os.path.join(bindir, name), stem, *args],
                              cwd=REPO, capture_output=True)
        _step(f"{name} succeeds", proc.returncode == 0,
              proc.stdout.decode()[-500:] + proc.stderr.decode()[-500:])

    dipoles = np.load(os.path.join(REPO, "static", f"TDM_Dx_{stem}.npy"))
    _step("TDM is 3x3 Hermitian", dipoles.shape == (3, 3)
          and np.allclose(dipoles, dipoles.conj().T, atol=1e-8), str(dipoles))
    _step("oscillator ground-to-first dipole matches sqrt(2.5)",
          abs(abs(dipoles[0, 1]) - np.sqrt(2.5)) < 0.02, str(dipoles))
    _step("oscillator forbidden ground-to-second dipole vanishes",
          abs(dipoles[0, 2]) < 0.02, str(dipoles))
    selected = os.path.join(REPO, "static", f"dij_select_{stem}_i0.csv")
    with open(selected, newline="") as f:
        rows = list(csv.DictReader(f))
    first = next(row for row in rows if int(row["j"]) == 1)
    _step("tdmselect agrees with full TDM",
          abs(float(first["abs_d2"]) - abs(dipoles[0, 1]) ** 2) < 1e-5,
          str(first))
    resolved = os.path.join(REPO, "static", f"drec_resolved_{stem}.csv")
    with open(resolved, newline="") as f:
        rows = list(csv.DictReader(f))
    first = next(row for row in rows if int(row["j"]) == 1)
    _step("drec regular dipole agrees with full TDM",
          abs(complex(float(first["Re_dj"]), float(first["Im_dj"])) - dipoles[0, 1]) < 1e-5,
          str(first))
    _step("drec continuum estimate is finite",
          all(np.isfinite(float(row[c])) for row in rows
              for c in ("E_j", "Re_d", "Im_d", "abs_d2", "rho")),
          str(rows))


def test_provenance_group():
    """Verify every eigen-output carries run_metadata with required keys."""
    print("\n  --- test_provenance_group ---")
    binary = zkit.find_binary()
    path = _write_tmp("box1d_test.inp", BOX1D)
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0, f"rc={r.returncode}")
    d = zkit.open_eig(out_h5)
    _step("run_metadata group present", d["has_metadata"], "group missing")
    if d["has_metadata"]:
        md = d["meta"]
        for k in EXPECTED_META_KEYS:
            _step(f"provenance key '{k}' present", k in md, f"missing key '{k}'")
        _step("eig_residual dataset present", "eig_residual" in d,
              "eig_residual missing")
        if "eig_residual" in d:
            resid_ok = bool(np.all(d["eig_residual"] < EXPECTED_RESID_TOL))
            _step(f"eig_residual < {EXPECTED_RESID_TOL}", resid_ok,
                  f"max resid = {np.max(d['eig_residual']):.2e}")
        print(f"  RESULT: version={md.get('code_version','?')}, resid={d.get('eig_residual', [None])[:3]}")


def test_knot_partition_of_unity():
    """Verify the reconstructed knot vector gives sum_i B_i(x) = 1 on the interior."""
    print("\n  --- test_knot_partition_of_unity ---")
    binary = zkit.find_binary()
    path = _write_tmp("box1d_test.inp", BOX1D)
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0, f"rc={r.returncode}")
    d = zkit.open_eig(out_h5)
    p = int(d["meta"]["SplineDegree"])
    kv = zkit.reconstruct_knots(d["knots_x"], p)
    _step("knots_x present", len(d.get("knots_x", [])) > 0, "no knots_x dataset")
    # sample strictly inside the domain (avoid clamped ends)
    xs = np.linspace(d["meta"]["LMinX"] + 0.5, d["meta"]["LMaxX"] - 0.5, 40)
    pou = zkit.partition_of_unity(kv, p, xs)
    max_dev = float(np.max(np.abs(pou - 1.0)))
    _step(f"sum B_i(x) = 1 (max dev = {max_dev:.2e})", max_dev < EXPECTED_POU_TOL,
          f"max|1-PoU|={max_dev:.2e} >= {EXPECTED_POU_TOL}")


def test_muparser_scalar_reference():
    """Verify a Potential expression referencing an input scalar works.

    Scalar parameters (e.g. Amplitude) are auto-bound to every muParser
    instance, so they can be referenced by name inside any expression.
    This test builds a smooth HO via Amplitude to verify that binding.
    """
    print("\n  --- test_muparser_scalar_reference ---")
    binary = zkit.find_binary()
    text = """\
Verbose           = 0
Dimension         = 1
Domain            = -20.0, 20.0
SplineDegree      = 5
NQuadratures      = 8
KnotSequence      = uniform
NSplines          = 80
Planck            = 1.0
Charge            = 1.0
Mass              = 1.0
MassIsConstant    = 1
Amplitude         = 0.02
Potential         = Amplitude * x * x
PotentialDerivativeX = 2.0 * Amplitude * x
UseDirectSolve    = 0
TargetEigenvalue  = 0.0
NBoundStates      = 3
NBoundStatesSave  = 0
EnablePropagation = 0
EnableCAP         = 0
Polarization      = x
"""
    path = _write_tmp("box1d_scalar.inp", text)
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0 (no mu::ParserError)", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    d = zkit.open_eig(out_h5)
    E = d["spectrum"][:3]
    for n, (En, ref) in enumerate(zip(E, EXPECTED_HO1D)):
        rel = abs(En - ref) / ref
        _step(f"scalar-ref E{n} = {En:.6f} (ref {ref})", rel < TOL_1D,
              f"rel err {rel:.2e} >= {TOL_1D}")
    print(f"  RESULT: E = {E}")


# ---------------------------------------------------------------------------
# Input-validation guards
# ---------------------------------------------------------------------------

def _expect_fatal(text, label):
    """Run an input expected to FATAL and verify the error message.

    Checks that the binary exits with a non-zero code and prints a
    FATAL message via either the parser or validation path.

    Args:
        text: Input file content.
        label: Short label for the test output.
    """
    print(f"\n  --- guard: {label} ---")
    binary = zkit.find_binary()
    path = _write_tmp(f"bad_{label}.inp", text)
    r = _run_guard(path, binary)
    _step(f"'{label}' -> non-zero exit", r.returncode != 0,
          f"binary succeeded (rc=0) but expected FATAL")
    out = r.stdout.decode() + r.stderr.decode()
    has_fatal = "FATAL" in out
    _step(f"'{label}' -> FATAL in output", has_fatal,
          f"no FATAL message; rc={r.returncode}\n{out[-800:]}")
    has_validation = "validation failed" in out or "unknown input key" in out
    _step(f"'{label}' -> validation/parser path", has_validation,
          f"no 'validation failed' or 'unknown input key'\n{out[-800:]}")


def test_guard_unknown_key():
    """Verify an unknown input key is a FATAL error under StrictInput (default)."""
    text = BOX1D + "BogusKeyThatDoesNotExist = 1.0\n"
    _expect_fatal(text, "unknown_key")


def test_guard_inverted_domain():
    """Verify LMax <= LMin is rejected."""
    text = BOX1D.replace("Domain            = -20.0, 20.0", "Domain            = 20.0, -20.0")
    _expect_fatal(text, "inverted_domain")


def test_guard_nonpositive_mass():
    """Verify a non-positive mass is rejected."""
    text = BOX1D.replace("Mass              = 1.0", "Mass              = 0.0")
    _expect_fatal(text, "zero_mass")


def test_guard_y_dependent_mass():
    """Reject a 2D mass that becomes negative away from the x axis."""
    text = (BOX1D.replace("Dimension         = 1", "Dimension         = 2")
            .replace("Mass              = 1.0", "Mass              = 1.0 - 0.02*y*y"))
    _expect_fatal(text, "y_dependent_mass")


def test_guard_y_dependent_potential():
    """Reject a non-finite 2D potential away from the x axis."""
    text = (BOX1D.replace("Dimension         = 1", "Dimension         = 2")
            .replace("Potential         = 0.5 * 1.0 * 0.2 * 0.2 * x * x",
                     "Potential         = sqrt(1.0-y*y)"))
    _expect_fatal(text, "y_dependent_potential")


def test_guard_bad_degree():
    """Verify SplineDegree out of [1,14] is rejected."""
    text = BOX1D.replace("SplineDegree      = 5", "SplineDegree      = 20")
    _expect_fatal(text, "bad_degree")


def test_guard_adaptive_missing_derivative():
    """Verify 'adaptive' knot sequence without PotentialDerivativeX falls back
    to uniform knots (implementation pending) instead of crashing.

    The full TDSEZAdaptiveKnots implementation requires PotentialDerivativeX,
    but the stub falls back to uniform interior knots without evaluating it.
    This test locks that behavior: zero rc + no crash / no FATAL.
    """
    print("\n  --- guard: adaptive_missing_derivative ---")
    binary = zkit.find_binary()
    text = (BOX1D
            .replace("KnotSequence      = uniform", "KnotSequence      = adaptive")
            .replace("PotentialDerivativeX = 0.2 * 0.2 * x", ""))
    path = _write_tmp("bad_adaptive_nodv.inp", text)
    r = _run_guard(path, binary)
    out = r.stdout.decode() + r.stderr.decode()
    _step("zero exit (stub fallback)", r.returncode == 0, f"binary failed: rc={r.returncode}\n{out[-800:]}")
    _step("NO 'TDSEZ FATAL'", "TDSEZ FATAL" not in out, f"REGRESSION:\n{out[-800:]}")
    _step("NO std::terminate", "terminate called" not in out, f"REGRESSION:\n{out[-800:]}")
    _step("NO SIGABRT", "SIGABRT" not in out, f"REGRESSION:\n{out[-800:]}")
    _step("NO raw mu::ParserError", "mu::ParserError" not in out, f"REGRESSION:\n{out[-800:]}")


def test_guard_adaptive_wf_missing_derivative():
    """Verify 'adaptive_wf' knot sequence without PotentialDerivativeX falls
    back to uniform knots (implementation pending) instead of crashing.

    The full TDSEZAdaptiveWFKnots bootstrap requires PotentialDerivativeX, but
    the stub falls back to uniform interior knots without evaluating it.
    This test locks that behavior: zero rc + no crash / no FATAL.
    """
    print("\n  --- guard: adaptive_wf_missing_derivative ---")
    binary = zkit.find_binary()
    text = (BOX1D
            .replace("KnotSequence      = uniform", "KnotSequence      = adaptive_wf")
            .replace("PotentialDerivativeX = 0.2 * 0.2 * x", ""))
    path = _write_tmp("bad_adaptivewf_nodv.inp", text)
    r = _run_guard(path, binary)
    out = r.stdout.decode() + r.stderr.decode()
    _step("zero exit (stub fallback)", r.returncode == 0, f"binary failed: rc={r.returncode}\n{out[-800:]}")
    _step("NO 'TDSEZ FATAL'", "TDSEZ FATAL" not in out, f"REGRESSION:\n{out[-800:]}")
    _step("NO std::terminate", "terminate called" not in out, f"REGRESSION:\n{out[-800:]}")
    _step("NO SIGABRT", "SIGABRT" not in out, f"REGRESSION:\n{out[-800:]}")


def test_strictinput_off_tolerates_unknown():
    """Verify StrictInput=0 tolerates unknown keys (no FATAL)."""
    print("\n  --- test_strictinput_off_tolerates_unknown ---")
    binary = zkit.find_binary()
    text = BOX1D + "StrictInput       = 0\nBogusKey = 1.0\n"
    path = _write_tmp("loose.inp", text)
    r, out_h5, _td = _run(path, binary, 5)
    _step("StrictInput=0 tolerates unknown key (rc=0)", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")


# ---------------------------------------------------------------------------
# InitialState selection (ground vs excited bound state)
# ---------------------------------------------------------------------------

def test_initial_state_ground_default():
    """Verify the default InitialState starts from the ground state (index 0)."""
    print("\n  --- test_initial_state_ground_default ---")
    import h5py
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d.inp")
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    with h5py.File(out_h5, "r") as h:
        spec = np.array(h["spectrum"])[:, 0]  # column 0 = real part
    _step(f"E0 = {spec[0]:.6f} (ref {EXPECTED_HO1D[0]:.6f})",
          abs(spec[0] - EXPECTED_HO1D[0]) < 1e-6,
          f"got {spec[0]}, expected {EXPECTED_HO1D[0]}")


def test_initial_state_excited():
    """Verify InitialState=state:1 starts from the 1st excited state."""
    print("\n  --- test_initial_state_excited ---")
    import h5py
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d_state1.inp")
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    out = r.stdout.decode() + r.stderr.decode()
    _step("'Bound state #1' printed", "Bound state #1" in out, f"got:\n{out[-800:]}")
    with h5py.File(out_h5, "r") as h:
        spec = np.array(h["spectrum"])[:, 0]
    _step(f"E1 = {spec[1]:.6f} (ref {EXPECTED_HO1D[1]:.6f})",
          abs(spec[1] - EXPECTED_HO1D[1]) < 1e-6,
          f"got {spec[1]}, expected {EXPECTED_HO1D[1]}")


def test_guard_initial_state_out_of_range():
    """Verify InitialState=state:N with N >= NBoundStates is a FATAL guard."""
    print("\n  --- guard: initial_state_out_of_range ---")
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d_statebad.inp")
    r = _run_guard(path, binary)
    out = r.stdout.decode() + r.stderr.decode()
    _step("non-zero exit (FATAL)", r.returncode != 0, "binary succeeded")
    _step("'InitialState' in error", "InitialState" in out, f"\n{out[-800:]}")
    _step("'NBoundStates' in error", "NBoundStates" in out, f"\n{out[-800:]}")


def test_initial_state_superposition():
    """Verify InitialState=sup:0.7*0+0.7*1 produces a normalized 50/50 split."""
    print("\n  --- test_initial_state_superposition ---")
    import h5py
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d_sup2.inp")
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    out = r.stdout.decode() + r.stderr.decode()
    _step("'Superposition assembled/normalized' printed",
          "Superposition assembled" in out or "Superposition normalized" in out,
          f"got:\n{out[-800:]}")
    with h5py.File(out_h5, "r") as h:
        spec = np.array(h["spectrum"])[:, 0]
    _step(f"E0 = {spec[0]:.6f} (ref {EXPECTED_HO1D[0]:.6f})",
          abs(spec[0] - EXPECTED_HO1D[0]) < 1e-6,
          f"got {spec[0]}")
    _step(f"E1 = {spec[1]:.6f} (ref {EXPECTED_HO1D[1]:.6f})",
          abs(spec[1] - EXPECTED_HO1D[1]) < 1e-6,
          f"got {spec[1]}")


def test_initial_state_superposition_population():
    """Verify a propagating superposition with zero field keeps the 0.5/0.5 split."""
    print("\n  --- test_initial_state_superposition_population ---")
    import h5py
    binary = zkit.find_binary()
    text = (BOX1D
            + "NBoundStates      = 5\n"
            + "NBoundStatesSave  = 1\n"
            + "InitialState      = sup: 0.7*0 + 0.7*1\n"
            + "EnablePropagation = 1\n"
            + "TimeStep          = 0.1\n"
            + "FinalTime         = 0.2\n"
            + "LaserX            = 0.0*sin(0.1*t)\n")
    path = _write_tmp("ho1d_supprop.inp", text)
    r, _out, td_h5 = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    _step("TimeEvolutionData file created", os.path.exists(td_h5),
          f"no output at {td_h5}")
    if os.path.exists(td_h5):
        with h5py.File(td_h5, "r") as h:
            p = np.array(h["populations"])[0]  # t=0 row
        _step(f"pop[0] = {p[1]:.4f} (ref {EXPECTED_POP_SPLIT:.4f})",
              abs(p[1] - EXPECTED_POP_SPLIT) < EXPECTED_POP_TOL,
              f"got {p[1]}, expected {EXPECTED_POP_SPLIT}")
        _step(f"pop[1] = {p[2]:.4f} (ref {EXPECTED_POP_SPLIT:.4f})",
              abs(p[2] - EXPECTED_POP_SPLIT) < EXPECTED_POP_TOL,
              f"got {p[2]}, expected {EXPECTED_POP_SPLIT}")


def test_compressed_time_series_flush():
    """Verify repeated HDF5 flushes preserve readable compressed time series."""
    import h5py

    binary = zkit.find_binary()
    text = (BOX1D
            + "EnablePropagation = 1\n"
            + "TimeStep = 0.1\n"
            + "FinalTime = 0.2\n"
            + "LaserX = 0.0*sin(0.1*t)\n"
            + "OutputStrideTS = 1\n"
            + "HDF5Compress = 1\n")
    path = _write_tmp("ho1d_compressed.inp", text)
    result, _out, td_h5 = _run(path, binary, 5)
    _step("compressed run exit code == 0", result.returncode == 0,
          f"rc={result.returncode}\n{result.stdout.decode()[-800:]}{result.stderr.decode()[-800:]}")
    with h5py.File(td_h5, "r") as h:
        for name in ("dipoles", "populations", "energies", "currents", "autocorrelation"):
            data = h[name]
            _step(f"{name} is compressed", data.compression == "gzip")
            _step(f"{name} has multiple time rows", data.shape[0] >= 2)
        _step("time rows are ordered", np.all(np.diff(h["energies"][:, 0]) >= 0))


def test_guard_superposition_out_of_range():
    """Verify a superposition index >= NBoundStates is a FATAL guard."""
    print("\n  --- guard: superposition_out_of_range ---")
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d_supbad.inp")
    r = _run_guard(path, binary)
    out = r.stdout.decode() + r.stderr.decode()
    _step("non-zero exit (FATAL)", r.returncode != 0, "binary succeeded")
    _step("'InitialState' in error", "InitialState" in out, f"\n{out[-800:]}")
    _step("'NBoundStates' in error", "NBoundStates" in out, f"\n{out[-800:]}")


def test_guard_superposition_too_many_terms():
    """Verify a superposition with >3 terms is a FATAL guard."""
    print("\n  --- guard: superposition_too_many_terms ---")
    binary = zkit.find_binary()
    text = (BOX1D
            + "NBoundStates      = 5\n"
            + "InitialState      = sup: 1*0 + 1*1 + 1*2 + 1*3\n"
            + "EnablePropagation = 0\n")
    path = _write_tmp("ho1d_sup4.inp", text)
    r = _run_guard(path, binary)
    out = r.stdout.decode() + r.stderr.decode()
    _step("non-zero exit (FATAL)", r.returncode != 0, "binary succeeded")
    _step("'must have 1..3' in error", "must have 1..3" in out, f"\n{out[-800:]}")


# ---------------------------------------------------------------------------
# Multi-dimensional Harmonic Oscillator (1D / 2D / 3D)
# ---------------------------------------------------------------------------
# The isotropic HO has a closed-form spectrum and a known degeneracy
# pattern that exposes dimensional bugs (wrong Laplacian, missing axis,
# broken tensor product):
#
#   1D:  E_n = (n + 0.5) * w          -> 0.1, 0.3, 0.5          (no degeneracy)
#   2D:  E_n = (nx+ny + 1) * w        -> 0.2, 0.4, 0.4, 0.6, 0.6, 0.6
#   3D:  E_n = (nx+ny+nz + 1.5) * w   -> 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7
#
# with w = 0.2.  Both the energy values and the degeneracy count at each
# level are asserted, because a correct multi-D Laplacian must produce
# exact degeneracies (1, 2, 3 in 2D;  1, 3, 6 in 3D).

def test_ho1d_energies():
    """Verify the 1D HO non-degenerate ladder: 0.1, 0.3, 0.5."""
    print("\n  --- test_ho1d_energies ---")
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d.inp")
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    d = zkit.open_eig(out_h5)
    E = d["spectrum"][:3]
    for n, (En, ref) in enumerate(zip(E, EXPECTED_HO1D)):
        rel = abs(En - ref) / ref
        _step(f"E{n} = {En:.6f} (ref {ref:.6f})", rel < TOL_1D,
              f"rel err {rel:.2e} >= {TOL_1D}")
    print(f"  RESULT: E = {E}")


def test_ho2d_energies():
    """Verify the 2D isotropic HO spectrum with degeneracy pattern 1,2,3.

    Expected: E0=0.2, E1=E2=0.4, E3=E4=E5=0.6
    """
    print("\n  --- test_ho2d_energies ---")
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho2d.inp")
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    d = zkit.open_eig(out_h5)
    E = d["spectrum"][:6]
    for n, (En, ref) in enumerate(zip(E, EXPECTED_HO2D)):
        rel = abs(En - ref) / ref
        _step(f"E{n} = {En:.6f} (ref {ref:.6f})", rel < TOL_2D,
              f"rel err {rel:.2e} >= {TOL_2D}")
    # explicit degeneracy checks
    _step("E1 == E2 (degenerate pair)", abs(E[1] - E[2]) < 1e-6,
          f"E1={E[1]}, E2={E[2]}")
    _step("E3 == E4 == E5 (degenerate triplet)",
          abs(E[3] - E[4]) < 1e-6 and abs(E[4] - E[5]) < 1e-6,
          f"E3={E[3]}, E4={E[4]}, E5={E[5]}")
    print(f"  RESULT: E = {E}")


def test_ho3d_energies():
    """Verify the 3D isotropic HO spectrum with degeneracy pattern 1,3,6.

    Expected: E0=0.3, E1=E2=E3=0.5, E4=E5=E6=0.7
    """
    print("\n  --- test_ho3d_energies ---")
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho3d.inp")
    r, out_h5, _td = _run(path, binary, 5)
    _step("binary exit code == 0", r.returncode == 0,
          f"rc={r.returncode}\n{r.stdout.decode()[-800:]}{r.stderr.decode()[-800:]}")
    d = zkit.open_eig(out_h5)
    E = d["spectrum"][:7]
    for n, (En, ref) in enumerate(zip(E, EXPECTED_HO3D)):
        rel = abs(En - ref) / ref
        _step(f"E{n} = {En:.6f} (ref {ref:.6f})", rel < TOL_3D,
              f"rel err {rel:.2e} >= {TOL_3D}")
    # explicit degeneracy checks
    _step("E1 == E2 == E3 (degenerate triplet)",
          abs(E[1] - E[2]) < 1e-4 and abs(E[2] - E[3]) < 1e-4,
          f"E1={E[1]}, E2={E[2]}, E3={E[3]}")
    _step("E4 == E5 == E6 (degenerate triplet)",
          abs(E[4] - E[5]) < 1e-4 and abs(E[5] - E[6]) < 1e-4,
          f"E4={E[4]}, E5={E[5]}, E6={E[6]}")
    print(f"  RESULT: E = {E}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
