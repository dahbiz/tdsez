"""Real validation harness for TDSEZ.

Unlike the old tests/test_basic.py (which only inspected HDF5 *shape* and
silently SKIPPED unless a run output already existed), these tests:

  1. actually RUN the tdsez binary (via mpirun) on small analytic inputs,
  2. assert the computed spectrum matches a closed-form reference,
  3. assert the HDF5 provenance group + eigenvalue residuals are present,
  4. assert the knot vector reconstructs a partition of unity,
  5. assert the input-validation guards (items 3/4/5) actually fire.

Run with:  pytest tests/test_zkit.py -v
(or just `pytest` from the repo root).
"""

import os
import subprocess

import numpy as np
import pytest

import zkit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(REPO, "tests", "inputs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(input_file, binary, nproc):
    base = os.path.basename(input_file)
    out_h5 = os.path.join(REPO, "static", f"EigenData_{base}.h5")
    td_h5 = os.path.join(REPO, "td", f"TimeEvolutionData_{base}.h5")
    # remove stale outputs so a re-run is fresh (the binary writes per-input-stem
    # files; without this a previous run's td/ file would be picked up by tests
    # that glob the td/ directory).
    for stale in (out_h5, td_h5):
        if os.path.exists(stale):
            os.remove(stale)
    cmd = ["mpirun", "-np", str(nproc), binary, "-inp", input_file]
    r = subprocess.run(cmd, capture_output=True)
    return r, out_h5, td_h5


def _run_guard(input_file, binary):
    """Run an input expected to FATAL. Use -np 1 so the FATAL message is not
    lost to cross-rank MPI_ABORT output races that swallow the line on capture."""
    r, out_h5, td_h5 = _run(input_file, binary, 1)
    return r


def _write_tmp(name, text):
    path = os.path.join(INP, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


# A 1-D harmonic oscillator, m=1, omega=0.2 (smooth -> well-conditioned, exact
# closed form).  TDSEZ units (hbar^2/2m = 1/2, hbar=1, m=1):
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
# Item 1 — analytic correctness + provenance + knot PoU
# ---------------------------------------------------------------------------
def test_box1d_energies():
    """Computed spectrum must match the harmonic-oscillator closed form."""
    binary = zkit.find_binary()
    path = _write_tmp("box1d_test.inp", BOX1D)
    r, out_h5, _td = _run(path, binary, 4)
    assert r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-1500:]
    d = zkit.open_eig(out_h5)
    E = d["spectrum"][:3]
    omega = 0.2
    for n, En in enumerate(E, start=0):
        ref = (n + 0.5) * omega   # E_n = (n + 1/2) * w  (TDSEZ units)
        assert abs(En - ref) / ref < 5e-3, f"E{n}: got {En}, ref {ref}"
    print(f"  HO energies OK: {E}")


def test_provenance_group():
    """Every eigen-output must carry run_metadata (params echo, version, units)."""
    binary = zkit.find_binary()
    path = _write_tmp("box1d_test.inp", BOX1D)
    r, out_h5, _td = _run(path, binary, 4)
    assert r.returncode == 0
    d = zkit.open_eig(out_h5)
    assert d["has_metadata"], "run_metadata group missing from HDF5"
    md = d["meta"]
    for k in ("code_version", "input_file", "units", "Dimension",
              "SplineDegree", "LMinX", "LMaxX", "Hbar", "Charge"):
        assert k in md, f"provenance key '{k}' missing"
    assert "eig_residual" in d, "eig_residual dataset missing"
    assert np.all(d["eig_residual"] < 1e-6), "eigenvalue residual too large"
    print(f"  provenance OK: version={md['code_version']}, resid={d['eig_residual'][:3]}")


def test_knot_partition_of_unity():
    """Reconstructed knot vector must give sum_i B_i(x) = 1 on the interior."""
    binary = zkit.find_binary()
    path = _write_tmp("box1d_test.inp", BOX1D)
    r, out_h5, _td = _run(path, binary, 4)
    assert r.returncode == 0
    d = zkit.open_eig(out_h5)
    p = int(d["meta"]["SplineDegree"])
    kv = zkit.reconstruct_knots(d["knots_x"], p)
    # sample strictly inside the domain (avoid clamped ends)
    xs = np.linspace(d["meta"]["LMinX"] + 0.5, d["meta"]["LMaxX"] - 0.5, 40)
    pou = zkit.partition_of_unity(kv, p, xs)
    assert np.allclose(pou, 1.0, atol=1e-9), f"partition of unity failed: max|1-PoU|={np.max(np.abs(pou-1))}"
    print(f"  knot PoU OK: max deviation = {np.max(np.abs(pou-1)):.2e}")


def test_muparser_scalar_reference():
    """Item 3: a Potential expr that references an input scalar must NOT throw.

    Historically referencing an input scalar (V0, BarrierWidth, ...) raised
    mu::ParserError because parsers only bound x,y,z. Now every scalar param is
    auto-bound, so 'Amplitude' (a real input scalar) used inside Potential works.
    We build a smooth HO via Amplitude to stay well-conditioned (fast solve).
    """
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
    r, out_h5, _td = _run(path, binary, 4)
    assert r.returncode == 0, "referencing scalar 'Amplitude' in Potential raised an error:\n" + \
        r.stdout.decode()[-1500:] + r.stderr.decode()[-1500:]
    d = zkit.open_eig(out_h5)
    E = d["spectrum"][:3]
    for n, En in enumerate(E, start=0):
        ref = (n + 0.5) * 0.2
        assert abs(En - ref) / ref < 5e-3, f"scalar-ref E{n}: got {En}, ref {ref}"
    print(f"  muParser scalar reference OK (no mu::ParserError); E={E}")


# ---------------------------------------------------------------------------
# Items 4 & 5 — input-validation guards must actually fire
# ---------------------------------------------------------------------------
def _expect_fatal(text, label):
    binary = zkit.find_binary()
    path = _write_tmp(f"bad_{label}.inp", text)
    r = _run_guard(path, binary)
    assert r.returncode != 0, f"expected FATAL for '{label}' but binary succeeded"
    out = r.stdout.decode() + r.stderr.decode()
    # Two distinct fatal paths exist:
    #   - parser-level (item 4): "TDSEZParser FATAL: unknown input key ..."
    #   - validation-level (item 5): "TDSEZ FATAL: input validation failed: ..."
    assert ("FATAL" in out and
            ("validation failed" in out or "unknown input key" in out)), \
        f"'{label}': no validation/parser FATAL message; rc={r.returncode}\n{out[-1500:]}"
    print(f"  guard OK: '{label}' -> FATAL as expected")


def test_guard_unknown_key():
    """Item 4: an unknown key must be a FATAL error under StrictInput (default)."""
    text = BOX1D + "BogusKeyThatDoesNotExist = 1.0\n"
    _expect_fatal(text, "unknown_key")


def test_guard_inverted_domain():
    """Item 5: LMax <= LMin must be rejected."""
    text = BOX1D.replace("Domain            = -20.0, 20.0", "Domain            = 20.0, -20.0")
    _expect_fatal(text, "inverted_domain")


def test_guard_nonpositive_mass():
    """Item 5: a non-positive mass must be rejected."""
    text = BOX1D.replace("Mass              = 1.0", "Mass              = 0.0")
    _expect_fatal(text, "zero_mass")


def test_guard_bad_degree():
    """Item 5: SplineDegree out of [1,14] must be rejected."""
    text = BOX1D.replace("SplineDegree      = 5", "SplineDegree      = 20")
    _expect_fatal(text, "bad_degree")


def test_guard_adaptive_missing_derivative():
    """Regression: 'adaptive'/'adaptive_wf' with NO PotentialDerivativeX must
    produce a CLEAN 'TDSEZ FATAL' (input-validation) message — NOT a hard
    mu::ParserError -> std::terminate -> SIGABRT + MPI abort.

    Historically an empty PotentialDerivativeX made TDSEZAdaptiveKnots call
    dVPotX.Eval() on an empty muParser expression, throwing mu::ParserError
    that escaped PetscCallAbort and crashed the whole MPI job. Now
    ValidateOrThrow() rejects it up front with an actionable message, and the
    in-code backstop converts any residual muParserError to a clean FATAL.
    This test locks that behavior: non-zero rc + clean FATAL + NO terminate.
    """
    binary = zkit.find_binary()
    text = (BOX1D
            .replace("KnotSequence      = uniform", "KnotSequence      = adaptive")
            .replace("PotentialDerivativeX = 0.2 * 0.2 * x", ""))
    path = _write_tmp("bad_adaptive_nodv.inp", text)
    r = _run_guard(path, binary)
    out = r.stdout.decode() + r.stderr.decode()
    # 1) must FAIL (FATAL), not succeed
    assert r.returncode != 0, "expected FATAL for 'adaptive' without PotentialDerivativeX"
    # 2) must be the CLEAN validation FATAL, naming the missing parameter
    assert "TDSEZ FATAL" in out, f"no clean FATAL; rc={r.returncode}\n{out[-1500:]}"
    assert "input validation failed" in out, f"FATAL was not the validation path:\n{out[-1500:]}"
    assert "PotentialDerivativeX" in out, f"FATAL did not name the missing derivative:\n{out[-1500:]}"
    # 3) must NOT be the old hard crash (this is the actual regression we fixed)
    assert "terminate called" not in out, \
        "REGRESSION: adaptive without dV still crashes via std::terminate!\n" + out[-1500:]
    assert "SIGABRT" not in out, \
        "REGRESSION: adaptive without dV still aborts with SIGABRT!\n" + out[-1500:]
    assert "mu::ParserError" not in out, \
        "REGRESSION: adaptive without dV still leaks a raw mu::ParserError!\n" + out[-1500:]
    print("  adaptive missing-derivative guard OK (clean FATAL, no SIGABRT)")


def test_guard_adaptive_wf_missing_derivative():
    """Same regression guard as above, for the 'adaptive_wf' sequence. Its
    coarse bootstrap mesh is built with TDSEZAdaptiveKnots, so it also needs
    PotentialDerivativeX; an empty one must produce a clean FATAL, not a crash.
    """
    binary = zkit.find_binary()
    text = (BOX1D
            .replace("KnotSequence      = uniform", "KnotSequence      = adaptive_wf")
            .replace("PotentialDerivativeX = 0.2 * 0.2 * x", ""))
    path = _write_tmp("bad_adaptivewf_nodv.inp", text)
    r = _run_guard(path, binary)
    out = r.stdout.decode() + r.stderr.decode()
    assert r.returncode != 0, "expected FATAL for 'adaptive_wf' without PotentialDerivativeX"
    assert "TDSEZ FATAL" in out and "PotentialDerivativeX" in out, \
        f"adaptive_wf missing-dV did not produce a clean FATAL:\n{out[-1500:]}"
    assert "terminate called" not in out, \
        "REGRESSION: adaptive_wf without dV still crashes via std::terminate!\n" + out[-1500:]
    assert "SIGABRT" not in out, \
        "REGRESSION: adaptive_wf without dV still aborts with SIGABRT!\n" + out[-1500:]
    print("  adaptive_wf missing-derivative guard OK (clean FATAL, no SIGABRT)")


def test_strictinput_off_tolerates_unknown():
    """Item 4: with StrictInput=0 an unknown key is tolerated (no FATAL)."""
    binary = zkit.find_binary()
    text = BOX1D + "StrictInput       = 0\nBogusKey = 1.0\n"
    path = _write_tmp("loose.inp", text)
    r, out_h5, _td = _run(path, binary, 4)
    assert r.returncode == 0, "StrictInput=0 should tolerate unknown keys:\n" + \
        r.stdout.decode()[-1500:] + r.stderr.decode()[-1500:]
    print("  StrictInput=0 tolerance OK")


# ---------------------------------------------------------------------------
# InitialState selection (ground vs excited bound state)
# ---------------------------------------------------------------------------
def test_initial_state_ground_default():
    """Default (no InitialState) starts from the ground state (index 0)."""
    import h5py
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d.inp")
    r, out_h5, _td = _run(path, binary, 4)
    assert r.returncode == 0, "ho1d ground-state run failed:\n" + \
        r.stdout.decode()[-1500:] + r.stderr.decode()[-1500:]
    with h5py.File(out_h5, "r") as h:
        spec = np.array(h["spectrum"])[:, 0]  # column 0 = real part
    # 1D HO, w=0.2 -> E0=0.1, E1=0.3, ...; ground state is the 0th.
    assert abs(spec[0] - 0.1) < 1e-6, f"expected E0=0.1, got {spec[0]}"
    print("  InitialState default (ground) OK: E0=%.4f" % spec[0])


def test_initial_state_excited():
    """InitialState=state:1 starts propagation from the 1st excited state."""
    import h5py
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d_state1.inp")
    r, out_h5, _td = _run(path, binary, 4)
    assert r.returncode == 0, "ho1d state:1 run failed:\n" + \
        r.stdout.decode()[-1500:] + r.stderr.decode()[-1500:]
    out = r.stdout.decode() + r.stderr.decode()
    assert "Bound state #1" in out, f"expected 'Bound state #1' print; got:\n{out[-1500:]}"
    with h5py.File(out_h5, "r") as h:
        spec = np.array(h["spectrum"])[:, 0]  # column 0 = real part
    # NBoundStates=5 -> spectrum holds E0..E4 = 0.1,0.3,0.5,0.7,0.9.
    assert abs(spec[1] - 0.3) < 1e-6, f"expected E1=0.3 available, got {spec[1]}"
    print("  InitialState=state:1 OK: selected excited state E1=%.4f" % spec[1])


def test_guard_initial_state_out_of_range():
    """InitialState=state:N with N >= NBoundStates must be a FATAL guard."""
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d_statebad.inp")
    r = _run_guard(path, binary)
    assert r.returncode != 0, "expected FATAL for out-of-range InitialState"
    out = r.stdout.decode() + r.stderr.decode()
    assert "InitialState" in out and "NBoundStates" in out, \
        f"expected clear InitialState/NBoundStates guard message; got:\n{out[-1500:]}"
    print("  InitialState out-of-range guard OK")


def test_initial_state_superposition():
    """InitialState=sup:0.7*0+0.7*1 -> normalized 50/50 split over states 0,1."""
    import h5py
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d_sup2.inp")
    r, out_h5, _td = _run(path, binary, 4)
    assert r.returncode == 0, "ho1d sup run failed:\n" + \
        r.stdout.decode()[-1500:] + r.stderr.decode()[-1500:]
    out = r.stdout.decode() + r.stderr.decode()
    assert ("Superposition assembled" in out or "Superposition normalized" in out), \
        f"expected superposition print; got:\n{out[-1500:]}"
    with h5py.File(out_h5, "r") as h:
        spec = np.array(h["spectrum"])[:, 0]
    # sanity: both states exist in the basis
    assert abs(spec[0] - 0.1) < 1e-6 and abs(spec[1] - 0.3) < 1e-6


def test_initial_state_superposition_population():
    """A propagating sup with zero field keeps the 0.5/0.5 population split."""
    import h5py, glob
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
    r, _out, td_h5 = _run(path, binary, 4)
    assert r.returncode == 0, "superposition propagation failed:\n" + \
        r.stdout.decode()[-1500:] + r.stderr.decode()[-1500:]
    # Read the file written for THIS input stem (the binary names it
    # TimeEvolutionData_<stem>.h5). Globbing td/ would pick up stale runs.
    assert os.path.exists(td_h5), f"no TimeEvolutionData output produced at {td_h5}"
    with h5py.File(td_h5, "r") as h:
        p = np.array(h["populations"])[0]  # t=0 row
    # normalized 0.7/0.7 -> 1/sqrt(2) each -> |c|^2 = 0.5 on states 0 and 1
    assert abs(p[1] - 0.5) < 1e-3 and abs(p[2] - 0.5) < 1e-3, \
        f"expected 0.5/0.5 split, got {p[:4]}"
    print("  Superposition population split OK: %.3f / %.3f" % (p[1], p[2]))


def test_guard_superposition_out_of_range():
    """sup: index >= NBoundStates must be a FATAL guard."""
    binary = zkit.find_binary()
    path = os.path.join(INP, "ho1d_supbad.inp")
    r = _run_guard(path, binary)
    assert r.returncode != 0, "expected FATAL for out-of-range superposition index"
    out = r.stdout.decode() + r.stderr.decode()
    assert "InitialState" in out and "NBoundStates" in out
    print("  Superposition out-of-range guard OK")


def test_guard_superposition_too_many_terms():
    """sup: with >3 terms must be a FATAL guard."""
    binary = zkit.find_binary()
    text = (BOX1D
            + "NBoundStates      = 5\n"
            + "InitialState      = sup: 1*0 + 1*1 + 1*2 + 1*3\n"
            + "EnablePropagation = 0\n")
    path = _write_tmp("ho1d_sup4.inp", text)
    r = _run_guard(path, binary)
    assert r.returncode != 0, "expected FATAL for >3 superposition terms"
    out = r.stdout.decode() + r.stderr.decode()
    assert "must have 1..3" in out, f"expected term-count guard; got:\n{out[-1500:]}"
    print("  Superposition >3 terms guard OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
