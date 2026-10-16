"""PhysicsOutput: computing only the requested quantities keeps the HDF5 layout.

Every dataset of td/TimeEvolutionData_<input>.h5 has the same shape whatever
is selected; a quantity that is not computed keeps its time column and is
zero elsewhere. The quantities that are computed equal those of a full run.
"""

import h5py
import numpy as np
import pytest

import zkit
from tests.test_tdsez import REPO, _expect_fatal, _run, _step, _write_tmp

DECK = """\
Verbose           = 0
Dimension         = 2
Domain            = -24.0, 24.0, -24.0, 24.0
SplineDegree      = 5
NQuadratures      = 8
KnotSequence      = uniform
NSplines          = 36
Charge            = -1.0
Mass              = 1.0
Potential         = -1/sqrt(x*x + y*y + 0.5)
PotentialDerivativeX = x/(x*x + y*y + 0.5)^1.5
PotentialDerivativeY = y/(x*x + y*y + 0.5)^1.5
TargetEigenvalue  = -1.5
NBoundStates      = 3
InitialState      = ground
EnablePropagation = 1
EnableCAP         = 1
CAPKmin           = 1.0
Polarization      = x
LaserX            = t<10 ? 0.08*sin(pi*t/10)^2*sin(1.1*t) : 0
Propagator        = pade22
TimeStep          = 0.05
FinalTime         = 10.0
OutputStrideWFS   = 0
"""

DATASETS = ("dipoles", "populations", "energies", "currents", "autocorrelation")
# columns of the 2D dipoles dataset: t, Ex, Ey, Dx, Dy, Ax, Ay
POS, ACC = slice(3, 5), slice(5, 7)


def _data(name, sel, extra=""):
    path = _write_tmp(name, DECK + extra + f"PhysicsOutput = {sel}\n")
    r, _, td = _run(path, zkit.find_binary(), 2)
    _step(f"PhysicsOutput = {sel} runs", r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-500:])
    with h5py.File(td) as f:
        return {k: np.asarray(f[k][()]) for k in DATASETS}


@pytest.fixture(scope="module")
def full():
    return _data("out_all.inp", "all")


def _same_layout(a, b, sel):
    for k in DATASETS:
        _step(f"{sel}: '{k}' has the shape of the full run {b[k].shape}", a[k].shape == b[k].shape)
        _step(f"{sel}: '{k}' keeps the time column", np.array_equal(a[k][:, 0], b[k][:, 0]))


def _rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


@pytest.mark.parametrize("method", ["operator", "discrete"])
@pytest.mark.parametrize(
    "sel,kept",
    [
        ("position", ["position"]),
        ("acceleration", ["acceleration"]),
        ("dipole", ["position", "acceleration"]),
        ("energy", ["energies"]),
        ("population", ["populations"]),
        ("current", ["currents"]),
        ("ac", ["autocorrelation"]),
        ("none", []),
    ],
)
def test_selection_keeps_layout_and_values(full, sel, kept, method):
    ref = full if method == "operator" else _data("out_all_disc.inp", "all", "AccelerationMethod = discrete\n")
    got = _data(f"out_{sel}_{method}.inp", sel, f"AccelerationMethod = {method}\n")
    _same_layout(got, ref, sel)
    parts = {
        "position": (got["dipoles"][:, POS], ref["dipoles"][:, POS]),
        "acceleration": (got["dipoles"][:, ACC], ref["dipoles"][:, ACC]),
        "energies": (got["energies"][:, 1:], ref["energies"][:, 1:]),
        "populations": (got["populations"][:, 1:], ref["populations"][:, 1:]),
        "currents": (got["currents"][:, 1:], ref["currents"][:, 1:]),
        "autocorrelation": (got["autocorrelation"][:, 1:], ref["autocorrelation"][:, 1:]),
    }
    for name, (a, b) in parts.items():
        if name in kept:
            err = _rel(a, b)
            _step(f"{sel}: {name} equals the full run ({err:.1e})", err < 1e-12)
        else:
            _step(f"{sel}: {name} not computed (zeros)", not np.any(a))
    _step(f"{sel}: zkit reads the time axis", np.allclose(got["dipoles"][:, 0], ref["dipoles"][:, 0]))


def test_unknown_output_token_is_rejected():
    _expect_fatal(DECK + "PhysicsOutput = dipole, popluation\n", "physics_output_typo")
