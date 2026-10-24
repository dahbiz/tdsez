"""OutputStrideObs = k: observables recorded every k steps. The propagation is
unchanged, so the rows equal those of the every-step run at the common steps
(to rounding: multi-rank runs differ by ~1e-15 from run to run anyway), and the
time column is uniform."""

import subprocess

import numpy as np
import pytest
import zkit

from tests.test_partition import DECK as DECK2D
from tests.test_tdsez import REPO, _step, _write_tmp

DECK1D = """Verbose = 1
Dimension = 1
Domain = -40, 40
SplineDegree = 5
NQuadratures = 7
NSplines = 200
Mass = 1.0
Potential = -1/sqrt(x*x + 2)
PotentialDerivativeX = x/(x*x + 2)^1.5
NBoundStates = 3
TargetEigenvalue = -1
InitialState = ground
EnablePropagation = 1
EnableCAP = 1
CAPLayerWidth = 10
Propagator = magnus4
TimeStep = 0.05
FinalTime = 20
OutputStrideWFS = 0
PhysicsOutput = dipole, acceleration, population, energy
Polarization = x
LaserX = 0.05*sin(pi*t/20)^2*sin(0.5*t)
"""


@pytest.mark.parametrize("name,deck,nproc", [("1d", DECK1D, 2), ("2d", DECK2D, 3)])
def test_observable_stride(name, deck, nproc):
    runs = {}
    for k in (1, 10):
        path = _write_tmp(f"stride_{name}_{k}.inp", deck + f"OutputStrideObs = {k}\nOutputStrideWFS = 0\n")
        r = subprocess.run(["mpiexec", "--bind-to", "none", "-n", str(nproc), zkit.find_binary(), "-inp", path],
                           check=False, cwd=REPO, capture_output=True)
        _step(f"{name}: stride {k} runs", r.returncode == 0, r.stdout.decode()[-1200:])
        runs[k] = zkit.Run(REPO, path).evolution
    for key in ("dipoles", "energies", "populations"):
        a, b = np.asarray(getattr(runs[1], key)), np.asarray(getattr(runs[10], key))
        same = a[::10].shape == b.shape
        err = np.max(np.abs(a[::10] - b)) if same else np.inf
        _step(f"{name}: {key} every 10th row equal ({a.shape[0]} -> {b.shape[0]} rows, {err:.1e})",
              same and err < 1e-12 * max(1.0, np.max(np.abs(a))))
    t = np.real(np.asarray(runs[10].dipoles))[:, 0]
    _step(f"{name}: uniform time column (dt_out = {t[1] - t[0]:.3f})", np.allclose(np.diff(t), t[1] - t[0]))


@pytest.mark.parametrize("stride,expect", [(20, "resolved"), (40, "twice the HHG cutoff"), (80, "are aliased")])
def test_stride_warning(stride, expect):
    """The start-up check compares pi/(N dt) with the HHG cutoff Ip + 3.17 Up:
    no warning when it is above twice the cutoff, a warning below that, and an
    aliasing warning below the cutoff (dry run, nothing propagated)."""
    deck = (DECK1D.replace("FinalTime = 20", "FinalTime = 60")
                  .replace("LaserX = 0.05*sin(pi*t/20)^2*sin(0.5*t)", "LaserX = 0.1*sin(pi*t/60)^2*sin(0.5*t)"))
    path = _write_tmp(f"stride_warn_{stride}.inp", deck + f"OutputStrideObs = {stride}\n")
    r = subprocess.run([zkit.find_binary(), "-inp", path, "-tdsez_estimate_memory", "1"],
                       check=False, cwd=REPO, capture_output=True)
    out = " ".join(r.stdout.decode().split())
    _step(f"stride {stride}: dry run", r.returncode == 0, out[-600:])
    _step(f"stride {stride}: '{expect}' reported", expect in out, out[out.find("OBSERVABLE"):][:700])
