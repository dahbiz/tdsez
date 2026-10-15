"""Geometric B-spline multigrid (Preconditioner = mg) and preconditioner refresh.

The preconditioner never changes the solution: every propagator solve is
converged to the same 1e-12 relative residual, so the dipole time series with
mg and asm must agree to rounding. The prolongation itself is checked inside
TDSE-Z at set-up (partition of unity and linear precision, also distributed).
"""

import os

import h5py
import numpy as np

import zkit
from tests.test_tdsez import REPO, _run, _step, _write_tmp, _expect_fatal

BASE2D = """\
Verbose           = 0
Dimension         = 2
Domain            = -20.0, 20.0
SplineDegree      = 4
NQuadratures      = 6
KnotSequence      = uniform
NSplines          = 64
Charge            = -1.0
Mass              = 1 + 0.5*tanh(x/3) + 0.3*exp(-y*y/4)
Potential         = -1/sqrt(x*x+y*y+1)
PotentialDerivativeX = x/(x*x+y*y+1)^1.5
PotentialDerivativeY = y/(x*x+y*y+1)^1.5
TargetEigenvalue  = -1.2
NBoundStates      = 1
Polarization      = xy
LaserX            = 0.1*sin(pi*t/2)^2*sin(1.5*t)
LaserY            = 0.05*sin(pi*t/2)^2*cos(1.5*t)
TimeStep          = 0.05
FinalTime         = 2.0
EnablePropagation = 1
EnableCAP         = 1
CAPKmin           = 1.0
OutputStrideWFS   = 100000
"""


def _dipoles(name, text, nproc):
    path = _write_tmp(name, text)
    r, _, td = _run(path, zkit.find_binary(), nproc)
    out = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
    _step(f"{name} runs", r.returncode == 0, out[-1500:])
    with h5py.File(td) as f:
        return np.asarray(f["dipoles"]), out


def test_multigrid_matches_asm():
    """Variable mass, Coulomb potential, CAP and a two-axis field: mg == asm,
    on 1 and 3 ranks, for Crank-Nicolson and Pade(2,2)."""
    for prop in ("cn", "pade22"):
        ref, _ = _dipoles(f"mg_ref_{prop}.inp", BASE2D + f"Propagator = {prop}\n", 1)
        for n in (1, 3):
            d, out = _dipoles(f"mg_{prop}_{n}.inp", BASE2D + f"Propagator = {prop}\nPreconditioner = mg\n", n)
            _step(f"{prop} np{n}: multigrid levels reported", "geometric B-spline multigrid" in out
                  or prop != "cn", out[-600:])
            err = abs(d - ref).max() / abs(ref).max()
            _step(f"{prop} np{n}: mg dipoles == asm dipoles", err < 1e-9, f"{err:.3e}")


def test_preconditioner_refresh_strong_field():
    """A strong field changes the operator so much that the preconditioner
    built at t = 0 stops working; it is rebuilt and the run converges."""
    text = BASE2D + """
Mass = 1.0
Potential = 0
PotentialDerivativeX = 0
PotentialDerivativeY = 0
NBoundStates = 0
InitialState = wavepacket
WavepacketMomentum = 1.0, 0.5
LaserX = 2.0*sin(1.0*t)
LaserY = 1.0*cos(1.0*t)
TimeStep = 0.2
FinalTime = 4.0
"""
    d, out = _dipoles("strong_refresh.inp", text, 2)
    _step("strong field converges", np.all(np.isfinite(d)))
    _step("preconditioner was rebuilt", "Preconditioner rebuilt" in out, out[-800:])


def test_multigrid_strong_field_warning():
    text = BASE2D + "Preconditioner = mg\nLaserX = 2.0\nTimeStep = 0.2\nFinalTime = 0.2\n"
    path = _write_tmp("mg_warning.inp", text)
    r, _, _ = _run(path, zkit.find_binary(), 1)
    out = r.stdout.decode(errors="replace")
    _step("warns when tau*max|E.r| is large", "tau*max|E.r|" in out, out[-800:])


def test_multigrid_guards():
    _expect_fatal(BASE2D + "Preconditioner = amg\n", "bad_preconditioner")
    _expect_fatal(BASE2D + "Preconditioner = mg\nMGLevels = 1\n", "mg_one_level")
    _expect_fatal(BASE2D + "Preconditioner = mg\nMGSmooth = 0\n", "mg_no_smoothing")
