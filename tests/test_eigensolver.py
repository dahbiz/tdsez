"""Eigensolver robustness: a failing inner solve falls back to the next inner
solver instead of aborting, and the result equals the direct solve."""

import os
import subprocess

import h5py
import numpy as np

import zkit
from tests.test_tdsez import _step, _write_tmp

DECK = """\
Verbose = 0
Dimension = 2
Domain = -12.0, 12.0
SplineDegree = 4
NSplines = 40
Charge = -1.0
Mass = 1.0
Potential = 0.02*(x*x + 1.7*y*y)
PotentialDerivativeX = 0.04*x
PotentialDerivativeY = 0.068*y
NBoundStates = 6
TargetEigenvalue = 0.0
InitialState = ground
EnablePropagation = 0
TimeStep = 0.05
FinalTime = 0.05
"""


def _solve(name, extra):
    path = _write_tmp(name + ".inp", DECK)
    d = os.path.dirname(path)
    r = subprocess.run(["mpirun", "--bind-to", "none", "-np", "2", zkit.find_binary(), "-inp", path] + extra,
                       capture_output=True, cwd=d)
    out = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
    out = " ".join(out.split())  # wrapped log lines
    _step(f"{name} runs", r.returncode == 0, out[-1500:])
    with h5py.File(os.path.join(d, "static", f"EigenData_{name}.inp.h5")) as f:
        return np.asarray(f["spectrum"])[:, 0], np.asarray(f["run_metadata/eig_residual"]), out


def test_failed_inner_solve_falls_back():
    ref, res_ref, _ = _solve("eps_direct", ["-tdsez_eps_inner", "direct"])
    # one FGMRES iteration per solve cannot converge: the iterative attempt fails
    E, res, out = _solve("eps_fallback", ["-tdsez_eps_inner", "iterative", "-st_ksp_max_it", "1"])
    _step("iterative attempt reported as failed", "inner solve failed" in out, out[-1500:])
    _step("same spectrum as the direct solve", len(E) == len(ref) and np.max(np.abs(E - ref)) < 1e-10,
          f"{E} vs {ref}")
    _step("backward errors small", np.max(res) < 1e-10 and np.max(res_ref) < 1e-10, f"{np.max(res):.1e}")
