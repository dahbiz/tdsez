"""-tdsez_estimate_memory: the dry run must see the grid of the real run.

The degrees of freedom and the nonzeros per rank are exact (knots of the run,
PetIGA partition), so on one rank they must equal the counts from the knot
vectors that the real run saves: a basis pair couples when the two functions
share an element of positive length (C0 interface knots included). The memory
model itself is calibrated against measured peaks (docs: CLI-Flags).
"""

import os
import re
import subprocess

import h5py
import numpy as np
import zkit

from tests.test_tdsez import REPO, _run, _step, _write_tmp

DECK = """Verbose = 1
Dimension = 2
Domain = -20, 20, -12, 12
SplineDegree = 4
NQuadratures = 6
KnotSequence = interface, symexp
NSplines = 30
Mass = (abs(x) < 5) ? 0.5 : 1.0
MassIsConstant = 0
Potential = (abs(x) < 5) ? -0.3 : 0.0
PotentialDerivativeX = 0
PotentialDerivativeY = 0
NBoundStates = 2
NBoundStatesSave = 1
TargetEigenvalue = -0.3
InitialState = ground
EnablePropagation = 0
"""


def _pairs_1d(U, p):
    """Number of coupled basis pairs on one axis (nonzeros of a 1D operator).
    The saved knots are PetIGA's compact form (right end with multiplicity p)."""
    U = np.concatenate([U, [U[-1]] * (p + 1 - int(np.sum(U == U[-1])))])
    n = len(U) - p - 1
    elems = [k for k in range(len(U) - 1) if U[k + 1] > U[k]]
    lo, hi = np.full(n, n), np.full(n, -1)
    for k in elems:                       # functions k-p..k are nonzero on [U_k, U_k+1)
        lo[k - p:k + 1] = np.minimum(lo[k - p:k + 1], k - p)
        hi[k - p:k + 1] = np.maximum(hi[k - p:k + 1], k)
    return n, int(np.sum(hi - lo + 1))


def _estimate(path, ranks, *opts):
    r = subprocess.run([zkit.find_binary(), "-inp", path, "-tdsez_estimate_memory", str(ranks), *opts],
                       check=False, cwd=REPO, capture_output=True)
    out = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout.decode())
    _step(f"estimate on {ranks} ranks runs", r.returncode == 0 and "MEMORY ESTIMATE" in out, out[-1500:])
    return " ".join(out.split())


def test_estimate_counts_equal_the_knots_of_the_run():
    path = _write_tmp("memest.inp", DECK)
    est = _estimate(path, 1)
    dof = int(re.search(r"Degrees of freedom = (\d+)", est).group(1))
    nnz = int(re.search(r"rows, (\d+) nonzeros", est).group(1))

    r, _, _ = _run(path, zkit.find_binary(), 1)
    _step("static run", r.returncode == 0, r.stdout.decode()[-800:])
    stem = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(os.path.join(REPO, "static", f"{stem}_eigen.h5"), "r") as f:
        Ux, Uy = np.array(f["knots_x"]).ravel().real, np.array(f["knots_y"]).ravel().real
    (nx, px), (ny, py) = _pairs_1d(Ux, 4), _pairs_1d(Uy, 4)
    _step(f"degrees of freedom {dof} = {nx} x {ny}", dof == nx * ny)
    _step(f"nonzeros {nnz} = {px} x {py} (from the saved knots)", nnz == px * py)


def test_estimate_scales_with_ranks_and_preconditioner():
    path = _write_tmp("memest_td.inp", DECK.replace("EnablePropagation = 0", "EnablePropagation = 1\n"
                                                    "Propagator = magnus4\nTimeStep = 0.1\nFinalTime = 1\nPolarization = x\nLaserX = 0.01*sin(t)"))
    peak = lambda s: float(re.search(r"Peak per rank = (\d+) MB", s).group(1))
    one, four = _estimate(path, 1, "-tdsez_estimate_base_mb", "0"), _estimate(path, 4, "-tdsez_estimate_base_mb", "0")
    _step(f"peak per rank falls with the rank count ({peak(one):.0f} -> {peak(four):.0f} MB)", peak(four) < peak(one))
    bj = _estimate(path, 4, "-tdsez_estimate_base_mb", "0", "-pade_stage1_pc_type", "bjacobi", "-pade_stage2_pc_type", "bjacobi")
    _step(f"block Jacobi stages need less than ASM(1) ({peak(bj):.0f} < {peak(four):.0f} MB)", peak(bj) < peak(four))
