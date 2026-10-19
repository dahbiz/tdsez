"""Node-balanced partition (-tdsez_balance) and in-place stage ILU.

The partition changes only which rank owns which rows: observables and the
wavefunction snapshots (written in natural order through PetIGA's scatters)
must equal PetIGA's own split to solver tolerance. The in-place ILU of the
stage blocks, also with a forced rebuild of the stage solvers at every solve,
must give the same propagation. The estimator must report the balanced rows.
"""

import re
import subprocess

import h5py
import numpy as np
import zkit

from tests.test_tdsez import REPO, _step, _write_tmp

DECK = """Verbose = 1
Dimension = 2
Domain = -24, 24, -24, 24
SplineDegree = 4
NQuadratures = 6
KnotSequence = hydrogenic
HydrogenicInnerSpacing  = 0.2, 0.2
HydrogenicInnerElements = 6, 6
HydrogenicMaxSpacing    = 1.2, 1.2
Mass = 1.0
MassIsConstant = 1
Potential = -1/sqrt(x*x + y*y + 0.64)
PotentialDerivativeX = x/(x*x + y*y + 0.64)^1.5
PotentialDerivativeY = y/(x*x + y*y + 0.64)^1.5
NBoundStates = 3
TargetEigenvalue = -1
InitialState = ground
EnablePropagation = 1
EnableCAP = 1
CAPLayerWidth = 6
Propagator = magnus4
TimeStep = 0.1
FinalTime = 6
OutputStrideWFS = 20
PhysicsOutput = dipole, acceleration, population, energy, wfs
AccelerationMethod = operator
Polarization = x
LaserX = 0.05*sin(pi*t/6)^2*sin(1.0*t)
"""


def _go(name, nproc, *opts):
    path = _write_tmp(name, DECK)
    cmd = ["mpiexec", "--bind-to", "none", "-n", str(nproc), zkit.find_binary(), "-inp", path, "-tdsez_mat_free", *opts]
    r = subprocess.run(cmd, check=False, cwd=REPO, capture_output=True)
    _step(f"{name} ({nproc} ranks {' '.join(opts)}) runs", r.returncode == 0, r.stdout.decode()[-1500:])
    return zkit.Run(REPO, path)


def _compare(a, b, label, tol=1e-9):
    ea, eb = a.evolution, b.evolution
    for key in ("dipoles", "energies", "populations"):
        x, y = np.asarray(getattr(ea, key)), np.asarray(getattr(eb, key))
        err = np.max(np.abs(x - y)) / max(np.max(np.abs(x)), 1e-300)
        _step(f"{label}: {key} equal ({err:.1e})", x.shape == y.shape and err < tol)


def test_balanced_partition_gives_the_same_run():
    petiga = _go("part_petiga.inp", 3, "-tdsez_balance", "0")
    balanced = _go("part_bal.inp", 3, "-tdsez_balance", "1")
    _compare(petiga, balanced, "balanced vs PetIGA split")
    wa, wb = petiga.wavefunctions.data, balanced.wavefunctions.data
    err = np.max(np.abs(wa - wb)) / np.max(np.abs(wa))
    _step(f"wavefunction snapshots equal in natural order ({len(wa)} snapshots, {err:.1e})",
          wa.shape == wb.shape and len(wa) > 1 and err < 1e-8)


def test_stage_ilu_in_place_and_rebuild():
    ref = _go("part_ref.inp", 3, "-tdsez_stage_ilu_inplace", "0")
    inp = _go("part_inp.inp", 3, "-tdsez_stage_ilu_inplace", "1")
    reb = _go("part_reb.inp", 3, "-tdsez_stage_ilu_inplace", "1", "-tdsez_pc_refresh_its", "1")
    _compare(ref, inp, "in-place stage ILU")
    _compare(ref, reb, "in-place stage ILU, rebuilt at every solve")


def test_estimator_reports_the_balanced_rows():
    path = _write_tmp("part_est.inp", DECK.replace("Dimension = 2", "Dimension = 3")
                      .replace("-24, 24, -24, 24", "-24, 24, -24, 24, -24, 24")
                      .replace("0.2, 0.2\n", "0.2, 0.2, 0.2\n").replace("6, 6\n", "6, 6, 6\n")
                      .replace("1.2, 1.2\n", "1.2, 1.2, 1.2\n")
                      .replace("x*x + y*y + 0.64", "x*x + y*y + z*z + 0.64") + "PotentialDerivativeZ = 0\n")
    ratio = {}
    for b in ("0", "1"):
        r = subprocess.run([zkit.find_binary(), "-inp", path, "-tdsez_mat_free", "-tdsez_estimate_memory", "64",
                            "-tdsez_balance", b, "-tdsez_balance_overlap", "0"], check=False, cwd=REPO, capture_output=True)
        out = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", r.stdout.decode()).split())
        m = re.search(r"rows (\d+)-(\d+) \(mean (\d+)\)", out)
        _step(f"estimate (balance {b}) runs", r.returncode == 0 and m is not None, out[-800:])
        ratio[b] = int(m.group(2)) / int(m.group(3))
    _step(f"largest rank / mean rows: {ratio['0']:.2f} (PetIGA) -> {ratio['1']:.2f} (balanced)",
          ratio["1"] < ratio["0"] and ratio["1"] < 1.15)


def test_eigensolver_ilu_in_place_gives_the_same_states():
    ref = _go("part_eps_ref.inp", 3, "-tdsez_eps_ilu_inplace", "0")
    inp = _go("part_eps_inp.inp", 3, "-tdsez_eps_ilu_inplace", "1")
    a, b = np.asarray(ref.spectrum.values), np.asarray(inp.spectrum.values)
    err = np.max(np.abs(a - b))
    _step(f"eigenvalues equal with in-place eigensolver ILU ({err:.1e})", a.shape == b.shape and err < 1e-10)
    _compare(ref, inp, "propagation after in-place eigensolver ILU")
