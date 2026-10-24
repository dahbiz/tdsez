"""Restart files (CheckpointStride, CheckpointWallTime, Restart, CheckpointGuess).

A run that writes checkpoints is unchanged; a run restarted from one of them
continues the uninterrupted run: the observables file is cut back to the
checkpoint time and appended, so it ends with the same rows and values (to
the tolerance of the stage solves; with CheckpointGuess = 1 on the same ranks
to rounding). The wavefunction is stored in the natural ordering, so the
restart may use another number of ranks. CheckpointWallTime writes a
checkpoint and stops cleanly; the restart finishes the run. With t-SURFF the
accumulated amplitudes are part of the checkpoint (Crank-Nicolson run).
"""

import os
import re
import shutil
import subprocess

import h5py
import numpy as np
import zkit

from tests.test_tdsez import REPO, _step, _write_tmp

# 3D H2+ (soft core), magnus4, CAP: 40 steps
DECK3D = """Verbose = 1
Dimension = 3
Domain = -12, 12, -12, 12, -14, 14
SplineDegree = 3
NQuadratures = 5
KnotSequence = hydrogenic
HydrogenicInnerSpacing  = 0.2, 0.2, 0.2
HydrogenicInnerElements = 4, 4, 4
HydrogenicMaxSpacing    = 1.6, 1.6, 1.2
Mass = 1.0
MassIsConstant = 1
Potential = -1 / sqrt(x^2 + y^2 + (z + 1.0)^2 + 0.01) - 1 / sqrt(x^2 + y^2 + (z - 1.0)^2 + 0.01)
PotentialDerivativeX = 0
PotentialDerivativeY = 0
PotentialDerivativeZ = 0
NBoundStates = 2
TargetEigenvalue = -1.1
InitialState = ground
EnablePropagation = 1
EnableCAP = 1
CAPLayerWidth = 4
Propagator = magnus4
TimeStep = 0.1
FinalTime = 4
OutputStrideWFS = 0
PhysicsOutput = dipole, population, energy
Polarization = z
ObserveAllAxes = 0
LaserZ = 0.05*sin(pi*t/4)^2*sin(1.0*t)
"""

BASE3D = DECK3D + "PhysicsOutput = dipole, acceleration, population, energy\n"
KEYS = ("dipoles", "energies", "populations")


def _go(name, text, nproc=4, extra=()):
    path = _write_tmp(name, text)
    r = subprocess.run(["mpiexec", "--bind-to", "none", "-n", str(nproc), zkit.find_binary(), "-inp", path, *extra],
                       capture_output=True, cwd=REPO)
    out = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", r.stdout.decode() + r.stderr.decode()).split())
    return r.returncode, out


def _obs(name):
    with h5py.File(os.path.join(REPO, "td", f"{zkit.run_name(name)}_observables.h5")) as f:
        return {k: np.array(f[k]) for k in KEYS}


def _clean(name):
    for what in ("observables.h5", "wfs.h5", "checkpoint.h5", "surff.h5"):
        p = os.path.join(REPO, "td", f"{zkit.run_name(name)}_{what}")
        if os.path.exists(p):
            os.remove(p)


def _compare(label, ref, new, tol):
    for k in KEYS:
        a, b = ref[k], new[k]
        err = np.abs(a - b).max() / np.abs(a).max() if a.shape == b.shape else np.inf
        _step(f"{label}: {k} rows {len(b)}/{len(a)}, difference {err:.1e}", a.shape == b.shape and err < tol)


def test_checkpoint_and_restart_3d():
    """magnus4, 40 steps: checkpoints every 10 steps leave the run
    unchanged; restarted from the last one (step 30) it continues the run, on
    the same and on another number of ranks."""
    rc, out = _go("rs_ref.inp", BASE3D)
    _step("reference runs", rc == 0, out[-800:])
    ref = _obs("rs_ref.inp")

    _clean("rs_ck.inp")
    rc, out = _go("rs_ck.inp", BASE3D + "CheckpointStride = 10\n")
    _step("run with checkpoints", rc == 0 and out.count("checkpoint: step") == 3, out[-800:])
    _compare("checkpoints written", ref, _obs("rs_ck.inp"), 1e-15)

    for nproc in (4, 2):
        rc, out = _go("rs_ck.inp", BASE3D + "CheckpointStride = 10\nRestart = 1\n", nproc)
        _step(f"restart on {nproc} ranks", rc == 0 and "Restarted = from step 30" in out, out[-800:])
        _step(f"restart on {nproc} ranks: observables cut back to 30 rows", "continued after 30 rows" in out, out[-600:])
        _compare(f"restart on {nproc} ranks", ref, _obs("rs_ck.inp"), 1e-11)


def test_restart_with_guess_history():
    """CheckpointGuess = 1, same ranks: the stage solves start from the same
    guesses, so the restart equals the uninterrupted run to rounding."""
    _clean("rs_g.inp")
    deck = BASE3D + "CheckpointStride = 10\nCheckpointGuess = 1\n"
    rc, out = _go("rs_g.inp", deck)
    _step("run with checkpoints and guess history", rc == 0, out[-800:])
    full = _obs("rs_g.inp")
    rc, out = _go("rs_g.inp", deck + "Restart = 1\n")
    _step("restart with the guess history", rc == 0, out[-800:])
    _compare("restart, guess history", full, _obs("rs_g.inp"), 1e-13)


def test_wall_time_stop_and_restart():
    """CheckpointWallTime: the budget (1 s) is used up during the set-up, so
    the run writes a checkpoint at the first step and stops with exit code 0;
    the restart finishes it."""
    rc, out = _go("rs_ref.inp", BASE3D)
    ref = _obs("rs_ref.inp")
    _clean("rs_w.inp")
    rc, out = _go("rs_w.inp", BASE3D + "CheckpointWallTime = 1\n")
    _step("stops on the wall-time budget with exit code 0",
          rc == 0 and "stopping at step 1" in out and "Stopped for a restart" in out, out[-800:])
    rc, out = _go("rs_w.inp", BASE3D + "Restart = 1\n")
    _step("restart finishes the run", rc == 0 and "Restarted = from step 1" in out, out[-800:])
    _compare("wall-time stop + restart", ref, _obs("rs_w.inp"), 1e-11)


def test_restart_crank_nicolson_tsurff():
    """Crank-Nicolson (TS) with t-SURFF in 2D: the restart continues the TS
    from the checkpoint's time and step, and the accumulated t-SURFF
    amplitudes (summed over ranks) continue the flux integral."""
    import inspect

    import tests.test_tsurff as T
    src = inspect.getsource(T.test_tsurff_end_term_2d_mpi)
    extra = re.search(r'text = END1D.replace\("Dimension         = 1", "Dimension         = 2"\) \+ """(.*?)"""',
                      src, re.S).group(1)
    text = T.END1D.replace("Dimension         = 1", "Dimension         = 2") + extra + "SurffEndTerm = 0\n"
    rc, out = _go("rs_cnref.inp", text, 3)
    _step("Crank-Nicolson reference", rc == 0, out[-800:])
    surff = os.path.join(REPO, "td", "{}_surff.h5")
    with h5py.File(surff.format("rs_cnref")) as f:
        aref = np.array(f["amplitude"])
    _clean("rs_cn.inp")
    rc, out = _go("rs_cn.inp", text + "CheckpointStride = 50\n", 3)
    _step("Crank-Nicolson with checkpoints", rc == 0 and "checkpoint: step 50" in out, out[-800:])
    rc, out = _go("rs_cn.inp", text + "CheckpointStride = 50\nRestart = 1\n", 2)
    _step("Crank-Nicolson restart (on 2 ranks)", rc == 0 and "Restarted = from step 50" in out, out[-800:])
    with h5py.File(surff.format("rs_cn")) as f:
        a = np.array(f["amplitude"])
    err = np.abs(a - aref).max() / np.abs(aref).max()
    _step(f"t-SURFF amplitude after the restart ({err:.1e})", err < 1e-10)


def test_restart_guards():
    """Restart without a checkpoint, with another time step, and a checkpoint
    stride that is not a multiple of the t-SURFF stride stop with a message."""
    _clean("rs_none.inp")
    rc, out = _go("rs_none.inp", BASE3D + "Restart = 1\n", 1)
    _step("Restart = 1 without a checkpoint", rc != 0 and "no checkpoint file" in out, out[-600:])
    _clean("rs_dt.inp")
    rc, out = _go("rs_dt.inp", BASE3D + "CheckpointStride = 10\n", 2)
    rc, out = _go("rs_dt.inp", BASE3D.replace("TimeStep = 0.1", "TimeStep = 0.05") + "Restart = 1\n", 2)
    _step("Restart with another TimeStep", rc != 0 and "was written with TimeStep" in out, out[-600:])
    rc, out = _go("rs_bad.inp", BASE3D + "TSurff = 1\nSurffRadius = 5\nOutputStrideSurff = 4\nCheckpointStride = 10\n", 1)
    _step("CheckpointStride not a multiple of OutputStrideSurff", rc != 0 and "multiple of OutputStrideSurff" in out,
          out[-600:])
