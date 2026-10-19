"""PhysicsOutput: computing only the requested quantities keeps the HDF5 layout.

Every dataset of td/<name>_observables.h5 has the same shape whatever
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


def test_output_file_names():
    """A run writes static/<name>_* and td/<name>_* (name = input without extension)."""
    import os

    name = "names_check"
    extra = ("Propagator = cn\nOutputStrideWFS = 50\nOutputStrideTS = 50\nNBoundStatesSave = 1\n"
             "SaveDipoleAxes = x\nTSurff = 1\nSurffRadius = 8\nSurffKmax = 2, 2\nSurffNk = 5, 5\n")
    deck = (DECK.replace("CAPKmin           = 1.0", "CAPKmin           = 2.0")
            .replace("Potential         = -1/sqrt(x*x + y*y + 0.5)",
                     "Potential         = -1/sqrt(x*x + y*y + 0.5)*(x*x + y*y < 36)"))   # free outside R = 8
    path = _write_tmp(f"{name}.inp", deck + extra)
    for f in os.listdir(os.path.join(REPO, "static")) + os.listdir(os.path.join(REPO, "td")):
        if name in f:
            os.remove(os.path.join(REPO, "static" if os.path.exists(os.path.join(REPO, "static", f)) else "td", f))
    r, _, _ = _run(path, zkit.find_binary(), 2)
    _step("run succeeds", r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-500:])
    expected = {"static": {f"{name}_eigen.h5", f"{name}_tdm.txt", f"{name}_Dx.bin", f"{name}_Dx.bin.info"},
                "td": {f"{name}_observables.h5", f"{name}_wfs.h5", f"{name}_ts.h5", f"{name}_surff.h5"}}
    for d, files in expected.items():
        got = {f for f in os.listdir(os.path.join(REPO, d)) if name in f}
        _step(f"{d}/ holds {sorted(files)}", got == files, f"got {sorted(got)}")
    _step("no ts file in the run directory", not os.path.exists(os.path.join(REPO, f"ts_{name}.inp.h5")))
    run = zkit.Run(REPO, path)
    _step("zkit finds every file", all(run.path(k).exists() for k in ("eigen", "tdm", "observables", "wfs", "surff", "ts")))


def test_post_processors_read_former_names(tmp_path):
    """dipoles.x reads a run stored under the names of earlier versions."""
    import os
    import shutil
    import subprocess

    name = "former_names"
    text = DECK.replace("EnablePropagation = 1", "EnablePropagation = 0") + "NBoundStatesSave = 1\nSaveDipoleAxes = x\n"
    path = _write_tmp(f"{name}.inp", text)
    r, _, _ = _run(path, zkit.find_binary(), 1)
    _step("solve succeeds", r.returncode == 0, r.stdout.decode()[-800:])
    old = tmp_path / "static"
    old.mkdir()
    shutil.copy(os.path.join(REPO, "static", f"{name}_eigen.h5"), old / f"EigenData_{name}.inp.h5")
    shutil.copy(os.path.join(REPO, "static", f"{name}_Dx.bin"), old / f"Dx_{name}.inp.bin")
    shutil.copy(os.path.join(REPO, "static", f"{name}_Dx.bin.info"), old / f"Dx_{name}.inp.bin.info")
    tool = os.path.join(os.path.dirname(zkit.find_binary()), "dipoles.x")
    p = subprocess.run([tool, f"{name}.inp"], cwd=tmp_path, capture_output=True)
    _step("dipoles.x reads EigenData_*.inp.h5 and Dx_*.inp.bin", p.returncode == 0,
          p.stdout.decode()[-500:] + p.stderr.decode()[-500:])
    _step("and writes static/<name>_tdm_Dx.npy", (old / f"{name}_tdm_Dx.npy").exists())
    _step("zkit reads the former eigen file", zkit.Run(tmp_path, f"{name}.inp").spectrum.values.size == 3)


def test_domain_per_axis_and_state_orientation():
    """Domain = xlo, xhi, ylo, yhi gives a 24 x 12 box (the flat list used to set
    every axis to the first pair), and zkit reconstructs the states with x and y
    in place (TDSE-Z stores [y][x]): oscillator w = (0.5, 1),
    <x^2> = (n_x + 1/2)/w_x, <y^2> = (n_y + 1/2)/w_y."""
    import h5py
    from zkit.io.wfs_field import reconstruct_static_wfs

    text = """Verbose = 0
Mode = td
EnablePropagation = 0
Dimension = 2
Domain = -12, 12, -6, 6
SplineDegree = 4
KnotSequence = hydrogenic
HydrogenicR1 = 0.5
HydrogenicNLin = 1
HydrogenicNExp = 10
HydrogenicHMax = 0.5
Potential = 0.5*(0.25*x*x + y*y)
NBoundStates = 3
NBoundStatesSave = 1
"""
    path = _write_tmp("domain_axes.inp", text)
    r, _, _ = _run(path, zkit.find_binary(), 1)
    _step("runs", r.returncode == 0, r.stdout.decode()[-800:])
    run = zkit.Run(REPO, path)
    with h5py.File(run.path("eigen")) as f:
        span = [(f[k][:].min(), f[k][:].max()) for k in ("knots_x", "knots_y")]
        n = [int(f[k].attrs["nfuncs"]) for k in ("knots_x", "knots_y")]
    _step(f"box {span} = [-12, 12] x [-6, 6]", span == [(-12.0, 12.0), (-6.0, 6.0)])
    _step(f"different basis sizes per axis {n}", n[0] != n[1])
    E = run.spectrum.values
    x, y = np.linspace(-11.5, 11.5, 231), np.linspace(-5.5, 5.5, 111)
    for i, (nx, ny) in zip(np.argsort(E), [(0, 0), (1, 0), (2, 0)]):
        a = reconstruct_static_wfs(run.path("eigen"), int(i), axes=[x, y])["abs2"]
        a /= a.sum()
        mx, my = (a.sum(1) * x * x).sum(), (a.sum(0) * y * y).sum()
        ex, ey = (nx + 0.5) / 0.5, (ny + 0.5) / 1.0
        _step(f"state ({nx},{ny}): <x^2> {mx:.4f} = {ex}, <y^2> {my:.4f} = {ey}",
              abs(mx - ex) < 2e-3 * ex and abs(my - ey) < 2e-3 * ey)


def test_domain_wrong_count_is_rejected():
    _expect_fatal(DECK.replace("Domain            = -24.0, 24.0, -24.0, 24.0", "Domain = -24, 24, -24"), "domain_three_values")
