"""Basis, QuadratureRule and the hydrogenic width cap."""

import os
import re
import subprocess

import h5py
import numpy as np

import zkit
from tests.test_tdsez import _step, _write_tmp

HO2D = """\
Verbose = 0
Dimension = 2
Domain = -12.0, 12.0
SplineDegree = 3
NSplines = 24
Charge = -1.0
Mass = 1.0
Potential = 0.02*(x*x + y*y)
PotentialDerivativeX = 0.04*x
PotentialDerivativeY = 0.04*y
NBoundStates = 3
TargetEigenvalue = 0.0
InitialState = ground
EnablePropagation = 0
TimeStep = 0.05
FinalTime = 0.05
"""


def _run(name, text):
    path = _write_tmp(name + ".inp", text)
    d = os.path.dirname(path)
    r = subprocess.run(["mpirun", "--bind-to", "none", "-np", "2", zkit.find_binary(), "-inp", path],
                       capture_output=True, cwd=d)
    out = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
    spec = None
    f = os.path.join(d, "static", f"EigenData_{name}.inp.h5")
    if r.returncode == 0 and os.path.exists(f):
        with h5py.File(f) as h:
            spec = np.sort(np.asarray(h["spectrum"])[:, 0])
    return r.returncode, spec, out


def test_basis_and_quadrature_rule():
    exact = np.array([0.2, 0.4, 0.4])
    for basis in ("bspline", "bernstein", "lagrange", "spectral"):
        for rule in ("legendre", "lobatto", "reduced"):
            name = f"basis_{basis}_{rule}"
            rc, E, out = _run(name, HO2D + f"Basis = {basis}\nQuadratureRule = {rule}\n")
            _step(f"{name} runs", rc == 0 and E is not None, out[-1200:])
            _step(f"{name}: 2D HO levels", np.max(np.abs(E[:3] - exact)) < 1e-4, f"{E[:3]}")


def test_hydrogenic_width_cap_and_singular_quadrature():
    deck = HO2D.replace("KnotSequence", "#").replace("NSplines = 24", "KnotSequence = hydrogenic\n"
           "HydrogenicR1 = 0.1, 0.1\nHydrogenicNLin = 3, 3\nHydrogenicNExp = 6, 6\nHydrogenicHMax = 1.0, 1.0")
    rc, E, out = _run("hyd_cap", deck)
    _step("capped hydrogenic grid runs", rc == 0, out[-1200:])
    widths = [float(w) for w in re.findall(r"h_max=([0-9.]+) au \(capped\)", out)]
    _step("largest element <= HydrogenicHMax", widths and max(widths) <= 1.0, str(widths))
    coul = deck.replace("Potential = 0.02*(x*x + y*y)", "Potential = -1/sqrt(x*x + y*y)")
    rc, E, out = _run("hyd_lobatto", coul + "QuadratureRule = lobatto\n")
    _step("Lobatto point on a Coulomb singularity is reported", rc != 0 and "singular at a quadrature point" in out,
          out[-800:])
