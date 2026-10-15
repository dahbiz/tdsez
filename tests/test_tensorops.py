"""Factorised (Kronecker) monitor operators against the assembled matrices.

The separable operators (M, dipoles, velocities, K) are applied in factorised
form in 2D/3D after a check against the assembled matrices; the recorded time
series must equal those of a run with -tdsez_tensor_ops 0 to rounding.
"""

import os
import subprocess

import h5py
import numpy as np

import zkit
from tests.test_tdsez import _step, _write_tmp

HEAD = """\
Verbose = 0
Charge = -1.0
KnotSequence = uniform
NBoundStates = 0
InitialState = wavepacket
EnablePropagation = 1
TimeStep = 0.05
FinalTime = 0.5
OutputStrideWFS = 100000
"""

CASES = {
    # x-polarised 2D: Dy is not coupled and is freed after the check
    "t2d_xpol": HEAD + """Dimension = 2
Domain = -12.0, 12.0
SplineDegree = 4
NSplines = 30
Mass = 1.0
Potential = -1/sqrt(x*x + 2*y*y + 1)
PotentialDerivativeX = x/(x*x + 2*y*y + 1)^1.5
PotentialDerivativeY = 2*y/(x*x + 2*y*y + 1)^1.5
WavepacketCenter = 1.0, 0.5
WavepacketMomentum = 0.5, 0.2
Polarization = x
LaserX = 0.05*sin(t)
""",
    # 3D with CAP and a two-axis field
    "t3d_cap": HEAD + """Dimension = 3
Domain = -10.0, 10.0
SplineDegree = 3
NSplines = 12
Mass = 0.7
Potential = 0.1*(x*x + y*y + z*z)
PotentialDerivativeX = 0.2*x
PotentialDerivativeY = 0.2*y
PotentialDerivativeZ = 0.2*z
WavepacketMomentum = 0.5, 0.0, 0.3
Polarization = xy
LaserX = 0.05*sin(t)
LaserY = 0.02*cos(t)
EnableCAP = 1
CAPKmin = 2.0
""",
    # 3D variable mass: only M and the dipoles are separable
    "t3d_varmass": HEAD + """Dimension = 3
Domain = -10.0, 10.0
SplineDegree = 3
NSplines = 12
Mass = 1 + 0.3*tanh(x/2)
Potential = 0
PotentialDerivativeX = 0
PotentialDerivativeY = 0
PotentialDerivativeZ = 0
WavepacketMomentum = 0.5, 0.0, 0.3
Polarization = x
LaserX = 0.05*sin(t)
""",
}


def _run(name, text, extra):
    path = _write_tmp(name + ".inp", text)
    d = os.path.dirname(path)
    td = os.path.join(d, "td", f"TimeEvolutionData_{name}.inp.h5")
    if os.path.exists(td):
        os.remove(td)
    r = subprocess.run(["mpirun", "--bind-to", "none", "-np", "3", zkit.find_binary(), "-inp", path] + extra,
                       capture_output=True, cwd=d)
    out = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
    _step(f"{name} {' '.join(extra) or 'factorised'} runs", r.returncode == 0, out[-1500:])
    with h5py.File(td) as f:
        return {k: np.asarray(f[k]) for k in ("dipoles", "energies", "currents")}, out


def test_factorised_observables_match_assembled():
    for name, text in CASES.items():
        ref, _ = _run(name + "_asm", text, ["-tdsez_tensor_ops", "0"])
        new, out = _run(name + "_fac", text, [])
        _step(f"{name}: factorised operators in use", "factorised form" in out, out[-800:])
        for k in ref:
            err = np.max(np.abs(ref[k] - new[k])) / max(np.max(np.abs(ref[k])), 1e-300)
            _step(f"{name}: {k} equal", err < 1e-11, f"{err:.2e}")
        if name == "t2d_xpol":
            _step("uncoupled Dy still recorded", np.max(np.abs(new["dipoles"][:, 4])) > 1e-6)
