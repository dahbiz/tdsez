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
    out = " ".join(out.split())  # wrapped log lines
    _step(f"{name} {' '.join(extra) or 'factorised'} runs", r.returncode == 0, out[-1500:])
    with h5py.File(td) as f:
        return {k: np.asarray(f[k]) for k in ("dipoles", "energies", "currents")}, out


def test_factorised_observables_match_assembled():
    for name, text in CASES.items():
        ref, _ = _run(name + "_asm", text, ["-tdsez_tensor_ops", "0"])
        new, out = _run(name + "_fac", text, [])
        _step(f"{name}: factorised operators in use", "Factorised operators" in out, out[-800:])
        for k in ref:
            err = np.max(np.abs(ref[k] - new[k])) / max(np.max(np.abs(ref[k])), 1e-300)
            _step(f"{name}: {k} equal", err < 1e-11, f"{err:.2e}")
        if name == "t2d_xpol":
            _step("uncoupled Dy still recorded", np.max(np.abs(new["dipoles"][:, 4])) > 1e-6)


def test_sumfact_force_corrections_match_per_point():
    """Operator-form acceleration with a position-dependent mass: the
    sum-factorised force matrices with their mass-gradient corrections give
    the same time series as the per-point kernels (-tdsez_sumfact 0)."""
    for dim in (2, 3):
        name = f"opacc_varmass_{dim}d"
        text = CASES["t3d_varmass"].replace("Dimension = 3", f"Dimension = {dim}")
        text = text.replace("Mass = 1 + 0.3*tanh(x/2)", "Mass = 1 + 0.3*tanh(x/2) + 0.2*exp(-y*y/3)")
        text += "AccelerationMethod = operator\nPolarization = xy\nLaserY = 0.02*cos(t)\n"
        if dim == 2:
            text = text.replace("WavepacketMomentum = 0.5, 0.0, 0.3", "WavepacketMomentum = 0.5, 0.2")
        ref, _ = _run(name + "_pp", text, ["-tdsez_sumfact", "0"])
        new, _ = _run(name + "_sf", text, [])
        for k in ref:
            err = np.max(np.abs(ref[k] - new[k])) / max(np.max(np.abs(ref[k])), 1e-300)
            _step(f"{name}: {k} equal", err < 1e-11, f"{err:.2e}")
        _step(f"{name}: acceleration non-trivial", np.max(np.abs(new["dipoles"][:, -1])) > 0)


def test_quadrature_expectations_match_matrices():
    """<V>, <1/m>, <K>, the forces and the velocities taken from psi at the
    quadrature points give the same time series as the assembled matrices
    (-tdsez_quad_obs 0): constant mass with CAP and operator acceleration,
    variable mass with operator (force corrections) and discrete acceleration."""
    cases = {
        "qo_cap": CASES["t3d_cap"] + "AccelerationMethod = operator\n",
        "qo_vm_op": CASES["t3d_varmass"].replace("Mass = 1 + 0.3*tanh(x/2)", "Mass = 1 + 0.3*tanh(x/2) + 0.2*exp(-y*y/3)")
        + "AccelerationMethod = operator\nPolarization = xy\nLaserY = 0.02*cos(t)\n",
        "qo_vm_disc": CASES["t3d_varmass"] + "AccelerationMethod = discrete\n",
    }
    cases["qo_vm_op_2d"] = cases["qo_vm_op"].replace("Dimension = 3", "Dimension = 2").replace(
        "WavepacketMomentum = 0.5, 0.0, 0.3", "WavepacketMomentum = 0.5, 0.2")
    for name, text in cases.items():
        ref, _ = _run(name + "_mat", text, ["-tdsez_quad_obs", "0"])
        new, out = _run(name + "_quad", text, [])
        _step(f"{name}: quadrature expectations in use", "Quadrature-point expectations" in out, out[-800:])
        runs = {"": new}
        if name == "qo_vm_op":
            # per-point assembly: the force coefficients are tabulated by the monitor itself
            runs[" (per-point assembly)"], _ = _run(name + "_pp", text, ["-tdsez_sumfact", "0"])
        for tag, got in runs.items():
            for k in ref:
                err = np.max(np.abs(ref[k] - got[k])) / max(np.max(np.abs(ref[k])), 1e-300)
                _step(f"{name}{tag}: {k} equal", err < 1e-11, f"{err:.2e}")


def test_matrix_free_matches_assembled():
    """-tdsez_mat_free assembles neither K, V, Md, the forces, the uncoupled
    dipoles nor (constant mass) the velocities, and forms the Pade stage
    matrices from M, H and D without the cached field-free parts: the time
    series must equal those of the default run to rounding."""
    cases = {
        "mf_2d": CASES["t2d_xpol"] + "Propagator = pade22\n",
        "mf_cap": CASES["t3d_cap"] + "Propagator = pade22\nAccelerationMethod = operator\n",
        "mf_vm_disc": CASES["t3d_varmass"] + "Propagator = pade22\nAccelerationMethod = discrete\n",
        "mf_vm_op": CASES["t3d_varmass"] + "Propagator = magnus4\nAccelerationMethod = operator\n",
    }
    for name, text in cases.items():
        ref, _ = _run(name + "_ref", text, [])
        new, out = _run(name + "_mf", text, ["-tdsez_mat_free"])
        _step(f"{name}: matrix-free in use", "Matrix-free" in out, out[-800:])
        for k in ref:
            err = np.max(np.abs(ref[k] - new[k])) / max(np.max(np.abs(ref[k])), 1e-300)
            _step(f"{name}: {k} equal", err < 1e-11, f"{err:.2e}")
