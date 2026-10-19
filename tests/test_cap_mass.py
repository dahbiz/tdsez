"""CAP for an (effective) mass m: the Manolopoulos profile is scaled by 1/m.

For mass m, potential V/m, field E(t/m)/m, step m dt and final time m T the
Hamiltonian is H_1/m on the same basis, so the run is the unit-mass run in the
time t/m: the norm absorbed by the CAP, the dipole and the populations must agree
to rounding. Before the 1/m scaling a GaAs-like mass (m = 0.067) absorbed nothing.

2D: a mass written as a function of x (element-kernel CAP, 1/m(r) at the
quadrature points) must equal the same constant mass (separable CAP).
"""

import re
import subprocess

import numpy as np
import pytest
import zkit
from zkit.layouts import dipole_layout

from tests.test_tdsez import REPO, _run, _step, _write_tmp

BASE1D = """Verbose = 0
Dimension = 1
Domain = -60, 60
SplineDegree = 6
NQuadratures = 8
NSplines = 240
Mass = {m}
Potential = -{s}*exp(-x*x/4)
PotentialDerivativeX = {s}*0.5*x*exp(-x*x/4)
NBoundStates = 2
TargetEigenvalue = -{s}
InitialState = ground
EnablePropagation = 1
EnableCAP = 1
CAPKmin = 0.3
Propagator = magnus4
TimeStep = {dt}
FinalTime = {T}
OutputStrideWFS = 0
PhysicsOutput = dipole, population, energy
Polarization = x
LaserX = {s}*0.8*sin(pi*t/{T})^2*sin(0.9*t/{m})
"""


def _ev(name, text, nproc=2):
    path = _write_tmp(name, text)
    r, _, _ = _run(path, zkit.find_binary(), nproc)
    _step(f"{name} runs", r.returncode == 0, r.stdout.decode()[-1200:])
    return zkit.Run(REPO, path).evolution


@pytest.mark.parametrize("m", [0.067, 4.0])
def test_cap_absorbs_like_unit_mass_in_scaled_time(m):
    T, dt = 80.0, 0.02
    one = _ev("capm_1.inp", BASE1D.format(m=1.0, s=1.0, dt=dt, T=T))
    mm = _ev(f"capm_{m}.inp", BASE1D.format(m=m, s=1.0 / m, dt=dt * m, T=T * m))
    n1, nm = one.energies[:, 6], mm.energies[:, 6]
    lost = 1 - n1[-1]
    _step(f"unit mass: CAP absorbs {lost:.3f} of the norm", lost > 0.02)
    _step(f"m = {m}: absorbed norm(t) = unit mass in t/m ({np.max(np.abs(nm - n1)):.1e})",
          np.max(np.abs(nm - n1)) < 1e-10)
    lay = dipole_layout(1)
    d1, dm = one.dipoles[:, lay.dipole_x], mm.dipoles[:, lay.dipole_x]
    _step(f"m = {m}: q<x>(t) identical ({np.max(np.abs(dm - d1)):.1e})", np.max(np.abs(dm - d1)) < 1e-9 * np.max(np.abs(d1)))


BASE2D = """Verbose = 0
Dimension = 2
Domain = -30, 30, -30, 30
SplineDegree = 5
NQuadratures = 6
NSplines = 70
Mass = {mass}
Potential = -1/sqrt(x*x + y*y + 0.64)
PotentialDerivativeX = x/(x*x + y*y + 0.64)^1.5
PotentialDerivativeY = y/(x*x + y*y + 0.64)^1.5
NBoundStates = 2
TargetEigenvalue = -1
InitialState = ground
EnablePropagation = 1
EnableCAP = 1
CAPKmin = 0.4
Propagator = pade22
TimeStep = 0.05
FinalTime = 30
OutputStrideWFS = 0
PhysicsOutput = population, energy
Polarization = x
LaserX = 0.1*sin(pi*t/30)^2*sin(0.3*t)
"""


def test_cap_varying_mass_kernel_equals_separable_for_constant_value():
    sep = _ev("capm_sep.inp", BASE2D.format(mass="0.5"))
    ker = _ev("capm_ker.inp", BASE2D.format(mass="0.5 + 0*x"))    # forces the element-kernel CAP
    a, b = sep.energies[:, 6], ker.energies[:, 6]
    _step(f"2D: CAP absorbs {1 - a[-1]:.3f}", 1 - a[-1] > 0.01)
    _step(f"2D: element-kernel CAP (varying mass) = separable CAP ({np.max(np.abs(a - b)):.1e})",
          np.max(np.abs(a - b)) < 1e-10)


HETERO1D = """Verbose = 1
Dimension = 1
Domain = -400, 400
SplineDegree = 7
NQuadratures = 10
KnotSequence = interface
NSplines = 300
Mass = (abs(x) < 100) ? 0.067 : 0.0879
Potential = (abs(x) < 100) ? 0.0 : 0.00893
PotentialDerivativeX = 0
NBoundStates = 2
TargetEigenvalue = 0.001
InitialState = ground
EnablePropagation = 1
EnableCAP = 1
CAPKmin = 0.03
Propagator = magnus4
TimeStep = 0.5
FinalTime = 6000
OutputStrideWFS = 0
PhysicsOutput = energy
AccelerationMethod = discrete
Polarization = x
LaserX = t < 6000 ? 1.5e-4*sin(pi*t/6000)^2*sin(0.0043*t) : 0
"""


def test_cap_with_interface_knots_assembles_and_absorbs():
    """C0 interface knots: functions across an interface share no element, so the
    separable CAP must not insert the full band (it raised 'new nonzero caused a
    malloc', was left incomplete and absorbed nothing)."""
    path = _write_tmp("capm_iface.inp", HETERO1D)
    r = subprocess.run([zkit.find_binary(), "-inp", path, "-tdsez_cap_check"], check=False, cwd=REPO, capture_output=True)
    out = " ".join(r.stdout.decode().split()) + r.stderr.decode()
    _step("runs without PETSc errors", r.returncode == 0 and "PETSC ERROR" not in out, out[-800:])
    m = re.search(r"CAP check = max row-sum difference ([0-9.eE+-]+)", out)
    _step(f"separable CAP = element kernel ({m.group(1) if m else None})", m is not None and float(m.group(1)) < 1e-9)
    ev = zkit.Run(REPO, path).evolution
    _step(f"CAP absorbs ({1 - ev.energies[-1, 6]:.3f})", 1 - ev.energies[-1, 6] > 0.01)
