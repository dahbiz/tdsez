"""t-SURFF photoelectron spectra against the analytic free wave packet.

A Gaussian packet psi0 = prod_d (2 pi w^2)^(-1/4) exp(-(x-c)^2/4w^2 + i k0 x)
of a free carrier keeps its canonical-momentum distribution in any uniform
field, for any mass, constant potential and carrier sign:

    |b(k)|^2 = prod_d (2 w^2/pi)^(1/2) exp(-2 w^2 (k_d - k0_d)^2).

With SurffInterior = 1 the projection of psi(T) inside the box is added, so the
spectrum is exact for any FinalTime and the remaining error is the propagator's.
"""

import os

import h5py
import numpy as np

import zkit
from tests.test_tdsez import REPO, _run, _step, _write_tmp, _expect_fatal

FREE1D = """\
Verbose           = 0
Dimension         = 1
Domain            = -40.0, 40.0
SplineDegree      = 7
NQuadratures      = 10
KnotSequence      = uniform
NSplines          = 400
Charge            = -1.0
Mass              = 1.0
Potential         = 0
PotentialDerivativeX = 0
NBoundStates      = 0
InitialState      = wavepacket
WavepacketCenter  = 0
WavepacketMomentum = 2.0
WavepacketWidth   = 1.0
Polarization      = x
LaserX            = 0
Propagator        = pade22
TimeStep          = 0.025
FinalTime         = 20.0
EnablePropagation = 1
EnableCAP         = 1
CAPKmin           = 0.5
OutputStrideWFS   = 100000
TSurff            = 1
SurffInterior     = 1
SurffRadius       = 10
SurffKmax         = 5
SurffKmin         = -1
SurffNk           = 61
"""

PULSE = "t<15 ? 0.2*sin(pi*t/15)^2*sin(1.3*t) : 0"


def _surff(path):
    return os.path.join(REPO, "td", f"surff_{os.path.basename(path)}.h5")


def _exact(k, w, k0):
    return np.prod([(2 * w * w / np.pi) ** 0.5 * np.exp(-2 * w * w * (k[:, d] - k0[d]) ** 2)
                    for d in range(k.shape[1])], axis=0)


def _run_spectrum(name, text, nproc=1):
    path = _write_tmp(name, text)
    if os.path.exists(_surff(path)):
        os.remove(_surff(path))
    r, _, _ = _run(path, zkit.find_binary(), nproc)
    _step(f"{name} runs", r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-800:])
    return h5py.File(_surff(path), "r")


def _maxrel(f, w, k0):
    k, pes = f["momenta"][:], f["pes"][:]
    ex = _exact(k, w, k0)
    return abs(pes - ex).max() / ex.max()


def test_tsurff_free_packet_1d():
    """Field-free packet: spectrum, output layout and attributes."""
    with _run_spectrum("tsurff_free1d.inp", FREE1D) as f:
        err = _maxrel(f, 1.0, [2.0])
        _step("free packet |b(k)|^2 matches the analytic spectrum", err < 1e-4, f"max rel err {err:.3e}")
        _step("grid axis and pes_grid written", f["k_x"].shape == (61,) and f["pes_grid"].shape == (61,))
        _step("flux-only amplitude kept", f["amplitude_flux"].shape == (61, 2))
        _step("attributes", f.attrs["final_time"][0] == 20.0 and f.attrs["interior"][0] == 1.0
              and f.attrs["n_surface_points"][0] == 2.0)


def test_tsurff_field_mass_offset_hole():
    """Canonical momentum is conserved: the same spectrum with a laser pulse,
    a light mass, a potential offset and either carrier sign."""
    for charge in ("-1.0", "1.0"):
        text = FREE1D + f"LaserX = {PULSE}\nMass = 0.5\nPotential = 0.3\nCharge = {charge}\nFinalTime = 15\n"
        with _run_spectrum(f"tsurff_field_{charge}.inp", text) as f:
            err = _maxrel(f, 1.0, [2.0])
            _step(f"Charge {charge}: spectrum in a field", err < 2e-4, f"max rel err {err:.3e}")
            _step(f"Charge {charge}: exterior potential is s*V", abs(f.attrs["exterior_potential"][0]
                  - (-float(charge)) * 0.3) < 1e-12)
            _step(f"Charge {charge}: A(T) != 0", abs(f.attrs["A_final"][0]) > 1e-3)


def test_tsurff_list_equals_grid():
    """The momentum-list path (direct sums) equals the separable grid path."""
    k = np.linspace(-1, 5, 61)
    mom = os.path.join(os.path.dirname(_write_tmp("tsurff_k.dat", "")), "tsurff_k.dat")
    np.savetxt(mom, k)
    base = FREE1D + f"LaserX = {PULSE}\nFinalTime = 15\n"
    with _run_spectrum("tsurff_grid.inp", base) as g:
        a = g["amplitude"][:]
    text = "\n".join(l for l in base.split("\n") if not l.startswith(("SurffKmax", "SurffKmin", "SurffNk")))
    with _run_spectrum("tsurff_list.inp", text + f"\nSurffMomenta = {mom}\n") as f:
        b = f["amplitude"][:]
        _step("list mode writes no grid", "pes_grid" not in f)
    _step("list amplitudes == grid amplitudes", abs(a - b).max() < 1e-12 * abs(a).max(),
          f"{abs(a - b).max():.3e}")


def test_tsurff_2d_mpi_consistency():
    """2D spectrum independent of the rank count; close to the analytic one."""
    text = FREE1D.replace("Dimension         = 1", "Dimension         = 2") + """
Domain = -20.0, 20.0
SplineDegree = 4
NQuadratures = 6
NSplines = 60
PotentialDerivativeY = 0
WavepacketCenter = 0, 0
WavepacketMomentum = 2.0, 1.0
WavepacketWidth = 1.0, 1.0
Polarization = xy
LaserY = 0
Propagator = cn
TimeStep = 0.02
FinalTime = 2
CAPKmin = 1.0
SurffRadius = 6
SurffKmax = 5, 4
SurffKmin = -1, -2
SurffNk = 31, 31
"""
    amps = []
    for n in (1, 3):
        with _run_spectrum(f"tsurff_2d_np{n}.inp", text, n) as f:
            amps.append(f["amplitude"][:])
            if n == 1:
                err = _maxrel(f, 1.0, [2.0, 1.0])
                _step("2D spectrum", err < 5e-3, f"max rel err {err:.3e}")
                _step("2D pes_grid shape", f["pes_grid"].shape == (31, 31))
    d = abs(amps[0] - amps[1]).max() / abs(amps[0]).max()
    _step("np1 == np3", d < 1e-10, f"{d:.3e}")


def test_tsurff_off_writes_nothing():
    """t-SURFF runs only when requested."""
    path = _write_tmp("tsurff_off.inp", FREE1D + "TSurff = 0\nFinalTime = 0.5\n")
    if os.path.exists(_surff(path)):
        os.remove(_surff(path))
    r, _, _ = _run(path, zkit.find_binary(), 1)
    _step("run succeeds", r.returncode == 0, r.stderr.decode()[-800:])
    _step("no surff file", not os.path.exists(_surff(path)))


def test_tsurff_rejects_varying_exterior():
    """A potential that is not constant outside the surface stops the run."""
    text = FREE1D + "Potential = -1/sqrt(x*x+2)\nPotentialDerivativeX = x/(x*x+2)^1.5\nFinalTime = 0.5\n"
    path = _write_tmp("tsurff_coulomb.inp", text)
    r, _, _ = _run(path, zkit.find_binary(), 1)
    out = r.stdout.decode() + r.stderr.decode()
    _step("non-zero exit", r.returncode != 0)
    _step("explains the free-exterior requirement", "free carrier outside the surface" in out, out[-800:])


def test_tsurff_guards():
    _expect_fatal(FREE1D + "SurffRadius = 30\n", "tsurff_in_cap")
    _expect_fatal(FREE1D + "SurffRadius = 0\n", "tsurff_radius")
    _expect_fatal(FREE1D + "SurffNk = 0\n", "tsurff_nk")
    _expect_fatal(FREE1D + "OutputStrideSurff = 0\n", "tsurff_stride")
    _expect_fatal(FREE1D + "InitialState = ground\n", "nbound0_needs_analytic")
    text = "\n".join(l for l in FREE1D.split("\n") if not l.startswith("PotentialDerivativeX"))
    _expect_fatal(text, "empty_potential_derivative")
