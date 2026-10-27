# Input files for the TDSE-Z paper

Z. Dahbi and A. Zaïr, *Unified strong-field dynamics simulations from atoms to heterostructures: beyond the
constant-mass approximation* (arXiv:2608.18472).

This folder holds the input file of every TDSE-Z calculation in the paper, one subfolder per figure or section.
Figure numbers refer to the paper. Each file is complete: the mass, the potential, and the laser field are written
in it as run-time expressions, so no other file is needed.

## Running

Build TDSE-Z as described in the top-level README, then run an input file with

```
mpiexec -n <ranks> tdsez.x -inp <file>.inp
```

Results are written to `static/<name>_eigen.h5` and `td/<name>_observables.h5` next to the input file (paper,
Appendix I). Unless noted below, a run takes seconds to minutes on a workstation.

## Contents

| Folder | Paper | Files | Notes |
|---|---|---|---|
| `fig01_knot_sequences` | Fig. 1 | 7 | one static run per knot sequence; only the knot vector is used |
| `fig03_fig04_dqw_interface` | Figs. 3, 4, Table I | 84 | DQW energies for five barrier widths (`table_wb*`), convergence with uniform, C^{p-1} and C^0 interface knots (`conv_*`), mass step (`step_*`); `exact_dqw.py` computes the transfer-matrix reference in 40-digit arithmetic |
| `sec5a_quesne` | Sec. V A | 1 | Quesne oscillator, first ten levels against E_n = (n + 1/2)/2 |
| `fig05_mass_step` | Fig. 5 | 2 + 2 | discrete and operator acceleration; `timestep_sweep` for the Δt dependence |
| `fig06_propagators` | Fig. 6 | 16 | driven harmonic oscillator with Crank–Nicolson, `pade22`, and `magnus4`; 2 ranks |
| `fig07_dqw_strong_field/ten_cycle` | Fig. 7 | 74 | field and photon-energy scans for BDD (interface and uniform knots) and the constant-mass models |
| `fig07_dqw_strong_field/short_pulses` | Sec. V D | 24 | three- and four-cycle pulses |
| `fig07_dqw_strong_field/surface_flux` | Sec. V D | 2 | t-SURFF inside the barrier at 100 and 150 kV/cm |
| `fig07_dqw_strong_field/convergence` | Sec. V D | 13 | basis, time-step, and absorber checks |
| `fig08_photoionization_hydrogen/tdsez` | Fig. 8, App. A | 2 | absorber designed for k_min = 0.3 (main run) and 0.6 (absorber test); 3D, use tens of ranks |
| `fig08_photoionization_hydrogen/scid_tdse` | Fig. 8, App. A | 5 | inputs for SCID-TDSE [S. Patchkovskii and H. G. Muller, Comput. Phys. Commun. 199, 153 (2016)]: reference (k_min = 0.15), absorber, l_max, and box checks, and the k_min = 0.6 comparison |
| `fig09_fig10_h2p/3d` | Figs. 9, 10, App. B | 4 | smaller box (±50 × ±50 × ±92 a.u.), finer grid, larger box (±60 × ±60 × ±110 a.u.), and a box extended to ±150 a.u. along the axis; 7–12 million unknowns, run on about 400 ranks |
| `fig09_fig10_h2p/1d` | Figs. 9, 10, App. B | 4 | soft-core model with the same I_p, boxes ±92 to ±200 |
| `fig11_nondipole_oscillator` | Fig. 11, App. C | 4 | anisotropic oscillators at order 1/c and 1/c^2; `reference.py` gives the exact oscillator-basis solution |
| `fig12_stabilization` | Fig. 12, App. C | 43 | 2D model atom of Kylstra et al.: dipole (`dp`), order 1/c (`ndp`), and order 1/c^2 (`ndp3`) |
| `fig13_scaling` | Fig. 13, App. F | 2 | strong scaling: 3D hydrogen (912 673 unknowns, Crank–Nicolson) and 3D H2+ (3.23 million unknowns, `magnus4`); run with 16 to 256 ranks |

## Notes

* The coupling beyond the dipole approximation (Figs. 11 and 12) is not in the public release. The input files are
  given for completeness; the module is available from the author on request.
* The 3D H2+ runs need a cluster. The input files use the default solver settings of the public release.
* Energies are reproduced to about 1e-10 meV; the last digits depend on the eigensolver tolerance and the build.
* The analysis scripts that produce the figures from the output files are available from the author on request.
