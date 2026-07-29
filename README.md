<p align="center">
  <img src="docs/brand/logo.png" width="220" alt="TDSE-Z logo">
</p>

<h1 align="center">TDSE-Z — Attosecond Quantum Dynamics, at Scale</h1>

<p align="center">
  <strong>C++17</strong> &middot; <strong>MPI</strong> &middot; <strong>PETSc</strong> &middot; <strong>SLEPc</strong> &middot; <strong>PetIGA</strong> &middot; <strong>CUDA</strong>
</p>

<p align="center">
  A high-performance C++ framework for solving the time-dependent Schrödinger equation
  in strong laser fields — atomic, molecular and solid-state systems resolved to the
  attosecond, scaled across MPI clusters and GPUs.
</p>

<p align="center">
  <a href="https://dahbiz.github.io/tdsez/">Website</a> &middot;
  <a href="https://dahbiz.github.io/tdsez/gallery.html">Gallery</a> &middot;
  <a href="https://dahbiz.github.io/tdsez/docs/">Documentation</a>
</p>

---

## Quick Start

```bash
# 1. Build (PETSc/SLEPc/PetIGA must be installed and discoverable)
mkdir build && cd build
cmake ..
make -j$(nproc)

# 2. Run
mpirun -np 4 ./tdsez -inp tests/inputs/ho1d.inp
```

## What It Does

TDSE-Z solves the time-dependent Schrödinger equation for single-particle quantum systems:

- **Static spectrum** — bound-state eigenvalues and eigenstates
- **Time propagation** — evolve an initial state under arbitrary laser fields
- **Dipole matrix** — transition elements $d_{ij} = \langle\psi_i|x|\psi_j\rangle$
- **Population tracking** — time-dependent bound-state occupations
- **Current / autocorrelation** — time-dependent observables
- **t-SURFF** — photoelectron energy spectra via the surface-flux method
- **Absorbing boundary (CAP)** — open-system dynamics

## Key Features

- B-spline basis functions via PetIGA for high-accuracy spatial discretisation
- SLEPc eigensolvers (EPS/PEP) for eigenvalue problems
- PETSc time-stepping (TS) for propagation
- MPI parallelism across multiple nodes
- CUDA GPU offload for matrix operations (auto-detected)
- HDF5 output with full provenance metadata
- muParser for user-defined potentials, laser fields, and mass profiles

## Repository Structure

```
├── CMakeLists.txt            Build configuration
├── LICENSE                   MIT License
├── README.md                 This file
├── Doxyfile                  Doxygen config
├── cmake/
│   └── banner.cmake          CMake config banner
├── docs/                     GitHub Pages website
│   ├── index.html            Landing page
│   ├── gallery.html          Simulation gallery
│   ├── docs/                 User documentation (5 sections)
│   ├── brand/                Logo, favicon, hero assets
│   └── gallery/              Gallery media (waveforms, trajectories, video)
├── include/
│   └── tdsez_internal.hpp    Internal interfaces, formatting helpers
├── src/
│   ├── tdsez.cpp             Main entry point
│   ├── parser.cpp            Input file parser
│   ├── core.cpp              Core solver (assemble, solve, output)
│   ├── core_knots.cpp        Knot sequence generators
│   ├── assembler.cpp         Hamiltonian assembly
│   ├── assembler_class.cpp   Assembler class
│   ├── assembly/             Low-level assembly routines
│   ├── manager.cpp           Propagation state management
│   ├── monitor.cpp           Time-stepping callbacks
│   ├── propagator.cpp        Propagator engine
│   ├── diagnostics.cpp       Dipole, current, gauge checks
│   ├── drecprocessor.cpp     Dipole recombination processor
│   ├── tdmprocessor.cpp      Dipole matrix post-processor
│   ├── tdmselect.cpp         Selective dipole element calculator
│   └── TDSE-Z_logo.txt       ASCII-art banner
└── tests/
    ├── test_tdsez.py         Validation tests (pytest)
    ├── conftest.py           Test configuration fixtures
    └── inputs/               Test input files (.inp)
```

## Tests

```bash
# From the build directory
cd build && make test          # builds + runs pytest with 5 MPI procs

# Or directly
pytest tests/ -v -s
```

The suite validates eigenvalue convergence, propagation, and observables against
analytical references (1D/2D/3D harmonic oscillators). See [tests/README.md](tests/README.md).

## Dependencies

| Package | Purpose |
|---------|---------|
| PETSc 3.x | Portable, Extensible Toolkit for Scientific Computation |
| SLEPc | Scalable Library for Eigenvalue Problem Computations |
| PetIGA | Isogeometric Analysis with PETSc |
| muParser | Fast C++ expression parser |
| CMake 3.15+ | Build system |
| MPI (OpenMPI/MPICH) | Distributed parallelism |
| CUDA (optional) | GPU acceleration |
| HDF5 | Simulation output |
| Python 3 + pytest + numpy + h5py | Tests (optional) |

## Citing

If you use TDSE-Z in your research, please cite:

> TDSE-Z — Time-Dependent Schrödinger Equation Solver (B-spline / IGA)
> Dr. Zakaria Dahbi, Attosecond Quantum Physics Lab, King's College London, UK

## License

MIT License — Copyright (c) 2024–2026 Dr. Zakaria Dahbi.
See [LICENSE](LICENSE) for details.

## Contact

- **Research:** [www.attokings.com](https://www.attokings.com)
- **Email:** zakaria.dahbi@kcl.ac.uk
- **GitHub:** [dahbiz/tdsez](https://github.com/dahbiz/tdsez)
