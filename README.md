# TDSE-ⵣ — Attosecond Quantum Dynamics at Scale

High-performance time-dependent Schrodinger equation solver using B-spline / isogeometric analysis (IGA). Solves the TDSE in 1D, 2D, and 3D with MPI parallelism and optional GPU acceleration.

## Quick Start

```bash
# 1. Set up environment
export PETSC_DIR=/path/to/petsc

# 2. Build
mkdir build && cd build
cmake .. -DPETSC_DIR=$PETSC_DIR
make -j$(nproc)

# 3. Run
mpirun -np 4 ./tdsez -inp tests/inputs/ho1d.inp
```

Full documentation: [docs/README.md](docs/README.md) — complete input file reference, examples, and API.

## What It Does

TDSE-ⵣ solves the time-dependent Schrodinger equation for single-particle quantum systems:

- **Static spectrum** — compute bound-state eigenvalues and eigenstates
- **Time propagation** — evolve an initial state with laser fields
- **Dipole matrix** — transition dipole elements $d_{ij} = \langle\psi_i|x|\psi_j\rangle$
- **Population tracking** — time-dependent bound-state occupations
- **Current / autocorrelation** — time-dependent observables
- **t-SURFF** — photoelectron energy spectra via surface-flux method
- **Absorbing boundary (CAP)** — open-system dynamics

## Key Features

- B-spline basis functions via PetIGA for high-accuracy spatial discretisation
- SLEPc eigensolvers (SLEPc EPS/PEP) for eigenvalue problems
- PETSc time-stepping (TS) for propagation
- MPI parallelism across multiple nodes
- CUDA GPU offload for matrix operations (auto-detected)
- HDF5 output with full provenance metadata
- muParser for user-defined potentials, laser fields, and mass profiles

## Repository Structure

```
├── CMakeLists.txt          # Build configuration
├── LICENSE                 # MIT License (AttoKings Research Group)
├── .gitignore
├── .editorconfig
├── README.md               # This file
├── cmake/
│   └── banner.cmake        # CMake config banner
├── docs/
│   ├── README.md           # Complete user documentation (750+ lines)
│   └── assets/             # Logo, favicon, demo assets
├── include/
│   ├── tdsez_internal.hpp  # Internal interfaces, formatting helpers
│   └── debug.hpp           # Debug utilities
├── src/
│   ├── tdsez.cpp           # Main entry point
│   ├── parser.cpp          # Input file parser
│   ├── core.cpp            # Core solver (assemble, solve, output)
│   ├── core_knots.cpp      # Knot sequence generators
│   ├── manager.cpp         # Propagation state management
│   ├── monitor.cpp         # Time-stepping callbacks
│   ├── propagator.cpp      # Propagator engine
│   ├── assembler.cpp       # Hamiltonian assembly
│   ├── assembler_class.cpp # Assembler class
│   ├── assembly/assembly.cpp # Low-level assembly routines
│   ├── diagnostics.cpp     # Dipole, current, gauge checks
│   ├── drecprocessor.cpp   # Dipole recombination processor
│   ├── tdmprocessor.cpp    # Dipole matrix post-processor
│   ├── tdmselect.cpp       # Selective dipole element calculator
│   ├── debug.cpp           # Debug utilities
│   └── TDSE-Z_logo.txt     # ASCII-art banner
└── tests/
    ├── README.md           # Test suite documentation
    ├── test_tdsez.py       # Validation tests (pytest)
    ├── check_asmbench.py   # Assembler benchmarks
    └── inputs/             # 23 test input files
```

## Documentation

| File | Content |
|------|---------|
| [docs/README.md](docs/README.md) | **Complete user guide** — all input file keys, examples, output format, CLI flags |
| tests/README.md | Validation test suite overview |
| [tests/inputs/](tests/inputs/) | 23 ready-to-run input examples |

The input file reference in `docs/README.md` covers every parameter with descriptions, valid values, and usage examples.

## Dependencies

| Package | Purpose |
|---------|---------|
| PETSc 3.x | Portable, Extensible Toolkit for Scientific Computation |
| SLEPc | Scalable Library for Eigenvalue Problem Computations |
| PetIGA | Isogeometric Analysis with PETSc |
| muparser | Fast C++ expression parser |
| CMake 3.15+ | Build system |
| MPI (OpenMPI/MPICH/etc.) | Distributed parallelism |
| Python 3 + pytest + numpy + h5py | Tests (optional) |

## Citing

If you use TDSE-ⵣ in your research, please cite:

> TDSE-ⵣ — Time-Dependent Schrodinger Equation Solver (B-spline / IGA)
> Dr. Zakaria Dahbi, AttoKings Research Group
> Attosecond Quantum Physics Lab, King's College London, UK

## License

MIT License — Copyright (c) 2024–2026 AttoKings Research Group.
See [LICENSE](LICENSE) for details.

## Contact

- **Research:** [www.attokings.com](https://www.attokings.com)
- **Email:** zakaria.dahbi@kcl.ac.uk
