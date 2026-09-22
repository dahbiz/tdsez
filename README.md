<p align="center">
  <img src="docs/brand/logo.png" width="220" alt="TDSE-Z logo">
</p>

<h1 align="center">TDSE-Z — Attosecond Quantum Dynamics, at Scale</h1>

<p align="center">
  <strong>C++17</strong> &middot; <strong>MPI</strong> &middot; <strong>PETSc</strong> &middot; <strong>SLEPc</strong> &middot; <strong>PetIGA</strong>
</p>

<p align="center">
  A high-performance C++ framework for solving the time-dependent Schrödinger equation
  in strong laser fields — atomic, molecular and solid-state systems resolved to the
  attosecond, scaled across MPI clusters.
</p>

<p align="center">
  <a href="https://dahbiz.github.io/tdsez/">Website</a> &middot;
  <a href="https://dahbiz.github.io/tdsez/docs/index.html">Documentation</a>
  <br />
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/9c/UKRI_EPSR_Council-Logo_Horiz-RGB.png?utm_source=commons.wikimedia.org&amp;utm_campaign=index&amp;utm_content=original" alt="UKRI Engineering and Physical Sciences Research Council" width="360" />
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
- **Absorbing boundary (CAP)** — open-system dynamics

## Code Layout

| Component | Responsibility |
|-----------|----------------|
| `include/tdsez_parser.hpp`, `src/parser.cpp`, `src/parser_validation.cpp` | Input fields, file parsing, expression setup, and validation |
| `include/tdsez_info.hpp` | Shared terminal output formatting |
| `include/tdsez_core.hpp`, `src/core.cpp`, `src/core_knots.cpp` | Grid construction and bound-state solve |
| `include/tdsez_assembler.hpp`, `src/assembly/`, `src/assembler_class.cpp` | Operator assembly |
| `include/tdsez_manager.hpp`, `include/tdsez_propagator.hpp`, `src/manager.cpp`, `src/propagator.cpp`, `src/monitor.cpp` | Time evolution and output |
| `src/diagnostics.cpp` | Physics diagnostics |

`include/tdsez_internal.hpp` collects shared solver types and function
declarations. Individual class headers can be included separately.

## Key Features

- B-spline basis functions via PetIGA for high-accuracy spatial discretisation
- SLEPc eigensolvers (EPS/PEP) for eigenvalue problems
- PETSc time-stepping (TS) for propagation
- MPI parallelism across multiple nodes
- HDF5 output with full provenance metadata
- muParser for user-defined potentials, laser fields, and mass profiles

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
| HDF5 | Simulation output |
| Python 3 + pytest + numpy + h5py | Tests (optional) |

## Documentation

The documentation is split into a user guide and a generated C++ API reference.
The user guide explains the model, input format, solver workflow, output schema,
and reproducible examples. The API reference follows the ownership boundaries
of the parser, core, assembler, manager, and propagator.

```bash
# User guide
cmake --build build --target docs-sphinx

# C++ API reference
cmake --build build --target docs
```

The generated pages are written to `docs/docs_sphinx/_build/html/` and
`docs/doxygen/html/`. See [Architecture](docs/docs_sphinx/Architecture.rst),
[API Reference](docs/docs_sphinx/API-Reference.rst), and
[Compilation](docs/docs_sphinx/Compilation.rst) for the supported build and
ownership model. The `Publish documentation` GitHub Actions workflow rebuilds
these pages and publishes them to [GitHub Pages](https://dahbiz.github.io/tdsez/docs/index.html)
when documentation-related changes reach `main`; it can also be started from
the Actions tab with `workflow_dispatch`.

## Citing

If you use TDSE-Z in your research, please cite:

> Dahbi, Zakaria, and Amelle Zaïr. “Unified Strong-Field Dynamics Simulations
> from Atoms to Heterostructures.” *arXiv preprint* arXiv:2608.18472 (2026).
> [arXiv:2608.18472](https://arxiv.org/abs/2608.18472)

The same metadata is available in [`CITATION.cff`](CITATION.cff), which GitHub
uses to display the repository's “Cite this repository” information.

## License & Commercial Use

This software is released under the **PolyForm Noncommercial License 1.0.0**.

* **Academic & Non-Commercial Use:** Free to use, modify, and distribute for
  non-commercial research and educational purposes.
* **Commercial Use:** For-profit companies or commercial projects require a
  separate commercial license. Please contact [zdahbi@outlook.es](mailto:zdahbi@outlook.es).

See [LICENSE](LICENSE) and the [full PolyForm terms](https://polyformproject.org/licenses/noncommercial/1.0.0)
for details.

## Contact

Dr. Zakaria Dahbi

[zakaria.dahbi@kcl.ac.uk](mailto:zakaria.dahbi@kcl.ac.uk) /
[zdahbi@outlook.es](mailto:zdahbi@outlook.es)

## Funding acknowledgement

Z.D. acknowledges funding from UK Research and Innovation (UKRI) under the UK
government’s Horizon Europe funding guarantee [Grant No. EP/Z000807/1].
