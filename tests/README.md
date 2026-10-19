# TDSE-Z Test Suite

## Overview

This directory contains serial parser-validation checks and MPI regression
tests for the TDSE-Z solver. Generated input files are written to a temporary
directory; checked-in files under `inputs/` are used as read-only fixtures.

## Structure

- `inputs/` - Input parameter files (.inp) for regression and validation tests
- `test_tdsez.py` - Spectra, HDF5 provenance, input guards, initial states,
  postprocessors
- `test_analytic.py` - Analytic regression tests against exact solutions
  (see below)
- `conftest.py` - Test configuration fixtures (locates `tdsez.x`)

The whole suite runs in about 20 s on 2-5 MPI ranks. Every case is small; the
largest single test takes about 5 s.

## Analytic regression tests (`test_analytic.py`)

Each test compares TDSE-Z with an exact result, with tolerances close to the
accuracy the discretisation reaches:

| Test | Reference | Tolerance |
|---|---|---|
| Infinite well, Dirichlet | $`E_n = n^2\pi^2/2L^2`$ | $`10^{-11}`$ relative |
| Quesne position-dependent mass | $`E_n = (n+\tfrac12)/2`$ | $`10^{-9}`$ |
| Mass step $`m = 1\,|\,4`$, interface knots, Dirichlet and Neumann | exact BDD transcendental equation | $`10^{-12}`$ |
| GaAs/AlGaAs double quantum well, interface knots | exact transfer-matrix solution | $`10^{-8}`$ meV |
| Driven oscillator, Crank–Nicolson and Magnus-4 | coherent state $`\langle x\rangle(t)`$ | accuracy, observed order 2 / 4, norm $`10^{-9}`$ |
| Driven mass step | $`a = d^2\langle x\rangle/dt^2`$, $`J = d\langle x\rangle/dt`$, length–velocity relation | $`10^{-4}`$, $`10^{-9}`$ |
| Sum-factorised assembly (2D/3D, $`x`$, $`z`$, $`xy`$) | per-point kernels, element by element | $`10^{-12}`$ |

`inputs/bench_h3d.inp` (3D, $`p = 7`$, about $`1.2\times10^6`$ DoFs) is a
cluster benchmark and is not part of the test suite.

## Running

```bash
# Run all tests (from the build directory)
cd build && make test

# Run with verbose output (from the source root)
pytest tests/ -v -s

# Run specific test
pytest tests/test_tdsez.py -v -s -k "test_ho1d"

# Only the analytic regression tests
pytest tests/test_analytic.py -v
```

`make test` runs the serial C++ parser validation test before the MPI physics
suite. Run only the parser test with `cmake --build build --target unit-test`.
Without `TDSEZ_BINARY` the tests use `build/tdsez.x` (or `build/tdsez` from
older builds); set `TDSEZ_BINARY` to the absolute path of the executable to
test another build.

### Dependencies

Install the Python dependencies before running tests:

```bash
pip install pytest h5py scipy
pip install git+https://github.com/dahbiz/zkit.git   # zkit is not on PyPI
```

## Input Files

Each `.inp` file defines a test case. Naming convention:

- `bad_*.inp` - Input files designed to test error handling / validation
- `ho1d*.inp` - 1D harmonic oscillator cases
- `box1d*.inp` - 1D box/particle-in-a-box cases
- `h2p*.inp` - H2+ (two-center) molecular cases
- `bench*.inp` - Benchmark configurations

## Adding a New Test

1. Create a new `.inp` file in `inputs/`
2. Add a corresponding test function in `test_tdsez.py`
3. If the test generates expected output, store it in `inputs/expected/`
