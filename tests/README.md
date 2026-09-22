# TDSE-Z Test Suite

## Overview

This directory contains serial parser-validation checks and MPI regression
tests for the TDSE-Z solver. Generated input files are written to a temporary
directory; checked-in files under `inputs/` are used as read-only fixtures.

## Structure

- `inputs/` - Input parameter files (.inp) for regression and validation tests
- `test_tdsez.py` - Main pytest test runner
- `conftest.py` - Test configuration fixtures

## Running

```bash
# Run all tests (from the build directory)
cd build && make test

# Run with verbose output (from the source root)
pytest tests/ -v -s

# Run specific test
pytest tests/test_tdsez.py -v -s -k "test_ho1d"
```

`make test` runs the serial C++ parser validation test before the MPI physics
suite. Run only the parser test with `cmake --build build --target unit-test`.
For an isolated build, set `TDSEZ_BINARY` to the absolute path of its `tdsez`
executable when running pytest directly.

### Dependencies

Install the Python dependencies before running tests:

```bash
pip install pytest h5py scipy zkit-lib
```

## Input Files

Each `.inp` file defines a test case. Naming convention:

- `bad_*.inp` - Input files designed to test error handling / validation
- `ho1d*.inp` - 1D harmonic oscillator cases
- `box1d*.inp` - 1D box/particle-in-a-box cases
- `h2p*.inp` - H2+ (two-center) molecular cases
- `bench*.inp` - Benchmark configurations
- `surff*.inp` - t-SURFF surface flux test cases

## Adding a New Test

1. Create a new `.inp` file in `inputs/`
2. Add a corresponding test function in `test_tdsez.py`
3. If the test generates expected output, store it in `inputs/expected/`
