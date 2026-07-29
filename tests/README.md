# TDSE-Z Test Suite

## Overview

This directory contains the automated test suite for the TDSE-Z solver.

## Structure

- `inputs/` - Input parameter files (.inp) for regression and validation tests
- `test_tdsez.py` - Main pytest test runner
- `conftest.py` - Test configuration fixtures

## Running

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_tdsez.py -v -k "test_ho1d"
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
