# TDSE-Z — Time-Dependent Schrodinger Equation Solver

B-spline / Isogeometric Analysis (IGA) solver for the time-dependent Schrodinger equation (TDSE) in 1D, 2D, and 3D. Built on [PETSc](https://www.mcs.anl.gov/petsc/), [SLEPc](https://slepc.upv.es/), and [PetIGA](https://www.imperial.ac.uk/research/developed-projects/software/petiga/) for scalable parallel eigenvalue problems and time propagation.

---

## Contents

- [What TDSE-Z Does](#what-tdse-z-does)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Running](#running)
- [Input File Reference](#input-file-reference)
  - [Parameter Summary Table](#parameter-summary-table)
  - [Domain / Per-Axis Syntax](#domain--per-axis-syntax)
  - [Knot Sequence Types](#knot-sequence-types)
  - [Physics Output Tokens](#physics-output-tokens)
  - [InitialState Format](#initialstate-format)
  - [muParser Expressions](#muparser-expressions)
  - [User-Defined Constants](#user-defined-constants)
  - [Comments & Blank Lines](#comments--blank-lines)
- [Output Files](#output-files)
- [Command-Line Flags](#command-line-flags)
- [Unit System](#unit-system)
- [Examples](#examples)
  - [1. Static Spectrum — 1D Harmonic Oscillator](#1-static-spectrum--1d-harmonic-oscillator)
  - [2. Time Propagation — H-Atom Ionisation](#2-time-propagation--h-atom-ionisation)
  - [3. t-SURFF — Photoelectron Spectrum](#3-t-surff--photoelectron-spectrum)
  - [4. Superposition Initial State](#4-superposition-initial-state)
  - [5. 2D / 3D Simulations](#5-2d--3d-simulations)
- [Validation Tests](#validation-tests)
- [Architecture Overview](#architecture-overview)
- [Known Limitations](#known-limitations)

---

## What TDSE-Z Does

TDSE-Z solves the TDSE:

$$i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \left[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r}) + V_{\text{laser}}(\mathbf{r},t)\right]\Psi(\mathbf{r},t)$$

using B-spline basis functions (isogeometric analysis) for spatial discretisation and implicit time-stepping (PETSc TS) for propagation.

**Key capabilities:**

- **Static spectrum** — compute bound-state eigenvalues and eigenstates via SLEPc EPS/PEP
- **Time propagation** — evolve an initial state with a laser field
- **Dipole matrix** — compute transition dipole elements $d_{ij} = \langle\psi_i|x|\psi_j\rangle$
- **Population tracking** — time-dependent occupation of bound states
- **Current / autocorrelation** — compute time-dependent observables
- **t-SURFF** — time-dependent surface flux method for photoelectron spectra
- **Absorbing boundary (CAP)** — complex absorbing potential for open systems
- **GPU acceleration** — offload assembly / matrix operations to CUDA via PETSc
- **Multi-GPU** — MPI-distributed problem with GPU devices

---

## Prerequisites

| Package | Minimum Version | Notes |
|---------|----------------|-------|
| PETSc | 3.x | With HDF5, BLAS/LAPACK, MPI |
| SLEPc | matches PETSc | Co-installed with PETSc under `$PETSC_DIR` |
| PetIGA | latest | Co-installed under `$PETSC_DIR` |
| CMake | 3.15+ | For build system |
| GCC / Clang | C++17 compatible | |
| muparser | system or bundled | Expression parser for potentials/lasers |
| Python 3 + pytest + numpy + h5py | optional | For validation tests |

**Environment:**

```bash
export PETSC_DIR=/path/to/petsc
# PETSC_ARCH may be empty
```

---

## Build

```bash
cd /path/to/tdsez
mkdir build && cd build
cmake .. -DPETSC_DIR=$PETSC_DIR
make -j$(nproc)
ctest                        # run validation tests
```

The build automatically detects your PETSc configuration (optimisation flags, CUDA architecture, HDF5 support, OpenMP) and inherits them. This guarantees ABI compatibility.

**CMake options:**

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_OPENMP` | ON | Enable OpenMP parallelism |
| `ENABLE_LTO` | ON | Link-time optimisation |
| `ENABLE_NATIVE` | ON | Use `-march=native` if PETSc didn't specify |
| `ENABLE_AUTO_CUDA` | ON | Auto-enable CUDA when PETSc was built with it |
| `PETSC_INHERIT_FLAGS` | ON | Inherit PETSc optimisation flags |

**Build targets:**

| Target | Description |
|--------|-------------|
| `tdsez` | Main solver — eigensolve + propagation + diagnostics |
| `tdmprocessor` | Full dipole-matrix post-processor (reads saved Dx + eigenstates) |
| `tdmselect` | Selective dipole-element calculator for a single bra state |

---

## Running

```bash
# Static solve (eigenstates only)
./tdsez -inp test.inp

# With MPI parallelism
mpirun -np 4 ./tdsez -inp test.inp

# Enable propagation
mpirun -np 4 ./tdsez -inp test.inp -EnablePropagation 1

# Time-propagation simulation
mpirun -np 8 ./tdsez -inp prop.inp
```

The executable reads the input file path from `-inp` (default: `tdse.prm`). All output files are written relative to the **current working directory** where you run the executable.

---

## Input File Reference

The input file is a plain-text `key = value` file. Each line is parsed independently.

### Syntax Rules

- **Comments:** `#` starts a comment (rest of line ignored)
- **Blank lines:** ignored
- **Trailing comments:** `#` after value is stripped
- **Case-insensitive:** keys and string values are matched case-insensitively
- **Scalar reference:** scalar parameters (e.g. `Amplitude = 0.02`) can be referenced inside muParser expressions (`Potential = Amplitude * x * x`)

### Parameter Summary Table

The table below lists every key accepted by TDSE-Z. "Required" means the parameter is mandatory for a valid simulation. Defaults are given where applicable.

#### Simulation & Grid

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Dimension` | int | — | Spatial dimension: 1, 2, or 3 |
| `Domain` | pair(s) | — | Simulation box bounds. See [Domain syntax](#domain--per-axis-syntax) below |
| `LMin`, `LMax` | real | — | Domain bounds (1D shorthand for all axes) |
| `LMinX..Z`, `LMaxX..Z` | real | — | Per-axis bounds (alternative to `Domain`) |
| `OffsetX..Z` | real | 0 | Spline knot offset per axis |
| `SplineDegree` | int | 3 | B-spline polynomial degree [1, 14] |
| `NQuadratures` | int | 8 | Gauss quadrature points per element |
| `KnotSequence` | string(s) | uniform | See [Knot Sequence types](#knot-sequence-types) below |
| `NSplines` / `Nelements` | int | 80 | Number of basis functions per axis. Many aliases: `ElementCount`, `NumberOfSplines`, `NumberOfElements` |

#### Particles & Constants

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Planck` / `Hbar` | real | 1.0 | Reduced Planck constant |
| `Charge` / `ParticleCharge` | real | 1.0 | Particle charge |
| `Mass` | string | 1.0 | Mass — either a constant string (e.g. `"1.0"`) or a muParser expression like `"1.0 + 0.1*x"` |
| `MassIsConstant` | bool | 1 | Whether `Mass` is a literal constant. Set to `0` for position-dependent mass |
| `dinvMassX..Z` | string | — | Inverse effective mass per axis (needed when `MassIsConstant = 0`) |

#### Potential

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Potential` / `ExternalPotential` | string | — | Potential energy expression in muParser syntax. Uses variables `x`, `y`, `z` |
| `PotentialDerivativeX` | string | — | $\partial V/\partial x$. Required for adaptive knot sequences |
| `PotentialDerivativeY` | string | 0 | $\partial V/\partial y$ (2D/3D) |
| `PotentialDerivativeZ` | string | 0 | $\partial V/\partial z$ (3D) |

#### Laser Field

##### Scalar form (single polarisation):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Amplitude` | real | 0 | Laser amplitude (electric field) |
| `Omega` | real | 0 | Carrier frequency |
| `Phase` / `CEP` | real | 0 | Carrier-envelope phase |
| `PulseDuration` | real | — | Pulse length in optical cycles (used with `EnvelopeType`) |
| `PulseCenter` | real | — | Pulse center time |
| `EnvelopeType` | string | "gaussian" | Envelope shape: `gaussian`, `sin2`, `flat`, `cos2`, `custom` |
| `Envelope` | string | — | Custom envelope expression in muParser (variable `t`) |

##### Per-component form (generalisation):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Ampx`, `Ampy`, `Ampz` | real | 0 | Per-axis amplitude |
| `Omegax`, `Omegay`, `Omegaz` | real | 0 | Per-axis frequency |
| `CEPx`, `CEPy`, `CEPz` | real | 0 | Per-axis carrier-envelope phase |
| `LaserX` / `DriverX` | string | 0 | Electric field $E_x(t)$ as muParser expression (variable `t`) |
| `LaserY` / `DriverY` | string | 0 | Electric field $E_y(t)$ |
| `LaserZ` / `DriverZ` | string | 0 | Electric field $E_z(t)$ |
| `Laser` / `Driver` | string | 0 | Scalar field expression (applies to all axes) |

##### Per-axis list form (shorthand):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `DriverAmplitude` | "a,b,c" | — | Sets `Ampx, Ampy, Ampz`. Single value replicates to all axes |
| `DriverFrequency` | "a,b,c" | — | Sets `Omegax, Omegay, Omegaz` |
| `DriverCEP` / `CarrierEnvelopePhase` | "a,b,c" | — | Sets CEPs per axis |
| `LaserAmplitude` | "a,b,c" | — | Alias for `DriverAmplitude` |
| `LaserFrequency` | "a,b,c" | — | Alias for `DriverFrequency` |

#### Solver Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `UseDirectSolve` / `DirectSolve` | bool | 0 | Use direct eigensolver instead of shift-and-invert |
| `TargetEigenvalue` / `TargetEnergy` | real | 0.0 | Shift value for shift-and-invert mode |
| `NBoundStates` / `NumberOfBoundStates` | int | 3 | Number of eigenstates to compute |
| `NBoundStatesSave` / `SaveBoundStates` | bool | 1 | Whether to save eigenstates to HDF5 |

#### Time Propagation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `EnablePropagation` | bool | 0 | Enable time propagation (set to `1` for dynamical simulation) |
| `TimeStep` / `TimeStepSize` | real | 0.1 | Integration timestep |
| `FinalTime` / `TotalTime` | real | 10.0 | Total simulation time |

#### Initial State

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `InitialState` / `StartState` | string | "state:0" | See [InitialState format](#initialstate-format) below |
| `NormalizeInitialState` / `NormalizePsi0` | bool | 1 | Auto-normalise the initial state |

#### Output & Diagnostics

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Verbose` | bool | 1 | Enable console output |
| `StrictInput` | bool | 1 | Abort on unknown keys (set to `0` to tolerate typos) |
| `PhysicsOutput` / `OutputQuantities` | string | "all" | Comma-separated list of quantities to output during propagation. Default: all. See [Physics Output tokens](#physics-output-tokens) |
| `OutputStrideWFS` | int | 1000 | Output stride for wavefunction snapshots |
| `OutputStrideTS` | int | 1 | Output stride for time-series (dipole, population, energy, current) |
| `OutputStrideAC` | int | 1 | Output stride for autocorrelation function |
| `HDF5Compress` | bool | 1 | Enable HDF5 compression in output files |
| `HDF5CompressLevel` | int | 5 | HDF5 compression level (0-9) |
| `SaveDipoleMatrix` | bool | 0 | Save assembled Dx operator to PETSc binary |
| `SaveDipoleAxes` | string | "x" | Which axes to save: `x`, `y`, `z`, `xy`, `xz`, `yz`, `xyz`, `all`, `none` |
| `BoundStateFormat` / `StateFormat` / `SaveStateFormat` | string | "complex" | State storage format: `"complex"` or `"real"` |

#### Boundary Conditions

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `BoundaryType` / `Boundary` | string | "natural" | Boundary type: `natural`/`neumann` or `wall`/`dirichlet` |
| `EnableCAP` / `AbsorbingBoundary` | bool | 0 | Enable complex absorbing potential |
| `CAPKmin` / `CapStrength` | real | 0.01 | CAP strength parameter |

#### Angular Momentum Diagonalisation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `EnableLzDiag` / `LzDiag` | bool | 0 | Diagonalise $L_z^2$ to resolve degenerate states (2D only) |

#### Hydrogenic Knot Sequence Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `HydrogenicNLin` | "n1,n2,n3" | "30,30,30" | Linear grid nodes per axis for hydrogenic knots |
| `HydrogenicNExp` | "n1,n2,n3" | "30,30,30" | Exponential grid nodes per axis |
| `HydrogenicR1` | "r1,r2,r3" | "1.0,1.0,1.0" | Transition radius per axis (linear -> exponential) |
| `KnotAlpha` | real | 0.5 | Knot sequence mixing parameter |

#### Adaptive Knot Sequence Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `AdaptiveKappa` | real | 5.0 | Adaptive refinement sensitivity |
| `AdaptivePower` | real | 2.0 | Power law for adaptive weights |
| `AdaptiveWFCoarseN` | int | 20 | Coarse grid size for adaptive WF refinement |
| `AdaptiveWFKinLambda` | real | 1.0 | Kinetic energy threshold for adaptive refinement |

#### t-SURFF Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `TSurff` | bool | 0 | Enable t-SURFF method |
| `SurffNk` | int | 30 | Number of momentum bins |
| `SurffKmax` | real | 3.0 | Maximum momentum |
| `OutputStrideSurff` | int | 1 | Output stride for t-SURFF data |
| `SurffCouplingSign` | real | 1.0 | Coupling sign convention |
| `SurffRadius` | real | 0.0 | SURFF surface radius |

#### Polarisation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Polarization` / `LaserPolarization` / `DriverPolarization` | string | "x" | Laser polarisation axis: `x`, `y`, `z`, `xy`, `xz`, `yz`, `xyz` |

---

### Domain / Per-Axis Syntax

The `Domain` key accepts two formats:

**Simple (uniform across axes):**
```
Domain = -20.0, 20.0
```
Sets LMin = -20, LMax = 20 for all axes.

**Per-axis pairs (bracket notation):**
```
Domain = [min,max], [min,max], [min,max]
```
Example:
```
Domain = [-20,20], [-15,15], [-10,10]
```
Sets X=[-20,20], Y=[-15,15], Z=[-10,10]. Missing axes inherit from X.

**Individual axis parameters** (alternative to `Domain`):
```
LMinX = -20    LMaxX = 20
LMinY = -15    LMaxY = 15
LMinZ = -10    LMaxZ = 10
```

---

### Knot Sequence Types

`KnotSequence` controls the distribution of B-spline basis functions. Accepts a comma-separated list of up to 3 sequences (one per axis). A single value replicates to all axes.

| Sequence | Description |
|----------|-------------|
| `uniform` | Uniform spacing. Good for simple problems, periodic potentials |
| `exp` | Exponential spacing (dense near origin, sparse at edges) |
| `symexp` | Symmetric exponential. Best for atomic/ionic systems |
| `adaptive` | Adaptive grid based on potential landscape (requires `PotentialDerivative*`) |
| `adaptive_wf` | Adaptive grid based on wavefunction curvature |
| `hydrogenic` | Hydrogen-like distribution with linear->exponential transition |
| `tan` / `tansym` | Tangent-based distributions |
| `user` / `custom` | User-defined via `LMin*`, `LMax*`, `Offset*` for custom control |

**Example:**
```
KnotSequence = symexp, exp, uniform
```

---

### Physics Output Tokens

`PhysicsOutput` controls which quantities are computed and saved during propagation. Set to `"all"` (default) to compute everything, or list specific tokens:

| Token | Output | Description |
|-------|--------|-------------|
| `dipole` | Time series | Dipole moment $D(t) = \langle\Psi|x|\Psi\rangle$ |
| `population` | Time series | Bound-state occupation probabilities $P_n(t)$ |
| `energy` | Time series | System energy expectation $\langle\Psi|H|\Psi\rangle$ |
| `current` | Time series | Probability current density |
| `autocorrelation` / `ac` | Time series | Autocorrelation function $A(t) = \langle\Psi(0)|\Psi(t)\rangle$ |
| `wfs` | HDF5 file | Wavefunction snapshots |
| `tsurff` | HDF5 file | t-SURFF momentum spectrum |

**Example:**
```
PhysicsOutput = dipole,population,energy
```
This computes only the dipole moment, populations, and energy — skipping current, autocorrelation, wavefunction snapshots, and t-SURFF data. This can significantly reduce runtime and disk I/O.

---

### InitialState Format

Controls which eigenstate(s) the propagation starts from:

| Format | Description |
|--------|-------------|
| `state:0` (default) | Ground state |
| `state:N` | Nth excited state (0-indexed) |
| `sup: c1*i1 [+ c2*i2 [+ c3*i3]]` | Superposition of up to 3 states |

**Examples:**
```
InitialState = state:0         # ground state
InitialState = state:2         # second excited state
InitialState = sup: 0.7*0 + 0.7*1   # superposition: 70% ground + 70% first excited (normalised)
```

The superposition syntax supports **1 to 3 terms**. Coefficients are automatically normalised so $\sum |c_i|^2 = 1$.

---

### muParser Expressions

All physical expressions (potential, laser field, mass, envelope) use [muParser](https://github.com/dnwrzl/muparser) — a fast C++ expression parser.

**Built-in variables:**

| Variable | Description |
|----------|-------------|
| `x`, `y`, `z` | Spatial coordinates |
| `t` | Time (for laser/envelope expressions) |

**Built-in constants:**

| Constant | Value |
|----------|-------|
| `pi` | $\pi = 3.14159...$ |
| `Amplitude` | From input file `Amplitude` key |
| `Phase` | From input file `Phase` key |
| `Omega` | From input file `Omega` key |
| All scalar parameters | Any key with a numeric value (e.g. `Mass`, `Charge`, `Hbar`, `Planck`, `NQuadratures`, `NSplines`, `TargetEigenvalue`, `TimeStep`, `FinalTime`, etc.) |

**Available functions:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `log10`, `sqrt`, `abs`, `pow`, `ceil`, `floor`, `min`, `max`, `sgn`, `gamma`, `erf`, `erfc`, `hypot`, `fmod`

**Conditional (ternary) operator:**
```
Potential = 0.5*k*x*x  (  ((-x)>5.0) + ((x-5.0)>0) )    # harmonic inside box, infinite walls outside
```

---

### User-Defined Constants

Define your own named constants that can be used anywhere in expressions:

```
Variables = V0=1.2, w=0.5, Rc=4.0
Potential = V0 * exp(-w * (x*x + y*y + z*z)) / (1 + w*Rc)
```

Each `Name=value` pair creates a constant. Multiple `Variables` lines accumulate.

---

### Comments & Blank Lines

```
# This is a full-line comment
EnablePropagation = 1    # inline comment (everything after # is stripped)
                     # blank lines are ignored
Dimension = 1            # this is valid
```

---

## Output Files

TDSE-Z writes files relative to your **current working directory** when you run the solver.

### HDF5 Output (`static/EigenData_*.h5`)

Written after the eigensolve step:

| Dataset / Group | Description |
|----------------|-------------|
| `spectrum` | Eigenvalues (N x 2: real part, imaginary part) |
| `states` | Eigenstate coefficients (N x Ns) |
| `run_metadata` | Provenance: code_version, input_file, units, Dimension, SplineDegree, LMinX..Z, Hbar, Charge, etc. |
| `eig_residual` | SLEPc eigenvalue solver residuals |
| `knots_x..z` | Knot vectors per axis |
| `basis_functions` | Number of basis functions per axis |

### Time Evolution (`td/TimeEvolutionData_*.h5`)

Written during propagation (if `EnablePropagation = 1`):

| Dataset | Description |
|---------|-------------|
| `time` | Time array |
| `dipole_x..z` | Dipole moment components |
| `populations` | Bound-state occupation probabilities |
| `energy` | System energy |
| `current_x..z` | Current density |
| `autocorrelation` | Autocorrelation function and modulus |
| `wfs` | Wavefunction snapshots (if `OutputStrideWFS` > 0) |
| `tsurff` | t-SURFF momentum spectrum (if `TSurff = 1`) |

### Dipole Operator (`static/Dx_*.bin`, `static/Dy_*.bin`, `static/Dz_*.bin`)

PETSc binary files containing the assembled dipole operator matrices. Reloadable with `MatLoad` for post-processing with `tdmprocessor`.

### Console Output

When `Verbose = 1`, the solver prints:
- Input parameters summary
- Knot sequence information
- Assembly progress
- Eigensolver convergence
- Dipole matrix elements
- Time propagation progress with timing metrics

---

## Command-Line Flags

| Flag | Value | Description |
|------|-------|-------------|
| `-inp` | string | Path to input file (default: `tdse.prm`) |
| `-enable_gpu` | 0/1 | Enable GPU acceleration |
| `-save_dipole` | string | Which dipole axes to save: `x`, `y`, `z`, `xy`, `xz`, `xyz`, `all`, `none` |
| `-mat_type` | string | PETSc matrix type (e.g. `mpiaij`, `cudaaij`) |
| `-vec_type` | string | PETSc vector type (e.g. `mvaaix`, `cuda`) |
| `-EnablePropagation` | 0/1 | Enable time propagation |
| `-EnableCAP` | 0/1 | Enable absorbing boundary |
| `-Verbose` | 0/1 | Console output level |
| `-NBoundStates` | int | Number of eigenstates to compute |
| `-TimeStep` | real | Integration timestep |
| `-FinalTime` | real | Total simulation time |
| `-inp` | string | Input file path |

All PETSc/SLEPc options are also available (e.g. `-eps_type`, `-ksp_type`, `-pc_type`). Use `./tdsez -help` for the full list.

---

## Unit System

TDSE-Z uses **atomic units (a.u.)**:

| Quantity | Unit | Value |
|----------|------|-------|
| Length | Bohr radius $a_0$ | 0.529177 Angstrom |
| Energy | Hartree | 27.211386 eV |
| Time | $\hbar/E_h$ | 0.024189 fs |
| Mass | Electron mass $m_e$ | 9.109e-31 kg |
| Charge | Elementary charge $e$ | 1.602e-19 C |
| $\hbar$ | 1.0 | |

The convention `hbar^2 / (2m) = 1/2` is used when `hbar = 1` and `m = 1`.

---

## Examples

### 1. Static Spectrum — 1D Harmonic Oscillator

```ini
# 1D harmonic oscillator: V(x) = 0.5*m*omega^2*x^2
# Analytic: E_n = (n + 0.5) * omega
Verbose           = 1
Dimension         = 1
Domain            = -20.0, 20.0
SplineDegree      = 5
NQuadratures      = 8
KnotSequence      = uniform
NSplines          = 80
Mass              = 1.0
Planck            = 1.0
Charge            = 1.0
MassIsConstant    = 1
Potential         = 0.5 * 1.0 * 0.2 * 0.2 * x * x
PotentialDerivativeX = 0.2 * 0.2 * x
UseDirectSolve    = 0
TargetEigenvalue  = 0.0
NBoundStates      = 3
NBoundStatesSave  = 1
EnablePropagation = 0
EnableCAP         = 0
Polarization      = x
```

Run:
```bash
mpirun -np 4 ./tdsez -inp ho1d.inp
```

Expected energies: E0=0.1, E1=0.3, E2=0.5 a.u.

---

### 2. Time Propagation — H-Atom Ionisation

```ini
# Single H-atom, soft-Coulomb potential, linear x-polarisation laser
# LaserX is the ELECTRIC FIELD E(t); TDSE-Z integrates E -> A internally
Verbose           = 0
Dimension         = 1
Domain            = -60.0, 60.0
SplineDegree      = 7
NQuadratures      = 10
KnotSequence      = symexp
NSplines          = 400
Mass              = 1.0
Planck            = 1.0
Charge            = 1.0

# Soft-Coulomb: V(x) = -1/sqrt(x^2 + a^2), a^2=1.44
Potential            = -1.0 / sqrt(x*x + 1.44)
PotentialDerivativeX = x / (x*x + 1.44)^1.5

# Laser (electric field in velocity gauge)
LaserX           = -0.0534 * ( (1.0/4.0)*sin(0.057*t/4.0)*sin(0.057*t) + sin(0.057*t/(2.0*4.0))*sin(0.057*t/(2.0*4.0)) * cos(0.057*t) )
LaserY           = 0.0
LaserZ           = 0.0
EnvelopeType     = custom

TimeStep         = 0.25
FinalTime        = 740.0
EnablePropagation = 1
EnableCAP        = 0
OutputStrideWFS  = 100000
```

This sets up a 1D hydrogen atom with a 4-cycle laser pulse at 0.057 a.u. frequency (808 nm) and intensity 0.0534 a.u. Run for 740 atomic units of time.

---

### 3. t-SURFF — Photoelectron Spectrum

```ini
# Same as Example 2, but with t-SURFF enabled
# ... (same setup as above) ...

# t-SURFF output
TSurff           = 1
SurffNk          = 80
SurffKmax        = 1.5
OutputStrideSurff = 1
SurffCouplingSign = 1.0
SurffRadius      = 0.0
```

Output: `td/TimeEvolutionData_<input>.h5` contains a `tsurff` dataset with the photoelectron energy spectrum.

---

### 4. Superposition Initial State

```ini
# Start from a superposition of ground + first excited state
Dimension         = 1
Domain            = -20.0, 20.0
SplineDegree      = 5
NQuadratures      = 8
KnotSequence      = uniform
NSplines          = 80
Mass              = 1.0
Planck            = 1.0
Charge            = 1.0
MassIsConstant    = 1
Potential         = 0.5 * 1.0 * 0.2 * 0.2 * x * x
PotentialDerivativeX = 0.2 * 0.2 * x
NBoundStates      = 5
NBoundStatesSave  = 1
InitialState      = sup: 0.7*0 + 0.7*1   # 50% ground + 50% first excited (normalised)
EnablePropagation = 1
TimeStep          = 0.1
FinalTime         = 100.0
LaserX            = 0.02 * cos(0.2*t) * sin^2(pi*t/100.0)
PhysicsOutput     = dipole,population,energy
```

The superposition `0.7*0 + 0.7*1` is automatically normalised to $1/\sqrt{2}$ each. The dipole oscillates at the Bohr frequency $\omega_{01} = (E_1 - E_0)/\hbar = 0.2$ a.u.

---

### 5. 2D / 3D Simulations

**2D example** — hydrogenic potential in x-y plane:
```
Dimension         = 2
Domain            = [-20,20], [-20,20]
KnotSequence      = symexp, exp
NSplines          = 60
Potential         = -1/sqrt(x^2 + 1.44) - 1/sqrt(y^2 + 1.44)
PotentialDerivativeX = (x-1.0)/((x-1.0)^2+1.44)^1.5 + (x+1.0)/((x+1.0)^2+1.44)^1.5
PotentialDerivativeY = 0.0
```

**3D example** — soft-Coulomb potential:
```
Dimension         = 3
Domain            = [-25,25], [-25,25], [-25,25]
KnotSequence      = symexp, symexp, symexp
NSplines          = 100
Potential         = -1.0 / sqrt(x*x + y*y + z*z + 1.44)
PotentialDerivativeX = x / (x*x + y*y + z*z + 1.44)^1.5
PotentialDerivativeY = y / (x*x + y*y + z*z + 1.44)^1.5
PotentialDerivativeZ = z / (x*x + y*y + z*z + 1.44)^1.5
```

---

## Validation Tests

Run the built-in validation suite:

```bash
cd build
ctest --output-on-failure
# or:
cd ../tests
pytest -v
```

Tests cover:

- **Analytic correctness** — harmonic oscillator spectrum matches closed-form
- **Input validation guards** — unknown keys, inverted domains, zero mass, bad degree
- **InitialState selection** — ground, excited, superposition, out-of-range
- **Knot partition of unity** — verifies B-spline basis properties
- **Adaptive knot sequence safety** — missing derivative produces clean FATAL, not crash
- **muParser scalar reference** — input scalars usable inside expressions

---

## Architecture Overview

```
main()
  |
  +-- TDSEZParser::PrmReader()    Parse input file
  +-- TDSEZParser::initParsers()   Bind muParser expressions
  |
  +-- TDSEZCore::Assemble()       Build H matrix (PETSc Mat)
  +-- TDSEZCore::Solve()          SLEPc eigensolve -> eigenstates
  +-- TDSEZCore::Output()         Write eigenstate info
  |
  +-- TDSEZAssembler              Build dipole operators (Dx, Dy, Dz)
  +-- TDSEZManager                Propagation state management
  |   +-- SetupSurff()            t-SURFF boundary operators
  |   +-- CloseHDF5()             Finalize HDF5 output
  |
  +-- TDSEZPropagator::Evolve()   Time-stepping loop (PETSc TS)
  |   |
  |   +-- Monitor callbacks:
  |   |   - TDSEZMonitorHDF5_1D/2D/3D  HDF5 writes
  |   |   - Dipole computation
  |   |   - Population tracking
  |   |   - Autocorrelation
  |   |   - Current density
  |
  +-- TDSEZCompUnifiedDipoleMatrix()  <psi_i|D|psi_j>
  +-- TDSEZPrecomputeMomentumMatrix() Momentum operator precomputation
  +-- TDSEZCheckLengthVelocity()     Gauge consistency check
```

---

## Known Limitations

1. **Single particle** — TDSE-Z solves the TDSE for one electron in an external potential. No many-body interactions.
2. **Velocity gauge** — laser interaction is implemented in velocity gauge ($\mathbf{A}\cdot\mathbf{p}$). Length gauge is not available.
3. **Periodic boundaries** — Only hard-wall (Dirichlet) or natural (Neumann) boundaries. No Bloch periodic BCs.
4. **No relativistic effects** — Non-relativistic Schrodinger equation only. No Dirac/Pauli terms.
5. **CPU-bound for large problems** — While GPU acceleration is available for matrix operations, the eigensolve (SLEPc) and time-stepping are primarily CPU-bound for large systems. Multi-GPU MPI is not yet implemented.
6. **Adaptive knots require derivative** — The `adaptive` and `adaptive_wf` knot sequences require `PotentialDerivativeX/Y/Z` to be specified.
