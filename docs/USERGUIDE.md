# TDSE-Z User Guide

## 1. Overview

**TDSE-Z** is a high-performance Time-Dependent Schrodinger Equation (TDSE) solver built on
PetIGA (Isogeometric Analysis), PETSc, and SLEPc. It solves the TDSE in 1D, 2D, and 3D using
B-spline basis functions on a structured tensor-product domain, with MPI parallelism and HDF5
output.

### Core workflow

```
1. Parse input file (.prm)
2. Build IGA mesh (knot vectors, basis functions)
3. Assemble Hamiltonian H and mass matrix M
4. Solve the time-independent Schrödinger equation (TISE): H|psi_n> = E_n M|psi_n>
5. Select initial state |psi(0)> (ground state, excited state, or superposition)
6. Time-propagate the TDSE: i d|psi>/dt = H|psi> + E(t)·D|psi>
7. Compute observables at each step: dipole, energy, populations, currents, autocorrelation
8. Write HDF5 output at configured strides
```

### Architecture

```
tdsez.cpp (main entry)
  ├── TDSEZParser  — key=value input parsing (muParser expressions)
  ├── TDSEZCore    — IGA setup, knot generation, TISE solve (SLEPc EPS)
  ├── TDSEZAssembler — builds H, M, K, V, Dx, Dy, Dz, VelX/Y/Z, dVdx/dy/dz, CAP, Lz
  ├── TDSEZManager — runtime: operator management, observables, HDF5 buffers
  └── TDSEZPropagator — PETSc TS driver (Crank-Nicolson theta=0.5)
```

### Supported dimensions

| Mode | Use case |
|------|----------|
| 1D | Electron in 1D potential, X-only polarization |
| 2D | Two-axis atom, X or Y or XY polarization, Lz angular momentum, degenerate splitting |
| 3D | Full 3D propagation, X/Y/Z or XY/XZ/XYZ polarization |

### Physics capabilities

- Position-dependent mass: `Mass(x,y,z)` with full quantum corrections
- Laser field: user-defined `LaserX(t)`, `LaserY(t)`, `LaserZ(t)` (or `Laser(t)`) expressions, envelope included
- Complex absorbing potential (CAP) for outgoing-wave boundary conditions
- Length-gauge (dipole) interaction: `E(t)·D`
- Velocity-gauge currents: intra-band, inter-band, bound-continuum decomposition
- Berry phase accumulation for circular polarization
- Transition dipole matrix computation and length-velocity gauge consistency check
- t-SURFF photoelectron momentum spectra (`TSurff = 1`, written to `td/<name>_surff.h5`; see the t-SURFF page of the documentation)

---

## 2. Quick Start

### 2.1 Compilation

```bash
cd fused-version
mkdir build && cd build
cmake .. -DPETSC_DIR=$PETSC_DIR
make -j$(nproc)
```

Requirements: PETSc (with HDF5, SLEPc, PetIGA), MPI, OpenBLAS, muParser.

### 2.2 Running a Simulation

```bash
mpirun -np 4 --bind-to none ./tdsez.x -inp ../inps/harmonic_oscillator_1d.prm
```

> **Important:** Always use `--bind-to none` (or `--oversubscribe`) with OpenMPI. The default `--bind-to core` pins each process to one CPU core and makes the solver 8-10x slower.

### 2.3 Output Structure

All output goes into two directories; every file starts with the run name
`<name>`, the input file name without extension (`h2p.inp` → `h2p`):

```
td/                         — time-dependent output
  <name>_observables.h5     — all time-series observables (incl. autocorrelation)
  <name>_wfs.h5             — wavefunction snapshots
  <name>_surff.h5           — t-SURFF spectrum (TSurff = 1)
  <name>_ts.h5              — time-stepper snapshots (OutputStrideTS > 0)
  <name>_checkpoint.h5      — restart file (CheckpointStride or CheckpointWallTime)
static/                     — stationary output
  <name>_eigen.h5           — energy spectrum + eigenstates
  <name>_tdm.txt            — transition dipoles and oscillator strengths
  <name>_Dx.bin, _Dy, _Dz   — saved dipole operators (if requested)
  <name>_gs.log             — degenerate-splitting log (2D, when EnableLzDiag=1)
```

---

## 3. Input File Format

The input file is a plain-text key=value file (`.prm`). Comments start with `#`.
Unknown keys cause a **fatal error** by default (`StrictInput=1`).

### 3.1 Simultaneous Multi-Axis Parsing

Keys that accept multi-axis values follow a standard convention:

- **Single value**: Applies to all 1D/2D/3D axes
- **Comma-separated**: First value, second value, (third value) — missing values fall back to the first
- **Per-axis arrays** (`Domain` key): Uses `[a,b]` syntax, e.g. `Domain = [-10,10], [-10,10]` for 2D

---

## 4. Parameter Reference

### 4.1 Simulation & Grid

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Dimension` | int | 1 | Spatial dimension: 1, 2, or 3 |
| `Domain` | string | `[-10,10]` | Spatial bounds per axis. Format: comma-separated `[min,max]` pairs, or a single pair applied to all axes. E.g., `Domain = [-10,10], [-8,8]` for 2D with asymmetric y-range. |
| `LMinX`, `LMaxX` | real | -10.0 / 10.0 | X-axis bounds (alternative to `Domain`) |
| `LMinY`, `LMaxY` | real | -10.0 / 10.0 | Y-axis bounds |
| `LMinZ`, `LMaxZ` | real | -10.0 / 10.0 | Z-axis bounds |
| `OffsetX`, `OffsetY`, `OffsetZ` | real | 0.0 | Origin offset along each axis |
| `Nelements` | int | 100 | Number of elements (knots) per axis. Also accepts aliases: `NSplines`, `NumberOfSplines`, `ElementCount`, `NumberOfElements` |
| `SplineDegree` | int | 3 | B-spline polynomial degree (1–14). Must satisfy p >= 1. Higher = more accuracy per DOF but more fill-in |
| `NQuadratures` | int | 8 | Quadrature points per element. Should be at least 2p+1 for full integration of degree-2p integrands |
| `KnotSequence` | string | `"uniform"` | Knot-vector distribution (see §4.5) |
| `KnotAlpha` | real | 0.0 | Grading strength for symexp/symtanu/symtann (0.9 when unset) and symlogtan (0.5 when unset). Aliases: `KnotGrading`, `GradingStrength` |
| `NQuadratures` | int | 8 | Gauss-Legendre quadrature order per element. Higher = better integration of spatially-varying operators |

### 4.2 Potential & Mass

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Potential` | muParser expr | Required | Scalar potential V(x,y,z). MuParser expressions using `x`, `y`, `z`. Default is `"0.0"` if not set. |
| `PotentialDerivativeX` | muParser expr | 0.0 | Analytic dV/dx. Auto-computed numerically if omitted, but explicit is preferred |
| `PotentialDerivativeY` | muParser expr | 0.0 | Analytic dV/dy |
| `PotentialDerivativeZ` | muParser expr | 0.0 | Analytic dV/dz |
| `Mass` | muParser expr | `"1.0"` | Position-dependent mass M(x,y,z). Use `"1.0"` for constant mass |
| `MassIsConstant` | bool | auto-detected | Set to `1` for constant mass. Disables quantum-correction terms in dV/dx, speeding assembly |
| `Variables` | string | — | User-defined muParser constants. Format: `V0=1.2, w=0.5`. These become available as constants in ALL expressions (Potential, Mass, Laser, etc.) |

> **Note:** All muParser expressions support `pi`, `abs()` (alias for `fabs()`), and every scalar input parameter is automatically defined as a constant (e.g., `Ampx`, `Omegax`, `TimeStep`, `LMinX`, etc.).

### 4.3 Eigensolver (TISE)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `NBoundStates` | int | 1 | Number of bound states to compute. Eigenstates 0..NBoundStates-1 are stored |
| `TargetEigenvalue` | real | 0.0 | Target eigenvalue for SLEPc EPS solver. States near this energy are prioritized |
| `UseDirectSolve` / `DirectSolve` | bool | false | Use direct factorization instead of iterative eigensolver |
| `NBoundStatesSave` / `SaveBoundStates` | bool | false | Whether to save eigenstates to `static/<name>_eigen.h5` |
| `BoundStateFormat` / `StateFormat` / `SaveStateFormat` | string | `"complex"` | Save format: `"complex"` (canonical eigenvectors with a fixed phase) or `"real"` (phase-rotated to real canonical form via `TDSEZMakeStateReal2`) |

### 4.4 Time Propagation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `EnablePropagation` | bool | 1 | Whether to run time propagation. Set to 0 for TISE-only (ground-state / eigenvalue computation) |
| `TimeStep` / `TimeStepSize` | real | 0.01 | Time step dt in atomic units |
| `FinalTime` / `TotalTime` | real | 100.0 | Total propagation time in atomic units |
| `CheckpointStride` | int | 0 | Write a restart file every N steps. 0 = none |
| `CheckpointWallTime` | real | 0 | Seconds after the program start at which the run writes a restart file and stops cleanly (set below the job time limit). 0 = no limit |
| `Restart` | bool | 0 | Continue from the restart file (same input, any number of ranks) |
| `CheckpointFile` | string | `td/<name>_checkpoint.h5` | Path of the restart file |
| `CheckpointGuess` | bool | 0 | Also store the stage-solver guess history (pade22/magnus4): a restart on the same ranks then equals the uninterrupted run to rounding |

### 4.5 Laser Field

The field on each polarised axis is exactly the expression you give (atomic
units, variable `t`); the pulse envelope is written inside it. Nothing is
multiplied in.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `LaserX`, `LaserY`, `LaserZ` | muParser expr | — | Field E_x(t), E_y(t), E_z(t) on each polarised axis |
| `Laser` | muParser expr | — | Used for every polarised axis without its own `LaserX/Y/Z` |
| `Polarization` / `LaserPolarization` / `DriverPolarization` | string | — | Coupled axes: `x`, `y`, `z`, `xy`, `xz`, `yz`, `xyz` (`all` = `xyz`). Each needs a field |
| `ObserveAllAxes` | bool | 1 | Record dipole, current and acceleration on every spatial axis (cross-polarized response). 0 = only the coupled axes, less memory |
| `Charge` | real | -1 | Carrier charge q: -1 electron, +1 hole. H = K - q (V + r·E); Potential and field are written for an electron |
| `Ampx/y/z`, `Omegax/y/z`, `CEPx/y/z` | real | 0 | Named constants for use inside the expressions; not applied on their own. Further constants via `Variables = T=440, t0=220` |

Example (4-cycle sin² pulse, 800 nm):

```
Polarization = x
Ampx   = 0.0534
Omegax = 0.057
LaserX = Ampx*sin(Omegax*t/8)^2*cos(Omegax*t + CEPx)
```

`Envelope` and `EnvelopeType` (never applied) and the legacy scalar `Amplitude`, `Omega`, `Phase`, `PulseDuration`, `PulseCenter` have been removed; inputs using them stop with a message naming the replacement (`EnvelopeType = custom` only warns).

### 4.6 Boundary Conditions & Absorbing Potentials

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `BoundaryType` / `Boundary` | string | `"Neumann"` | `"Neumann"` / `"Natural"` = reflecting wall (free Galerkin, default). `"Dirichlet"` / `"Wall"` = ψ=0 enforced at boundary |
| `EnableCAP` / `AbsorbingBoundary` | bool | 0 | Enable complex absorbing potential for outgoing-wave boundary conditions |
| `CAPKmin` | real | 0.1 | Lowest momentum absorbed efficiently by the Manolopoulos CAP. Sets the absorbing layer at each wall, width 6.555/CAPKmin (65.5 a.u. for 0.1); at most 45% of each axis |

### 4.7 Knot Sequences

| Sequence | Description | Tuning Keys |
|----------|-------------|-------------|
| `uniform` | Uniform spacing | — |
| `symexp` | Symmetric exponential: dense near origin, sparse at edges | `KnotAlpha` (grading), `KnotSequence` = `"symexp, uniform, uniform"` for per-axis |
| `symtanu` | Inverse-tanh map: dense near origin, sparse at edges | `KnotAlpha` |
| `symtann` | tanh map: dense near the edges, sparse at origin | `KnotAlpha` |
| `symlogtan` | Tangent map, mildly dense near origin | `KnotAlpha` |
| `interface` (alias `interfaces`) | C0 knots (multiplicity p) at every potential/mass breakpoint; elements uniform within each layer, element count per layer proportional to its width | Breakpoints read from `x < c` / `abs(x) < c` comparisons in `Potential` and `Mass` |
| `hydrogenic` | Linear near origin, exponential tail. For Coulombic problems | `HydrogenicR1`/`InnerSpacing`, `HydrogenicNLin`/`InnerElements`, `HydrogenicNExp`/`OuterElements` |

**Per-axis knot sequences** (comma-separated): `"symexp, uniform, symtann"` means X=symexp, Y=uniform, Z=symtann. Missing axes default to the first value.

**Hydrogenic sequence parameters** (per-axis, comma-separated like `Nelements`; each half-axis gets exactly `NLin` linear + `NExp` exponential elements, mirrored about the origin):
- `HydrogenicR1` (alias `InnerSpacing`, `HydrogenicInnerSpacing`): size of the linear near-origin elements in bohr. Default 0.1.
- `HydrogenicNLin` (alias `InnerElements`, `HydrogenicInnerElements`): number of linear elements; r_cross = NLin*R1 must be < Lmax. Default 0 = use Nelements-1.
- `HydrogenicNExp` (alias `OuterElements`, `HydrogenicOuterElements`): number of exponentially growing elements from r_cross to Lmax. Default 10.

Example:
```
KnotSequence  = hydrogenic
InnerSpacing  = 0.1, 0.1, 0.1
InnerElements = 15, 15, 15
OuterElements = 47, 47, 152
```

### 4.8 Output Control

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `OutputStrideObs` / `ObservableStride` | int | 1 | Compute and record the observables every N steps; warns when N is too large for the harmonic spectrum |
| `OutputStrideTS` | int | 0 | Output stride (in steps) for time-series diagnostics. 0 = disabled |
| `OutputStrideWFS` | int | 100 | Output stride (in steps) for wavefunction snapshots. 0 = disabled |
| `OutputStrideAC` | int | 0 | Output stride (in steps) for autocorrelation diagnostics. 0 = disabled |
| `HDF5Compress` | bool | 0 | Enable HDF5 chunk compression |
| `HDF5CompressLevel` | int | 6 | Compression level (0–9, default 6). Higher = smaller files, more CPU |
| `Verbose` | bool | 1 | Verbose console output. Set to 0 for quiet runs |

### 4.9 Dipole Operator Save

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `SaveDipoleMatrix` | bool | 0 | Save assembled dipole operator Dx to PETSc binary file |
| `SaveDipoleAxes` | string | "" | Which axes to save: `"x"`, `"xy"`, `"xyz"`, `"all"`, `"none"`. CLI override via `-save_dipole` |

Dipole operators are saved to `static/<name>_Dx.bin` (and Dy, Dz for multi-axis runs). Loadable with `MatLoad` in PETSc.

### 4.11 Initial State

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `InitialState` / `StartState` | string | `"ground"` | Initial state selector. See below for formats |
| `NormalizeInitialState` / `NormalizePsi0` | bool | 1 | Normalize the assembled initial state so <ψ₀\|M\|ψ₀> = 1 |
| `AccelerationMethod` | string | `"auto"` | Dipole acceleration a(t): `operator` (Ehrenfest force −⟨(1/m)(∇V+E)⟩ plus mass-gradient terms; needs smooth m and V), `discrete` (a = d⟨v⟩/dt from the Galerkin equation of motion, one M solve per output step; exact for sharp interfaces and hard walls), `auto` (`discrete` when `Mass` depends on x, y or z and `MassIsConstant = 0`, otherwise `operator`) |
| `Preconditioner` | string | `"asm"` | Propagator preconditioner: `asm` (additive Schwarz, ILU(0)), `gamg` (PETSc algebraic multigrid; use the ASM/ILU smoother options of `benchmarks/preconditioners/run.sh`) or `mg` (geometric B-spline multigrid, exact knot-insertion prolongation, Galerkin coarse levels, ASM/ILU(0) smoothing). Same result; `mg` needs fewer iterations but is meant for large rank counts, and is not robust when tau*max\|E.r\| > 1.5 (a warning is printed). The preconditioner is rebuilt automatically when a solve needs far more iterations than after the last build (strong fields) |
| `MGLevels` / `MGSmooth` | int | 0 / 1 | Multigrid levels (0 = automatic) and smoothing sweeps |
| `Propagator` | string | `"cn"` | Time stepper. `cn`: Crank-Nicolson (Pade(1,1), second order, 1 sparse solve per step). `pade22`: exponential midpoint rule with a Pade(2,2) exponential (fourth-order field-free phase, second order in the field variation, 2 solves per step). `magnus4`: commutator-free fourth-order Magnus with Pade(2,2) exponentials (fourth order, 4 solves per step). Both rational propagators are unitary for Hermitian H and allow much larger `TimeStep` for the same accuracy; stage solvers accept `-pade_stage1_` / `-pade_stage2_` PETSc options. Each stage solve starts from the previous step's stage increment (about 15% fewer GMRES iterations); `-pade_extrapolate_guess 0` restores the plain initial guess |

**InitialState formats:**
- `"ground"` — Use the 0th converged bound state (ground state)
- `"state:N"` — Use the Nth converged bound state (0-based index). E.g., `"state:2"` = 3rd eigenstate
- `"sup: a*N + b*M [+ c*P]"` — Coherent superposition. Real coefficients a, b, c (up to 3 states). E.g., `"sup: 0.5*0 + 0.866*1"` = 0.5·ψ₀ + 0.866·ψ₁
- `"wavepacket"` — Gaussian wave packet ∏_d (2π w_d²)^(-1/4) exp(−(x_d−c_d)²/4w_d² + i k_d x_d), set by `WavepacketCenter`, `WavepacketMomentum`, `WavepacketWidth` (per axis). With an analytic initial state, `NBoundStates = 0` skips the eigensolver.

### 4.12 Degenerate State Splitting (2D only)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `EnableLzDiag` / `LzDiag` | bool | 0 | Enable Lz² diagonalisation to split degenerate energy shells by angular momentum quantum number. 2D only |

When enabled, degenerate states are re-orthogonalized in the Lz² eigenbasis, assigning quantum numbers (n_r, m) and providing a clean mapping between energy levels and angular momentum.

### 4.13 t-SURFF Photoelectron Spectra

Computed only with `TSurff = 1`; output in `td/<name>_surff.h5`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `TSurff` | bool | 0 | Compute the t-SURFF momentum spectrum |
| `SurffRadius` | real | — | Surface \|x_d\| = R (inside every axis, in front of the CAP layer) |
| `SurffKmax` / `SurffKmin` | real[3] | — / −Kmax | Momentum grid limits per axis |
| `SurffNk` | int[3] | — | Grid points per axis |
| `SurffMomenta` | file | — | Momentum list (rows `kx [ky kz]`) instead of the grid |
| `OutputStrideSurff` | int | 1 | Sampling stride (final time always sampled) |
| `SurffQuadrature` | int | 0 | Gauss points per knot span on the surface (0 = `NQuadratures`) |
| `SurffInterior` | bool | 0 | Add the projection of ψ(T) inside the box (free-particle tests) |
| `SurffMassTolerance` / `SurffPotentialTolerance` | real | 1e-6 / 1e-6 | Allowed variation of mass (relative) and potential (Ha) outside the surface |

### 4.14 Miscellaneous

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `StrictInput` | bool | 1 | Whether unknown keys are fatal errors. Set to 0 to tolerate unknown keys (not recommended) |

---

## 5. CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-inp <file>` | Input parameter file (default: `tdse.prm`) | `-inp inps/harmonic_1d.prm` |
| `-save_dipole <axes>` | Save dipole operator axes at runtime | `-save_dipole xyz` |
| `-mat_type <type>` | PETSc matrix type for IGA matrices | `-mat_type mpiaij` |
| `-vec_type <type>` | PETSc vector type for IGA vectors | `-vec_type mpi` |
| `-tdsez_eps_extra <n>` | Converge n eigenpairs beyond `NBoundStates` (saves the second solve of the degenerate-shell check) | `-tdsez_eps_extra 4` |
| `-blas_num_threads <n>` | BLAS threads per rank (default 1 unless `OMP_NUM_THREADS` etc. is set) | `-blas_num_threads 4` |

CLI flags are processed before PETSc/SLEPc initialization and stripped from argv. The `-mat_type` and `-vec_type` flags are re-injected as `-iga_mat_type` and `-iga_vec_type` to reach PetIGA.

---

## 6. HDF5 Output Schema

### 6.1 Time Evolution Data (`td/<name>_observables.h5`)

Every step is recorded in a ring buffer of size `FLUSH_INTERVAL = 5000` steps and written to HDF5 at multiples of `OutputStrideTS` steps (if >0), but at most once every 2 s of wall time, and in any case every 5000 steps and at the end of the run. The written data are the same for any stride; only how often the file on disk is updated during the run changes.

**Dipole table** (`dipole` enabled):

| Column | Index | Description |
|--------|-------|-------------|
| 0 | t | Time (a.u.) |
| 1 | Ex(t) | Electric field x-component (a.u.) |
| 2 | Ey(t) | Electric field y-component (a.u.) |
| 3 | Ez(t) | Electric field z-component (a.u.) — 3D only |
| 4 | Dx | Dipole moment x-component (a.u.) = q·<ψ\|x\|ψ> |
| 5 | Dy | Dipole moment y-component (a.u.) |
| 6 | Dz | Dipole moment z-component (a.u.) — 3D only |
| 7 | Ax | Acceleration x-component (a.u.) |
| 8 | Ay | Acceleration y-component (a.u.) |
| 9 | Az | Acceleration z-component (a.u.) — 3D only |

Row widths: 7 (1D), 7 (2D), 10 (3D)

**Population table** (`population` enabled):

| Column | Index | Description |
|--------|-------|-------------|
| 0 | t | Time (a.u.) |
| 1..N+1 | P_n = \|c_n\|² | Population of bound state n (0-based) |

Row width: NBoundStates + 1

**Energy table** (`energy` enabled):

| Column | Index | Description |
|--------|-------|-------------|
| 0 | t | Time (a.u.) |
| 1 | Kin | Kinetic energy (a.u.) = <ψ\|K\|ψ> |
| 2 | Pot | Potential energy (a.u.) = <ψ\|V\|ψ> |
| 3 | Int | Interaction energy (a.u.) = -E(t)·d(t) |
| 4 | Tot | Total energy (a.u.) = Kin + Pot + Int |
| 5 | InvMass | <ψ\|1/m(r)\|ψ> — inverse mass expectation |
| 6 | Norm | <ψ\|M\|ψ> — norm conservation check |

Row width: 7

**Current table** (`current` enabled):

| Column | Index | Description |
|--------|-------|-------------|
| 0 | t | Time (a.u.) |
| 1 | Lz | Angular momentum z-component expectation |
| 2 | dGamma | Berry phase increment per step |
| 3 | Jx_tot | Total current x-component |
| 4 | Jy_tot | Total current y-component |
| 5 | Jz_tot | Total current z-component — 3D only |
| 6 | Jx_intra | Intra-band current x (sum |c_n|² · v_nn) |
| 7 | Jy_intra | Intra-band current y |
| 8 | Jz_intra | Intra-band current z — 3D only |
| 9 | Jx_inter | Inter-band current x (sum_{m≠n} c_m* c_n · v_mn) |
| 10 | Jy_inter | Inter-band current y |
| 11 | Jz_inter | Inter-band current z — 3D only |
| 12 | Jx_bc | Bound-continuum current x (remainder) |
| 13 | Jy_bc | Bound-continuum current y |
| 14 | Jz_bc | Bound-continuum current z — 3D only |

Row widths: 11 (1D), 11 (2D), 15 (3D)

**Autocorrelation table** (`autocorrelation` enabled):

| Column | Index | Description |
|--------|-------|-------------|
| 0 | t | Time (a.u.) |
| 1 | Re⟨ψ(0)\|ψ(t)⟩ | Real part of autocorrelation |
| 2 | Im⟨ψ(0)\|ψ(t)⟩ | Imaginary part of autocorrelation |

Row width: 3

### 6.2 Wavefunction Snapshots (`td/<name>_wfs.h5`)

HDF5 multi-timestep dataset. Each timestep stores a PETSc vector (complex, distributed). Knot vectors are embedded as attributes at destruction time (rank 0, after the collective viewer is closed).

### 6.3 Autocorrelation

The `autocorrelation` dataset of `td/<name>_observables.h5` stores ⟨ψ(0)\|ψ(t)⟩ at every step.

### 6.4 t-SURFF Spectra (`td/<name>_surff.h5`)

Only with `TSurff = 1`. Datasets `momenta` (Nk x d), `amplitude` (Re, Im of b(k)), `pes` = |b(k)|^2 = dP/d^dk; on a grid also `k_x`/`k_y`/`k_z` and `pes_grid`; `amplitude_flux` with `SurffInterior = 1`. Attributes: `final_time`, `radius`, `exterior_mass`, `exterior_potential`, `charge`, `A_final`, `stride`, `n_surface_points`, `n_samples`, `interior`.

### 6.5 Eigenstate Data (`static/<name>_eigen.h5`)

**Datasets:**
- `spectrum` — eigenvalues (real, 1D array, size = NBoundStates)
- `psi_0`, `psi_1`, ... — eigenstate vectors (complex, if `NBoundStatesSave=1`)

**Attributes of `run_metadata` group:**
- `code_version` — Version string
- `input_file` — Input filename
- `units` — "energy: Hartree (a.u.); length: Bohr (a.u.); mass: electron mass"
- `Dimension`, `SplineDegree`, `Nelements` — Grid parameters
- `LMinX`, `LMaxX`, `LMinY`, `LMaxY`, `LMinZ`, `LMaxZ` — Domain bounds
- `Charge` — Carrier charge q (-1 electron, +1 hole)
- `TargetEigenvalue`, `NBoundStatesSave`, `NBoundStates` — Solver parameters
- `state_format` — "complex" or "real"
- `eig_residual` — SLEPc relative error estimate per eigenvalue

**Attributes of each axis dataset (`knots_x`, `knots_y`, `knots_z`):**
- `mult_start_end` — [start_multiplicity, end_multiplicity] (2-int)
- `SplineDegree` — B-spline degree (int)
- `nfuncs` — Number of basis functions (int) = knot_count - degree

### 6.6 Dipole Operator Save

PETSc binary files: `static/<name>_Dx.bin`, `static/<name>_Dy.bin`, `static/<name>_Dz.bin`. Load with:
```cpp
MatLoad(dx, "-Dx_input.bin");
```

---

## 7. Example Input Files

See the `inps/` directory for complete, validated examples.

### Example: 1D Harmonic Oscillator (no field)

A bound-state computation with no laser field, no propagation.

See `inps/harmonic_oscillator_1d.prm`.

### Example: 1D Atomic Ionization

Hydrogen-like atom in a Gaussian laser pulse. Computes ionization probability via population monitoring and dipole spectrum.

See `inps/atom_laser_1d.prm`.

### Example: 2D Circularly Polarized Field

Two-axis atom with circular polarization (Ex + iEy). Computes Berry phase, angular momentum, and degenerate state splitting.

See `inps/harmonic_2d_circular.prm`.

### Example: 3D Multi-Center Potential


See `inps/multi_center_3d.prm`.

### Example: Superposition Initial State

Creates a coherent superposition of the ground and first excited state with user-defined coefficients.

See `inps/superposition_1d.prm`.

---

## 8. Observable Reference

### Dipole moment d(t) = q⟨ψ(t)\|r\|ψ(t)⟩

Units: charge·length (e·a₀). The Fourier transform of d(t) gives the absorption spectrum.

### Energy decomposition

```
E_total(t) = ⟨ψ\|K\|ψ⟩ + ⟨ψ\|V\|ψ⟩ + ⟨ψ\|H⟩ + E(t)·d(t)
```

Where:
- K = -½∇²/m(r) — kinetic energy operator
- V — potential energy
- ⟨ψ\|H\|ψ⟩ — bare Hamiltonian expectation
- E(t)·d(t) — interaction energy (length gauge)

### Current decomposition

The velocity-gauge current J = ⟨ψ\|v\|ψ⟩ decomposes into:

1. **Intra-band**: Σ_n |c_n|² · v_nn — current within each occupied state
2. **Inter-band**: Σ_{m≠n} c_m* c_n · v_mn — current from state coherences
3. **Bound-continuum**: J_total - J_intra - J_inter — current from continuum components

This decomposition is useful for diagnosing whether ionization is driven by inter-band transitions (semiconductor-like) or by the field ejecting electrons directly (atomic-like).

### Autocorrelation A(t) = ⟨ψ(0)\|ψ(t)⟩

The overlap between initial and current state. |A(t)|² gives the survival probability. The Fourier transform of A(t) gives the energy spectrum.

### Berry phase

For circularly polarized fields, the accumulated geometric phase:
```
Γ(t) = Σ_steps -arg(Σ_n c_n*(t-dt) · c_n(t))
```

Accumulates in `manager.BerryPhase`.

---

## 9. Troubleshooting

### Common issues

**"Dimension must be 1, 2 or 3"**
- Set `Dimension = 1`, `2`, or `3` explicitly. Default is 1.

**"SplineDegree must be in [1,14]"**
- Set `SplineDegree` between 1 and 14. Default is 3.

**"Domain bound is not finite"**
- Check the `Domain` (or `LMinX`/`LMaxX`, ...) values. Must be finite numbers.

**"Initial state index >= number of converged eigenstates"**
- Increase `NBoundStates` to compute more eigenstates, or use a lower index in `InitialState`.

**Segmentation fault during propagation**
- Check that `OutputStrideTS > 0` or `OutputStrideWFS > 0` is set if you expect output.
- For multi-axis polarization, the buffer size was bumped to 16 temporary vectors (from 12) to prevent segfault when using xyz polarization.

**Silent all-zero autocorrelation**
- The autocorrelation buffer is filled BEFORE the per-step HDF5 flush. If `OutputStrideTS=1`, the flush happens at every step. This is correct in the current implementation (AC is computed first, then `recordedSteps++`, then flush). Verify your `OutputStrideAC` setting separately.

**8-10x slowdown with mpirun**
- Use `mpirun -np N --bind-to none` or `--oversubscribe`. OpenMPI's default `--bind-to core` pins each process to one physical core.

**"Unknown Mat type given: "**
- This happens when `-mat_type` or `-vec_type` is not properly handled by PetIGA. Use the CLI flags as-is; TDSEZ re-injects them as `-iga_mat_type` / `-iga_vec_type`.

**"Overlap-matrix norms: NOT POSITIVE"**
- An eigenvector has a non-positive norm ψᴴMψ. Check the eigensolver settings and the mesh. This check does not test current conservation; the length–velocity gauge check printed before propagation is the operator-level test of the BDD velocity.

**StrictInput fatal on unknown key**
- Either fix the typo in your input file, or set `StrictInput = 0` to tolerate unknown keys (not recommended for physics codes).

---

## 10. Physics Conventions

- **Units**: All quantities are in atomic units (Hartree). Energy in Hartree, length in Bohr (a₀), mass in electron mass (m_e), charge in elementary charge (e).
- **Hamiltonian**: H = -½∇²/m(r) + V(r) + CAP
- **TDSE (length gauge)**: i·∂ψ/∂t = H·ψ + E(t)·D·ψ
- **Residual assembly (Crank-Nicolson)**: F = iM·ψ̇ - H·ψ - E(t)·D·ψ
- **Mass matrix inner product**: ⟨φ\|M\|ψ⟩ used for all projections, populations, and norms
- **Dipole operator**: D_x = x (position operator)
- **Velocity operator**: V_x = -i/(2m(r)) (B·∂_x - ∂_x·B), anti-symmetric

---

## 11. Performance Notes

- **Constant mass is much faster**: Setting `MassIsConstant = 1` disables all finite-difference gradient computations of 1/m, which are the dominant cost in the variable-mass assembly path.
- **Output buffering**: every step is recorded; rows are appended to the HDF5 file at most every 2 s (at multiples of `OutputStrideTS`), every 5000 steps, when the buffer is full and at the end, so RAM stays bounded.
- **EnableCAP for large domains**: Without CAP, waves reflect off domain boundaries. With CAP=1, outgoing waves are absorbed, allowing smaller domains.
- **LTO**: Link-time optimization is enabled by default in Release builds. Disabling with `-DENABLE_LTO=OFF` speeds compilation but may reduce performance by 5-15%.
