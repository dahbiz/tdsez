# TDSE-Z code quality and release assessment

Assessment date: 2026-09-21.

## Release decision

**Do not release the full advertised feature set as numerically validated.**
The core harmonic-oscillator cases and selected CPU paths pass the checks below,
but three selectable knot modes are placeholders, GPU execution could not run
on this host, and the physical accuracy of CAP reflection,
multidimensional variable mass, and continuum postprocessing has
no representative reference calculation. A restricted release should state
these limits explicitly and exclude the pending knot modes from supported
specialized features.

## Findings by path

| Path | Evidence from this review | Release claim supported |
| --- | --- | --- |
| `interface`, `adaptive`, `adaptive_wf` | `src/core_knots.cpp` returns uniform interior knots; parameterized tests compare each saved vector with exact uniform spacing. `TDSEZExtractBreakpoints` returns no breakpoints. | Uniform fallback only. Interface alignment and adaptation are not implemented. |
| CAP | A short 1D propagation near the box boundary has finite, monotonically decreasing norm. | Qualitative absorption in this fixture only. Reflection error, strength calibration, higher dimensions, and laser-driven dynamics remain unverified. |
| Variable mass | The first three 1D oscillator eigenvalues for `m(x)=1+0.02x²` agree within 1% with an independent flux-form finite-difference discretization. A 2D spatially varying mass case is finite and agrees across one and two MPI ranks. | Static 1D reference accuracy plus 2D MPI consistency. The variable-mass Ehrenfest corrections, time propagation, and material-interface physics remain unverified. |
| Postprocessors | On the oscillator fixture, `tdmprocessor` gives a Hermitian 3×3 matrix, the allowed `0→1` x dipole agrees with `sqrt(2.5)`, the forbidden `0→2` dipole is small, and the regular and recombination CSV fields from `tdmselect` and `drecprocessor` are finite and consistent. | CPU bound-state and finite-output checks for this fixture. Continuum physical accuracy and GPU postprocessing still require external validation. |
| GPU | `EnableGPU=1` now configures PetIGA with `aijcusparse` and `cuda` before IGA creation. The startup check reaches PETSc's CUDA backend, but `nvidia-smi` cannot communicate with the driver and PETSc reports `cudaErrorMemoryAllocation`. | Backend selection is covered; numerical GPU execution and performance still require a working CUDA host. |

## Quality changes

- Removed automatic finite-math compilation flags, which let Release builds
  optimize away `std::isfinite` input guards.
- Corrected validation sampling to vary every active spatial coordinate and
  include corners. Expression errors now produce validation errors.
- Rejects trailing characters in initial-state indices and coefficients.
- Propagates solver and HDF5 write failures instead of reporting success.
- Added serial and MPI regressions for input errors, compressed HDF5 writes,
  variable-mass eigenvalues in 1D and 2D, CAP norm loss, knot fallbacks, and
  CPU postprocessors.
- Replaced inaccurate postprocessor commands and output claims in the guide.
- Writes generated test inputs to a temporary directory.

## Verification and limits

The default Release configuration built all targets with LTO enabled against
PETSc 3.24.3, SLEPc 3.24.0, PetIGA, and MPI. The serial parser unit test and
the complete MPI regression suite must pass on the final tree before release.
The complete command and test count are recorded in the release checklist.

Domain sampling is a preflight check; it cannot prove every point of a user
expression is finite or positive. No continuous-integration workflow is
present. The installed `clang-tidy` wrapper could not complete an analysis
because its Clang toolchain could not locate C++ standard headers; this report
does not claim static-analysis coverage. Numerical performance after removing
finite-math flags was not measured.
