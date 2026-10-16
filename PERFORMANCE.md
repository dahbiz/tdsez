# Performance optimisations in TDSE-Z

This page documents the optimisations of the propagation and output path, the
speed-ups measured for each, and why none of them changes the computed results
beyond rounding or the linear-solver tolerance.

All timings were measured on a workstation with 2–4 MPI ranks, using the
end-of-run **TIMING SUMMARY** printed by `tdsez.x` (wall time, maximum over
ranks). Speed-ups on a cluster are expected to be similar but have not been
re-measured at scale.

## Notation

The semi-discrete Schrödinger equation is

```math
i\,\mathbf M\,\dot{\mathbf c}(t) = \mathbf H(t)\,\mathbf c(t),
\qquad
\mathbf H(t) = \mathbf H_0 + \mathbf E(t)\cdot\mathbf D ,
```

with overlap matrix $`\mathbf M`$, field-free Hamiltonian $`\mathbf H_0`$
(kinetic $`\mathbf K`$ plus potential $`\mathbf V`$, plus the CAP when enabled) and
dipole matrices $`\mathbf D = (\mathbf D_x,\mathbf D_y,\mathbf D_z)`$.

## Summary

| Optimisation | Test case | Before | After | Gain |
|---|---|---|---|---|
| Preconditioner reuse, right preconditioning, assembly savings | 3D propagation | — | — | propagation $`2.5\times`$, run $`1.7`$–$`2.3\times`$ |
| Magnus-4 instead of Crank–Nicolson (better accuracy) | 1D driven HO | 11.4 s | 0.77 s | $`15\times`$ ($`\approx 70\times`$ at error $`10^{-6}`$) |
| Padé/Magnus stage solver | 3D HO, 200 Magnus steps | 249 s | 219 s | $`-12\%`$ |
| Third-order extrapolated guess, fused stage update (3) | 3D, $`p=3`$, 32k DOFs, `pade22`, 40 steps: time stepping | 3.40 s | 2.90 s | $`-15\%`$ |
| Monitor products + batched HDF5 writes | 1D HO, $`5\times10^4`$ steps | 15.8 s | 12.3 s | $`-22\%`$ |
| | 1D with CAP, $`2.2\times10^4`$ steps | 5.6 s | 4.1 s | $`-27\%`$ |
| | 3D variable mass, 100 steps | 3.42 s | 3.24 s | $`-5\%`$ |
| Sum-factorised assembly (BLAS contractions) | 3D, $`p=7`$, $`10^3`$ elements: $`\mathbf H,\mathbf M`$ | 114 s | 8.0 s | $`14\times`$ |
| | same, time-dependent operators | 464 s | 32 s | $`14\times`$ |
| | 3D $`z`$-polarisation, $`p=5`$: whole run | 62.6 s | 7.1 s | $`8.9\times`$ |
| Banded insertion | 3D, $`p=7`$, $`10^3`$ elements, 4 ranks: $`\mathbf H,\mathbf M`$ | 12.1 s | 2.1 s | $`5.9\times`$ |
| | same, time-dependent operators | 54.2 s | 6.5 s | $`8.4\times`$ |
| | 3D, $`p=7`$, $`20^3`$ elements, 4 ranks: $`\mathbf H,\mathbf M`$ | 258 s | 11.5 s | $`22\times`$ |
| Monitor expectations at the quadrature points (1e) | 3D varying mass, discrete acceleration, 50 steps: whole run | 142 s | 90 s | $`1.6\times`$ |
| | same, operator acceleration: monitors | 6.24 s | 4.43 s | $`-29\%`$ |
| Allocation-free $`1/m`$ derivatives (1e) | 3D, $`p=4`$, varying mass: force assembly | 28.1 s | 18.5 s | $`-34\%`$ |
| Eigensolver inner solves and fallback (1f) | 3D, $`p=7`$, 15.6k DOFs, 4 states | 49.0 s | 19.8 s | $`2.5\times`$ |
| `-tdsez_mat_free` (1g) | 3D, $`p=3`$, 43k DOFs, CAP, 50 steps: whole run | 11.5 s | 9.0 s | $`1.3\times`$ |
| | same: peak memory, all ranks | 11.4 GB | 4.6 GB | $`-60\%`$ |
| Discrete acceleration: CG/block-Jacobi mass solve instead of LU (1g) | same grid, discrete acceleration: whole run | 37.7 s | 16.5 s | $`2.3\times`$ |

## 1. Crank–Nicolson solver set-up

Crank–Nicolson solves at every step

```math
\Bigl(\mathbf M + \tfrac{i\Delta t}{2}\,\mathbf H^{n+1/2}\Bigr)\mathbf c^{n+1}
= \Bigl(\mathbf M - \tfrac{i\Delta t}{2}\,\mathbf H^{n+1/2}\Bigr)\mathbf c^{n},
\qquad
\mathbf H^{n+1/2} = \mathbf H\!\left(t_n + \tfrac{\Delta t}{2}\right),
```

with GMRES and a restricted additive-Schwarz preconditioner with ILU(0) subdomain
solves.

* **Preconditioner reuse.** The system matrix changes between steps only
  through the scalar field $`\mathbf E(t)`$, so the ILU factorisation of the first
  step remains an effective preconditioner. The SNES layer used to reset the
  KSP reuse flag at every step and refactor; it now builds the preconditioner
  once (`SNESSetLagPreconditioner(snes, -2)` with persistence).
* **Right preconditioning.** GMRES monitors the true residual, so the relative
  tolerance $`10^{-12}`$ bounds the actual Crank–Nicolson solve error,
  independently of the preconditioner quality.
* **Assembly.** Real-valued upper-triangle assembly of the symmetric operators,
  and no finite-difference mass derivatives when the discrete Ehrenfest
  acceleration is used.

Measured on 3D test problems: propagation $`2.5\times`$ faster, complete runs
$`1.7`$–$`2.3\times`$ faster.

* **ILU(0) subdomains.** Subdomains above 4000 rows used ILU(1). In 3D, degree
  4 ($`110\,592`$ DOFs, 8 ranks), ILU(0) needs the same 7 GMRES iterations, its
  set-up takes 4.6 s instead of 45.7 s, the step 4.5 s instead of 11.4 s, and the
  dipole time series agree to $`8\times10^{-15}`$. In 2D ILU(1) saved at most one
  iteration and was 10–20% slower per step. ILU(0) is the default;
  `-tdsez_ilu_levels n` restores another level.
* **Constant mass detected.** A `Mass` expression without `x`, `y`, `z` now sets
  `MassIsConstant` automatically. Without it, the multi-axis force kernels
  computed finite-difference derivatives of $`1/m`$ (about 260 expression
  evaluations per quadrature point and axis): the 3D degree-4 benchmark spent
  more than 10 minutes per rank in the time-dependent operator assembly, 19 s
  with the detection. The derivatives of a constant are exactly zero, so the
  operators are unchanged.

## 1b. Preconditioner refresh and geometric multigrid

* **Refresh.** The preconditioner is built once and reused. With a strong field
  ($`E = 2`$ a.u., $`\Delta t = 0.2`$, 2D) the operator drifts away from the one it
  was built from: 4, 13, 24, 65, 249, 641 GMRES iterations in successive steps,
  then failure (ASM and multigrid alike). A solve that needs more than twice the
  iterations of the first solve after the last build (plus 5) now triggers a
  rebuild at the next step; the run converges with 4–8 rebuilds. Ordinary runs
  never reach the threshold.
* **Geometric multigrid** (`Preconditioner = mg`): nested coarse B-spline spaces
  (every other simple knot dropped), exact knot-insertion prolongation, Galerkin
  coarse operators, ASM/ILU(0) Richardson smoothing, MUMPS on the coarsest
  level, flexible GMRES outside. Same results as ASM to $`10^{-12}`$.

| Case (ranks) | ASM its | MG its | MG / ASM time per step |
|---|---|---|---|
| 2D $`p=5`$, 40k DOFs, CN $`\Delta t=0.02`$ (1, 4, 8, 12) | 3–7 | 2–4 | 1.4–1.8 |
| same, $`\Delta t=0.2`$ | 4–11 | 2–6 | 1.3–1.7 |
| 2D variable mass / Coulomb / no CAP (8) | 6–13 | 3–5 | 1.1–1.5 |
| 3D $`p=3`$, 110k DOFs, CN (4, 8) | 6–8 | 4 | 1.2 |
| 2D field: $`\tau\max\lvert E\cdot r\rvert`$ = 0.075–1.5 (8) | 5–20 | 3–18 | 1.4–1.7 |
| 2D field: $`\tau\max\lvert E\cdot r\rvert`$ = 3 (8) | 5–33 | diverges | — |

  SOR smoothing was tried first and failed the degree test: 14–26 iterations on
  one rank, 60–80 on four ($`p=5`$). On a single 12-core machine multigrid does
  not pay off yet; it is meant for large rank counts where one-level ASM, having
  no coarse space, degrades. ASM stays the default.

## 1c. Algebraic multigrid, memory, CAP and variable-mass assembly

* **GAMG** (`Preconditioner = gamg`): with PETSc's default Chebyshev/Jacobi
  smoothing 218–279 iterations (2D) — unusable on this complex operator. With
  the ASM/ILU(0) smoother: 3D $`p=3`$, 110k DOFs, 8 ranks, 40 CN steps:

  | | its | set-up | solve / step | wall / step |
  |---|---|---|---|---|
  | asm | 7.0 | 1.1 s | 0.089 s | 0.835 s |
  | mg | 4.0 | 1.7 s | 0.158 s | 0.982 s |
  | gamg | 4.0 | 5.7 s | 0.134 s | 1.069 s |

  No speed-up on one node; see `benchmarks/preconditioners` for the cluster test.
* **Memory.** The 3D observables ran 6× slower per mat-vec than the propagation
  (5.3 against 33 GFlop/s, same flops) because the run was paging: about 15
  stored operators at 114 MB per rank each. The unused copy `Ht` of H is gone,
  the Crank–Nicolson Jacobian is only created for `cn`, and the monitor frees
  `K` (no CAP: $`K\psi = H\psi - V\psi`$) and `Md` (constant mass: $`M/m`$) after
  checking the identities on the first step. 3D benchmark: 1797 → 1642 MB per
  rank, 0.835 → 0.746 s per step, outputs identical to $`10^{-15}`$.
* **Separable CAP.** $`W(\mathbf r)=\sum_d W_d(x_d)`$ on a tensor rule, so the CAP
  matrix is $`-i\sum_d W_d\otimes\prod_{e\ne d}M_e`$ of 1D matrices built from
  PetIGA's 1D quadrature tables, filled row by row (no element loop over the
  free core). 3D, 8 ranks: 3.30 s → 0.053 s; equal to the element assembly to
  $`7\times10^{-16}`$ (`-tdsez_cap_check`).
* **Variable-mass assembly.** With a position-dependent mass the
  time-dependent operators were built point by point: twelve dense element
  matrices accumulated at every quadrature point, plus finite-difference
  derivatives of $`1/m`$ for the mass-gradient corrections of the force
  matrices. Those corrections are used only by the operator-form acceleration;
  with the discrete form (the default for a varying mass) the force matrices are
  never read. In that case the sum-factorised path, which already takes $`1/m`$
  at every quadrature point for K, Md and the velocities, is now used, and the
  unused force matrices are freed. The remaining per-point kernels use real
  arithmetic (bit-identical) and share one memoised derivative evaluation per
  point. 3D variable mass, $`p=4`$, 16$`^3`$ elements, 8 ranks: operator assembly
  68.6 s → 2.1 s, run 73.6 s → 6.1 s; element matrices equal to the per-point
  kernels to $`2\times10^{-15}`$ (`-tdsez_sumfact_check`), outputs equal to
  run-to-run noise.

## 1c'. Force corrections for a position-dependent mass

With `AccelerationMethod = operator` and a varying mass the force matrices
carry mass-gradient corrections. Every term has the form
$`\int c(\mathbf r)\,D^\alpha B_i\,D^\beta B_j`$ with at most first derivatives
per axis, so it is sum-factorised like the other operators (ten coefficient
arrays per axis: $`(B,B)`$, $`(B,\partial_e B)`$, $`(B,\partial_a\partial_e B)`$,
$`(\partial_e B,\partial_e B)`$; general, non-symmetric storage). Element
matrices equal the per-point kernels to $`4\times10^{-15}`$ in 2D and 3D. 3D,
$`p=3`$, 16$`^3`$ elements, 8 ranks: operator assembly 34.3 s → 3.9 s, run
34.9 s → 4.5 s, outputs equal to $`6\times10^{-15}`$.

`TDSEZFormPhyX` is the 1D kernel; its unused 2D/3D branches and the matching
sum-factorised path were removed (2D/3D always use `TDSEZFormPhyAxes`). The
corrections are skipped in every dimension when the discrete acceleration is
used (the force matrices are then not read).

## 1d. Factorised monitor operators

On a tensor B-spline basis with a tensor Gauss rule the operators the monitors
apply every step are Kronecker products of 1D banded matrices: $`M`$, the
dipoles $`D_a`$, and for a constant mass the velocities
$`-\tfrac{i}{2m}(G-G^T)_a`$ and $`K=\tfrac1{2m}\sum_a S_a`$ (1D mass, position,
first-derivative and stiffness matrices from PetIGA's 1D quadrature tables).
They are applied with one halo exchange of $`\psi`$ per step and three 1D band
contractions per term (about $`6d(2p+1)`$ flops per coefficient instead of
$`8(2p+1)^d`$, no matrix traffic). Without a CAP, $`H\psi=K\psi+V\psi`$ also
drops the sparse product with $`H`$.

Each operator is checked against its assembled matrix on the state at the first
monitor call (agreement to $`10^{-10}`$ relative required; observed
$`2\text{–}5\times10^{-15}`$); the matrices then read by nothing else are freed
(velocities, dipoles of uncoupled axes, $`K`$). Dirichlet walls (identity rows
in the assembled matrices) keep the assembled path; `-tdsez_tensor_ops 0`
switches the factorised path off.

| 3D, $`p=3`$, 110k DOFs, 8 ranks, 40 CN steps | observables | wall / step |
|---|---|---|
| x-polarised, before | 26.1 s | 0.80 s |
| x-polarised, factorised | 5.2 s | 0.30 s |
| xyz-polarised, before | 25.8 s | 0.87 s |
| xyz-polarised, factorised | 12.8 s | 0.54 s |

Outputs (dipoles, energies, currents) equal to $`4\times10^{-15}`$. The
products with $`V`$ and the force matrices are replaced in 1e. With xyz
polarisation the propagator also keeps all three dipole matrices.

## 1e. Monitor expectation values from the quadrature points

Every monitor operator is $`A_{ij}=\sum_q c(q)\,D^\alpha B_i(q)\,D^\beta B_j(q)`$
on the assembly quadrature, so
$`\langle\psi|A|\psi\rangle=\sum_q c(q)\,\overline{D^\alpha\psi(q)}\,D^\beta\psi(q)`$
is the same number without the matrix. $`\psi`$ and its derivatives (first
order per axis; the mixed second derivatives for the mass-gradient force
corrections) are evaluated element by element by sum factorisation from the
ghosted coefficients, and the sums are reduced once per monitor call
(`src/quadobs.cpp`). This gives $`\langle V\rangle`$, $`\langle 1/m\rangle`$,
$`\langle K\rangle`$ for a varying mass, the CAP, the forces
$`\langle\partial_aV\rangle`$ with their corrections, the velocities for a
varying mass, and $`w^H\mathrm{Vel}_a\psi`$ for the discrete acceleration.
Each value is checked against its assembled matrix on the state at the first
monitor call (agreement to $`10^{-10}`$ relative required; observed
$`10^{-15}`$), then $`V`$, $`\partial_aV`$ and $`\mathrm{Vel}_a`$ are freed.
Dirichlet walls and 1D keep the matrices; `-tdsez_quad_obs 0` switches the path
off.

The per-point coefficients of the force corrections are the ones the
sum-factorised assembly has just computed; it keeps them per owned element and
the monitor takes them over (with `-tdsez_sumfact 0` the monitor tabulates them
itself). The point sums accumulate per point and are summed once per call: with
`-fno-fast-math` a running scalar sum is a serial dependency chain the compiler
may not vectorise.

The $`1/m`$ derivatives of the force corrections (finite-difference stencil up
to third order, 264 evaluations per point in 3D) are computed without
allocations: the same stencil points, the same arithmetic and the same
de-duplication as before, so the values are bitwise identical.

| 3D, $`p=4`$, $`28^3`$ splines, 8 ranks, 50 steps | monitors, matrices | monitors, quadrature | run |
|---|---|---|---|
| varying mass, operator acceleration | 6.24 s | 4.43 s | 66.1 → 61.0 s |
| varying mass, discrete acceleration | 81.1 s | 40.0 s | 142 → 90 s |

| same case, 20 steps | before | after |
|---|---|---|
| TD operator assembly (forces with corrections) | 28.1 s | 18.5 s |

Outputs equal to the matrix path to $`8\times10^{-15}`$ (relative). Peak memory
does not change: it is reached during the assembly, before the matrices are
freed.

## 1f. Eigensolver (shift-invert inner solves, validation, fallback)

The previous set-up passed its preconditioner settings as unprefixed global
options (`-sub_pc_factor_levels`, `-sub_pc_factor_mat_ordering_type rcm`, ...).
SLEPc's shift-invert KSP has the prefix `st_`, so the eigensolver ran plain
ILU(0) on ASM blocks with overlap 2 (100k–1M DOFs) or block Jacobi (< 10k), and
the options leaked to the propagator's ASM. A non-converged inner solve
aborted the run, and the inner tolerance ($`10^{-10}`$) limited the accuracy of
states near $`E=0`$ while SLEPc reported convergence at $`10^{-12}`$.

Now (see Eigensolver in the documentation): direct LU in 1D/2D and for small
3D problems of degree $`\ge 6`$, FGMRES + restricted ASM(1)/ILU(0) otherwise;
every attempt checked (inner solve, number of converged pairs, backward error
$`\le 10^{-8}`$) with fallback iterative → direct → ASM(2)/ILU(1). The
degenerate-shell check re-solves at full tolerance with the deflation space set
again (it does not persist between `EPSSolve` calls; before, the loose
$`10^{-6}`$ probe was kept, backward error $`3\times10^{-9}`$ instead of
$`10^{-15}`$).

8 ranks, eigensolve wall time (eigenvalues identical to the previous build):

| Case | DOFs | before | after |
|---|---|---|---|
| 3D HO, $`p=7`$, 4 states | 15.6k | 49.0 s | 19.8 s (direct) |
| 3D varying mass, $`p=4`$, 1 state | 32.8k | 5.1 s | 3.9 s |
| 3D HO / soft-core H, $`p=3`$ (back to back) | 64k / 46k | 16.2 / 10.8 s | 14.0 / 8.8 s |
| 2D soft-core H, Dirichlet, 12 states | 7.7k | 1.74 s | 0.58 s |
| 2D soft-core H, $`p=4`$, 10 states | 254k | 64 s (iterative) | 34 s (direct, 3× memory) |
| 2D HO / soft-core H, 30 / 15 states | 13k / 21k | 1.07 / 1.7 s | 1.6 / 2.2 s |
| 1D soft-core H, 30 states | 3k | 0.22 s | 0.25 s |
| 3D $`p=7`$, ASM overlap 0 forced | 15.6k | abort (KSP diverged) | 25 s (fallback to direct) |

Small 2D problems are somewhat slower with the exact solve; in exchange every
state is exact to rounding (1D Rydberg states: relative residual
$`1.6\times10^{-8}`$ before, backward error $`6\times10^{-15}`$ now).
`-tdsez_eps_inner iterative` restores the iterative inner solve.

## 1g. Matrix-free monitors and the mass solve of the discrete acceleration

`-tdsez_mat_free` (2D/3D, no Dirichlet walls, B-spline basis, CPU) skips the
assembly of everything only the monitors and the bound-state diagnostics read:
$`\mathbf K`$, $`\mathbf V`$, $`\mathbf M_{1/m}`$, the forces, the dipoles of
uncoupled axes and, for a constant mass, the velocities. The factorised (1d)
and quadrature (1e) forms are used without the check against a matrix. Only
the coupled dipoles (and for a varying mass the velocities) are assembled, in
one sum-factorised pass (`TDSEZFormMatFreeAxes`). The Padé/Magnus stage
matrices are formed as $`(g_k\mathbf H + \mathbf M) + g_k s\,\mathbf E\cdot\mathbf D`$
in one pass, the same values in the same order of operations as from the
cached $`\mathbf M + g_k\mathbf H`$, which are therefore not kept (two full
matrices less). Outputs equal the default run to $`10^{-12}`$ (2D/3D, CAP,
varying mass, discrete acceleration, Magnus-4; `tests/test_tensorops.py`).

Whole run and peak memory of all ranks; 3D, $`p=3`$, 42 875 DOFs, CAP,
`pade22`, 50 steps, 4 ranks (no swapping in any run):

| mass, acceleration | old LU mass solve | default | `-tdsez_mat_free` |
|---|---|---|---|
| constant, operator | – | 11.5 s, 11.4 GB | 9.0 s, 4.6 GB |
| constant, discrete | 37.7 s, 16.6 GB | 16.5 s, 11.3 GB | 13.9 s, 4.7 GB |
| varying, operator | – | 16.6 s, 12.4 GB | 14.5 s, 7.5 GB |
| varying, discrete | 40.1 s, 17.1 GB | 18.4 s, 12.5 GB | 16.1 s, 6.9 GB |

The time per step is unchanged by the flag (the stage solves use the same
assembled matrices): the operator assembly falls from 3.2 s (6.3 s for a
varying mass) to 0.4 s (1.0 s) and the memory by 40-60 %. In 2D (41 209 DOFs,
200 steps) the run times are unchanged and the memory falls by 30-45 %. For a
varying mass the first monitor call tabulates the mass-gradient force
coefficients itself, a one-time cost of a few seconds in 3D. On a 91 125-DOF
hydrogen grid (Becke-masked Coulomb, t-SURFF) the peak memory falls from 11.0
to 8.8 GB; there the rest is $`\mathbf H`$, $`\mathbf M`$, the two stage
matrices and their ASM/ILU preconditioners.

Every configuration was also run with and without the flag and all HDF5
outputs and saved dipole matrices compared (2D/3D; `cn`, `pade22`, `magnus4`;
operator and discrete acceleration; constant and varying mass; CAP; x, xy, z,
xyz polarisation; `ObserveAllAxes = 0`; t-SURFF; multigrid; `EnableLzDiag`;
`SaveDipoleAxes`): equal to $`10^{-14}`$. 1D, Dirichlet walls and the
Bernstein basis warn and run as without the flag. The test suite also passes
with the flag forced on for every run (`PETSC_OPTIONS=-tdsez_mat_free`).

The discrete acceleration solves $`\mathbf M\mathbf w = \mathbf r`$ at every
monitor call. With a MUMPS LU this was a full triangular solve on a factor with
3D fill (5.5 s per call on the grid above, 116 s for 21 calls, and a larger
peak memory). $`\mathbf M`$ is Hermitian positive definite, so conjugate
gradients with a block-Jacobi ILU(0) preconditioner (symmetric for a symmetric
$`\mathbf M`$) converges to $`10^{-14}`$ relative in about 16 iterations:
5.4 s for the same 21 calls ($`21\times`$), accelerations equal to the LU
result to $`2\times10^{-12}`$. The LU stays available with
`-accel_ksp_type preonly -accel_pc_type lu`.

## 1h. Propagation solver: alternatives measured and not adopted

**Preconditioners.** Every `PCType` PETSc and PetIGA register was tried on the
`pade22` stage solves (8 ranks): restricted ASM with ILU(0) stays the fastest.
3D $`p=3`$, 32k DOFs, 40 steps / 2D $`p=4`$, 14k DOFs, 100 steps:
`asm` 4.5 s, 5.5 its / 0.65 s, 5.0 its; `bjacobi`, `gasm`, `bjacobi`+ICC
9.0–9.3 s, 25 its / 1.5–1.6 s, 19 its; `sor` 40 s / 6.8 s; `jacobi` 50 s / 9.5 s;
PetIGA `igaebe` 69 s, 222 its / 7.2 s, 102 its; PetIGA `igabbb` did not converge
(also with maximal overlap). Both PetIGA preconditioners invert small dense
element or basis-function blocks: smoothers without global coupling. ASM with
overlap 2 needs fewer iterations in 2D (3.3) but is not faster.

**Accuracy** is set by the right-preconditioned GMRES tolerance, not by the
preconditioner: against an exact MUMPS LU solve (2D, 100 steps) the ASM run
differs by at most $`1.3\times10^{-11}`$ (dipoles), block Jacobi by
$`2\times10^{-11}`$; norm drift $`8\times10^{-14}`$.

**Load imbalance.** PetIGA gives the last rank of each axis the $`p`$ trailing
basis functions (3D, 32 cubic elements per axis on 2×2×2 ranks: 16³ vs 19³
owned rows, ASM blocks 19³–22³, ILU time ratio 1.6). The element count can
shift one element per axis only. Rebalancing the element cuts after
`IGASetUp_Stage1` made time stepping 7–12 % faster but the element-based
monitors and assembly slower; the run changed by 0–5 % (noise level), so it
was reverted. Details in the MPI Notes of the documentation.

**Pipelined GMRES** (`-pade_stage{1,2}_ksp_type pgmres`, `MPI_Iallreduce`
overlapped with the next product): 20–35 % slower on one machine (3D, 8 ranks,
same 4.1 its/solve); reductions are cheap in shared memory. Possibly useful
across many nodes.

## 2. Fourth-order propagators (`Propagator = pade22 | magnus4`)

Both use the $`(2,2)`$ Padé approximant of the step exponential,

```math
R_{22}(z) = \frac{1 + z/2 + z^2/12}{1 - z/2 + z^2/12},
\qquad z = -i\,h\,\mathbf M^{-1}\mathbf H_{\mathrm E},
```

which is fourth order and unitary for Hermitian $`\mathbf H_{\mathrm E}`$. Its
numerator has the roots $`z_\pm = -3 \pm i\sqrt3`$, so $`R_{22}`$ factorises into
two Cayley-type stages

```math
\bigl(\mathbf M + g_k\,\mathbf H_{\mathrm E}\bigr)\,\mathbf y
= \bigl(\mathbf M + \overline{g_k}\,\mathbf H_{\mathrm E}\bigr)\,\mathbf x,
\qquad
g_{1,2} = \frac{h\,(\mp\sqrt3 + 3i)}{12}.
```

`magnus4` is the commutator-free fourth-order Magnus scheme (CF4, Blanes and
Moan 2006),

```math
U = \exp\!\bigl[h(a_1A_1 + a_2A_2)\bigr]\,\exp\!\bigl[h(a_2A_1 + a_1A_2)\bigr],
\qquad
A_j = -i\,\mathbf M^{-1}\mathbf H(t_n + c_j h),
```

with Gauss nodes $`c_{1,2} = \tfrac12 \mp \tfrac{\sqrt3}{6}`$ and
$`a_{1,2} = (3 \mp 2\sqrt3)/12`$. Because $`\mathbf H(t)`$ is linear in the field
and $`a_1 + a_2 = \tfrac12`$, each factor is an ordinary half step $`h/2`$ with the
effective field $`2(a_1\mathbf E_1 + a_2\mathbf E_2)`$ (respectively
$`2(a_2\mathbf E_1 + a_1\mathbf E_2)`$); no commutator is ever formed.

Accuracy versus cost for a strongly driven 1D harmonic oscillator,
$`E(t) = \cos(0.6\,t)`$, $`T = 50`$ ($`p = 7`$, 1200 elements on $`[-60, 60]`$, 2 ranks),
against the exact solution $`\langle x\rangle(t) = -\dfrac{E_0}{1-\omega^2}\bigl(\cos\omega t - \cos t\bigr)`$
with $`E_0 = 1`$, $`\omega = 0.6`$:

| Propagator | $`\Delta t`$ | Solves per step | $`\max_t \lvert\langle x\rangle - x_{\mathrm{ex}}\rvert`$ | Propagation |
|---|---|---|---|---|
| Crank–Nicolson | $`10^{-3}`$ | 1 | $`1.7\times10^{-4}`$ | 11.4 s |
| `pade22` | $`0.02`$ | 2 | $`2.5\times10^{-5}`$ | 0.91 s |
| `magnus4` | $`0.05`$ | 4 | $`4.3\times10^{-5}`$ | 0.77 s |

![Dipole of the driven oscillator for the three propagators](docs/perf/propagator_dipoles.png)

*(a) Dipole $`\langle x\rangle(t)`$ from the `dipoles` dataset of each run's HDF5 output, on top of the exact solution. (b) Absolute error; the legend gives the maximal error and the propagation wall time.*

Observed orders of accuracy: $`2.00`$ (Crank–Nicolson), $`3.97`$–$`4.00`$ (`magnus4`);
`pade22` is fourth order in the field-free phase but second order in the field
variation. For an error of $`10^{-6}`$, extrapolating the second-order behaviour
of Crank–Nicolson gives a cost ratio of about $`70`$ relative to `magnus4`.

The figure is reproduced by `docs/perf/run.sh` (inputs `docs/perf/ho1d_*.inp`,
plot script `docs/perf/plot_propagator_dipoles.py`).

## 3. Padé/Magnus stage solver

**Cached field-free parts.** The stage matrices are

```math
\mathbf A_k = \mathbf M + g_k\,\mathbf H_{\mathrm E}
= \underbrace{\mathbf M + g_k\,\mathbf H_0}_{\mathbf B_k}
+ g_k\,\mathbf E\cdot\mathbf D .
```

$`\mathbf B_k`$ depends only on the step length $`h`$ (and $`\mathbf H_0`$ is fixed
during propagation, including the CAP), so it is computed once per step length.
Each stage update is then one copy and one field term.

**Paired right-hand sides.** Since $`\overline{g_1} = -g_2`$,

```math
\bigl(\mathbf M + \overline{g_1}\,\mathbf H_{\mathrm E}\bigr)\mathbf x
= 2\,\mathbf M\mathbf x - \mathbf A_2\,\mathbf x,
\qquad
\bigl(\mathbf M + \overline{g_2}\,\mathbf H_{\mathrm E}\bigr)\mathbf x
= 2\,\mathbf M\mathbf x - \mathbf A_1\,\mathbf x ,
```

so both stage matrices are built once per $`R_{22}`$ application and each
right-hand side needs two sparse products ($`\mathbf M`$ and $`\mathbf A_{\text{other}}`$)
instead of three ($`\mathbf H_0`$, $`\mathbf D`$, $`\mathbf M`$).

**Extrapolated initial guess.** Each stage solve starts from the stage input
plus the increment $`\boldsymbol\delta = \mathbf y - \mathbf x`$ of the same
stage extrapolated from the last $`q`$ steps (polynomial of degree $`q-1`$),

```math
\mathbf y_0^{(n)} = \mathbf x^{(n)} + \sum_{j=1}^{q} (-1)^{j-1}\binom{q}{j}\,\boldsymbol\delta^{(n-j)},
```

so the initial residual is $`O(h^q)`$. $`q = 3`$ by default
(`-pade_guess_order 1..5`; $`q=1`$ was the previous behaviour); orders above 3
gain little. `-pade_extrapolate_guess 0` starts from $`\mathbf x`$ alone.

**Fused stage update.** $`\mathbf A_k = \mathbf B_k + g_k \mathbf E\cdot\mathbf D`$
is formed in one pass over the stored values instead of a copy and one
`MatAXPY` per field axis; the matrices are checked once to share the exact
sparsity structure (else the copy/AXPY path is used).

| 8 ranks, `pade22`, $`\Delta t=0.05`$ | time stepping | GMRES its/solve | operator updates |
|---|---|---|---|
| 3D soft-core H, $`p=3`$, 32k DOFs, 40 steps, xy field: before | 3.40 s | 5.51 | 0.61 s |
| same, after | 2.90 s | 4.11 | 0.35 s |
| 2D soft-core H, $`p=4`$, 14k DOFs, 100 steps: before | 0.52 s | 5.01 | 0.072 s |
| same, after | 0.41 s | 3.38 | 0.050 s |

Outputs agree with the previous build to the solve tolerance (dipoles
$`\le 9\times10^{-11}`$ relative, other columns $`\le 7\times10^{-12}`$).
The solves stop at $`10^{-12}`$ relative; `-pade_stage{1,2}_ksp_rtol 1e-11`
saves another 10–13 % (differences $`5\times10^{-11}`$).

3D harmonic oscillator, $`19\,683`$ DoFs, `magnus4`, $`\Delta t = 0.05`$, 200 steps,
4 ranks:

| | Before | Cached + paired | + extrapolated guess |
|---|---|---|---|
| Propagation | 249 s | 252 s | **219 s** |
| GMRES iterations per solve | 5.98 | 5.98 | **5.07** |
| PC applications per solve | 7.0 | 7.0 | **6.1** |
| Operator updates | 800 (10.2 s) | 400 (6.3 s) | 400 (5.4 s) |

## 4. Monitors and output

The observables ($`\langle H\rangle`$, energies, $`\langle x\rangle`$, acceleration,
populations, currents) are still computed and written at **every** step.

**Verified product identities.** Without a CAP, $`\mathbf H_0 = \mathbf K + \mathbf V`$,
and for a constant mass the inverse-mass matrix is
$`\mathbf M_{m^{-1}} = m^{-1}\mathbf M`$. On the first monitor call both
identities are checked on the current state,

```math
\frac{\lVert \mathbf H_0\boldsymbol\psi - \mathbf K\boldsymbol\psi - \mathbf V\boldsymbol\psi\rVert}
     {\lVert \mathbf H_0\boldsymbol\psi\rVert} \le 10^{-12},
\qquad
\frac{\lVert \mathbf M_{m^{-1}}\boldsymbol\psi - c\,\mathbf M\boldsymbol\psi\rVert}
     {\lVert \mathbf M_{m^{-1}}\boldsymbol\psi\rVert} \le 10^{-12},
\quad
c = \frac{\boldsymbol\psi^\dagger \mathbf M_{m^{-1}}\boldsymbol\psi}{\boldsymbol\psi^\dagger \mathbf M\boldsymbol\psi},
```

and only the identities that hold are used afterwards, replacing up to two
sparse products per step by vector operations. With a CAP or a variable mass
the full products are kept automatically.

**Batched HDF5 writes.** Every step is recorded in the ring buffer. The buffer
is written at multiples of `OutputStrideTS`, but at most once every 2 s of wall
time, and always every `FLUSH_INTERVAL` $`= 5000`$ steps and at the end of the
run. With `OutputStrideTS = 1` this replaces one HDF5 write per step by a few
per run; the file contents and dataset shapes are identical.

| Case | Monitor | HDF5 writes | Propagation |
|---|---|---|---|
| 1D HO, $`5\times10^4`$ steps | 6.04 → 2.31 s | 50 001 → 10 (3.28 → 0.01 s) | 15.8 → 12.3 s |
| 1D with CAP, $`2.2\times10^4`$ steps | 2.23 → 0.76 s | 22 051 → 4 (1.37 → 0.003 s) | 5.58 → 4.05 s |
| 3D variable mass, 100 steps | 1.00 → 0.86 s | 101 → 2 | 3.42 → 3.24 s |

## 5. Sum-factorised assembly

The element matrices are integrals of the form

```math
E_{ab} = \sum_{q} c(\mathbf r_q)\, D^{\alpha} N_a(\mathbf r_q)\, D^{\beta} N_b(\mathbf r_q),
```

where $`c`$ contains the quadrature weight and the coefficient
($`\tfrac{1}{2m}`$, $`V`$, $`m^{-1}`$, $`x_d`$, $`\partial_d V/m`$, ...) and
$`D^\alpha, D^\beta`$ are values or first derivatives. The B-spline basis and the
Gauss rule are tensor products,

```math
N_a(\mathbf r) = N_{a_1}(x)\,N_{a_2}(y)\,N_{a_3}(z),
\qquad
\mathbf r_q = (x_{q_1}, y_{q_2}, z_{q_3}),
```

so the sum over the $`Q^3`$ points can be contracted one axis at a time:

```math
\begin{aligned}
T_1[a_1 b_1](q_2,q_3) &= \sum_{q_1} N^{\alpha_1}_{a_1}(x_{q_1})\,N^{\beta_1}_{b_1}(x_{q_1})\;c(q_1,q_2,q_3),\\
T_2[a_1 b_1 a_2 b_2](q_3) &= \sum_{q_2} N^{\alpha_2}_{a_2}(y_{q_2})\,N^{\beta_2}_{b_2}(y_{q_2})\;T_1[a_1 b_1](q_2,q_3),\\
E_{ab} &= \sum_{q_3} N^{\alpha_3}_{a_3}(z_{q_3})\,N^{\beta_3}_{b_3}(z_{q_3})\;T_2[a_1 b_1 a_2 b_2](q_3).
\end{aligned}
```

With $`n = p+1`$ functions and $`Q`$ points per axis, the work per element and
term drops from $`Q^3 n^6`$ to $`n^6 Q + n^4 Q^2 + n^2 Q^3`$ in 3D (from
$`Q^2 n^4`$ to $`n^4 Q + n^2 Q^2`$ in 2D). The same quadrature points and the same
pointwise coefficients are used, so the element matrices are identical up to
rounding; symmetric matrices are symmetrised exactly and the velocity matrix is
built as $`i(\mathbf E - \mathbf E^{\mathsf T})`$, exactly antisymmetric.

Applies to $`\mathbf H, \mathbf M`$ and to the time-dependent operators
$`\mathbf K, \mathbf V, \mathbf M_{m^{-1}}, \mathbf D_d, \mathbf{Vel}_d, \partial_d V`$
for every polarisation, in 2D and 3D, including the mass-gradient corrections
of the force operator (section 1c'). The $`y`$/$`z`$/multi-axis
polarisations are now assembled in one batched pass instead of six separate
per-operator passes. Options: `-tdsez_sumfact 0` disables it;
`-tdsez_sumfact_check` computes every element both ways and prints the largest
relative difference ($`\le 5\times10^{-15}`$ in all tests).

**Matrix products (BLAS).** Each axis contraction is a dense matrix product
and is evaluated with the BLAS that PETSc was configured with (through PETSc's
`BLASREALgemm_` macro, so OpenBLAS, MKL, BLIS, ESSL, Cray LibSci, 32- or 64-bit
BLAS integers all work; CMake links PETSc's `BLASLAPACK_LIB`). In 3D, with
$`m = n^2`$ products per axis,

```math
\begin{aligned}
\mathbf T_1 &= \mathbf P_1\,\mathbf C^{\mathsf T} && (m \times Q^2,\ \text{inner } Q),\\
\mathbf T_2 &= \mathbf T_1\,\mathbf P_2^{\mathsf T} && (mQ \times m,\ \text{inner } Q),\\
\mathbf E &= \mathbf V\,\mathbf P_3^{\mathsf T} && (m^2 \times m,\ \text{inner } Q),
\end{aligned}
```

where $`\mathbf V`$ is $`\mathbf T_2`$ with the $`q_3`$ and $`a_2b_2`$ indices
swapped. All terms of one operator (the three kinetic terms and, for
$`\mathbf H`$, the potential) accumulate into the same real buffer in the last
product ($`\beta = 1`$), and the buffer is written into the complex element
matrix once per operator, symmetrised ($`\tfrac12(\mathbf E+\mathbf E^{\mathsf T})`$)
or, for the velocity, antisymmetrised ($`\mathbf E-\mathbf E^{\mathsf T}`$) in
the same pass. During the element loops BLAS is set to one thread
(`PetscBLASSetNumThreads`, PETSc $`\ge 3.21`$) and restored afterwards, so a
threaded BLAS does not start one thread per core in every MPI rank.

**Element work only** ($`\mathbf H, \mathbf M`$, one rank, time per element):

| $`p`$ | per-point kernels | sum factorised, loops | sum factorised, BLAS | total gain | flop ratio |
|---|---|---|---|---|---|
| 5 | 7.1 ms | 0.44 ms | 0.18 ms | 39× | 23× |
| 7 | 75.6 ms | 4.5 ms | 1.27 ms | 60× | 38× |

The per-point kernels scale as $`Q^3 n^6 \sim p^9`$ (measured growth $`10.7`$
from $`p=5`$ to $`7`$, predicted $`11.0`$). The gain exceeds the flop ratio
because the matrix products run near the floating-point peak while the
per-point kernels are limited by memory traffic; the remaining element time is
mostly the symmetrised write into the complex element matrix
($`2n^6`$ reads per operator).

**Whole assembly phases** (3D harmonic oscillator, $`10^3`$ elements, 4 ranks):

| Case | $`\mathbf H,\mathbf M`$ | Time-dependent operators | Whole run |
|---|---|---|---|
| $`x`$-pol., $`p=3`$ | 0.12 → 0.08 s | 0.52 → 0.32 s | 1.04 → 0.80 s |
| $`x`$-pol., $`p=5`$ | 2.89 → 1.00 s | 9.75 → 4.07 s | 14.6 → 7.0 s |
| $`x`$-pol., $`p=7`$ | 114 → 8.5 s | 464 → 36 s | |
| $`x`$-pol., $`p=7`$, BLAS contractions | 8.5 → 8.0 s | 36 → 32 s | 53 → 49 s |
| $`z`$-pol., $`p=3`$ | 0.12 → 0.08 s | 2.47 → 0.31 s | 3.00 → 0.78 s |
| $`z`$-pol., $`p=5`$ | 2.81 → 1.05 s | 57.9 → 4.0 s | 62.6 → 7.1 s |

With the BLAS contractions the element work is about 0.3 s of the 8 s for
$`\mathbf H,\mathbf M`$ in the $`p=7`$ case; the remaining assembly cost is the insertion of the element
matrices into PETSc ($`(p+1)^6`$ values per element and matrix, about 25 ns each)
and the communication of entries owned by other ranks; section 6 removes most of it.

**High-order eigensolves.** At $`p \ge 7`$ in 3D the B-spline overlap matrix is
very ill-conditioned and $`\boldsymbol x^\dagger\mathbf M\boldsymbol x`$ carries a
rounding-level imaginary part that exceeded SLEPc's default definiteness check
($`\sim 10\,\varepsilon`$), aborting the eigensolve. The check is now set to a
relative $`10^{-10}`$ (overridable with `-bv_definite_tol`); the eigenpairs have
residuals of $`10^{-11}`$–$`10^{-13}`$ and agree between the two assembly paths.

## 6. Banded insertion of element matrices

After sum factorisation the assembly time is the insertion of the element
matrices into PETSc. Element-by-element insertion (`MatSetValuesLocal` per
element) adds every stored entry once for each element that shares it, up to

```math
\frac{(p+1)^{2d}}{(2p+1)^{d}}
```

times ($`78`$ at $`p=7`$ in 3D). Each addition is a search in a row of
$`(2p+1)^d`$ columns, plus a copy into the communication stash when the row
belongs to another rank. With $`p=7`$ nearly every element touches rows of a
neighbouring rank, so more ranks made the assembly slower.

On a tensor-product B-spline space, function $`i = (i_1,i_2,i_3)`$ couples only
to $`j`$ with $`|j_d - i_d| \le p_d`$. Each rank therefore sums its element
matrices into a dense band over its ghosted node box,

```math
B\bigl[r(i)\bigr]\bigl[(j_1-i_1+p_1) + w_1\,\bigl((j_2-i_2+p_2) + w_2\,(j_3-i_3+p_3)\bigr)\bigr],
\qquad w_d = 2p_d+1,
```

with $`r(i)`$ PetIGA's ghosted local index (checked against PetIGA's element
mapping on the first element; on a mismatch the code falls back to element
insertion). Each row is then inserted once. Rows of other ranks are still
communicated by PETSc, once per entry. The matrices are the same sums of the
same element matrices in a different order.

The buffer holds (ghosted nodes) $`\times\ \prod_d (2p_d+1)`$ scalars per matrix,
about the size of the assembled matrix. Matrices are accumulated in groups that
fit in `-tdsez_assembly_buffer_mb` (default 1024 MB per rank, at least one matrix
per group); each group is one element pass, and the sum-factorised element work
computes only the matrices of the current group. `-tdsez_band 0` restores
element-by-element insertion.

| Case (4 ranks unless stated) | element insertion | banded | gain |
|---|---|---|---|
| 3D, $`p=7`$, $`10^3`$ elements, $`\mathbf H,\mathbf M`$ | 12.1 s | 2.1 s | $`5.9\times`$ |
| same, time-dependent operators ($`x`$-pol.) | 54.2 s | 6.5 s | $`8.4\times`$ |
| same, whole run | 81.4 s | 22.7 s | $`3.6\times`$ |
| 3D, $`p=7`$, $`20^3`$ elements (19 683 DoFs), $`\mathbf H,\mathbf M`$ | 258 s | 11.5 s | $`22\times`$ |
| same, 1 rank | 129 s | 16.8 s | $`7.7\times`$ |

The first three rows were measured back to back on a partly loaded machine.
On the $`20^3`$-element case the eigensolve (shift-invert with the iterative
inner solver) now dominates the run.

## Effect on accuracy

* **Cached and paired stages** are exact algebraic rewritings; results differ
  only by floating-point rounding.
* **Extrapolated initial guess** changes only the GMRES starting point. The
  stopping test (true residual below $`10^{-12}`$ relative, right preconditioning)
  is unchanged. On the 3D Magnus test the new and old results differ by at most
  $`1\times10^{-10}`$ in $`\langle x\rangle`$ and $`2\times10^{-10}`$ in
  $`\langle H\rangle`$, the accumulated solver tolerance over 800 solves, five
  orders of magnitude below the time-step error of the method at the same
  $`\Delta t`$.
* **Monitor shortcuts** only read $`\boldsymbol\psi`$ and do not affect the
  propagation; they are used only when verified to $`10^{-12}`$. Across the
  regression set the written energies, dipoles and accelerations agree with the
  previous code to $`\le 3\times10^{-14}`$.
* **Batched writes** do not change any written value.
* **Sum-factorised assembly** uses the same quadrature points and coefficients;
  element matrices agree with the per-point kernels to $`\le 5\times10^{-15}`$
  relative. On the 28 repository test inputs, energies, dipoles and spectra
  agree to $`\le 5\times10^{-14}`$.
* **Banded insertion** sums the same element matrices in a different order.
  Across the test inputs, outputs agree with element insertion to rounding
  ($`\le 10^{-13}`$ in spectra and dipoles).
* **Eigenpairs** are made reproducible independently of these changes (see the
  eigensolver documentation): whole degenerate shells, a canonical basis inside
  each shell and a fixed phase. Per-state populations, currents and saved
  wavefunctions agree across assembly paths and MPI rank counts to
  $`\le 5\times10^{-12}`$; before, any rounding change could rotate the states of a
  degenerate shell.

## Reproducing the measurements

Each run prints a TIMING SUMMARY (phase wall times with their parts: eigensolver
setup and iterations, each operator, linear solves, preconditioner setup,
operator updates, observables, HDF5 writes) and a PERFORMANCE block (DoFs,
ranks, time per step, solver counts, peak memory). To
compare propagators on the same problem, run with the same number of ranks and
change only

```text
Propagator = cn        # or pade22, magnus4
TimeStep   = 0.001     # e.g. 0.05 for magnus4 at comparable accuracy
```
