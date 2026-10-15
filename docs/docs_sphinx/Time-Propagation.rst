Time Propagation
================

Controls the time-evolution of the quantum state after eigensolver initialization.
The propagation integrates the TDSE using PETSc's time-stepping framework.

Time Parameters
---------------

.. list-table::
   :widths: 40 15 10 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - EnablePropagation
     - bool
     - 1
     - Whether to run time propagation. Set to 0 for TISE-only
   * - TimeStep / TimeStepSize
     - real
     - 0.01
     - Time step dt in atomic units
   * - FinalTime / TotalTime
     - real
     - 100.0
     - Total propagation time in atomic units

Time Stepper
------------

``Propagator`` selects the scheme; all are unitary for a Hermitian
Hamiltonian:

- ``cn`` (default): Crank-Nicolson (PETSc TS, :math:`\theta = 1/2`), second
  order, one sparse solve per step.
- ``pade22``: exponential midpoint rule with a Pade(2,2) exponential, fourth
  order in the field-free phase, two solves per step.
- ``magnus4``: commutator-free fourth-order Magnus with Pade(2,2) exponentials,
  four solves per step.

The Galerkin equation of motion is

.. math::
   :nowrap:

   \begin{equation}
   iM \frac{d c}{d t} = \big(H + s\,\mathbf E(t)\cdot\mathbf D\big)c ,
   \qquad s = -q ,
   \end{equation}

with the mass matrix :math:`M`, the Hamiltonian :math:`H` (kinetic, potential
and CAP) and the dipole matrices :math:`\mathbf D`.

Each linear system is solved with right-preconditioned GMRES to a relative
residual of :math:`10^{-12}`, preconditioned by restricted additive Schwarz
with one subdomain per rank and ILU(0) on each subdomain. The
preconditioner is built once and reused. ``-tdsez_ilu_levels n`` changes the
fill level; in our tests ILU(1) saved at most one iteration in 2D and none in
3D, where its factorisation is about 10 times slower (degree 4).

Preconditioners
---------------

``Preconditioner`` selects how each linear system is preconditioned. The outer
solve is always the full B-spline operator to the same tolerance, so the choice
changes the cost, never the result (dipole series agree to :math:`10^{-12}`).

- ``asm`` (default): restricted additive Schwarz, one subdomain per rank with
  ILU(0) (see above).
- ``gamg``: PETSc algebraic multigrid (smoothed aggregation) inside flexible
  GMRES. Its default Chebyshev/Jacobi smoothing needs hundreds of iterations on
  this complex operator; use it with the ASM/ILU(0) smoother,
  ``-mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -mg_levels_pc_type asm
  -mg_levels_sub_pc_type ilu`` (``benchmarks/preconditioners/run.sh`` does).
- ``mg``: geometric multigrid on the B-spline space, inside flexible GMRES.
  Coarse levels drop every other simple interior knot of each axis (C0 and
  other repeated knots are kept), so the coarse spaces are nested in the fine
  one and the prolongation is exact: knot insertion per axis, as a tensor
  product, mapped through each level's parallel ordering (checked at set-up on
  a linear function). Coarse operators are Galerkin products
  :math:`P^TAP`; smoothing is one Richardson sweep with ASM/ILU(0) before and
  after the coarse correction (``MGSmooth``); the coarsest level (about 2000
  DOFs, or ``MGLevels`` levels) is solved directly with MUMPS. PETSc options
  with the solver prefix (``-mg_levels_*``, ``-mg_coarse_*``; ``-pade_stage1_``
  / ``-pade_stage2_`` prefixes for the rational propagators) override the
  defaults.

Gauss-Seidel/SOR smoothing is not used: on degree-5 splines it needed 14-26
iterations on one rank (60-80 on four) against 1-4 with ILU smoothing, the
known degree dependence of standard smoothers for B-spline discretisations.

Measured (iterations per solve, time per step relative to ``asm``):

.. list-table::
   :widths: 46 18 18 18
   :header-rows: 1

   * - Case
     - ``asm`` its
     - ``mg`` its
     - ``mg`` time
   * - 2D, p = 5, 40k DOFs, ``cn``, :math:`\Delta t` = 0.02, 1-12 ranks
     - 3-7
     - 2-4
     - 1.4-1.8x
   * - same, :math:`\Delta t` = 0.2
     - 4-11
     - 2-6
     - 1.3-1.7x
   * - 2D variable mass / Coulomb / no CAP, 8 ranks
     - 6-13
     - 3-5
     - 1.1-1.5x
   * - 3D, p = 3, 110k DOFs, ``cn``, 4 and 8 ranks
     - 6-8
     - 4
     - 1.2x
   * - same, 8 ranks, 40 steps: solve time per step (set-up)
     - 0.089 s (1.1 s)
     - 0.158 s (1.7 s)
     - ``gamg``: 4 its, 0.134 s (5.7 s)

Multigrid needs fewer and more uniform iterations, but on up to 12 ranks of
one machine each V-cycle costs more than the iterations it saves. It is meant
for large rank counts, where one-level ASM degrades because it has no coarse
space; measure both on the target machine.

**Other PETSc and PetIGA preconditioners.** Any ``PCType`` can be selected per
stage solver (``-pade_stage1_pc_type X -pade_stage2_pc_type X``). Measured with
``pade22``, 8 ranks, :math:`\Delta t = 0.05`:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - ``pc_type``
     - 3D, p = 3, 32k DOFs, 40 steps
     - 2D, p = 4, 14k DOFs, 100 steps
   * - ``asm`` + ILU(0) (default)
     - 4.5 s, 5.5 its/solve
     - 0.65 s, 5.0
   * - ``asm``, overlap 2
     - --
     - 0.63 s, 3.3
   * - ``bjacobi`` / ``gasm`` / ``bjacobi`` + ICC
     - 9.0-9.3 s, 25
     - 1.5-1.6 s, 19
   * - ``sor``
     - 40 s, 91
     - 6.8 s, 95
   * - ``jacobi``
     - 50 s, 320
     - 9.5 s, 238
   * - ``igaebe`` (PetIGA, element by element)
     - 69 s, 222
     - 7.2 s, 102
   * - ``igabbb`` (PetIGA, basis by basis)
     - not converged in 1000 its
     - not converged, also with ``-pc_bbb_overlap 4,4``

(Before the extrapolated initial guess below; the ranking is unchanged.)
``igaebe`` and ``igabbb`` invert one small dense block per element or per basis
function (overlap :math:`p/2` by default): they damp local error like a
multigrid smoother but have no global coupling, so on their own they need
hundreds of iterations or stall. Block Jacobi, GASM and ICC lack the overlap of
ASM and need about four times the iterations.

**Accuracy does not depend on the preconditioner.** GMRES is right
preconditioned, so it stops on the true residual of the original system
(:math:`10^{-12}` relative) whatever the preconditioner. The same 2D run
(100 ``pade22`` steps) against an exact solve (MUMPS LU), largest relative
difference:

.. list-table::
   :widths: 25 15 15 15 15 15
   :header-rows: 1

   * - Preconditioner
     - dipoles
     - energies
     - currents
     - populations
     - autocorrelation
   * - ``asm`` (default)
     - 1.3e-11
     - 8.1e-13
     - 1.3e-12
     - 1.3e-12
     - 6.0e-13
   * - ``bjacobi``
     - 2.0e-11
     - 1.8e-12
     - 3.9e-12
     - 2.3e-12
     - 9.7e-13

The norm drifts by :math:`8\times10^{-14}` over the run. These differences are
far below the time-step and grid error (about :math:`10^{-5}` on validation
grids).

**Cost of the GMRES solves.** Orthogonalisation is 1-2 % of a solve; the rest
is preconditioner applications (ILU triangular solves, about 60 %) and
matrix-vector products. What reduces it is fewer iterations:

- the stage solves start from the increment extrapolated from the last three
  steps (``-pade_guess_order 1..5``, default 3; 5.5 -> 4.1 its/solve in 3D,
  5.0 -> 3.4 in 2D);
- a looser tolerance, ``-pade_stage1_ksp_rtol 1e-11 -pade_stage2_ksp_rtol
  1e-11``, saves another 10-13 % with output differences of
  :math:`5\times10^{-11}` (default :math:`10^{-12}`).

**Pipelined GMRES** (``-pade_stage{1,2}_ksp_type pgmres``) overlaps each
iteration's global reduction (``MPI_Iallreduce``) with the next product. On
one machine a reduction costs microseconds through shared memory and the extra
vector work dominates: 20-35 % slower in time stepping (3D, 8 ranks, same
4.1 its/solve). It can pay off across many nodes, where reductions cross the
network; measure it there. It does not remove load imbalance (see MPI Notes).

**Strong fields.** The operator is
:math:`M + i\tau(H + s\,\mathbf E\cdot\mathbf r)` (:math:`\tau = \Delta t/2`
for ``cn``). When :math:`\tau\max|\mathbf E\cdot\mathbf r|` exceeds about 1.5
the field term dominates and the operator behaves like an indefinite Helmholtz
operator that the coarse levels cannot represent: in 2D tests multigrid needed
3-18 iterations up to 1.5 and diverged at 3, where ``asm`` still converged.
TDSE-Z prints a warning at start-up in that regime. Physical strong-field runs
(:math:`E \le 0.2` a.u., :math:`\Delta t \le 0.05`, box :math:`\le 100` a.u.)
stay well below it.

**Preconditioner refresh.** The preconditioner is built from the operator of
one step and reused. When a solve needs more than twice the iterations of the
first solve after the last build (plus 5), it is rebuilt from the current
operator (both preconditioners, all propagators). Without this, a field of
2 a.u. took 4, 13, 24, 65, 249, 641 iterations in successive steps and then
failed; with it the run converges with a few rebuilds (printed at the end).

Accuracy of the Rational Propagators
------------------------------------

The rational propagators reach a given accuracy with a much larger step: on
the free-packet t-SURFF benchmark (:doc:`TSurff`, T = 40) ``pade22`` at
:math:`\Delta t = 0.025` reaches a spectrum error of :math:`1.2\times10^{-6}`
against :math:`3.3\times10^{-4}` for ``cn`` at :math:`\Delta t = 0.0125`.

Choosing a Time Step
--------------------

The time step must be small enough to resolve:

1. **Highest energy component**: :math:`\Delta t \ll 2\pi / E_{\text{max}}`
2. **Laser frequency**: :math:`\Delta t \ll 2\pi / \Omega`
3. **Fastest potential variation**: Ensure the wavefunction doesn't change much
   within one step

Rule of thumb:

- **Ground-state dynamics**: 0.01–0.1 a.u. (typically fine)
- **High-frequency laser**: :math:`\Delta t < 0.01` a.u.
- **Broadband excitation**: :math:`\Delta t < 0.001` a.u.
- **Very fast processes**: :math:`\Delta t < 0.0001` a.u.

If results depend on time step (run with dt and dt/2 and compare), reduce it.

Propagation Control
-------------------

**EnablePropagation = 0** — Run eigensolver only, no time propagation.

This is useful for:
- Verifying eigenvalues and eigenstates before propagation
- Computing stationary properties
- Quick convergence checks

**EnablePropagation = 1** — Run full time propagation.

The propagation starts from the initial state (set via ``InitialState``) and
integrates for ``FinalTime`` time units with ``TimeStep`` spacing.

Monitoring Propagation
----------------------

Add ``Verbose = 1`` (default) to see timestep-by-timestep output:

::

    Timestep 0001, time = 0.0100, norm = 1.000000
    Timestep 0002, time = 0.0200, norm = 1.000000
    ...

If norm deviates from 1.0, there's an issue with:
- Time step too large (numerical instability)
- Boundary conditions (particle escaping through reflecting wall)
- CAP parameters (absorbing too much or too little)

Common Propagation Scenarios
----------------------------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Scenario
     - TimeStep
     - FinalTime
     - Notes
   * - Ground-state evolution
     - 0.01
     - 100
     - Standard, conservative
   * - Fast laser dynamics
     - 0.001
     - 10
     - High-frequency field
   * - Ionization study
     - 0.001
     - 200
     - Long time, fine step
   * - Eigenvalue verification
     - N/A
     - N/A
     - EnablePropagation = 0
   * - Short pulse interaction
     - 0.005
     - 50
     - Moderate resolution

Troubleshooting
---------------

**Propagation stops early**
  - Check if ``ts_max_steps`` is reached (default: 10000)
  - Increase with ``-ts_max_steps N`` if needed

**Norm not conserved**
  - Reduce ``TimeStep``
  - Check boundary conditions
  - Verify CAP is properly configured if studying ionization

**No output after propagation**
  - Set ``PhysicsOutput = all`` to enable all diagnostics
  - Check output stride settings
