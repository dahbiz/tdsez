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

Multigrid needs fewer and more uniform iterations, but on up to 12 ranks of
one machine each V-cycle costs more than the iterations it saves. It is meant
for large rank counts, where one-level ASM degrades because it has no coarse
space; measure both on the target machine.

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
