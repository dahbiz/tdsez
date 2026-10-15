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
