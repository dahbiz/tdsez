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

TDSE-Z uses the PETSc TS framework with **Crank-Nicolson (theta = 0.5)** as the
implicit time-stepper. The method is unitary (norm-preserving) and second-order
accurate.

The residual at each step is evaluated as:

.. math::
   :nowrap:

   \begin{equation}
   F = iM \frac{\partial \psi}{\partial t} - H\psi - E(t)D\psi = 0
   \end{equation}

where M is the mass matrix, H is the Hamiltonian, E(t) is the laser field, and D
is the dipole operator.

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
