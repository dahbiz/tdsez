physics-conventions
===================

Physics Conventions
===================

Atomic units, gauge choices, Hamiltonian form, time stepper details.

Convention Reference
--------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Quantity
     - Convention
   * - Units
     - All in atomic units (Hartree). Energy: Hartree, length: Bohr (a_0),
       mass: m_e, charge: e
   * - Hamiltonian
     - H = -1/2 * nabla^2/m(r) + V(r) + CAP
   * - TDSE (length gauge)
     - i * dpsi/dt = H * psi + E(t) * D * psi
   * - Residual (Crank-Nicolson)
     - F = iM * psidot - H * psi - E(t) * D * psi
   * - Mass inner product
     - <phi|M|psi> used for all projections, populations, norms
   * - Dipole operator
     - D_x = x (position operator)
   * - Velocity operator
     - V_x = -i/(2m(r)) (B * d_x - d_x * B), anti-symmetric
   * - Time stepper
     - PETSc TS with Crank-Nicolson (theta = 0.5)
