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
     - :math:`H = -\frac{1}{2} \nabla \cdot \left( \frac{1}{m(\mathbf{r})} \nabla \right) + V(\mathbf{r}) + \text{CAP}` (BenDaniel–Duke)
   * - TDSE (length gauge)
     - :math:`i \frac{\partial \psi}{\partial t} = H\psi + \mathbf E(t)\cdot\mathbf r\,\psi`, i.e. the
       coupling of an electron (charge :math:`-e`) to the field :math:`\mathbf E(t)`
       given by ``LaserX/Y/Z``; the force on the electron is :math:`-\mathbf E`
   * - Charge
     - ``Charge`` (:math:`q`, default 1) scales the reported dipole
       :math:`q\langle\mathbf r\rangle`, current :math:`q\langle\mathbf v\rangle` and
       acceleration :math:`q\,d\langle\mathbf v\rangle/dt` consistently. It does not
       change the propagated Hamiltonian.
   * - Residual (Crank-Nicolson)
     - :math:`F = iM \dot{\psi} - H\psi - E(t)D\psi`
   * - Mass inner product
     - :math:`\langle \phi | M | \psi \rangle` used for all projections, populations, norms
   * - Dipole operator
     - :math:`D_x = x` (position operator)
   * - Velocity operator
     - :math:`V_x = -\frac{i}{2m(\mathbf{r})} (\mathbf{B} \cdot \nabla - \nabla \cdot \mathbf{B})`, anti-symmetric
   * - Time stepper
     - PETSc TS with Crank-Nicolson (theta = 0.5)
