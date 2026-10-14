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
     - :math:`H = -\frac{1}{2} \nabla \cdot \left( \frac{1}{m(\mathbf{r})} \nabla \right) + s\,V(\mathbf{r}) + \text{CAP}` (BenDaniel–Duke),
       :math:`s = -q`
   * - TDSE (length gauge)
     - :math:`i \frac{\partial \psi}{\partial t} = H\psi + s\,\mathbf E(t)\cdot\mathbf r\,\psi`, with
       :math:`\mathbf E(t)` given by ``LaserX/Y/Z`` on the polarised axes. ``Potential``
       and the field are written for an electron: for :math:`q = -1` (:math:`s = 1`)
       they enter as written and the force of the field is :math:`-\mathbf E`
   * - Charge
     - ``Charge`` :math:`q` in units of :math:`e`: :math:`-1` electron (default),
       :math:`+1` hole or positive quasiparticle, for which the potential and the
       field term are inverted, :math:`H = K - (V + \mathbf r\cdot\mathbf E)`. General
       :math:`q` scales both, :math:`H = K - q\,(V + \mathbf r\cdot\mathbf E)`. The outputs are
       the dipole :math:`q\langle\mathbf r\rangle`, current :math:`q\langle\mathbf v\rangle` and
       acceleration :math:`q\,d\langle\mathbf v\rangle/dt`.
   * - Residual (Crank-Nicolson)
     - :math:`F = iM \dot{\psi} - H\psi - s\,\mathbf E(t)\cdot\mathbf D\,\psi`
   * - Mass inner product
     - :math:`\langle \phi | M | \psi \rangle` used for all projections, populations, norms
   * - Dipole operator
     - :math:`D_x = x` (position operator)
   * - Velocity operator
     - :math:`V_x = -\frac{i}{2m(\mathbf{r})} (\mathbf{B} \cdot \nabla - \nabla \cdot \mathbf{B})`, anti-symmetric
   * - Time stepper
     - PETSc TS with Crank-Nicolson (theta = 0.5)
