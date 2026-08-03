Potential & Mass
================

muParser expressions, position-dependent mass, and constants.

Potential
---------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Potential
     - muParser
     - ``0.0``
     - Scalar potential V(x,y,z). MuParser expressions using ``x``, ``y``, ``z``
   * - PotentialDerivativeX / Y / Z
     - muParser
     - ``0.0``
     - Analytic dV/dx, dV/dy, dV/dz. Auto-computed numerically if omitted

Mass
----

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Mass
     - muParser
     - ``1.0``
     - Position-dependent mass M(x,y,z). Use ``1.0`` for constant mass
   * - invMassX / Y / Z
     - muParser
     - ``0.0``
     - Gradients of 1/m
   * - MassIsConstant
     - bool
     - auto
     - Set to ``1`` for constant mass. Disables quantum-correction terms,
       speeding assembly

Physical Constants
------------------

.. list-table::
   :widths: 30 15 10 45
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Hbar / Planck
     - real
     - 1.0
     - Reduced Planck constant (atomic units)
   * - Charge / ParticleCharge
     - real
     - 1.0
     - Particle charge

Custom Variables
----------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - Variables
     - string
     - User-defined muParser constants. Format:
       ``V0=1.2, w=0.5``. Available in ALL expressions including
       Potential, Laser, Mass.

All scalar input parameters are automatically defined as muParser constants
(e.g., Amplitude, Omega, TimeStep, LMin). No need to redefine them in
Variables.
