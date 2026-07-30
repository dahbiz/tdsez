boundary-conditions
===================

Boundary Conditions
===================

Reflecting walls, Dirichlet, and complex absorbing potential.

Boundary Types
--------------

.. list-table::
   :widths: 35 15 20 30
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - BoundaryType / Boundary
     - string
     - ``Neumann``
     - ``Neumann``/``Natural`` = reflecting wall.
       ``Dirichlet``/``Wall`` = psi=0 at boundary

Complex Absorbing Potential
---------------------------

.. list-table::
   :widths: 35 15 10 40
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - EnableCAP / AbsorbingBoundary
     - bool
     - 0
     - Enable complex absorbing potential for outgoing-wave boundary conditions
   * - CAPKmin / CapStrength
     - real
     - 0.1
     - Inner edge of CAP region (a.u.)
   * - Gamma
     - real
     - 5.0
     - CAP strength. Higher = stronger absorption but more numerical reflection
