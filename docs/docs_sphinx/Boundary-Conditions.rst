Boundary Conditions
===================

Controls how the wavefunction behaves at the edges of the computational domain.
Choosing the right boundary condition is critical for accuracy — reflecting walls
cause spurious reflections, while CAP or t-SURFF allow outgoing flux to leave
cleanly.

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
       ``Dirichlet``/``Wall`` = :math:`\psi = 0` at boundary

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
     - CAP strength :math:`\gamma`. Higher = stronger absorption but more numerical reflection

How CAP Works
-------------

The CAP adds an imaginary potential :math:`-i\gamma(x)` near the domain edges:

.. math::
   V_{\text{CAP}}(x) = -i\gamma \left(\frac{x - x_{\text{edge}}}{L_{\text{region}}}\right)^n \quad \text{for } x > x_{\text{edge}}

where :math:`x_{\text{edge}}` is ``CAPKmin``, :math:`\gamma` is ``Gamma``, and
:math:`n=2` by default. The imaginary part causes exponential decay of outgoing
waves, mimicking an open boundary.

Tuning CAP Parameters
~~~~~~~~~~~~~~~~~~~~~

- **``CAPKmin``**: Move inward to absorb more, outward to reduce reflections.
  Start with ``0.1`` and adjust based on reflected population.
- **``Gamma``**: Higher values absorb more strongly but can cause numerical
  reflection at the CAP edge. Typical range: 1–10.
- **Higher Gamma** → stronger absorption, but potential numerical reflections
- **Lower Gamma** → less absorption, fewer reflections

Recommendation: Set ``PhysicsOutput = population`` to monitor bound-state
population. If population decreases too quickly, lower Gamma. If population
oscillates (reflecting), increase CAPKmin or Gamma.

Reflecting vs Absorbing Boundaries
----------------------------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1
   :stub-columns: 1

   * -
     - Reflecting
     - CAP
   * - Flux
     - No flux leaves domain
     - Outgoing flux absorbed
   * - Use case
     - Good for bound states only
     - Good for ionization/continuum
   * - Population
     - Population conserved
     - Population can decrease
   * - Tuning
     - Simple, no tuning needed
     - Requires parameter tuning

Use reflecting boundaries when studying bound-state dynamics only. Use CAP when
ionization or ionization-like processes are present.

Dirichlet Boundary
------------------

``BoundaryType = Dirichlet`` (or ``Wall``) forces :math:`\psi = 0` at the domain
edges. This is a hard wall — stronger than Neumann but still reflects.

Use Dirichlet when:
- You want a stricter confinement than Neumann
- You're studying systems with natural nodes at boundaries
- You need to ensure wavefunction vanishes at edges
