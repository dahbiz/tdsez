Boundary Conditions
===================

Controls how the wavefunction behaves at the edges of the computational domain.
Choosing the right boundary condition is critical for accuracy — reflecting walls
cause spurious reflections, while CAP allows outgoing flux to leave cleanly.

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
   * - CAPKmin
     - real
     - 0.1
     - Lowest momentum absorbed efficiently; sets the layer width w = 6.555/CAPKmin (see below)

How CAP Works
-------------

The CAP adds :math:`-iW(\mathbf r)` to the Hamiltonian, with the reflection-free
profile of Manolopoulos (J. Chem. Phys. 117, 9552 (2002)). Every axis gets an
absorbing layer of width

.. math::

   w = \frac{c}{2\,\delta\,k_{\min}} \approx \frac{6.555}{k_{\min}}, \qquad c = 2.62206,\ \delta = 0.2,

at both of its walls, :math:`k_{\min}` = ``CAPKmin``. Inside a layer, with
:math:`u` the distance from the layer's inner edge and :math:`r = c\,u/w`,

.. math::

   W = E_{\min}\left(a r - b r^3 + \frac{4}{(c-r)^2} - \frac{4}{(c+r)^2}\right),
   \qquad E_{\min} = \tfrac12 k_{\min}^2 ,

which starts at zero (no step at the inner edge) and diverges at the wall. The
contributions of the axes add, so non-cubic and asymmetric boxes are absorbed on
every face. Electrons with momentum above :math:`k_{\min}` are absorbed with
reflection below about 1%; slower ones are partly reflected. The layer may take
at most 45% of each axis; the run stops otherwise.

Tuning CAP Parameters
---------------------

- ``CAPKmin`` sets both the lowest absorbed momentum and the layer width. Lower
  values absorb slower electrons but need a wider layer (a larger box).
- Keep the physically relevant region (for HHG, the electron excursion
  :math:`E_0/\omega^2` and the returning trajectories) inside the free core,
  which is printed at start-up.

Recommendation: set ``PhysicsOutput = population`` to monitor the bound-state
population, and compare HHG spectra with a larger box or a smaller ``CAPKmin``
to check that reflections are negligible.

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
