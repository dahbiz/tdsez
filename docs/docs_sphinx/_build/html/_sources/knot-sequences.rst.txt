knot-sequences
==============

Knot Sequences
==============

B-spline knot vector generation: uniform, symexp, symmetric, interface, hydrogenic,
adaptive.

Knot Sequence Types
-------------------

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Sequence
     - Description
     - Tuning Keys
   * - **uniform**
     - Uniform spacing
     - --
   * - **symexp**
     - Symmetric exponential: dense near origin, sparse at edges
     - KnotAlpha
   * - **symtanu**
     - Symmetric tangent, unbounded: tanh-compressed near origin
     - KnotAlpha
   * - **symtan**
     - Symmetric tangent, bounded in [Lmin,Lmax]
     - KnotAlpha
   * - **logtan**
     - Logarithmic tangent
     - KnotAlpha
   * - **interface**
     - Auto-aligned at potential/mass breakpoints
     - --
   * - **hydrogenic**
     - Linear near origin, exponential tail. For Coulombic problems
     - HydrogenicNLin[d], HydrogenicNExp[d], HydrogenicR1[d]
   * - **adaptive**
     - Potential-driven: clusters knots where \|gradient V\| + wells are strong
     - AdaptiveKappa, AdaptivePower
   * - **adaptive_wf**
     - Density-driven (two-pass): first coarse solve, then cluster on \|psi_0\|^2
     - AdaptiveWFCoarseN, AdaptiveWFKinLambda

Per-Axis Assignment
-------------------

Knot sequences are comma-separated per axis:

::

    KnotSequence = "symexp, uniform, symtan"

X=symexp, Y=uniform, Z=symtan. Missing axes default to the first value.
