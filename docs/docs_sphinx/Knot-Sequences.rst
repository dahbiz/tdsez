Knot Sequences
==============

B-spline knot vector generation supports uniform, symmetric exponential and
tangent, logarithmic tangent, and hydrogenic sequences. The ``interface``,
``adaptive``, and ``adaptive_wf`` names currently select uniform interior knots;
their specialized generators are pending implementation.

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
     - Pending: currently uses uniform interior knots, without interface alignment
     - None currently active
   * - **hydrogenic**
     - Linear near origin, exponential tail. For Coulombic problems
     - HydrogenicNLin[d], HydrogenicNExp[d], HydrogenicR1[d]
   * - **adaptive**
     - Pending: currently uses uniform interior knots, without potential-driven clustering
     - AdaptiveKappa and AdaptivePower are currently ignored
   * - **adaptive_wf**
     - Pending: currently skips the coarse solve and uses uniform interior knots
     - AdaptiveWFCoarseN and AdaptiveWFKinLambda are currently ignored

Per-Axis Assignment
-------------------

Knot sequences are comma-separated per axis:

::

    KnotSequence = "symexp, uniform, symtan"

X=symexp, Y=uniform, Z=symtan. Missing axes default to the first value.
