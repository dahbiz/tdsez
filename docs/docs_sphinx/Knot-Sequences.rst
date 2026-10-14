Knot Sequences
==============

B-spline knot vector generation supports uniform, symmetric exponential and
tangent, logarithmic tangent, hydrogenic, and interface-aligned sequences. The
``adaptive`` and ``adaptive_wf`` names currently select uniform interior knots;
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
     - KnotAlpha (alias KnotGrading, GradingStrength); default 0.9
   * - **symtanu**
     - Inverse-tanh map: dense near origin, sparse at edges
     - KnotAlpha; default 0.9
   * - **symtann**
     - tanh map: dense near the edges, sparse at origin
     - KnotAlpha; default 0.9
   * - **symlogtan**
     - Tangent map, mildly dense near origin
     - KnotAlpha; default 0.5
   * - **interface** (alias **interfaces**)
     - C0 knots (multiplicity p) at every breakpoint of the Potential and Mass expressions; uniform elements within each layer, element count per layer proportional to its width. Required for spectral accuracy at abrupt BDD heterojunctions.
     - Breakpoints from ``x < c`` / ``abs(x) < c`` comparisons
   * - **hydrogenic**
     - Per half-axis: NLin linear elements of size R1 up to r_cross = NLin*R1, then NExp exponentially growing elements up to Lmax; mirrored about the origin. For Coulombic problems
     - HydrogenicR1 (alias InnerSpacing, HydrogenicInnerSpacing), HydrogenicNLin (alias InnerElements, HydrogenicInnerElements), HydrogenicNExp (alias OuterElements, HydrogenicOuterElements); per-axis lists
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
