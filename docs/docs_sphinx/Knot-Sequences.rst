Knot Sequences
==============

B-spline knot vector generation supports uniform, symmetric exponential and
tangent, logarithmic tangent, hydrogenic, and interface-aligned sequences. Any
other name (including the removed placeholders ``adaptive`` and
``adaptive_wf``) stops the run with a message.

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
     - HydrogenicR1 (alias InnerSpacing, HydrogenicInnerSpacing), HydrogenicNLin (alias InnerElements, HydrogenicInnerElements), HydrogenicNExp (alias OuterElements, HydrogenicOuterElements), HydrogenicHMax (alias HydrogenicMaxSpacing; largest element width, uniform beyond, 0 = off); per-axis lists

Per-Axis Assignment
-------------------

Knot sequences are comma-separated per axis:

::

    KnotSequence = "symexp, uniform, symtan"

X=symexp, Y=uniform, Z=symtan. Missing axes default to the first value.

Coulomb Potentials (hydrogenic sequence)
----------------------------------------

Measured on the 3D hydrogen atom, :math:`V = -1/r`, degree 3, 8 Gauss points,
``HydrogenicR1 = 0.05``, ``HydrogenicNLin = 5``, ``HydrogenicNExp = 10``
(35 937 DOFs), exact ground state :math:`-0.5`.

**Box size and boundaries dominate.** With the default natural (Neumann)
boundaries the wavefunction is not forced to zero at the box, and the energy
minimum then drops *below* the exact value when the box cuts the tail of the
state (the cut-off tail has positive local energy). The 1s tail beyond 5 au
holds 0.3 % of the probability:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Box
     - Neumann (default)
     - Dirichlet
   * - :math:`\pm 5`
     - -0.501454
     - -0.498718
   * - :math:`\pm 10`
     - -0.4999968
     - -0.4999966
   * - :math:`\pm 15`
     - -0.4999935
     - -0.4999935

Use :math:`\pm 10` au for the ground state, :math:`\pm 20` for n = 2
(on :math:`\pm 10` the n = 2 energies are 3-6e-3 too low), about
:math:`\pm 35` for n = 3. On :math:`\pm 20` with 15 exponential elements
(79 507 DOFs) 1s and the four n = 2 states are within 2e-6.

**Grading.** Refining the inner region (5 x 0.05 au) changes the ground state by
less than 1e-6; the outer, exponentially graded elements limit the accuracy
(13 instead of 10 exponential elements: error 3.2e-6 -> 7.8e-7). Smaller
inner elements at a fixed element count make it worse (R1 = 0.01-0.02: 8-18e-6),
because the outer elements coarsen. The exponential tail grows to elements
of :math:`L_{max}(1 - r_{cross}/L_{max})^{1/N_{exp}}` (3.1 au on
:math:`\pm 10`, 5 au on :math:`\pm 20`), too coarse for ionised or excited
electrons during propagation (momentum about 1 au needs widths below
about 1 au). ``HydrogenicHMax`` caps the width: from the first element that
would be wider, the rest of the half-axis is uniform with width
:math:`\le` HMax. It does not change bound states noticeably
(:math:`\pm 10`, HMax = 1: 68 921 DOFs, E0 = -0.49999866) and is meant for
propagation. The knot line printed at start-up reports the largest width.

**Quadrature.** The nucleus sits on a knot (an element corner), where Gauss
points never fall. Gauss-Legendre integrates the corner singularity of
:math:`1/r` with a relative error of 1.6e-4 (8 points) on the 8 elements that
touch it; the ground state moves by 7e-7 from 8 to 10 points. Fewer points
under-integrate the singularity and overestimate the attraction, which can
cancel the discretisation error by accident (4 points: -0.500009, below the
exact value; 6 points: -0.4999990) - converge in the number of points, not
toward -0.5. ``QuadratureRule = reduced`` behaves like Gauss with fewer points
(-0.4999976 at 8); ``lobatto`` puts points on the knots, where :math:`1/r` is
infinite, and is stopped at assembly with a message. Assembly time is
0.24-0.39 s for 4-10 points and the eigensolve does not depend on the rule.

**Basis.** At equal DOFs (50 653) the smooth B-splines are about 100 times more
accurate than the C0 element bases, which span the same space as each other
and differ only in conditioning:

.. list-table::
   :widths: 30 25 25
   :header-rows: 1

   * - ``Basis``
     - error of E0
     - eigensolve
   * - ``bspline``
     - 1.1e-6
     - 13.3 s
   * - ``bernstein``
     - 1.3e-4
     - 6.9 s
   * - ``lagrange``
     - 1.3e-4
     - 14.3 s
   * - ``spectral``
     - 1.3e-4
     - 16.8 s

(``Preconditioner = mg`` requires ``bspline``: its prolongation is B-spline
knot insertion.)

