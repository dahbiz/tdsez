Grid & Domain
=============

Spatial discretization: dimension, bounds, elements, splines, quadrature.

Dimension
---------

.. list-table::
   :widths: 20 10 10 60
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Dimension
     - int
     - 1
     - Spatial dimension: 1, 2, or 3

Spatial Bounds
--------------

.. list-table::
   :widths: 35 15 15 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Domain
     - string
     - ``[-10,10]``
     - Spatial bounds per axis. Format: comma-separated ``[min,max]``
       pairs, or single pair for all axes. E.g. ``Domain = [-10,10], [-8,8]``
       for 2D asymmetric y
   * - LMinX, LMaxX, LMinY, LMaxY, LMinZ, LMaxZ
     - real
     - +/-10.0
     - Per-axis bounds (alternative to Domain)
   * - OffsetX, OffsetY, OffsetZ
     - real
     - 0.0
     - Origin offset along each axis

Mesh Resolution
---------------

.. list-table::
   :widths: 25 10 10 55
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Nelements
     - int
     - 100
     - Number of elements per axis. Aliases: ``NSplines``,
       ``NumberOfSplines``, ``ElementCount``, ``NumberOfElements``
   * - SplineDegree
     - int
     - 3
     - B-spline degree (1-14). Higher = more accuracy per DOF but more
       fill-in. For most attosecond problems, **p=5** is the sweet spot.
   * - NQuadratures
     - int
     - 8
     - Quadrature points per element and axis (at most 10). n Gauss points
       integrate polynomials of degree 2n-1 exactly: p+1 points are exact for
       M and K; singular potentials need more (see Knot Sequences).
   * - QuadratureRule
     - string
     - legendre
     - ``legendre`` (Gauss), ``lobatto`` (Gauss-Lobatto, includes the element
       ends) or ``reduced`` (PetIGA's reduced B-spline rule).
   * - Basis
     - string
     - bspline
     - ``bspline`` (smooth, C^{p-1}) or the C0 element bases ``bernstein``,
       ``lagrange``, ``spectral`` (Lagrange at Gauss-Lobatto points); for these
       every interior knot is raised to multiplicity p.

Note: SplineDegree range is **1-14**. Higher degrees reduce DOF count for the
same accuracy but increase matrix fill.
