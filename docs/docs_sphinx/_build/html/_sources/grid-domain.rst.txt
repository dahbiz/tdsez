grid-domain
===========

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
   * - LMin / LMax
     - real
     - -10.0 / 10.0
     - Symmetric domain bounds (shorthand)
   * - LMinX, LMaxX, LMinY, LMaxY, LMinZ, LMaxZ
     - real
     - +/-10.0
     - Per-axis bounds (default to LMin/LMax)
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
     - Gauss-Legendre quadrature per element. Use >= 2p+1 for full integration.

Note: SplineDegree range is **1-14**. Higher degrees reduce DOF count for the
same accuracy but increase matrix fill.
