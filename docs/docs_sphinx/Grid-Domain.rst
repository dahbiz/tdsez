Grid & Domain
=============

TDSE-Z discretises the wave function in a tensor-product B-spline basis
[deBoor2001]_ [Bachau2001]_ on a rectangular box, using the isogeometric
analysis framework PetIGA [Hughes2005]_ [Dalcin2016]_. This page describes the
keys that define the box, the basis and the quadrature. The placement of the
knots along each axis is described in :doc:`Knot-Sequences`; all keys are
listed in :doc:`Input-File`.

Discretisation
--------------

Along each axis :math:`a` the basis consists of the :math:`n_a` B-splines
:math:`B_{i}^{(a)}` of degree :math:`p` (``SplineDegree``) on a clamped knot
vector (end knots of multiplicity :math:`p+1`). In :math:`d` dimensions the
wave function is expanded as

.. math::
   :label: eq-grid-domain-expansion

   \psi(\mathbf r, t) = \sum_{i_1,\dots,i_d} c_{i_1\dots i_d}(t)\,
   B^{(1)}_{i_1}(x_1)\cdots B^{(d)}_{i_d}(x_d),

so the number of degrees of freedom is :math:`N_{\mathrm{DOF}} = \prod_a n_a`.
With ``Basis = bspline`` and simple interior knots,

.. math::
   :label: eq-grid-domain-nfun

   n_a = N_{\mathrm{el}} + p ,

where :math:`N_{\mathrm{el}}` is the number of elements (knot spans) of the
axis. Within an element the basis is polynomial of degree :math:`p`; across a
simple knot it is :math:`C^{p-1}`.

Dimension and bounds
--------------------

.. list-table::
   :widths: 30 10 12 48
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``Dimension``
     - int
     - 1
     - Spatial dimension, 1, 2 or 3.
   * - ``Domain``
     - string
     - --
     - ``lo, hi`` sets every axis; ``[lox,hix], [loy,hiy], [loz,hiz]`` or
       ``lox, hix, loy, hiy[, loz, hiz]`` sets the axes separately (missing
       pairs copy the x pair). Another number of values stops the run.
   * - ``LMinX``, ``LMaxX``, ``LMinY``, ``LMaxY``, ``LMinZ``, ``LMaxZ``
     - real
     - -10, 10
     - Bounds of each axis (a.u.). They must be finite with
       ``LMax > LMin``.
   * - ``OffsetX``, ``OffsetY``, ``OffsetZ``
     - real
     - 0
     - Translation of the mesh along each axis.

Example of an asymmetric 2D box::

    Dimension = 2
    Domain    = [-60, 60], [-30, 30]

``OffsetX/Y/Z`` translate the computational box: axis :math:`a` spans
:math:`[L_{\min,a} + o_a,\ L_{\max,a} + o_a]`. The offsets are added to the
bounds once at start-up, so the knots, the potential, mass and dipole
operators, the input validation, the CAP layers, the t-SURFF checks and the
printed reports all use the shifted box. The ``hydrogenic`` sequence is built
symmetrically about the origin from :math:`L_{\max}` (see
:doc:`Knot-Sequences`).

Mesh resolution
---------------

.. list-table::
   :widths: 30 10 12 48
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``Nelements`` (``NSplines``, ``NumberOfSplines``, ``ElementCount``,
       ``NumberOfElements``)
     - int
     - 100
     - Number of elements per axis (the same value on every axis), at least 1.
       The ``hydrogenic`` and ``interface`` sequences set or redistribute the
       element count themselves (:doc:`Knot-Sequences`).
   * - ``SplineDegree``
     - int
     - 3
     - Polynomial degree :math:`p`, 1 to 14.

A warning is printed when the x axis has fewer than :math:`2p+1` interior
knots (``Nelements - 1 < 2 SplineDegree + 1``), since such a basis cannot
represent localised states reliably.

A higher degree gives a higher convergence order per degree of freedom at the
price of wider matrix bands (each row couples :math:`(2p+1)^d` basis
functions).

Basis
-----

.. list-table::
   :widths: 30 10 12 48
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``Basis`` (``BasisType``)
     - string
     - ``bspline``
     - ``bspline``, ``bernstein``, ``lagrange`` or ``spectral`` (Lagrange
       polynomials at the Gauss-Lobatto points); case-insensitive.

The element bases ``bernstein``, ``lagrange`` and ``spectral`` are
:math:`C^0`: after the knot vector of an axis has been generated, every
distinct interior knot is raised to multiplicity :math:`p`, which gives
:math:`n_a = p\,N_{\mathrm{el}} + 1` functions on an axis with
:math:`N_{\mathrm{el}}` elements. The three element bases span the same space
and differ only in conditioning. ``Preconditioner = mg`` requires
``Basis = bspline``, because its prolongation is B-spline knot insertion. A
measured comparison is given in :doc:`Knot-Sequences`.

Quadrature
----------

.. list-table::
   :widths: 30 10 12 48
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``NQuadratures``
     - int
     - 8
     - Quadrature points per element and axis. PetIGA provides Gauss-Legendre
       rules with 1 to 10 points and Gauss-Lobatto rules with 2 to 10 points.
   * - ``QuadratureRule`` (``QuadratureType``)
     - string
     - ``legendre``
     - ``legendre`` (Gauss), ``lobatto`` (Gauss-Lobatto, includes the element
       ends) or ``reduced`` (PetIGA's reduced B-spline rule);
       case-insensitive.

An :math:`n_q`-point Gauss-Legendre rule integrates polynomials up to degree
:math:`2n_q - 1` exactly. The integrands of the overlap and kinetic matrices
for a constant mass have degree :math:`2p` and :math:`2p-2` per axis, so

.. math::
   :label: eq-grid-domain-gauss

   n_q \ge p + 1

integrates them exactly. Position-dependent masses and potentials are not
polynomial; singular potentials such as :math:`1/r` require a convergence
study in :math:`n_q` (:doc:`Knot-Sequences`). A Gauss-Lobatto rule evaluates
the potential at the knots; if it is infinite there the Hamiltonian contains
non-finite entries and the run stops after assembly with a message.
