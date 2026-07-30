hdf5-output
===========

HDF5 Output Schema
==================

Complete dataset layout: dipole, energy, current, population, autocorrelation,
eigenstates.

Ring Buffer
-----------

Output is buffered in a ring buffer of **5000 steps**. Flushed every
``OutputStrideTS`` steps or every 5000 steps, whichever comes first.

Time Evolution Data
-------------------

File: ``td/TimeEvolutionData_*.h5``

Dipole Table
~~~~~~~~~~~~

.. list-table::
   :widths: 8 20 20 52
   :header-rows: 1

   * - Col
     - 1D / 2D
     - 3D
     - Description
   * - 0
     - t
     - t
     - Time (a.u.)
   * - 1
     - Ex(t)
     - Ex(t)
     - Electric field x-component
   * - 2
     - Ey(t)
     - Ey(t)
     - Electric field y-component
   * - 3
     - --
     - Ez(t)
     - Electric field z-component
   * - 4
     - Dx
     - Dx
     - Dipole moment x = q * <psi\|x\|psi>
   * - 5
     - Dy
     - Dy
     - Dipole moment y
   * - 6
     - Ax
     - Dz / Az
     - Acceleration x / Dipole z / Acceleration z

Energy Table
~~~~~~~~~~~~

.. list-table::
   :widths: 10 90
   :header-rows: 1

   * - Col
     - Description
   * - 0
     - t -- time
   * - 1
     - Kin -- kinetic energy <psi\|K\|psi>
   * - 2
     - Pot -- potential energy <psi\|V\|psi>
   * - 3
     - Int -- interaction energy E(t)*d(t)
   * - 4
     - Tot -- total energy Kin + Pot + Int
   * - 5
     - InvMass -- <psi\|1/m(r)\|psi>
   * - 6
     - Norm -- <psi\|M\|psi> (norm conservation check)

Current Table
~~~~~~~~~~~~~

.. list-table::
   :widths: 10 90
   :header-rows: 1

   * - Col
     - Description
   * - 0
     - t
   * - 1
     - Lz -- angular momentum z-expectation
   * - 2
     - dGamma -- Berry phase increment per step
   * - 3--5
     - Jx_tot, Jy_tot, Jz_tot -- total currents
   * - 6--8
     - J_intra -- intra-band currents
   * - 9--11
     - J_inter -- inter-band currents
   * - 12--14
     - J_bc -- bound-continuum currents (remainder)

Population Table
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 90
   :header-rows: 1

   * - Col
     - Description
   * - 0
     - t -- time
   * - 1..N+1
     - P_n = \|c_n\|^2 -- population of bound state n

Autocorrelation
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 90
   :header-rows: 1

   * - Col
     - Description
   * - 0
     - t
   * - 1
     - Re<psi(0)|psi(t)>
   * - 2
     - Im<psi(0)|psi(t)>

Eigenstate Data
---------------

File: ``static/EigenData_*.h5``

**Datasets**: ``spectrum`` (1D, float64), ``psi_0``, ``psi_1``, ... (complex
vectors), ``run_metadata`` group with ``code_version``, ``input_file``, ``units``,
grid params, eigenvalue residuals. Each axis dataset (``knots_x``, ``knots_y``,
``knots_z``) has attributes: ``mult_start_end`` (2-int), ``SplineDegree``,
``nfuncs``.
