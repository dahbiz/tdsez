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
     - Dipole moment x = :math:`q \langle \psi | x | \psi \rangle`
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
     - Kin -- kinetic energy :math:`\langle \psi | K | \psi \rangle`
   * - 2
     - Pot -- potential energy :math:`\langle \psi | V | \psi \rangle`
   * - 3
     - Int -- interaction energy :math:`E(t) \cdot d(t)`
   * - 4
     - Tot -- total energy :math:`E_{\text{total}} = \text{Kin} + \text{Pot} + \text{Int}`
   * - 5
     - InvMass -- :math:`\langle \psi | 1/m(\mathbf{r}) | \psi \rangle`
   * - 6
     - Norm -- :math:`\langle \psi | M | \psi \rangle` (norm conservation check)

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
     - Lz -- angular momentum z-expectation :math:`\langle L_z \rangle`
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
     - :math:`P_n = |c_n|^2` -- population of bound state n

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
     - :math:`\text{Re} \langle \psi(0) | \psi(t) \rangle`
   * - 2
     - :math:`\text{Im} \langle \psi(0) | \psi(t) \rangle`

Eigenstate Data
---------------

File: ``static/EigenData_*.h5``

**Datasets**: ``spectrum`` (1D, float64), ``psi_0``, ``psi_1``, ... (complex
vectors), ``run_metadata`` group with ``code_version``, ``input_file``, ``units``,
grid params, eigenvalue residuals. Each axis dataset (``knots_x``, ``knots_y``,
``knots_z``) has attributes: ``mult_start_end`` (2-int), ``SplineDegree``,
``nfuncs``.
