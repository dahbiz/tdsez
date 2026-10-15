HDF5 Output Schema
==================

Complete dataset layout: dipole, energy, current, population, autocorrelation,
eigenstates.

Ring Buffer
-----------

Rows are buffered in memory and appended to the file at most every 2 s of wall
time (at multiples of ``OutputStrideTS``), every 5000 steps, whenever the buffer
is full, and at the end of the run. Every step is recorded; the buffering only
affects when rows reach the file.

Time Evolution Data
-------------------

File: ``td/TimeEvolutionData_*.h5``

Dipole Table
~~~~~~~~~~~~

Dataset ``dipoles``, one row per step. With :math:`q` = ``Charge``:

- **1D** (4 columns): ``t, Ex, Dx, Ax``
- **2D** (7 columns): ``t, Ex, Ey, Dx, Dy, Ax, Ay``
- **3D** (10 columns): ``t, Ex, Ey, Ez, Dx, Dy, Dz, Ax, Ay, Az``

where ``E*`` is the field at time ``t``, :math:`D_a = q\langle x_a\rangle` the
dipole and :math:`A_a = q\, d\langle v_a\rangle/dt` the dipole acceleration
(:math:`= \ddot D_a`; see :doc:`Observables`). All spatial axes are recorded,
including those not in ``Polarization`` (cross-polarized and asymmetric
response); with ``ObserveAllAxes = 0`` only the polarised axes are computed and
the other columns are zero.

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
     - Int -- interaction energy :math:`\mathbf E(t) \cdot \langle \mathbf r \rangle`, the expectation of the coupling term
   * - 4
     - Tot -- total energy :math:`E_{\text{total}} = \text{Kin} + \text{Pot} + \text{Int}`
   * - 5
     - InvMass -- :math:`\langle \psi | 1/m(\mathbf{r}) | \psi \rangle`
   * - 6
     - Norm -- :math:`\langle \psi | M | \psi \rangle` (norm conservation check)

Current Table
~~~~~~~~~~~~~

Dataset ``currents``. Columns 0–2 are ``t``, :math:`\langle L_z\rangle` (2D/3D;
0 in 1D) and the Berry-phase increment :math:`d\Gamma`. The remaining columns are
the current :math:`J_a = q\langle v_a\rangle` (:math:`= \dot D_a`) and its
decomposition into intra-band, inter-band and bound–continuum parts:

- **1D and 2D** (11 columns): ``t, Lz, dGamma, Jx, Jy, Jx_intra, Jy_intra,
  Jx_inter, Jy_inter, Jx_bc, Jy_bc`` (``Jy*`` are 0 in 1D)
- **3D** (15 columns): ``t, Lz, dGamma, Jx, Jy, Jz, Jx_intra, Jy_intra,
  Jz_intra, Jx_inter, Jy_inter, Jz_inter, Jx_bc, Jy_bc, Jz_bc``

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

t-SURFF Spectra
---------------

File: ``td/surff_*.h5`` (only with ``TSurff = 1``). Datasets ``momenta``,
``amplitude`` (Re, Im), ``pes`` :math:`=|b(\mathbf k)|^2`, on a grid also
``k_x``/``k_y``/``k_z`` and ``pes_grid``; ``amplitude_flux`` with
``SurffInterior = 1``. Attributes ``final_time``, ``radius``,
``exterior_mass``, ``exterior_potential``, ``charge``, ``A_final``,
``stride``, ``n_surface_points``, ``n_samples``, ``interior``. See
:doc:`TSurff`.

Eigenstate Data
---------------

File: ``static/EigenData_*.h5``

**Datasets**: ``spectrum`` (1D, float64), ``psi_0``, ``psi_1``, ... (complex
vectors), ``run_metadata`` group with ``code_version``, ``input_file``, ``units``,
grid params, eigenvalue residuals. Each axis dataset (``knots_x``, ``knots_y``,
``knots_z``) has attributes: ``mult_start_end`` (2-int), ``SplineDegree``,
``nfuncs``.
