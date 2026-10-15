Output Control
==============

These keys select the observables computed during propagation, the snapshot
and flush strides, the static output for post-processing, and HDF5
compression. The files themselves are described in :doc:`HDF5-Output`.

Observable Selection
--------------------

.. list-table::
   :widths: 28 10 12 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - PhysicsOutput / OutputQuantities
     - string
     - ``all``
     - Comma-separated list of ``dipole``, ``population``, ``energy``,
       ``current``, ``autocorrelation`` (or ``ac``) and ``wfs``; case and
       spaces are ignored. ``all`` or an empty value enables everything.
   * - ObserveAllAxes
     - bool
     - 1
     - Observe the dipole, current and acceleration on every spatial axis.
       With 0 only the axes in ``Polarization`` are observed; the operators
       of the other axes are not assembled and their columns are zero.

A quantity missing from ``PhysicsOutput`` is not computed: its products and
reductions are skipped. Its dataset is still written, filled with zeros, so
that every ``TimeEvolutionData`` file has the same datasets and shapes. The
products needed by several quantities are formed when any of them is enabled
(the inverse-mass expectation value for ``energy`` or ``dipole``, the
bound-state projections for ``population`` or ``current``). ``wfs`` enables
the wavefunction snapshots controlled by ``OutputStrideWFS``.

Output Strides
--------------

The time series of :doc:`Observables`, including the autocorrelation, are
recorded at every step in ``td/TimeEvolutionData_<input>.h5``, independently
of the strides below.

.. list-table::
   :widths: 22 8 10 60
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - OutputStrideWFS
     - int
     - 100
     - Write the state to ``td/wfs_<input>.h5`` every N steps (dataset
       ``wavefunction``, time-step index :math:`\text{step}/N`); requires
       ``wfs`` in ``PhysicsOutput``. 0 disables the snapshots.
   * - OutputStrideTS
     - int
     - 0
     - If :math:`N > 0`: (i) ``TimeEvolutionData`` is also flushed at
       multiples of N steps when at least 2 s of wall time have passed since
       the last write; (ii) with ``Propagator = cn`` the 1D and 3D monitors
       write the state of the PETSc ``TS`` object (``TSView``) every N steps
       to ``ts_<input>.h5`` in the run directory. 0 disables both.
   * - OutputStrideSurff
     - int
     - 1
     - Sampling stride of the t-SURFF observer (:doc:`TSurff`).

The rows of ``TimeEvolutionData`` are buffered in memory and appended to the
file when the buffer is full, every 5000 recorded steps, at multiples of
``OutputStrideTS`` as above, and at the end of the run. A small
``OutputStrideTS`` therefore makes partial results visible in the file during
a long run without changing what is recorded.

Static Output
-------------

.. list-table::
   :widths: 28 10 12 50
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - NBoundStatesSave / SaveBoundStates
     - bool
     - 0
     - Write the eigenstates to ``static/EigenData_<input>.h5`` (datasets
       ``psi_i`` and ``coeff_i``). Required by the post-processing tools.
   * - BoundStateFormat / StateFormat / SaveStateFormat
     - string
     - ``complex``
     - ``complex`` writes the solver eigenvectors; ``real`` writes them
       rotated to a real canonical phase.
   * - SaveDipoleMatrix
     - bool
     - 0
     - Write the assembled position matrix :math:`\mathbf D_x` to
       ``static/Dx_<input>.bin`` (PETSc binary).
   * - SaveDipoleAxes
     - string
     - (empty)
     - Axes to write instead: ``x``, ``xy``, ``xz``, ``xyz``, ``all`` or
       ``none``; overrides ``SaveDipoleMatrix``. The command-line flag
       ``-save_dipole`` (:doc:`CLI-Flags`) overrides this key. Only axes
       whose operators are assembled are written.

Compression and Console Output
------------------------------

.. list-table::
   :widths: 22 8 10 60
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - HDF5Compress
     - bool
     - 0
     - Compress the HDF5 output. The ``TimeEvolutionData`` datasets
       (chunks of 256 rows) use the shuffle filter and deflate at
       ``HDF5CompressLevel``; the files written through PETSc viewers
       (``EigenData``, ``wfs``, ``ts``) use PETSc's HDF5 compression.
   * - HDF5CompressLevel
     - int
     - 6
     - Deflate level of the ``TimeEvolutionData`` datasets, clamped to 1-9.
   * - Verbose
     - bool
     - 1
     - With 1 the propagation prints a progress line every step; with 0
       about twenty lines over the run (:doc:`Time-Propagation`).
