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
     - Comma-separated list of ``position`` (dipole :math:`\langle q\mathbf r\rangle`),
       ``acceleration``, ``dipole`` (both), ``population``, ``energy``,
       ``current``, ``autocorrelation`` (or ``ac``) and ``wfs``; case and
       spaces are ignored. ``all`` or an empty value enables everything,
       ``none`` only the time axis. An unknown entry stops the run
       (``StrictInput = 1``).
   * - ObserveAllAxes
     - bool
     - 1
     - Observe the dipole, current and acceleration on every spatial axis.
       With 0 only the axes in ``Polarization`` are observed; the operators
       of the other axes are not assembled and their columns are zero.

A quantity missing from ``PhysicsOutput`` is not computed: its products and
reductions are skipped. Its dataset is still written with the same shape, the
time in column 0 and zeros elsewhere, so every ``td/<name>_observables.h5`` file has
the same layout and readers (zkit) always find the time axis. The quantities
that are computed are the same as in a full run (to rounding).

Each monitor call forms only the products the requested quantities read:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Entry
     - Products per step
   * - ``position``
     - dipole :math:`D_a\psi` on the observed axes
   * - ``acceleration``
     - operator form: force operators, :math:`\langle m^{-1}\rangle`;
       discrete form: :math:`H\psi` and one CG solve with :math:`M`
       (:doc:`Observables`)
   * - ``energy``
     - :math:`M\psi`, :math:`H\psi`, :math:`K\psi`, :math:`V\psi`, :math:`M_d\psi`, dipoles
   * - ``population``, ``ac``
     - :math:`M\psi` and the bound-state projections
   * - ``current``
     - velocities, :math:`L_z`, projections, Berry phase

:math:`\langle\psi|\psi\rangle` and :math:`\langle H\rangle` of the progress
log are computed at the logged steps only. On a 2D atom (27 225 DOFs, 4 ranks,
``pade22``, 400 steps) the monitors take

.. list-table::
   :widths: 30 25 25
   :header-rows: 1

   * - ``PhysicsOutput``
     - operator acceleration
     - discrete acceleration
   * - ``all``
     - 2.44 s (run 12.2 s)
     - 14.3 s (run 23.9 s)
   * - ``position``
     - 0.53 s (run 10.9 s)
     - 0.55 s (run 11.0 s)
   * - ``population``
     - 0.32 s
     - 0.33 s
   * - ``none``
     - 0.09 s
     - 0.10 s

The discrete acceleration (default for a position-dependent mass or a
``FieldTerm``) is the expensive observable: record it only when needed, e.g.
``PhysicsOutput = position, population`` for dynamics and ``acceleration`` for
HHG. Its mass solve starts from the previous step's solution (same
:math:`10^{-14}` tolerance).

``wfs`` enables the wavefunction snapshots controlled by ``OutputStrideWFS``.

Output Strides
--------------

The time series of :doc:`Observables`, including the autocorrelation, are
recorded in ``td/<name>_observables.h5`` every ``OutputStrideObs`` steps
(default 1: every step), independently of the other strides below.

.. list-table::
   :widths: 22 8 10 60
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - OutputStrideObs
     - int
     - 1
     - Compute and record the observables every N steps (rows at multiples of
       N); the propagation is unchanged. The run warns when the sampling
       frequency :math:`\pi/(N\Delta t)` is too low for the harmonic spectrum
       and prints the largest safe N (:doc:`Input-File`).
   * - CheckpointStride
     - int
     - 0
     - Write a restart file every N steps (:ref:`restart-files`).
   * - OutputStrideWFS
     - int
     - 100
     - Write the state to ``td/<name>_wfs.h5`` every N steps (dataset
       ``wavefunction``, time-step index :math:`\text{step}/N`); requires
       ``wfs`` in ``PhysicsOutput``. 0 disables the snapshots.
   * - OutputStrideTS
     - int
     - 0
     - If :math:`N > 0`: (i) ``td/<name>_observables.h5`` is also flushed at
       multiples of N steps when at least 2 s of wall time have passed since
       the last write; (ii) with ``Propagator = cn`` the 1D and 3D monitors
       write the state of the PETSc ``TS`` object (``TSView``) every N steps
       to ``td/<name>_ts.h5`` in the run directory. 0 disables both.
   * - OutputStrideSurff
     - int
     - 1
     - Sampling stride of the t-SURFF observer (:doc:`TSurff`).

The rows of ``td/<name>_observables.h5`` are buffered in memory and appended to the
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
     - Write the eigenstates to ``static/<name>_eigen.h5`` (datasets
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
       ``static/<name>_Dx.bin`` (PETSc binary).
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
     - Compress the HDF5 output. The ``<name>_observables.h5`` datasets
       (chunks of 256 rows) use the shuffle filter and deflate at
       ``HDF5CompressLevel``; the files written through PETSc viewers
       (``_eigen``, ``_wfs``, ``_ts``) use PETSc's HDF5 compression.
   * - HDF5CompressLevel
     - int
     - 6
     - Deflate level of the ``<name>_observables.h5`` datasets, clamped to 1-9.
   * - Verbose
     - bool
     - 1
     - With 1 the propagation prints a progress line every step; with 0
       about twenty lines over the run (:doc:`Time-Propagation`).

Terminal log
------------

Rank 0 prints the run as a sequence of stages, each under a centred title
between two rules: GRID AND BASIS, STATIONARY PROBLEM, TIME-DEPENDENT
OPERATORS, PROPAGATION and TIMING SUMMARY. Inside a stage, centred
``──── LABEL ────`` lines open its parts (for example EIGENSOLVER,
EIGENVALUES, COMPLEX ABSORBING POTENTIAL, PROGRESS). Settings and results are
``Label = value`` rows and notes start with ``ⵣ``; warnings
(``ⵣ WARNING:``) and fatal errors are centred with a blank line around them;
long text is wrapped. The body is a column centred under the
rules, which span the terminal (70 to 94 columns) or 80 columns when the
output is redirected to a file. Errors that stop the run start with
``✖ TDSEZ FATAL:``.

The EIGENVALUES table lists each state with its energy in atomic units
(``%+.5e``), in eV with two decimals, and its backward error
(:doc:`Eigensolver`).

The TIMING SUMMARY lists the wall time of every phase (slowest rank), its
share of the run and the ratio of the slowest to the fastest rank, with the
parts of a phase indented below it: the eigensolver setup, the Krylov-Schur
iterations and the degenerate-shell step; each time-dependent operator; the
bound-state dipole and momentum matrices and the gauge check; the time
stepping (linear solves, preconditioner setup, operator updates) and the
observables. The PERFORMANCE part that follows gives the time per step, the
solver counts and the peak memory (:doc:`Performance`).
