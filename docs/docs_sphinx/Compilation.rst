Compilation
===========

CMake configuration, dependencies, build targets, execution modes, and output structure for TDSE-Z.

Dependencies
------------

Requires PETSc, SLEPc, PetIGA, MPI, OpenBLAS, muParser.

Build Steps
-----------

::

    # Export PETSc before configuring
    export PETSC_DIR=/path/to/petsc
    cd fused-version
    mkdir build && cd build
    cmake .. # auto-detects PETSc/O3/march
    make -j $(nproc)
    # -> produces build/tdsez, build/tdmprocessor, build/tdmselect, build/drecprocessor

Auto-Detection
--------------

The build **auto-detects** PETSc's optimisation flags (-O3, -march, -mtune),
and OpenMP settings. TDSEZ inherits the same ABI and CPU target automatically -- no
manual flag tuning needed.

Build Targets
-------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Target
     - Description
   * - **tdsez**
     - Main TDSE solver -- runs full simulation pipeline
   * - **tdmprocessor**
     - Transition dipole matrix post-processor
   * - **tdmselect**
     - Selective dipole element extraction
   * - **drecprocessor**
     - Dipole recorder / spectrum analysis

Execution Modes
---------------

**Serial Execution**

::

    $ ./tdsez -inp inps/harmonic_oscillator_1d.prm

**Parallel Execution**

::

    $ mpirun --bind-to none -np 4 ./tdsez -inp inps/model.prm

**Eigenmode Only**

Solve TISE and produce eigenstates without time propagation.

::

    $ ./tdsez -inp model.prm -ts_max_steps 0

Output Directory Structure
--------------------------

::

    td/                  # time-evolution output
      TimeEvolutionData_<input>.h5  # time-series observables
      wfs_<input>.h5             # wavefunction snapshots
      ac_<input>.h5             # autocorrelation function
    static/                # eigenproblem output
      EigenData_<input>.h5    # energy spectrum + eigenstates
      GS_<input>.log         # degenerate-splitting log (2D)

Performance Note
----------------

.. warning::

   OpenMPI 5.x: ``mpirun -np 1`` defaults to ``--bind-to core``, pinning to
   one CPU and causing an 8--10x slowdown. Always use ``--bind-to none`` or
   ``--oversubscribe``.
