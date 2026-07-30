run
===

Run
===

Launch simulations, parallel execution, and output directory structure.

Serial Execution
----------------

::

    $ ./tdsez -inp inps/harmonic_oscillator_1d.prm

Parallel Execution
------------------

::

    $ mpirun --bind-to none -np 4 ./tdsez -inp inps/model.prm

GPU Mode
--------

Requires PETSc built with ``--with-cuda=1``.

::

    $ mpirun --bind-to none -np 4 ./tdsez -inp model.prm   \
      -vec_type cuda -mat_type aijcusparse

.. note::

   Non-GPU-aware MPI: If your MPI installation is not GPU-aware, add
   ``-use_gpu_aware_mpi 0`` to avoid hangs.

Eigenmode Only
--------------

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
