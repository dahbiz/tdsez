cli-flags
=========

CLI Flags
=========

Command-line overrides for all input parameters and PETSc options.

All flags override input-file values.

::

    $ ./tdsez -inp file.prm          # specify input file
    $ ./tdsez -ts_max_steps 0         # eigenvalues only, no propagation
    $ ./tdsez -vec_type cuda          # GPU vectors
    $ ./tdsez -mat_type aijcusparse   # GPU matrices
    $ ./tdsez -use_gpu_aware_mpi 0    # non-GPU-aware MPI workaround
    $ ./tdsez -save_dipole xyz        # save dipole axes at runtime
    $ ./tdsez -enable_gpu 1           # enable GPU acceleration
    $ ./tdsez -help                   # full PETSc options dump

Note: ``-mat_type`` and ``-vec_type`` are re-injected as ``-iga_mat_type`` /
``-iga_vec_type`` to reach PetIGA. The solver strips PETSc-known flags before
processing.
