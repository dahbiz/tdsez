MPI Notes
=========

OpenMPI 5.x bind-to pitfalls, GPU-aware MPI, parallel best practices.

OpenMPI 5.x bind-to Pitfall
---------------------------

With OpenMPI 5.x, ``mpirun -np 1`` defaults to ``--bind-to core``, pinning the
process to a single physical core. PETSc/SLEPc internal BLAS threads then compete
for one core -- **8-10x slowdown**.

::

    $ mpirun --bind-to none -np 1 ./tdsez -inp model.prm
    $ # or equivalently:
    $ mpirun --oversubscribe -np 1 ./tdsez -inp model.prm

Also Applies to Pytest
----------------------

tests/test_tdsez.py ``_run`` helper uses ``mpirun --bind-to none`` -- without it
the 3D HO test takes 80s+ instead of ~10s.

GPU-Aware MPI
-------------

If your MPI is not GPU-aware, add ``-use_gpu_aware_mpi 0`` to avoid hangs when
running on GPU vectors/matrices.
