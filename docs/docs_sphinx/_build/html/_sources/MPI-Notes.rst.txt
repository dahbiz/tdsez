MPI Notes
=========

OpenMPI 5.x bind-to pitfalls and parallel best practices.

OpenMPI 5.x bind-to Pitfall
---------------------------

With OpenMPI 5.x, ``mpirun -np 1`` defaults to ``--bind-to core``, pinning the
process to a single physical core. PETSc/SLEPc internal BLAS threads then compete
for one core -- **8-10x slowdown**.

::

    $ mpirun --bind-to none -np 1 ./tdsez.x -inp model.prm
    $ # or equivalently:
    $ mpirun --oversubscribe -np 1 ./tdsez.x -inp model.prm

Also Applies to Pytest
----------------------

tests/test_tdsez.py ``_run`` helper uses ``mpirun --bind-to none`` -- without it
the 3D HO test takes 80s+ instead of ~10s.

Load Balance
------------

PetIGA splits the elements of each axis evenly over the ranks of that axis
(leftover elements go to the first ranks). A rank owns one basis function per
element, and the last rank of each axis also owns the :math:`p` trailing ones.
Ranks therefore own different numbers of rows, and the solves (sparse products,
ILU triangular solves, global dot products every GMRES iteration) wait for the
largest. With degree 3 on 2x2x2 ranks:

.. list-table::
   :widths: 30 20 20 15 15
   :header-rows: 1

   * - Elements per axis
     - first rank owns
     - last rank owns
     - ratio per axis
     - ratio in 3D
   * - 32
     - 16
     - 16 + 3 = 19
     - 1.19
     - 1.67
   * - 33
     - 17
     - 16 + 3 = 19
     - 1.12
     - 1.40
   * - 200 (100 per rank)
     - 100
     - 103
     - 1.03
     - 1.09

On the 32-element case the ASM subdomains measured 19\ :sup:`3` to
22\ :sup:`3` rows (with overlap) and the ILU solves had a max/min time ratio of
1.6. Choosing the element count can move one element per axis, not :math:`p`;
the imbalance only matters for small subdomains, so use enough elements per rank
(or fewer ranks for small problems).

**Rebalancing was tried and not adopted.** Moving the element cuts so that every
rank owns the same number of basis functions (after PetIGA's partitioning
stage) made time stepping 7-12 % faster, but put the element-based work
(quadrature observables, operator assembly) out of balance: monitoring up to
60 % and assembly 5-23 % slower. The whole run changed by 0-5 %, within the timing
noise, and the change relied on PetIGA's internal partition fields. With
:math:`w` the fraction of the :math:`p` trailing basis functions moved off the
last rank (3D, 32k DOFs, 8 ranks, two runs each):

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - :math:`w`
     - time stepping
     - monitoring
     - total
   * - 0 (PetIGA)
     - 2.68 / 2.97 s
     - 1.14 / 1.33 s
     - 9.11 / 9.91 s
   * - 0.5
     - 2.50 / 2.60 s
     - 1.33 / 1.35 s
     - 9.11 / 9.46 s
   * - 1
     - 3.60 / 2.56 s
     - 1.83 / 1.60 s
     - 11.29 / 10.07 s

**Waiting cannot be hidden by MPI.** The waiting is the symptom: a rank with
more rows does more arithmetic, and the others idle at the next collective
whether it is blocking or not (``MPI_Wait`` after ``MPI_Iallreduce``). PETSc
already overlaps the ghost exchange of each sparse product with the local part;
pipelined GMRES (see Time Propagation) overlaps the reductions and is slower on
one machine.

**Mixed core types.** Apple M-series processors combine performance and
efficiency cores (M4 Pro: 8 + 4). With ``--bind-to none`` and more ranks than
performance cores some ranks run on efficiency cores and the others wait for
them; use at most as many ranks as performance cores.

