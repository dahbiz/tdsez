Troubleshooting
===============

Common errors, known issues, performance tips, and debugging strategies.

Unknown Input Key Fatal Error
-----------------------------

Set ``StrictInput = 0`` to tolerate unknown keys, or fix the typo. Physics codes
treat unknown keys as fatal because a typo'd potential or mass can silently produce
wrong physics.

Dimension Must Be 1, 2 or 3
----------------------------

Set ``Dimension = 1``, ``2``, or ``3`` explicitly. Default is 1.

SplineDegree Must Be in [1,14]
------------------------------

Set ``SplineDegree`` between 1 and 14.

Initial State Index >= Converged Eigenstates
--------------------------------------------

Increase ``NBoundStates`` to compute more eigenstates, or use a lower index in
``InitialState``.

8-10x Slowdown with mpirun
---------------------------

Use ``mpirun -np N --bind-to none``. OpenMPI's default ``--bind-to core`` pins
each process to one physical core.

Silent All-Zero Autocorrelation
-------------------------------

AC is computed BEFORE the per-step HDF5 flush. With ``OutputStrideTS=1`` the
flush fires at every step. This is correct in the current implementation. Verify
``OutputStrideAC`` is set independently.

"BDD Current: NOT CONSERVED"
----------------------------

The mass matrix test failed. ``Mass(x,y,z)`` is non-positive somewhere in the
domain. Check for negative or zero values.

Segmentation Fault During Propagation
-------------------------------------

When using xyz polarization, buffer size is bumped to 16 temporary vectors to
prevent segfault. Ensure ``OutputStrideTS > 0`` or ``OutputStrideWFS > 0`` if
you expect output.

Performance in 3D
-----------------

- **Assembly**: a constant ``Mass`` expression is detected automatically
  (``MassIsConstant``). All 2D/3D time-dependent operators, including the
  mass-gradient corrections of the force matrices for
  ``AccelerationMethod = operator`` with a varying mass, are sum-factorised;
  ``-tdsez_sumfact_check`` compares every element with the per-point kernels.
- **Eigensolver**: shift-invert near the bottom of a gapless spectrum (free or
  nearly free carriers) converges slowly, and the 3D direct factorisation is
  memory-bound. With an analytic initial state use ``NBoundStates = 0``. A
  failed inner solve or inaccurate pairs are reported (``WARNING: inner solve
  failed ... trying the next inner solver``) and the next inner solver runs;
  "Inner solver used" and "Max backward error" in the RESULTS block show what
  was accepted.
- **Memory**: every stored operator has :math:`(2p+1)^3` non-zeros per row
  (729 for degree 4), about 24 bytes each. With 8 ranks the 110 592-DOF degree-4
  benchmark peaks at 16–20 GB; past the physical memory the run swaps and
  mat-vecs slow down by an order of magnitude. Prefer degree 3 in 3D, set
  ``ObserveAllAxes = 0`` when only the driven axis is needed, and check
  "Peak memory" in the run summary.
- **Observables**: in 2D/3D M, the dipoles, the velocities and K are applied
  in factorised (Kronecker) form, checked against the assembled matrices at the
  first step (printed at start-up). <V>, the forces dV/dx_a, the CAP and, for
  a varying mass, <1/m>, <K> and the velocities are taken from psi at the
  quadrature points (also checked at the first step); ``-tdsez_quad_obs 0``
  restores the matrix products.
