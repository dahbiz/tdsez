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
  (``MassIsConstant``) and uses the fast assembly. A position-dependent mass
  computes finite-difference derivatives of 1/m at every quadrature point,
  which dominates the 3D operator assembly.
- **Eigensolver**: shift-invert near the bottom of a gapless spectrum (free or
  nearly free carriers) converges slowly, and the 3D direct factorisation is
  memory-bound. With an analytic initial state use ``NBoundStates = 0``.
- **Memory**: every stored operator has :math:`(2p+1)^3` non-zeros per row
  (729 for degree 4), about 24 bytes each. With 8 ranks the 110 592-DOF degree-4
  benchmark peaks at 16–20 GB; past the physical memory the run swaps and
  mat-vecs slow down by an order of magnitude. Prefer degree 3 in 3D, set
  ``ObserveAllAxes = 0`` when only the driven axis is needed, and check
  "Peak memory" in the run summary.
- **Observables**: in 3D the per-step observables (about 14 sparse products)
  cost more than the time step itself.
