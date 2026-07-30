troubleshooting
===============

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

Performance Tip
---------------

Constant mass is much faster. Set ``MassIsConstant = 1`` to disable finite-difference
gradient computations of 1/m -- the dominant cost in the variable-mass assembly path.
