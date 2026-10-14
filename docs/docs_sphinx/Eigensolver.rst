Eigensolver
===========

SLEPc EPS configuration, bound states, and solve mode.

Bound States
------------

.. list-table::
   :widths: 35 15 10 40
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - NBoundStates
     - int
     - 1
     - Number of bound states to compute (stored as states 0..N-1). If the
       last requested state belongs to a degenerate level, the whole level is
       kept (N > NBoundStates, with a printed note). Aliases: ``NumberOfBoundStates``
   * - TargetEigenvalue / TargetEnergy
     - real
     - 0.0
     - Target eigenvalue for SLEPc EPS. States near this energy are prioritized

Degenerate Levels and Reproducible States
-----------------------------------------

Inside a degenerate level any orthonormal basis is an eigenbasis, and the one
SLEPc returns depends on rounding (assembly order, MPI layout, BLAS). Per-state
populations, currents and saved wavefunctions would then change from run to
run. TDSE-Z therefore post-processes the converged pairs:

1. States are ordered by distance to ``TargetEigenvalue`` (ties by energy). The
   first ``NBoundStates`` are kept, extended to the end of a degenerate level.
2. Krylov solvers can miss copies of a repeated eigenvalue. Whenever the kept
   set contains a degenerate level, or ends at the last converged pair, the
   kept states are deflated and SLEPc is asked for the nearest remaining pair.
   If it is not farther from the target, or has the energy of a kept level, it
   was missed and is added (``Degenerate-shell check: added ...``). The check
   costs about one extra eigensolve and runs only when degenerate levels are
   kept (or when SLEPc converged no pair beyond the kept set).
3. Inside each degenerate level the states are rotated to diagonalise a fixed
   generic weight (a quadratic polynomial of position at the Greville points),
   sorted by that weight. Energies and the spanned subspace are unchanged.
4. Every state gets a fixed phase: real, with a positive overlap with
   :math:`\prod_d (1 + 0.3\hat x_d)` (or, when that overlap vanishes, with
   :math:`\prod_d e^{a_d \hat x_d}`).

The resulting states, and everything computed from them, agree across assembly
paths and MPI rank counts to rounding. Two eigenvalues count as degenerate when
:math:`|E_i - E_j| \le \mathrm{tol}\,\max(1, |E_i|)`, with ``tol`` set by
``-tdsez_degeneracy_tol`` (default ``1e-9``).

Solve Mode
----------

.. list-table::
   :widths: 40 15 10 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - UseDirectSolve / DirectSolve
     - bool
     - false
     - Use direct factorization instead of iterative eigensolver

Saving Eigenstates
------------------

.. list-table::
   :widths: 45 15 10 30
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - NBoundStatesSave / SaveBoundStates
     - bool
     - false
     - Save eigenstates to ``static/EigenData_*.h5``
   * - BoundStateFormat / StateFormat / SaveStateFormat
     - string
     - ``complex``
     - Save format: ``complex`` (eigenvectors in the fixed gauge above) or ``real``
       (phase-rotated to real canonical form via TDSEZMakeStateReal2)
