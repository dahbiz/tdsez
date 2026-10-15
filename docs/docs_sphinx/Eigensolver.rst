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
     - Solve the shift-invert systems with a direct factorisation first

SLEPc's Krylov-Schur solver runs on the shift-invert operator
:math:`(H-\sigma M)^{-1}M` (GHEP, eigenvalues nearest ``TargetEigenvalue``;
the shift :math:`\sigma` sits :math:`10^{-3}\max(1,|\mathrm{target}|)` below
the target so that a target on an eigenvalue does not make
:math:`H-\sigma M` singular). Each application solves a system with
:math:`H-\sigma M`, the inner solve:

* **direct**: LU factorisation (MUMPS; SuperLU_DIST or PETSc's LU when MUMPS
  is missing). Exact and factorised once. Default in 1D and 2D (up to 2M
  DOFs) and in 3D for degree :math:`\ge 6` up to 30k DOFs, where the
  overlapping ASM blocks would cover most of the domain.
* **iterative**: FGMRES with restricted additive Schwarz (overlap 1), ILU(0)
  on each block with in-block shifts against zero pivots of the indefinite
  :math:`H-\sigma M`, relative tolerance :math:`10^{-10}`. Default in 3D.

Every attempt is checked: the inner solve converged, ``NBoundStates`` pairs
converged, and every backward error
:math:`\|Hv-EMv\|/((\|H\|_\infty+|E|\,\|M\|_\infty)\|v\|)` is below
:math:`10^{-8}` (the eigensolver tolerance is :math:`10^{-12}`). If fewer
pairs converged the attempt is repeated once with a twice larger subspace;
otherwise the next inner solver runs: iterative :math:`\to` direct (3D up to
300k DOFs) :math:`\to` FGMRES with ASM overlap 2 and ILU(1). The run stops
only when no pair converges at all. ``-tdsez_eps_inner direct|iterative``
chooses the first inner solver; ``-st_ksp_*``, ``-st_pc_*`` and
``-st_sub_pc_*`` options apply on top of the defaults.

The printed residual of each state (``res``) and ``eig_residual`` in
``EigenData_*.h5`` are this backward error, which stays meaningful for
:math:`E\to 0` (e.g. the constant ground state of a free particle with
natural boundaries).

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
