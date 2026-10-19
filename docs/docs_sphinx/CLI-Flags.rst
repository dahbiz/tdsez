CLI Flags
=========

The physical problem is defined in the input file (:doc:`Input-File`). The
command line selects the input file, overrides two input keys, and passes
TDSE-Z diagnostic switches and PETSc, SLEPc and PetIGA options
[PETSc]_ [Hernandez2005]_ [Dalcin2016]_.

Basic usage
-----------

::

    $ mpiexec -n 4 ./tdsez.x -inp run.prm
    $ ./tdsez.x -inp run.prm -save_dipole xyz
    $ ./tdsez.x -inp run.prm -enable_gpu 1
    $ ./tdsez.x -help

For a bound-state calculation without propagation set
``EnablePropagation = 0`` in the input file.

Option handling
---------------

Input-file keys cannot be set on the command line, with two exceptions:
``-enable_gpu`` overrides ``EnableGPU`` and ``-save_dipole`` overrides
``SaveDipoleAxes``. These two flags and ``-mat_type``/``-vec_type`` are
removed from the argument list before PETSc is initialised. All other options
go to the PETSc options database and are read by the object that uses them.

Program flags
-------------

.. list-table::
   :widths: 30 12 58
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``-inp <file>``
     - string
     - Input file (default ``tdse.prm``). Output file names are derived from
       its base name.
   * - ``-enable_gpu <0|1>``
     - int
     - Overrides ``EnableGPU``.
   * - ``-save_dipole <axes>``
     - string
     - Overrides ``SaveDipoleAxes``: any combination of ``x``, ``y``, ``z``,
       ``all`` or ``none``. The requested dipole operators are written to
       ``static/<name>_D<a>.bin``; only assembled operators are written.
   * - ``-mat_type <type>``, ``-vec_type <type>``
     - string
     - Matrix and vector types of all PetIGA objects; passed on as
       ``-iga_mat_type`` and ``-iga_vec_type``.

Assembly and eigensolver flags
------------------------------

.. list-table::
   :widths: 30 12 58
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``-tdsez_sumfact <bool>``
     - bool
     - Sum-factorised element kernels (default 1); 0 uses the per-point
       kernels.
   * - ``-tdsez_sumfact_check``
     - flag
     - Compute every element matrix both ways and print the largest relative
       difference (diagnostic, slow).
   * - ``-tdsez_band <bool>``
     - bool
     - Banded accumulation of the element matrices with one insertion per
       matrix row (default 1); 0 inserts element by element.
   * - ``-tdsez_assembly_buffer_mb <MB>``
     - real
     - Memory budget per rank for the banded accumulation (default 1024).
       Larger sets of matrices are assembled in several element passes.
   * - ``-tdsez_eps_inner <mode>``
     - string
     - First inner solver of the shift-invert eigensolve: ``direct`` or
       ``iterative``. By default a direct solver is used when a direct
       package is available and the problem is small enough (1D and 2D up to
       :math:`2\cdot 10^6` rows; 3D with degree :math:`\ge 6` and at most
       :math:`3\cdot 10^4` rows), iterative otherwise. A failed attempt falls
       back to the next solver. See :doc:`Eigensolver`.
   * - ``-tdsez_degeneracy_tol <tol>``
     - real
     - Relative energy tolerance for treating eigenvalues as one degenerate
       level (default ``1e-9``); see :doc:`Eigensolver`.

The eigensolver also reads the SLEPc options (``-eps_*``, ``-st_*``,
``-st_ksp_*``, ``-st_pc_*``, ``-st_sub_*``), which apply on top of the
settings chosen by TDSE-Z.

Propagation flags
-----------------

.. list-table::
   :widths: 30 12 58
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``-tdsez_cap_check``
     - bool
     - Also assemble the CAP with the element kernel and print the largest
       difference to the separable assembly (diagnostic).
   * - ``-tdsez_tensor_ops <bool>``
     - bool
     - Apply :math:`M`, the dipoles, the velocities and :math:`K` in the
       monitors in factorised (Kronecker) form (default 1, 2D/3D, not with
       Dirichlet walls); 0 uses the assembled sparse matrices.
   * - ``-tdsez_quad_obs <bool>``
     - bool
     - Evaluate :math:`\langle V\rangle`, :math:`\langle 1/m\rangle`,
       :math:`\langle K\rangle`, the CAP, the forces
       :math:`\partial V/\partial x_a` and the velocities of the monitors from
       :math:`\psi` at the quadrature points (default 1, 2D/3D, not with
       Dirichlet walls); 0 uses the assembled matrices.
   * - ``-tdsez_mat_free``
     - bool
     - Matrix-free monitors and diagnostics (2D/3D, no Dirichlet walls,
       B-spline basis, CPU): :math:`K`, :math:`V`, :math:`M_{1/m}`, the forces,
       the dipoles of axes not coupled to the field and, for a constant mass,
       the velocities are never assembled; their factorised and
       quadrature-point forms (``-tdsez_tensor_ops``, ``-tdsez_quad_obs``) are
       used directly, and the Padé/Magnus stage matrices are formed from
       :math:`\mathbf M`, :math:`\mathbf H` and :math:`\mathbf D` without
       cached field-free parts. :math:`\mathbf H`, :math:`\mathbf M`, the CAP
       and the coupled dipoles stay assembled. Outputs equal the default run
       to rounding (:doc:`Performance`).
   * - ``-tdsez_estimate_memory <N>``
     - int
     - Dry run: predict the memory per rank of this input on ``N`` ranks and
       stop (nothing is assembled; seconds on one rank). Uses the knots of the
       run and PetIGA's processor grid, so the rows and nonzeros of every rank
       are exact, and sums the objects alive in the eigensolve, operator
       assembly and propagation phases (matrices with the pattern of H,
       overlapped ASM blocks, ILU(0) factors, Krylov vectors). Pass the options
       of the production run (``-tdsez_mat_free``, ``-st_matmode``,
       ``-pade_stage1_pc_type`` ...). With ``-tdsez_estimate_ranks_per_node``
       and ``-tdsez_estimate_node_mem <GB>`` it reports whether a node fits.
       ``-tdsez_estimate_base_mb`` is the per-rank runtime baseline (default
       250, calibrated on a Linux cluster; about 550 on macOS). Calibration:
       within 3% of the measured peak for 3D runs with ``-tdsez_mat_free``;
       conservative (up to 50% high) with assembled monitors. It reports the
       range over ranks of the rows and of the peak, and suggests options when
       a node does not fit.
   * - ``-tdsez_balance <bool>``
     - bool
     - Node-balanced partition (default 1). PetIGA splits the elements evenly,
       so the last part of every axis also owns the :math:`p` trailing basis
       functions and the corner rank owns :math:`(n+p)^3` rows instead of
       :math:`n^3` (1.25--1.6 times the mean at :math:`p=3`--5). The element
       boundaries are moved so that ``owned + c * overlap`` is equal on every
       part; the processor grid is PetIGA's. 18% less time per step in a 4-rank
       3D test, outputs identical to rounding. 0 restores PetIGA's split.
   * - ``-tdsez_balance_overlap <c>``
     - real
     - Weight of the ASM overlap / ghost layer in the balance (default 0.5):
       the end parts of an axis, which have the overlap on one side only, get
       :math:`c\,p` more nodes than the interior parts. 0 balances the rows.
   * - ``-tdsez_stage_ilu_inplace <bool>``
     - bool
     - ``pade22``/``magnus4``: factor each ASM block in place (ILU(0) only),
       one overlapped block per stage instead of a copy and its factor, about a
       quarter of the propagation memory; the stage solves are ~20% slower
       (plain AIJ triangular solves). Default 0; for runs that do not fit
       otherwise. A preconditioner refresh then rebuilds the stage solver.
       Implies ``-tdsez_eps_ilu_inplace``.
   * - ``-tdsez_eps_ilu_inplace <bool>``
     - bool
     - Factor the ASM blocks of the eigensolver's inner solves in place
       (ILU(0), first attempt only; the shift-invert operator does not change
       during the eigensolve). One overlapped block per rank instead of two;
       inner solves ~20% slower. Default: the value of
       ``-tdsez_stage_ilu_inplace``.
   * - ``-tdsez_pc_refresh_its <n>``
     - int
     - Also rebuild the propagator preconditioner whenever a solve takes more
       than ``n`` iterations (default: only when the count exceeds twice the
       count after the last build plus 5).
   * - ``-tdsez_ilu_levels <n>``
     - int
     - ILU fill level of the ASM subdomain blocks of the propagator
       preconditioner (default 0). Level 1 saves at most one GMRES iteration,
       while its fill makes 3D factorisations about ten times slower.
   * - ``-pade_guess_order <q>``
     - int
     - Order of the extrapolated initial guess of the ``pade22`` and
       ``magnus4`` stage solves, 1 to 5 (default 3).
   * - ``-pade_extrapolate_guess <bool>``
     - bool
     - Extrapolated initial guess of the ``pade22`` and ``magnus4`` stage
       solves (default 1); 0 disables it.

``Propagator = cn`` runs on a PETSc ``TS`` and reads the ``-ts_*``,
``-snes_*``, ``-ksp_*`` and ``-pc_*`` options; ``-ts_max_steps`` affects only
this propagator. See :doc:`Time-Propagation`.

Environment variables
---------------------

``TDSEZ_DUMP_KNOTS``
   When set, the interior knots of a non-uniform x axis are printed after the knot
   vector has been generated (:doc:`Knot-Sequences`).

Help
----

``-help`` lists the PETSc, SLEPc and PetIGA options that the run queries.
