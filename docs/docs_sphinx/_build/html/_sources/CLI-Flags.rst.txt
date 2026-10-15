CLI Flags
=========

Command-line overrides for all input parameters and PETSc options. CLI flags take
precedence over values in the input file.

Basic Usage
-----------

::

    $ ./tdsez.x -inp file.prm          # specify input file
    $ ./tdsez.x -ts_max_steps 0         # eigenvalues only, no propagation
    $ ./tdsez.x -save_dipole xyz        # save dipole axes at runtime
    $ ./tdsez.x -help                   # full PETSc options dump

Option Precedence
-----------------

1. Default values (hardcoded in code)
2. Input file values
3. Command-line flags (highest priority)

For example, if the input file sets ``EnablePropagation = 1`` and you run with
``-ts_max_steps 0``, propagation will be skipped because the CLI flag wins.

Key CLI Flags
-------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``-inp <file>``
     - string
     - Input parameter file (required for non-interactive runs)
   * - ``-ts_max_steps <n>``
     - int
     - Maximum time steps. Set to 0 for eigenvalue solve only (no propagation).
   * - ``-save_dipole <axes>``
     - string
     - Which dipole axes to save at runtime. Accepts ``x``, ``xy``, ``xyz``, ``all``, or ``none``.

Assembly and Eigensolver Flags
------------------------------

.. list-table::
   :widths: 30 12 58
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``-tdsez_sumfact 0``
     - bool
     - Use the per-point element kernels instead of sum factorisation.
   * - ``-tdsez_sumfact_check``
     - flag
     - Compute every element matrix both ways and print the largest relative
       difference (diagnostic, slow).
   * - ``-tdsez_band 0``
     - bool
     - Insert element matrices one element at a time instead of the banded
       accumulation (one insertion per matrix row).
   * - ``-tdsez_assembly_buffer_mb <MB>``
     - real
     - Memory budget per rank for the banded accumulation (default 1024). Larger
       sets of matrices are assembled in several element passes.
   * - ``-tdsez_degeneracy_tol <tol>``
     - real
     - Relative energy tolerance for treating eigenvalues as one degenerate
       level (default ``1e-9``); see :doc:`Eigensolver`.

Propagation Flags
-----------------

.. list-table::
   :widths: 30 12 58
   :header-rows: 1

   * - Flag
     - Type
     - Description
   * - ``-tdsez_cap_check``
     - flag
     - Also assemble the CAP with the element kernel and print the largest
       difference to the separable assembly (diagnostic).
   * - ``-tdsez_tensor_ops 0``
     - bool
     - Apply M, the dipoles, the velocities and K in the monitors as assembled
       sparse matrices instead of in factorised (Kronecker) form (2D/3D).
   * - ``-tdsez_ilu_levels <n>``
     - int
     - ILU fill level of the ASM subdomain preconditioner of every propagator
       (default 0). Level 1 saves at most one GMRES iteration but its fill makes
       3D factorisations about 10x slower; the 1e-12 solve tolerance is the same.

Note: ``-mat_type`` and ``-vec_type`` are re-injected as ``-iga_mat_type`` /
``-iga_vec_type`` to reach PetIGA. The solver strips PETSc-known flags before
processing.

PetIGA-Specific Options
-----------------------

The solver injects ``-iga_`` prefixes for PetIGA-specific options:

- ``-iga_mat_type`` — Controls sparse matrix format
- ``-iga_vec_type`` — Controls vector storage backend
- ``-iga_ksp_monitor`` — KSP solver convergence output
- ``-iga_ksp_converged_reason`` — Print KSP convergence reason

Help and Diagnostics
--------------------

::

    $ ./tdsez.x -help           # All available options
    $ ./tdsez.x -help <term>    # Search for specific option
    $ ./tdsez.x -help_group all # Grouped options list
