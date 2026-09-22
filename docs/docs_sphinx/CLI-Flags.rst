CLI Flags
=========

Command-line overrides for all input parameters and PETSc options. CLI flags take
precedence over values in the input file.

Basic Usage
-----------

::

    $ ./tdsez -inp file.prm          # specify input file
    $ ./tdsez -ts_max_steps 0         # eigenvalues only, no propagation
    $ ./tdsez -save_dipole xyz        # save dipole axes at runtime
    $ ./tdsez -help                   # full PETSc options dump

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

    $ ./tdsez -help           # All available options
    $ ./tdsez -help <term>    # Search for specific option
    $ ./tdsez -help_group all # Grouped options list
