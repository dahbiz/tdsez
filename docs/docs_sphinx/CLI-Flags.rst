CLI Flags
=========

Command-line overrides for all input parameters and PETSc options. CLI flags take
precedence over values in the input file.

Basic Usage
-----------

::

    $ ./tdsez -inp file.prm          # specify input file
    $ ./tdsez -ts_max_steps 0         # eigenvalues only, no propagation
    $ ./tdsez -vec_type cuda          # GPU vectors
    $ ./tdsez -mat_type aijcusparse   # GPU matrices
    $ ./tdsez -use_gpu_aware_mpi 0    # non-GPU-aware MPI workaround
    $ ./tdsez -save_dipole xyz        # save dipole axes at runtime
    $ ./tdsez -enable_gpu 1           # enable GPU acceleration
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
   * - ``-vec_type <type>``
     - string
     - PETSc vector type. Accepts ``cuda`` for GPU vectors or ``mpi`` for CPU.
   * - ``-mat_type <type>``
     - string
     - PETSc matrix type. Accepts ``aijcusparse`` for GPU sparse matrices or ``aij`` for CPU.
   * - ``-use_gpu_aware_mpi <0/1>``
     - bool
     - Whether MPI supports GPU-aware communication. Set to 0 on systems where MPI is not GPU-aware (most systems).
   * - ``-save_dipole <axes>``
     - string
     - Which dipole axes to save at runtime. Accepts ``x``, ``xy``, ``xyz``, ``all``, or ``none``.
   * - ``-enable_gpu <0/1>``
     - bool
     - Enable GPU acceleration for propagation.

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

GPU Configuration
-----------------

When running on GPU:

1. Set ``-vec_type cuda`` and ``-mat_type aijcusparse``
2. Set ``-use_gpu_aware_mpi 0`` (unless your MPI library supports GPU-aware)
3. Set ``-enable_gpu 1`` to activate GPU propagation
4. Verify CUDA is accessible via ``deviceQuery`` or ``nvidia-smi``

For non-GPU-aware MPI, data must be transferred between GPU and CPU for
communication. This adds overhead but works on most systems.

Help and Diagnostics
--------------------

::

    $ ./tdsez -help           # All available options
    $ ./tdsez -help <term>    # Search for specific option
    $ ./tdsez -help_group all # Grouped options list
