Architecture
============

TDSE-Z is organized as a small pipeline. Each stage owns one concern and
passes PETSc objects to the next stage.

Execution pipeline
------------------

.. code-block:: text

   input file
       |
       v
   TDSEZParser ---- expressions, defaults, validation
       |
       v
   TDSEZCore ------ IGA grid, knots, H/M, eigenstates
       |
       v
   TDSEZAssembler - K, V, dipoles, velocity, CAP
       |
       +----------------------+
       |                      |
       v                      v
   TDSEZManager --------> TDSEZPropagator
   observables/HDF5       PETSc TS time stepping

The command-line executable in ``src/tdsez.cpp`` coordinates these objects. The
individual classes are intentionally usable from tests and small applications.

Public modules
--------------

``TDSEZParser``
   Reads ``key = value`` files, binds muParser expressions, normalizes aliases,
   and rejects invalid domains, masses, dimensions, and state selectors before
   PETSc objects are assembled. Public declarations are in
   ``include/tdsez_parser.hpp``.

``TDSEZCore``
   Owns the PetIGA context and static generalized eigenproblem
   :math:`H\psi=EM\psi`. It creates the knot vectors, assembles the Hamiltonian
   and mass matrices, solves with SLEPc, and writes eigenstate provenance.

``TDSEZAssembler``
   Builds the operator matrices used by the static and time-dependent stages.
   The implementation is split between ``src/assembler_class.cpp`` and the
   kernels in ``src/assembly/``.

``TDSEZManager``
   Owns runtime observables, HDF5 buffers, state projections, and output
   metadata. It keeps the output schema stable when individual observables are
   disabled.

``TDSEZPropagator``
   Configures PETSc TS with the Crank–Nicolson theta method, residual and
   Jacobian callbacks, and the selected preconditioner.

``TDSEZInfo``
   Provides consistent terminal headings, tables, units, and timing summaries.

Ownership rules
---------------

* PETSc and PetIGA objects are created collectively on ``PETSC_COMM_WORLD``.
* Classes destroy only objects they own; aliases exposed by getters remain
  owned by the creating class.
* HDF5 handles are opened lazily and closed by ``CloseHDF5`` after the final
  buffered rows are written.
* MPI rank 0 owns human-readable summaries and sequential postprocessor files;
  all ranks still participate in collective PETSc and HDF5 operations.

Generated API reference
-----------------------

Build the class and function reference with::

    cmake --build build --target docs

The generated entry point is ``docs/doxygen/html/index.html``. Doxygen reads
the public headers first, while source files provide implementation details and
call-site links.
