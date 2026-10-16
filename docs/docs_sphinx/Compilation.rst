Compilation
===========

Dependencies, CMake configuration, build targets and first runs of TDSE-Z.

Dependencies
------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Dependency
     - Requirement
   * - C++17 compiler, CMake ≥ 3.15, MPI
     - MPI is located with ``find_package(MPI)``.
   * - PETSc [PETSc]_
     - Complex scalars (``--with-scalar-type=complex``) and HDF5; MUMPS for
       the direct inner solves of the eigensolver. ``PETSC_DIR`` must be
       exported; ``PETSC_ARCH`` is optional.
   * - SLEPc [Hernandez2005]_
     - Installed in the PETSc prefix (``$PETSC_DIR/lib/libslepc``).
   * - PetIGA [Dalcin2016]_
     - Installed in the PETSc prefix (``$PETSC_DIR/include/petiga.h``,
       ``$PETSC_DIR/lib/libpetiga``).
   * - HDF5
     - ``hdf5`` library in ``$PETSC_DIR/lib`` (the high-level ``hdf5_hl`` library is not needed).
   * - muParser
     - Found through its CMake package, otherwise linked as ``muparser``
       from ``$PETSC_DIR/lib``.
   * - BLAS/LAPACK
     - Taken from PETSc's ``BLASLAPACK_LIB``.
   * - Python 3
     - Required at configure time; ``pytest``, ``h5py``, ``scipy`` and
       ``zkit`` enable the ``test`` target.

A PETSc configuration that provides all of the above:

.. code-block:: bash

   ./configure --prefix=$HOME/petsc --with-scalar-type=complex --with-64-bit-indices \
               --with-cxx-dialect=c++17 --with-debugging=0 \
               --download-slepc --download-hdf5 --download-mumps --download-scalapack \
               --download-metis --download-parmetis --download-muparser --download-openblas

PetIGA is then built against this PETSc and installed into the same prefix.

Build
-----

From the repository root:

.. code-block:: bash

   export PETSC_DIR=/path/to/petsc
   cmake -S . -B build
   cmake --build build -j

The default build type is ``Release``. CMake reads PETSc's compiler flags
(``-O3``, ``-march``, ``-mtune``), BLAS/LAPACK link line, OpenMP and CUDA
settings from ``$PETSC_DIR/lib/pkgconfig/PETSc.pc``,
``$PETSC_DIR/include/petscconf.h`` and ``petscvariables``, so that TDSE-Z is
compiled with the same ABI and CPU target as PETSc. ``$PETSC_DIR/lib`` is
embedded as run path.

.. list-table:: CMake options
   :header-rows: 1
   :widths: 30 12 58

   * - Option
     - Default
     - Effect
   * - ``ENABLE_OPENMP``
     - ``ON``
     - Link OpenMP when available.
   * - ``ENABLE_LTO``
     - ``ON``
     - Link-time optimisation (disabled if unsupported by the compiler).
   * - ``ENABLE_NATIVE``
     - ``ON``
     - Compile for the native CPU (``-march=native``, or ``-mcpu=native`` on
       arm64) when PETSc's flags specify no CPU target.
   * - ``ENABLE_AUTO_CUDA``
     - ``ON``
     - Enable CUDA when PETSc was built with it.
   * - ``PETSC_INHERIT_FLAGS``
     - ``ON``
     - Add PETSc's C++ optimisation flags in Release builds (fast-math
       flags excluded).

Build targets
-------------

All executables are written to the build directory.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Target
     - Description
   * - ``tdsez``
     - Solver executable ``tdsez.x``.
   * - ``tdmprocessor``
     - Transition dipole matrix between all saved eigenstates.
   * - ``tdmselect``
     - Dipole elements from one bra state, written as CSV.
   * - ``drecprocessor``
     - Recombination dipole :math:`d_{\mathrm{rec}}(E)` from neighbouring box
       eigenstates.
   * - ``unit-test``
     - Builds and runs the serial parser-validation test.
   * - ``test``
     - Runs the pytest suite in ``tests/`` (requires the Python packages
       above).
   * - ``docs``
     - Doxygen HTML in ``docs/doxygen/html/`` (requires Doxygen).
   * - ``docs-sphinx``
     - This manual in ``docs/docs_sphinx/_build/html/`` (requires Sphinx).

The post-processors are described in :doc:`Post-Processing`.

Running
-------

The input file is passed with ``-inp`` (default ``tdse.prm``):

.. code-block:: bash

   ./build/tdsez.x -inp tests/inputs/ho1d.inp                      # serial
   mpirun --bind-to none -np 4 ./build/tdsez.x -inp tests/inputs/ho3d.inp

``--bind-to none`` avoids the core pinning of OpenMPI 5.x (:doc:`MPI-Notes`).
To compute only the bound states, set ``EnablePropagation = 0`` in the input
file. PETSc and SLEPc options (``-eps_*``, ``-ksp_*``, ``-log_view``) and the
TDSE-Z flags of :doc:`CLI-Flags` are appended to the command line. Output
files are listed in :doc:`Overview`.
