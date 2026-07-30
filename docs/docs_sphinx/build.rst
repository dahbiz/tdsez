build
=====

Build
=====

CMake configuration, dependencies, and build targets for TDSE-Z.

Dependencies
------------

Requires PETSc (with or without CUDA), SLEPc, PetIGA, MPI, OpenBLAS, muParser.

Build Steps
-----------

::

    # Export PETSc before configuring
    export PETSC_DIR=/path/to/petsc
    cd fused-version
    mkdir build && cd build
    cmake .. # auto-detects PETSc/CUDA/O3/march
    make -j $(nproc)
    # -> produces build/tdsez, build/tdmprocessor, build/tdmselect, build/drecprocessor

Auto-Detection
--------------

The build **auto-detects** PETSc's optimisation flags (-O3, -march, -mtune), CUDA arch,
and OpenMP settings. TDSEZ inherits the same ABI and CPU target automatically -- no
manual flag tuning needed.

Build Targets
-------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Target
     - Description
   * - **tdsez**
     - Main TDSE solver -- runs full simulation pipeline
   * - **tdmprocessor**
     - Transition dipole matrix post-processor
   * - **tdmselect**
     - Selective dipole element extraction
   * - **drecprocessor**
     - Dipole recorder / spectrum analysis
