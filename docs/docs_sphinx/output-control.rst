output-control
==============

Output Control
==============

Stride configuration, observables selection, and compression.

Observable Selection
--------------------

.. list-table::
   :widths: 35 15 15 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - PhysicsOutput / OutputQuantities
     - string
     - ``all``
     - Comma-separated: ``dipole,population,energy,current,autocorrelation,wfs``.
       Any not listed are **OFF** -- their per-step computations are entirely
       skipped. ``all`` or empty = everything

Output Strides
--------------

.. list-table::
   :widths: 25 10 10 55
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - OutputStrideTS
     - int
     - 0
     - Output stride (steps) for time-series observables. 0 = disabled
   * - OutputStrideWFS
     - int
     - 100
     - Output stride (steps) for wavefunction snapshots. 0 = disabled
   * - OutputStrideAC
     - int
     - 0
     - Output stride (steps) for autocorrelation. 0 = disabled

Other Output Settings
---------------------

.. list-table::
   :widths: 25 10 10 55
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - HDF5Compress
     - bool
     - 0
     - Enable HDF5 chunk compression
   * - HDF5CompressLevel
     - int
     - 6
     - Compression level (0-9)
   * - Verbose
     - bool
     - 1
     - Verbose console output. Set to 0 for quiet runs
