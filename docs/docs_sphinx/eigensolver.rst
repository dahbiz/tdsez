eigensolver
===========

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
     - Number of bound states to compute (stored as states 0..NBoundStates-1).
       Aliases: ``NumberOfBoundStates``
   * - TargetEigenvalue / TargetEnergy
     - real
     - 0.0
     - Target eigenvalue for SLEPc EPS. States near this energy are prioritized

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
     - Save format: ``complex`` (raw SLEPc eigenvectors) or ``real``
       (phase-rotated to real canonical form via TDSEZMakeStateReal2)
