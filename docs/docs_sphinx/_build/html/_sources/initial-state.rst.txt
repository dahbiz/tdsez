initial-state
=============

Initial State
=============

Ground state, excited state, and superposition selection.

InitialState Selector
---------------------

.. list-table::
   :widths: 40 15 10 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - InitialState / StartState
     - string
     - ``ground``
     - Initial state selector. See formats below
   * - NormalizeInitialState / NormalizePsi0
     - bool
     - 1
     - Normalize the assembled initial state so <psi_0|M|psi_0> = 1

Formats
-------

- **ground** -- Use the 0th converged bound state (ground state)
- **state:N** -- Use the Nth bound state (0-based). E.g., ``state:2`` = 3rd
  eigenstate
- **sup: a*N + b*M [+ c*P]** -- Coherent superposition. Real coefficients (up
  to 3). E.g., ``sup: 0.5*0 + 0.866*1``
