Initial State
=============

Selects the initial quantum state for time propagation. The state is assembled
from the bound eigenstates computed during the eigensolver phase, then optionally
normalized before propagation begins.

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
     - Normalize the assembled initial state so :math:`\langle \psi_0 | M | \psi_0 \rangle = 1`

Format Reference
----------------

Three formats are supported. The eigensolver must have computed at least as many
states as referenced by the selector.

**ground** — Use the 0th converged bound state (ground state)

::

    InitialState = ground

This is the lowest-energy eigenstate from the eigensolver. For most problems,
this corresponds to the bound ground state of the potential.

**state:N** — Use the Nth bound state (0-based index)

::

    InitialState = state:2    # 3rd eigenstate

Valid indices are 0 through NBoundStates-1. For example, ``state:2`` uses the
third eigenstate (index 2). If the eigensolver computed 5 states, valid indices
are 0–4.

**sup: a*N + b*M [+ c*P]** — Coherent superposition of up to 3 bound states

::

    InitialState = sup: 0.5*0 + 0.866*1
    InitialState = sup: 0.3*0 + 0.6*1 + 0.6*2

The superposition state is:

.. math::
   :nowrap:

   \begin{equation}
   |\psi(0)\rangle = a|\psi_N\rangle + b|\psi_M\rangle + c|\psi_P\rangle
   \end{equation}

Rules for superposition:

- Coefficients must be **real numbers**
- Number of terms: 2 or 3 (up to 3 states)
- Each term is ``coefficient*state_index``
- Coefficients are not automatically normalized — you must set them so that
  :math:`a^2 + b^2 + c^2 = 1` for a normalized superposition
- When ``NormalizeInitialState = 1``, the solver will normalize for you
  after assembly

Example Superpositions
~~~~~~~~~~~~~~~~~~~~~~

Two-state superposition (ground + first excited):

::

    InitialState = sup: 0.5*0 + 0.866*1

Three-state superposition:

::

    InitialState = sup: 0.3*0 + 0.6*1 + 0.6*2

Useful for studying:
- Quantum beating between states with different energies
- Rabi oscillations in a driven system
- Multi-state interference effects

Normalization
-------------

When ``NormalizeInitialState = 1`` (default), the assembled state is normalized
so that :math:`\langle \psi(0) | M | \psi(0) \rangle = 1` where M is the mass
matrix. This ensures the total probability is conserved.

For superpositions, the normalization factor is:

.. math::
   N = \sqrt{\sum_i a_i^2 + \sum_{i \neq j} a_i a_j \langle \psi_i | \psi_j \rangle}

Since eigenstates are orthonormal, the cross-terms vanish and
:math:`N = \sqrt{\sum_i a_i^2}`. Set your coefficients so this equals 1, or
enable normalization to let the solver handle it.

Common Use Cases
----------------

.. list-table::
   :widths: 30 25 45
   :header-rows: 1

   * - Use Case
     - Initial State
     - Notes
   * - Ground-state dynamics
     - ``ground``
     - Most common; starting from lowest energy
   * - Excited-state dynamics
     - ``state:N``
     - N must be < NBoundStates
   * - Quantum beating
     - ``sup: a*N + b*M``
     - Different energies cause oscillations
   * - Rabi oscillations
     - ``sup: a*N + b*M``
     - With laser driver, population transfers
   * - Multi-state interference
     - ``sup: a*N + b*M + c*P``
     - Three-state dynamics

Requirements
------------

1. The eigensolver must have completed successfully
2. ``NBoundStates`` must be large enough to access the desired state
3. For superpositions, coefficients must be real numbers
4. Eigenstates must be orthonormal (guaranteed by SLEPc/PETSc)
