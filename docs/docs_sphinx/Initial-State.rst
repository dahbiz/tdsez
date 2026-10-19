Initial State
=============

``InitialState`` selects the state :math:`\psi(0)` at the start of the time
propagation. It is either built from the bound states of the eigensolver
(:doc:`Eigensolver`) or projected from an analytic function onto the basis.
All norms use the overlap matrix :math:`M` of the basis,
:math:`\|\psi\|_M^2 = \mathbf c^\dagger M\,\mathbf c` for the coefficient
vector :math:`\mathbf c` (:doc:`Physics-Conventions`).

Keys
----

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
     - Initial-state selector; the accepted forms are listed below.
   * - NormalizeInitialState / NormalizePsi0
     - bool
     - ``1``
     - Normalise a superposition (``sup:``) to :math:`\|\psi(0)\|_M = 1`.
   * - WavepacketCenter
     - real list
     - ``0``
     - Centre :math:`c_d` of the wave packet per axis.
   * - WavepacketMomentum
     - real list
     - ``0``
     - Mean momentum :math:`k_{0,d}` per axis.
   * - WavepacketWidth
     - real list
     - ``1``
     - Width :math:`w_d > 0` per axis.

The wave-packet keys take comma-separated per-axis values; a single value is
used for every axis.

Selector Forms
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - ``InitialState``
     - Initial state
   * - ``ground``
     - Bound state 0 (equivalent to ``state:0``).
   * - ``state:N``
     - Bound state :math:`N` (0-based).
   * - ``sup: c1*N1 + c2*N2 [+ c3*N3]``
     - Superposition of one to three bound states,
       :eq:`eq-initial-state-superposition`.
   * - ``wavepacket``
     - Gaussian wave packet :eq:`eq-initial-state-wavepacket`, projected onto
       the basis.
   * - ``analytic:quesne01``
     - 1D benchmark: exact Quesne-oscillator state, projected onto the basis.
   * - ``analytic:massstep01``
     - 1D benchmark: exact mass-step state, projected onto the basis.

The selector is case-sensitive. Any other value stops the run with a message
listing the accepted forms.

Bound states
~~~~~~~~~~~~

Bound state :math:`N` is the :math:`N`-th state of the eigensolver after its
canonical ordering: states are ordered by their distance to
``TargetEigenvalue`` (ties by energy). ``ground`` is therefore the lowest
state only when ``TargetEigenvalue`` lies at or below the ground-state energy
(:doc:`Eigensolver`). An index must satisfy :math:`0 \le N <`
``NBoundStates``; a larger index is rejected before the solve. The state is
used as returned by the eigensolver, with :math:`\|\psi_N\|_M = 1` and the
fixed phase described in :doc:`Eigensolver`.

::

    InitialState = ground
    InitialState = state:2      # third state of the canonical ordering

Superpositions
~~~~~~~~~~~~~~

.. math::
   :label: eq-initial-state-superposition

   \psi(0) = \frac{1}{\mathcal N} \sum_{k=1}^{n} c_k\,\psi_{N_k},
   \qquad n \le 3 ,

with :math:`\mathcal N = \|\sum_k c_k\psi_{N_k}\|_M` when
``NormalizeInitialState = 1`` and :math:`\mathcal N = 1` otherwise. For
distinct indices the bound states are :math:`M`-orthonormal and
:math:`\mathcal N = (\sum_k |c_k|^2)^{1/2}`. The norm before and after
normalisation is printed.

Syntax rules:

- Each term is ``coefficient*index``; a term without ``*`` has coefficient 1.
- Terms are separated by ``+`` or ``-``; a ``-`` negates the following term.
- Coefficients are real (``0.6``, ``-1e-3``) or complex: ``a+bi``, ``a-bi``,
  ``bi`` (for example ``0.5i``), ``i`` (``j`` is accepted for ``i``). A complex coefficient
  with an internal sign is enclosed in parentheses, for example
  ``(0.7+0.7i)*0``.
- Every index must be smaller than ``NBoundStates``.

Because every bound state carries a fixed phase (:doc:`Eigensolver`), the
relative phase of the terms is reproducible across runs and MPI layouts.

::

    InitialState = sup: 0.5*0 + 0.866*1
    InitialState = sup: 0.6*0 - 0.8*2
    InitialState = sup: (0.7+0.7i)*0 + 1.0*1

Gaussian wave packet
~~~~~~~~~~~~~~~~~~~~

``InitialState = wavepacket`` starts from the normalised Gaussian

.. math::
   :label: eq-initial-state-wavepacket

   \psi_0(\mathbf r) = \prod_{d=1}^{D} (2\pi w_d^2)^{-1/4}
   \exp\!\left(-\frac{(x_d-c_d)^2}{4w_d^2} + i\,k_{0,d}\,x_d\right),

for which :math:`|\psi_0|^2` has standard deviation :math:`w_d` along axis
:math:`d` and the momentum density is
:math:`\prod_d (2w_d^2/\pi)^{1/2}\, e^{-2w_d^2(k_d-k_{0,d})^2}`.

Analytic benchmark states
~~~~~~~~~~~~~~~~~~~~~~~~~

The two ``analytic:`` selectors are 1D test states with parameters fixed in
the code; each is the equal-weight superposition
:math:`(\phi_0 + \phi_1)/\sqrt 2` of the two lowest exact eigenstates of a
benchmark problem, and is meaningful only with the matching input:

- ``analytic:quesne01``: Quesne oscillator with
  :math:`m(x) = (1 + x/2)^{-1}` and :math:`V(x) = x^2/(4(x+2))` on
  :math:`x > -2` (see the example in :doc:`Potential-Mass`).
- ``analytic:massstep01``: :math:`V = 0` on :math:`[-1, 1]` with Dirichlet
  walls and a mass step :math:`m = 1` for :math:`x < 0`, :math:`m = 4` for
  :math:`x > 0`.

Projection and normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The wave packet and the analytic states are :math:`L^2`-projected onto the
basis,

.. math::
   :label: eq-initial-state-projection

   M\,\mathbf c = \mathbf b, \qquad
   b_i = \int B_i(\mathbf r)\,\psi_0(\mathbf r)\,d\mathbf r ,

solved with conjugate gradients (Jacobi preconditioner, relative tolerance
:math:`10^{-13}`), and then always normalised to :math:`\|\psi(0)\|_M = 1`,
independently of ``NormalizeInitialState``. The packet must lie well inside
the box; the part outside the box is lost in the projection.

Running without the eigensolver
-------------------------------

The projected states do not need the eigenstates, and ``NBoundStates = 0``
skips the eigensolver; it is accepted only with ``wavepacket`` or an
``analytic:`` selector. This matters for free or weakly bound systems, where
a shift-invert solve at the bottom of a continuum is slow, particularly in
3D. Without bound states no ``<name>_eigen.h5`` file is written and the
populations and the bound-state decomposition of the current are empty.
The t-SURFF benchmarks use this mode (:doc:`TSurff`).

Typical uses
------------

.. list-table::
   :widths: 30 25 45
   :header-rows: 1

   * - Use case
     - ``InitialState``
     - Notes
   * - Ground-state dynamics (HHG, ionisation)
     - ``ground``
     - ``TargetEigenvalue`` at or below the ground-state energy
   * - Excited-state dynamics
     - ``state:N``
     - :math:`N <` ``NBoundStates``
   * - Quantum beats, coherent control
     - ``sup: c1*N1 + c2*N2``
     - Relative phase set by the coefficients
   * - Free propagation, scattering, t-SURFF tests
     - ``wavepacket``
     - ``NBoundStates = 0`` skips the eigensolver
