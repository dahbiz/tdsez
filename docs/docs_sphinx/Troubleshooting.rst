Troubleshooting
===============

Error messages, warnings and performance problems, with their causes and
remedies. Messages are quoted as printed; ``...`` stands for run-specific
values.

Input Errors
------------

Unknown, removed or repeated keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TDSEZParser FATAL: unknown input key '...'`` stops the run, because a
misspelt key (for example of the potential or the mass) would otherwise be
ignored silently. Correct the key (:doc:`Input-File`). ``StrictInput = 0``
turns the error into a warning; the deck is read line by line, so the
``StrictInput`` line must precede the unknown key.

Keys of removed features are reported with the reason and the replacement
(``input key '...' has been removed: ...``); a removed key whose value had no
effect (for example ``Planck = 1``) only produces a warning. A key given twice
is fatal (``input key(s) given more than once``) unless ``StrictInput = 0``, in
which case the last value is used. ``Variables`` is the only key that may be
repeated.

Validation errors
~~~~~~~~~~~~~~~~~

After reading the deck, the parameters are checked as a whole; a failed check
stops the run with ``ValidateOrThrow: ...`` and names the offending key and
value, for example:

- ``Dimension must be 1, 2 or 3``
- ``SplineDegree must be in [1,14]``
- ``Mass must be > 0 and finite in the open domain``
- ``InitialState='state:N' but only NBoundStates=... bound states are
  computed`` -- increase ``NBoundStates`` to at least :math:`N+1`, or choose a
  lower index.
- ``NBoundStates = 0 (no eigenproblem) needs an analytic InitialState
  (wavepacket, analytic:quesne01, analytic:massstep01)``
- t-SURFF: ``TSurff = 1 needs EnablePropagation = 1``, ``... needs SurffRadius
  > 0``, ``the t-SURFF surface must lie in front of the CAP layer`` (the
  message gives the largest admissible radius on that axis); see
  :doc:`TSurff`.

A number must be the whole value: ``NSplines = 100, 16, 16`` for a key with one
value, ``NBoundStates = 4.5`` or ``TimeStep = 0.05a`` stop the run with
``'...' is not a single number`` or ``'...' is not an integer`` (an integral
real such as ``4.0`` is accepted where an integer is expected). Earlier versions
kept the leading number and dropped the rest without a warning.

The ``tests/inputs/bad_*.inp`` decks reproduce several of these errors
(:doc:`Examples`).

Initial state index beyond the converged eigenstates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Initial state index N >= number of converged eigenstates M`` appears when the
index passed validation but fewer eigenpairs converged than requested. Check
the eigensolver report (next section) and ``TargetEigenvalue``.

Eigensolver
-----------

The bound states are computed with Krylov-Schur [Stewart2002]_ on the
shift-invert operator (:doc:`Eigensolver`). The inner solves of
:math:`H - \sigma M` follow a plan of up to three solvers (direct, iterative,
stronger iterative). Each attempt is accepted only if the inner solve succeeds,
at least ``NBoundStates`` pairs converge and the maximum backward error is at
most :math:`10^{-8}`. The run reports a failed attempt and moves on:

- ``WARNING: inner solve failed (..., PETSc error ...); trying the next inner
  solver``
- ``WARNING: n of N pairs converged (...); retrying with a larger subspace``;
  the attempt is repeated once with a doubled subspace.
- ``WARNING: backward error ... > 1e-08 with ...; trying the next inner
  solver``

The ``RESULTS`` block lists ``Inner solver used``, ``Converged pairs`` and, for
an accepted attempt, ``Max backward error``. If no attempt passes every check,
the converged pairs of the last attempt are used with ``WARNING: no inner
solver met every check``; if no pair converged at all, the run stops. Remedies:

- Set ``TargetEigenvalue`` just below the lowest wanted state.
- ``-tdsez_eps_inner direct`` forces an exact inner solve first
  (:doc:`CLI-Flags`).
- With an analytic initial state (``wavepacket``, ``analytic:...``), set
  ``NBoundStates = 0`` to skip the eigensolver.
- Shift-invert near the bottom of a gapless spectrum (free or nearly free
  carriers) converges slowly, and the 3D direct factorisation is memory-bound.

Non-finite Hamiltonian
----------------------

``TDSEZ FATAL: the Hamiltonian has infinite or NaN entries: the potential is
singular at a quadrature point.`` A potential such as :math:`-1/r` is infinite
at its centre; ``QuadratureRule = lobatto`` places quadrature points on the
knots, so a Coulomb centre on a knot gives non-finite matrix entries. Use
``QuadratureRule = legendre`` (default) or ``reduced``, or move the centre off
the knots.

Energies Below the Exact Value
------------------------------

With the default natural (Neumann) boundaries the wave function is not forced
to vanish at the box edge. When the box truncates the tail of a state, the
computed energy can lie *below* the exact value, because the discarded tail
carries positive local energy. This is not a variational violation of the
solver. Enlarge the box until the energy is stable, or compare with
``BoundaryType = Dirichlet``; measured values for hydrogen are given in
:doc:`Knot-Sequences` (Coulomb section).

Quadrature Order
----------------

:math:`n` Gauss points per element integrate polynomials of degree
:math:`2n-1` exactly; the mass and kinetic matrices of degree-:math:`p`
B-splines therefore need ``NQuadratures`` :math:`\ge p+1`. The code does not
enforce this; the default of 8 covers :math:`p \le 7`. Singular or rapidly
varying potentials need more points (:doc:`Grid-Domain`,
:doc:`Knot-Sequences`).

Strong-Field Warnings at Start-up
---------------------------------

For a propagation with a non-zero field, the code estimates the peak field
:math:`E_0`, the frequency :math:`\omega` (from ``Omegax/y/z`` or from the zero
crossings of the field), :math:`U_p = E_0^2/4\omega^2`,
:math:`I_p = -E_{\rm target}` (``TargetEigenvalue``; 0.5 if the target is not
negative), :math:`k_{\max} = \sqrt{2(3.17\,U_p + I_p)}` and
:math:`\alpha = E_0/\omega^2` (:doc:`HHG`).

**Field resolution.** The FIELD RESOLUTION part of the log gives the widest
element allowed (:math:`\pi/(2k_{\max})` on the polarised axes,
``TransverseElementFactor`` times that on the others) and, per axis, either
``widest element ... : resolved`` or ``ⵣ WARNING: x axis (polarised): widest
element ... the returning electron and the harmonic cutoff are under-resolved
there``. The grid is not changed. Increase ``NSplines``, reduce the grading, or
use ``KnotSequence = hydrogenic``, whose outer element width is capped by
``HydrogenicHMax``: a positive value sets the cap, 0 (default) takes the widths
above from the field (printed as ``HydrogenicHMax from the field``), a
negative value removes the cap.

**CAP report.** With ``EnableCAP = 1`` the block ``COMPLEX ABSORBING POTENTIAL
(CAP)`` gives the layer width per wall, the slowest momentum absorbed with
little reflection, and the free core of every axis. On a driven axis whose
free-core half width is below :math:`2\alpha + 10` it prints ``WARNING: x free
core half width ... < 2 alpha + 10``: returning electrons enter the absorber.
Enlarge the domain or reduce ``CAPWidth`` (increase ``CAPKmin``).

t-SURFF Errors
--------------

- ``t-SURFF needs a free carrier outside the surface: the mass deviates by ...
  and the potential by ...``: the mass or potential varies beyond
  ``SurffRadius``. Move the surface out, switch the potential off smoothly
  before it, or raise ``SurffMassTolerance`` / ``SurffPotentialTolerance``
  deliberately.
- ``t-SURFF: SurffMomenta file '...' is missing, empty or has a row with fewer
  than d values``.

See :doc:`TSurff`.

Slow Runs under MPI
-------------------

OpenMPI binds each rank to one core by default, so the threads of the BLAS
library compete for that core; the slowdown is typically 8-10×. TDSE-Z uses
one BLAS thread per rank unless ``OMP_NUM_THREADS`` (or the variable of the
BLAS library) is set; if you set it, keep ranks × threads within the cores of
the node. Launch with ``mpiexec -n N --bind-to none`` (:doc:`MPI-Notes`).

A Job Stops on One Rank, or Hangs
---------------------------------

An error on a single rank (for example out of memory, ``std::bad_alloc``)
ends the whole job with ``MPI_Abort`` and ``[r] TDSEZ FATAL: ...`` from that
rank, instead of leaving the other ranks waiting until the time limit. Look for
the first ``FATAL`` or ``PETSC ERROR`` line in the error output of all ranks.
For a long run under a time limit use restart files
(:ref:`restart-files`).

Performance in 3D
-----------------

- **Assembly**: a ``Mass`` expression without :math:`x, y, z` is detected as
  constant automatically (as with ``MassIsConstant = 1``).
- **Memory**: every stored operator has :math:`(2p+1)^3` non-zeros per row
  (729 for degree 4), about 24 bytes each. With 8 ranks the 110 592-DOF
  degree-4 benchmark peaks at 16–20 GB (measured); beyond the physical memory
  the run swaps and matrix-vector products slow down by an order of magnitude.
  Prefer degree 3 in 3D, set ``ObserveAllAxes = 0`` when only the driven axis
  is needed, and check ``Peak memory per rank`` in the run summary.
- **Observables**: in 2D/3D without Dirichlet walls, the mass matrix, the
  dipoles, the velocities and the kinetic energy are applied in factorised
  (Kronecker) form, each checked against the assembled matrix on the first
  state and used only if they agree to :math:`10^{-10}` relative
  (``Factorised operators = ...`` in the PROPAGATION stage;
  ``-tdsez_tensor_ops 0`` keeps the matrix products). :math:`\langle V\rangle`,
  the forces, the CAP and, for a varying mass, :math:`\langle 1/m\rangle`,
  :math:`\langle K\rangle` and the velocities are evaluated from :math:`\psi`
  at the quadrature points, also after a check against the matrices
  (``-tdsez_quad_obs 0`` restores the matrix products).
