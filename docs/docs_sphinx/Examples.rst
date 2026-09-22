Examples
========

Reference input files for well-known physics systems demonstrating the core
capabilities of TDSE-Z.

.. toctree::
   :maxdepth: 1
   :hidden:



1D Systems
----------

Harmonic Oscillator
~~~~~~~~~~~~~~~~~~~

File: ``inps/harmonic_oscillator_1d.prm``

A bound-state-only computation for the 1D harmonic oscillator:

.. math::
   :nowrap:

   \begin{equation}
   V(x) = \frac{1}{2} k x^2, \quad k = 1.0
   \end{equation}

**What it demonstrates:**
- Ground-state eigensolver (``EnablePropagation = 0``)
- Uniform B-spline grid with degree 3
- Extraction of eigenvalues and eigenstates
- Verification against analytical solution

**Exact solutions for reference:**

.. math::
   :nowrap:

   \begin{equation}
   E_n = \left(n + \frac{1}{2}\right) \omega = n + 0.5
   \end{equation}

.. math::
   :nowrap:

   \begin{equation}
   \psi_n(x) = \left(\frac{\alpha}{\pi}\right)^{1/4} \frac{1}{\sqrt{2^n n!}} H_n(\sqrt{\alpha}\,x) \, e^{-\alpha x^2/2}
   \end{equation}

where :math:`\alpha = \omega = 1.0`, :math:`H_n` is the Hermite polynomial of
degree :math:`n`.

**Key parameters:**

- ``Domain = [-10, 10]`` — sufficient to capture the exponential tails
- ``SplineDegree = 3``, ``Nelements = 200`` — fine grid for accurate eigenvalues
- ``NBoundStates = 5`` — compute first 5 eigenstates
- ``TargetEigenvalue = 0.5`` — start search near ground state

Run: ``./tdsez -inp inps/harmonic_oscillator_1d.prm``

Results in ``static/EigenData_1d.prm.h5``: eigenvalues, eigenstates, energy
spectrum.

---

1D Atomic Ionization
~~~~~~~~~~~~~~~~~~~~

File: ``inps/atom_laser_1d.prm``

A hydrogen-like atom driven by a Gaussian-envelope laser pulse at 800 nm:

.. math::
   :nowrap:

   \begin{equation}
   V(x) = -\frac{\alpha}{|x| + \beta}, \quad \alpha = 1.0,\; \beta = 0.5
   \end{equation}

**What it demonstrates:**
- Time-dependent propagation with field-ionization
- Complex absorbing potential (CAP) to absorb outgoing electron flux
- Population decay from ground state
- Dipole emission spectrum
- Autocorrelation function and survival probability

**Laser parameters:**

- Wavelength: :math:`\lambda = 800\,\text{nm}` → :math:`\omega = 0.056954` a.u.
- Intensity: :math:`I \approx 2.2 \times 10^{13}\,\text{W/cm}^2` (``Amplitude = 0.04``)
- Pulse duration (FWHM): 6 fs = 247 a.u.
- CEP (carrier-envelope phase): 0

**Expected behavior:**

- Ground-state population :math:`P_0(t)` decays as the atom ionizes
- Autocorrelation :math:`|A(t)|^2` drops from 1 to near 0 as electron escapes
- Dipole spectrum shows ionization continuum
- CAP at ``CAPKmin = 15.0`` absorbs electrons beyond :math:`|x| = 15`

**Key parameters:**

- ``EnableCAP = 1``, ``CAPKmin = 15.0``, ``Gamma = 10.0`` — absorbing boundary
- ``PhysicsOutput = all`` — all diagnostics enabled
- ``TimeStep = 0.02``, ``FinalTime = 300.0`` — propagate for ~7 fs

Run: ``./tdsez -inp inps/atom_laser_1d.prm``

Results in ``td/TimeEvolutionData_atom_laser_1d.h5``: dipole, population,
energy, current, autocorrelation tables.

---

1D Superposition — Quantum Beating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

File: ``inps/superposition_1d.prm``

A coherent superposition of the ground state (n=0) and first excited state
(n=1) in a 1D potential:

.. math::
   :nowrap:

   \begin{equation}
   V(x) = \frac{1}{2} k x^2, \quad k = 4.0
   \end{equation}

The superposition is:

.. math::
   :nowrap:

   \begin{equation}
   |\psi(0)\rangle = 0.6\,|\psi_0\rangle + 0.8\,|\psi_1\rangle
   \end{equation}

**What it demonstrates:**
- Superposition initial state (``InitialState = "sup: 0.6*0 + 0.8*1"``)
- Quantum beating — oscillating dipole at the transition frequency
- Zero laser field — free evolution of the superposition
- Population oscillation between states
- Wavefunction snapshots (WFS)

**Expected behavior:**

The dipole moment oscillates at:

.. math::
   :nowrap:

   \begin{equation}
   \omega_{01} = \frac{E_1 - E_0}{\hbar} = \omega = 2.0
   \end{equation}

The population of each state oscillates as:

.. math::
   :nowrap:

   \begin{equation}
   P_0(t) = |c_0|^2 = 0.36, \qquad P_1(t) = |c_1|^2 = 0.64
   \end{equation}

For a pure harmonic oscillator these are constant (degenerate spacing), but
any anharmonicity in the potential causes population exchange.

**Key parameters:**

- ``InitialState = "sup: 0.6*0 + 0.8*1"`` — superposition of n=0 and n=1
- ``NormalizeInitialState = 1`` — normalize after assembly
- ``Amplitude = 0.0`` — no driving field, free evolution
- ``OutputStrideWFS = 20`` — save wavefunction snapshots

Run: ``./tdsez -inp inps/superposition_1d.prm``

2D Systems
----------

2D Circularly Polarized Field — Berry Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

File: ``inps/harmonic_2d_circular.prm``

A 2D harmonic oscillator driven by circularly polarized light:

.. math::
   :nowrap:

   \begin{equation}
   V(x,y) = \frac{1}{2}(x^2 + y^2)
   \end{equation}

with laser field:

.. math::
   :nowrap:

   \begin{equation}
   E_x(t) = E_0 \sin(\omega t), \qquad E_y(t) = E_0 \sin(\omega t + \pi/2)
   \end{equation}

**What it demonstrates:**
- Circular polarization (``Polarization = xy``)
- Berry phase accumulation in a time-dependent field
- Angular momentum :math:`L_z` expectation value
- Degenerate state splitting (``EnableLzDiag = 1``)
- Full current decomposition in 2D

**Expected behavior:**

The Berry phase :math:`\Gamma(t)` accumulates linearly for a pure circularly
polarized field. The angular momentum :math:`\langle L_z \rangle` oscillates
as the electron is driven in a circular orbit. The current decomposition
shows dominant inter-band current from the circular driving.

**Key parameters:**

- ``Polarization = xy``, ``Ampx = 0.03``, ``Ampy = 0.03``
- ``CEPy = 1.570796`` (``\pi/2``) — 90° phase shift for circular polarization
- ``EnableLzDiag = 1`` — diagonalize :math:`L_z^2` within degenerate shells
- ``PhysicsOutput = all``

Run: ``./tdsez -inp inps/harmonic_2d_circular.prm``

