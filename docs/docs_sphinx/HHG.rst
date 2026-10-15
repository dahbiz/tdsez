Simulating HHG Correctly
========================

High-harmonic spectra are the Fourier transform of a time signal that spans
many orders of magnitude, so small numerical defects (reflections, a field with
a DC component, an unresolved cutoff, drifting free electrons) show up as
spurious structure long before they affect norms or energies. This page lists
what to set and what to check, with a validated 1D example.

Worked Example
--------------

Files: ``examples/hhg/hhg_1d.inp`` and ``examples/hhg/hhg_spectrum.py``.

1D soft-core atom :math:`V=-1/\sqrt{x^2+1}` (:math:`I_p = 0.670` Ha), 800 nm
(:math:`\omega = 0.057`), :math:`10^{14}` W/cm² (:math:`E_0 = 0.0534`),
16-cycle flat-top pulse (2-cycle sin² ramps):

::

    Dimension            = 1
    Domain               = -70.0, 70.0
    SplineDegree         = 7
    NSplines             = 560
    Charge               = -1.0
    Potential            = -1.0 / sqrt(x*x + 1.0)
    PotentialDerivativeX = x / (x*x + 1.0)^1.5
    TargetEigenvalue     = -0.8
    NBoundStates         = 4
    InitialState         = ground
    Variables            = E0=0.0534, w=0.057
    Variables            = Tr=220.4625, Tf=1543.2375, Tp=1763.7
    Polarization         = x
    LaserX               = -(E0/w)*( f'(t)*sin(w*t) + w*f(t)*cos(w*t) )   # E = -dA/dt, see the file
    Propagator           = pade22
    TimeStep             = 0.05
    FinalTime            = 1763.7
    EnableCAP            = 1
    CAPKmin              = 0.3

::

    tdsez.x -inp hhg_1d.inp                      # 15 s on one core
    python hhg_spectrum.py td/TimeEvolutionData_hhg_1d.inp.h5 --omega 0.057 --plot hhg.png

Result: clean odd harmonics (even/odd peak ratio :math:`10^{-4}`–:math:`10^{-3}`),
plateau up to H27–H29 and a sharp cutoff at
:math:`1.32\,I_p + 3.17\,U_p = 1.58` Ha (H27.7); the three equivalent spectra
(below) agree to a few per cent up to H39; final norm 0.992 (0.8% ionised, no
depletion).

The Spectrum
------------

The dipole table (``td/TimeEvolutionData_<input>.h5``, dataset ``dipoles``)
holds, per axis, the dipole :math:`D = q\langle x\rangle` and the dipole
acceleration :math:`A = q\,d\langle v\rangle/dt` (:doc:`Observables`); the
``currents`` table holds :math:`J = q\langle v\rangle`. The emitted spectrum is

.. math::

   S(\omega) = |\tilde A(\omega)|^2 = \omega^2|\tilde J(\omega)|^2
             = \omega^4|\tilde D(\omega)|^2 ,

identical for exact dynamics. ``hhg_spectrum.py`` computes all three with a
window over the whole record (Hann by default) and writes them side by side:

- Use :math:`|\tilde A|^2` as the result. The acceleration is dominated by the
  recolliding electron near the core and is insensitive to the slowly drifting
  free part.
- :math:`\omega^2|\tilde J|^2` should agree with it across the plateau; a
  disagreement points to an unconverged run.
- :math:`\omega^4|\tilde D|^2` amplifies any low-frequency error by
  :math:`\omega^4`. It agrees in a clean run (example above) but is the first to
  show reflections or free-electron drift; a dipole spectrum far above the
  acceleration spectrum is a diagnostic, not a result.

Checklist
---------

**Field**

- Give the field as :math:`E = -dA/dt` of a vector potential that vanishes at
  both ends (``LaserX`` is the electric field). Then
  :math:`\int E\,dt = 0`: no DC kick, no drift of the ionised electrons, no
  spurious low-order signal. A sin² envelope written directly on :math:`E` has
  a non-zero area for most phases.
- Short sin² pulses chirp the harmonics (the intensity changes every cycle) and
  the single-atom spectrum then shows broad, shifted and even-order-looking
  structure. That is physics, not noise; use a flat-top pulse, as in the
  example, when you need clean harmonic lines, e.g. for convergence tests.
- Set ``Polarization`` to the driven axes; the response on the other axes is
  recorded too (``ObserveAllAxes = 1``, default), which gives the
  cross-polarised emission in 2D/3D.

**Box and absorber**

- The free core (printed at start-up) must hold the electron excursion: half
  width at least about :math:`2\alpha + 10` a.u., with
  :math:`\alpha = E_0/\omega^2` (16.4 a.u. at 800 nm, :math:`10^{14}` W/cm²).
- ``EnableCAP = 1``. The CAP removes the electrons that do not return; without
  it they reflect from the wall and recollide at wrong times. ``CAPKmin`` sets
  the slowest momentum absorbed and the layer width (:math:`6.555/k_{\min}`):
  0.2–0.3 a.u. is a good start; check that a smaller value (wider layer, larger
  domain) does not change the spectrum.

**Resolution**

- Space: the highest momentum is about :math:`\sqrt{2(3.17U_p+I_p)}`
  (1.8 a.u. here), wavelength 3.5 a.u.; use at least 8–10 basis functions per
  wavelength (spacing 0.25 a.u. with degree 7 above). Near a Coulomb-like core
  refine the knots (:doc:`Knot-Sequences`). With ``KnotSequence = hydrogenic``
  the measured requirement is an outer width :math:`\le \pi/(2k_{max})` (set
  automatically when propagating) and an innermost spacing
  ``HydrogenicR1`` :math:`\le` 0.05 a.u.
- Time: resolve the cutoff, :math:`\omega_c\Delta t \ll 1`. Crank-Nicolson
  (``cn``) has a phase error :math:`(\omega\Delta t)^2/12` per step; the
  rational propagators (``pade22``, ``magnus4``) are fourth order and allow
  :math:`\Delta t` = 0.05 at 800 nm (:doc:`Time-Propagation`).
- Every step is recorded, so the Nyquist frequency is :math:`\pi/\Delta t`,
  far above any cutoff.

**Atom**

- Ground state from the eigensolver (``InitialState = ground``) and a
  ``TargetEigenvalue`` just below it.
- ``PotentialDerivativeX/Y/Z`` must be the exact gradient of ``Potential``:
  the acceleration uses it (``AccelerationMethod = operator`` for a constant
  mass). A wrong derivative gives a wrong :math:`\tilde A` while
  :math:`\tilde J` and :math:`\tilde D` stay right: the three-way comparison
  catches it.
- ``Charge = -1`` for electrons. The dipole and acceleration columns carry the
  charge (:math:`D = q\langle x\rangle`); spectra are unaffected.
- Watch depletion: the final norm (``energies`` column 6) and the ground-state
  population. Above about 50% ionisation the plateau is saturated and the
  spectrum depends strongly on the pulse shape. The softer potential
  :math:`-1/\sqrt{x^2+2}` (:math:`I_p = 0.5`) is ionised by 94% under the same
  pulse.

**Convergence**

Repeat with :math:`\Delta t/2`, a finer grid, a larger box and a smaller
``CAPKmin``; the harmonic peaks of :math:`|\tilde A|^2` should move by much
less than the feature you interpret.

**Higher dimensions**

The same rules apply in 2D/3D. The cost is dominated by the number of
non-zeros per row, :math:`(2p+1)^d`: prefer degree 3–4 in 3D, use
``ObserveAllAxes = 0`` when only the driven axis is needed, and keep the box
just large enough (the CAP does the rest).
