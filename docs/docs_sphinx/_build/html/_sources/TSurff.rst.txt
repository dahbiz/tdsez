Photoelectron Spectra (t-SURFF)
===============================

The time-dependent surface flux method (t-SURFF; Tao & Scrinzi, New J. Phys.
14, 013021 (2012)) computes momentum-resolved photoelectron spectra from the
flux of the wave function through a surface that encloses the bound region,
so the box only has to hold the bound dynamics and the quiver motion: the
CAP behind the surface removes the outgoing electrons.

t-SURFF is a passive observer: it reads :math:`\psi` at the sampled time steps
and never changes the propagation. It is computed only when the input asks for
it (``TSurff = 1``) and writes its own file, ``td/surff_<input>.h5``.

Method
------

The surface is the box :math:`|x_d| = R` (``SurffRadius``) on every axis
(two points in 1D, the edges of a square in 2D, the faces of a cube in 3D).
Outside it the carrier must be free: constant mass :math:`m`, constant potential
:math:`V_0` and the field coupling of the run,

.. math::

   H = -\frac{1}{2m}\nabla^2 + V_0 + s\,\mathbf E(t)\cdot\mathbf r ,
   \qquad s = -q ,\ q = \text{Charge}.

Its length-gauge Volkov states are

.. math::

   \chi_{\mathbf k}(\mathbf r,t) = (2\pi)^{-d/2}\,
   e^{\,i(\mathbf k+\mathbf A(t))\cdot\mathbf r - i\Phi_{\mathbf k}(t)},
   \qquad \dot{\mathbf A} = -s\,\mathbf E ,\quad
   \dot\Phi_{\mathbf k} = \frac{|\mathbf k+\mathbf A|^2}{2m} + V_0 ,

and the amplitude of the state that has left the box is the time integral of
the flux through the surface,

.. math::

   b(\mathbf k) = \frac{i}{2}\int_0^T\!dt \oint dS\;
   \chi^*_{\mathbf k}\Big[-\frac{1}{m}\partial_n\psi
   - \frac{i}{m}\,\hat n\cdot(\mathbf k+\mathbf A)\,\psi\Big].

:math:`|b(\mathbf k)|^2 = dP/d^d k` is the distribution of the **canonical**
momentum :math:`\mathbf k`; the final kinetic momentum is
:math:`\mathbf k + \mathbf A(T)` (``A_final`` in the file, zero for a pulse of
zero area). The charge sign is the run's (``Charge = -1`` electron, ``+1``
hole), and the mass and potential are read from the ``Mass`` and
``Potential`` expressions at the surface.

Numerics:

- :math:`\psi` and :math:`m^{-1}\partial_n\psi` at the surface quadrature points
  (Gauss points of every knot span inside :math:`[-R, R]`, ``SurffQuadrature``
  per span) come from one sparse product with a precomputed trace matrix; at
  set-up the trace is checked against direct B-spline evaluation.
- The time integral uses the trapezoidal rule over the samples (every
  ``OutputStrideSurff``-th step and the final time): second order in the
  sampling interval.
- :math:`\mathbf A`, :math:`\int\mathbf A` and :math:`\int(A^2/2m+V_0)` are
  integrated with RK4 sub-steps of at most ``TimeStep/2``, so the Volkov phase
  does not depend on the stride.
- On a momentum grid the Volkov factors separate by axis: per sample and face
  the surface sum is one small GEMM plus axpy updates, instead of
  (#momenta × #surface points) complex exponentials. Each rank works on the
  surface points it owns; the amplitudes are reduced once at the end.

Before the run starts, the mass and potential are checked along outward rays
from every surface point to the end of each axis; a deviation above
``SurffMassTolerance`` (relative) or ``SurffPotentialTolerance`` (Ha) stops the
run. A Coulomb tail therefore has to be cut off smoothly before :math:`R`
(see the ATI example), or the tolerance raised knowingly.

Input Keys
----------

.. list-table::
   :widths: 28 10 14 48
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - TSurff
     - bool
     - 0
     - 1 = compute the t-SURFF spectrum (needs ``EnablePropagation = 1``).
   * - SurffRadius
     - real
     - --
     - Half-width :math:`R` of the box surface. Must lie inside every axis and in
       front of the CAP layer (the run stops otherwise).
   * - SurffKmax / SurffKmin
     - real[3]
     - -- / :math:`-k_{\max}`
     - Momentum grid limits per axis (comma-separated, one value for all axes).
   * - SurffNk
     - int[3]
     - --
     - Grid points per axis (1 = the single value ``SurffKmin``).
   * - SurffMomenta
     - file
     - (none)
     - Text file of momenta, one ``kx [ky kz]`` row per momentum (``#``
       comments); replaces the grid. Direct sums: use for short lists only.
   * - OutputStrideSurff
     - int
     - 1
     - Sample every n-th step (and always the final time). Error grows as
       (n·dt)²; keep 1 unless the flux sum dominates the run time.
   * - SurffQuadrature
     - int
     - 0
     - Gauss points per knot span on the surface (0 = ``NQuadratures``).
   * - SurffInterior
     - bool
     - 0
     - 1 = add :math:`\langle\chi_{\mathbf k}(T)|\theta_{\rm box}\psi(T)\rangle`,
       the part still inside the box at ``FinalTime``. Exact for a free carrier
       at any ``FinalTime``; for bound systems it adds the bound states'
       plane-wave content, so keep 0 there and propagate long enough instead.
   * - SurffMassTolerance
     - real
     - 1e-6
     - Allowed relative mass variation outside the surface.
   * - SurffPotentialTolerance
     - real
     - 1e-6
     - Allowed potential variation (Ha) outside the surface.

Output: ``td/surff_<input>.h5``
-------------------------------

Written by rank 0 at the end of the run.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Dataset
     - Shape
     - Content
   * - ``momenta``
     - (Nk, d)
     - canonical momenta :math:`\mathbf k`
   * - ``amplitude``
     - (Nk, 2)
     - Re, Im of :math:`b(\mathbf k)` (including the interior term with
       ``SurffInterior = 1``)
   * - ``pes``
     - (Nk,)
     - :math:`|b(\mathbf k)|^2 = dP/d^dk`
   * - ``amplitude_flux``
     - (Nk, 2)
     - flux part only (with ``SurffInterior = 1``)
   * - ``k_x``, ``k_y``, ``k_z``
     - (n_d,)
     - grid axes (grid mode)
   * - ``pes_grid``
     - (n_x[, n_y[, n_z]])
     - ``pes`` on the grid, C order (``k_x`` slowest)

Attributes: ``final_time``, ``radius``, ``exterior_mass``,
``exterior_potential`` (:math:`V_0`, includes the carrier sign), ``charge``,
``A_final``, ``stride``, ``n_surface_points``, ``n_samples``, ``interior``.

``examples/tsurff/plot_tsurff.py`` writes :math:`dP/dk` and
:math:`dP/dE` (1D: :math:`dP/dE = (m/|k|)(|b(k)|^2+|b(-k)|^2)`; 2D/3D:
angle-integrated from the grid) and plots them.

Example 1: free wave packet (analytic benchmark)
------------------------------------------------

``InitialState = wavepacket`` starts from the Gaussian

.. math::

   \psi_0 = \prod_d (2\pi w_d^2)^{-1/4}
   \exp\!\Big(-\frac{(x_d-c_d)^2}{4w_d^2} + i k_{0,d}x_d\Big)

(``WavepacketCenter``, ``WavepacketMomentum``, ``WavepacketWidth``). For a free
carrier the canonical momentum is conserved in any uniform field, so the exact
spectrum is

.. math::

   |b(\mathbf k)|^2 = \prod_d \Big(\frac{2w_d^2}{\pi}\Big)^{1/2}
   e^{-2w_d^2(k_d-k_{0,d})^2}

for any pulse, mass, constant potential and charge sign. ``NBoundStates = 0``
skips the eigensolver (allowed with an analytic initial state).

::

    Dimension          = 1
    Domain             = -40, 40
    SplineDegree       = 7
    NSplines           = 400
    Potential          = 0
    PotentialDerivativeX = 0
    NBoundStates       = 0
    InitialState       = wavepacket
    WavepacketMomentum = 2.0
    WavepacketWidth    = 1.0
    Polarization       = x
    LaserX             = t<20 ? 0.1*sin(pi*t/20)^2*sin(1.3*t) : 0
    Propagator         = pade22
    TimeStep           = 0.025
    FinalTime          = 40
    EnableCAP          = 1
    CAPKmin            = 0.5
    TSurff             = 1
    SurffInterior      = 1
    SurffRadius        = 10
    SurffKmin          = -1
    SurffKmax          = 5
    SurffNk            = 121

File: ``examples/tsurff/free_packet_1d.inp``. Measured accuracy
(max :math:`||b|^2-|b|^2_{\rm exact}|` / peak):

.. list-table::
   :widths: 55 20 25
   :header-rows: 1

   * - Case
     - error
     - note
   * - 1D, field-free, ``pade22``, dt = 0.025
     - 1.2e-6
     - with interior term
   * - 1D, sin² pulse, mass 0.5 / offset 0.3 / ``Charge = +1``
     - 0.9–3e-5
     - all agree
   * - 1D, ``cn``, dt = 0.05 → 0.00625
     - 4.9e-3 → 7.7e-5
     - second order (propagator)
   * - 1D, flux only (no interior), ``cn`` dt = 0.0125, T = 40 / 80
     - 3.4e-3 / 2.0e-3
     - slow electrons still inside
   * - 1D, ``OutputStrideSurff`` = 1, 2, 4, 8
     - 1.2, 4.2, 16, 65 ×1e-6
     - second order (trapezoid)
   * - 2D, (2,1), degree 5, ``pade22``
     - 8.9e-5
     - interior term
   * - 3D, (1.5,1,0.5), degree 4, ``cn``, 8 ranks
     - 1.6e-3
     - coarse grid, dt = 0.02
   * - 2D, 1 rank vs 4 ranks
     - 8e-13
     - MPI independence
   * - 2D, grid vs ``SurffMomenta`` list
     - 1e-15
     - same amplitudes

Cost: 0.11 ms per sample for 784 surface points × 3721 momenta in 2D against
24 ms per propagation step (the direct sum over the same momentum list takes
29.6 ms per sample, 260× more); 149 ms per sample for 86 400 points × 68 921
momenta in 3D against 6.2 s per step.

Example 2: above-threshold ionisation of a 1D atom
--------------------------------------------------

File: ``examples/tsurff/ati_1d.inp``. Soft-core atom
:math:`V=-1/\sqrt{x^2+1}` (:math:`I_p = 0.670` Ha) switched off smoothly before
the surface, 800 nm, :math:`10^{14}` W/cm², 8-cycle sin² pulse, then 400 a.u.
field-free so the slow electrons cross :math:`|x| = 35`:

::

    Variables            = Rc=20, d=2
    Potential            = -0.5*(1 - tanh((abs(x)-Rc)/d)) / sqrt(x*x + 1)
    PotentialDerivativeX = 0.5*(1 - tanh((abs(x)-Rc)/d)) * x/(x*x + 1)^1.5 + 0.5*(1 - tanh((abs(x)-Rc)/d)^2)/d * sign(x) / sqrt(x*x + 1)
    ...
    TSurff      = 1
    SurffRadius = 35
    SurffKmax   = 2.5
    SurffNk     = 501

::

    tdsez.x -inp ati_1d.inp
    python examples/tsurff/plot_tsurff.py td/surff_ati_1d.inp.h5 --energy --omega 0.057

Result (7 s on one core): ATI peaks spaced by :math:`\omega` (0.88–1.09 ω
between 0.19 and 0.82 Ha) at :math:`n\omega - I_p - U_p` modulo :math:`\omega`;
total yield :math:`\int|b|^2dk = 9.46\times10^{-4}` against
:math:`9.31\times10^{-4}` absorbed by the CAP (the difference is still between
the surface and the CAP).

Choosing the parameters
-----------------------

- **Surface radius**: outside the bound states and the potential (cut it off
  smoothly), and preferably outside the quiver radius
  :math:`\alpha = E_0/\omega^2`. The CAP starts behind it.
- **Final time**: long enough after the pulse for the slowest electron of
  interest to reach the surface, :math:`T - T_{\rm pulse} \gtrsim (R-\text{core})/k_{\min}`;
  check by comparing two final times. For free-particle tests use
  ``SurffInterior = 1``.
- **Momentum grid**: :math:`k_{\max}\gtrsim\sqrt{2m\cdot 10U_p}` for the full
  rescattering plateau; resolve the ATI spacing
  :math:`\Delta k \approx m\omega/k`.
- **Time step**: t-SURFF inherits the propagator error; the rational
  propagators (``pade22``, ``magnus4``) are much more accurate per step.
- **Charge**: as for the run (``-1`` electron). Holes (``+1``) are free carriers
  with the opposite coupling sign; t-SURFF follows automatically.
