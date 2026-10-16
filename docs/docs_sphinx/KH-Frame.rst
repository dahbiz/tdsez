Kramers-Henneberger Frame
=========================

``Mode = kh`` solves the static Kramers-Henneberger (KH) problem: the
eigenstates of the potential averaged over the quiver motion of the carrier in
the laser field [Henneberger1968]_ [Gavrila2002]_. No time propagation is
run. ``Mode = td`` (default) is the usual laboratory-frame calculation.

Theory
------

A free carrier of charge :math:`q` and mass :math:`m` in the uniform field
:math:`\mathbf E(t)` oscillates along the quiver path

.. math::
   :label: eq-kh-quiver

   m\,\ddot{\boldsymbol\alpha}(t) = q\,\mathbf E(t) .

In the frame that follows this motion, :math:`\mathbf r \to \mathbf r +
\boldsymbol\alpha(t)`, the length-gauge Hamiltonian of
:eq:`eq-laser-field-coupling` becomes
:math:`p^2/2m + s\,V(\mathbf r + \boldsymbol\alpha(t))`: the field acts only
through the displaced potential. For high frequencies its cycle average,

.. math::
   :label: eq-kh-potential

   V_{\text{KH}}(\mathbf r) = \frac{1}{P}\int_{t_0}^{t_0 + P}
   V\big(\mathbf r + \boldsymbol\alpha(t)\big)\,dt ,
   \qquad P = 2\pi/\omega ,

is the leading term of the high-frequency Floquet theory; its eigenstates are
the dressed states of atomic stabilisation, with the characteristic
dichotomy of the dressed density for large quiver amplitudes
:math:`\alpha_0 = |q|E_0/m\omega^2`.

What TDSE-Z does
----------------

With ``Mode = kh`` the potential in every operator is replaced by
:math:`V_{\text{KH}}`; everything else is a normal stationary run
(``static/<name>_eigen.h5``, ``SaveDipoleAxes``, the post-processing tools of
:doc:`Post-Processing`, zkit). The quiver is computed from the laser
expressions ``LaserX/Y/Z`` (or ``Laser``) of the polarised axes:

1. **Window.** One optical cycle centred at the peak of
   :math:`|\mathbf E(t)|` in :math:`[0, \texttt{FinalTime}]`. The period is
   :math:`2\pi/\omega` with :math:`\omega` the largest ``Omegax/y/z`` of the
   polarised axes, otherwise it is measured from the first and the last zero
   crossing of the field (located to rounding by bisection).
   ``KHWindow = t0, t1`` sets the window directly.
2. **Quiver.** Integrating :eq:`eq-kh-quiver` twice over the window and
   removing the mean velocity (the drift) and the mean position leaves the
   pure quiver; for a monochromatic field
   :math:`\boldsymbol\alpha(t) = -(q\mathbf E_0/m\omega^2)\sin(\omega t +
   \varphi)`. With Cauchy's formula for the repeated integral,
   :math:`\boldsymbol\alpha` at every node needs only single integrals of
   :math:`\mathbf E`, evaluated by composite 8-point Gauss-Legendre
   quadrature (accurate to rounding).
3. **Average.** :eq:`eq-kh-potential` is evaluated at every quadrature point
   of the assembly by composite Gauss-Legendre quadrature over the window,
   which converges exponentially for a smooth quiver, periodic (CW) or not (a
   pulse envelope across the cycle). The number of nodes doubles from 16
   until :math:`V_{\text{KH}}` changes by less than :math:`10^{-12}` on a
   test lattice; ``KHSamples`` fixes it.

The log reports the window, the period, the quiver amplitude and rms per axis
and the number of nodes; the attributes ``Mode``, ``KH_window_start``,
``KH_window_end``, ``KH_omega``, ``KH_alpha_x/y/z`` and ``KH_samples`` of
``run_metadata`` in ``static/<name>_eigen.h5`` record them.

Input
-----

.. list-table::
   :widths: 20 12 12 56
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``Mode``
     - ``td`` | ``kh``
     - ``td``
     - ``kh``: static KH problem (no propagation, ``EnablePropagation`` is
       ignored).
   * - ``KHWindow``
     - ``t0, t1``
     - --
     - Averaging window (a.u.); default one cycle at the field peak.
   * - ``KHSamples``
     - int
     - 0
     - Gauss nodes per window (rounded up to a multiple of 8); 0 chooses
       them automatically.

The laser (``Polarization``, ``LaserX/Y/Z``, ``FinalTime`` covering the
pulse) is required; the mass must be constant (the quiver of a free carrier);
``FieldTerm`` is not supported. Linear, elliptical and circular polarisation in
1D, 2D and 3D are handled alike.

Example: a soft-core atom dressed by a strong high-frequency field, linear
along :math:`z`:

.. code-block:: text

   Mode         = kh
   Dimension    = 3
   Potential    = -1/sqrt(x*x + y*y + z*z + 1)
   Polarization = z
   Variables    = E0=5.0, w=1.0, T=62.83
   LaserZ       = E0*sin(pi*t/T)^2*sin(w*t)
   FinalTime    = 62.83
   NBoundStates = 6
   NBoundStatesSave = 1

Validation
----------

``tests/test_kh.py`` compares ``Mode = kh`` with exact results. For an
oscillator :math:`V = \tfrac12 m\omega_0^2 r^2`,
:math:`V_{\text{KH}} = V + \tfrac12 m\omega_0^2\langle|\boldsymbol\alpha|^2\rangle`,
so every level shifts by :math:`\tfrac14 m\omega_0^2\alpha_0^2` for a linear
field and :math:`\tfrac12 m\omega_0^2\alpha_0^2` for a circular one (constant
:math:`|\boldsymbol\alpha|`). The shift is checked against a ``Mode = td``
run on the same grid (the discretisation cancels, so this tests the quiver),
and the levels against :math:`\sum_a \omega_a(n_a + \tfrac12)` + shift. A
Gaussian well in a :math:`\sin^2` pulse (window, period and quiver found
automatically) is compared with an independent reference: the quiver
integrated with SciPy and :math:`V_{\text{KH}}` diagonalised by fourth-order
finite differences.

.. list-table:: Relative errors
   :widths: 40 30 30
   :header-rows: 1

   * - Case
     - Shift vs ``Mode = td``
     - Levels vs exact / reference
   * - 1D oscillator, :math:`m = 2`, linear
     - :math:`7\times10^{-12}`
     - :math:`3\times10^{-11}`
   * - 2D oscillator, circular
     - :math:`1\times10^{-13}`
     - :math:`2\times10^{-8}`
   * - 3D anisotropic oscillator, linear along :math:`z`
     - :math:`6\times10^{-12}`
     - :math:`2\times10^{-3}` (coarse test grid)
   * - 1D Gaussian well, :math:`\sin^2` pulse
     - --
     - :math:`1.5\times10^{-7}` (finite-difference reference)
