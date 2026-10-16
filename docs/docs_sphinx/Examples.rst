Examples
========

The repository contains two sets of input decks: the strong-field examples in
``examples/``, documented with measured results, and the reference decks in
``tests/inputs/``, which the test suite runs against analytic values.

Running a Deck
--------------

``tdsez.x`` is built in ``build/`` (:doc:`Compilation`). It writes its output
relative to the working directory: ``static/<name>_eigen.h5`` for the
eigenstates and ``td/<name>_observables.h5`` for the time series, where
``<name>`` is the deck's file name without extension (:doc:`HDF5-Output`). Always launch
through ``mpiexec`` with ``--bind-to none`` (:doc:`MPI-Notes`):

::

    mpiexec -n 1 --bind-to none build/tdsez.x -inp tests/inputs/ho1d.inp
    mpiexec -n 4 --bind-to none build/tdsez.x -inp tests/inputs/ho2d.inp

Strong-Field Examples (``examples/``)
-------------------------------------

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Files
     - Content
   * - ``examples/hhg/hhg_1d.inp``,
       ``examples/hhg/hhg_spectrum.py``
     - High-harmonic generation from a 1D soft-core atom (800 nm,
       :math:`10^{14}` W/cm², flat-top pulse); spectrum from the
       acceleration, the current and the dipole. See :doc:`HHG`.
   * - ``examples/tsurff/free_packet_1d.inp``,
       ``examples/tsurff/plot_tsurff.py``
     - t-SURFF benchmark: free Gaussian wave packet in a laser pulse against
       the analytic spectrum. See :doc:`TSurff`.
   * - ``examples/tsurff/ati_1d.inp``,
       ``examples/tsurff/plot_tsurff.py``
     - Above-threshold ionisation spectrum of a 1D soft-core atom with
       t-SURFF. See :doc:`TSurff`.

Run these decks from their own directory, as shown on the linked pages, so
that the post-processing scripts find ``td/`` next to them.

Reference Decks (``tests/inputs/``)
-----------------------------------

Harmonic oscillator
~~~~~~~~~~~~~~~~~~~

The oscillator decks use :math:`m = 1`, :math:`\omega = 0.2`, degree-5
B-splines on uniform knots and no field. In atomic units

.. math::
   :label: eq-examples-ho

   H = -\frac{1}{2}\nabla^2 + \frac{1}{2}\omega^2 r^2 ,\qquad
   E = \Big(n_1 + \dots + n_d + \frac{d}{2}\Big)\,\omega ,

so the levels of :eq:`eq-examples-ho` have the degeneracies 1, 2, 3 in 2D and
1, 3, 6 in 3D.

.. list-table::
   :widths: 30 40 30
   :header-rows: 1

   * - Deck
     - Content
     - Reference values (a.u.)
   * - ``ho1d.inp``
     - 1D, 3 eigenstates, no propagation
     - 0.1, 0.3, 0.5
   * - ``ho2d.inp``
     - 2D, 6 eigenstates
     - 0.2, 0.4 (×2), 0.6 (×3)
   * - ``ho3d.inp``
     - 3D, 7 eigenstates, direct inner solver
     - 0.3, 0.5 (×3), 0.7 (first 3 of the six-fold level)
   * - ``ho1d_state1.inp``
     - ``InitialState = state:1``
     - initial state :math:`E_1 = 0.3`
   * - ``ho1d_sup2.inp``
     - ``InitialState = sup: 0.7*0 + 0.7*1``, normalised to
       :math:`1/\sqrt2` per state
     - populations 0.5 / 0.5
   * - ``ho1d_sup4.inp``
     - four-term superposition (rejected: at most three terms)
     - validation error
   * - ``ho1d_supprop.inp``
     - superposition of states 0 and 1, short propagation
     - populations 0.5 / 0.5
   * - ``ho1d_supq_real.inp``, ``ho1d_supq_imag.inp``
     - :math:`(|0\rangle + |1\rangle)/\sqrt2` and
       :math:`(|0\rangle + i|1\rangle)/\sqrt2`, one beat period
       :math:`T = 2\pi/\omega`
     - :math:`\langle x\rangle` oscillates with amplitude
       :math:`x_{01} = (2\omega)^{-1/2} = 1.5811`; the two signals are
       :math:`90^\circ` out of phase
   * - ``ho1d_sup_complexi.inp``
     - :math:`(|0\rangle + i|1\rangle)/\sqrt2` with propagation
     - :math:`\langle x\rangle(0) = 0`

Soft-core molecules and atoms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Deck
     - Content
   * - ``h2p.inp``, ``h2p_2d.inp``
     - Soft-core :math:`\mathrm{H}_2^+` model
       :math:`V = -1/\sqrt{(x+1)^2+1.44} - 1/\sqrt{(x-1)^2+1.44}` in 1D (and
       the same potential, independent of :math:`y`, in 2D) with a
       trapezoidal 800 nm pulse; short propagation (:math:`T = 10`) through
       the full eigensolver, propagation and HDF5 output path.
   * - ``bench_h1d.inp``
     - 1D soft-core hydrogen :math:`V = -1/\sqrt{x^2+1.44}`, 4-cycle sin²
       vector-potential pulse at 800 nm (:math:`E_0 = 0.0534`), 740 a.u. of
       propagation without CAP.
   * - ``bench_h3d.inp``
     - The same pulse on the 3D soft-core atom :math:`-1/\sqrt{r^2+1.44}`
       (degree 7, 100 elements per axis); a large run intended for
       benchmarking on many cores.

Decks that must fail
~~~~~~~~~~~~~~~~~~~~

These decks check the input validation; each run stops with a ``FATAL`` or
``ValidateOrThrow`` message (:doc:`Troubleshooting`).

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Deck
     - Rejected because
   * - ``bad_unknown_key.inp``
     - unknown input key (``StrictInput = 1``)
   * - ``bad_bad_degree.inp``
     - ``SplineDegree = 20`` outside [1, 14]
   * - ``bad_inverted_domain.inp``
     - lower domain bound above the upper one
   * - ``bad_zero_mass.inp``
     - ``Mass = 0``
   * - ``bad_adaptive_nodv.inp``, ``bad_adaptivewf_nodv.inp``
     - unknown ``KnotSequence`` names
   * - ``ho1d_statebad.inp``
     - ``InitialState = state:5`` with fewer bound states requested
   * - ``ho1d_supbad.inp``
     - superposition term with state 5 while ``NBoundStates = 4``
