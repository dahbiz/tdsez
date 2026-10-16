Post-Processing
===============

TDSE-Z ships three post-processing programs for bound-continuum transition
dipoles, built with the solver (``tdmprocessor``, ``tdmselect``,
``drecprocessor``), and the Python package ``zkit`` for general analysis of
the HDF5 output (:doc:`HDF5-Output`).

Input Files
-----------

The programs read the static output of a completed run and are started from
the run directory (the one that contains ``static/``). Their first argument
is the input file, ``file.inp`` or just ``file``; every file of the run is
named after it (``static/file_*``). The run must save the eigenstates and the
position matrix::

    NBoundStatesSave = 1
    SaveDipoleMatrix = 1        # Dx only; or SaveDipoleAxes = xyz

This writes ``static/file_eigen.h5`` (with ``psi_i`` and ``coeff_i``)
and ``static/file_Dx.bin`` (and ``_Dy``, ``_Dz`` with
``SaveDipoleAxes``). Runs of earlier versions (``EigenData_file.inp.h5``,
``Dx_file.inp.bin``) are read as well. The eigenstates are read from ``coeff_i``, which has the
ordering of the saved matrix (``psi_i`` for files without it). The PETSc
binary matrices are sequential, so each program runs on a single MPI rank;
``-gpu`` loads them as CUDA matrices and vectors on one GPU.

The programs compute matrix elements

.. math::
   :label: eq-post-processing-tdm

   d^{(a)}_{ij} = \langle\psi_i|\,x_a\,|\psi_j\rangle
   = \mathbf c_i^\dagger\,\mathbf D_a\,\mathbf c_j

between eigenstates of the box. They are covered by numerical regression
tests for a one-dimensional harmonic oscillator; results for continuum
(box) states have not been validated against a physical reference.

tdmprocessor
------------

Full transition-dipole matrix for every saved Cartesian component::

    ./tdmprocessor file.inp

Since :math:`\mathbf D_a` is Hermitian only the upper triangle is computed.
Output in ``static/``:

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - File
     - Content
   * - ``file_tdm_Dx.npy`` (``_Dy``, ``_Dz``)
     - complex :math:`N\times N` matrix, element ``[i, j]`` is
       :math:`d^{(a)}_{ij}`, one file per saved component
   * - ``file_tdm_Dx.txt`` (``_Dy``, ``_Dz``)
     - the same as text, for :math:`N \le 200`
   * - ``file_tdm_magnitude.npy``
     - :math:`|\mathbf d_{ij}| = \big(\sum_a |d^{(a)}_{ij}|^2\big)^{1/2}`
       over the saved components
   * - ``file_energies.npy``
     - energies :math:`E_i` (``float64``)
   * - ``file_states.npy``
     - state indices :math:`0, \dots, N-1` (``int32``)
   * - ``file_frequencies.npy``
     - :math:`\omega_{ij} = E_j - E_i`
   * - ``file_fewlevel.json``
     - manifest with the file names, units and the length-gauge coupling
       :math:`H_{ij}(t) = E_i\,\delta_{ij} + \mathbf E(t)\cdot\mathbf d_{ij}`

The ``.npy`` files are read with ``numpy.load``; together they define a
truncated eigenstate model of the system.

tdmselect
---------

One row of the transition-dipole matrix, :math:`d_j = d^{(x)}_{ij}` for a
fixed bra state :math:`i` and all states with :math:`E_j > E_{\text{thr}}`::

    ./tdmselect file.inp -i 0 -Ethr 0.2 [-w0 0.05655]

.. list-table::
   :widths: 18 18 64
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``-i``
     - 0
     - index of the bra state in ``static/file_eigen.h5``
   * - ``-Ethr``
     - 0.0
     - energy threshold; states with :math:`E_j > E_{\text{thr}}` are kept
   * - ``-w0``
     - 0.05655
     - reference frequency for the column ``x``

Output: ``static/file_dij_i0.csv`` with the columns ``j, E_j, x,
Re_d, Im_d, abs_d2, log10_abs_d2, rho, log10_rho_abs_d2``, where
:math:`x = (E_j - E_i)/\omega_0`, ``abs_d2`` is :math:`|d_j|^2` and
:math:`\rho_j = 1/(E_{j+1} - E_j)` is the density of states from the
energy-ordered spectrum.

drecprocessor
-------------

Recombination dipole for one open channel from neighbouring box
eigenstates::

    ./drecprocessor file.inp -i 0 -Ethr 0.2 [-w0 0.05655] [-g 0.5]

A box eigenstate :math:`\psi_j` is a real mixture of the regular solution and
its energy derivative. The program approximates the derivative by a central
difference over the neighbouring states, sorted by energy,

.. math::
   :label: eq-post-processing-drec

   \mathbf w_j = \frac{\psi_{j+1} - \psi_{j-1}}{E_{j+1} - E_{j-1}},
   \qquad
   d_{\text{rec}}(E_j) = \langle\psi_i|x|\psi_j\rangle
   + i\,g\,\frac{E_{j+1} - E_{j-1}}{2}\,\langle\psi_i|x|\mathbf w_j\rangle ,

(one-sided at the ends of the spectrum), after fixing the phase of every
state so that it varies smoothly with energy.

.. list-table::
   :widths: 18 18 64
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``-i``
     - 0
     - bra state, as index in the energy-ordered spectrum
   * - ``-Ethr``
     - -5.0
     - energy threshold; states with :math:`E_j > E_{\text{thr}}` are kept
   * - ``-w0``
     - 0.05655
     - reference frequency for the column ``x``
   * - ``-g``
     - 0.5
     - weight :math:`g` of the energy-derivative term

Output: ``static/file_drec.csv`` with the columns ``j, E_j, x,
Re_d, Im_d, abs_d2, log10_abs_d2, Re_dj, Im_dj, Re_dw, Im_dw, dEhalf, rho``.
``abs_d2`` is the energy-normalised :math:`\rho_j\,|d_{\text{rec}}|^2` with
the central-difference density of states :math:`\rho_j = 2/(E_{j+1} -
E_{j-1})`; ``dj`` and ``dw`` are the two matrix elements of
:eq:`eq-post-processing-drec`. The construction approximates an energy
derivative from finite-box states; its accuracy for a given system should be
checked against a reference calculation.

Python Analysis: zkit
---------------------

``zkit`` reads every HDF5 file of a run and provides spectra, wavefunction
reconstruction from the stored knot vectors, transition-dipole diagrams and
Wigner transforms. The TDSE-Z test suite uses it::

    import zkit

    run = zkit.load("run_dir", "h2p.inp")   # run directory, input file name
    E   = run.spectrum.values               # eigenvalues
    ev  = run.evolution                     # dipoles, energies, currents, ...
    eig = zkit.open_eig("run_dir/static/h2p_eigen.h5")

``zkit`` also installs a command-line tool (``zkit summary``, ``zkit batch``,
``zkit plot-wfs``, ``zkit tdm-plot``; ``zkit --help`` lists all options).
