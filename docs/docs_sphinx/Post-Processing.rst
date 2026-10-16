Post-Processing
===============

TDSE-Z builds three post-processing tools next to ``tdsez.x``. They compute
transition dipoles between the eigenstates of a finished run, in 1D, 2D and
3D, on any saved axis:

.. list-table::
   :widths: 22 40 38
   :header-rows: 1

   * - Tool
     - Computes
     - Writes (``static/``)
   * - ``dipoles.x``
     - the full matrices :math:`d^{(a)}_{ij} = \langle i|x_a|j\rangle` for
       every saved axis, the energies, :math:`\omega_{ij}` and a few-level
       model manifest
     - ``<name>_tdm_Dx.npy`` (``_Dy``, ``_Dz``), ``<name>_energies.npy``,
       ``<name>_frequencies.npy``, ``<name>_tdm_magnitude.npy``,
       ``<name>_states.npy``, ``<name>_fewlevel.json``
   * - ``dipole_spectrum.x``
     - one row :math:`d_j = \langle i|x_a|j\rangle` from a chosen state
       :math:`i` to every state above an energy, with the density of states
     - ``<name>_dij_D<a>_i<i>.csv``
   * - ``recombination.x``
     - a recombination dipole :math:`d_{\text{rec}}(E)` for one open channel,
       from neighbouring box states
     - ``<name>_drec_D<a>.csv``

``<name>`` is the input file name without its extension (``h2p.inp`` →
``h2p``), as for every output of a run (:doc:`HDF5-Output`). The Python
package ``zkit`` reads these files (:ref:`post-processing-zkit`).

Quick Start
-----------

A 2D oscillator with different frequencies along :math:`x` and :math:`y`,
from the input file to the dipole matrices. Add to the input file
(``ho2.inp``):

.. code-block:: text

   Dimension        = 2
   Domain           = -14, 14, -14, 14
   SplineDegree     = 6
   NSplines         = 50
   Potential        = 0.5*(0.36*x*x + 0.6889*y*y)   # omega = 0.6, 0.83
   NBoundStates     = 6
   NBoundStatesSave = 1          # save the eigenvectors
   SaveDipoleAxes   = xy         # save the dipole operators D_x, D_y
   EnablePropagation = 0         # eigenstates only

Then, in the run directory:

.. code-block:: bash

   mpiexec --bind-to none -n 4 build/tdsez.x -inp ho2.inp   # static/ho2_eigen.h5, ho2_Dx.bin, ho2_Dy.bin
   build/dipoles.x ho2.inp                                   # all matrices
   build/dipole_spectrum.x ho2.inp -i 0 -Ethr -1 -axis y     # row of D_y from state 0
   build/recombination.x ho2.inp -i 0 -Ethr -1 -axis x       # d_rec(E) along x

and read the results in Python:

.. code-block:: python

   import numpy as np
   E  = np.load("static/ho2_energies.npy")      # energies, in the order of the eigen file
   Dx = np.load("static/ho2_tdm_Dx.npy")        # complex (N, N), Dx[i, j] = <i|x|j>
   print(abs(Dx[0, 1]), np.sqrt(1 / (2 * 0.6)))  # 0.912871 0.912871

Requirements and Conventions
----------------------------

**What the run must save.** ``NBoundStatesSave = 1`` writes the eigenvectors
to ``static/<name>_eigen.h5`` (``coeff_i``, and ``psi_i``);
``SaveDipoleAxes = x``, ``xy``, ``xyz`` (or ``all``) writes the dipole
operators :math:`\mathbf D_a`, :math:`(D_a)_{kl} = \int B_k\,x_a\,B_l`, to
``static/<name>_D<a>.bin`` (PETSc binary); ``SaveDipoleMatrix = 1`` saves
:math:`D_x` only, and ``-save_dipole xyz`` on the command line does the same as
``SaveDipoleAxes``. A stationary run (``EnablePropagation = 0``) is enough; it
needs no ``PotentialDerivative`` expressions.

**Running.** Start the tools in the run directory (the one that contains
``static/``). The first argument is the input file, ``ho2.inp`` or ``ho2``;
runs written by earlier versions (``static/EigenData_ho2.inp.h5``,
``static/Dx_ho2.inp.bin``) are read as well. The saved matrices are
sequential PETSc matrices, so each tool runs on one MPI rank (``mpiexec -n 1``
or no ``mpiexec``). ``-gpu`` loads them as CUDA matrices and vectors on one GPU
(PETSc built with CUDA).

**State order.** The eigen file stores the states in the order of the
eigensolver (by distance from ``TargetEigenvalue``), which is not always the
energy order. ``dipoles.x`` and ``dipole_spectrum.x`` index states in this
stored order (``<name>_energies.npy`` gives their energies);
``recombination.x`` sorts them by energy and its ``-i`` is an index in that
order.

**What is computed.** With the eigenvector coefficients :math:`\mathbf c_i`,

.. math::
   :label: eq-post-processing-tdm

   d^{(a)}_{ij} = \langle\psi_i|\,x_a\,|\psi_j\rangle
   = \mathbf c_i^\dagger\,\mathbf D_a\,\mathbf c_j .

The phase of each eigenvector is arbitrary, so the phase of
:math:`d_{ij}` is too; :math:`|d_{ij}|` and products around closed loops are
physical. Within a degenerate level the individual states, and their
:math:`d_{ij}`, depend on the eigensolver; sums over the level do not.

dipoles.x
---------

.. code-block:: text

   dipoles.x <input> [-gpu]

Computes :math:`d^{(a)}_{ij}` for every saved axis :math:`a` (every
``static/<name>_D<a>.bin`` found). :math:`\mathbf D_a` is Hermitian, so only
the upper triangle is computed and the lower one is its conjugate. Output in
``static/``:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Content
   * - ``<name>_tdm_Dx.npy`` (``_Dy``, ``_Dz``)
     - complex :math:`N\times N` matrix, element ``[i, j]`` is
       :math:`d^{(a)}_{ij}`, one file per saved axis
   * - ``<name>_tdm_Dx.txt`` (``_Dy``, ``_Dz``)
     - the same as text (``i j Re Im |d|``), for :math:`N \le 200`
   * - ``<name>_tdm_magnitude.npy``
     - :math:`|\mathbf d_{ij}| = \big(\sum_a |d^{(a)}_{ij}|^2\big)^{1/2}`
       over the saved axes
   * - ``<name>_energies.npy``
     - energies :math:`E_i` (``float64``)
   * - ``<name>_states.npy``
     - state indices :math:`0, \dots, N-1` (``int32``)
   * - ``<name>_frequencies.npy``
     - :math:`\omega_{ij} = E_j - E_i`
   * - ``<name>_fewlevel.json``
     - manifest of the files above, with units and the length-gauge coupling
       :math:`H_{ij}(t) = E_i\,\delta_{ij} + \mathbf E(t)\cdot\mathbf d_{ij}`

Together the ``.npy`` files define a truncated eigenstate (few-level) model of
the system:

.. code-block:: python

   import json, numpy as np
   m  = json.load(open("static/ho2_fewlevel.json"))
   E  = np.load("static/" + m["energies"])
   D  = [np.load("static/" + f) for f in m["dipole_components"]]   # D_x, D_y
   H  = lambda t, Ex, Ey: np.diag(E) + Ex(t) * D[0] + Ey(t) * D[1]

dipole_spectrum.x
-----------------

.. code-block:: text

   dipole_spectrum.x <input> -i <state> -Ethr <E> [-axis x|y|z] [-w0 <omega0>] [-gpu]

One row of the dipole matrix, :math:`d_j = d^{(a)}_{ij}`, from the state
:math:`i` to every state with :math:`E_j > E_{\text{thr}}`, with the density
of states, e.g. for ionisation or absorption spectra from the ground state.

.. list-table::
   :widths: 18 18 64
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``-i``
     - 0
     - the state :math:`i`, index in ``static/<name>_eigen.h5`` (stored order)
   * - ``-Ethr``
     - 0.0
     - keep the states with :math:`E_j > E_{\text{thr}}` (a.u.); a large
       negative value keeps all
   * - ``-axis``
     - ``x``
     - dipole component; ``static/<name>_D<a>.bin`` must be saved
   * - ``-w0``
     - 0.05655
     - reference frequency for the column ``x``
   * - ``-gpu``
     - off
     - CUDA matrices and vectors

Output: ``static/<name>_dij_D<a>_i<i>.csv``, full double precision:

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Column
     - Meaning
   * - ``j``, ``E_j``
     - state index (stored order) and energy
   * - ``x``
     - :math:`(E_j - E_i)/\omega_0` (photon order at :math:`\omega_0`)
   * - ``Re_d``, ``Im_d``
     - :math:`d_j`
   * - ``abs_d2``, ``log10_abs_d2``
     - :math:`|d_j|^2` and its logarithm
   * - ``rho``, ``log10_rho_abs_d2``
     - density of states :math:`\rho_j = 1/(E_{j+1} - E_j)` from the
       energy-ordered spectrum, and :math:`\log_{10}(\rho_j |d_j|^2)`

Example, ground-state dipoles along :math:`z` in 3D:

.. code-block:: bash

   build/dipole_spectrum.x ho3.inp -i 0 -Ethr -1e9 -axis z   # static/ho3_dij_Dz_i0.csv

recombination.x
---------------

.. code-block:: text

   recombination.x <input> -i <state> -Ethr <E> [-axis x|y|z] [-w0 <omega0>] [-g <g>] [-gpu]

A box eigenstate :math:`\psi_j` in the continuum is a real standing wave, a
mixture of the regular solution and its energy derivative. The tool
approximates the derivative by a central difference over the neighbouring
states (sorted by energy, after fixing the phase of every state so that it
varies smoothly with energy) and forms

.. math::
   :label: eq-post-processing-drec

   \mathbf w_j = \frac{\psi_{j+1} - \psi_{j-1}}{E_{j+1} - E_{j-1}},
   \qquad
   d_{\text{rec}}(E_j) = \langle\psi_i|x_a|\psi_j\rangle
   + i\,g\,\frac{E_{j+1} - E_{j-1}}{2}\,\langle\psi_i|x_a|\mathbf w_j\rangle

(one-sided at the ends of the spectrum).

.. list-table::
   :widths: 18 18 64
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``-i``
     - 0
     - the state :math:`i`, index in the **energy-ordered** spectrum
   * - ``-Ethr``
     - -5.0
     - keep the states with :math:`E_j > E_{\text{thr}}`
   * - ``-axis``
     - ``x``
     - dipole component
   * - ``-w0``
     - 0.05655
     - reference frequency for the column ``x``
   * - ``-g``
     - 0.5
     - weight :math:`g` of the energy-derivative term
   * - ``-gpu``
     - off
     - CUDA matrices and vectors

Output: ``static/<name>_drec_D<a>.csv`` (full double precision), one row per
state in energy order:

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Column
     - Meaning
   * - ``j``, ``E_j``, ``x``
     - index in energy order, energy, :math:`(E_j - E_i)/\omega_0`
   * - ``Re_d``, ``Im_d``
     - :math:`d_{\text{rec}}(E_j)`
   * - ``abs_d2``, ``log10_abs_d2``
     - :math:`\rho_j\,|d_{\text{rec}}|^2` with
       :math:`\rho_j = 2/(E_{j+1} - E_{j-1})`, and its logarithm
   * - ``Re_dj``, ``Im_dj``
     - the regular term :math:`\langle\psi_i|x_a|\psi_j\rangle`
   * - ``Re_dw``, ``Im_dw``
     - :math:`\langle\psi_i|x_a|\mathbf w_j\rangle`
   * - ``dEhalf``, ``rho``
     - :math:`(E_{j+1} - E_{j-1})/2` and :math:`\rho_j`

The regular term equals the corresponding row of ``dipoles.x``. The
energy-derivative construction itself has no exact reference here; check it
against a scattering calculation for the system at hand.

Validation
----------

``tests/test_postprocessing.py`` and ``zkit validate`` (:mod:`zkit.analytic`)
run a stationary TDSE-Z calculation for each system below, then the three
tools. They compare the spectrum and every :math:`|d_{ij}|` of ``dipoles.x``
with exact results, and the rows of ``dipole_spectrum.x`` and
``recombination.x`` (regular term) with the full matrix on every axis. The
oscillators have a different frequency per axis, so the low levels are
non-degenerate and are matched by energy.

.. list-table:: Largest relative errors (6 states)
   :widths: 30 34 18 18
   :header-rows: 1

   * - System
     - Exact result
     - Energies
     - :math:`|d_{ij}|`
   * - Oscillator 1D, :math:`\omega = 0.5`
     - :math:`\langle n|x|n{+}1\rangle = \sqrt{(n+1)/2\omega}`
     - :math:`1.6\times10^{-12}`
     - :math:`3.1\times10^{-12}`
   * - Box 1D, :math:`L = 10`, hard walls
     - :math:`\langle n|x|m\rangle = -\frac{8Lnm}{\pi^2(n^2-m^2)^2}`, :math:`n+m` odd
     - :math:`1.8\times10^{-15}`
     - :math:`4.9\times10^{-15}`
   * - Oscillator 2D, :math:`\omega = (0.6, 0.83)`
     - products of the 1D results, :math:`x` and :math:`y`
     - :math:`6.4\times10^{-8}`
     - :math:`8.2\times10^{-8}`
   * - Oscillator 3D, :math:`\omega = (0.6, 0.83, 1.17)`, 24 elements per axis
     - :math:`x`, :math:`y` and :math:`z`
     - :math:`4.6\times10^{-4}`
     - :math:`1.1\times10^{-4}`

The rows of ``dipole_spectrum.x`` and ``recombination.x`` equal those of
``dipoles.x`` to rounding (:math:`<10^{-26}`). The 3D errors are those of the
coarse test grid and decrease with it.

Troubleshooting
---------------

``no saved dipole operator found``, ``static/<name>_D<a>.bin not found``
   The run did not save that axis: set ``SaveDipoleAxes`` (``xyz`` for every
   axis) and rerun the stationary calculation.

``sequential dipole files require a single MPI rank``
   Run the tool without ``mpiexec`` (or with ``-n 1``).

``-i`` out of range
   ``-i`` counts from 0 and must be below ``NBoundStates``; remember that
   ``recombination.x`` uses the energy order.

Dipoles that disagree with a reference by a phase or within a degenerate level
   Compare :math:`|d_{ij}|`, or sums over the degenerate level; the individual
   phases and degenerate partners are not unique.

.. _post-processing-zkit:

Python Analysis: zkit
---------------------

``zkit`` reads every file of a run. For the transition dipoles:

.. code-block:: python

   import zkit
   from zkit.io.tdm import read_tdm
   from zkit.analytic import oscillator_tdm, compare_tdm

   run = zkit.Run(".", "ho2.inp")                # run directory, input file
   tdm = read_tdm(run.path("eigen"))             # {"tdm_x": (N, N), "tdm_y": ...} from dipoles.x
   ref = oscillator_tdm((0.6, 0.83), n_states=6)
   print(compare_tdm(tdm, run.spectrum.values, ref))   # relative errors vs exact

and from the shell:

.. code-block:: bash

   zkit --run-dir . tdm-plot ho2.inp --outdir figures   # level diagram and |d_ij| map
   zkit validate --quick                                # TDSE-Z and the tools against exact results

More generally ``zkit`` provides spectra, the time series, wavefunction
reconstruction from the stored knot vectors and Wigner transforms. The TDSE-Z
test suite uses it::

    run = zkit.load("run_dir", "h2p.inp")   # run directory, input file name
    E   = run.spectrum.values               # eigenvalues
    ev  = run.evolution                     # dipoles, energies, currents, ...

``zkit --help`` lists every command.
