Post-Processing
===============

Three standalone post-processor binaries built alongside tdsez. These operate on
HDF5 and PETSc binary files produced during a run, enabling analysis without
rerunning the full propagation.

tdmprocessor
------------

Loads saved dipole operator :math:`\mathbf{D}_x` (PETSc binary) and eigenstates
from HDF5, computes :math:`d_{ij} = \langle \psi_i | D_x | \psi_j \rangle`
block-by-block.

Usage
~~~~~

::

    $ ./tdmprocessor -inp file.prm

The input file must contain ``SaveDipoleMatrix = 1`` and a path to the saved
dipole operator. The post-processor loads the operator and all computed
eigenstates, then computes the full dipole matrix.

Output
~~~~~~

The computed matrix :math:`D_{ij}` is stored in the HDF5 output file under the
key ``/dipole_matrix``. This matrix enables computation of transition rates,
oscillator strengths, and multiphoton spectra without recomputing overlaps.

Requirements
~~~~~~~~~~~~

- PETSc installed and linked
- HDF5 library (linked at configure time)
- ``.h5`` file from a completed run
- ``.bin`` dipole operator file in ``static/``

tdmselect
---------

Computes dipole matrix elements above an energy threshold, ideal for selecting
states with significant coupling for further analysis.

Usage
~~~~~

::

    $ ./tdmselect -i state0.h5 -Ethr -1.0 [-w0 0.057] [-gpu]

Options
~~~~~~~

- ``-i <file>`` — Input HDF5 state file (required)
- ``-Ethr <E>`` — Energy threshold in atomic units (required)
- ``-w0 <w>`` — Reference frequency for transition energy calculation (default: 0.057)
- ``-gpu`` — Compute on GPU if available (requires CUDA build)

Output
~~~~~~

Writes a CSV file ``tdm_select_<E>.csv`` with columns:

- State pair indices (i, j)
- Energy difference :math:`E_j - E_i`
- Dipole matrix element magnitude :math:`|d_{ij}|`
- Oscillator strength proxy :math:`f_{ij} \propto (E_j - E_i) |d_{ij}|^2`

Filtering
~~~~~~~~~

Only entries with :math:`|E_j - E_i| > |E_{thr}|` are written. This is useful
for selecting resonant transitions or identifying coupled state pairs for
time-dependent analysis.

Save Dipole Operators
---------------------

Persist the assembled dipole operator :math:`\mathbf{D}_x` for later use in
post-processing tools.

Input File
~~~~~~~~~~

::

    SaveDipoleMatrix = 1
    SaveDipoleAxes = xyz

The ``SaveDipoleAxes`` parameter accepts:

- ``x`` — x-component only
- ``xy`` — x and y components
- ``xyz`` — all three components
- ``all`` — all available components
- ``none`` — do not save (default: ``none``)

Storage
~~~~~~~

The operator is saved to ``static/Dx_*.bin``, ``static/Dy_*.bin``, or
``static/Dz_*.bin`` (one per MPI rank). Load in PETSc with:

::

    $ petscviewer -viewer_binary file.bin

Each file is in PETSc binary format and can be loaded programmatically:

.. code-block:: c

    MatLoad(Dx, PETSC_COMM_WORLD, "static/Dx_0000.bin");

The full dipole operator for any axis combination can be reconstructed by
assembling the per-axis components. This is necessary for computing
:math:`\mathbf{d} = \langle \psi | \mathbf{r} | \psi \rangle` for multi-axis
propagation.
