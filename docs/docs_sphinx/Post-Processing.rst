Post-Processing
===============

The standalone tools read the static outputs of a completed TDSE-Z run. Run
them from the directory containing ``static/``. For input ``file.inp``, use
``file.inp`` as the stem, including its extension. Generate the required files
with::

    NBoundStatesSave = 1
    SaveDipoleMatrix = 1
    SaveDipoleAxes = x

This writes ``static/EigenData_file.inp.h5`` (including ``psi_0``, ``psi_1``,
and later states) and ``static/Dx_file.inp.bin``. The tools below operate on
the x dipole operator. They currently have CPU numerical regression coverage
for a one-dimensional harmonic oscillator; continuum results remain
unvalidated.

tdmprocessor
------------

Compute the full transition dipole matrix
:math:`d_{ij}=\langle\psi_i|D_x|\psi_j\rangle`::

    ./tdmprocessor file.inp

The outputs are ``static/TDM_Dx_file.inp.npy`` (complex matrix),
``static/EigenEnergies_file.inp.npy``, and ``static/States_file.inp.npy``.

tdmselect
---------

Compute one row of the transition dipole matrix for states whose energy is
strictly above ``Ethr``::

    ./tdmselect file.inp -i 0 -Ethr 0.2 [-w0 0.05655]

``-i`` is the bra state index. The output is
``static/dij_select_file.inp_i0.csv``. It includes each selected state energy,
complex dipole, squared magnitude, and density-of-states estimate. ``-w0``
sets the reference frequency used for the ``x`` column.

drecprocessor
-------------

Compute a finite-difference recombination-dipole estimate from neighboring
box eigenstates::

    ./drecprocessor file.inp -i 0 -Ethr 0.2 [-w0 0.05655] [-g 0.5]

The output is ``static/drec_resolved_file.inp.csv``. This construction uses
neighboring finite-box eigenstates to approximate an energy derivative. Its
continuum interpretation and numerical accuracy require a separate physical
reference case before release claims are made.
