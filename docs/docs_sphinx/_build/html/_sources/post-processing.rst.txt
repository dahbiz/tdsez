post-processing
===============

Post-Processing
===============

Three standalone post-processor binaries built alongside tdsez.

tdmprocessor
------------

Loads saved dipole operator Dx (PETSc binary) and eigenstates from HDF5, computes
d_ij = <psi_i|Dx|psi_j> block-by-block. Standalone; needs only PETSc.

tdmselect
---------

CLI: ``-i <bra> -Ethr <E> [-w0 <omega0>] [-gpu]``. Writes CSV of dipole matrix
elements above energy threshold.

Save Dipole Operators
---------------------

Set ``SaveDipoleMatrix = 1`` and ``SaveDipoleAxes = x | xy | xyz | all | none``
to persist Dx to ``static/Dx_*.bin``. Loadable with ``MatLoad`` in PETSc.
