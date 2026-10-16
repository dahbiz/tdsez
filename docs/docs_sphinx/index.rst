TDSE-Z: Time-Dependent Schrödinger Equation Solver
==================================================

TDSE-Z solves the time-dependent Schrödinger equation (TDSE) for a single
particle in one, two and three dimensions, from atoms and molecules in strong
laser fields to semiconductor heterostructures with a position-dependent
effective mass. Space is discretised with B-splines through isogeometric
analysis [Hughes2005]_ using PetIGA [Dalcin2016]_; linear algebra, time
stepping, eigenproblems and MPI parallelism are provided by PETSc [PETSc]_ and
SLEPc [Hernandez2005]_. All quantities are in atomic units (a.u.).

Key features
------------

**Physics**

- Potentials :math:`V(\mathbf r)` and effective masses :math:`m(\mathbf r)`
  given as muParser expressions in the input file.
- Hermitian BenDaniel–Duke kinetic operator
  :math:`-\tfrac12\nabla\cdot m^{-1}(\mathbf r)\nabla` [BenDanielDuke1966]_.
- Length-gauge coupling to laser fields along :math:`x`, :math:`y` and
  :math:`z`; electrons (``Charge = -1``, default) or holes (``Charge = +1``).
- Neumann or Dirichlet boundaries and a Manolopoulos complex absorbing
  potential (CAP) [Manolopoulos2002]_.

**Discretisation and solvers**

- B-spline basis of degree p from 1 to 14 (``Basis = bspline``, default), or
  the :math:`C^0` element bases ``bernstein``, ``lagrange`` and ``spectral``.
- Uniform, graded, hydrogenic and interface-aligned knot sequences.
- Bound states from shift-invert Krylov–Schur [Stewart2002]_ with an ordered
  list of inner linear solvers (direct LU, then iterative ASM/ILU), a retry
  with a larger subspace, and acceptance by the eigenpair backward error.
- Time propagators ``cn`` (Crank–Nicolson [Crank1947]_), ``pade22``
  (Padé(2,2) exponential midpoint rule) and ``magnus4`` (fourth-order
  commutator-free Magnus [BlanesMoan2006]_ [Alvermann2011]_).
- Preconditioners ``asm`` (restricted additive Schwarz with ILU
  [CaiSarkis1999]_), ``mg`` (geometric B-spline multigrid by knot insertion)
  and ``gamg`` (PETSc algebraic multigrid).

**Observables and output**

- Dipole, velocity and acceleration, energy, currents, bound-state
  populations, autocorrelation and wavefunction snapshots, written to HDF5.
- Photoelectron momentum spectra by t-SURFF [TaoScrinzi2012]_ (``TSurff = 1``).
- Post-processors for transition dipole matrices and recombination dipoles.

**Performance**

- Sum-factorised element assembly with BLAS contractions and banded insertion
  of element matrices.
- Factorised (Kronecker) application of separable operators and
  quadrature-point evaluation of monitor expectation values in 2D and 3D.

Where to start
--------------

- :doc:`Overview` — governing equations and solver workflow.
- :doc:`Compilation` — dependencies, build and first run.
- :doc:`Input-File` — input syntax and the full key reference.
- :doc:`Examples` — worked inputs.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   Overview
   Compilation
   Input-File
   CLI-Flags
   Examples

.. toctree::
   :maxdepth: 2
   :caption: Model and discretisation

   Grid-Domain
   Knot-Sequences
   Boundary-Conditions
   Potential-Mass
   Laser-Field
   Physics-Conventions

.. toctree::
   :maxdepth: 2
   :caption: Solvers

   Eigensolver
   KH-Frame
   Initial-State
   Time-Propagation

.. toctree::
   :maxdepth: 2
   :caption: Output and analysis

   Observables
   Output-Control
   HDF5-Output
   Post-Processing
   TSurff
   HHG

.. toctree::
   :maxdepth: 2
   :caption: Internals and operation

   Architecture
   API-Reference
   Performance
   MPI-Notes
   Troubleshooting
   References

Citation
--------

If you use TDSE-Z in your research, please cite:

   Dahbi, Zakaria, and Amelle Zaïr. “Unified Strong-Field Dynamics Simulations
   from Atoms to Heterostructures.” *arXiv preprint* arXiv:2608.18472 (2026).
   `arXiv:2608.18472 <https://arxiv.org/abs/2608.18472>`_.

.. raw:: html

   <div style="text-align: center; margin-top: 2em;">
   <p><strong>Zakaria Dahbi</strong><br>King's College London</p>
   <p>Z.D. acknowledges funding from UK Research and Innovation (UKRI) under the UK
   government's Horizon Europe funding guarantee [Grant No. EP/Z000807/1].</p>
   <p><img src="https://upload.wikimedia.org/wikipedia/commons/9/9c/UKRI_EPSR_Council-Logo_Horiz-RGB.png"
           alt="UKRI Engineering and Physical Sciences Research Council" width="220"></p>
   </div>
