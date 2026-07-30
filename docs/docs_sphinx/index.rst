TDSE-Z: Time-Dependent Schrödinger Equation Solver
==================================================

TDSE-Z is a finite-element solver for the time-dependent Schrödinger equation in 1D, 2D, and 3D. It uses PETSc/SLEPc for linear algebra and eigensolvers, with PetIGA for isogeometric analysis (B-spline basis functions).

Key Features
------------

- **Finite-Element Method**: High-order B-spline basis (degree 1-14), arbitrary spatial dimensions
- **PETSc/SLEPc Backend**: Scalable linear algebra, Krylov solvers, and eigensolvers
- **Isogeometric Analysis**: IGA via PetIGA for C1-continuous basis functions
- **Time Propagation**: Crank-Nicolson implicit time-stepper (unitary, second-order)
- **Complex Arithmetic**: Full complex-field support for quantum observables
- **GPU Acceleration**: CUDA-enabled PETSc for large-scale wavefunction propagation
- **HDF5 Output**: Ring-buffered time-series observables and wavefunction snapshots
- **Position-Dependent Mass**: muParser expressions for M(x,y,z) with quantum correction terms
- **Flexible Potentials**: muParser expressions with analytic or numerical derivatives

Solver Architecture
-------------------

The solver operates in two phases:

1. **Eigensolver Phase** (optional): SLEPc EPS to compute bound states. Eigenstates are stored for use as initial states.
2. **Time Propagation Phase**: PETSc TS with Crank-Nicolson (θ = 0.5). The residual at each step:

    F = iM · ψ̇ − H · ψ − E(t)·D · ψ

The Hamiltonian: H = −½∇²/m(r) + V(r) + CAP

Input files are plain-text ``key = value`` format. muParser expressions are used for potentials, laser fields, mass, and custom variables. All scalar input parameters are automatically defined as muParser constants.

Output Directory Structure
--------------------------

::

    td/                        # time-evolution output
      TimeEvolutionData_<input>.h5    # time-series observables
      wfs_<input>.h5                  # wavefunction snapshots
      ac_<input>.h5                   # autocorrelation function
    static/                    # eigenproblem output
      EigenData_<input>.h5          # energy spectrum + eigenstates
      GS_<input>.log                # degenerate-splitting log (2D)

Three standalone post-processor binaries:
    - ``tdmprocessor`` — Transition dipole matrix computation
    - ``tdmselect``    — Selective dipole elements for one bra state
    - ``drecprocessor`` — Dipole recombination spectrum

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   build
   run
   input-file
   grid-domain
   potential-mass
   eigensolver
   time-propagation
   laser-field
   boundary-conditions
   knot-sequences
   output-control
   initial-state
   hdf5-output
   post-processing
   cli-flags
   mpi-notes
   observables
   physics-conventions
   troubleshooting
