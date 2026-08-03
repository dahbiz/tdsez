Overview & Architecture
=======================

High-level workflow and file structure of the TDSE-Z framework.

Solver Workflow
---------------

TDSE-Z solves the Time-Dependent Schrödinger Equation (TDSE) using isogeometric analysis (B-spline basis functions) with MPI parallelism and HDF5 output.

::

    1. Parse input file (.prm)
    2. Build IGA mesh (knot vectors, basis functions)
    3. Assemble Hamiltonian H and mass matrix M
    4. Solve TISE: H|psi_n> = E_n M|psi_n>  (SLEPc EPS)
    5. Select initial state |psi(0)> -- ground, excited, or superposition
    6. Propagate: i dpsi/dt = H|psi> + E(t)*D|psi>
    7. Compute observables: dipole, energy, populations, currents, autocorrelation
    8. Write HDF5 output at configured strides

File Structure
--------------

::

    tdsez.cpp         // main entry -- orchestrates full pipeline
    +-- TDSEZParser   // key=value input parsing (muParser)
    +-- TDSEZCore     // IGA setup, knot generation, TISE solve
    +-- TDSEZAssembler // builds H, M, K, V, Dx, Dy, Dz, Lz
    +-- TDSEZManager  // runtime: observables, HDF5 buffers, output
    +-- TDSEZPropagator // PETSc TS driver (Crank-Nicolson theta=0.5)

Residual Assembly
-----------------

The Crank-Nicolson time-stepper evaluates the residual equation:

.. math::
   :nowrap:

   \begin{equation}
   F = iM \frac{\partial \psi}{\partial t} - H\psi - E(t)D\psi = 0
   \end{equation}

where :math:`\frac{\partial \psi}{\partial t}` is the time derivative, M is the mass matrix, H is the Hamiltonian,
and :math:`E(t)D` is the laser coupling term.

Output Structure
----------------

::

    td/                  # time-evolution output
      TimeEvolutionData_<input>.h5  # time-series observables
      wfs_<input>.h5             # wavefunction snapshots
      ac_<input>.h5             # autocorrelation function
    static/                # eigenproblem output
      EigenData_<input>.h5    # energy spectrum + eigenstates
      GS_<input>.log         # degenerate-splitting log (2D)
