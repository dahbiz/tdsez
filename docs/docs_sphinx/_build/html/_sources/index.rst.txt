TDSE-Z: Time-Dependent Schrödinger Equation Solver
==================================================

TDSE-Z is a finite-element solver for the time-dependent Schrödinger equation in 1D, 2D, and 3D. It uses PETSc/SLEPc for linear algebra and eigensolvers, with PetIGA for isogeometric analysis (B-spline basis functions).

Key Features
------------

- **Finite-Element Method**: High-order B-spline basis (degree 1-14), arbitrary spatial dimensions
- **PETSc/SLEPc Backend**: Scalable linear algebra, Krylov solvers, and eigensolvers
- **Isogeometric Analysis**: IGA via PetIGA for C1-continuous basis functions
- **Time Propagation**: Crank-Nicolson implicit time-stepper (unitary, second-order)
-  **Complex Arithmetic**: Full complex-field support for quantum observables
-  **HDF5 Output**: Ring-buffered time-series observables and wavefunction snapshots
- **Position-Dependent Mass**: muParser expressions for M(x,y,z) with quantum correction terms
- **Flexible Potentials**: muParser expressions with analytic or numerical derivatives

Solver Architecture
-------------------

The solver operates in two phases:

1. **Eigensolver Phase** (optional): SLEPc EPS to compute bound states. Eigenstates are stored for use as initial states.
2. **Time Propagation Phase**: PETSc TS with Crank-Nicolson (θ = 0.5). The residual at each step:

.. math::
   :nowrap:

   \begin{equation}
   F = iM \frac{\partial \psi}{\partial t} - H\psi - E(t)D\psi = 0
   \end{equation}

The Hamiltonian in BenDaniel–Duke form (for spatially varying mass):

.. math::
   :nowrap:

   \begin{equation}
   H = -\frac{1}{2} \nabla \cdot \left( \frac{1}{m(\mathbf{r})} \nabla \right) + V(\mathbf{r}) + \text{CAP}
   \end{equation}

For uniform mass :math:`m(\mathbf{r}) = m_0` this reduces to :math:`-\frac{1}{2m_0}\nabla^2`. The divergence term properly accounts for the gradient of the inverse mass, ensuring the correct quantum momentum operator in the position-dependent mass formalism.

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
   :hidden:

   Overview
   Architecture
   API-Reference
   Compilation
   Input-File
   Grid-Domain
   Potential-Mass
   Laser-Field
   Eigensolver
   Time-Propagation
   Output-Control
   Initial-State
   Examples
   Post-Processing
   CLI-Flags
   MPI-Notes
   Troubleshooting
   Boundary-Conditions
   Knot-Sequences
   HDF5-Output
   Observables
   Physics-Conventions

Citation
--------

If you use TDSE-Z in your research, please cite:

   Dahbi, Zakaria, and Amelle Zaïr. “Unified Strong-Field Dynamics Simulations
   from Atoms to Heterostructures.” *arXiv preprint* arXiv:2608.18472 (2026).
   `arXiv:2608.18472 <https://arxiv.org/abs/2608.18472>`_.

Funding acknowledgement
-----------------------

Z.D. acknowledges funding from UK Research and Innovation (UKRI) under the UK
government’s Horizon Europe funding guarantee [Grant No. EP/Z000807/1].
