#ifndef TDSEZ_MANAGER_HPP
#define TDSEZ_MANAGER_HPP

#include <petscviewerhdf5.h>
#include <petscts.h>
#include <hdf5.h>
#include <string>
#include <vector>

class TDSEZCore;
class TDSEZAssembler;

/**
 * @brief Orchestrates time propagation, diagnostics, HDF5 output, and t-SURFF.
 * @details Owns the operator matrices (H, K, V, dipoles, velocity, CAP, Lz),
 * diagnostic buffers, HDF5 file handle, and the t-SURFF surface-flux
 * machinery.  The Manager is constructed with pointers to a TDSEZCore
 * (eigenproblem results) and a TDSEZAssembler (operator builder) and
 * drives the PETSc TS time-stepping through a TDSEZPropagator.
 */
class TDSEZManager
{
    private:
    /// Pointer to the owning TDSEZCore (eigenproblem / IGA setup).
    TDSEZCore* core_;
    /// Pointer to the TDSEZAssembler (operator builder).
    TDSEZAssembler* assembler_;

    public:
        /// Construct the manager, binding it to a core and assembler.
        TDSEZManager(TDSEZCore* core, TDSEZAssembler* assembler);

        /// Destructor — destroys all owned PETSc objects and closes HDF5.
        ~TDSEZManager();

        /// @cond
        // disable copy to avoid double Mat/Vec destroy
        TDSEZManager(const TDSEZManager&) = delete;
        TDSEZManager& operator=(const TDSEZManager&) = delete;
        TDSEZManager(TDSEZManager&&) = delete;
        TDSEZManager& operator=(TDSEZManager&&) = delete;
        /// @endcond


    // ── Operator / state objects (private: owned by the Manager lifecycle) ──
    // Callers must NOT reach in and null these (it would break the propagator);
    // use the getters below. Naming: m_ prefix = private member.
    /// @name Operator / state matrices
    /// @brief All PETSc Mat/Vec objects owned by the Manager lifecycle.
    /// @details Use the getters (M(), H(), K(), etc.) rather than touching
    /// these directly.  The propagator and diagnostics read them through
    /// the getters.
    /// @{
    Mat m_M  = PETSC_NULLPTR;  ///< mass matrix
    Mat m_H  = PETSC_NULLPTR;  ///< Hamiltonian (H = -∇²/2m + V + CAP, post-CAP-apply)
    Mat m_K  = PETSC_NULLPTR;  ///< kinetic-energy matrix
    Mat m_V  = PETSC_NULLPTR;  ///< potential-energy matrix
    Mat m_Dx = PETSC_NULLPTR, m_Dy = PETSC_NULLPTR, m_Dz = PETSC_NULLPTR; ///< dipole (position) operators
    Mat m_VelX = PETSC_NULLPTR, m_VelY = PETSC_NULLPTR, m_VelZ = PETSC_NULLPTR; ///< velocity-gauge operators
    Mat m_dVdx = PETSC_NULLPTR, m_dVdy = PETSC_NULLPTR, m_dVdz = PETSC_NULLPTR; ///< -∇V components
    Mat m_Md = PETSC_NULLPTR;  ///< mass-derivative operator (energy/dipole diagnostics)
    Mat m_CAP = PETSC_NULLPTR; ///< complex absorbing potential (consumed by propagator)
    Mat m_Ht  = PETSC_NULLPTR; ///< transient H copy used inside the TS step
    Mat m_Lz  = PETSC_NULLPTR; ///< angular-momentum operator (2D degeneracy split; NULL in 3D)
    Vec m_psi0 = PETSC_NULLPTR; ///< initial state |ψ(0)⟩ (for autocorrelation)
    /// @}

    /// @name Read-only getters for operator/state objects
    /// @{
    Mat       M()     const { return m_M; }     ///< mass matrix
    Mat       H()     const { return m_H; }     ///< Hamiltonian
    Mat       K()     const { return m_K; }     ///< kinetic energy
    Mat       V()     const { return m_V; }     ///< potential energy
    Mat       Dx()    const { return m_Dx; }    ///< x-dipole
    Mat       Dy()    const { return m_Dy; }    ///< y-dipole
    Mat       Dz()    const { return m_Dz; }    ///< z-dipole
    Mat       VelX()  const { return m_VelX; }  ///< x-velocity operator
    Mat       VelY()  const { return m_VelY; }  ///< y-velocity operator
    Mat       VelZ()  const { return m_VelZ; }  ///< z-velocity operator
    Mat       dVdx()  const { return m_dVdx; }  ///< -dV/dx
    Mat       dVdy()  const { return m_dVdy; }  ///< -dV/dy
    Mat       dVdz()  const { return m_dVdz; }  ///< -dV/dz
    Mat       Md()    const { return m_Md; }    ///< mass-derivative operator
    Mat       CAP()   const { return m_CAP; }   ///< complex absorbing potential
    Mat       Ht()    const { return m_Ht; }    ///< transient H copy inside TS step
    Mat       Lz()    const { return m_Lz; }    ///< angular momentum (2D only)
    Vec       psi0()  const { return m_psi0; }  ///< initial state |ψ(0)⟩
    /// @}

    /// @name Setters (used only during setup / CAP consumption)
    /// @{
    void setCAP(Mat c)  { m_CAP  = c; }  ///< Set the CAP matrix.
    void setPsi0(Vec p) { m_psi0 = p; }  ///< Set the initial state vector.
    void setHt(Mat h)   { m_Ht   = h; }  ///< Set the transient H copy.
    /// @}
    /// @brief Active polarization flags (set by PolarizationSelector()).
    PetscBool hasX = PETSC_FALSE, hasY = PETSC_FALSE, hasZ = PETSC_FALSE;

    /// @name Diagnostic buffers (allocated on rank 0 only)
    /// @{
    PetscScalar *dipBuffer = PETSC_NULLPTR;    ///< dipole moment history
    PetscScalar *popBuffer = PETSC_NULLPTR;   ///< population history
    PetscScalar *energyBuffer = PETSC_NULLPTR; ///< energy history
    PetscScalar *currBuffer = PETSC_NULLPTR;   ///< current history
    PetscScalar *acBuffer = PETSC_NULLPTR;     ///< autocorrelation history
    /// @}

    /// IFunction scratch vectors (length numIFuncVec).
    Vec *IFuncVec = PETSC_NULLPTR;
    /// Accumulated Berry phase (for circularly polarized fields).
    PetscReal BerryPhase = 0.0;

    /// @name P-vector storage for velocity-gauge propagation
    /// @{
    std::vector<PetscScalar> vMat_x, vMat_y, vMat_z;
    /// @}

    /// @name Access vectors and state tracking
    /// @{
    std::vector<Vec>            boundstates;    ///< converged bound states
    std::vector<Vec>            dipoleVecs;     ///< dipole operator action vectors
    std::vector<PetscReal>      population;     ///< per-state populations
    std::vector<PetscReal>      energies;        ///< eigen-energies
    std::vector<PetscScalar>    popDots;        ///< current population dot products
    std::vector<PetscScalar>    popDots_prev;   ///< previous-step dot products
    /// @}

    /// Temporary scratch vector.
    Vec *tmpVec = PETSC_NULLPTR;
    /// Maximum number of dipole/population vectors retained.
    PetscInt numVecs = 20;
    /// Number of IFunction scratch vectors.
    PetscInt numIFuncVec = 4;
    /// Allocation and recording counters.
    PetscInt maxAllocatedSteps = 0, recordedSteps = 0, dipWidth = 0, popWidth = 0, energyWidth = 0;

    /// @name Output control
    /// @{
    PetscInt stride = 2000;                     ///< diagnostic write stride
    static constexpr PetscInt FLUSH_INTERVAL = 5000; ///< HDF5 flush interval (steps)
    PetscInt lastWrittenSteps = 0;              ///< last step flushed to HDF5
    std::string inputFile;                     ///< input file path
    std::string outputFilename;                 ///< HDF5 output filename
    /// @}

    /// @name Persistent HDF5 handle
    /// @details Kept open across flushes for efficiency. -1 means not open.
    /// @{
    hid_t       h5File           = -1;
    bool        h5Compress       = false;
    int         h5CompressLevel  = 6;
    /// @}

    /// @name Physical constants (cached from parser)
    /// @{
    PetscReal mass = 0.0, invMass = 0.0, hbar = 0.0, charge = 0.0, chargeOvMass = 0.0;
    PetscInt  NPOP = 1, rank = 0;
    /// @}

    /// Get the owning TDSEZCore.
    TDSEZCore* getCore() { return core_; }
    /// Get the TDSEZAssembler.
    TDSEZAssembler* getAssembler() { return assembler_; }

    /// @name PETSc viewers
    /// @{
    PetscViewer TIMEViewer = PETSC_NULLPTR;  ///< time-series viewer
    PetscViewer WFSViewer = PETSC_NULLPTR;  ///< wavefunction snapshot viewer
    PetscViewer ACViewer  = PETSC_NULLPTR;   ///< autocorrelation viewer
    /// @}

    /// @brief Function pointers for the IFunction and IJacobian callbacks
    ///        (selected by PolarizationSelector()).
    /// @{
    PetscErrorCode (*TDSEZIFunctionPtr)(TS,PetscReal,Vec,Vec,Vec,void*) = PETSC_NULLPTR;
    PetscErrorCode (*TDSEZIJacobianPtr)(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*) = PETSC_NULLPTR;
    /// @}

    /// @brief Write the accumulated diagnostic buffers to an HDF5 file.
    /// @param filename Output HDF5 filename.
    /// @return PetscErrorCode (0 on success).
    PetscErrorCode WriteHDF5(const std::string& filename);
    /// @brief Close the persistent HDF5 file handle.
    /// @return PetscErrorCode (0 on success).
    PetscErrorCode CloseHDF5();
    /// @brief Select active polarization axes and wire IFunction/IJacobian pointers.
    /// @return PetscErrorCode (0 on success).
    PetscErrorCode PolarizationSelector();

    // ---------------- t-SURFF (time-dependent surface flux) ----------------
    /// @name t-SURFF surface-flux machinery
    /// @details Faces indexed 0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+ (i.e.
    /// (axis,side) = (f/2, f%2)).  Only faces consistent with
    /// hasX/hasY/hasZ are built/used.
    /// @{

    /// Boundary value operators per face (MatMult: psi -> boundary values).
    Mat  Bval_face[6] = {PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR,
                          PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR};
    /// Boundary derivative operators per face (MatMult: psi -> boundary normals).
    Mat  Bder_face[6] = {PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR,
                          PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR};
    /// Boundary value vectors (rank 0 only, MatMult output).
    Vec  faceValVec[6] = {PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR,
                           PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR};
    /// Boundary derivative vectors (rank 0 only, MatMult output).
    Vec  faceDerVec[6] = {PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR,
                           PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR};

    /// Number of quadrature points per face (rank 0 only).
    PetscInt faceNquad[6] = {0,0,0,0,0,0};
    /// Sequential copy of psi on rank 0 for SELF face MatMult.
    Vec       psiZero       = PETSC_NULLPTR;
    /// Scatter from distributed psi to the rank-0 sequential copy.
    VecScatter surffScatter = PETSC_NULLPTR;
    /// Quadrature-point coordinates per face (rank 0 only).
    std::vector<PetscReal> faceX[6], faceY[6], faceZ[6];
    /// Quadrature weights per face (includes boundary Jacobian, rank 0 only).
    std::vector<PetscReal> faceW[6];
    /// Face axis index: 0=x, 1=y, 2=z.
    PetscInt   faceAxis[6]     = {0,0,1,1,2,2};
    /// Outward-normal sign: -1 for lower face, +1 for upper face.
    PetscReal  faceSign[6]     = {-1,+1,-1,+1,-1,+1};
    /// Per-quad weight mask: 1 if |r_q|>SurffRadius (active shell), else 0.
    std::vector<PetscReal> faceMask[6];

    /// @name Running vector potential A(t) = -∫₀^t E(t')dt' (trapezoidal)
    /// @{
    PetscReal Ax = 0.0, Ay = 0.0, Az = 0.0;
    PetscReal surffTprev = 0.0, surffExPrev = 0.0, surffEyPrev = 0.0, surffEzPrev = 0.0;
    PetscBool surffInitialized = PETSC_FALSE;
    /// @}

    /// @name Momentum grid + running accumulators (rank 0 only, size Nk³)
    /// @{
    PetscInt   surffNk = 0;                 ///< number of k-grid points per axis
    PetscReal  surffKmax = 0.0;             ///< k-grid cutoff
    PetscReal  surffMass = 1.0;             ///< sampled particle mass (constant-mass approx)
    std::vector<PetscReal>   kAxis;         ///< Nk values, -Kmax..+Kmax
    std::vector<PetscReal>   surffPhi;      ///< running Φ(k,t), size Nk³
    std::vector<PetscScalar> surffAmp;      ///< running b(k,t) accumulator, size Nk³
    /// @}

    /// @name Per-face diagnostic PES (allocated when TSURFF_FACE_SPLIT env set)
    /// @{
    std::vector<PetscScalar> surffAmpFace[6];
    PetscBool surffFaceSplit = PETSC_FALSE;
    /// @}

    /// @name Per-fold separable buffers (rank 0)
    /// @details For an active face the boundary flux factorizes as
    /// J_f(k) = e^{-i k_d X_d} [ A_f(k_u) + i·sign·k_d · B_f(k_u) ]
    /// where (k_u, k_d) split the momentum along tangent/normal axes.
    /// This makes the per-fold cost O(Nk^{d-1}·Nq + Nk^d) instead of
    /// O(Nk^d·Nq) — an exact factor-Nk speedup with no approximation.
    /// @{
    PetscInt   surffTangentDim = 0;        ///< dim-1
    std::vector<PetscScalar> surffAf[6];   ///< size (Nk)^(d-1) per active face
    std::vector<PetscScalar> surffBf[6];   ///< size (Nk)^(d-1) per active face
    /// @}

    /// @brief Set up the t-SURFF momentum grid, face operators, and scatters.
    /// @return PetscErrorCode (0 on success).
    PetscErrorCode SetupSurff();
    /// @brief Build the per-face boundary value/derivative operators.
    /// @return PetscErrorCode (0 on success).
    PetscErrorCode BuildFaceOperators();
    /// @brief Accumulate the surface-flux contribution at one time step.
    /// @param psi Current wavefunction.
    /// @param t Current time.
    /// @param dt Time-step size.
    /// @param step Current step index.
    /// @return PetscErrorCode (0 on success).
    PetscErrorCode AccumulateSurff(Vec psi, PetscReal t, PetscReal dt, PetscInt step);
    /// @brief Finalize the t-SURFF reconstruction and write the PES to disk.
    /// @return PetscErrorCode (0 on success).
    PetscErrorCode FinalizeSurff();
    /// @}  // end t-SURFF group
};


#endif // TDSEZ_MANAGER_HPP
