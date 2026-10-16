#ifndef TDSEZ_MANAGER_HPP
#define TDSEZ_MANAGER_HPP

#include <petscviewerhdf5.h>
#include <petscts.h>
#include <hdf5.h>
#include <string>
#include <vector>
#include "tdsez_tensorops.hpp"
#include "tdsez_quadobs.hpp"

class TDSEZCore;
class TDSEZAssembler;
class TDSEZTSurff;

/**
 * @brief Orchestrates time propagation, diagnostics and HDF5 output.
 * @details Owns the operator matrices (H, K, V, dipoles, velocity, CAP, Lz),
 * diagnostic buffers and the HDF5 file handle.  The Manager is constructed with pointers to a TDSEZCore
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
    Mat m_Lz  = PETSC_NULLPTR; ///< angular-momentum operator (2D degeneracy split; NULL in 3D)
    Vec m_psi0 = PETSC_NULLPTR; ///< initial state |ψ(0)⟩ (for autocorrelation)
    std::vector<Mat> m_G;       ///< FieldTerm matrices <B| w_k |B> (aliased from the assembler)
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
    Mat       Lz()    const { return m_Lz; }    ///< angular momentum (2D only)
    Vec       psi0()  const { return m_psi0; }  ///< initial state |ψ(0)⟩
    const std::vector<Mat> &G() const { return m_G; }   ///< FieldTerm matrices
    /// @}

    /// @brief Field terms of H(t) = H + sum_i c_i(t) T_i: the dipoles of the
    ///        coupled axes (c = s E_a(t)) followed by the FieldTerm matrices
    ///        (c = s f_k(t)). TermMatrices() is fixed; TermCoefficients() fills c.
    /// @{
    std::vector<Mat> TermMatrices() const;
    void TermCoefficients(PetscReal t, std::vector<PetscScalar> &c) const;
    /// @}
    /// @brief y_k = G_k psi and e_k = psi^H y_k for every FieldTerm, cached
    ///        for this psi (state id) so the energy and the acceleration of one
    ///        monitor call share the products.
    PetscErrorCode FieldTermProducts(Vec psi);
    /// Sum_k s f_k(t) <psi|G_k|psi> (the FieldTerm part of the interaction energy).
    PetscErrorCode FieldTermEnergy(PetscReal t, Vec psi, PetscScalar *e);
    std::vector<Vec>         termPsi_;          ///< G_k psi
    std::vector<PetscScalar> termExp_;          ///< psi^H G_k psi
    PetscObjectId            termPsiId_ = 0;    ///< psi of the cached products
    PetscObjectState         termPsiState_ = -1;

    /// @name Setters (used only during setup / CAP consumption)
    /// @{
    void setCAP(Mat c)  { m_CAP  = c; }  ///< Set the CAP matrix.
    void setPsi0(Vec p) { m_psi0 = p; }  ///< Set the initial state vector.
    /// @}
    /// t-SURFF observer sampled by the monitors (not owned; null unless TSurff = 1).
    TDSEZTSurff *tsurff = PETSC_NULLPTR;
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
    PetscBool      monitorChecked_ = PETSC_FALSE;  ///< MonitorProducts shortcuts verified
    PetscBool      hFromKV_        = PETSC_FALSE;  ///< H psi = K psi + V psi holds
    PetscScalar    invMassConst_   = 0.0;          ///< Md = invMassConst_ * M (0: not constant)
    PetscLogDouble lastFlushTime_  = 0.0;          ///< wall time of the last HDF5 write
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

    /// True when the dipole acceleration is taken from the discrete
    /// equation of motion rather than the force operator (AccelerationMethod).
    PetscBool UseDiscreteAcceleration() const;
    /// Dipole acceleration a = q d<v>/dt from the Galerkin equation of motion
    /// i M c' = (H + E.D) c:  a_alpha = -2 q Im( w^H Vel_alpha psi ),
    /// w = M^{-1} (H + E.D) psi.  Needs no derivatives of m or V, so it is
    /// valid across sharp material interfaces. Inactive axes return 0.
    /// Hpsi, if not NULL, must hold H psi (as computed by the monitor) and saves a MatMult.
    PetscErrorCode DiscreteAcceleration(PetscReal t, Vec psi, Vec Hpsi, PetscScalar acc[3]);

    /// @brief Monitor products tmpVec[0..4] = (M, H, Md, K, V) psi; Md/K/V only
    ///        when @p needKVMd. Uses H psi = K psi + V psi and
    ///        Md psi = M psi / m when both are verified to rounding for this
    ///        run (no CAP; constant mass), which saves up to two sparse products
    ///        per step without changing any output.
    PetscErrorCode MonitorProducts(Vec psi, PetscBool needKVMd);
    /// Factorised separable operators (2D/3D), validated at the first monitor call.
    TDSEZTensorOps *tensorOps_ = PETSC_NULLPTR;
    /// y = op x: factorised when validated, MatMult(A, x, y) otherwise.
    PetscErrorCode ApplyOp(TDSEZTensorOps::Op op, Mat A, Vec x, Vec y);
    /// Whether op can be applied (assembled matrix present or factorised form validated).
    bool HasOp(TDSEZTensorOps::Op op, Mat A) const { return A || (tensorOps_ && tensorOps_->Usable(op)); }
    /// Build and validate the factorised operators on psi; free the matrices
    /// only the monitors used (velocities, uncoupled dipoles, K).
    PetscErrorCode SetupTensorOps(Vec psi);
    /// -tdsez_mat_free: build the factorised operators on the layout of
    /// @p templ without assembled matrices to compare with (the assembler did
    /// not build them). Called before the bound-state diagnostics.
    PetscErrorCode SetupMatrixFree(Vec templ);
    bool matFree_ = false;                    ///< -tdsez_mat_free in effect
    /// Expectation values from psi at the quadrature points (2D/3D), each
    /// checked against its matrix at the first monitor call; the matrices they
    /// replace are freed (V, force dV_a, and for a varying mass K, Md, Vel_a).
    TDSEZQuadObs *quadObs_ = PETSC_NULLPTR;
    TDSEZQuadObs::Values qv_;                 ///< values of the current monitor call
    bool qPot_ = false, qKin_ = false, qInvM_ = false, qCap_ = false, qH_ = false;
    bool qForce_[3] = {false, false, false}, qVel_[3] = {false, false, false};
    PetscErrorCode SetupQuadObs(Vec psi);
    /// Replace res[1..4] (<H>, <1/m>, <K>, <V>) by the quadrature values in use.
    void OverrideExpectations(PetscScalar res[]) const;
    bool QuadForce(int a) const { return qForce_[a]; }
    bool QuadVel(int a) const { return qVel_[a]; }
    PetscScalar QuadForceValue(int a) const { return qv_.force[a]; }
    PetscScalar QuadVelValue(int a) const { return qv_.vel[a]; }
    /// @brief Whether the time-series buffers should be written now: always at
    ///        multiples of FLUSH_INTERVAL; at multiples of OutputStrideTS only if
    ///        at least kMinFlushSeconds have passed since the last write. Every
    ///        recorded row is still written (the remainder at the end of the run).
    bool FlushDue();
    static constexpr PetscLogDouble kMinFlushSeconds = 2.0;
    /// Direct solver for the overlap matrix M (built on first use).
    KSP kspMass = PETSC_NULLPTR;
    /// Work vectors for DiscreteAcceleration (built on first use).
    Vec accelWork[3] = {PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR};
};


#endif // TDSEZ_MANAGER_HPP
