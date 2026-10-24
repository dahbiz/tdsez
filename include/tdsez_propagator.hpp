#ifndef TDSEZ_PROPAGATOR_HPP
#define TDSEZ_PROPAGATOR_HPP

#include <petscts.h>
#include <petscmat.h>
#include <vector>

class TDSEZCore;
class TDSEZManager;

/// Preconditioner refresh: the preconditioner is built from the operator of
/// one step and reused; when a solve needs more than twice the iterations of
/// the first solve after the last build (plus 5), it is rebuilt from the
/// current operator. With strong fields s E(t).r changes the operator so much
/// that a preconditioner built at t = 0 stops working. The solve tolerance is
/// not affected.
struct TDSEZPCRefresh {
    PetscInt base = -1, rebuilds = 0;
    /// Record a solve; true when the preconditioner should be rebuilt.
    bool Stale(PetscInt its)
    {
        // -tdsez_pc_refresh_its N: also rebuild whenever a solve takes more than N iterations
        static PetscInt cap = -2;
        if (cap == -2) { cap = -1; PetscOptionsGetInt(NULL, NULL, "-tdsez_pc_refresh_its", &cap, NULL); }
        if (cap >= 0 && its > cap) { base = -1; ++rebuilds; return true; }
        if (base < 0) { base = its; return false; }
        if (its <= 2 * base + 5) return false;
        base = -1; ++rebuilds;
        return true;
    }
};

/**
 * @brief Time-stepping driver wrapping the PETSc TS integrator.
 * @details Owns the TS context and the Jacobian matrix.  Evolve() sets up
 * the IFunction/IJacobian callbacks (selected by the Manager's
 * PolarizationSelector), drives the time loop from t=0 to FinalTime,
 * and calls the HDF5 monitor at each stride.
 */
class TDSEZPropagator
{

public:

    /// Construct the propagator, binding it to a TDSEZManager.
    TDSEZPropagator(TDSEZManager &manager);

    /// Destructor — destroys the Jacobian matrix and TS context.
    ~TDSEZPropagator() {
        PetscFunctionBeginUser;
        // cleanup
        if (J_) MatDestroy(&J_);
        if (ts_) TSDestroy(&ts_);
        if (kspStage_[0]) KSPDestroy(&kspStage_[0]);
        if (kspStage_[1]) KSPDestroy(&kspStage_[1]);
        if (Astage_[0]) MatDestroy(&Astage_[0]);
        if (Astage_[1]) MatDestroy(&Astage_[1]);
        if (Base_[0]) MatDestroy(&Base_[0]);
        if (Base_[1]) MatDestroy(&Base_[1]);
        for (Vec &v : work_) if (v) VecDestroy(&v);
        for (auto &h : guessDelta_) for (Vec &v : h) if (v) VecDestroy(&v);
        PetscFunctionReturnVoid();
    }

    /// @brief Run the time propagation from t=0 to FinalTime.
    /// @return PetscErrorCode (0 on success).
    PetscErrorCode Evolve();

private:
    /// Reference to the owning TDSEZCore (for IGA/grid parameters).
    TDSEZCore &tdse_;
    /// Reference to the TDSEZManager (for operators, buffers, diagnostics).
    TDSEZManager  &manager_;

    TS  ts_  = PETSC_NULLPTR;  ///< PETSc time-stepping context
    Mat J_   = PETSC_NULLPTR;  ///< Jacobian matrix for the TS step

    /// @name Rational (Pade(2,2)) propagators: Propagator = pade22 | magnus4
    /// @{
    /// Time loop for the rational propagators (Crank-Nicolson uses PETSc TS).
    PetscErrorCode EvolveRational();
    /// Restart files: set up the checkpointer and its hooks for the guess history.
    PetscErrorCode SetupRestartFiles(Vec psi);
    /// Restart = 1: load the checkpoint into psi and continue the output files from it.
    PetscErrorCode Restart(Vec psi, PetscInt *step, PetscReal *t);
    /// psi <- R22(-i h M^{-1} H_E) psi, H_E = H + sum_i c_i T_i, as two
    /// unitary Cayley-type stages with complex shifts.
    /// @p c holds the coefficients of terms_ (H_E = H + sum_i c_i terms_[i]).
    PetscErrorCode Pade22Step(PetscReal h, const std::vector<PetscScalar> &c, Vec psi, int slot = 0);
    /// Set A = M + g (H + sum_i c_i terms_[i]).
    PetscErrorCode BuildStageMatrix(Mat A, PetscScalar g, const std::vector<PetscScalar> &c);
    /// Field terms of H(t): coupled dipoles, then FieldTerm matrices (TDSEZManager::TermMatrices).
    std::vector<Mat> terms_;
    /// Cache the field-independent parts B_k = M + g_k(h) H for step length h.
    PetscErrorCode UpdateStageBase(PetscReal h);
    Mat Base_[2]     = {PETSC_NULLPTR, PETSC_NULLPTR};  ///< cached M + g_k H (field-free)
    int fusedUpdate_ = -1;   ///< A_k = B_k + sum c_a D_a in one pass over the values (-1: not checked yet)
    int leanStage_ = -1;                                ///< -tdsez_mat_free: stages formed from M, H, D (no Base_)
    PetscReal baseH_ = -1.0;                            ///< step length of Base_
    /// Initial-guess extrapolation: stage output minus input from the previous
    /// step, per stage and per Pade call of a step (slot 0..3).
    Vec  guessDelta_[4][5] = {};      // per stage: increments of the last steps, newest first
    PetscInt guessCount_[4] = {0, 0, 0, 0};
    PetscInt guessOrder_ = 3;
    PetscBool extrapolateGuess_ = PETSC_TRUE;
    Mat Astage_[2]   = {PETSC_NULLPTR, PETSC_NULLPTR};  ///< stage matrices M + g_k H_E
    KSP kspStage_[2] = {PETSC_NULLPTR, PETSC_NULLPTR};  ///< one solver per stage shift
    Vec work_[3]     = {PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR};
    PetscInt nStageSolves_ = 0, nStageIts_ = 0;        ///< solver statistics
    TDSEZPCRefresh stageRefresh_[2];                    ///< preconditioner refresh per stage solver
    /// @}
};


#endif // TDSEZ_PROPAGATOR_HPP
