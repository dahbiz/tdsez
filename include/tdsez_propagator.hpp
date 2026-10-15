#ifndef TDSEZ_PROPAGATOR_HPP
#define TDSEZ_PROPAGATOR_HPP

#include <petscts.h>
#include <petscmat.h>

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
        for (Vec &v : guessDelta_) if (v) VecDestroy(&v);
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
    /// psi <- R22(-i h M^{-1} H_E) psi, H_E = H + sum_a E_a D_a, as two
    /// unitary Cayley-type stages with complex shifts.
    PetscErrorCode Pade22Step(PetscReal h, const PetscScalar E[3], Vec psi, int slot = 0);
    /// Set A = M + g (H + sum_a E_a D_a).
    PetscErrorCode BuildStageMatrix(Mat A, PetscScalar g, const PetscScalar E[3]);
    /// Cache the field-independent parts B_k = M + g_k(h) H for step length h.
    PetscErrorCode UpdateStageBase(PetscReal h);
    Mat Base_[2]     = {PETSC_NULLPTR, PETSC_NULLPTR};  ///< cached M + g_k H (field-free)
    PetscReal baseH_ = -1.0;                            ///< step length of Base_
    /// Initial-guess extrapolation: stage output minus input from the previous
    /// step, per stage and per Pade call of a step (slot 0..3).
    Vec  guessDelta_[4] = {PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR};
    PetscBool guessValid_[4] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};
    PetscBool extrapolateGuess_ = PETSC_TRUE;
    Mat Astage_[2]   = {PETSC_NULLPTR, PETSC_NULLPTR};  ///< stage matrices M + g_k H_E
    KSP kspStage_[2] = {PETSC_NULLPTR, PETSC_NULLPTR};  ///< one solver per stage shift
    Vec work_[3]     = {PETSC_NULLPTR, PETSC_NULLPTR, PETSC_NULLPTR};
    PetscInt nStageSolves_ = 0, nStageIts_ = 0;        ///< solver statistics
    TDSEZPCRefresh stageRefresh_[2];                    ///< preconditioner refresh per stage solver
    /// @}
};


#endif // TDSEZ_PROPAGATOR_HPP
