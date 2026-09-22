#ifndef TDSEZ_PROPAGATOR_HPP
#define TDSEZ_PROPAGATOR_HPP

#include <petscts.h>
#include <petscmat.h>

class TDSEZCore;
class TDSEZManager;

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
};


#endif // TDSEZ_PROPAGATOR_HPP
