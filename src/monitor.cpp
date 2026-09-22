#include "tdsez_internal.hpp"

/**
 * @file monitor.cpp
 * @brief TS monitor callbacks computing per-step observables (norm, energy,
 *        dipole, acceleration, populations, currents, Berry phase,
 *        autocorrelation) and streaming them to HDF5, plus the IFunction /
 *        IJacobian callbacks for each polarization combination.
 */

/// IFunction for x-polarized propagation: F = i M psidot - H psi - Ex Dx psi.
/**
 * @param[in]  ts      Time-stepper context (unused).
 * @param[in]  t       Current simulation time.
 * @param[in]  psi     State vector (unused directly; used via MatMult).
 * @param[in]  psidot  Time derivative of state (unused directly).
 * @param[out] F       Residual vector F = i M psidot - H psi - Ex Dx psi.
 * @param[in]  ctx     TDSEZManager providing H, Dx, M, IFuncVec scratch.
 * @return PETSc error code.
 */
PetscErrorCode TDSEZIFunctionPolX(TS ts, PetscReal t, Vec psi, Vec psidot, Vec F, void *ctx)
{
  (void)ts; (void)psi; (void)psidot;
  auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    PetscScalar Ex = TDSEZParser::Ex(t);

    MatMult(TDSEZ->H(),  psi,    TDSEZ->IFuncVec[0]); // vec1 = H * psi
    MatMult(TDSEZ->Dx(), psi,    TDSEZ->IFuncVec[1]); // vec2 = Dx * psi
    MatMult(TDSEZ->M(),  psidot, F);                // F    = M * psidot
    
    // We want: F = (-1.0)*vec1 + (-Ex)*vec2 + (i)*F
    VecAXPBYPCZ(F, -1.0, -Ex, PETSC_i, TDSEZ->IFuncVec[0], TDSEZ->IFuncVec[1]);

    return 0;
}


/// IFunction for y-polarized propagation: F = i M psidot - H psi - Ey Dy psi.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[out] F Residual = i M psidot - H psi - Ey Dy psi.
 *  @param[in] ctx TDSEZManager. @return PETSc error code. */
PetscErrorCode TDSEZIFunctionPolY(TS ts, PetscReal t, Vec psi, Vec psidot, Vec F, void *ctx)
{
  (void)ts; (void)psi; (void)psidot;
  auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    PetscScalar Ey = TDSEZParser::Ey(t);
    MatMult(TDSEZ->H(),  psi,    TDSEZ->IFuncVec[0]); // vec1 = H * psi
    MatMult(TDSEZ->Dy(), psi,    TDSEZ->IFuncVec[1]); // vec2 = Dy * psi
    MatMult(TDSEZ->M(),  psidot, F);                // F    = M * psidot
    // We want: F = (-1.0)*vec1 + (-Ey)*vec2 + (i)*F
    VecAXPBYPCZ(F, -1.0, -Ey, PETSC_i, TDSEZ->IFuncVec[0], TDSEZ->IFuncVec[1]);

    return 0;
}


/// IFunction for z-polarized propagation: F = i M psidot - H psi - Ez Dz psi.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[out] F Residual = i M psidot - H psi - Ez Dz psi.
 *  @param[in] ctx TDSEZManager. @return PETSc error code. */
PetscErrorCode TDSEZIFunctionPolZ(TS ts, PetscReal t, Vec psi, Vec psidot, Vec F, void *ctx)
{
  (void)ts; (void)psi; (void)psidot;
  auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    PetscScalar Ez = TDSEZParser::Ez(t);
    MatMult(TDSEZ->H(),  psi,    TDSEZ->IFuncVec[0]); // vec1 = H * psi
    MatMult(TDSEZ->Dz(), psi,    TDSEZ->IFuncVec[1]); // vec2 = Dz * psi
    MatMult(TDSEZ->M(),  psidot, F);                // F    = M * psidot
    // We want: F = (-1.0)*vec1 + (-Ez)*vec2 + (i)*F
    VecAXPBYPCZ(F, -1.0, -Ez, PETSC_i, TDSEZ->IFuncVec[0], TDSEZ->IFuncVec[1]);

    return 0;
}



/// IFunction for xy-polarized propagation: F = i M psidot - H psi - Ex Dx - Ey Dy.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[out] F Residual. @param[in] ctx TDSEZManager. @return PETSc error code. */
PetscErrorCode TDSEZIFunctionPolXY(TS ts, PetscReal t, Vec psi, Vec psidot, Vec F, void *ctx)
{
  (void)ts; (void)psi; (void)psidot;
  auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    PetscScalar Ex = TDSEZParser::Ex(t);
    PetscScalar Ey = TDSEZParser::Ey(t);
    PetscFunctionBeginUser;
    // 1. MatMults (These stay the same)
    MatMult(TDSEZ->H(),  psi, TDSEZ->IFuncVec[0]); // H0 * psi
    MatMult(TDSEZ->Dx(), psi, TDSEZ->IFuncVec[1]); // Dx * psi
    MatMult(TDSEZ->Dy(), psi, TDSEZ->IFuncVec[2]); // Dy * psi
    // F has the inertial term: F = i * M * psidot
    MatMult(TDSEZ->M(), psidot, F);
    VecScale(F, PETSC_i);
    // F = F + (-1.0)*H0 + (-Ex)*Dx + (-Ey)*Dy
    const PetscScalar coeffs[3] = {-1.0, -Ex, -Ey};
    VecMAXPY(F, 3, coeffs, TDSEZ->IFuncVec);
        PetscFunctionReturn(PETSC_SUCCESS);
}




/// IFunction for xz-polarized propagation: F = i M psidot - H psi - Ex Dx - Ez Dz.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[out] F Residual. @param[in] ctx TDSEZManager. @return PETSc error code. */
PetscErrorCode TDSEZIFunctionPolXZ(TS ts, PetscReal t, Vec psi, Vec psidot, Vec F, void *ctx)
{
  (void)ts; (void)psi; (void)psidot;
  auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ex = TDSEZParser::Ex(t);
    const PetscScalar Ez = TDSEZParser::Ez(t);
    PetscFunctionBeginUser;
    // We use slots 0, 1, and 2 for H, Dx, and Dz
    PetscCall(MatMult(TDSEZ->H(),  psi, TDSEZ->IFuncVec[0])); // H0 * psi
    PetscCall(MatMult(TDSEZ->Dx(), psi, TDSEZ->IFuncVec[1])); // Dx * psi
    PetscCall(MatMult(TDSEZ->Dz(), psi, TDSEZ->IFuncVec[2])); // Dz * psi
    // 2. Inertial term: F = i * M * psidot
    PetscCall(MatMult(TDSEZ->M(), psidot, F));                // F = M * psidot
    PetscCall(VecScale(F, PETSC_i));                        // F = i * M * psidot
    // F = F + (-1.0)*H0 + (-Ex)*Dx + (-Ez)*Dz
    const PetscScalar coeffs[3] = {-1.0, -Ex, -Ez};        //
    PetscCall(VecMAXPY(F, 3, coeffs, TDSEZ->IFuncVec));    //
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// IFunction for yz-polarized propagation: F = i M psidot - H psi - Ey Dy - Ez Dz.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[out] F Residual. @param[in] ctx TDSEZManager. @return PETSc error code. */
PetscErrorCode TDSEZIFunctionPolYZ(TS ts, PetscReal t, Vec psi, Vec psidot, Vec F, void *ctx)
{
  (void)ts; (void)psi; (void)psidot;
  auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ey = TDSEZParser::Ey(t);
    const PetscScalar Ez = TDSEZParser::Ez(t);
    PetscFunctionBeginUser;
    // Mapping: slot 0 = H, slot 1 = Dy, slot 2 = Dz
    PetscCall(MatMult(TDSEZ->H(),  psi, TDSEZ->IFuncVec[0])); // H0 * psi
    PetscCall(MatMult(TDSEZ->Dy(), psi, TDSEZ->IFuncVec[1])); // Dy * psi
    PetscCall(MatMult(TDSEZ->Dz(), psi, TDSEZ->IFuncVec[2])); // Dz * psi
    // Inertial term: F = i * M * psidot
    PetscCall(MatMult(TDSEZ->M(), psidot, F));                // F = M * psidot
    PetscCall(VecScale(F, PETSC_i));                        // F = i * M * psidot
    // F = F + (-1.0)*H0 + (-Ey)*Dy + (-Ez)*Dz
    const PetscScalar coeffs[3] = {-1.0, -Ey, -Ez};
    PetscCall(VecMAXPY(F, 3, coeffs, TDSEZ->IFuncVec));

    PetscFunctionReturn(PETSC_SUCCESS);
}




/// IFunction for xyz-polarized propagation: F = i M psidot - H - Ex Dx - Ey Dy - Ez Dz.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[out] F Residual. @param[in] ctx TDSEZManager. @return PETSc error code. */
PetscErrorCode TDSEZIFunctionPolXYZ(TS ts, PetscReal t, Vec psi, Vec psidot, Vec F, void *ctx)
{
  (void)ts; (void)psi; (void)psidot;
  auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ex = TDSEZParser::Ex(t);
    const PetscScalar Ey = TDSEZParser::Ey(t);
    const PetscScalar Ez = TDSEZParser::Ez(t);
    PetscFunctionBeginUser;
    // Mapping: slot 0=H, 1=Dx, 2=Dy, 3=Dz
    PetscCall(MatMult(TDSEZ->H(),  psi, TDSEZ->IFuncVec[0])); 
    PetscCall(MatMult(TDSEZ->Dx(), psi, TDSEZ->IFuncVec[1])); 
    PetscCall(MatMult(TDSEZ->Dy(), psi, TDSEZ->IFuncVec[2])); 
    PetscCall(MatMult(TDSEZ->Dz(), psi, TDSEZ->IFuncVec[3]));
    // 2. Inertial term: F = i * M * psidot
    PetscCall(MatMult(TDSEZ->M(), psidot, F));
    PetscCall(VecScale(F, PETSC_i));
    // F = F + (-1.0)*H0 + (-Ex)*Dx + (-Ey)*Dy + (-Ez)*Dz
    const PetscScalar coeffs[4] = {-1.0, -Ex, -Ey, -Ez};
    PetscCall(VecMAXPY(F, 4, coeffs, TDSEZ->IFuncVec));
    PetscFunctionReturn(PETSC_SUCCESS);
}
/// IJacobian for x-polarized propagation: J = i a M - H - Ex Dx.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[in] a Shift parameter. @param[out] J Jacobian matrix.
 *  @param[in] P Preconditioner (unused). @param[in] ctx TDSEZManager.
 *  @return PETSc error code. */
PetscErrorCode TDSEZIJacobianPolX(TS ts, PetscReal t, Vec psi, Vec psidot, PetscReal a, Mat J, Mat P, void *ctx)
{
    (void)ts; (void)psi; (void)psidot; (void)P;
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ex = TDSEZParser::Ex(t);
    const PetscScalar ai = a * PETSC_i; 
    PetscFunctionBeginUser;
    // MatCopy is generally faster 
    PetscCall(MatCopy(TDSEZ->M(), J, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(J, ai)); 
    // J = ai*M - 1.0*H - Ex*Dx
    PetscCall(MatAXPY(J, -1.0, TDSEZ->H(),  SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ex,  TDSEZ->Dx(), SAME_NONZERO_PATTERN));

    PetscFunctionReturn(PETSC_SUCCESS);
}




/// IJacobian for y-polarized propagation: J = i a M - H - Ey Dy.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[in] a Shift parameter. @param[out] J Jacobian matrix.
 *  @param[in] P Preconditioner (unused). @param[in] ctx TDSEZManager.
 *  @return PETSc error code. */
PetscErrorCode TDSEZIJacobianPolY(TS ts, PetscReal t, Vec psi, Vec psidot, PetscReal a, Mat J, Mat P, void *ctx)
{
    (void)ts; (void)psi; (void)psidot; (void)P;
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ey = TDSEZParser::Ey(t);
    const PetscScalar ai = a * PETSC_i;
    PetscFunctionBeginUser;
    // We copy H and scale it by -1.0 immediately to get J = -H
    PetscCall(MatCopy(TDSEZ->H(), J, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(J, -1.0)); 
    // J = (-1.0*H) + (ai)*M + (-Ey)*Dy
    PetscCall(MatAXPY(J, ai,  TDSEZ->M(),  SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ey, TDSEZ->Dy(), SAME_NONZERO_PATTERN));


    PetscFunctionReturn(PETSC_SUCCESS);
}


/// IJacobian for z-polarized propagation: J = i a M - H - Ez Dz.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[in] a Shift parameter. @param[out] J Jacobian matrix.
 *  @param[in] P Preconditioner (unused). @param[in] ctx TDSEZManager.
 *  @return PETSc error code. */
PetscErrorCode TDSEZIJacobianPolZ(TS ts, PetscReal t, Vec psi, Vec psidot, PetscReal a, Mat J, Mat P, void *ctx)
{
    (void)ts; (void)psi; (void)psidot; (void)P;
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ez = TDSEZParser::Ez(t);
    const PetscScalar ai = a * PETSC_i;
    PetscFunctionBeginUser;
    // J = -H
    PetscCall(MatCopy(TDSEZ->H(), J, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(J, -1.0)); 
    // J = J + (ai)*M + (-Ez)*Dz -> results in: (ai)M - H - Ez*Dz
    PetscCall(MatAXPY(J, ai,  TDSEZ->M(),  SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ez, TDSEZ->Dz(), SAME_NONZERO_PATTERN));

    PetscFunctionReturn(PETSC_SUCCESS);
}




/// IJacobian for xy-polarized propagation: J = i a M - H - Ex Dx - Ey Dy.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[in] a Shift parameter. @param[out] J Jacobian matrix.
 *  @param[in] P Preconditioner (unused). @param[in] ctx TDSEZManager.
 *  @return PETSc error code. */
PetscErrorCode TDSEZIJacobianPolXY(TS ts, PetscReal t, Vec psi, Vec psidot, PetscReal a, Mat J, Mat P, void *ctx)
{
    (void)ts; (void)psi; (void)psidot; (void)P;
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ex = TDSEZParser::Ex(t);
    const PetscScalar Ey = TDSEZParser::Ey(t);
    const PetscScalar ai = a * PETSC_i; 
    PetscFunctionBeginUser;
    // Starting with H and negate it in one pass
    PetscCall(MatCopy(TDSEZ->H(), J, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(J, -1.0)); 
    // Fused Matrix Accumulation: (ai)M - H - Ex*Dx - Ey*Dy
    PetscCall(MatAXPY(J, ai,  TDSEZ->M(),  SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ex, TDSEZ->Dx(), SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ey, TDSEZ->Dy(), SAME_NONZERO_PATTERN));

    PetscFunctionReturn(PETSC_SUCCESS);
}


/// IJacobian for xz-polarized propagation: J = i a M - H - Ex Dx - Ez Dz.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[in] a Shift parameter. @param[out] J Jacobian matrix.
 *  @param[in] P Preconditioner (unused). @param[in] ctx TDSEZManager.
 *  @return PETSc error code. */
PetscErrorCode TDSEZIJacobianPolXZ(TS ts, PetscReal t, Vec psi, Vec psidot, PetscReal a, Mat J, Mat P, void *ctx)
{
    (void)ts; (void)psi; (void)psidot; (void)P;
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ex = TDSEZParser::Ex(t);
    const PetscScalar Ez = TDSEZParser::Ez(t);
    const PetscScalar ai = a * PETSC_i;
    PetscFunctionBeginUser;
    // Starting with H and negate it immediately
    PetscCall(MatCopy(TDSEZ->H(), J, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(J, -1.0)); 
    // Fused Matrix Accumulation: (ai)M - H - Ex*Dx - Ez*Dz
    PetscCall(MatAXPY(J, ai,  TDSEZ->M(),  SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ex, TDSEZ->Dx(), SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ez, TDSEZ->Dz(), SAME_NONZERO_PATTERN));

    PetscFunctionReturn(PETSC_SUCCESS);
}


/// IJacobian for yz-polarized propagation: J = i a M - H - Ey Dy - Ez Dz.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[in] a Shift parameter. @param[out] J Jacobian matrix.
 *  @param[in] P Preconditioner (unused). @param[in] ctx TDSEZManager.
 *  @return PETSc error code. */
PetscErrorCode TDSEZIJacobianPolYZ(TS ts, PetscReal t, Vec psi, Vec psidot, PetscReal a, Mat J, Mat P, void *ctx)
{
    (void)ts; (void)psi; (void)psidot; (void)P;
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ey = TDSEZParser::Ey(t);
    const PetscScalar Ez = TDSEZParser::Ez(t);
    const PetscScalar ai = a * PETSC_i;
    PetscFunctionBeginUser;
    // Starting with H and negate it immediately
    PetscCall(MatCopy(TDSEZ->H(), J, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(J, -1.0)); 
    // Fused Matrix Accumulation: (ai)M - H - Ey*Dy - Ez*Dz
    PetscCall(MatAXPY(J, ai,  TDSEZ->M(),  SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ey, TDSEZ->Dy(), SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ez, TDSEZ->Dz(), SAME_NONZERO_PATTERN));

    PetscFunctionReturn(PETSC_SUCCESS);
}



/// IJacobian for xyz-polarized propagation: J = i a M - H - Ex Dx - Ey Dy - Ez Dz.
/** @param[in] ts Time-stepper (unused). @param[in] t Current time.
 *  @param[in] psi State (unused). @param[in] psidot State derivative (unused).
 *  @param[in] a Shift parameter. @param[out] J Jacobian matrix.
 *  @param[in] P Preconditioner (unused). @param[in] ctx TDSEZManager.
 *  @return PETSc error code. */
PetscErrorCode TDSEZIJacobianPolXYZ(TS ts, PetscReal t, Vec psi, Vec psidot, PetscReal a, Mat J, Mat P, void *ctx)
{
    (void)ts; (void)psi; (void)psidot; (void)P;
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);
    const PetscScalar Ex = TDSEZParser::Ex(t);
    const PetscScalar Ey = TDSEZParser::Ey(t);
    const PetscScalar Ez = TDSEZParser::Ez(t);
    const PetscScalar ai = a * PETSC_i;
    PetscFunctionBeginUser;
    // Startig with H and negate it in one pass
    PetscCall(MatCopy(TDSEZ->H(), J, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(J, -1.0)); 
    // Fused Matrix Accumulation: (ai)M - H - Ex*Dx - Ey*Dy - Ez*Dz
    PetscCall(MatAXPY(J, ai,  TDSEZ->M(),  SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ex, TDSEZ->Dx(), SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ey, TDSEZ->Dy(), SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(J, -Ez, TDSEZ->Dz(), SAME_NONZERO_PATTERN));

    PetscFunctionReturn(PETSC_SUCCESS);
}





// 

/// 1D TS monitor: computes per-step observables and buffers them for HDF5.
/**
 * Computes norm, <H>, kinetic/potential/interaction/total energy, x-dipole,
 * acceleration, populations, and autocorrelation via a fused VecMDot
 * reduction, writes them into in-memory buffers, and flushes periodically to
 * HDF5. Also accumulates t-SURFF, writes wavefunction snapshots, and logs
 * progress.
 *
 * @param[in] ts   Time-stepper context.
 * @param[in] step Current time-step index.
 * @param[in] t    Current simulation time.
 * @param[in] psi  Current state vector.
 * @param[in] ctx  TDSEZManager providing operators and output buffers.
 * @return PETSc error code.
 */
PetscErrorCode TDSEZMonitorHDF5_1D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx)
{
    PetscFunctionBeginUser;
    (void)ts; // unused callback parameter
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);

    if (TDSEZParser::out_tsurff) PetscCall(TDSEZ->AccumulateSurff(psi, t, TDSEZParser::TimeStep, step));

    // ------------------------
    // Compute temporary vectors
    // ------------------------
    PetscCall(MatMult(TDSEZ->M(),  psi, TDSEZ->tmpVec[0]));
    PetscCall(MatMult(TDSEZ->H(),  psi, TDSEZ->tmpVec[1]));
    // Md/K/V feed energy AND dipole (accel needs invMassAvg); skip when neither is on.
    if (TDSEZParser::out_energy || TDSEZParser::out_dipole) {
        PetscCall(MatMult(TDSEZ->Md(), psi, TDSEZ->tmpVec[2]));
        PetscCall(MatMult(TDSEZ->K(),  psi, TDSEZ->tmpVec[3]));
        PetscCall(MatMult(TDSEZ->V(),  psi, TDSEZ->tmpVec[4]));
    }

    PetscInt nBatch = 5;
    PetscInt idxDx=-1, idxAccX=-1;
    if (TDSEZ->hasX) {
        idxDx   = nBatch++; PetscCall(MatMult(TDSEZ->Dx(),   psi, TDSEZ->tmpVec[idxDx]));
        idxAccX = nBatch++; PetscCall(MatMult(TDSEZ->dVdx(), psi, TDSEZ->tmpVec[idxAccX]));
    }

    // ------------------------
    // Fused reduction
    // ------------------------
    PetscScalar res[15];
    PetscCall(VecMDot(psi, nBatch, TDSEZ->tmpVec, res));

    const PetscReal   normSq       = PetscRealPart(res[0]);
    const PetscScalar expectationH = res[1];
    const PetscScalar invMassAvg   = (TDSEZParser::out_energy || TDSEZParser::out_dipole) ? res[2] : 0.0;
    const PetscScalar kineticE     = (TDSEZParser::out_energy) ? res[3] : 0.0;
    const PetscScalar potentialE   = (TDSEZParser::out_energy) ? res[4] : 0.0;

    PetscScalar laserEx=0.0, dipoleMX=0.0, accelX=0.0, projDx=0.0;

    if (TDSEZ->hasX)
    {
        laserEx  = TDSEZParser::Ex(t);
        projDx   = res[idxDx];
        dipoleMX = TDSEZ->charge * projDx;
        accelX   = (TDSEZParser::out_energy || TDSEZParser::out_dipole)
                   ? res[idxAccX] - invMassAvg*(TDSEZ->charge*laserEx) : 0.0;
    }


    const PetscScalar interactionE = laserEx*projDx;
    const PetscScalar totalEnergy  = expectationH + interactionE;

    // ------------------------
    // Population projection (needed for population output AND current decomposition)
    // ------------------------
    if (TDSEZParser::out_population || TDSEZParser::out_current) {
        PetscCall(VecMDot(TDSEZ->tmpVec[0], TDSEZ->NPOP, TDSEZ->boundstates.data(), TDSEZ->popDots.data()));
    }


    // Compute TRUE autocorrelation: <psi(0)|M|psi(t)>
    // ------------------------
    PetscScalar autocorr = 0.0;
    if (TDSEZParser::out_autocorr && TDSEZ->psi0() != PETSC_NULLPTR)
    {
        PetscCall(VecDot(TDSEZ->tmpVec[0], TDSEZ->psi0(), &autocorr));  // ⟨ψ(0)| M |ψ(t)⟩
    }

    if (TDSEZ->rank == 0)
    {
        PetscInt sIdx = TDSEZ->recordedSteps % TDSEZ->maxAllocatedSteps;

        // Dipoles Table [t, Ex, Dx, Ax]
        if (TDSEZParser::out_dipole) {
            PetscInt dOff = sIdx * 4;
            TDSEZ->dipBuffer[dOff+0] = t;
            TDSEZ->dipBuffer[dOff+1] = laserEx;
            TDSEZ->dipBuffer[dOff+2] = dipoleMX;
            TDSEZ->dipBuffer[dOff+3] = accelX;
        }

        // Populations Table [t, P0, P1, P2, ...]
        if (TDSEZParser::out_population) {
            PetscInt pWidth = TDSEZ->NPOP + 1;
            PetscInt pOff   = sIdx * pWidth;
            TDSEZ->popBuffer[pOff+0] = t;
            for (PetscInt i=0; i < TDSEZ->NPOP; ++i)
            {
                PetscScalar dot = TDSEZ->popDots[i];
                TDSEZ->popBuffer[pOff + i + 1] = PetscAbsScalar(dot * PetscConj(dot));
            }
        }

        // Energy Table [t, Kin, Pot, Int, Tot, Norm]
        if (TDSEZParser::out_energy) {
            PetscInt eOff = sIdx * 7;
            TDSEZ->energyBuffer[eOff+0] = t;
            TDSEZ->energyBuffer[eOff+1] = kineticE;
            TDSEZ->energyBuffer[eOff+2] = potentialE;
            TDSEZ->energyBuffer[eOff+3] = interactionE;
            TDSEZ->energyBuffer[eOff+4] = totalEnergy;
            TDSEZ->energyBuffer[eOff+5] = invMassAvg;
            TDSEZ->energyBuffer[eOff+6] = (PetscScalar)normSq;
        }
        // autocorrelation Table [t, Re<Psi(0)|Psi(t)>, Im<Psi(0)|Psi(t)>]
        // MUST be filled BEFORE the per-step flush below: the flush writes the
        // autocorrelation dataset from acBuffer for the window
        // [lastWrittenSteps, recordedSteps). If we incremented recordedSteps and
        // flushed first, the current step's AC slot would be flushed as stale
        // (zero) before it is computed -> silent all-zero autocorrelation
        // whenever the flush fires every step (e.g. OutputStrideTS=1).
        if (TDSEZParser::out_autocorr) {
            PetscInt acOff = sIdx * 3;
            TDSEZ->acBuffer[acOff+0] = t;
            TDSEZ->acBuffer[acOff+1] = PetscRealPart(autocorr);
            TDSEZ->acBuffer[acOff+2] = PetscImaginaryPart(autocorr);
        }

        TDSEZ->recordedSteps++;

        // In-progress flush: write the new window every OutputStrideTS (if set)
        // or every FLUSH_INTERVAL so RAM does not grow unbounded for long propagations.
        if (TDSEZ->recordedSteps > 0) {
            PetscInt stride = TDSEZParser::OutputStrideTS;
            if ((stride > 0 && TDSEZ->recordedSteps % stride == 0) ||
                TDSEZ->recordedSteps % TDSEZ->FLUSH_INTERVAL == 0) {
                PetscCallAbort(PETSC_COMM_WORLD,
                               TDSEZ->WriteHDF5(TDSEZ->outputFilename));
            }
        }
    }


    // SNAPSHOT WFS
    PetscInt snapshotStride = TDSEZParser::OutputStrideWFS;
    if (TDSEZParser::out_wfs && snapshotStride > 0 && step % snapshotStride == 0)
    {
        PetscInt step_index = step / snapshotStride;
        // WFSViewer is opened in COLLECTIVE mode, so VecView of the parallel
        // psi Vec is a collective HDF5 operation that MUST be called by every
        // rank. Gating it behind rank==0 deadlocked ranks 1..N (they wait
        // forever for the collective write), which hung the whole run. Calling
        // it collectively writes the full wavefunction correctly.
        PetscCall(PetscViewerHDF5SetTimestep(TDSEZ->WFSViewer, step_index));
        PetscObjectSetName((PetscObject)psi, "wavefunction");
        PetscCall(VecView(psi, TDSEZ->WFSViewer));
    }

    // TS stride snapshot
    PetscInt tsStride = TDSEZParser::OutputStrideTS;
    if (tsStride > 0 && step % tsStride == 0) {
        PetscInt ts_index = step / tsStride;
        PetscCall(PetscViewerHDF5SetTimestep(TDSEZ->TIMEViewer, ts_index));
        PetscObjectSetName((PetscObject)ts, "timestepper");
        PetscCall(TSView(ts, TDSEZ->TIMEViewer));
    }


    // ------------------------
    // Logging
    // ------------------------
    if (step % 100 == 0 || TDSEZParser::Verbose)
    {
        PetscPrintf(PETSC_COMM_WORLD,
            "      [%6" PetscInt_FMT "] t=%8.3f   <Psi|Psi>=%12.3e     <H>=%12.4e\n",
            step, t,
            (PetscReal)PetscRealPart(normSq),
            (PetscReal)PetscRealPart(totalEnergy));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/// 2D TS monitor: computes per-step observables and buffers them for HDF5.
/**
 * Extends the 1D monitor to two Cartesian axes (x, y): computes norm,
 * energy, dipoles, accelerations, velocity-gauge currents (intra/inter/bc
 * decomposition), Berry phase from population overlaps, Lz expectation,
 * populations, and autocorrelation. Buffers results and flushes to HDF5.
 *
 * @param[in] ts   Time-stepper context.
 * @param[in] step Current time-step index.
 * @param[in] t    Current simulation time.
 * @param[in] psi  Current state vector.
 * @param[in] ctx  TDSEZManager providing operators and output buffers.
 * @return PETSc error code.
 */
PetscErrorCode TDSEZMonitorHDF5_2D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx)
{
    (void)ts; (void)step; (void)t; (void)psi;
    PetscFunctionBeginUser;
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);

    if (TDSEZParser::out_tsurff) PetscCall(TDSEZ->AccumulateSurff(psi, t, TDSEZParser::TimeStep, step));

    /* ----------------------------------------------------------------
       MatMults
       ---------------------------------------------------------------- */
    PetscCall(MatMult(TDSEZ->M(),  psi, TDSEZ->tmpVec[0]));
    PetscCall(MatMult(TDSEZ->H(),  psi, TDSEZ->tmpVec[1]));
    // Md/K/V feed energy AND dipole (accel needs invMassAvg); skip when neither is on.
    if (TDSEZParser::out_energy || TDSEZParser::out_dipole) {
        PetscCall(MatMult(TDSEZ->Md(), psi, TDSEZ->tmpVec[2]));
        PetscCall(MatMult(TDSEZ->K(),  psi, TDSEZ->tmpVec[3]));
        PetscCall(MatMult(TDSEZ->V(),  psi, TDSEZ->tmpVec[4]));
    }

    PetscInt nBatch = 5;
    PetscInt idxDx=-1, idxDy=-1, idxAccX=-1, idxAccY=-1;
    PetscInt idxVx=-1, idxVy=-1, idxLz=-1;

    if (TDSEZ->hasX && TDSEZ->Dx() != PETSC_NULLPTR) {
        idxDx   = nBatch++; PetscCall(MatMult(TDSEZ->Dx(),   psi, TDSEZ->tmpVec[idxDx]));
        idxAccX = nBatch++; PetscCall(MatMult(TDSEZ->dVdx(), psi, TDSEZ->tmpVec[idxAccX]));
    }
    if (TDSEZ->hasY && TDSEZ->Dy() != PETSC_NULLPTR) {
        idxDy   = nBatch++; PetscCall(MatMult(TDSEZ->Dy(),   psi, TDSEZ->tmpVec[idxDy]));
        idxAccY = nBatch++; PetscCall(MatMult(TDSEZ->dVdy(), psi, TDSEZ->tmpVec[idxAccY]));
    }
    if (TDSEZ->hasX && TDSEZ->VelX() != PETSC_NULLPTR) {
        idxVx = nBatch++; PetscCall(MatMult(TDSEZ->VelX(), psi, TDSEZ->tmpVec[idxVx]));
    }
    if (TDSEZ->hasY && TDSEZ->VelY() != PETSC_NULLPTR) {
        idxVy = nBatch++; PetscCall(MatMult(TDSEZ->VelY(), psi, TDSEZ->tmpVec[idxVy]));
    }
    if (TDSEZ->Lz() != PETSC_NULLPTR) {
        idxLz = nBatch++; PetscCall(MatMult(TDSEZ->Lz(), psi, TDSEZ->tmpVec[idxLz]));
    }
    

    /* ----------------------------------------------------------------
       fused reduction
       ---------------------------------------------------------------- */
    PetscScalar res[16];
    PetscCall(VecMDot(psi, nBatch, TDSEZ->tmpVec, res));

    const PetscReal   normSq     = PetscRealPart(res[0]);
    const PetscScalar expectH    = res[1];
    const PetscScalar invMassAvg = (TDSEZParser::out_energy || TDSEZParser::out_dipole) ? res[2] : 0.0;
    const PetscScalar kineticE   = (TDSEZParser::out_energy) ? res[3] : 0.0;
    const PetscScalar potentialE = (TDSEZParser::out_energy) ? res[4] : 0.0;

    PetscScalar laserEx=0,  laserEy=0;
    PetscScalar projDx=0,   projDy=0;
    PetscScalar dipoleMX=0, dipoleMY=0;
    PetscScalar accelX=0,   accelY=0;
    PetscScalar Jx_total=0, Jy_total=0;

    if (idxDx >= 0) 
    {
        laserEx  = TDSEZParser::Ex(t);
        projDx   = res[idxDx];
        dipoleMX = TDSEZ->charge * projDx;
        accelX   = (TDSEZParser::out_energy || TDSEZParser::out_dipole)
                   ? res[idxAccX] - invMassAvg * (TDSEZ->charge * laserEx) : 0.0;
    }

    if (idxDy >= 0) {
        laserEy  = TDSEZParser::Ey(t);
        projDy   = res[idxDy];
        dipoleMY = TDSEZ->charge * projDy;
        accelY   = (TDSEZParser::out_energy || TDSEZParser::out_dipole)
                   ? res[idxAccY] - invMassAvg * (TDSEZ->charge * laserEy) : 0.0;
    }
    if (idxVx >= 0) Jx_total = res[idxVx];
    if (idxVy >= 0) Jy_total = res[idxVy];

    const PetscScalar interactionE = laserEx*projDx + laserEy*projDy;
    const PetscScalar totalEnergy  = expectH + interactionE;

    /* ----------------------------------------------------------------
       autocorrelation  ⟨ψ(0)|M|ψ(t)⟩: tmpVec[0] = M*psi
       ---------------------------------------------------------------- */
    PetscScalar autocorr = 0.0;
    if (TDSEZParser::out_autocorr && TDSEZ->psi0() != PETSC_NULLPTR)
        PetscCall(VecDot(TDSEZ->tmpVec[0], TDSEZ->psi0(), &autocorr));

    /* ---------------------------------------------------------------- 
       population projections  c_n = ⟨φ_n|M|ψ⟩: tmpVec[0] = M*psi
       (needed for population output AND current decomposition)
       ---------------------------------------------------------------- */
    if (TDSEZParser::out_population || TDSEZParser::out_current) {
        PetscCall(VecMDot(TDSEZ->tmpVec[0], TDSEZ->NPOP,
                          TDSEZ->boundstates.data(),
                          TDSEZ->popDots.data()));
    }

    /* ---------------------------------------------------------------- 
       Lz projections  m_n = ⟨φ_n|Lz|ψ⟩: tmpVec[0] = M*psi
       ---------------------------------------------------------------- */
        PetscScalar Lz_expect = 0.0;
        if (TDSEZParser::out_current && idxLz >= 0) Lz_expect = res[idxLz];

    /* ----------------------------------------------------------------
    Berry phase — geometric phase from consecutive overlaps
    Gamma_{step} = -arg⟨ψ_prev|M|ψ(t)⟩
    Accumulated: Γ(t) = sum Gamma_{step}
    Berry phase — from popDots coefficients, no VecCopy needed
    Gamma_{step} = -arg( Sum_n c_n*(t-dt) · c_n(t) )
    --------------------------------------------------------------- */
    const PetscInt N = TDSEZ->NPOP;
    PetscReal dGamma = 0.0;
    if (!TDSEZ->popDots_prev.empty())
    {
        PetscScalar overlap = 0.0;
        for (PetscInt n = 0; n < N; n++)
            overlap += PetscConj(TDSEZ->popDots_prev[n]) * TDSEZ->popDots[n];

        dGamma = -PetscAtan2Real(PetscImaginaryPart(overlap),
                                        PetscRealPart(overlap));
        TDSEZ->BerryPhase += dGamma;
    }
    TDSEZ->popDots_prev = TDSEZ->popDots;


    /* ----------------------------------------------------------------
       currents — reuse popDots, no extra MatMults
       ---------------------------------------------------------------- */
    PetscScalar Jx_intra=0, Jy_intra=0;
    PetscScalar Jx_inter=0, Jy_inter=0;
    PetscScalar Jx_bc=0,    Jy_bc=0;

    const PetscBool doVx = (idxVx >= 0 && !TDSEZ->vMat_x.empty());
    const PetscBool doVy = (idxVy >= 0 && !TDSEZ->vMat_y.empty());

    if (doVx || doVy)
    {
        // intraband: Σ_n |c_n|^2 * v_nn
        for (PetscInt n = 0; n < N; n++) {
            PetscReal pop = PetscRealPart(PetscConj(TDSEZ->popDots[n]) * TDSEZ->popDots[n]);
            if (doVx) Jx_intra += pop * TDSEZ->vMat_x[n*N+n];
            if (doVy) Jy_intra += pop * TDSEZ->vMat_y[n*N+n];
        }

        // interband: Σ_{m≠n} c_m* * c_n * v_mn
        for (PetscInt m = 0; m < N; m++)
        for (PetscInt n = 0; n < N; n++) {
            if (m == n) continue;
            PetscScalar rho_mn = PetscConj(TDSEZ->popDots[m]) * TDSEZ->popDots[n];
            if (doVx) Jx_inter += rho_mn * TDSEZ->vMat_x[m*N+n];
            if (doVy) Jy_inter += rho_mn * TDSEZ->vMat_y[m*N+n];
        }

        // bound-continuum: remainder
        if (doVx) Jx_bc = Jx_total - Jx_intra - Jx_inter;
        if (doVy) Jy_bc = Jy_total - Jy_intra - Jy_inter;
    }

    /* ----------------------------------------------------------------
       buffer (rank 0 only)
       ---------------------------------------------------------------- */
    if (TDSEZ->rank == 0)
    {
        PetscInt sIdx = TDSEZ->recordedSteps % TDSEZ->maxAllocatedSteps;

        // Dipoles [t, Ex, Ey, Dx, Dy, Ax, Ay]
        if (TDSEZParser::out_dipole) {
            PetscInt dOff = sIdx * 7;
            TDSEZ->dipBuffer[dOff+0] = t;
            TDSEZ->dipBuffer[dOff+1] = PetscRealPart(laserEx);
            TDSEZ->dipBuffer[dOff+2] = PetscRealPart(laserEy);
            TDSEZ->dipBuffer[dOff+3] = PetscRealPart(dipoleMX);
            TDSEZ->dipBuffer[dOff+4] = PetscRealPart(dipoleMY);
            TDSEZ->dipBuffer[dOff+5] = PetscRealPart(accelX);
            TDSEZ->dipBuffer[dOff+6] = PetscRealPart(accelY);
        }

        // Populations [t, P0, P1, ..., P_{N-1}]
        if (TDSEZParser::out_population) {
            PetscInt pWidth = TDSEZ->NPOP + 1;
            PetscInt pOff   = sIdx * pWidth;
            TDSEZ->popBuffer[pOff+0] = t;
            for (PetscInt i = 0; i < N; i++) {
                PetscScalar dot = TDSEZ->popDots[i];
                TDSEZ->popBuffer[pOff+i+1] = PetscRealPart(PetscConj(dot) * dot);
            }
        }

        // Energy [t, Kin, Pot, Int, Tot, InvMass, Norm]
        if (TDSEZParser::out_energy) {
            PetscInt eOff = sIdx * 7;
            TDSEZ->energyBuffer[eOff+0] = t;
            TDSEZ->energyBuffer[eOff+1] = PetscRealPart(kineticE);
            TDSEZ->energyBuffer[eOff+2] = PetscRealPart(potentialE);
            TDSEZ->energyBuffer[eOff+3] = PetscRealPart(interactionE);
            TDSEZ->energyBuffer[eOff+4] = PetscRealPart(totalEnergy);
            TDSEZ->energyBuffer[eOff+5] = PetscRealPart(invMassAvg);
            TDSEZ->energyBuffer[eOff+6] = (PetscScalar)normSq;
        }

        // Currents [t, Lz, dGamma, Jx_tot, Jy_tot, Jx_intra, Jy_intra, Jx_inter, Jy_inter, Jx_bc, Jy_bc]
        if (TDSEZParser::out_current) {
            PetscInt cOff = sIdx * 11;
            TDSEZ->currBuffer[cOff+0] = t;
            TDSEZ->currBuffer[cOff+1] = PetscRealPart(Lz_expect);
            TDSEZ->currBuffer[cOff+2] = dGamma;  
            TDSEZ->currBuffer[cOff+3] = PetscRealPart(Jx_total);
            TDSEZ->currBuffer[cOff+4] = PetscRealPart(Jy_total);
            TDSEZ->currBuffer[cOff+5] = PetscRealPart(Jx_intra);
            TDSEZ->currBuffer[cOff+6] = PetscRealPart(Jy_intra);
            TDSEZ->currBuffer[cOff+7] = PetscRealPart(Jx_inter);
            TDSEZ->currBuffer[cOff+8] = PetscRealPart(Jy_inter);
            TDSEZ->currBuffer[cOff+9] = PetscRealPart(Jx_bc);
            TDSEZ->currBuffer[cOff+10] = PetscRealPart(Jy_bc);
        }

        TDSEZ->recordedSteps++;

        // autocorrelation Table [t, Re<Psi(0)|Psi(t)>, Im<Psi(0)|Psi(t)>]
        // Filled BEFORE the per-step flush (see 3D monitor for rationale).
        if (TDSEZParser::out_autocorr) {
            PetscInt acOff = sIdx * 3;
            TDSEZ->acBuffer[acOff+0] = t;
            TDSEZ->acBuffer[acOff+1] = PetscRealPart(autocorr);
            TDSEZ->acBuffer[acOff+2] = PetscImaginaryPart(autocorr);
        }

        // In-progress flush: write the new window every OutputStrideTS (if set)
        // or every FLUSH_INTERVAL so RAM does not grow unbounded for long propagations.
        if (TDSEZ->recordedSteps > 0) {
            PetscInt stride = TDSEZParser::OutputStrideTS;
            if ((stride > 0 && TDSEZ->recordedSteps % stride == 0) ||
                TDSEZ->recordedSteps % TDSEZ->FLUSH_INTERVAL == 0) {
                PetscCallAbort(PETSC_COMM_WORLD,
                               TDSEZ->WriteHDF5(TDSEZ->outputFilename));
            }
        }
    }

    /* ----------------------------------------------------------------
       wavefunction snapshot
       ---------------------------------------------------------------- */
    PetscInt snapshotStride = TDSEZParser::OutputStrideWFS;
    if (TDSEZParser::out_wfs && snapshotStride > 0 && step % snapshotStride == 0) {
        PetscInt step_index = step / snapshotStride;
        // WFSViewer is opened in COLLECTIVE mode, so VecView of the parallel
        // psi Vec is a collective HDF5 operation that MUST be called by every
        // rank. Gating it behind rank==0 deadlocked ranks 1..N (they wait
        // forever for the collective write), which hung the whole run. Calling
        // it collectively writes the full wavefunction correctly.
        PetscCall(PetscViewerHDF5SetTimestep(TDSEZ->WFSViewer, step_index));
        PetscObjectSetName((PetscObject)psi, "wavefunction");
        PetscCall(VecView(psi, TDSEZ->WFSViewer));
    }

    // ------------------------
    // Logging 
    // ------------------------
    if (step % 100 == 0 || TDSEZParser::Verbose)
    {
        PetscPrintf(PETSC_COMM_WORLD,
            "      [%6" PetscInt_FMT "] t=%8.3f   ⟨Ψ|Ψ⟩=%12.3e     ⟨H⟩=%12.4e\n",
            step, t,
            (PetscReal)PetscRealPart(normSq),
            (PetscReal)PetscRealPart(totalEnergy));
    }


    PetscFunctionReturn(PETSC_SUCCESS);
}





/// 3D TS monitor: computes per-step observables and buffers them for HDF5.
/**
 * Extends the 2D monitor to three Cartesian axes (x, y, z): computes norm,
 * energy, dipoles, accelerations, velocity-gauge currents (intra/inter/bc
 * decomposition), Berry phase, Lz expectation, populations, and
 * autocorrelation. Buffers results and flushes to HDF5.
 *
 * @param[in] ts   Time-stepper context.
 * @param[in] step Current time-step index.
 * @param[in] t    Current simulation time.
 * @param[in] psi  Current state vector.
 * @param[in] ctx  TDSEZManager providing operators and output buffers.
 * @return PETSc error code.
 */
PetscErrorCode TDSEZMonitorHDF5_3D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx)
{
    PetscFunctionBeginUser;
    (void)ts; // unused callback parameter
    auto *TDSEZ = static_cast<TDSEZManager*>(ctx);

    if (TDSEZParser::out_tsurff) PetscCall(TDSEZ->AccumulateSurff(psi, t, TDSEZParser::TimeStep, step));

    // ------------------------
    // Compute temporary vectors
    // ------------------------
    PetscCall(MatMult(TDSEZ->M(),  psi, TDSEZ->tmpVec[0]));
    PetscCall(MatMult(TDSEZ->H(),  psi, TDSEZ->tmpVec[1]));
    // Md/K/V feed energy AND dipole (accel needs invMassAvg); skip when neither is on.
    if (TDSEZParser::out_energy || TDSEZParser::out_dipole) {
        PetscCall(MatMult(TDSEZ->Md(), psi, TDSEZ->tmpVec[2]));
        PetscCall(MatMult(TDSEZ->K(),  psi, TDSEZ->tmpVec[3]));
        PetscCall(MatMult(TDSEZ->V(),  psi, TDSEZ->tmpVec[4]));
    }

    PetscInt nBatch = 5;
    PetscInt idxDx=-1, idxDy=-1, idxDz=-1, idxAccX=-1, idxAccY=-1, idxAccZ=-1;
    PetscInt idxVx=-1, idxVy=-1, idxVz=-1, idxLz=-1;
    if (TDSEZ->hasX) {
        idxDx   = nBatch++; PetscCall(MatMult(TDSEZ->Dx(),   psi, TDSEZ->tmpVec[idxDx]));
        idxAccX = nBatch++; PetscCall(MatMult(TDSEZ->dVdx(), psi, TDSEZ->tmpVec[idxAccX]));
        if (TDSEZ->VelX() != PETSC_NULLPTR) { idxVx = nBatch++; PetscCall(MatMult(TDSEZ->VelX(), psi, TDSEZ->tmpVec[idxVx])); }
    }
    if (TDSEZ->hasY) {
        idxDy   = nBatch++; PetscCall(MatMult(TDSEZ->Dy(),   psi, TDSEZ->tmpVec[idxDy]));
        idxAccY = nBatch++; PetscCall(MatMult(TDSEZ->dVdy(), psi, TDSEZ->tmpVec[idxAccY]));
        if (TDSEZ->VelY() != PETSC_NULLPTR) { idxVy = nBatch++; PetscCall(MatMult(TDSEZ->VelY(), psi, TDSEZ->tmpVec[idxVy])); }
    }

    if (TDSEZ->hasZ) {
        idxDz   = nBatch++; PetscCall(MatMult(TDSEZ->Dz(),   psi, TDSEZ->tmpVec[idxDz]));
        idxAccZ = nBatch++; PetscCall(MatMult(TDSEZ->dVdz(), psi, TDSEZ->tmpVec[idxAccZ]));
        if (TDSEZ->VelZ() != PETSC_NULLPTR) { idxVz = nBatch++; PetscCall(MatMult(TDSEZ->VelZ(), psi, TDSEZ->tmpVec[idxVz])); }
    }
    if (TDSEZ->Lz() != PETSC_NULLPTR) {
        idxLz = nBatch++; PetscCall(MatMult(TDSEZ->Lz(), psi, TDSEZ->tmpVec[idxLz]));
    }

    // ------------------------
    // Fused reduction
    // ------------------------
    PetscScalar res[16]; 
    PetscCall(VecMDot(psi, nBatch, TDSEZ->tmpVec, res));

    const PetscReal   normSq       = PetscRealPart(res[0]);
    const PetscScalar expectationH = res[1];
    const PetscScalar invMassAvg   = (TDSEZParser::out_energy || TDSEZParser::out_dipole) ? res[2] : 0.0;
    const PetscScalar kineticE     = (TDSEZParser::out_energy) ? res[3] : 0.0;
    const PetscScalar potentialE   = (TDSEZParser::out_energy) ? res[4] : 0.0;

    PetscScalar laserEx=0.0, laserEy=0.0, laserEz=0.0;
    PetscScalar dipoleMX=0.0, dipoleMY=0.0, dipoleMZ=0.0;
    PetscScalar accelX=0.0, accelY=0.0, accelZ=0.0;
    PetscScalar projDx=0.0, projDy=0.0, projDz=0.0;

    if (TDSEZ->hasX) {
        laserEx  = TDSEZParser::Ex(t);
        projDx   = res[idxDx];
        dipoleMX = TDSEZ->charge * projDx;
        accelX   = (TDSEZParser::out_energy || TDSEZParser::out_dipole)
                   ? res[idxAccX] - invMassAvg*(TDSEZ->charge*laserEx) : 0.0;
    }
    if (TDSEZ->hasY) {
        laserEy  = TDSEZParser::Ey(t);
        projDy   = res[idxDy];
        dipoleMY = TDSEZ->charge * projDy;
        accelY   = (TDSEZParser::out_energy || TDSEZParser::out_dipole)
                   ? res[idxAccY] - invMassAvg*(TDSEZ->charge*laserEy) : 0.0;
    }

    if (TDSEZ->hasZ) {
        laserEz  = TDSEZParser::Ez(t);
        projDz   = res[idxDz];
        dipoleMZ = TDSEZ->charge * projDz;
        accelZ   = (TDSEZParser::out_energy || TDSEZParser::out_dipole)
                   ? res[idxAccZ] - invMassAvg*(TDSEZ->charge*laserEz) : 0.0;
    }

    const PetscScalar interactionE = (laserEx*projDx) + (laserEy*projDy) + (laserEz*projDz);
    const PetscScalar totalEnergy  = expectationH + interactionE;



    // Compute TRUE autocorrelation: ⟨ψ(0)|M|ψ(t)⟩
    // ------------------------
    PetscScalar autocorr = 0.0;
    if (TDSEZParser::out_autocorr && TDSEZ->psi0() != PETSC_NULLPTR)
    {
        PetscCall(VecDot(TDSEZ->tmpVec[0], TDSEZ->psi0(), &autocorr));  // ⟨ψ(0)| M |ψ(t)⟩
    }

    // ------------------------
    // Population projection (needed for population output AND current decomposition)
    // ------------------------
    if (TDSEZParser::out_population || TDSEZParser::out_current) {
        PetscCall(VecMDot(TDSEZ->tmpVec[0], TDSEZ->NPOP, TDSEZ->boundstates.data(), TDSEZ->popDots.data()));
    }

    /* ----------------------------------------------------------------
       Lz projection  m = ⟨ψ|Lz|ψ⟩ (needed for current output)
       ---------------------------------------------------------------- */
    PetscScalar Lz_expect = 0.0;
    if (TDSEZParser::out_current && idxLz >= 0) Lz_expect = res[idxLz];

    /* ----------------------------------------------------------------
       Berry phase — geometric phase from consecutive overlaps
       ---------------------------------------------------------------- */
    const PetscInt BerryN = TDSEZ->NPOP;
    PetscReal dGamma = 0.0;
    if (TDSEZParser::out_current && !TDSEZ->popDots_prev.empty()) {
        PetscScalar overlap = 0.0;
        for (PetscInt n = 0; n < BerryN; n++)
            overlap += PetscConj(TDSEZ->popDots_prev[n]) * TDSEZ->popDots[n];
        dGamma = -PetscAtan2Real(PetscImaginaryPart(overlap), PetscRealPart(overlap));
        TDSEZ->BerryPhase += dGamma;
    }
    TDSEZ->popDots_prev = TDSEZ->popDots;

    /* ----------------------------------------------------------------
       currents — reuse popDots, no extra MatMults
       ---------------------------------------------------------------- */
    PetscScalar Jx_total=0, Jy_total=0, Jz_total=0;
    if (idxVx >= 0) Jx_total = res[idxVx];
    if (idxVy >= 0) Jy_total = res[idxVy];
    if (idxVz >= 0) Jz_total = res[idxVz];

    PetscScalar Jx_intra=0, Jy_intra=0, Jz_intra=0;
    PetscScalar Jx_inter=0, Jy_inter=0, Jz_inter=0;
    PetscScalar Jx_bc=0,    Jy_bc=0,    Jz_bc=0;

    const PetscBool doVx = (idxVx >= 0 && !TDSEZ->vMat_x.empty());
    const PetscBool doVy = (idxVy >= 0 && !TDSEZ->vMat_y.empty());
    const PetscBool doVz = (idxVz >= 0 && !TDSEZ->vMat_z.empty());

    if (doVx || doVy || doVz) {
        for (PetscInt n = 0; n < BerryN; n++) {
            PetscReal pop = PetscRealPart(PetscConj(TDSEZ->popDots[n]) * TDSEZ->popDots[n]);
            if (doVx) Jx_intra += pop * TDSEZ->vMat_x[n*BerryN+n];
            if (doVy) Jy_intra += pop * TDSEZ->vMat_y[n*BerryN+n];
            if (doVz) Jz_intra += pop * TDSEZ->vMat_z[n*BerryN+n];
        }
        for (PetscInt m = 0; m < BerryN; m++)
        for (PetscInt n = 0; n < BerryN; n++) {
            if (m == n) continue;
            PetscScalar rho_mn = PetscConj(TDSEZ->popDots[m]) * TDSEZ->popDots[n];
            if (doVx) Jx_inter += rho_mn * TDSEZ->vMat_x[m*BerryN+n];
            if (doVy) Jy_inter += rho_mn * TDSEZ->vMat_y[m*BerryN+n];
            if (doVz) Jz_inter += rho_mn * TDSEZ->vMat_z[m*BerryN+n];
        }
        if (doVx) Jx_bc = Jx_total - Jx_intra - Jx_inter;
        if (doVy) Jy_bc = Jy_total - Jy_intra - Jy_inter;
        if (doVz) Jz_bc = Jz_total - Jz_intra - Jz_inter;
    }

    if (TDSEZ->rank == 0) {
        PetscInt sIdx = TDSEZ->recordedSteps % TDSEZ->maxAllocatedSteps;

        // Dipoles Table [t, Ex, Ey, Ez, Dx, Dy, Dz, Ax, Ay, Az]
        if (TDSEZParser::out_dipole) {
            PetscInt dOff = sIdx * 10;
            TDSEZ->dipBuffer[dOff+0] = t;
            TDSEZ->dipBuffer[dOff+1] = laserEx;
            TDSEZ->dipBuffer[dOff+2] = laserEy;
            TDSEZ->dipBuffer[dOff+3] = laserEz;
            TDSEZ->dipBuffer[dOff+4] = dipoleMX;
            TDSEZ->dipBuffer[dOff+5] = dipoleMY;
            TDSEZ->dipBuffer[dOff+6] = dipoleMZ;
            TDSEZ->dipBuffer[dOff+7] = accelX;
            TDSEZ->dipBuffer[dOff+8] = accelY;
            TDSEZ->dipBuffer[dOff+9] = accelZ;
        }

        // Populations Table [t, P0, P1, P2, ...]
        if (TDSEZParser::out_population) {
            PetscInt pWidth = TDSEZ->NPOP + 1;
            PetscInt pOff   = sIdx * pWidth;
            TDSEZ->popBuffer[pOff+0] = t; 
            for (PetscInt i=0; i < TDSEZ->NPOP; ++i) 
            {
                PetscScalar dot = TDSEZ->popDots[i];
                TDSEZ->popBuffer[pOff + i + 1] = PetscAbsScalar(dot * PetscConj(dot));
            }
        }

        // Energy Table [t, Kin, Pot, Int, Tot, Norm]
        if (TDSEZParser::out_energy) {
            PetscInt eOff = sIdx * 7;
            TDSEZ->energyBuffer[eOff+0] = t;
            TDSEZ->energyBuffer[eOff+1] = kineticE;
            TDSEZ->energyBuffer[eOff+2] = potentialE;
            TDSEZ->energyBuffer[eOff+3] = interactionE;
            TDSEZ->energyBuffer[eOff+4] = totalEnergy;
            TDSEZ->energyBuffer[eOff+5] = invMassAvg;
            TDSEZ->energyBuffer[eOff+6] = (PetscScalar)normSq;
        }
        // autocorrelation Table [t, Re⟨Ψ(0)|Ψ(t)⟩, Im⟨Ψ(0)|Ψ(t)⟩]
        // MUST be filled BEFORE the per-step flush below: the flush writes the
        // autocorrelation dataset from acBuffer for the window
        // [lastWrittenSteps, recordedSteps). If we incremented recordedSteps and
        // flushed first, the current step's AC slot would be flushed as stale
        // (zero) before it is computed -> silent all-zero autocorrelation
        // whenever the flush fires every step (e.g. OutputStrideTS=1).
        if (TDSEZParser::out_autocorr) {
            PetscInt acOff = sIdx * 3;
            TDSEZ->acBuffer[acOff+0] = t;
            TDSEZ->acBuffer[acOff+1] = PetscRealPart(autocorr);
            TDSEZ->acBuffer[acOff+2] = PetscImaginaryPart(autocorr);
        }

        TDSEZ->recordedSteps++;

        // In-progress flush: write the new window every OutputStrideTS (if set)
        // or every FLUSH_INTERVAL so RAM does not grow unbounded for long propagations.
        if (TDSEZ->recordedSteps > 0) {
            PetscInt stride = TDSEZParser::OutputStrideTS;
            if ((stride > 0 && TDSEZ->recordedSteps % stride == 0) ||
                TDSEZ->recordedSteps % TDSEZ->FLUSH_INTERVAL == 0) {
                PetscCallAbort(PETSC_COMM_WORLD,
                               TDSEZ->WriteHDF5(TDSEZ->outputFilename));
            }
        }

        // Currents [t, Lz, dGamma, Jx_tot, Jy_tot, Jz_tot, Jx_intra, Jy_intra, Jz_intra, Jx_inter, Jy_inter, Jz_inter, Jx_bc, Jy_bc, Jz_bc]
        if (TDSEZParser::out_current) {
            PetscInt cOff = sIdx * 15;
            TDSEZ->currBuffer[cOff+0]  = t;
            TDSEZ->currBuffer[cOff+1]  = PetscRealPart(Lz_expect);
            TDSEZ->currBuffer[cOff+2]  = dGamma;
            TDSEZ->currBuffer[cOff+3]  = PetscRealPart(Jx_total);
            TDSEZ->currBuffer[cOff+4]  = PetscRealPart(Jy_total);
            TDSEZ->currBuffer[cOff+5]  = PetscRealPart(Jz_total);
            TDSEZ->currBuffer[cOff+6]  = PetscRealPart(Jx_intra);
            TDSEZ->currBuffer[cOff+7]  = PetscRealPart(Jy_intra);
            TDSEZ->currBuffer[cOff+8]  = PetscRealPart(Jz_intra);
            TDSEZ->currBuffer[cOff+9]  = PetscRealPart(Jx_inter);
            TDSEZ->currBuffer[cOff+10] = PetscRealPart(Jy_inter);
            TDSEZ->currBuffer[cOff+11] = PetscRealPart(Jz_inter);
            TDSEZ->currBuffer[cOff+12] = PetscRealPart(Jx_bc);
            TDSEZ->currBuffer[cOff+13] = PetscRealPart(Jy_bc);
            TDSEZ->currBuffer[cOff+14] = PetscRealPart(Jz_bc);
        }
    }

    // SNAPSHOT WFS — set the timestep FIRST, then write ONCE (rank 0 only).
    // The previous code set the timestep only AFTER a first VecView, so it wrote
    // the field twice: once at the previous timestep index and once at the
    // correct one, producing duplicate / mislabelled snapshots.
    PetscInt snapshotStride = TDSEZParser::OutputStrideWFS;
    if (TDSEZParser::out_wfs && snapshotStride > 0 && step % snapshotStride == 0)
    {
        PetscInt step_index = step / snapshotStride;
        // WFSViewer is opened in COLLECTIVE mode, so VecView of the parallel
        // psi Vec is a collective HDF5 operation that MUST be called by every
        // rank. Gating it behind rank==0 deadlocked ranks 1..N (they wait
        // forever for the collective write), which hung the whole run. Calling
        // it collectively writes the full wavefunction correctly.
        PetscCall(PetscViewerHDF5SetTimestep(TDSEZ->WFSViewer, step_index));
        PetscObjectSetName((PetscObject)psi, "wavefunction");
        PetscCall(VecView(psi, TDSEZ->WFSViewer));
    }

    // TS stride snapshot
    PetscInt tsStride = TDSEZParser::OutputStrideTS;
    if (tsStride > 0 && step % tsStride == 0) {
        PetscInt ts_index = step / tsStride;
        PetscCall(PetscViewerHDF5SetTimestep(TDSEZ->TIMEViewer, ts_index));
        PetscObjectSetName((PetscObject)ts, "timestepper");
        PetscCall(TSView(ts, TDSEZ->TIMEViewer));
    }


    // -------------------------
    // Logging 
    // -------------------------
    if (step % 100 == 0 || TDSEZParser::Verbose)
    {
        PetscPrintf(PETSC_COMM_WORLD,
            "      [%6" PetscInt_FMT "] t=%8.3f   ⟨Ψ|Ψ⟩=%12.3e     ⟨H⟩=%12.4e\n",
            step, t,
            (PetscReal)PetscRealPart(normSq),
            (PetscReal)PetscRealPart(totalEnergy));
    }
    
    PetscFunctionReturn(PETSC_SUCCESS);
}




/*  TDSEZCore propagator class  */
