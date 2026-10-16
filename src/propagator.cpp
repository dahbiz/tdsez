#include "tdsez_internal.hpp"
#include "tdsez_multigrid.hpp"
#include "tdsez_quadobs.hpp"

/// ILU fill level of the ASM subdomain solves (-tdsez_ilu_levels overrides).
/// ILU(0): ILU(1) saved at most one GMRES iteration (none in 3D) while its
/// fill made the 3D factorisation 10x slower (degree 4: 45 s against 4.6 s,
/// 2.5x slower steps). The 1e-12 solve tolerance is unchanged.
static PetscInt TDSEZILULevels(PetscInt localRows)
{
    (void)localRows;
    PetscInt levels = 0;
    PetscOptionsGetInt(NULL, NULL, "-tdsez_ilu_levels", &levels, NULL);
    return levels;
}

/// Crank-Nicolson: one SNES/KSP solve per TS step (SNESKSPONLY).
static TDSEZPCRefresh cnRefresh;
static PetscErrorCode TDSEZCNPostStep(TS ts)
{
    PetscFunctionBeginUser;
    SNES snes; KSP ksp; PetscInt its;
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    if (cnRefresh.Stale(its)) PetscCall(SNESSetLagPreconditioner(snes, -2));   // rebuild at the next solve
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @file propagator.cpp
 * @brief TDSEZPropagator: PETSc TS time-stepping for the TDSE, including
 *        CAP activation, Crank-Nicolson (TSTHETA θ=0.5) integration, and
 *        bound-state post-processing (M-orthogonalization, Lz²
 *        diagonalisation, quantum number assignment).
 * @author TDSEZ Project
 */



/// @brief Construct the TDSEZPropagator. Activates the complex absorbing
///        potential (CAP) by adding it to the Hamiltonian, creates work
///        the CAP is merged into H and destroyed.
/// @param manager  Reference to the TDSEZManager owning the operators and
///                 callback pointers.
TDSEZPropagator::TDSEZPropagator(TDSEZManager &manager) : tdse_(*manager.getCore()), manager_(manager)
{
    PetscFunctionBeginUser;

    // set/unset complex absorbing potential
    PetscBool TDSEZEnableCAP = TDSEZParser::EnableCAP;
    Mat CAP = manager.CAP();
    if (TDSEZEnableCAP && CAP != PETSC_NULLPTR)
    {
        MatAXPY(manager.H(), 1.0, CAP, SAME_NONZERO_PATTERN);
        MatDestroy(&CAP);
        manager.setCAP(PETSC_NULLPTR);
        manager.getAssembler()->CAP = PETSC_NULLPTR;
    }
    // The Crank-Nicolson Jacobian J_ is created in Evolve(), only when the
    // TS path runs (the rational propagators use their own stage matrices).

    PetscFunctionReturnVoid();
}


/// @brief Run the time propagation using PETSc TS (TSTHETA with θ=0.5,
///        i.e. Crank-Nicolson). Configures the TS with the IFunction/
///        IJacobian callbacks, GMRES linear solver with ILU preconditioner,
///        attaches HDF5 monitors, and calls TSSolve.
/// @return PetscErrorCode — PETSC_SUCCESS on success, or a PETSc error
///         code on invalid time parameters or solver failure.
PetscErrorCode TDSEZPropagator::Evolve()
{
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    PetscReal dt   = TDSEZParser::TimeStep;
    PetscReal endT = TDSEZParser::FinalTime;
    if (dt <= 0.0 || endT <= 0.0)
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid time step or final time.");

    if (TDSEZParser::Propagator != "cn") {
        PetscCall(EvolveRational());
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // ---------------- TS ----------------
    ierr = TSCreate(PETSC_COMM_WORLD, &ts_); CHKERRQ(ierr);
    ierr = TSSetType(ts_, TSTHETA); CHKERRQ(ierr);
    ierr = TSThetaSetTheta(ts_, 0.5); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts_, TS_LINEAR); CHKERRQ(ierr);
    ierr = TSSetSolution(ts_, tdse_.initialPsi); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts_, dt); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts_, endT); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);

    ierr = TSSetIFunction(ts_, PETSC_NULLPTR, manager_.TDSEZIFunctionPtr, &manager_); CHKERRQ(ierr);
    if (!J_) PetscCall(MatDuplicate(manager_.H(), MAT_DO_NOT_COPY_VALUES, &J_));
    ierr = TSSetIJacobian(ts_, J_, J_, manager_.TDSEZIJacobianPtr, &manager_); CHKERRQ(ierr);

    // Jacobian structure fixed
    ierr = MatSetOption(J_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(J_, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(J_, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE); CHKERRQ(ierr);

    PetscInt dim = TDSEZParser::Dimension;
    switch (dim) {
        case 1: ierr = TSMonitorSet(ts_, TDSEZMonitorHDF5_1D, &manager_, PETSC_NULLPTR); CHKERRQ(ierr); break;
        case 2: ierr = TSMonitorSet(ts_, TDSEZMonitorHDF5_2D, &manager_, PETSC_NULLPTR); CHKERRQ(ierr); break;
        case 3: ierr = TSMonitorSet(ts_, TDSEZMonitorHDF5_3D, &manager_, PETSC_NULLPTR); CHKERRQ(ierr); break;
        default: SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %d", (int)dim);
    }

    // ---------------- SNES / KSP / PC ----------------
    SNES snes; KSP ksp; PC pc;
    ierr = TSGetSNES(ts_, &snes); CHKERRQ(ierr);
    ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);

    ierr = SNESSetType(snes, SNESKSPONLY); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRQ(ierr);

    PetscInt mat_rows;
    PetscCall(MatGetSize(J_, &mat_rows, NULL));
    const PetscBool isLarge = (mat_rows > 100000) ? PETSC_TRUE : PETSC_FALSE;
    // Restart: the Crank-Nicolson step converges in a handful of iterations
    // (typically <10) with the ILU-PC, so a large Krylov space only wastes
    // memory and orthogonalization setup at high rank counts. 50 is ample.
    const PetscInt  restart = isLarge ? 50 : 30;
    ierr = KSPGMRESSetRestart(ksp, restart); CHKERRQ(ierr);
    // Use PETSc's default MODIFIED Gram-Schmidt orthogonalization (single
    // reduction pass). The previous Classical-GS + CGS-refine doubled the
    // collective inner products (VecMDot) per iteration, which dominates
    // communication cost at high rank counts for a well-preconditioned system
    // where modified GS is numerically sufficient. Convergence tolerance is
    // unchanged (still 1e-12), so accuracy is identical.
    ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1000); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRQ(ierr);
    // The Crank-Nicolson Jacobian J = a*i*M - H - E(t)*Dx has a CONSTANT
    // nonzero pattern every step (only the scalar E(t) rescales Dx values);
    // the ILU(0) subdomain factorization therefore remains a valid
    // preconditioner across all steps. Reusing it skips the per-step
    // PCSetUp/ILU-refactor without changing the problem, and the 1e-12
    // KSP tolerance still drives every linear solve to the identical residual
    // (PETSc only refreshes the factorization if a solve actually diverges).
    ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE); CHKERRQ(ierr);
    // The KSP flag alone is not enough: on every SNESSolve the SNES layer
    // resets the KSP reuse flag from its own preconditioner lag (default 1,
    // i.e. rebuild every solve), so ASM re-extracted the subdomain matrices
    // and refactored ILU at every time step. Lag -2 builds the preconditioner
    // once (at the first step) and never again; "persists" keeps that across
    // the successive SNESSolve calls made by TS. Both are still overridable
    // with -snes_lag_preconditioner on the command line.
    ierr = SNESSetLagPreconditioner(snes, -2); CHKERRQ(ierr);
    ierr = SNESSetLagPreconditionerPersists(snes, PETSC_TRUE); CHKERRQ(ierr);
    // Right preconditioning: GMRES then monitors the true (unpreconditioned)
    // residual, so the 1e-12 tolerance bounds the actual Crank-Nicolson solve
    // error independently of preconditioner quality. With left preconditioning
    // the tolerance applied to the preconditioned residual, which for
    // ill-conditioned systems (e.g. C0 interface knots) left ~1e-6 errors in
    // <x>(t); right preconditioning brings these to ~1e-14 at the same cost.
    ierr = KSPSetPCSide(ksp, PC_RIGHT); CHKERRQ(ierr);

    if (TDSEZParser::Preconditioner != "asm") {
        // Geometric B-spline multigrid (mg) or PETSc algebraic multigrid
        // (gamg); flexible GMRES keeps the full operator and the 1e-12
        // tolerance. The coarse levels are built at the first solve (J is
        // filled then) and reused like the ASM factors.
        PetscInt levels = 0;
        PetscCall(KSPSetType(ksp, KSPFGMRES));
        PetscCall(KSPGMRESSetRestart(ksp, restart));
        if (TDSEZParser::Preconditioner == "mg") {
            PetscCall(TDSEZSetupMultigrid(tdse_.iga, pc, &levels));
            TDSEZInfo().kv("Preconditioner", "geometric B-spline multigrid, %d levels", (int)levels);
        } else {
            PetscCall(PCSetType(pc, PCGAMG));
            TDSEZInfo().kv("Preconditioner", "PETSc algebraic multigrid (GAMG)");
        }
        PetscCall(TSSetFromOptions(ts_));
        PetscCall(TSSetUp(ts_));
        PetscCall(TSSetPostStep(ts_, TDSEZCNPostStep));
        PetscCall(TSSolve(ts_, tdse_.initialPsi));
        if (cnRefresh.rebuilds)
            TDSEZInfo().kv("Preconditioner rebuilt", "%d times (operator changed with the field)", (int)cnRefresh.rebuilds);
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    ierr = PCSetType(pc, PCASM); CHKERRQ(ierr);
    // ierr = PCASMSetOverlap(pc, 1); CHKERRQ(ierr);
    ierr = PCASMSetType(pc, PC_ASM_RESTRICT); CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc, 0);                                CHKERRQ(ierr);

    ierr = TSSetFromOptions(ts_); CHKERRQ(ierr);
    ierr = SNESSetUp(snes); CHKERRQ(ierr);
    ierr = KSPSetUp(ksp); CHKERRQ(ierr);

// ---------------- Configure ASM subsolvers ONCE ----------------
    PetscInt nlocal; KSP *subksp;
    ierr = PCASMGetSubKSP(pc, &nlocal, NULL, &subksp); CHKERRQ(ierr);

    if (subksp) 
    {
        for (PetscInt i = 0; i < nlocal; ++i)
        {
            PC subpc;
            ierr = KSPGetPC(subksp[i], &subpc); CHKERRQ(ierr);

            // Enforce structured Incomplete LU (Bypasses full LU fill-in slowness)
            ierr = PCSetType(subpc, PCILU);                                    CHKERRQ(ierr);

            PetscInt local_rows = 0;
            PetscCall(MatGetLocalSize(J_, &local_rows, NULL));
            PetscCall(PCFactorSetLevels(subpc, TDSEZILULevels(local_rows)));

            // Lock memory structures now that the drop threshold is gone
            ierr = PCFactorSetReuseOrdering(subpc, PETSC_TRUE);                CHKERRQ(ierr);
            ierr = PCFactorSetReuseFill(subpc, PETSC_TRUE);                    CHKERRQ(ierr);

            // Numerical stability pivots for complex wave propagation
            ierr = PCFactorSetShiftType(subpc, MAT_SHIFT_INBLOCKS);            CHKERRQ(ierr);
            ierr = PCFactorSetShiftAmount(subpc, 1e-12);                        CHKERRQ(ierr);

            // Subdomain internal solution constraints (reuse is decided by the
            // outer solver, so a refresh also refactors the subdomains)
            ierr = KSPSetTolerances(subksp[i], 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1000); CHKERRQ(ierr);
            ierr = KSPSetType(subksp[i], KSPPREONLY); CHKERRQ(ierr);
        }
    }

    ierr = PCSetUp(pc);  CHKERRQ(ierr);
    ierr = TSSetUp(ts_); CHKERRQ(ierr);
    PetscCall(TSSetPostStep(ts_, TDSEZCNPostStep));

    // ---------------- Solve ----------------
    ierr = TSSolve(ts_, tdse_.initialPsi); CHKERRQ(ierr);
    if (cnRefresh.rebuilds)
        TDSEZInfo().kv("Preconditioner rebuilt", "%d times (operator changed with the field)", (int)cnRefresh.rebuilds);

    PetscFunctionReturn(PETSC_SUCCESS);
}


// ─────────────────────────────────────────────────────────────────────────────
//  Rational propagators: Propagator = pade22 | magnus4
// ─────────────────────────────────────────────────────────────────────────────
//
// The semi-discrete equation is i M c' = H(t) c with H(t) = H + E(t).D (H
// includes the CAP once applied). Crank-Nicolson is the Pade(1,1)
// approximant of the step exponential. These propagators use the Pade(2,2)
// approximant
//     R(z) = (1 + z/2 + z^2/12) / (1 - z/2 + z^2/12),   z = -i h M^{-1} H_E,
// which is fourth order and unitary for Hermitian H_E. Its numerator has the
// roots z_+- = -3 +- i sqrt(3), so R factorises into two Cayley-type stages,
//     (M + g_k H_E) y = (M + conj(g_k) H_E) x,   g_{1,2} = h (-+sqrt(3) + 3i)/12,
// each of which is itself unitary. Every stage costs one sparse solve with the
// sparsity pattern of the Crank-Nicolson Jacobian.
//
//   pade22  : exponential midpoint rule, one R22 with H_E = H(t + h/2).
//             Second order in the field variation, fourth order in the
//             field-free phase (removes the Crank-Nicolson dispersion
//             1/(1+(E h/2)^2) of high-energy components). 2 solves/step.
//   magnus4 : commutator-free fourth-order Magnus (Blanes & Moan 2006),
//             U = exp(h(a1 A1 + a2 A2)) exp(h(a2 A1 + a1 A2)), A_j = -i M^{-1}
//             H(t + c_j h), with Gauss nodes c_j = 1/2 -+ sqrt(3)/6 and
//             a_{1,2} = (3 -+ 2 sqrt(3))/12. Since a1 + a2 = 1/2, each factor is
//             a half step with the effective field 2(a E1 + b E2), evaluated
//             with R22. Fourth order overall. 4 solves/step.

/// @brief Configure a stage solver like the Crank-Nicolson KSP: GMRES with
///        right preconditioning (tolerance on the true residual), restricted
///        additive Schwarz with ILU subdomain solves, preconditioner built once
///        and reused. Options prefix pade_stage{1,2}_.
static PetscErrorCode TDSEZConfigureStageKSP(KSP ksp, Mat A, const char *prefix, IGA iga)
{
    PetscFunctionBeginUser;
    PC pc;
    PetscInt rows = 0;
    PetscCall(MatGetSize(A, &rows, NULL));
    PetscCall(KSPSetOptionsPrefix(ksp, prefix));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPGMRESSetRestart(ksp, rows > 100000 ? 50 : 30));
    PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
    PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
    PetscCall(KSPGetPC(ksp, &pc));
    if (TDSEZParser::Preconditioner != "asm") {
        // flexible GMRES: the outer solve keeps the full operator and its
        // tolerance whatever the multigrid cycle does
        PetscInt levels = 0;
        PetscCall(KSPSetType(ksp, KSPFGMRES));
        PetscCall(KSPGMRESSetRestart(ksp, rows > 100000 ? 50 : 30));
        if (TDSEZParser::Preconditioner == "mg") PetscCall(TDSEZSetupMultigrid(iga, pc, &levels));
        else PetscCall(PCSetType(pc, PCGAMG));
        PetscCall(KSPSetFromOptions(ksp));
        PetscCall(KSPSetUp(ksp));
        PetscCall(PCSetUp(pc));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PCSetType(pc, PCASM));
    PetscCall(PCASMSetType(pc, PC_ASM_RESTRICT));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetUp(ksp));

    PetscBool isASM = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCASM, &isASM));
    if (isASM) {
        PetscInt nlocal = 0, localRows = 0;
        KSP     *subksp = NULL;
        PetscCall(PCASMGetSubKSP(pc, &nlocal, NULL, &subksp));
        PetscCall(MatGetLocalSize(A, &localRows, NULL));
        for (PetscInt i = 0; i < nlocal; ++i) {
            PC subpc;
            PetscCall(KSPGetPC(subksp[i], &subpc));
            PetscCall(PCSetType(subpc, PCILU));
            PetscCall(PCFactorSetLevels(subpc, TDSEZILULevels(localRows)));
            PetscCall(PCFactorSetReuseOrdering(subpc, PETSC_TRUE));
            PetscCall(PCFactorSetReuseFill(subpc, PETSC_TRUE));
            PetscCall(PCFactorSetShiftType(subpc, MAT_SHIFT_INBLOCKS));
            PetscCall(PCFactorSetShiftAmount(subpc, 1e-12));
            PetscCall(KSPSetType(subksp[i], KSPPREONLY));
        }
        PetscCall(PCSetUp(pc));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief Set A = M + g (H + sum_a E_a D_a) on the active polarization axes.
PetscErrorCode TDSEZPropagator::BuildStageMatrix(Mat A, PetscScalar g, const PetscScalar E[3])
{
    PetscFunctionBeginUser;
    const Mat       D[3]  = {manager_.Dx(), manager_.Dy(), manager_.Dz()};
    const PetscBool on[3] = {(PetscBool)TDSEZParser::Coupled(0), (PetscBool)TDSEZParser::Coupled(1),
                             (PetscBool)TDSEZParser::Coupled(2)};   // axes coupled to the field
    PetscCall(MatCopy(manager_.H(), A, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(A, g));
    PetscCall(MatAXPY(A, 1.0, manager_.M(), SAME_NONZERO_PATTERN));
    for (int a = 0; a < 3; ++a)
        if (on[a] && D[a] && E[a] != 0.0)
            PetscCall(MatAXPY(A, g * E[a], D[a], SAME_NONZERO_PATTERN));
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// Local AIJ blocks of A (diagonal and off-diagonal for MPIAIJ, the matrix
/// itself for SeqAIJ) and the off-diagonal column map; false for other types.
static bool TDSEZAIJBlocks(Mat A, Mat blk[2], const PetscInt **colmap)
{
    PetscBool mpi = PETSC_FALSE, seq = PETSC_FALSE;
    if (PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &mpi) || PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &seq)) return false;
    blk[0] = blk[1] = NULL; *colmap = NULL;
    if (seq) { blk[0] = A; return true; }
    if (!mpi) return false;
    return MatMPIAIJGetSeqAIJ(A, &blk[0], &blk[1], colmap) == PETSC_SUCCESS;
}

/// Whether A and B store the same sparsity structure (row pointers, column
/// indices, off-diagonal column map) on every rank (collective).
static PetscErrorCode TDSEZSameStructure(Mat A, Mat B, bool *same)
{
    PetscFunctionBeginUser;
    Mat a[2], b[2];
    const PetscInt *ca, *cb;
    int ok = TDSEZAIJBlocks(A, a, &ca) && TDSEZAIJBlocks(B, b, &cb);
    for (int k = 0; k < 2 && ok; ++k) {
        if (!a[k] || !b[k]) { ok = !a[k] && !b[k]; continue; }
        PetscInt na, nb;
        const PetscInt *ia, *ja, *ib, *jb;
        PetscBool da, db;
        PetscCall(MatGetRowIJ(a[k], 0, PETSC_FALSE, PETSC_FALSE, &na, &ia, &ja, &da));
        PetscCall(MatGetRowIJ(b[k], 0, PETSC_FALSE, PETSC_FALSE, &nb, &ib, &jb, &db));
        ok = da && db && na == nb && !memcmp(ia, ib, sizeof(PetscInt) * (na + 1)) &&
             !memcmp(ja, jb, sizeof(PetscInt) * ia[na]);
        if (ok && k == 1) {
            PetscInt nc;
            PetscCall(MatGetLocalSize(a[1], NULL, &nc));
            PetscInt ncb;
            PetscCall(MatGetLocalSize(b[1], NULL, &ncb));
            ok = nc == ncb && (!nc || !memcmp(ca, cb, sizeof(PetscInt) * nc));
        }
        PetscCall(MatRestoreRowIJ(a[k], 0, PETSC_FALSE, PETSC_FALSE, &na, &ia, &ja, &da));
        PetscCall(MatRestoreRowIJ(b[k], 0, PETSC_FALSE, PETSC_FALSE, &nb, &ib, &jb, &db));
    }
    int all;
    PetscCallMPI(MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, PetscObjectComm((PetscObject)A)));
    *same = all != 0;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// A = B + sum_i c_i D_i in one pass over the stored values (same structure).
static PetscErrorCode TDSEZFusedCombine(Mat A, Mat B, PetscInt n, const Mat D[], const PetscScalar c[])
{
    PetscFunctionBeginUser;
    Mat a[2], b[2], d[3][2];
    const PetscInt *cm;
    TDSEZAIJBlocks(A, a, &cm);
    TDSEZAIJBlocks(B, b, &cm);
    for (PetscInt i = 0; i < n; ++i) TDSEZAIJBlocks(D[i], d[i], &cm);
    for (int k = 0; k < 2; ++k) {
        if (!a[k]) continue;
        PetscInt nz;
        {
            PetscInt nr; const PetscInt *ia, *ja; PetscBool done;
            PetscCall(MatGetRowIJ(a[k], 0, PETSC_FALSE, PETSC_FALSE, &nr, &ia, &ja, &done));
            nz = ia[nr];
            PetscCall(MatRestoreRowIJ(a[k], 0, PETSC_FALSE, PETSC_FALSE, &nr, &ia, &ja, &done));
        }
        PetscScalar *va;
        const PetscScalar *vb, *vd[3] = {NULL, NULL, NULL};
        PetscCall(MatSeqAIJGetArrayWrite(a[k], &va));
        PetscCall(MatSeqAIJGetArrayRead(b[k], &vb));
        for (PetscInt i = 0; i < n; ++i) PetscCall(MatSeqAIJGetArrayRead(d[i][k], &vd[i]));
        switch (n) {
        case 0: for (PetscInt j = 0; j < nz; ++j) va[j] = vb[j]; break;
        case 1: for (PetscInt j = 0; j < nz; ++j) va[j] = vb[j] + c[0] * vd[0][j]; break;
        case 2: for (PetscInt j = 0; j < nz; ++j) va[j] = vb[j] + c[0] * vd[0][j] + c[1] * vd[1][j]; break;
        default:
            for (PetscInt j = 0; j < nz; ++j) va[j] = vb[j] + c[0] * vd[0][j] + c[1] * vd[1][j] + c[2] * vd[2][j];
        }
        for (PetscInt i = 0; i < n; ++i) PetscCall(MatSeqAIJRestoreArrayRead(d[i][k], &vd[i]));
        PetscCall(MatSeqAIJRestoreArrayRead(b[k], &vb));
        PetscCall(MatSeqAIJRestoreArrayWrite(a[k], &va));
    }
    PetscCall(PetscObjectStateIncrease((PetscObject)A));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// A = (g H + M) + sum_i c_i D_i in one pass over the value arrays (all
/// matrices with the structure of A): the same values, in the same order of
/// operations, as UpdateStageBase followed by TDSEZFusedCombine, without the
/// cached B_k (-tdsez_mat_free).
static PetscErrorCode TDSEZFusedStage(Mat A, Mat M, Mat H, PetscScalar g, PetscInt n, const Mat D[], const PetscScalar c[])
{
    PetscFunctionBeginUser;
    Mat a[2], m[2], h[2], d[3][2];
    const PetscInt *cm;
    TDSEZAIJBlocks(A, a, &cm);
    TDSEZAIJBlocks(M, m, &cm);
    TDSEZAIJBlocks(H, h, &cm);
    for (PetscInt i = 0; i < n; ++i) TDSEZAIJBlocks(D[i], d[i], &cm);
    for (int k = 0; k < 2; ++k) {
        if (!a[k]) continue;
        PetscInt nz;
        {
            PetscInt nr; const PetscInt *ia, *ja; PetscBool done;
            PetscCall(MatGetRowIJ(a[k], 0, PETSC_FALSE, PETSC_FALSE, &nr, &ia, &ja, &done));
            nz = ia[nr];
            PetscCall(MatRestoreRowIJ(a[k], 0, PETSC_FALSE, PETSC_FALSE, &nr, &ia, &ja, &done));
        }
        PetscScalar *va;
        const PetscScalar *vm, *vh, *vd[3] = {NULL, NULL, NULL};
        PetscCall(MatSeqAIJGetArrayWrite(a[k], &va));
        PetscCall(MatSeqAIJGetArrayRead(m[k], &vm));
        PetscCall(MatSeqAIJGetArrayRead(h[k], &vh));
        for (PetscInt i = 0; i < n; ++i) PetscCall(MatSeqAIJGetArrayRead(d[i][k], &vd[i]));
        switch (n) {
        case 0: for (PetscInt j = 0; j < nz; ++j) va[j] = g * vh[j] + vm[j]; break;
        case 1: for (PetscInt j = 0; j < nz; ++j) va[j] = (g * vh[j] + vm[j]) + c[0] * vd[0][j]; break;
        case 2: for (PetscInt j = 0; j < nz; ++j) va[j] = (g * vh[j] + vm[j]) + c[0] * vd[0][j] + c[1] * vd[1][j]; break;
        default:
            for (PetscInt j = 0; j < nz; ++j)
                va[j] = (g * vh[j] + vm[j]) + c[0] * vd[0][j] + c[1] * vd[1][j] + c[2] * vd[2][j];
        }
        for (PetscInt i = 0; i < n; ++i) PetscCall(MatSeqAIJRestoreArrayRead(d[i][k], &vd[i]));
        PetscCall(MatSeqAIJRestoreArrayRead(h[k], &vh));
        PetscCall(MatSeqAIJRestoreArrayRead(m[k], &vm));
        PetscCall(MatSeqAIJRestoreArrayWrite(a[k], &va));
    }
    PetscCall(PetscObjectStateIncrease((PetscObject)A));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Cache B_k = M + g_k H for the step length h. H (including any CAP)
///        is fixed during propagation, so this is recomputed only when h
///        changes (first step, and a shortened final step).
PetscErrorCode TDSEZPropagator::UpdateStageBase(PetscReal h)
{
    PetscFunctionBeginUser;
    if (h == baseH_) PetscFunctionReturn(PETSC_SUCCESS);
    const PetscReal   s3   = PetscSqrtReal(3.0);
    const PetscScalar g[2] = {h * (-s3 + 3.0 * PETSC_i) / 12.0,
                              h * ( s3 + 3.0 * PETSC_i) / 12.0};
    for (int k = 0; k < 2; ++k) {
        if (!Base_[k]) PetscCall(MatDuplicate(manager_.H(), MAT_DO_NOT_COPY_VALUES, &Base_[k]));
        PetscCall(MatCopy(manager_.H(), Base_[k], SAME_NONZERO_PATTERN));
        PetscCall(MatScale(Base_[k], g[k]));
        PetscCall(MatAXPY(Base_[k], 1.0, manager_.M(), SAME_NONZERO_PATTERN));
    }
    baseH_ = h;
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief psi <- R22(-i h M^{-1} H_E) psi with H_E = H + sum_a E_a D_a.
/// @details The two stages are (M + g_k H_E) y = (M + conj(g_k) H_E) x with
///          conj(g_1) = -g_2. The right-hand side of each stage is therefore
///          2 M x - A_other x, where A_other = M + g_other H_E is the other
///          stage matrix, so both stage matrices are built once per call
///          (from the cached field-free parts) and no products with H or D
///          are needed.
/// @param[in]     h    Step length of this exponential.
/// @param[in]     E    Field components used for H_E.
/// @param[in,out] psi  State vector.
PetscErrorCode TDSEZPropagator::Pade22Step(PetscReal h, const PetscScalar E[3], Vec psi, int slot)
{
    PetscFunctionBeginUser;
    const PetscReal   s3   = PetscSqrtReal(3.0);
    const PetscScalar g[2] = {h * (-s3 + 3.0 * PETSC_i) / 12.0,
                              h * ( s3 + 3.0 * PETSC_i) / 12.0};
    const Mat       D[3]  = {manager_.Dx(), manager_.Dy(), manager_.Dz()};
    const PetscBool on[3] = {(PetscBool)TDSEZParser::Coupled(0), (PetscBool)TDSEZParser::Coupled(1),
                             (PetscBool)TDSEZParser::Coupled(2)};   // axes coupled to the field
    Vec rhs = work_[0], Ax = work_[1];

    // A_k = B_k + g_k E.D for both stages.
    PetscCall(PetscLogEventBegin(TDSEZProfile::Get().eventStageUpdate, 0, 0, 0, 0));
    Mat      Dn[3];
    PetscInt n = 0;
    for (int a = 0; a < 3; ++a) if (on[a] && D[a] && E[a] != 0.0) Dn[n++] = D[a];
    if (leanStage_ < 0) {                   // -tdsez_mat_free: no cached B_k if H, M, D share the structure
        bool same = TDSEZMatrixFree();
        for (int a = 0; a < 3 && same; ++a) if (D[a]) PetscCall(TDSEZSameStructure(Astage_[0], D[a], &same));
        if (same) PetscCall(TDSEZSameStructure(Astage_[0], manager_.M(), &same));
        if (same) PetscCall(TDSEZSameStructure(Astage_[0], manager_.H(), &same));
        if (same) PetscCall(TDSEZSameStructure(Astage_[0], Astage_[1], &same));
        leanStage_ = same ? 1 : 0;
    }
    if (leanStage_) {
        for (int k = 0; k < 2; ++k) {
            PetscScalar c[3];
            for (PetscInt i = 0, a = 0; a < 3; ++a) if (on[a] && D[a] && E[a] != 0.0) c[i++] = g[k] * E[a];
            PetscCall(TDSEZFusedStage(Astage_[k], manager_.M(), manager_.H(), g[k], n, Dn, c));
        }
    } else {
    PetscCall(UpdateStageBase(h));
    if (fusedUpdate_ < 0) {                 // once: every matrix in the sum has the stage structure
        bool same = true;
        for (int a = 0; a < 3 && same; ++a) if (D[a]) PetscCall(TDSEZSameStructure(Astage_[0], D[a], &same));
        for (int k = 0; k < 2 && same; ++k) {
            PetscCall(TDSEZSameStructure(Astage_[0], Base_[k], &same));
            if (same) PetscCall(TDSEZSameStructure(Astage_[0], Astage_[k], &same));
        }
        fusedUpdate_ = same ? 1 : 0;
    }
    for (int k = 0; k < 2; ++k) {
        if (fusedUpdate_) {
            PetscScalar c[3];
            for (PetscInt i = 0, a = 0; a < 3; ++a) if (on[a] && D[a] && E[a] != 0.0) c[i++] = g[k] * E[a];
            PetscCall(TDSEZFusedCombine(Astage_[k], Base_[k], n, Dn, c));
            continue;
        }
        PetscCall(MatCopy(Base_[k], Astage_[k], SAME_NONZERO_PATTERN));
        for (int a = 0; a < 3; ++a)
            if (on[a] && D[a] && E[a] != 0.0)
                PetscCall(MatAXPY(Astage_[k], g[k] * E[a], D[a], SAME_NONZERO_PATTERN));
    }
    }
    PetscCall(PetscLogEventEnd(TDSEZProfile::Get().eventStageUpdate, 0, 0, 0, 0));

    for (int k = 0; k < 2; ++k) {
        // rhs = (M + conj(g_k) H_E) x = 2 M x - A_{1-k} x
        PetscCall(MatMult(manager_.M(), psi, rhs));
        PetscCall(MatMult(Astage_[1 - k], psi, Ax));
        PetscCall(VecAXPBY(rhs, -1.0, 2.0, Ax));
        // Initial guess: psi plus the increment y - x of this stage
        // extrapolated from the last steps (polynomial of order guessOrder_;
        // the increment varies smoothly with t, so the first residual is
        // O(h^q) smaller than with psi alone).
        const int s = 2 * slot + k;
        Vec *hist = guessDelta_[s];
        if (extrapolateGuess_) {
            for (PetscInt j = 0; j < guessOrder_; ++j) if (!hist[j]) PetscCall(VecDuplicate(psi, &hist[j]));
            PetscCall(VecCopy(psi, Ax));                     // Ax <- stage input x
            static const PetscScalar w[5][5] = {{1, 0, 0, 0, 0}, {2, -1, 0, 0, 0}, {3, -3, 1, 0, 0},
                                                {4, -6, 4, -1, 0}, {5, -10, 10, -5, 1}};
            const PetscInt q = PetscMin(guessOrder_, guessCount_[s]);
            if (q > 0) PetscCall(VecMAXPY(psi, q, w[q - 1], hist));
        }
        PetscCall(KSPSolve(kspStage_[k], rhs, psi));
        if (extrapolateGuess_) {
            Vec oldest = hist[guessOrder_ - 1];
            for (PetscInt j = guessOrder_ - 1; j > 0; --j) hist[j] = hist[j - 1];
            hist[0] = oldest;
            PetscCall(VecWAXPY(hist[0], -1.0, Ax, psi));     // y - x
            guessCount_[s] = PetscMin(guessCount_[s] + 1, guessOrder_);
        }
        KSPConvergedReason reason;
        PetscInt its = 0;
        PetscCall(KSPGetConvergedReason(kspStage_[k], &reason));
        PetscCall(KSPGetIterationNumber(kspStage_[k], &its));
        // preconditioner refresh: rebuild from the next stage matrix if this
        // solve took far more iterations than after the last build
        PetscBool reuse;
        PetscCall(KSPGetReusePreconditioner(kspStage_[k], &reuse));
        if (!reuse) PetscCall(KSPSetReusePreconditioner(kspStage_[k], PETSC_TRUE));   // just rebuilt
        if (stageRefresh_[k].Stale(its)) PetscCall(KSPSetReusePreconditioner(kspStage_[k], PETSC_FALSE));
        PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
                   "Pade(2,2) stage %d solve did not converge (reason %d)", k + 1, (int)reason);
        nStageSolves_++; nStageIts_ += its;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief Time loop for Propagator = pade22 | magnus4. Calls the same HDF5
///        monitors as the TS path (with a NULL TS) at t = 0 and after every
///        step; the last step is shortened to end exactly at FinalTime.
PetscErrorCode TDSEZPropagator::EvolveRational()
{
    PetscFunctionBeginUser;
    const PetscReal dt   = TDSEZParser::TimeStep;
    const PetscReal endT = TDSEZParser::FinalTime;
    const bool      m4   = (TDSEZParser::Propagator == "magnus4");
    const PetscBool on[3] = {(PetscBool)TDSEZParser::Coupled(0), (PetscBool)TDSEZParser::Coupled(1),
                             (PetscBool)TDSEZParser::Coupled(2)};   // axes coupled to the field
    Vec psi = tdse_.initialPsi;

    PetscErrorCode (*monitor)(TS, PetscInt, PetscReal, Vec, void *) = NULL;
    const PetscInt dim = (PetscInt)TDSEZParser::Dimension;
    switch (dim) {
        case 1: monitor = TDSEZMonitorHDF5_1D; break;
        case 2: monitor = TDSEZMonitorHDF5_2D; break;
        case 3: monitor = TDSEZMonitorHDF5_3D; break;
        default: SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension");
    }
    auto field = [&](PetscReal t, PetscScalar E[3]) {
        // field term of H, s E(t) (s = -Charge), on the coupled axes
        for (int a = 0; a < 3; ++a) E[a] = on[a] ? TDSEZParser::Coupling(a, t) : 0.0;
    };

    // Stage matrices share the Crank-Nicolson Jacobian sparsity (that of H).
    for (int k = 0; k < 2; ++k)
        PetscCall(MatDuplicate(manager_.H(), MAT_DO_NOT_COPY_VALUES, &Astage_[k]));
    for (Vec &v : work_) PetscCall(VecDuplicate(psi, &v));

    // Build both stage matrices for the first step and set up the solvers;
    // their preconditioners are then reused for the whole propagation.
    {
        const PetscReal   h0 = m4 ? 0.5 * PetscMin(dt, endT) : PetscMin(dt, endT);
        const PetscReal   s3 = PetscSqrtReal(3.0);
        const PetscScalar g[2] = {h0 * (-s3 + 3.0 * PETSC_i) / 12.0, h0 * (s3 + 3.0 * PETSC_i) / 12.0};
        PetscScalar E0[3];
        field(0.0, E0);
        const char *prefix[2] = {"pade_stage1_", "pade_stage2_"};
        for (int k = 0; k < 2; ++k) {
            PetscCall(BuildStageMatrix(Astage_[k], g[k], E0));
            PetscCall(KSPCreate(PETSC_COMM_WORLD, &kspStage_[k]));
            PetscCall(TDSEZConfigureStageKSP(kspStage_[k], Astage_[k], prefix[k], tdse_.iga));
        }
    }
    TDSEZInfo().kv("Propagator", "%s: %s, %d solves/step", TDSEZParser::Propagator.c_str(),
        m4 ? "CF Magnus 4, Pade(2,2)" : "Pade(2,2) midpoint", m4 ? 4 : 2);

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-pade_extrapolate_guess", &extrapolateGuess_, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-pade_guess_order", &guessOrder_, NULL));
    PetscCheck(guessOrder_ >= 1 && guessOrder_ <= 5, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "-pade_guess_order must be 1..5");

    // Commutator-free Magnus 4 constants.
    const PetscReal s3 = PetscSqrtReal(3.0);
    const PetscReal c1 = 0.5 - s3 / 6.0, c2 = 0.5 + s3 / 6.0;
    const PetscReal a1 = (3.0 - 2.0 * s3) / 12.0, a2 = (3.0 + 2.0 * s3) / 12.0;

    // Step times t_k = k dt (not accumulated, which drifts over many steps);
    // the last step ends exactly at endT.
    const PetscInt nSteps = (PetscInt)PetscCeilReal(endT / dt - 1e-9);
    PetscReal t = 0.0;
    PetscInt  step = 0;
    PetscCall(monitor(NULL, step, t, psi, &manager_));
    while (step < nSteps) {
        const PetscReal t1 = (step + 1 == nSteps) ? endT : (step + 1) * dt;
        PetscReal h = t1 - step * dt;
        if (PetscAbsReal(h - dt) <= 1e-12 * dt) h = dt;   // keep the cached stage matrices
        t = step * dt;
        if (!m4) {
            PetscScalar Em[3];
            field(t + 0.5 * h, Em);
            PetscCall(Pade22Step(h, Em, psi));
        } else {
            PetscScalar E1[3], E2[3], Ea[3], Eb[3];
            field(t + c1 * h, E1);
            field(t + c2 * h, E2);
            for (int a = 0; a < 3; ++a) {
                Ea[a] = 2.0 * (a2 * E1[a] + a1 * E2[a]);   // applied first
                Eb[a] = 2.0 * (a1 * E1[a] + a2 * E2[a]);
            }
            PetscCall(Pade22Step(0.5 * h, Ea, psi, 0));
            PetscCall(Pade22Step(0.5 * h, Eb, psi, 1));
        }
        t = t1;
        ++step;
        PetscCall(monitor(NULL, step, t, psi, &manager_));
    }
    TDSEZInfo().blank();
    TDSEZInfo().kv("Steps, stage solves", "%" PetscInt_FMT ", %" PetscInt_FMT " (%.2f GMRES its/solve)",
        step, nStageSolves_, nStageSolves_ ? (double)nStageIts_ / (double)nStageSolves_ : 0.0);
    PetscFunctionReturn(PETSC_SUCCESS);
}


// ─────────────────────────────────────────────────────────────────────────────
//  ApplyLzPhysical:  u = M⁻¹ Lz v
// ─────────────────────────────────────────────────────────────────────────────
PetscErrorCode ApplyLzPhysical(TDSEZManager *TDSEZ, KSP ksp_M, Vec psi, Vec u)
{
    PetscFunctionBeginUser;
    Vec tmp;
    PetscCall(VecDuplicate(psi, &tmp));
    PetscCall(MatMult(TDSEZ->Lz(), psi, tmp));
    PetscCall(KSPSolve(ksp_M, tmp, u));
    PetscCall(VecDestroy(&tmp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ─────────────────────────────────────────────────────────────────────────────
//  DiagDenseHermitian:
//    Diagonalise a dense dim×dim Hermitian matrix (row-major) with LAPACK.
//    Returns eigenvalues ascending, eigenvectors as rows of evecs.
//    Phase canonicalized: largest component real-positive.
// ─────────────────────────────────────────────────────────────────────────────
static PetscErrorCode DiagDenseHermitian(
    PetscInt                               dim,
    const std::vector<std::complex<double>> &Amat,
    std::vector<double>                    &evals,
    std::vector<std::vector<std::complex<double>>> &evecs,
    double                                  tol = 1e-12) // degeneracy tolerance
{
    PetscFunctionBeginUser;

    // Step 1: Create PETSc dense matrix
    Mat A;
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, dim, dim,
                                reinterpret_cast<PetscScalar*>(const_cast<std::complex<double>*>(Amat.data())), &A));

    // Step 2: Create SLEPc EPS solver for Hermitian problem
    EPS eps;
    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, A, NULL));
    PetscCall(EPSSetProblemType(eps, EPS_HEP));            // Hermitian
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL)); // deterministic ordering
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));

    PetscInt nconv;
    PetscCall(EPSGetConverged(eps, &nconv));
    if (nconv < dim)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not all eigenvalues converged");

    evals.resize(dim);
    evecs.assign(dim, std::vector<std::complex<double>>(dim));

    Vec xr;
    PetscCall(MatCreateVecs(A, &xr, NULL));

    // Step 3: Extract eigenpairs
    std::vector<std::vector<std::complex<double>>> raw_evecs(dim, std::vector<std::complex<double>>(dim));
    for (PetscInt i = 0; i < dim; i++) {
        PetscScalar kr;
        PetscCall(EPSGetEigenpair(eps, i, &kr, NULL, xr, NULL));
        evals[i] = PetscRealPart(kr);

        const PetscScalar *xdata;
        PetscCall(VecGetArrayRead(xr, &xdata));
        for (PetscInt j = 0; j < dim; j++)
            raw_evecs[i][j] = xdata[j];
        PetscCall(VecRestoreArrayRead(xr, &xdata));
    }

    // Step 4: Identify degenerate blocks and sort deterministically
    std::vector<bool> used(dim,false);
    PetscInt idx = 0;
    while (idx < dim) {
        // Find degenerate block
        PetscInt block_start = idx;
        PetscInt block_end = idx;
        while (block_end+1 < dim && std::abs(evals[block_end+1]-evals[block_start]) < tol)
            block_end++;

        PetscInt block_size = block_end - block_start + 1;

        // Deterministic sort: by max component index in original basis
        std::vector<std::pair<PetscInt,std::complex<double>>> maxidx(block_size);
        for (PetscInt k = 0; k < block_size; k++) {
            PetscInt max_j = 0;
            double max_val = 0.0;
            for (PetscInt j = 0; j < dim; j++) {
                double av = std::abs(raw_evecs[block_start+k][j]);
                if (av > max_val) { max_val = av; max_j = j; }
            }
            maxidx[k] = {max_j, 0.0}; // store max index for sorting
        }

        // sort indices deterministically
        std::vector<PetscInt> order(block_size);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](PetscInt a, PetscInt b){ return maxidx[a].first < maxidx[b].first; });

        // Assign sorted + phase-fixed eigenvectors
        for (PetscInt k = 0; k < block_size; k++) {
            PetscInt src = block_start + order[k];
            evecs[block_start+k] = raw_evecs[src];

            // phase-fix: largest component positive real
            PetscInt max_j = 0;
            double max_val = 0.0;
            for (PetscInt j = 0; j < dim; j++) {
                double av = std::abs(evecs[block_start+k][j]);
                if (av > max_val) { max_val = av; max_j = j; }
            }
            if (max_val > 1e-14) {
                std::complex<double> phase = evecs[block_start+k][max_j] / max_val;
                for (PetscInt j = 0; j < dim; j++) evecs[block_start+k][j] /= phase;
            }
        }

        idx = block_end + 1;
    }

    PetscCall(VecDestroy(&xr));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ─────────────────────────────────────────────────────────────────────────────
//  RotateSubspace:
//    boundstates[grp[k]] <- sum_j evecs[k][j] * boundstates[grp[j]]
//    then M-renormalise each result.
// ─────────────────────────────────────────────────────────────────────────────
static PetscErrorCode RotateSubspace(
    TDSEZManager                                *TDSEZ,
    const std::vector<PetscInt>                 &grp,
    const std::vector<std::vector<PetscScalar>> &evecs,
    Vec                                          Mphi)
{
    PetscFunctionBeginUser;

    PetscCheck(TDSEZ, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "TDSEZ is NULL");
    PetscCheck(Mphi,  PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Mphi is NULL");

    const PetscInt dim = (PetscInt)grp.size();
    if (dim <= 1) PetscFunctionReturn(PETSC_SUCCESS);

    PetscCheck((PetscInt)evecs.size() == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "evecs row count does not match group size");

    std::vector<Vec> tmp(dim, NULL);

    for (PetscInt k = 0; k < dim; ++k) {
        PetscCheck((PetscInt)evecs[k].size() == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                   "evecs[%" PetscInt_FMT "] size does not match group size", k);

        PetscCall(VecDuplicate(TDSEZ->boundstates[grp[0]], &tmp[k]));
        PetscCall(VecZeroEntries(tmp[k]));

        for (PetscInt j = 0; j < dim; ++j) {
            const PetscScalar alpha = evecs[k][j];
            if (alpha != (PetscScalar)0.0) {
                PetscCall(VecAXPY(tmp[k], alpha, TDSEZ->boundstates[grp[j]]));
            }
        }
    }

    for (PetscInt k = 0; k < dim; ++k) {
        Vec vk = TDSEZ->boundstates[grp[k]];

        PetscCall(VecCopy(tmp[k], vk));

        /*
           Modified M-Gram-Schmidt:
           proj = <v_q | M | v_k>, then v_k <- v_k - proj v_q.
           Previously accepted states v_q are already M-normalized.
        */
        for (PetscInt q = 0; q < k; ++q) {
            Vec vq = TDSEZ->boundstates[grp[q]];

            PetscCall(MatMult(TDSEZ->M(), vk, Mphi));

            PetscScalar proj;
            PetscCall(VecDot(Mphi, vq, &proj));

            if (proj != (PetscScalar)0.0) {
                PetscCall(VecAXPY(vk, -proj, vq));
            }
        }

        PetscCall(MatMult(TDSEZ->M(), vk, Mphi));

        PetscScalar norm2;
        PetscCall(VecDot(Mphi, vk, &norm2));

        const PetscReal nrm = PetscRealPart(norm2);

        PetscCheck(nrm > PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
                   "Rotated state %" PetscInt_FMT " has near-zero M-norm", k);

        PetscCall(VecScale(vk, 1.0 / PetscSqrtReal(nrm)));
    }

    for (PetscInt k = 0; k < dim; ++k) {
        PetscCall(VecDestroy(&tmp[k]));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}




// ─────────────────────────────────────────────────────────────────────────────
//  TDSEZOrthogonalizeDegenerates
// ─────────────────────────────────────────────────────────────────────────────
/// @brief Orthogonalize degenerate bound states by diagonalising Lz² within
///        each degenerate energy shell. Performs M-orthogonalization
///        (double-pass Gram-Schmidt), computes Lz² matrix elements in the
///        bound-state subspace, diagonalises each degenerate block, rotates
///        the states into the Lz² eigenbasis, and assigns quantum numbers
///        (nr, m) plus reordered energies.
/// @param TDSEZ  Pointer to the TDSEZManager holding bound states and operators.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZOrthogonalizeDegenerates(TDSEZManager *TDSEZ)
{
    PetscFunctionBeginUser;

    // -------------------------------------------------------------------------
    //  Logging
    // -------------------------------------------------------------------------
    FILE *logfp = nullptr;
    const PetscInt rank = TDSEZ->rank;
    if (rank == 0) 
    {   // Use only the input basename (inputFile may be an absolute path, which
        // would embed slashes and break the "static/GS_..." log path).
        std::string base = std::filesystem::path(TDSEZ->inputFile).filename().string();
        std::error_code ec;
        std::filesystem::create_directories("static", ec); // best-effort
        std::string fname = "static/GS_" + base + ".log";
        logfp = fopen(fname.c_str(), "w");
        if (!logfp)
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN,
                    "Cannot open GS file!");
    }

    // -------------------------------------------------------------------------
    //  Printing helpers  (rank-0 only)
    // -------------------------------------------------------------------------
    //  HR   — heavy unicode rule (matches the rest of the solver output)
    //  Both — stdout + logfile
    //  Log  — logfile only
    // -------------------------------------------------------------------------
    std::string hrule((size_t)TDSEZInfo::INDENT, ' ');
    for (int i = 0; i < TDSEZInfo::width(); ++i) hrule += "═";
    hrule += "\n";
    const char *HR = hrule.c_str();
    // stdout copy shifted to the centred body column (rules stay full width)
    auto Shifted = [](const char *text) {
        const int pad = TDSEZInfo::col() - TDSEZInfo::INDENT;
        std::string out;
        bool start = true;
        for (const char *c = text; *c; ++c) {
            if (start && *c != '\n' && !strstr(text, "═")) out.append((size_t)pad, ' ');
            start = (*c == '\n');
            out += *c;
        }
        fputs(out.c_str(), stdout);
    };
    auto Std = [&](const char *fmt, ...) {
        char buf[2048];
        va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
        Shifted(buf);
    };

    auto Log = [&](const char *fmt, ...) {
        if (rank != 0 || !logfp) return;
        va_list ap; va_start(ap, fmt); vfprintf(logfp, fmt, ap); va_end(ap);
    };
    auto Both = [&](const char *fmt, ...) {
        if (rank != 0) return;
        char buf[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        Shifted(buf);
        if (logfp) fputs(buf, logfp);
    };
    // Section banner: ══ header ══
    auto Section = [&](const char *label) {
        Both("\n%s  ▸ %-66s\n%s\n", HR, label, HR);
    };
    // Sub-step header: plain indented line
    auto Step = [&](const char *fmt, ...) {
        if (rank != 0) return;
        char buf[256];
        va_list ap; va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
        Both("  ┌─ %s\n", buf);
    };

    // -------------------------------------------------------------------------
    //  Single purity tolerance used EVERYWHERE in this function.
    //  Rationale: Lz^2 eigenvalue errors ~1–5% arise from Cartesian B-spline
    //  discretisation of Lz = -i(x∂_y - y∂_x); the tolerance below accepts
    //  states whose <Lz^2> is within 1% of an integer².  Tighten when the
    //  grid has a knot at the origin.
    // -------------------------------------------------------------------------
    const PetscReal LZ2_PURE_TOL  = 1e-2;   // |<Lz^2> - m²| < this → PURE
    const PetscReal LZ2_WARN_TOL  = 0.10;   // above this → MIXED (grid issue)

    auto PurityLabel = [&](PetscReal disp) -> const char * {
        if (disp < LZ2_PURE_TOL)  return "PURE";
        if (disp < LZ2_WARN_TOL)  return "MARGINAL";
        return "MIXED";
    };

    // -------------------------------------------------------------------------
    const PetscInt N = TDSEZ->NPOP;

    // ---- Sanity checks -------------------------------------------------------
    // Lz is assembled ONLY in 2D (see TDSEZAssembler ctor: dim==2). In 3D the
    // degenerate shells are split by the (l,m) angular part, not a single
    // Cartesian Lz, so there is nothing to orthogonalise here. Defensively
    // no-op instead of crashing if called without Lz (the caller in tdsez.cpp
    // already gates on Lz, but this guards every other call site).
    if (!TDSEZ->Lz()) {
        TDSEZInfo().status("ⵣ", "Degenerate Lz orthogonalisation skipped: Lz is not assembled in this dimension/polarisation");
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (!TDSEZ->H())
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
                "TDSEZ->H() is NULL — required for energy re-diagonalisation");
    for (PetscInt i = 0; i < N; i++)
        if (!TDSEZ->boundstates[i])
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
                    "boundstates[%d] is NULL", (int)i);

    // ---- M-solver ------------------------------------------------------------
    KSP ksp_M;
    {
        PC pc;
        PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_M));
        PetscCall(KSPSetOperators(ksp_M, TDSEZ->M(), TDSEZ->M()));
        PetscCall(KSPSetType(ksp_M, KSPCG));
        PetscCall(KSPSetTolerances(ksp_M, 1e-14,
                                   PETSC_DEFAULT, PETSC_DEFAULT, 1000));
        PetscCall(KSPGetPC(ksp_M, &pc));
        PetscCall(PCSetType(pc, PCBJACOBI));
        PetscCall(KSPSetFromOptions(ksp_M));
        PetscCall(KSPSetUp(ksp_M));
    }

    PetscLogDouble t0; PetscCall(PetscTime(&t0));

    // ---- Work vectors --------------------------------------------------------
    Vec Mphi, Hphi, Lzphi;
    PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Mphi));
    PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Hphi));
    PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Lzphi));

    // =========================================================================
    //  STEP A — Global M-GS  +  full-space Lz^2 diagonalisation
    // =========================================================================
    Section("STEP A : global M-GS + full-space Lz^2 diagonalisation");

    // A.1 — Global M-Gram-Schmidt, double pass --------------------------------
    Step("A.1  Global M-Gram-Schmidt (double pass)");
    for (int pass = 0; pass < 2; pass++) {
        for (PetscInt i = 0; i < N; i++) {
            PetscCall(MatMult(TDSEZ->M(), TDSEZ->boundstates[i], Mphi));
            for (PetscInt j = 0; j < i; j++) {
                PetscScalar ov;
                PetscCall(VecDot(Mphi, TDSEZ->boundstates[j], &ov));
                PetscCall(VecAXPY(TDSEZ->boundstates[i], -ov,
                                  TDSEZ->boundstates[j]));
                PetscCall(MatMult(TDSEZ->M(), TDSEZ->boundstates[i], Mphi));
            }
            PetscScalar n2;
            PetscCall(VecDot(Mphi, TDSEZ->boundstates[i], &n2));
            PetscCall(VecScale(TDSEZ->boundstates[i],
                               1.0 / PetscSqrtReal(PetscRealPart(n2))));
        }
    }
    {
        PetscReal max_err = 0.0;
        for (PetscInt i = 0; i < N; i++) {
            PetscCall(MatMult(TDSEZ->M(), TDSEZ->boundstates[i], Mphi));
            for (PetscInt j = 0; j < N; j++) {
                PetscScalar val;
                PetscCall(VecDot(Mphi, TDSEZ->boundstates[j], &val));
                PetscReal err = PetscAbsScalar(val - (i == j ? 1.0 : 0.0));
                if (err > max_err) max_err = err;
            }
        }
        Both("  │   M-ortho after GS : max err = %.2e  %s\n\n",
             max_err, max_err < 1e-10 ? "✔ OK" : "✘ WARNING");
    }

    // A.2 — Build full N×N Lz^2 matrix -----------------------------------------
    Step("A.2  Building %d × %d Lz^2 matrix", (int)N, (int)N);

    std::vector<Vec> Lzphys_all(N);
    for (PetscInt j = 0; j < N; j++) {
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Lzphys_all[j]));
        PetscCall(ApplyLzPhysical(TDSEZ, ksp_M,
                                  TDSEZ->boundstates[j], Lzphys_all[j]));
    }
    std::vector<PetscScalar> Lz2full(N * N, 0.0);
    for (PetscInt kj = 0; kj < N; kj++) {
        PetscCall(MatMult(TDSEZ->M(), Lzphys_all[kj], Mphi));
        for (PetscInt ki = 0; ki < N; ki++) {
            PetscScalar val;
            PetscCall(VecDot(Mphi, Lzphys_all[ki], &val));
            Lz2full[ki * N + kj] = val;
        }
    }
    for (PetscInt j = 0; j < N; j++)
        PetscCall(VecDestroy(&Lzphys_all[j]));

    // A.3 — Diagonalise full Lz^2 matrix ----------------------------------------
    Step("A.3  Diagonalising full Lz^2 matrix");

    std::vector<PetscReal>                lz2_evals_full;
    std::vector<std::vector<PetscScalar>> lz2_evecs_full;
    PetscCall(DiagDenseHermitian(N, Lz2full, lz2_evals_full, lz2_evecs_full));

    Both("  │\n");
    Both("  │   %4s  %+12s  %8s\n", "psi", "λ(Lz^2)", "|m|~");
    Both("  │   ──────────────────────────────\n");
    for (PetscInt k = 0; k < N; k++)
        Both("  │   %4d  %+12.6f  %8.4f\n",
             (int)k, lz2_evals_full[k],
             PetscSqrtReal(PetscAbsReal(lz2_evals_full[k])));
    Both("  │\n");

    // A.4 — Rotate all states into Lz^2 eigenbasis ------------------------------
    {
        std::vector<PetscInt> all_idx(N);
        for (PetscInt k = 0; k < N; k++) all_idx[k] = k;
        PetscCall(RotateSubspace(TDSEZ, all_idx, lz2_evecs_full, Mphi));
    }

    // Verify Lz^2 after rotation — uses THE SAME tolerance as the final check
    Both("  │   Verification after rotation  (tol=%.0e)\n", LZ2_PURE_TOL);
    Both("  │\n");
    Both("  │   %4s  %+12s  %8s  %10s  %s\n",
         "psi", "<Lz^2>", "|m|~", "err", "status");
    Both("  │   ─────────────────────────────────────────────\n");

    PetscInt n_marginal_A = 0;
    for (PetscInt n = 0; n < N; n++) {
        Vec u, Mu;
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &u));
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Mu));
        PetscCall(ApplyLzPhysical(TDSEZ, ksp_M, TDSEZ->boundstates[n], u));
        PetscCall(MatMult(TDSEZ->M(), u, Mu));
        PetscScalar lz2;
        PetscCall(VecDot(Mu, u, &lz2));
        PetscReal lz2_re   = PetscRealPart(lz2);
        PetscReal m_approx = PetscSqrtReal(PetscAbsReal(lz2_re));
        PetscReal nearest  = (PetscReal)std::round((double)m_approx);
        PetscReal disp     = PetscAbsReal(lz2_re - nearest * nearest);
        const char *label  = PurityLabel(disp);

        if (disp >= LZ2_PURE_TOL) n_marginal_A++;

        Both("  │   %4d  %+12.6f  %8.4f  %10.2e  %s\n",
             (int)n, lz2_re, m_approx, disp, label);

        // Store for grouping in Step B
        lz2_evals_full[n] = lz2_re;
        PetscCall(VecDestroy(&u));
        PetscCall(VecDestroy(&Mu));
    }
    Both("  │\n");
    if (n_marginal_A > 0)
        Both("  │   ⚠  %d state(s) MARGINAL/MIXED — Lz discretisation error\n"
             "  │      likely cause: no B-spline knot at origin (even ninterior)\n"
             "  │      fix: use odd ninterior so a knot falls at x=0\n  │\n",
             n_marginal_A);

    // =========================================================================
    //  STEP B — Per-|m|-block H diagonalisation
    // =========================================================================
    Section("STEP B : per-|m|-block H diagonalisation");

    // Group states by rounded |m|
    std::map<PetscInt, std::vector<PetscInt>> m2_map;
    for (PetscInt i = 0; i < N; i++) {
        PetscReal m_approx = PetscSqrtReal(PetscAbsReal(lz2_evals_full[i]));
        PetscInt  m_round  = (PetscInt)std::round((double)m_approx);
        m2_map[m_round * m_round].push_back(i);
    }
    std::vector<std::vector<PetscInt>> lz2_groups;
    for (auto &kv : m2_map) lz2_groups.push_back(kv.second);

    Both("  │   |m| groups\n");
    Both("  │\n");
    Both("  │   %6s  %5s  %s\n", "<Lz^2>~", "dim", "states");
    Both("  │   ─────────────────────────────\n");
    for (auto &grp : lz2_groups) {
        Both("  │   %6.2f  %5d  {", lz2_evals_full[grp[0]], (int)grp.size());
        for (PetscInt k : grp) Both(" %d", (int)k);
        Both(" }\n");
    }
    Both("  │\n");

    for (auto &grp : lz2_groups) {
        PetscInt dim = (PetscInt)grp.size();

        Both("  ├─ |m|-block  <Lz^2>=%.4f  dim=%d\n",
             lz2_evals_full[grp[0]], (int)dim);

        if (dim == 1) {
            PetscCall(MatMult(TDSEZ->H(), TDSEZ->boundstates[grp[0]], Hphi));
            PetscScalar En;
            PetscCall(VecDot(Hphi, TDSEZ->boundstates[grp[0]], &En));
            TDSEZ->energies[grp[0]] = PetscRealPart(En);
            Both("  │   singleton  psi_%d  E = %+.10f\n\n",
                 (int)grp[0], TDSEZ->energies[grp[0]]);
            continue;
        }

        // Build dim×dim H matrix
        std::vector<PetscScalar> Hmat(dim * dim, 0.0);
        for (PetscInt kj = 0; kj < dim; kj++) {
            PetscCall(MatMult(TDSEZ->H(), TDSEZ->boundstates[grp[kj]], Hphi));
            for (PetscInt ki = 0; ki < dim; ki++) {
                PetscScalar val;
                PetscCall(VecDot(Hphi, TDSEZ->boundstates[grp[ki]], &val));
                Hmat[ki * dim + kj] = val;
            }
        }

        // Log detailed H matrix info (file only)
        if (rank == 0) {
            Log("  │   H diagonal: ");
            for (PetscInt k = 0; k < dim; k++)
                Log("%.8f ", PetscRealPart(Hmat[k * dim + k]));
            Log("\n");
            PetscReal offdiag = 0.0;
            for (PetscInt i = 0; i < dim; i++)
            for (PetscInt j = 0; j < dim; j++)
                if (i != j)
                    offdiag += PetscAbsScalar(Hmat[i*dim+j]) *
                               PetscAbsScalar(Hmat[i*dim+j]);
            Log("  │   H off-diag Frobenius = %.2e\n", PetscSqrtReal(offdiag));
        }

        // Diagonalise H block
        std::vector<PetscReal>                H_evals;
        std::vector<std::vector<PetscScalar>> H_evecs;
        PetscCall(DiagDenseHermitian(dim, Hmat, H_evals, H_evecs));

        Both("  │   H eigenvalues:");
        for (PetscInt k = 0; k < dim; k++) Both("  %+.10f", H_evals[k]);
        Both("\n\n");

        PetscCall(RotateSubspace(TDSEZ, grp, H_evecs, Mphi));

        for (PetscInt k = 0; k < dim; k++)
            TDSEZ->energies[grp[k]] = H_evals[k];
    }

    // =========================================================================
    //  STEP C — Lz diagonalisation within energy-degenerate pairs (±m)
    // =========================================================================
    Section("STEP C : Lz diagonalisation within energy-degenerate pairs");

    for (auto &grp : lz2_groups) {
        PetscInt gdim = (PetscInt)grp.size();
        if (gdim == 1) continue;

        const PetscReal E_tol = 1e-5;
        std::vector<bool>                  assigned(gdim, false);
        std::vector<std::vector<PetscInt>> e_pairs;

        for (PetscInt ki = 0; ki < gdim; ki++) {
            if (assigned[ki]) continue;
            std::vector<PetscInt> pair = {ki};
            assigned[ki] = true;
            for (PetscInt kj = ki + 1; kj < gdim; kj++)
                if (!assigned[kj] &&
                    PetscAbsReal(TDSEZ->energies[grp[kj]] -
                                 TDSEZ->energies[grp[ki]]) < E_tol)
                { pair.push_back(kj); assigned[kj] = true; }
            e_pairs.push_back(pair);
        }

        for (auto &pair : e_pairs) {
            PetscInt pdim = (PetscInt)pair.size();
            if (pdim == 1) continue;

            std::vector<PetscScalar> Lzblk(pdim * pdim, 0.0);
            for (PetscInt kj = 0; kj < pdim; kj++) {
                PetscCall(ApplyLzPhysical(TDSEZ, ksp_M,
                                          TDSEZ->boundstates[grp[pair[kj]]],
                                          Lzphi));
                PetscCall(MatMult(TDSEZ->M(), Lzphi, Mphi));
                for (PetscInt ki = 0; ki < pdim; ki++) {
                    PetscScalar val;
                    PetscCall(VecDot(Mphi, TDSEZ->boundstates[grp[pair[ki]]], &val));
                    Lzblk[ki * pdim + kj] = val;
                }
            }

            std::vector<PetscReal>                lz_evals;
            std::vector<std::vector<PetscScalar>> lz_evecs;
            PetscCall(DiagDenseHermitian(pdim, Lzblk, lz_evals, lz_evecs));

            std::vector<PetscInt> pair_grp(pdim);
            for (PetscInt k = 0; k < pdim; k++) pair_grp[k] = grp[pair[k]];
            PetscCall(RotateSubspace(TDSEZ, pair_grp, lz_evecs, Mphi));

            Both("  │   Lz pair {");
            for (PetscInt k : pair_grp) Both(" %d", (int)k);
            Both(" }  →  Lz evals:");
            for (PetscReal v : lz_evals) Both("  %+.6f", v);
            Both("\n");
        }
    }
    Both("  │\n");

    PetscCall(VecDestroy(&Mphi));
    PetscCall(VecDestroy(&Hphi));
    PetscCall(VecDestroy(&Lzphi));

    // =========================================================================
    //  VERIFICATION  — M-orthonormality, Lz^2 purity, energies
    //  Uses the SAME LZ2_PURE_TOL as Step A — no more inconsistency.
    // =========================================================================
    Section("VERIFICATION : M-orthonormality · Lz^2 purity · energies");

    {
        Vec Mphi2, Lzphi2, Lzphys2, Lz2phi2, Hphi2;
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Mphi2));
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Lzphi2));
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Lzphys2));
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Lz2phi2));
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &Hphi2));

        PetscBool ortho_ok = PETSC_TRUE, lz_ok = PETSC_TRUE;
        PetscInt  n_ortho_warn = 0, n_mixed = 0, n_marginal = 0;
        PetscReal max_ortho_err = 0.0, max_disp = 0.0;

        if (rank == 0) {
            Std("  │\n");
            Std("  │   %-6s  %-14s  %7s  %6s  %14s  %10s  %s\n",
                   "state", "<Lz>", "<Lz^2>", "|m|", "E", "E_diff", "purity");
            Std("  │   %s\n",
                   "────────────────────────────────────────────────────────────────────");
        }

        for (PetscInt n = 0; n < N; n++) {

            // M-orthonormality
            PetscCall(MatMult(TDSEZ->M(), TDSEZ->boundstates[n], Mphi2));
            for (PetscInt mi = 0; mi < N; mi++) {
                PetscScalar val;
                PetscCall(VecDot(Mphi2, TDSEZ->boundstates[mi], &val));
                PetscReal err = PetscAbsScalar(val - (mi == n ? 1.0 : 0.0));
                if (err > max_ortho_err) max_ortho_err = err;
                if (err > 1e-10) {
                    ortho_ok = PETSC_FALSE; n_ortho_warn++;
                    if (rank == 0)
                        Std("  │   !! ORTHO  <ψ_%d|M|ψ_%d> = (%+.3e,%+.3e)  err=%.2e\n",
                               (int)mi, (int)n,
                               (double)PetscRealPart(val),
                               (double)PetscImaginaryPart(val), (double)err);
                }
            }

            // <Lz>
            PetscCall(MatMult(TDSEZ->Lz(), TDSEZ->boundstates[n], Lzphi2));
            PetscScalar lz_exp;
            PetscCall(VecDot(Lzphi2, TDSEZ->boundstates[n], &lz_exp));

            // <Lz^2>  — physical inner product
            PetscCall(ApplyLzPhysical(TDSEZ, ksp_M,
                                      TDSEZ->boundstates[n], Lzphys2));
            PetscCall(MatMult(TDSEZ->M(), Lzphys2, Lz2phi2));
            PetscScalar lz2_exp;
            PetscCall(VecDot(Lz2phi2, Lzphys2, &lz2_exp));

            // <H>
            PetscCall(MatMult(TDSEZ->H(), TDSEZ->boundstates[n], Hphi2));
            PetscScalar H_exp;
            PetscCall(VecDot(Hphi2, TDSEZ->boundstates[n], &H_exp));
            PetscReal E_new  = PetscRealPart(H_exp);
            PetscReal E_diff = PetscAbsReal(E_new - TDSEZ->energies[n]);

            PetscReal lz2_re   = PetscRealPart(lz2_exp);
            PetscReal m_approx = PetscSqrtReal(PetscAbsReal(lz2_re));
            PetscReal m_round  = (PetscReal)std::round((double)m_approx);
            PetscReal disp     = PetscAbsReal(lz2_re - m_round * m_round);

            // ── SAME threshold as Step A ──────────────────────────────────────
            const char *plabel = PurityLabel(disp);
            PetscBool   pure   = (disp < LZ2_PURE_TOL);
            // ─────────────────────────────────────────────────────────────────

            if (!pure) {
                lz_ok = PETSC_FALSE;
                if (disp < LZ2_WARN_TOL) n_marginal++;
                else                     n_mixed++;
            }
            if (disp > max_disp) max_disp = disp;

            if (rank == 0)
                Std("  │   psi_%-2d  (%+5.2f,%+5.2f)  %7.3f  %6.3f"
                       "  %+.10f  %.2e  %s\n",
                       (int)n,
                       (double)PetscRealPart(lz_exp),
                       (double)PetscImaginaryPart(lz_exp),
                       (double)lz2_re, (double)m_approx,
                       (double)E_new, (double)E_diff,
                       plabel);
        }

        if (rank == 0) {
            Std("  │\n");
            Std("  │   M-orthonormality : %s  (max err=%.1e, warnings=%d)\n",
                   ortho_ok ? "✔ PASSED" : "✘ FAILED",
                   (double)max_ortho_err, (int)n_ortho_warn);
            Std("  │   Lz^2 purity       : %s  (tol=%.0e, max disp=%.1e,"
                   " marginal=%d, mixed=%d / %d)\n",
                   lz_ok ? "✔ PASSED" : "✘ FAILED",
                   LZ2_PURE_TOL, (double)max_disp,
                   (int)n_marginal, (int)n_mixed, (int)N);
            if (!lz_ok)
                Std("  │\n"
                       "  │   ⚠  Lz^2 impurity is a discretisation artefact.\n"
                       "  │      Use odd ninterior to place a B-spline knot at x=0.\n");
            Std("  │\n");
        }

        PetscCall(VecDestroy(&Mphi2));   PetscCall(VecDestroy(&Lzphi2));
        PetscCall(VecDestroy(&Lzphys2)); PetscCall(VecDestroy(&Lz2phi2));
        PetscCall(VecDestroy(&Hphi2));
    }

    PetscCall(KSPDestroy(&ksp_M));

    // =========================================================================
    //  STEP D — Sort boundstates by (n_r, m)
    // =========================================================================
    Section("STEP D : sort states by (n_r, m)");

    {
        struct StateQN {
            PetscInt  idx;
            PetscInt  m;
            PetscInt  nr;
            PetscReal energy;
        };

        Vec tmpLz;
        PetscCall(VecDuplicate(TDSEZ->boundstates[0], &tmpLz));

        std::vector<StateQN> qns(N);
        for (PetscInt n = 0; n < N; n++) {
            qns[n].idx    = n;
            qns[n].energy = TDSEZ->energies[n];
            PetscCall(MatMult(TDSEZ->Lz(), TDSEZ->boundstates[n], tmpLz));
            PetscScalar lz_val;
            PetscCall(VecDot(tmpLz, TDSEZ->boundstates[n], &lz_val));
            qns[n].m = (PetscInt)std::round((double)PetscRealPart(lz_val));
        }
        PetscCall(VecDestroy(&tmpLz));

        // Assign n_r within each |m| group
        std::map<PetscInt, std::vector<PetscInt>> abs_m_groups;
        for (PetscInt n = 0; n < N; n++)
            abs_m_groups[std::abs((int)qns[n].m)].push_back(n);

        for (auto &kv : abs_m_groups) {
            auto &grp = kv.second;
            std::sort(grp.begin(), grp.end(),
                      [&](PetscInt a, PetscInt b) {
                          return qns[a].energy < qns[b].energy;
                      });
            PetscInt nr = 0, count = 0;
            for (PetscInt idx : grp) {
                qns[idx].nr = nr;
                count++;
                PetscInt mult = (kv.first == 0) ? 1 : 2;
                if (count % mult == 0) nr++;
            }
        }

        // Sort by (n_r, |m|, m)
        std::vector<PetscInt> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](PetscInt a, PetscInt b) {
                      if (qns[a].nr != qns[b].nr)
                          return qns[a].nr < qns[b].nr;
                      PetscInt absa = std::abs((int)qns[a].m);
                      PetscInt absb = std::abs((int)qns[b].m);
                      if (absa != absb) return absa < absb;
                      return qns[a].m < qns[b].m;
                  });

        // Permute in-place
        std::vector<Vec>       new_bs(N);
        std::vector<PetscReal> new_en(N);
        for (PetscInt k = 0; k < N; k++) {
            new_bs[k] = TDSEZ->boundstates[order[k]];
            new_en[k] = TDSEZ->energies[order[k]];
        }
        for (PetscInt k = 0; k < N; k++) {
            TDSEZ->boundstates[k] = new_bs[k];
            TDSEZ->energies[k]    = new_en[k];
        }

        if (rank == 0) {
            Std("  │   %4s  %4s  %4s  %4s  %16s\n",
                   "idx", "n_r", "m", "|m|", "energy");
            Std("  │   %s\n",
                   "──────────────────────────────────────────────");
            for (PetscInt k = 0; k < N; k++) {
                PetscInt orig = order[k];
                Std("  │   %4d  %4d  %+4d  %4d  %+16.10f\n",
                       (int)k, (int)qns[orig].nr, (int)qns[orig].m,
                       (int)std::abs((int)qns[orig].m),
                       (double)TDSEZ->energies[k]);
            }
            Std("  │\n");
        }
    }

    // =========================================================================
    PetscLogDouble t_end; PetscCall(PetscTime(&t_end));
    Both("%s  Total wall time : %.3f s\n%s\n",
         HR, (PetscReal)(t_end - t0), HR);

    if (rank == 0 && logfp) fclose(logfp);
    PetscFunctionReturn(PETSC_SUCCESS);
}






