#include "tdsez_internal.hpp"

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
///        matrices (Ht = copy of H, J_ = sparsity clone of H).
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
    // create work vectors and copy matrices
    Mat Ht = PETSC_NULLPTR;
    MatDuplicate(manager.H(), MAT_COPY_VALUES, &Ht);  // Ht = H0
    manager.setHt(Ht);
    MatDuplicate(manager.H(), MAT_DO_NOT_COPY_VALUES, &J_);

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

            // Adaptive ILU: stronger fill on larger subdomains,
            // zero-fill keeps overhead low on small subdomain problems.
            PetscInt local_rows = 0;
            PetscCall(MatGetLocalSize(J_, &local_rows, NULL));
            PetscCall(PCFactorSetLevels(subpc, (local_rows > 4000) ? 1 : 0));  CHKERRQ(ierr);

            // Lock memory structures now that the drop threshold is gone
            ierr = PCFactorSetReuseOrdering(subpc, PETSC_TRUE);                CHKERRQ(ierr);
            ierr = PCFactorSetReuseFill(subpc, PETSC_TRUE);                    CHKERRQ(ierr);

            // Numerical stability pivots for complex wave propagation
            ierr = PCFactorSetShiftType(subpc, MAT_SHIFT_INBLOCKS);            CHKERRQ(ierr);
            ierr = PCFactorSetShiftAmount(subpc, 1e-12);                        CHKERRQ(ierr);

            // Subdomain internal solution constraints
            ierr = KSPSetReusePreconditioner(subksp[i], PETSC_TRUE);           CHKERRQ(ierr);
            ierr = KSPSetTolerances(subksp[i], 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1000); CHKERRQ(ierr);
            ierr = KSPSetType(subksp[i], KSPPREONLY); CHKERRQ(ierr);
        }
    }

    ierr = PCSetUp(pc);  CHKERRQ(ierr);
    ierr = TSSetUp(ts_); CHKERRQ(ierr);

    // ---------------- Solve ----------------
    ierr = TSSolve(ts_, tdse_.initialPsi); CHKERRQ(ierr);

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
    const char *HR =
        "  ══════════════════════════════════════════════════════════════════════\n";

    auto Log = [&](const char *fmt, ...) {
        if (rank != 0 || !logfp) return;
        va_list ap; va_start(ap, fmt); vfprintf(logfp, fmt, ap); va_end(ap);
    };
    auto Both = [&](const char *fmt, ...) {
        if (rank != 0) return;
        va_list ap, ap2;
        va_start(ap, fmt);
        va_copy(ap2, ap);
        vprintf(fmt, ap);   va_end(ap);
        if (logfp) { vfprintf(logfp, fmt, ap2); }
        va_end(ap2);
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
        PetscPrintf(PETSC_COMM_WORLD,
            "  (skipping degenerate Lz-orthogonalisation: Lz not assembled in this dimension/polarization)\n");
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
            printf("  │\n");
            printf("  │   %-6s  %-14s  %7s  %6s  %14s  %10s  %s\n",
                   "state", "<Lz>", "<Lz^2>", "|m|", "E", "E_diff", "purity");
            printf("  │   %s\n",
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
                        printf("  │   !! ORTHO  <ψ_%d|M|ψ_%d> = (%+.3e,%+.3e)  err=%.2e\n",
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
                printf("  │   psi_%-2d  (%+5.2f,%+5.2f)  %7.3f  %6.3f"
                       "  %+.10f  %.2e  %s\n",
                       (int)n,
                       (double)PetscRealPart(lz_exp),
                       (double)PetscImaginaryPart(lz_exp),
                       (double)lz2_re, (double)m_approx,
                       (double)E_new, (double)E_diff,
                       plabel);
        }

        if (rank == 0) {
            printf("  │\n");
            printf("  │   M-orthonormality : %s  (max err=%.1e, warnings=%d)\n",
                   ortho_ok ? "✔ PASSED" : "✘ FAILED",
                   (double)max_ortho_err, (int)n_ortho_warn);
            printf("  │   Lz^2 purity       : %s  (tol=%.0e, max disp=%.1e,"
                   " marginal=%d, mixed=%d / %d)\n",
                   lz_ok ? "✔ PASSED" : "✘ FAILED",
                   LZ2_PURE_TOL, (double)max_disp,
                   (int)n_marginal, (int)n_mixed, (int)N);
            if (!lz_ok)
                printf("  │\n"
                       "  │   ⚠  Lz^2 impurity is a discretisation artefact.\n"
                       "  │      Use odd ninterior to place a B-spline knot at x=0.\n");
            printf("  │\n");
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
            printf("  │   %4s  %4s  %4s  %4s  %16s\n",
                   "idx", "n_r", "m", "|m|", "energy");
            printf("  │   %s\n",
                   "──────────────────────────────────────────────");
            for (PetscInt k = 0; k < N; k++) {
                PetscInt orig = order[k];
                printf("  │   %4d  %4d  %+4d  %4d  %+16.10f\n",
                       (int)k, (int)qns[orig].nr, (int)qns[orig].m,
                       (int)std::abs((int)qns[orig].m),
                       (double)TDSEZ->energies[k]);
            }
            printf("  │\n");
        }
    }

    // =========================================================================
    PetscLogDouble t_end; PetscCall(PetscTime(&t_end));
    Both("%s  Total wall time : %.3f s\n%s\n",
         HR, (PetscReal)(t_end - t0), HR);

    if (rank == 0 && logfp) fclose(logfp);
    PetscFunctionReturn(PETSC_SUCCESS);
}






