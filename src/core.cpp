#include "tdsez_internal.hpp"

/**
 * @file core.cpp
 * @brief Core TDSEZ setup: IGA configuration, Hamiltonian/mass assembly, and
 *        the generalized eigenproblem solve via SLEPc (Krylov-Schur with
 *        shift-invert spectral transformation).
 * @author TDSEZ Project
 */




/* Laser */
/// @brief Assemble the Hamiltonian (H) and mass (M) matrices via IGA in a
///        single-pass quadrature loop. Handles Dirichlet or Neumann boundary
///        conditions and prints assembly timing plus matrix norm diagnostics.
/// @return PetscErrorCode — PETSC_SUCCESS on success, or a SETERRQ code on
///         invalid dimension or unknown boundary type.
PetscErrorCode TDSEZCore::Assemble()
{
    PetscFunctionBegin;

    if (TDSEZParser::Dimension < 1 || TDSEZParser::Dimension > 3)
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                "Dimension must be 1, 2, or 3.");

    // Destroy previous matrices if reassembling
    if (H) PetscCall(MatDestroy(&H));
    if (M) PetscCall(MatDestroy(&M));

    // Create H; clone sparsity pattern for M (avoids duplicate graph build)
    PetscCall(IGACreateMat(iga, &H));
    PetscCall(MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &M));

    for (Mat mat : {H, M}) {
        PetscCall(MatSetOption(mat, MAT_HERMITIAN, PETSC_TRUE));
        PetscCall(MatSetOption(mat, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE));
    }

    PetscInt nm; PetscCall(MatGetSize(H, &nm, NULL));

    TDSEZInfo info;
    // ── ASSEMBLY SETUP header + FILENAME|DOFs ──
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    info.rule();
    info.center("ASSEMBLY SETUP");
    info.rule();
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "FILENAME: %s  |  DOFs: %" PetscInt_FMT,
                 inputFile.c_str(), nm);
        info.center(buf);
    }
    info.rule();

    // Single-pass assembly: H and M in one quadrature loop
    PetscLogDouble t0, t1;
    PetscCall(PetscTime(&t0));

    Mat mats[2] = {H, M};
    // Boundary condition at the box wall (declarative input option:
    // BoundaryType / Boundary). Neumann (default) = reflecting free Galerkin;
    // Dirichlet = homogeneous psi=0 enforced via IGAElementFixSystem.
    std::string btype = TDSEZParser::BoundaryType;
    for (auto &c : btype) c = (char)std::tolower((unsigned char)c);
    const bool dirichlet =
        (btype == "dirichlet") || (btype == "wall");
    if (dirichlet) {
        PetscCall(TDSEZCompOperatorsDirichlet(iga, 2, mats, TDSEZFormHam, NULL));
    } else if (btype == "neumann" || btype == "natural" || btype.empty()) {
        PetscCall(TDSEZCompOperators(iga, 2, mats, TDSEZFormHam, NULL));
    } else {
        PetscPrintf(PETSC_COMM_WORLD,
            "\nTDSEZ FATAL: unknown BoundaryType '%s' (use Neumann or Dirichlet)\n",
            TDSEZParser::BoundaryType.c_str());
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    PetscCall(PetscTime(&t1));
    // Sanity checks
    PetscReal normH, normM;
    PetscCall(MatNorm(H, NORM_INFINITY, &normH));
    PetscCall(MatNorm(M, NORM_INFINITY, &normM));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "                ||H||_inf = %.3e - ||M||_inf = %.3e\n", normH, normM));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "                          TISE Assembly time: %.3gs\n", t1 - t0));

#ifndef NDEBUG
    {
        Mat Ht;
        PetscCall(MatHermitianTranspose(H, MAT_INITIAL_MATRIX, &Ht));
        PetscCall(MatAXPY(Ht, -1.0, H, SAME_NONZERO_PATTERN));
        PetscReal err; PetscCall(MatNorm(Ht, NORM_FROBENIUS, &err));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  ||H - H†||_F = %.2e  [%s]\n", err, err < 1e-10 ? "OK" : "WARN"));
        PetscCall(MatDestroy(&Ht));
    }
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}





/// @brief Solve the generalized Hermitian eigenproblem H psi = E M psi using
///        SLEPc's Krylov-Schur method with shift-invert spectral transformation.
///        Selects inner solver strategy (direct LU or iterative FGMRES+ILU)
///        based on problem size, extracts bound states, and prepares the
///        initial wavefunction for propagation.
/// @return PetscErrorCode — PETSC_SUCCESS on convergence, or a SLEPc/PETSc
///         error code on failure.
PetscErrorCode TDSEZCore::Solve()
{
    PetscFunctionBegin;

    ST       st;
    KSP      ksp;
    PC       pc;
    PetscInt mat_rows;

    // Output helper (centralised in TDSEZInfo — single source of truth for
    // bar width, indentation, and centering).
    TDSEZInfo info;

    PetscCall(PetscMemorySetGetMaximumUsage());

    // User parameters & matrix metadata
    PetscScalar targetEV       = TDSEZParser::TargetEigenvalue;
    PetscInt    useDirectSolve = TDSEZParser::UseDirectSolve;
    PetscInt    nev            = TDSEZParser::NBoundStates;

    PetscCall(MatGetSize(H, &mat_rows, PETSC_NULLPTR));
    PetscInt nprocs;
    int mpi_nprocs;
    MPI_Comm_size(PETSC_COMM_WORLD, &mpi_nprocs);
    nprocs = mpi_nprocs;

    const PetscBool isHuge   = (mat_rows > 1000000) ? PETSC_TRUE : PETSC_FALSE;
    const PetscBool isLarge  = (!isHuge && mat_rows > 100000) ? PETSC_TRUE : PETSC_FALSE;
    const PetscBool isMedium = (!isHuge && !isLarge && mat_rows > 10000) ? PETSC_TRUE : PETSC_FALSE;
    const PetscBool useDirect = (useDirectSolve == 1 && !isHuge) ? PETSC_TRUE : PETSC_FALSE;

    PetscInt ncv, mpd;
    PetscInt deg_block_est = 7; 
    if (isHuge) {
        ncv = nev + 2*deg_block_est;  
        mpd = nev + deg_block_est;
    } else if (isLarge) {
        ncv = nev + 2*deg_block_est;  
        mpd = nev + deg_block_est;
    } else if (isMedium) {
        ncv = PetscMax(nev + 2*deg_block_est, nev*2);  
        mpd = nev + deg_block_est;
    } else {
        ncv = PetscMax(nev + 2*deg_block_est, PetscMin(nev*4, (PetscInt)200));
        mpd = PetscMax(nev + deg_block_est, ncv / 2);
    }

    // ── update inner_solver merho
    const char* inner_solver =
        useDirect           ? "MUMPS  (direct LU)"                        :
        isHuge              ? "FGMRES(50)  + ASM(1) + ILU(1) + RCM"      :
        isLarge             ? "FGMRES(100) + ASM(1) + ILU(4) + RCM"      :
        isMedium            ? "FGMRES(100) + ASM(2) + ILU(3) + RCM"      :
                              "FGMRES(100) + BJACOBI + ILU(2) + RCM"     ;

    // Problem  info
    info.section("PROBLEM");
    info.blank();
    info.kv("Problem DOFs",    "%" PetscInt_FMT,  mat_rows);
    info.kv("Boundary",        "%s",              TDSEZParser::BoundaryType.c_str());
    // Knot sequence + active domain — print only the line matching the dimension
    auto fmtDom = [](double lo, double hi) -> std::string {
        char b[32];
        snprintf(b, sizeof(b), "[%+.1f, %+.1f]", lo, hi);
        return std::string(b);
    };
    if (TDSEZParser::Dimension >= 3) {
        std::string ks3 = std::string(TDSEZParser::KnotSeq[0]) + ", " + TDSEZParser::KnotSeq[1] + ", " + TDSEZParser::KnotSeq[2];
        info.kv("Knot sequence",  "%s", ks3.c_str());
        std::string dom3 = fmtDom(TDSEZParser::LMinX, TDSEZParser::LMaxX) + "x"
                         + fmtDom(TDSEZParser::LMinY, TDSEZParser::LMaxY) + "x"
                         + fmtDom(TDSEZParser::LMinZ, TDSEZParser::LMaxZ);
        info.kv("ActiveDomain:", "%s", dom3.c_str());
    } else if (TDSEZParser::Dimension >= 2) {
        std::string ks2 = std::string(TDSEZParser::KnotSeq[0]) + ", " + TDSEZParser::KnotSeq[1];
        info.kv("Knot sequence",  "%s", ks2.c_str());
        std::string dom2 = fmtDom(TDSEZParser::LMinX, TDSEZParser::LMaxX) + "x"
                         + fmtDom(TDSEZParser::LMinY, TDSEZParser::LMaxY);
        info.kv("ActiveDomain:", "%s", dom2.c_str());
    } else {
        info.kv("Knot sequence",  "%s", TDSEZParser::KnotSeq[0].c_str());
        info.kv("ActiveDomain:", "%s", fmtDom(TDSEZParser::LMinX, TDSEZParser::LMaxX).c_str());
    }
    info.blank();
    info.kv("MPI ranks",                  "%" PetscInt_FMT,  (PetscInt)nprocs);
    info.kv("DOFs per rank",              "%.3e",            (double)mat_rows / nprocs);
    info.kv("Basis vector footprint",     "%.3e MB",         (double)mat_rows * 16.0 / 1e6);
    info.blank();
    // Initial state info
    if (TDSEZParser::InitialStateMode == 2) {
        std::string desc;
        for (const auto &term : TDSEZParser::InitialStateTerms) {
            if (!desc.empty()) desc += " + ";
            char cbuf[32];
            snprintf(cbuf, sizeof(cbuf), "%.3f*state:%" PetscInt_FMT, term.second, term.first);
            desc += cbuf;
        }
        info.kv("Initial state",  "%s", desc.c_str());
        info.kv("Normalization",  "%s", TDSEZParser::NormalizeInitialState ? "yes" : "no");
    } else {
        info.kv("Initial state",  "state:%" PetscInt_FMT, TDSEZParser::InitialStateIndex);
    }
    info.blank();

    // Solver infp
    info.section("SOLVER CONFIGURATION");
    info.blank();
    info.kv("Eigensolver",        "%s", "Krylov-Schur  (EPSKRYLOVSCHUR)");
    info.kv("Spectral transform", "%s", "Shift-invert  (STSINVERT)");
    info.kv("Inner linear solver","%s", inner_solver);
    info.blank();
    info.kv("Target eigenvalue (sigma)",       "%s a.u.", TDSEZ_fmtAu((double)PetscRealPart(targetEV)).c_str());
    info.kv("Eigenpairs requested (nev)", "%" PetscInt_FMT, nev);
    info.kv("Search subspace dim  (ncv)", "%" PetscInt_FMT, ncv);
    info.kv("Max projected dim    (mpd)", "%" PetscInt_FMT, mpd);
    info.kv("Convergence tolerance",      "%.0e",      1e-10);
    info.kv("Max iterations",             "%" PetscInt_FMT, (PetscInt)1000);
    info.kv("Krylov restart ratio",       "%.2e",      isHuge ? 0.3 : 0.5);
    info.blank();
    info.kv("Subspace RAM estimate", "%.3e GB",
       (double)ncv * mat_rows * 16.0 / 1e9);
    info.blank();

    // Memory estimate before factorization / setup
    info.section("MEMORY  (process 0)");
    info.blank();
    info.memKV("Baseline");

    // EPS creation and setup
    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, H, M));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    // With STSINVERT (shift-invert, below) the spectral transformation maps the
    // original eigenvalue lambda to mu = 1/(lambda - sigma). EPS_SMALLEST_REAL
    // would then return the eigenvalues FARTHEST from the shift (wrong end of the
    // spectrum) and silently miss the true ground state. The correct pairing for
    // "eigenvalues nearest the target" under shift-invert is EPS_TARGET_REAL.
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL));
    PetscCall(EPSSetTarget(eps, targetEV));
    PetscCall(EPSSetTolerances(eps, 1e-15, 8000));
    PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR));
    PetscCall(EPSKrylovSchurSetDimensions(eps, nev, ncv, mpd));
    PetscCall(EPSKrylovSchurSetRestart(eps, isHuge ? 0.3 : 0.5));

    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetShift(st, targetEV));
    PetscCall(STSetMatStructure(st, SAME_NONZERO_PATTERN));
    // PetscCall(STSetMatMode(st, ST_MATMODE_COPY));
    PetscCall(STSetType(st, STSINVERT));
    

    PetscCall(STGetKSP(st, &ksp));
    PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(EPSSetTolerances(eps, 1e-12, 2000));

if (useDirect) {

        // Direct solve via MUMPS or cuSPARSE
        PetscCall(KSPSetType(ksp, KSPPREONLY));
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCLU));

        MatType mtype;
        PetscCall(MatGetType(H, &mtype));

        if (strstr(mtype, "cuda") || strstr(mtype, "cusparse")) {
            PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERCUSPARSE));
        } else {
            PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
            PetscCall(KSPSetOptionsPrefix(ksp, "st_"));
            PetscCall(PCSetOptionsPrefix(pc,  "st_"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR, "-st_mat_mumps_icntl_14", "50"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR, "-st_mat_mumps_icntl_28", "2"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR, "-st_mat_mumps_icntl_29", "2"));
        }

    } else {

        // Iterative solve: strategy depends on problem size 
        //
        //  isHuge  : > 10M DOFs  — FGMRES + ASM + ILU(2)
        //  isLarge :  1M-10M     — FGMRES + ASM + ILU(3)  ← 5M lives here
        //  isMedium: 100k-1M     — FGMRES + ASM + ILU(3)
        //  small   : < 100k      — FGMRES + BJACOBI + ILU(2)
        //
        //  Key fixes vs original:
        //   - dtol 1e8 → 1e3  (stops silent divergence)
        //   - rtol 1e-10 → 1e-6  (shift-invert needs ~1e-6, not 1e-11)
        //   - restart 50 → 100  (5M needs larger Krylov space)
        //   - ILU levels increased for large problems
        //   - sub_ksp_type preonly (no inner iterative, just apply ILU)
        //   - RCM ordering respects B-spline band structure

        PetscCall(KSPSetType(ksp, KSPFGMRES));
        PetscCall(KSPGMRESSetRestart(ksp, isHuge ? 50 : 100));

        // dtol=1e3 catches divergence early instead of spinning
        PetscCall(KSPSetTolerances(ksp,
                  isHuge ? 1e-8 : 1e-10,    // rtol
                  PETSC_DEFAULT,           // abstol
                  1e3,                     // dtol  (was 1e8 — dangerous)
                  isHuge ? 100 : 200));    // maxits

        PetscCall(KSPGetPC(ksp, &pc));

        if (isHuge) {
            // ── > 10M DOFs ─────────────────────────────────────
            // Minimal ILU — just enough to help FGMRES direction
            PetscCall(PCSetType(pc, PCASM));
            PetscCall(PCASMSetOverlap(pc, 1));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_ksp_type",                    "preonly"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_type",                     "ilu"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_levels",            "1"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_mat_ordering_type", "rcm"));

        } else if (isLarge) {
            // ILU(4) + RCM + overlap 1 — strongest iterative option
            // before resorting to direct solve
            // If still not converging: use -use_direct_solve 1
            PetscCall(PCSetType(pc, PCASM));
            PetscCall(PCASMSetOverlap(pc, 1));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_ksp_type",                    "preonly"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_type",                     "ilu"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_levels",            "4"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_mat_ordering_type", "rcm"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_fill",              "20"));

        } else if (isMedium) {
            // 100k–1M DOFs
            PetscCall(PCSetType(pc, PCASM));
            PetscCall(PCASMSetOverlap(pc, 2));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_ksp_type",                    "preonly"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_type",                     "ilu"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_levels",            "3"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_mat_ordering_type", "rcm"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_fill",              "15"));

        } else {
            // < 100k DOFs
            PetscCall(PCSetType(pc, PCBJACOBI));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_ksp_type",                    "preonly"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_type",                     "ilu"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_levels",            "2"));
            PetscCall(PetscOptionsSetValue(PETSC_NULLPTR,
                      "-sub_pc_factor_mat_ordering_type", "rcm"));
        }
    }


    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(STSetFromOptions(st));
    PetscCall(EPSSetFromOptions(eps));


    PetscInt nbound = TDSEZParser::NBoundStates;  // = 21

    PetscCall(EPSMonitorSet(eps,
        [](EPS, PetscInt its, PetscInt nconv,
        PetscScalar *, PetscScalar *,
        PetscReal *errest, PetscInt nest, void* ctx) -> PetscErrorCode
        {
            PetscFunctionBegin;
            PetscInt  nbound = *static_cast<PetscInt*>(ctx);
            static PetscInt prev_nconv = 0;
            PetscInt  delta = nconv - prev_nconv;
            prev_nconv = nconv;
            (void)errest; (void)nest;
            PetscPrintf(PETSC_COMM_WORLD,
                "      EPS  it=%-4d  nconv=%4d/%-7d  (%5.1f%%)  Δconv=%+4d\n",
                (int)its, (int)nconv, (int)nbound,
                100.0 * nconv / nbound,
                (int)delta);
            PetscFunctionReturn(PETSC_SUCCESS);
        },
        &nbound, PETSC_NULLPTR));

    PetscCall(EPSSetUp(eps));
    info.memKV("After factorization / setup");

    if (useDirect) 
    {
        Mat F;
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCFactorGetMatrix(pc, &F));
        PetscCall(MatMumpsSetIcntl(F, 14, 20));
        PetscCall(MatMumpsSetIcntl(F, 28,  2));
        PetscCall(MatMumpsSetIcntl(F, 29,  2));
    }

    // Solve
    info.section("SOLVE");
    info.blank();
    info.status("◆", "Starting EPSSolve ...");
    info.blank();

    PetscCall(EPSSolve(eps));

    info.blank();
    info.memKV("After solve");
    info.blank();

    // ── Results ──────────────────────────────────────────────────────
    PetscCall(EPSGetConverged(eps, &nconv));
    EPSConvergedReason reason;
    PetscInt its;
    PetscCall(EPSGetConvergedReason(eps, &reason));
    PetscCall(EPSGetIterationNumber(eps, &its));

    info.section("RESULTS");
    info.blank();
    info.kv("Outer iterations",    "%" PetscInt_FMT, its);
    info.kv("Converged pairs",     "%" PetscInt_FMT " / %" PetscInt_FMT, nconv, nev);
    info.kv("Stop reason",         "%s",             EPSConvergedReasons[reason]);
    info.blank();

    if (nconv == 0) 
    {
        info.status("ⵀ", "No eigenpairs converged  —  error estimates:");
        info.blank();
        for (PetscInt i = 0; i < PetscMin(nev, (PetscInt)5); i++) 
        {
            PetscReal errest;
            PetscCall(EPSGetErrorEstimate(eps, i, &errest));
            char lbl[48];
            snprintf(lbl, sizeof(lbl), "  pair %" PetscInt_FMT " residual", i);
            info.kv(lbl, "%.3e", (double)errest);
        }
        info.blank();
        info.footer();
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // Print converged eigenvalues
    info.status("ⵣ", "Eigenvalues (converged):");
    info.blank();
    for (PetscInt i = 0; i < nconv && i < nev; i++) {
        PetscScalar eigr;
        PetscReal   errest;
        PetscCall(EPSGetEigenvalue(eps, i, &eigr, PETSC_NULLPTR));
        PetscCall(EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &errest));
        char lbl[48], val[80];
        snprintf(lbl, sizeof(lbl), "  E[%" PetscInt_FMT "]", i);
        snprintf(val, sizeof(val), "%s a.u. | %s eV  (rtol: %.2e)",
                 TDSEZ_fmtAu((double)PetscRealPart(eigr)).c_str(),
                 TDSEZ_fmtEv((double)PetscRealPart(eigr)).c_str(), (double)errest);
        info.kv(lbl, "%s", val);
    }
    info.blank();

    // Extract initial state
    //   mode 0/1 -> single converged eigenstate (ground or state:N)
    //   mode 2   -> coherent superposition of up to 3 bound states
    PetscInt initIdx = TDSEZParser::InitialStateIndex;
    auto buildFromIndex = [&](PetscInt k, Vec *out) -> PetscErrorCode {
        if (k >= nconv)
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                    "Initial state index %" PetscInt_FMT " >= number of converged eigenstates %"
                    PetscInt_FMT " — increase NBoundStates or check solver convergence.",
                    k, nconv);
        PetscCall(IGACreateVec(iga, out));
        PetscCall(EPSGetEigenvector(eps, k, *out, PETSC_NULLPTR));
        PetscFunctionReturn(PETSC_SUCCESS);
    };

    if (TDSEZParser::InitialStateMode == 2) {
        // Assemble psi0 = sum_i coeff_i * eigenvector(idx_i)
        PetscCall(IGACreateVec(iga, &initialPsi));
        PetscCall(VecSet(initialPsi, 0.0));
        std::string desc;
        for (const auto &term : TDSEZParser::InitialStateTerms) {
            PetscInt  k = term.first;
            PetscReal c = term.second;
            Vec evec = PETSC_NULLPTR;
            PetscCall(buildFromIndex(k, &evec));
            PetscCall(VecAXPY(initialPsi, c, evec));
            PetscCall(VecDestroy(&evec));
            if (!desc.empty()) desc += " + ";
            char cbuf[32];
            snprintf(cbuf, sizeof(cbuf), "%.3f*state:%" PetscInt_FMT, c, k);
            desc += cbuf;
        }
        // Normalize so <psi0|M|psi0> = 1 (physical convention for observables)
        if (TDSEZParser::NormalizeInitialState) {
            Vec Mpsi = PETSC_NULLPTR;
            PetscCall(MatCreateVecs(M, &Mpsi, PETSC_NULLPTR));
            PetscCall(MatMult(M, initialPsi, Mpsi));
            PetscScalar nrm2;
            PetscCall(VecDot(Mpsi, initialPsi, &nrm2));
            PetscReal nrm = PetscSqrtReal(PetscRealPart(nrm2));
            if (nrm <= 0.0)
                SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                        "Initial-state superposition has zero M-norm — check coefficients/indices");
            PetscCall(VecScale(initialPsi, 1.0 / nrm));
            PetscCall(VecDestroy(&Mpsi));
            // Verify: recompute <psi|M|psi> after normalization
            PetscScalar chk2;
            PetscCall(MatCreateVecs(M, &Mpsi, PETSC_NULLPTR));
            PetscCall(MatMult(M, initialPsi, Mpsi));
            PetscCall(VecDot(Mpsi, initialPsi, &chk2));
            PetscCall(VecDestroy(&Mpsi));
            info.status("ⵣ", "Superposition normalized (%s)", desc.c_str());
            info.kv("  ||psi||_M (before)", "%.6f", (double)nrm);
            info.kv("  ||psi||_M (after)",  "%.6f", (double)PetscSqrtReal(PetscRealPart(chk2)));
        } else {
            // Print raw M-norm even when not normalizing
            Vec Mpsi = PETSC_NULLPTR;
            PetscCall(MatCreateVecs(M, &Mpsi, PETSC_NULLPTR));
            PetscCall(MatMult(M, initialPsi, Mpsi));
            PetscScalar nrm2;
            PetscCall(VecDot(Mpsi, initialPsi, &nrm2));
            PetscCall(VecDestroy(&Mpsi));
            PetscReal nrm = PetscSqrtReal(PetscRealPart(nrm2));
            info.status("ⵣ", "Superposition assembled (%s, unnormalized)", desc.c_str());
            info.kv("  ||psi||_M", "%.6f", (double)nrm);
        }
        info.blank();
    } else {
        if (initIdx >= nconv)
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                    "Initial state index %" PetscInt_FMT " >= number of converged eigenstates %"
                    PetscInt_FMT " — increase NBoundStates or check solver convergence.",
                    initIdx, nconv);
        Vec epsPsi;
        PetscCall(MatCreateVecs(H, &epsPsi, PETSC_NULLPTR));
        PetscCall(EPSGetEigenvector(eps, initIdx, epsPsi, PETSC_NULLPTR));
        PetscCall(IGACreateVec(iga, &initialPsi));
        PetscCall(VecCopy(epsPsi, initialPsi));
        PetscCall(VecDestroy(&epsPsi));

        if (initIdx == 0)
            info.status("ⵣ", "Ground-state wavefunction extracted  ");
        else
            info.status("ⵣ", "Bound state #%" PetscInt_FMT " extracted   (InitialState=state:%" PetscInt_FMT ")",
                   initIdx, initIdx);
        info.blank();
    }

    // BDD current sanity check: verify mass matrix positive definiteness
    // for the computed eigenstates (relevant when mass is not constant)
    {
        PetscBool conserved = PETSC_TRUE;
        Vec tmp = PETSC_NULLPTR;
        Vec psi_i = PETSC_NULLPTR;
        PetscCall(MatCreateVecs(H, &tmp, PETSC_NULLPTR));
        PetscCall(MatCreateVecs(H, &psi_i, PETSC_NULLPTR));
        for (PetscInt i = 0; i < nconv && i < nev; ++i) {
            PetscCall(EPSGetEigenvector(eps, i, psi_i, PETSC_NULLPTR));
            PetscCall(MatMult(M, psi_i, tmp));
            PetscScalar mnorm;
            PetscCall(VecDot(tmp, psi_i, &mnorm));
            PetscReal mnorm_r = PetscRealPart(mnorm);
            if (mnorm_r <= 0.0) {
                conserved = PETSC_FALSE;
                break;
            }
        }
        PetscCall(VecDestroy(&tmp));
        PetscCall(VecDestroy(&psi_i));

        info.status(conserved ? "✔" : "✖",
               conserved ? "BDD current: CONSERVED" : "BDD current: NOT CONSERVED");
        info.blank();
    }

    info.footer();

    PetscFunctionReturn(PETSC_SUCCESS);
}










/// @brief Output eigenstates and spectrum to HDF5 (EigenData) and/or text
///        formats. Writes the energy spectrum vector and individual eigenstate
///        wavefunctions (psi_0, psi_1, ...) for post-processing tools.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZCore::Output()
{
    PetscFunctionBegin;

    // MPI rank
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // create vectors
    Vec psi;
    IGACreateVec(iga, &psi);

    if (nconv == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "No eigenvalues converged! Exiting...\n");
        VecDestroy(&psi);
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    PetscInt  nsave   = std::min((PetscInt)TDSEZParser::NBoundStates, nconv);
    PetscBool savewfs = TDSEZParser::NBoundStatesSave;
    PetscInt  nsavePop = std::min((PetscInt)TDSEZParser::NBoundStates, nconv);

    // HDF5 setup — only if enabled
    std::string h5file = "static/EigenData_" + std::filesystem::path(inputFile).filename().string() + ".h5";
    PetscViewer viewerH5 = PETSC_NULLPTR;
    {
        if (rank == 0) {
            std::string absPath = std::filesystem::absolute(h5file).string();
            hid_t tf = H5Fcreate(absPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            if (tf >= 0) H5Fclose(tf);
        }
        MPI_Barrier(PETSC_COMM_WORLD);
        PetscViewerHDF5Open(PETSC_COMM_WORLD, h5file.c_str(), FILE_MODE_WRITE, &viewerH5);
        if (TDSEZParser::HDF5Compress) PetscViewerHDF5SetCompress(viewerH5, PETSC_TRUE);
    }

    // Spectrum vector — only needed if saving
    Vec spectrum = PETSC_NULLPTR;
    {
        VecCreate(PETSC_COMM_WORLD, &spectrum);
        VecSetSizes(spectrum, PETSC_DECIDE, nsave);
        VecSetFromOptions(spectrum);
    }

    // Clear stale old vectors (prevent massive leak)
    for (auto &v : boundstates) VecDestroy(&v);
    boundstates.clear();

    for (int i = 0; i < nsave; ++i) {
        PetscScalar kr;
        EPSGetEigenvalue(eps, i, &kr, PETSC_NULLPTR);
        energies.push_back(PetscRealPart(kr));

        VecSetValue(spectrum, i, PetscRealPart(kr), INSERT_VALUES);

        EPSGetEigenvector(eps, i, psi, PETSC_NULLPTR);

        if (savewfs) {
            // Choose what to persist: the raw complex solver state, or the
            // phase-rotated real canonical form (TDSEZMakeStateReal2).
            const bool saveReal = (TDSEZParser::BoundStateFormat == "real");
            Vec psi_save = psi;
            Vec psi_real = PETSC_NULLPTR;
            if (saveReal) {
                IGACreateVec(iga, &psi_real);
                PetscCall(TDSEZMakeStateReal2(psi, psi_real, M));
                psi_save = psi_real;
            }
            std::string name = "psi_" + std::to_string(i);
            PetscObjectSetName((PetscObject)psi_save, name.c_str());
            VecView(psi_save, viewerH5);
            if (psi_real) PetscCall(VecDestroy(&psi_real));
        }

        if (i < nsavePop) {
            Vec psi_copy;
            IGACreateVec(iga, &psi_copy);
            VecCopy(psi, psi_copy);
            boundstates.push_back(psi_copy);
        }
    }

    {
        VecAssemblyBegin(spectrum);
        VecAssemblyEnd(spectrum);
        PetscObjectSetName((PetscObject)spectrum, "spectrum");
        VecView(spectrum, viewerH5);

        // ── Eigenvalue residuals (collective, all ranks) ────────────────
        // EPSComputeError() is a COLLECTIVE SLEPc call (internal MatMult +
        // VecScatter across all ranks). It MUST be invoked by every rank, so
        // compute it here, OUTSIDE any rank==0 guard, before the provenance
        // block below. Only the HDF5 write of the result is rank-0-only.
        PetscInt nres = nconv;
        std::vector<double> errs(nres > 0 ? nres : 1, 0.0);
        for (PetscInt i = 0; i < nres; ++i) {
            PetscReal errest = 0.0;
            PetscCall(EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &errest));
            errs[i] = (double)errest;
        }

        // ── Provenance group (always written, independent of savewfs) ───────
        // Echo parsed input + code version so every output file is self-
        // describing and reproducible. This is the hard-number guard against
        // silently-wrong spectra and "which input produced this?" confusion.
        {
            hid_t h5_file = 0;
            PetscViewerHDF5GetFileId(viewerH5, &h5_file);
            // NOTE: the viewer was opened with PETSC_COMM_WORLD, so the file is
            // collectively-opened. Raw H5 group/attribute/dataset calls on it
            // MUST be performed by ALL ranks (they are collective). Gating them
            // behind rank==0 desyncs HDF5's collective state and deadlocks
            // PetscViewerDestroy at the flush/close barrier on np>=2.
            if (h5_file > 0) {
                hid_t g = H5Gcreate2(h5_file, "run_metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (g < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "HDF5: failed to create run_metadata group");
                auto write_attr = [&](const char *name, const std::string &val) {
                    hid_t st = H5Screate(H5S_SCALAR);
                    hid_t dt = H5Tcopy(H5T_C_S1);
                    H5Tset_size(dt, val.size());
                    hid_t a = H5Acreate2(g, name, dt, st, H5P_DEFAULT, H5P_DEFAULT);
                    if (a >= 0) { H5Awrite(a, dt, val.c_str()); H5Aclose(a); }
                    H5Tclose(dt); H5Sclose(st);
                };
                auto write_real = [&](const char *name, PetscReal v) {
                    hid_t st = H5Screate(H5S_SCALAR);
                    hid_t a = H5Acreate2(g, name, H5T_NATIVE_DOUBLE, st, H5P_DEFAULT, H5P_DEFAULT);
                    if (a >= 0) { H5Awrite(a, H5T_NATIVE_DOUBLE, &v); H5Aclose(a); }
                    H5Sclose(st);
                };
                auto write_int = [&](const char *name, PetscInt v) {
                    hid_t st = H5Screate(H5S_SCALAR);
                    hid_t a = H5Acreate2(g, name, H5T_NATIVE_INT, st, H5P_DEFAULT, H5P_DEFAULT);
                    if (a >= 0) { H5Awrite(a, H5T_NATIVE_INT, &v); H5Aclose(a); }
                    H5Sclose(st);
                };
                write_attr("code_version", TDSEZParser::VersionString());
                write_attr("input_file",   std::filesystem::path(inputFile).filename().string());
                write_attr("units",        "energy: Hartree (a.u.); length: Bohr (a.u.); mass: electron mass");
                write_int ("Dimension",    TDSEZParser::Dimension);
                write_int ("SplineDegree", TDSEZParser::SplineDegree);
                write_int ("Nelements",    TDSEZParser::Nelements);
                write_real("LMinX", TDSEZParser::LMinX); write_real("LMaxX", TDSEZParser::LMaxX);
                write_real("LMinY", TDSEZParser::LMinY); write_real("LMaxY", TDSEZParser::LMaxY);
                write_real("LMinZ", TDSEZParser::LMinZ); write_real("LMaxZ", TDSEZParser::LMaxZ);
                write_real("Hbar",  TDSEZParser::Hbar);
                write_real("Charge",TDSEZParser::Q);
                write_real("TargetEigenvalue", TDSEZParser::TargetEigenvalue);
                write_int ("NBoundStatesSave", (PetscInt)TDSEZParser::NBoundStatesSave);
                write_int ("NBoundStates", TDSEZParser::NBoundStates);
                write_attr("state_format", TDSEZParser::BoundStateFormat);
                // Eigenvalue residual (relative error estimate) per saved state,
                // computed collectively above; written here (rank 0 only).
                if (nres > 0) {
                    hsize_t dims[1] = {(hsize_t)nres};
                    hid_t sp = H5Screate_simple(1, dims, NULL);
                    hid_t ds = H5Dcreate2(g, "eig_residual", H5T_NATIVE_DOUBLE, sp,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    if (ds >= 0) { H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, errs.data()); H5Dclose(ds); }
                    H5Sclose(sp);
                }
                H5Gclose(g);
            }
        }

        // Knot vectors are ALWAYS written — they are cheap and required to
        // reconstruct any downstream quantity. (Previously they were gated on
        // savewfs, which also triggered the much slower WFS-grid evaluation.)
        {
            // Save knot vectors (dimension-aware, read from IGA)
            // Store only unique knot values to avoid large repeated-end blocks
            // Reconstruct full knot vector by repeating:
            //   first multiplicity(p+1), last multiplicity(p+1), interior multiplicity 1
            const char *axis_names[] = {"knots_x", "knots_y", "knots_z"};
            hid_t h5_file = 0;
            PetscViewerHDF5GetFileId(viewerH5, &h5_file);
            // Collective, not rank-gated (see provenance block note above).
            if (h5_file > 0) {
            for (PetscInt d = 0; d < TDSEZParser::Dimension; ++d) {
                IGAAxis axis;
                IGAGetAxis(iga, d, &axis);
                PetscInt   nknots;
                PetscReal *knotvals;
                IGAAxisGetKnots(axis, &nknots, &knotvals);

                // Count multiplicity of first/last unique knot
                PetscInt start_mult = 1;
                for (PetscInt i = 1; i < nknots; ++i) {
                    if (knotvals[i] == knotvals[i-1]) ++start_mult;
                    else break;
                }
                PetscInt end_mult = 1;
                for (PetscInt i = nknots-2; i >= 0; --i) {
                    if (knotvals[i] == knotvals[i+1]) ++end_mult;
                    else break;
                }

                // Save the EXACT PetIGA knot vector (compact form): PetIGA keeps
                // the right end at multiplicity p (not p+1) and drops the final
                // trailing Lmax. Its length is nknots = nfuncs + p, so a reader
                // recovers the standard open vector (length nfuncs + p + 1) by
                // appending one final knot = last value (= Lmax). This compact
                // form is the minimal non-redundant representation; trimming it
                // further to length nfuncs would drop the domain endpoint and
                // break basis/dof alignment, so we keep it as-is.
                const PetscInt pdeg = TDSEZParser::SplineDegree;
                std::vector<PetscReal> compressed(knotvals, knotvals + nknots);
                hsize_t dims[1] = {(hsize_t)compressed.size()};
                // Every HDF5 call below is checked; previously all return codes
                // were silently discarded, so a failed write produced a corrupt /
                // empty dataset with no diagnostic.
                hid_t space = H5Screate_simple(1, dims, NULL);
                if (space < 0)
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                            "HDF5: failed to create dataspace for axis '%s'", axis_names[d]);

                hid_t dset = H5Dcreate2(h5_file, axis_names[d], H5T_NATIVE_DOUBLE, space,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (dset < 0) { H5Sclose(space);
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                            "HDF5: failed to create dataset for axis '%s'", axis_names[d]);
                }
                if (H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, compressed.data()) < 0) {
                    H5Dclose(dset); H5Sclose(space);
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                            "HDF5: failed to write knots for axis '%s'", axis_names[d]);
                }

                hsize_t mdims[1] = {2};
                hid_t mspace = H5Screate_simple(1, mdims, NULL);
                if (mspace < 0) { H5Dclose(dset); H5Sclose(space);
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                            "HDF5: failed to create attribute dataspace for axis '%s'", axis_names[d]);
                }
                hid_t attr = H5Acreate2(dset, "mult_start_end", H5T_NATIVE_INT, mspace,
                                        H5P_DEFAULT, H5P_DEFAULT);
                if (attr < 0) { H5Sclose(mspace); H5Dclose(dset); H5Sclose(space);
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                            "HDF5: failed to create mult_start_end attribute for axis '%s'", axis_names[d]);
                }
                // Record the ACTUAL endpoint multiplicities (PetIGA may keep the
                // right end at multiplicity p, not p+1). A reader uses these plus
                // SplineDegree to rebuild the basis exactly.
                PetscInt mult[2] = {start_mult, end_mult};
                if (H5Awrite(attr, H5T_NATIVE_INT, mult) < 0) {
                    H5Aclose(attr); H5Sclose(mspace); H5Dclose(dset); H5Sclose(space);
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                            "HDF5: failed to write mult_start_end for axis '%s'", axis_names[d]);
                }
                H5Aclose(attr);
                // Spline degree p — required to rebuild the B-spline basis (Cox-de
                // Boor) from the knot vector and map the dof coefficients in
                // psi_*/wavefunction onto basis functions. Without it the stored
                // wavefunctions cannot be evaluated on a spatial grid.
                {
                    hid_t pspace = H5Screate(H5S_SCALAR);
                    hid_t pattr  = H5Acreate2(dset, "SplineDegree", H5T_NATIVE_INT,
                                             pspace, H5P_DEFAULT, H5P_DEFAULT);
                    if (pattr < 0) {
                        H5Sclose(pspace); H5Dclose(dset); H5Sclose(space);
                        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                                "HDF5: failed to create SplineDegree attribute for axis '%s'", axis_names[d]);
                    }
                    PetscInt pval = pdeg;
                    if (H5Awrite(pattr, H5T_NATIVE_INT, &pval) < 0) {
                        H5Aclose(pattr); H5Sclose(pspace); H5Dclose(dset); H5Sclose(space);
                        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                                "HDF5: failed to write SplineDegree for axis '%s'", axis_names[d]);
                    }
                    H5Aclose(pattr);
                    H5Sclose(pspace);
                }
                // Number of basis functions (dof count) = nknots - p. This is the
                // exact size of each psi_*/wavefunction coefficient vector and is
                // what a reader must use as the basis-function count (NOT the
                // generic nknots - p - 1, because PetIGA's knot vector has the
                // right end at multiplicity p).
                {
                    PetscInt nfuncs = (PetscInt)compressed.size() - pdeg;
                    hid_t nspace = H5Screate(H5S_SCALAR);
                    hid_t nattr  = H5Acreate2(dset, "nfuncs", H5T_NATIVE_INT,
                                             nspace, H5P_DEFAULT, H5P_DEFAULT);
                    if (nattr < 0) {
                        H5Sclose(nspace); H5Dclose(dset); H5Sclose(space);
                        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                                "HDF5: failed to create nfuncs attribute for axis '%s'", axis_names[d]);
                    }
                    if (H5Awrite(nattr, H5T_NATIVE_INT, &nfuncs) < 0) {
                        H5Aclose(nattr); H5Sclose(nspace); H5Dclose(dset); H5Sclose(space);
                        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                                "HDF5: failed to write nfuncs for axis '%s'", axis_names[d]);
                    }
                    H5Aclose(nattr);
                    H5Sclose(nspace);
                }
                H5Sclose(mspace);
                H5Dclose(dset);
                H5Sclose(space);
            }
            }
        }

        VecDestroy(&spectrum);
        PetscViewerDestroy(&viewerH5);
    }

    // Cleanup
    VecDestroy(&psi);
    PetscCall(EPSDestroy(&eps));

    PetscFunctionReturn(PETSC_SUCCESS);
}





// Store direct pointers during setup


// ---- TDSEZCore constructor / destructor (out-of-line) ----

        TDSEZCore::TDSEZCore(const std::string &input) : 
            inputFile(input), 
            iga(PETSC_NULLPTR), 
            eps(PETSC_NULLPTR), 
            H(PETSC_NULLPTR),           
            M(PETSC_NULLPTR)
        {
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            MPI_Comm_size(PETSC_COMM_WORLD, &size);

            H5Eset_auto(H5E_DEFAULT, NULL, NULL);

            if (rank == 0) {
                PetscMkdir("td");
                PetscMkdir("static");
                std::string base = std::filesystem::path(inputFile).filename().string();
                std::filesystem::remove("td/TimeEvolutionData_" + base + ".h5");
                std::filesystem::remove("td/wfs_" + base + ".h5");
                std::filesystem::remove("ts_" + base + ".h5");
                std::filesystem::remove("ac_" + base + ".h5");
                std::filesystem::remove("static/EigenData_" + base + ".h5");
            }
            MPI_Barrier(PETSC_COMM_WORLD);

            // Parse the input file (replaces the former PULSE wrapper, which
            // existed only to call PrmReader + initParsers). Validation below
            // relies on the parser state being populated.
            TDSEZParser::PrmReader(input);
            TDSEZParser::initParsers();

            // Validate the parsed input BEFORE touching PETSc/SLEPc. A bad input
            // (non-positive mass, inverted domain, etc.) would otherwise produce
            // silent garbage or a deep solver crash. Catch the exception and abort
            // with a clear message.
            try {
                TDSEZParser::ValidateOrThrow();
            } catch (const std::exception &e) {
                PetscPrintf(PETSC_COMM_WORLD,
                    "\nTDSEZ FATAL: input validation failed:\n  %s\n", e.what());
                MPI_Abort(PETSC_COMM_WORLD, 3);
            }

            // EnableGPU is a complete device request. Configure PetIGA before
            // IGACreate/IGASetFromOptions so every assembled matrix and vector
            // gets the CUDA backend. A host build fails early with an actionable
            // message instead of silently running on CPU.
            if (TDSEZParser::EnableGPU) {
#if defined(PETSC_HAVE_CUDA)
                PetscBool hasMat = PETSC_FALSE, hasVec = PETSC_FALSE;
                PetscCallAbort(PETSC_COMM_WORLD,
                    PetscOptionsHasName(PETSC_NULLPTR, PETSC_NULLPTR,
                                        "-iga_mat_type", &hasMat));
                PetscCallAbort(PETSC_COMM_WORLD,
                    PetscOptionsHasName(PETSC_NULLPTR, PETSC_NULLPTR,
                                        "-iga_vec_type", &hasVec));
                if (!hasMat)
                    PetscCallAbort(PETSC_COMM_WORLD,
                        PetscOptionsSetValue(PETSC_NULLPTR, "-iga_mat_type", "aijcusparse"));
                if (!hasVec)
                    PetscCallAbort(PETSC_COMM_WORLD,
                        PetscOptionsSetValue(PETSC_NULLPTR, "-iga_vec_type", "cuda"));
#else
                PetscPrintf(PETSC_COMM_WORLD,
                    "TDSEZ FATAL: EnableGPU=1 but PETSc was built without CUDA support.\n");
                MPI_Abort(PETSC_COMM_WORLD, 4);
#endif
            }

            // Set domain/IGA options
            // Set domain/IGA options (per-axis intervals collapsed to the X span
            // for the legacy -L option; the knot vectors below use Lmin*/Lmax*).
            std::string domain = std::to_string(TDSEZParser::LMinX) + "," + std::to_string(TDSEZParser::LMaxX);
            IGAOptionsAlias("-L", domain.c_str(), "-iga_limits");
            IGAOptionsAlias("-N", std::to_string(TDSEZParser::Nelements).c_str(), "-iga_elements");
            IGAOptionsAlias("-D", std::to_string(TDSEZParser::SplineDegree).c_str(), "-iga_degree");
            IGAOptionsAlias("-T", std::to_string(TDSEZParser::TargetEigenvalue).c_str(), "-eps_target");
            IGAOptionsAlias("-E", std::to_string(TDSEZParser::NBoundStates).c_str(), "-eps_nev");

            // 3. Create IGA
            IGACreate(PETSC_COMM_WORLD, &iga);
            IGASetDim(iga, TDSEZParser::Dimension);
            IGASetFromOptions(iga);   
            IGASetQuadrature(iga, 0, TDSEZParser::NQuadratures);
            IGASetDof(iga, 1);
            IGASetOrder(iga, TDSEZParser::SplineDegree);

            // ── adaptive_wf two-pass bootstrap ───────────────────────────────
            // If any axis requests KnotSequence = adaptive_wf, run a cheap coarse
            // ground-state solve now to obtain the electron-density marginals
            // rho_x_/rho_y_/rho_z_ that the knot generator below consumes. This
            // must happen BEFORE the knot block. No-op for other sequences.
            {
                bool need_wf = (TDSEZParser::KnotSeq[0] == "adaptive_wf")
                             || (TDSEZParser::KnotSeq[1] == "adaptive_wf")
                             || (TDSEZParser::KnotSeq[2] == "adaptive_wf");
                if (need_wf) {
                    // Wrap in try/catch: BootstrapDensity() runs a coarse solve
                    // and may throw a C++ exception (e.g. TDSEZAdaptiveKnots
                    // hitting an empty PotentialDerivativeX under adaptive_wf).
                    // PetscCallAbort only catches PETSc error codes, not C++
                    // exceptions, so without this the throw escapes to
                    // std::terminate + SIGABRT + MPI abort. Convert to a clean
                    // TDSEZ FATAL instead.
                    PetscErrorCode ierr_b = PETSC_SUCCESS;
                    try {
                        ierr_b = BootstrapDensity();
                    } catch (const std::exception &be) {
                        PetscPrintf(PETSC_COMM_WORLD,
                            "\nTDSEZ FATAL: adaptive_wf bootstrap failed:\n  %s\n",
                            be.what());
                        ierr_b = PETSC_ERR_LIB;
                    }
                    PetscCallAbort(PETSC_COMM_WORLD, ierr_b);
                }
            }

            // Modified LMin/Lmax (offset + per-axis support).
            PetscReal LminX = TDSEZParser::LMinX + TDSEZParser::OffsetX;
            PetscReal LmaxX = TDSEZParser::LMaxX + TDSEZParser::OffsetX;
            PetscReal LminY = TDSEZParser::LMinY + TDSEZParser::OffsetY;
            PetscReal LmaxY = TDSEZParser::LMaxY + TDSEZParser::OffsetY;
            PetscReal LminZ = TDSEZParser::LMinZ + TDSEZParser::OffsetZ;
            PetscReal LmaxZ = TDSEZParser::LMaxZ + TDSEZParser::OffsetZ;

            PetscReal alphax = 0.9, alphay = 0.9, alphaz = 0.9;
            IGAAxis axisx, axisy, axisz;

            // ── KNOT VECTOR SETUP ──────────────────────────────────────────────
            // Wrapped in try/catch so that a throw from any KnotSequence generator
            // (e.g. hydrogenic with r_cross >= Lmax) is reported as a clean
            // TDSEZ FATAL instead of escaping as std::terminate + SIGABRT + MPI
            // abort, matching the input-validation error path above.
            try {

            // ── X-AXIS KNOTS ─────────────────────────────────────────────────────
            {
                const std::string seq = TDSEZParser::KnotSeq[0];
                IGAGetAxis(iga, 0, &axisx);
                // Robustness guard: too few interior knots => basis cannot
                // represent the ground state; warn (not fatal) so the user
                // is not silently handed garbage at N<=~2p.
                {
                    const PetscInt nint = TDSEZParser::Nelements - 1;
                    const PetscInt min_n = 2 * TDSEZParser::SplineDegree + 1;
                    if (nint < min_n && rank == 0)
                        PetscPrintf(PETSC_COMM_WORLD,
                            "TDSEZ: WARNING X axis has only %" PetscInt_FMT
                            " interior knots (< %" PetscInt_FMT " recommended for p=%"
                            PetscInt_FMT "); result may be unreliable.\n",
                            nint, min_n, TDSEZParser::SplineDegree);
                }
                if (seq != "uniform")
                {
                if (rank == 0)
                {
                    if (seq == "symexp")
                        knots_x = TDSEZExpSymKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, alphax, TDSEZParser::SplineDegree);
                    else if (seq == "interfaces")
                        knots_x = TDSEZInterfaceKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, TDSEZParser::SplineDegree, TDSEZParser::Potential, TDSEZParser::Mass);
                    else if (seq == "symtanu")
                        knots_x = TDSEZTanUSymKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, alphax, TDSEZParser::SplineDegree);
                    else if (seq == "symtann")
                        knots_x = TDSEZTanNSymKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, alphax, TDSEZParser::SplineDegree);
                    else if (seq == "symlogtan")
                        knots_x = TDSEZLogTanKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, 0.5, TDSEZParser::SplineDegree);
                    else if (seq == "hydrogenic")
                        knots_x = TDSEZHydrogenicKnots(LmaxX,
                            TDSEZParser::HydrogenicNLin[0] > 0 ? TDSEZParser::HydrogenicNLin[0] : TDSEZParser::Nelements - 1,
                            TDSEZParser::HydrogenicR1[0] > 0.0 ? TDSEZParser::HydrogenicR1[0] : 0.1,
                            TDSEZParser::HydrogenicNExp[0] > 0 ? TDSEZParser::HydrogenicNExp[0] : 10,
                            TDSEZParser::SplineDegree);
                    else if (seq == "adaptive")
                        knots_x = TDSEZAdaptiveKnots(LminX, LmaxX, TDSEZParser::Nelements - 1,
                            TDSEZParser::SplineDegree, 0,
                            TDSEZParser::AdaptiveKappa, TDSEZParser::AdaptivePower);
                    else if (seq == "adaptive_wf")
                        knots_x = TDSEZAdaptiveWFKnots(LminX, LmaxX, TDSEZParser::Nelements - 1,
                            TDSEZParser::SplineDegree, rho_x_);
                    else
                    {
                        PetscPrintf(PETSC_COMM_WORLD,
                            "TDSEZ: WARNING unknown KnotSequence '%s' on X axis — "
                            "falling back to uniform.\n", seq.c_str());
                        IGAAxisInitUniform(axisx, TDSEZParser::Nelements, LminX, LmaxX, TDSEZParser::SplineDegree - 1);
                    }
                }
                // Broadcast the computed knot vector from rank 0 to all ranks.
                // The Bcast MUST be called by EVERY rank (collective) — do NOT
                // gate it behind !knots_x.empty(), because on non-root ranks
                // knots_x is never assigned and stays empty, so they would skip
                // the Bcast while rank 0 blocks inside it => MPI deadlock under
                // -np > 1. The empty case (unknown-sequence fallback, which
                // already called IGAAxisInitUniform above) simply skips
                // IGAAxisSetKnots below.
                {
                    PetscMPIInt nk = (PetscMPIInt)knots_x.size();
                    MPI_Bcast(&nk, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
                    if (rank != 0) knots_x.resize(nk);
                    if (nk > 0)
                        MPI_Bcast(knots_x.data(), nk, MPIU_REAL, 0, PETSC_COMM_WORLD);
                    if (nk > 0)
                        IGAAxisSetKnots(axisx, knots_x.size() - 1, knots_x.data());
                    if (nk > 0 && getenv("TDSEZ_DUMP_KNOTS"))
                    {
                        PetscPrintf(PETSC_COMM_WORLD, "[knots_x interior] ");
                        for (PetscInt i = (PetscInt)TDSEZParser::SplineDegree; i < (PetscInt)knots_x.size() - (PetscInt)TDSEZParser::SplineDegree; ++i)
                            PetscPrintf(PETSC_COMM_WORLD, "%.3f ", (double)knots_x[i]);
                        PetscPrintf(PETSC_COMM_WORLD, "\n");
                    }
                }
                }
                else
                {
                    // Per-axis uniform interval (uses LminX/LmaxX).
                    IGAAxisInitUniform(axisx, TDSEZParser::Nelements, LminX, LmaxX, TDSEZParser::SplineDegree - 1);
                }
            }


            // ── Y-AXIS KNOTS ─────────────────────────────────────────────────────
            if (TDSEZParser::Dimension >= 2)
            {
                const std::string seq = TDSEZParser::KnotSeq[1];
                IGASetQuadrature(iga, 1, TDSEZParser::NQuadratures);
                IGAGetAxis(iga, 1, &axisy);
                if (seq != "uniform")
                {
                    if (rank == 0)
                    {
                        if (seq == "symexp")
                            knots_y = TDSEZExpSymKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, alphay, TDSEZParser::SplineDegree);
                        else if (seq == "interfaces")
                            knots_y = TDSEZInterfaceKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, TDSEZParser::SplineDegree, TDSEZParser::Potential, TDSEZParser::Mass);
                        else if (seq == "symtanu")
                            knots_y = TDSEZTanUSymKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, alphay, TDSEZParser::SplineDegree);
                        else if (seq == "symtann")
                            knots_y = TDSEZTanNSymKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, alphay, TDSEZParser::SplineDegree);
                        else if (seq == "symlogtan")
                            knots_y = TDSEZLogTanKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, 0.5, TDSEZParser::SplineDegree);
                        else if (seq == "hydrogenic")
                            knots_y = TDSEZHydrogenicKnots(LmaxY,
                                TDSEZParser::HydrogenicNLin[1] > 0 ? TDSEZParser::HydrogenicNLin[1] : TDSEZParser::Nelements - 1,
                                TDSEZParser::HydrogenicR1[1] > 0.0 ? TDSEZParser::HydrogenicR1[1] : 0.1,
                                TDSEZParser::HydrogenicNExp[1] > 0 ? TDSEZParser::HydrogenicNExp[1] : 10,
                                TDSEZParser::SplineDegree);
                        else if (seq == "adaptive")
                            knots_y = TDSEZAdaptiveKnots(LminY, LmaxY, TDSEZParser::Nelements - 1,
                                TDSEZParser::SplineDegree, 1,
                                TDSEZParser::AdaptiveKappa, TDSEZParser::AdaptivePower);
                        else if (seq == "adaptive_wf")
                            knots_y = TDSEZAdaptiveWFKnots(LminY, LmaxY, TDSEZParser::Nelements - 1,
                                TDSEZParser::SplineDegree, rho_y_);
                        else
                        {
                            PetscPrintf(PETSC_COMM_WORLD,
                                "TDSEZ: WARNING unknown KnotSequence '%s' on Y axis — "
                                "falling back to uniform.\n", seq.c_str());
                            IGAAxisInitUniform(axisy, TDSEZParser::Nelements, LminY, LmaxY, TDSEZParser::SplineDegree - 1);
                        }
                    }
                    // Broadcast knot vector from rank 0 (collective on ALL ranks;
                    // see X-axis note — never gate the Bcast behind !empty()).
                    {
                        PetscMPIInt nk = (PetscMPIInt)knots_y.size();
                        MPI_Bcast(&nk, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
                        if (rank != 0) knots_y.resize(nk);
                        if (nk > 0)
                            MPI_Bcast(knots_y.data(), nk, MPIU_REAL, 0, PETSC_COMM_WORLD);
                        if (nk > 0)
                            IGAAxisSetKnots(axisy, knots_y.size() - 1, knots_y.data());
                    }
                }
                else
                {
                    // Per-axis uniform interval (uses LminY/LmaxY, not the X span).
                    IGAAxisInitUniform(axisy, TDSEZParser::Nelements, LminY, LmaxY, TDSEZParser::SplineDegree - 1);
                }
            }


            // ── Z-AXIS KNOTS ─────────────────────────────────────────────────────
            if (TDSEZParser::Dimension == 3)
            {
                const std::string seq = TDSEZParser::KnotSeq[2];
                IGASetQuadrature(iga, 2, TDSEZParser::NQuadratures);
                IGAGetAxis(iga, 2, &axisz);
                if (seq != "uniform")
                {
                    if (rank == 0)
                    {
                        if (seq == "symexp")
                            knots_z = TDSEZExpSymKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, alphaz, TDSEZParser::SplineDegree);
                        else if (seq == "interfaces")
                            knots_z = TDSEZInterfaceKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, TDSEZParser::SplineDegree, TDSEZParser::Potential, TDSEZParser::Mass);
                        else if (seq == "symtanu")
                            knots_z = TDSEZTanUSymKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, alphaz, TDSEZParser::SplineDegree);
                        else if (seq == "symtann")
                            knots_z = TDSEZTanNSymKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, alphaz, TDSEZParser::SplineDegree);
                        else if (seq == "hydrogenic")
                            knots_z = TDSEZHydrogenicKnots(LmaxZ,
                                TDSEZParser::HydrogenicNLin[2] > 0 ? TDSEZParser::HydrogenicNLin[2] : TDSEZParser::Nelements - 1,
                                TDSEZParser::HydrogenicR1[2] > 0.0 ? TDSEZParser::HydrogenicR1[2] : 0.1,
                                TDSEZParser::HydrogenicNExp[2] > 0 ? TDSEZParser::HydrogenicNExp[2] : 10,
                                TDSEZParser::SplineDegree);
                        else if (seq == "adaptive")
                            knots_z = TDSEZAdaptiveKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1,
                                TDSEZParser::SplineDegree, 2,
                                TDSEZParser::AdaptiveKappa, TDSEZParser::AdaptivePower);
                        else if (seq == "adaptive_wf")
                            knots_z = TDSEZAdaptiveWFKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1,
                                TDSEZParser::SplineDegree, rho_z_);
                        else
                        {
                            PetscPrintf(PETSC_COMM_WORLD,
                                "TDSEZ: WARNING unknown KnotSequence '%s' on Z axis — "
                                "falling back to uniform.\n", seq.c_str());
                            IGAAxisInitUniform(axisz, TDSEZParser::Nelements, LminZ, LmaxZ, TDSEZParser::SplineDegree - 1);
                        }
                    }
                    // Broadcast knot vector from rank 0 (collective on ALL ranks;
                    // see X-axis note — never gate the Bcast behind !empty()).
                    {
                        PetscMPIInt nk = (PetscMPIInt)knots_z.size();
                        MPI_Bcast(&nk, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
                        if (rank != 0) knots_z.resize(nk);
                        if (nk > 0)
                            MPI_Bcast(knots_z.data(), nk, MPIU_REAL, 0, PETSC_COMM_WORLD);
                        if (nk > 0)
                            IGAAxisSetKnots(axisz, knots_z.size() - 1, knots_z.data());
                    }
                }
                else
                {
                    // Per-axis uniform interval (uses LminZ/LmaxZ, not the X span).
                    IGAAxisInitUniform(axisz, TDSEZParser::Nelements, LminZ, LmaxZ, TDSEZParser::SplineDegree - 1);
                }
            }

            } catch (const std::exception &e) {
                PetscPrintf(PETSC_COMM_WORLD,
                    "\nTDSEZ FATAL: knot-sequence setup failed:\n  %s\n", e.what());
                MPI_Abort(PETSC_COMM_WORLD, 3);
            }

            // Boundary Conditions
            for (PetscInt d = 0; d < TDSEZParser::Dimension; d++) {
                IGASetBoundaryValue(iga, d, 0, 0, 0.0);
                IGASetBoundaryValue(iga, d, 1, 0, 0.0);
            }

            // Final Setup
            IGASetUp(iga);
        }

        TDSEZCore::~TDSEZCore() 
        {
            // MPI_Barrier(PETSC_COMM_WORLD);
            if (iga) IGADestroy(&iga);
            if (eps) EPSDestroy(&eps);
            if (initialPsi) VecDestroy(&initialPsi);
            if (H) MatDestroy(&H);
            if (M) MatDestroy(&M);

            // destroy bound states
            for (auto& vec : boundstates) VecDestroy(&vec);

            boundstates.clear();
        };
// ============================================================================
// adaptive_wf two-pass bootstrap support
// Full implementations saved in dev/core_bootstrap_implementations.cpp for a
// future push. Stubs are no-ops (return success, leave rho_*_ empty).
// ============================================================================

/// @brief Bootstrap a coarse-grid solve to obtain a reference density for
///        adaptive wavefunction-based knot placement.
/// @note Full implementation saved in dev/core_bootstrap_implementations.cpp
///       for a future push. Stub is a no-op (returns PETSC_SUCCESS).
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZCore::BootstrapDensity()
{
    PetscFunctionBegin;
    PetscPrintf(PETSC_COMM_WORLD,
        "  adaptive_wf BOOTSTRAP: implementation pending — skipping coarse solve.\n");
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sample |psi|^2 and marginalise to 1D per-axis density histograms.
/// @note Full implementation saved in dev/core_bootstrap_implementations.cpp
///       for a future push. Stub is a no-op.
/// @param iga  IGA context used for volume evaluation.
/// @param psi  Wavefunction vector to sample.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZCore::SampleDensity(IGA iga, Vec psi)
{
    PetscFunctionBegin;
    (void)iga; (void)psi;
    PetscFunctionReturn(PETSC_SUCCESS);
}
