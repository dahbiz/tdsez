// ============================================================================
//  assembler_class.cpp  —  out-of-line definitions for TDSEZAssembler and its
//  matrix-timing helpers, extracted from the legacy AttoPulse.cpp monolith so
//  the modular `tdsez` target links. The class itself is declared in
//  tdsez_internal.hpp; AttoPulse.cpp (the AttoPulseOld reference binary) keeps
//  its own inline copy and does NOT compile this file, so there is no duplicate
//  symbol.
// ============================================================================

/**
 * @file assembler_class.cpp
 * @brief Out-of-line definitions for TDSEZAssembler and matrix-timing helpers.
 *        Assembles CAP, Hamiltonian, mass, dipole, velocity, and gradient
 *        operators per polarization and dimension.
 * @author TDSEZ Project
 */

#include "tdsez_internal.hpp"
#include "tdsez_quadobs.hpp"

/// @brief Create, assemble, and time a single IGA matrix using the
///        PetIGA form-matrix callback mechanism.
/// @param iga    IGA context.
/// @param M      Pointer to the output Mat (created internally).
/// @param form   IGAFormMatrix callback for the operator.
/// @param name   Label for timing output.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZAssembleMatrixTimed(IGA iga, Mat *M, IGAFormMatrix form, const char *name)
{
    PetscLogDouble t0, t1;
    PetscFunctionBeginUser;
    PetscCall(PetscTime(&t0));
    PetscCall(IGACreateMat(iga, M));
    PetscCall(IGASetFormMatrix(iga, form, PETSC_NULLPTR));
    PetscCall(IGAComputeMatrix(iga, *M));
    PetscCall(PetscTime(&t1));
    TDSEZInfo().kv(name, "%.3f s", (double)(t1 - t0));
    TDSEZProfile::Get().Sub(TDSEZProfile::OperatorAssembly, name, t1 - t0);
    PetscFunctionReturn(0);
}


/// @brief Batch-create and assemble multiple IGA matrices in a single
///        element loop using a TDSEZPhysicsKernel callback, with timing.
/// @param iga    IGA context.
/// @param n      Number of matrices to assemble (1–12).
/// @param mats   Array of pointers to Mat (pre-zeroed, created internally).
/// @param form   Physics kernel callback for multi-operator assembly.
/// @param name   Label for timing output.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZAssembleMatBatchTimed(
    IGA                  iga,
    PetscInt             n,
    Mat                **mats,     // array of pointers to Mat
    TDSEZPhysicsKernel   form,
    const char          *name)
{
    PetscFunctionBegin;
    PetscCheck(iga,  PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IGA is NULL");
    PetscCheck(mats, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "mats is NULL");
    PetscCheck(form, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "form is NULL");
    PetscCheck(n >= 1 && n <= 12, PETSC_COMM_SELF,
               PETSC_ERR_ARG_OUTOFRANGE,
               "n %" PetscInt_FMT " must be in [1,12]", n);

    // Pre-zero through pointers
    for (PetscInt i = 0; i < n; i++) {
        PetscCheck(mats[i], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
                   "mats[%" PetscInt_FMT "] is NULL pointer", i);
        *mats[i] = NULL;
    }

    PetscLogDouble t0, t1;
    PetscCall(PetscTime(&t0));

    // Create via pointer — updates caller's variables directly
    for (PetscInt i = 0; i < n; i++) {
        PetscCall(IGACreateMat(iga, mats[i]));
    }

    // Build flat array for TDSEZCompOperators
    Mat flat[12];
    for (PetscInt i = 0; i < n; i++) flat[i] = *mats[i];
    PetscCall(TDSEZCompOperators(iga, n, flat, form, PETSC_NULLPTR));

    PetscCall(PetscTime(&t1));
    TDSEZInfo().kv(name, "%.3f s", (double)(t1 - t0));
    TDSEZProfile::Get().Sub(TDSEZProfile::OperatorAssembly, name, t1 - t0);

    PetscFunctionReturn(PETSC_SUCCESS);
}



/// @brief Construct the TDSEZAssembler. Assembles all operators needed for
///        propagation and diagnostics: CAP, Hamiltonian (H), mass (M),
///        kinetic (K), potential (V), dipole (Dx/Dy/Dz), velocity
///        (VelX/VelY/VelZ), gradient (dVdx/dVdy/dVdz), mass-dist (Md),
///        and angular momentum (Lz). The per-axis operators are built for every
///        spatial axis (the field couples only the polarised ones).
/// @param iga  IGA context reference for matrix creation.
TDSEZAssembler::TDSEZAssembler(IGA& iga) : iga(iga)
{
    // Read polarization direction
    const std::string& pol = TDSEZParser::Polarization;
    const PetscInt     dim = TDSEZParser::Dimension;

    // assemble core TDSE operators first and time them
    TDSEZInfo info;
    info.header("TIME-DEPENDENT OPERATORS");
    info.section("ASSEMBLY");
    info.kv("Dimension, polarisation", "%dD, %s", (int)dim, pol.c_str());


    if (TDSEZParser::EnableCAP)
    {
        {
            PetscLogDouble t0, t1;
            PetscTime(&t0);
            TDSEZAssembleSeparableCAP(iga, &CAP);
            PetscTime(&t1);
            info.kv("Absorbing CAP", "%.3f s", (double)(t1 - t0));
            TDSEZProfile::Get().Sub(TDSEZProfile::OperatorAssembly, "Absorbing CAP", t1 - t0);
            // -tdsez_cap_check: compare with the element-kernel assembly
            PetscBool check = PETSC_FALSE;
            PetscOptionsGetBool(NULL, NULL, "-tdsez_cap_check", &check, NULL);
            if (check) {
                Mat ref = PETSC_NULLPTR;
                PetscReal diff = 0, scale = 0;
                TDSEZAssembleMatrixTimed(iga, &ref, TDSEZformCap, "CAP (element check)");
                MatNorm(ref, NORM_INFINITY, &scale);
                MatAXPY(ref, -1.0, CAP, SAME_NONZERO_PATTERN);
                MatNorm(ref, NORM_INFINITY, &diff);
                info.kv("CAP check", "max row-sum difference %.3e (relative %.3e)",
                        (double)diff, (double)(diff / PetscMax(scale, (PetscReal)1e-300)));
                MatDestroy(&ref);
            }
        }

    }


    if (dim == 2 && TDSEZParser::EnableLzDiag)
    {
        TDSEZAssembleMatrixTimed(iga, &Lz, TDSEZFormLz, "Angular momentum Lz");
    }

    // Dipole, velocity and force operators on every spatial axis by default
    // (ObserveAllAxes = 1): the field couples only the polarised axes, but the
    // response (dipole, current, acceleration) is recorded on all of them, so
    // cross-polarized emission and asymmetry effects are not lost. With
    // ObserveAllAxes = 0 only the coupled axes are built (less memory). In 2D/3D
    // [K, V, Md] plus [D_d, Vel_d, dV_d] for each axis are built in one batched
    // pass (TDSEZFormPhyAxes, sum-factorised for a constant mass); 1D uses the
    // x kernel.
    if (dim == 1)
    {
        Mat *mats[6] = {&K, &V, &Md, &Dx, &VelX, &dVdx};
        TDSEZAssembleMatBatchTimed(iga, 6, mats, TDSEZFormPhyX, "Axis operators (x)");
    }
    else if (TDSEZMatrixFree())
    {
        // -tdsez_mat_free: the monitors and diagnostics use the factorised and
        // quadrature forms directly (TDSEZManager::SetupMatrixFree). Only the
        // dipoles of the coupled axes are assembled (propagator stage
        // matrices) and, for a varying mass, the velocities (not separable).
        Mat *Dm[3] = {&Dx, &Dy, &Dz}, *Vm[3] = {&VelX, &VelY, &VelZ}, *mats[6];
        PetscInt n = 0;
        std::string dax, vax;
        // dipole matrices requested for saving (-save_dipole / SaveDipoleAxes,
        // SaveDipoleMatrix = x only), as in the save step of the main program
        std::string save = TDSEZParser::SaveDipoleAxes;
        for (auto &c : save) c = (char)std::tolower((unsigned char)c);
        if (save == "all") save = "xyz";
        if (save.empty() && TDSEZParser::SaveDipoleMatrix) save = "x";
        if (save == "none") save.clear();
        for (int d = 0; d < 3; ++d) {
            const bool saved = save.find("xyz"[d]) != std::string::npos;
            TDSEZMatFreeDipOn[d] = (d < dim && (TDSEZParser::Coupled(d) || saved)) ? PETSC_TRUE : PETSC_FALSE;
            if (TDSEZMatFreeDipOn[d]) { mats[n++] = Dm[d]; dax += "xyz"[d]; }
        }
        for (int d = 0; d < 3; ++d) {
            TDSEZMatFreeVelOn[d] = (d < dim && !TDSEZParser::MassIsConstant && TDSEZParser::Observed(d)) ? PETSC_TRUE : PETSC_FALSE;
            if (TDSEZMatFreeVelOn[d]) { mats[n++] = Vm[d]; vax += "xyz"[d]; }
        }
        if (n > 0)
            TDSEZAssembleMatBatchTimed(iga, n, mats, TDSEZFormMatFreeAxes,
                                       ("Dipoles (" + dax + ")" + (vax.empty() ? "" : ", velocities (" + vax + ")")).c_str());
        info.kv("Matrix-free", "K, V, Md, dV, other dipoles%s", TDSEZParser::MassIsConstant ? ", velocities" : "");
    }
    else if (dim == 2 || dim == 3)
    {
        for (int d = 0; d < 3; ++d) TDSEZPhyAxesOn[d] = TDSEZParser::Observed(d) ? PETSC_TRUE : PETSC_FALSE;
        Mat *mats[12];
        PetscInt n = 0;
        mats[n++] = &K; mats[n++] = &V; mats[n++] = &Md;
        Mat *Dm[3] = {&Dx, &Dy, &Dz}, *Vm[3] = {&VelX, &VelY, &VelZ}, *Gm[3] = {&dVdx, &dVdy, &dVdz};
        std::string axes;
        for (int d = 0; d < dim; ++d)
            if (TDSEZPhyAxesOn[d]) { mats[n++] = Dm[d]; mats[n++] = Vm[d]; mats[n++] = Gm[d]; axes += "xyz"[d]; }
        TDSEZAssembleMatBatchTimed(iga, n, mats, TDSEZFormPhyAxes, ("Axis operators (" + axes + ")").c_str());
    }
    else
    {
        throw std::runtime_error("Dimension must be 1, 2 or 3.");
    }

    // Coupling terms beyond the dipoles (FieldTerms, multipoles): G_i assembled
    // once; the propagator adds c_i(t) G_i to the stage matrices with the
    // dipoles. Batches of up to 12 terms share one element loop (sum-factorised
    // in 2D/3D).
    const PetscInt nterm = TDSEZParser::NCouplingTerms();
    G.assign((size_t)nterm, PETSC_NULLPTR);
    for (PetscInt k0 = 0; k0 < nterm; k0 += 12)
    {
        const PetscInt n = PetscMin((PetscInt)12, nterm - k0);
        Mat *mats[12];
        for (PetscInt i = 0; i < n; ++i) mats[i] = &G[(size_t)(k0 + i)];
        TDSEZCouplingIndex = k0;
        const std::string name = n == 1 ? "Coupling term " + std::to_string((int)k0 + 1)
                                        : "Coupling terms " + std::to_string((int)k0 + 1) + "-" + std::to_string((int)(k0 + n));
        TDSEZAssembleMatBatchTimed(iga, n, mats, TDSEZFormCouplingTerms, name.c_str());
    }
    for (PetscInt k = 0; k < nterm; ++k)
        info.kv(k == 0 ? "Coupling terms" : "", "%d: %s", (int)k + 1, TDSEZParser::CouplingTerms_[(size_t)k].label.c_str());

    if (TDSEZParser::EnableCAP)
    {
        // ── CAP absorption-region diagnostic (printed before propagation) ──
        // Mirrors TDSEZformCap: a Manolopoulos layer of width w = c/(2 delta kmin)
        // at both walls of every active axis (separable, W = sum_d W_d).
        const PetscReal cap_width = TDSEZCapWidth();
        const PetscReal lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
        const PetscReal hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
        info.section("COMPLEX ABSORBING POTENTIAL");
        info.line("Manolopoulos layer w = %.4g a.u. per wall (%s); absorbs momenta k >= kmin = %.4g a.u. "
            "(energies >= %.4g a.u.) with little reflection, slower electrons are partly reflected. "
            "A wider layer (CAPWidth) gives a smaller kmin.",
            (double)cap_width, TDSEZParser::CAPWidth > 0 ? "CAPWidth" : "6.555/CAPKmin",
            (double)TDSEZParser::CAPKmin, (double)(0.5 * TDSEZParser::CAPKmin * TDSEZParser::CAPKmin));
        const TDSEZFieldScales &fs = TDSEZEstimateFieldScales();
        for (PetscInt d = 0; d < dim; ++d) {
            const PetscReal core = hi[d] - lo[d] - 2.0 * cap_width;
            char lab[16];
            snprintf(lab, sizeof(lab), "%c axis", "xyz"[d]);
            info.kv(lab, "%c < %.4g and %c > %.4g; free core %.4g a.u.",
                "xyz"[d], (double)(lo[d] + cap_width), "xyz"[d], (double)(hi[d] - cap_width), (double)core);
            // returning electrons must stay in the free core: half width >= 2 alpha + 10
            if (fs.valid && TDSEZParser::Coupled((int)d)) {
                const PetscReal need = 2 * fs.alpha + 10;
                if (0.5 * core < need)
                    info.warn("%c free core half width %.4g a.u. < 2 alpha + 10 = %.4g a.u. (quiver amplitude "
                        "alpha = E0/omega^2 = %.3g): returning electrons enter the absorber; enlarge the domain or "
                        "reduce CAPWidth", "xyz"[d], (double)(0.5 * core), (double)need, (double)fs.alpha);
                else
                    info.kv("", "half core %.4g >= 2 alpha + 10 = %.4g a.u. (alpha %.3g)",
                        (double)(0.5 * core), (double)need, (double)fs.alpha);
            }
        }
    }

    // Dirichlet walls: the wall DoFs are pinned in H and M (core assembly), so
    // they must not couple to the rest of the system through the time-dependent
    // or diagnostic operators either. No-op for Neumann boundaries.
    for (Mat A : {CAP, Lz, K, V, Md, Dx, Dy, Dz, VelX, VelY, VelZ, dVdx, dVdy, dVdz})
        PetscCallAbort(PETSC_COMM_WORLD, TDSEZZeroDirichletRows(A, 0.0));
    for (Mat A : G) PetscCallAbort(PETSC_COMM_WORLD, TDSEZZeroDirichletRows(A, 0.0));
}
