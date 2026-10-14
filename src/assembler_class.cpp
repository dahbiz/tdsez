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
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "      %-20s : %7.4f sec\n", name, t1 - t0));
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
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "      %-20s : %7.4f sec\n", name, t1 - t0));

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
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");
    PetscPrintf(PETSC_COMM_WORLD, "      TD Operator Assembly Timing (%dD): %s \n", (int)dim, pol.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");


    if (TDSEZParser::EnableCAP)
    {
        TDSEZAssembleMatrixTimed(iga, &CAP, TDSEZformCap, "Absorbing CAP");

        // ── CAP absorption-region diagnostic (printed before propagation) ──
        // Mirrors TDSEZformCap: a Manolopoulos layer of width w = c/(2 delta kmin)
        // at both walls of every active axis (separable, W = sum_d W_d).
        const PetscReal cap_width = TDSEZCapWidth();
        const PetscReal lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
        const PetscReal hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
        PetscPrintf(PETSC_COMM_WORLD,
            "  ══════════════════════════════════════════════════════════════════════\n"
            "    ▸ COMPLEX ABSORBING POTENTIAL (CAP)\n"
            "  ══════════════════════════════════════════════════════════════════════\n");
        PetscPrintf(PETSC_COMM_WORLD,
            "      Manolopoulos layer w = %12.6f   (c/(2*delta*kmin), kmin=%g, Emin=%g)\n",
            (double)cap_width, (double)TDSEZParser::CAPKmin,
            (double)(0.5 * TDSEZParser::CAPKmin * TDSEZParser::CAPKmin));
        for (PetscInt d = 0; d < dim; ++d)
            PetscPrintf(PETSC_COMM_WORLD,
                "      %c: absorbing for %c < %.6f and %c > %.6f  (free core %.6f a.u.)\n",
                "xyz"[d], "xyz"[d], (double)(lo[d] + cap_width), "xyz"[d], (double)(hi[d] - cap_width),
                (double)(hi[d] - lo[d] - 2.0 * cap_width));
        PetscPrintf(PETSC_COMM_WORLD,
            "  ══════════════════════════════════════════════════════════════════════\n");
    }


    if (dim == 2 && TDSEZParser::EnableLzDiag)
    {
        TDSEZAssembleMatrixTimed(iga, &Lz, TDSEZFormLz, "Angular Momentum Lz");
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
        TDSEZAssembleMatBatchTimed(iga, 6, mats, TDSEZFormPhyX, "Axis Ops (x)");
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
        TDSEZAssembleMatBatchTimed(iga, n, mats, TDSEZFormPhyAxes, ("Axis Ops (" + axes + ")").c_str());
    }
    else
    {
        throw std::runtime_error("Dimension must be 1, 2 or 3.");
    }

    // Dirichlet walls: the wall DoFs are pinned in H and M (core assembly), so
    // they must not couple to the rest of the system through the time-dependent
    // or diagnostic operators either. No-op for Neumann boundaries.
    for (Mat A : {CAP, Lz, K, V, Md, Dx, Dy, Dz, VelX, VelY, VelZ, dVdx, dVdy, dVdz})
        PetscCallAbort(PETSC_COMM_WORLD, TDSEZZeroDirichletRows(A, 0.0));
}
