// ============================================================================
//  assembler_class.cpp  —  out-of-line definitions for TDSEZAssembler and its
//  matrix-timing helpers, extracted from the legacy AttoPulse.cpp monolith so
//  the modular `tdsez` target links. The class itself is declared in
//  tdsez_internal.hpp; AttoPulse.cpp (the AttoPulseOld reference binary) keeps
//  its own inline copy and does NOT compile this file, so there is no duplicate
//  symbol.
// ============================================================================

#include "tdsez_internal.hpp"

// General function to create, assemble, and time a matrix
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



TDSEZAssembler::TDSEZAssembler(IGA& iga) : iga(iga)
{
    // Read polarization direction
    const std::string& pol = TDSEZParser::Polarization;
    const PetscInt     dim = TDSEZParser::Dimension;

    // assemble core TDSE operators first and time them
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");
    PetscPrintf(PETSC_COMM_WORLD, "      TD Operator Assembly Timing (%dD): %s \n", dim, pol.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");


    // TDSEZAssembleMatrixTimed(iga, &K, TDSEZFormKinetic, "Kinetic Term");
    // TDSEZAssembleMatrixTimed(iga, &V, TDSEZFormPotential, "Potential Term");
    if (TDSEZParser::EnableCAP)
    {
        TDSEZAssembleMatrixTimed(iga, &CAP, TDSEZformCap, "Absorbing CAP");

        // ── CAP absorption-region diagnostic (printed before propagation) ──
        // Mirrors the geometry used inside TDSEZformCap / Manolopoulos_CAP_profile:
        //   R_outer = |LMaxX|, R_inner = 0.90*R_outer, activated for r = max(|x|,|y|,|z|) > R_inner.
        //   Manolopoulos absorb-region width w = c / (2*delta*kmin), c=2.62206, delta=0.2.
        const PetscReal cap_c     = 2.62206;
        const PetscReal cap_delta = 0.2;
        const PetscReal R_outer  = PetscAbsReal(TDSEZParser::LMaxX);
        const PetscReal R_inner  = 0.90 * R_outer;
        const PetscReal cap_width = (TDSEZParser::CAPKmin > 0.0)
                                  ? cap_c / (2.0 * cap_delta * TDSEZParser::CAPKmin)
                                  : PETSC_INFINITY;
        const PetscBool cap_width_inf = (TDSEZParser::CAPKmin <= 0.0);
        PetscPrintf(PETSC_COMM_WORLD,
            "  ══════════════════════════════════════════════════════════════════════\n"
            "    ▸ COMPLEX ABSORBING POTENTIAL (CAP) ACTIVATION REGION\n"
            "  ══════════════════════════════════════════════════════════════════════\n");
        PetscPrintf(PETSC_COMM_WORLD,
            "      dimension            = %dD\n", dim);
        PetscPrintf(PETSC_COMM_WORLD,
            "      R_outer (|LMaxX|)     = %12.6f   (CAP off outside this radius)\n", R_outer);
        PetscPrintf(PETSC_COMM_WORLD,
            "      R_inner (0.90*R_out)  = %12.6f   (CAP on for r = max(|x|,|y|,|z|) > R_inner)\n", R_inner);
        if (cap_width_inf)
            PetscPrintf(PETSC_COMM_WORLD,
                "      Manolopoulos width w = INFINITE  (CAPKmin<=0 -> absorbs ENTIRE domain; divergence risk)\n");
        else
            PetscPrintf(PETSC_COMM_WORLD,
                "      Manolopoulos width w = %12.6f   (c/(2*delta*kmin), kmin=%g)\n", cap_width, (double)TDSEZParser::CAPKmin);
        PetscPrintf(PETSC_COMM_WORLD,
            "      active shell         = [%.6f, %.6f]  (width in r = %.6f)\n", R_inner, R_outer, R_outer - R_inner);
        PetscPrintf(PETSC_COMM_WORLD,
            "  ══════════════════════════════════════════════════════════════════════\n");
    }

    // TDSEZAssembleMatrixTimed(iga, &Md, TDSEZFormMassDist, "Mass Distribution");

    if (dim == 2 && TDSEZParser::EnableLzDiag)
    {
        TDSEZAssembleMatrixTimed(iga, &Lz, TDSEZFormLz, "Angular Momentum Lz");
    }

    if (pol != "x")
    {
        TDSEZAssembleMatrixTimed(iga, &K, TDSEZFormKinetic, "Kinetic Term");
        TDSEZAssembleMatrixTimed(iga, &V, TDSEZFormPotential, "Potential Term");
        TDSEZAssembleMatrixTimed(iga, &Md, TDSEZFormMassDist, "Mass Distribution");
    }

    // Construct dipole matrices based on specified polarization
    if (pol == "x")
    {
        // Bulk assemble all 6 operators for x-polarization in one go
        Mat *mats[6] = {&K, &V, &Md, &Dx, &VelX, &dVdx};
        TDSEZAssembleMatBatchTimed(iga, 6, mats, TDSEZFormPhyX, "X-Polarization Ops");

    }
    else if (pol == "y" && dim >= 2)
    {        // construct Dy
        TDSEZAssembleMatrixTimed(iga, &Dy, DipoleY, "Dipole Y");
        // construct VelY
        TDSEZAssembleMatrixTimed(iga, &VelY, VelocityY, "Velocity Y");
        // construct dVdy
        TDSEZAssembleMatrixTimed(iga, &dVdy, TDSEZformPotentialGradY, "Potential Gradient Y");
    }
    else if (pol == "z" && dim == 3)
    {
        // construct Dz
        TDSEZAssembleMatrixTimed(iga, &Dz, DipoleZ, "Dipole Z");
        // construct VelZ
        TDSEZAssembleMatrixTimed(iga, &VelZ, VelocityZ, "Velocity Z");
        // construct dVdz
        TDSEZAssembleMatrixTimed(iga, &dVdz, TDSEZformPotentialGradZ, "Potential Gradient Z");
    }
    else if (pol == "xy" && dim >= 2)
    {
        // construct Dx, Dy
        TDSEZAssembleMatrixTimed(iga, &Dx, DipoleX, "Dipole X");
        TDSEZAssembleMatrixTimed(iga, &Dy, DipoleY, "Dipole Y");
        // construct VelX, VelY
        TDSEZAssembleMatrixTimed(iga, &VelX, VelocityX, "Velocity X");
        TDSEZAssembleMatrixTimed(iga, &VelY, VelocityY, "Velocity Y");
        // construct dVdx, dVdy
        TDSEZAssembleMatrixTimed(iga, &dVdx, TDSEZformPotentialGradX, "Potential Gradient X");
        TDSEZAssembleMatrixTimed(iga, &dVdy, TDSEZformPotentialGradY, "Potential Gradient Y");
    }
    else if (pol == "xz" && dim == 3)
    {
        // construct Dx, Dz
        TDSEZAssembleMatrixTimed(iga, &Dx, DipoleX, "Dipole X");
        TDSEZAssembleMatrixTimed(iga, &Dz, DipoleZ, "Dipole Z");
        // construct VelX, VelZ
        TDSEZAssembleMatrixTimed(iga, &VelX, VelocityX, "Velocity X");
        TDSEZAssembleMatrixTimed(iga, &VelZ, VelocityZ, "Velocity Z");
        // construct dVdx, dVdz
        TDSEZAssembleMatrixTimed(iga, &dVdx, TDSEZformPotentialGradX, "Potential Gradient X");
        TDSEZAssembleMatrixTimed(iga, &dVdz, TDSEZformPotentialGradZ, "Potential Gradient Z");
    }
    else if (pol == "yz" && dim == 3)
    {
        // construct Dy, Dz
        TDSEZAssembleMatrixTimed(iga, &Dy, DipoleY, "Dipole Y");
        TDSEZAssembleMatrixTimed(iga, &Dz, DipoleZ, "Dipole Z");
        // construct VelY, VelZ
        TDSEZAssembleMatrixTimed(iga, &VelY, VelocityY, "Velocity Y");
        TDSEZAssembleMatrixTimed(iga, &VelZ, VelocityZ, "Velocity Z");
        // construct dVdy, dVdz
        TDSEZAssembleMatrixTimed(iga, &dVdy, TDSEZformPotentialGradY, "Potential Gradient Y");
        TDSEZAssembleMatrixTimed(iga, &dVdz, TDSEZformPotentialGradZ, "Potential Gradient Z");
    }
    else if ((pol == "xyz" || pol == "all") && dim == 3)
    {
        // construct Dx, Dy, Dz
        TDSEZAssembleMatrixTimed(iga, &Dx, DipoleX, "Dipole X");
        TDSEZAssembleMatrixTimed(iga, &Dy, DipoleY, "Dipole Y");
        TDSEZAssembleMatrixTimed(iga, &Dz, DipoleZ, "Dipole Z");
        // construct VelX, VelY, VelZ
        TDSEZAssembleMatrixTimed(iga, &VelX, VelocityX, "Velocity X");
        TDSEZAssembleMatrixTimed(iga, &VelY, VelocityY, "Velocity Y");
        TDSEZAssembleMatrixTimed(iga, &VelZ, VelocityZ, "Velocity Z");
        // construct dVdx, dVdy, dVdz
        TDSEZAssembleMatrixTimed(iga, &dVdx, TDSEZformPotentialGradX, "Potential Gradient X");
        TDSEZAssembleMatrixTimed(iga, &dVdy, TDSEZformPotentialGradY, "Potential Gradient Y");
        TDSEZAssembleMatrixTimed(iga, &dVdz, TDSEZformPotentialGradZ, "Potential Gradient Z");

        PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------\n");
    }
    else
    {
        throw std::runtime_error("The polarization choice is not compatible with the  dimension of the problem. Please check your input file.");
    }

}
