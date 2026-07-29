// ============================================================================
//  assembly.cpp  —  ALL matrix/vector ASSEMBLY routines for TDSEZ
// ----------------------------------------------------------------------------
//  PETSc-style organization: every PetIGA form-kernel, operator batch-assembly
//  wrapper, and the core element-loop assembler live in this single translation
//  unit. This is the only TU that knows how to turn physics (potential, mass,
//  dipole, velocity, gradient, CAP) into PETSc Mat/Vec objects.
//
//  Contents:
//    - Manolopoulos_CAP_profile      (complex-absorbing-potential profile)
//    - TDSEZCompOperators          (core element loop: IGABeginElement ...)
//    - TDSEZAssembleMatrixTimed / TDSEZAssembleMatBatchTimed  (timed wrappers)
//    - TDSEZFormHam / TDSEZFormPhyX  (multi-operator kernels, 1D/2D/3D)
//    - TDSEZFormHamiltonian, TDSEZFormLz, TDSEZFormKinetic, TDSEZFormPotential,
//      TDSEZFormMass, TDSEZFormMassDist   (single-operator form kernels)
//    - DipoleX/Y/Z, VelocityX/Y/Z, TDSEZformPotentialGradX/Y/Z,
//      TDSEZformPotentialGrad_old, TDSEZformCap   (form kernels)
//    - TDSEZGetCoeffs, TDSEZProjState   (initial-state projection assembly)
//    - ComputeCommutator                    (Lz commutator matrix op)
//
//  NO logic was rewritten vs the prior split files (form_functions.cpp,
//  physics_kernels.cpp, assembler.cpp, and the assembly parts of
//  diagnostics.cpp). The split was purely organizational.
// ============================================================================

#include "tdsez_internal.hpp"

#include "tdsez_internal.hpp"






// Manolopoulos CAP profile
PetscReal Manolopoulos_CAP_profile(const PetscReal xq, const PetscReal, const PetscReal x_max, const PetscReal ma_kmin) 
{
    // Constants from Manolopoulos (precompute powers to avoid repeated calls)
    const PetscReal ma_c = 2.62206;
    const PetscReal ma_c2 = ma_c * ma_c;
    const PetscReal ma_c3 = ma_c2 * ma_c;
    const PetscReal ma_a = 1.0 - (16.0 / ma_c3);
    const PetscReal ma_b = (1.0 - (17.0 / ma_c3)) / ma_c2;
    const PetscReal ma_delta = 0.2;
    const PetscReal ma_Emin = 0.5 * ma_kmin * ma_kmin;
    

    // Absorption region
    const PetscReal absorbregionwidth = ma_c / (2.0 * ma_delta * ma_kmin);
    // const PetscReal absorbregionwidth = ma_delta;
    const PetscReal rcap = x_max - absorbregionwidth;
    PetscReal abs_xq = PetscAbsReal(xq);


    // Skip potential calculation if inside core region
    if (abs_xq <= rcap) return 0.0;

    // Map to scaled variable r
    PetscReal denom = x_max - rcap;
    PetscReal r = ma_c * ((abs_xq - rcap) / denom);

    // Handle boundary to avoid division by zero
    if (abs_xq >= x_max) r = ma_c * (1.0 - PETSC_MACHINE_EPSILON);

    // Precompute terms for potential_strength
    PetscReal ma_c_minus_r = ma_c - r;
    PetscReal ma_c_plus_r  = ma_c + r;
    PetscReal r2 = r * r;
    PetscReal r3 = r2 * r;

    PetscReal potential_strength = ma_Emin * (ma_a * r - ma_b * r3 + 4.0 / (ma_c_minus_r * ma_c_minus_r) - 4.0 / (ma_c_plus_r * ma_c_plus_r));

    return potential_strength;
}




typedef PetscErrorCode (*TDSEZPhysicsKernel)(
    IGAPoint p,
    PetscInt nmat,
    PetscScalar *K[],
    void *ctx);


    

PetscErrorCode TDSEZCompOperators(
    IGA                iga,
    PetscInt           nmat,
    Mat                mats[],
    TDSEZPhysicsKernel kernel,
    void              *ctx)
{
    MPI_Comm comm;

    PetscFunctionBegin;

    PetscCheck(iga,    PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IGA is NULL");
    PetscCheck(mats,   PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "mats[] is NULL");
    PetscCheck(kernel, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "kernel is NULL");
    PetscCheck(nmat >= 1 && nmat <= 12, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "nmat %" PetscInt_FMT " must be in [1,12]", nmat);

    comm = PetscObjectComm((PetscObject)iga);

    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCheck(mats[k], comm, PETSC_ERR_ARG_NULL,
                   "mats[%" PetscInt_FMT "] is NULL", k);

        PetscCall(MatSetOption(mats[k], MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
        PetscCall(MatSetOption(mats[k], MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
        PetscCall(MatSetOption(mats[k], MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE));
        PetscCall(MatSetOption(mats[k], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

        /* Required because IGAElementAssembleMat uses ADD_VALUES semantics. */
        PetscCall(MatZeroEntries(mats[k]));
    }

    const size_t size_max = std::numeric_limits<size_t>::max();

    size_t nen_max = 1;
    for (PetscInt d = 0; d < iga->dim; ++d) {
        const size_t q = (size_t)(iga->axis[d]->p + 1);

        PetscCheck(q > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
                   "Invalid IGA degree on axis %" PetscInt_FMT, d);
        PetscCheck(nen_max <= size_max / q, comm, PETSC_ERR_ARG_OUTOFRANGE,
                   "nen_max overflow");

        nen_max *= q;
    }

    PetscCheck(nen_max > 0 && nen_max <= size_max / nen_max, comm,
               PETSC_ERR_ARG_OUTOFRANGE, "nen_max^2 overflow");

    const size_t n2_max = nen_max * nen_max;

    PetscCheck((size_t)nmat <= size_max / n2_max, comm,
               PETSC_ERR_ARG_OUTOFRANGE, "work allocation overflow");

    const size_t alloc_cnt   = (size_t)nmat * n2_max;
    const size_t alloc_bytes = alloc_cnt * sizeof(PetscScalar);

    PetscScalar *work_storage = NULL;
    PetscScalar *work[12];

    IGAElement element = NULL;
    IGAPoint   point   = NULL;
    PetscBool  element_open = PETSC_FALSE;
    PetscBool  point_open   = PETSC_FALSE;
    PetscErrorCode ierr = PETSC_SUCCESS;

    PetscCall(PetscMalloc1(alloc_cnt, &work_storage));

    for (PetscInt k = 0; k < nmat; ++k) {
        work[k] = work_storage + (size_t)k * n2_max;
    }

    ierr = IGABeginElement(iga, &element);
    if (PetscUnlikely(ierr)) goto cleanup;
    element_open = PETSC_TRUE;

    while (IGANextElement(iga, element)) {
        const PetscInt nen = element->nen;
        const size_t   n2  = (size_t)nen * (size_t)nen;

#if defined(PETSC_USE_DEBUG)
        if (PetscUnlikely((size_t)nen > nen_max)) {
            ierr = PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__,
                              PETSC_ERR_ARG_WRONGSTATE, PETSC_ERROR_INITIAL,
                              "element->nen=%" PetscInt_FMT
                              " exceeds computed nen_max=%zu", nen, nen_max);
            goto cleanup;
        }
#endif

        if (PetscLikely(n2 == n2_max)) {
            ierr = PetscMemzero(work_storage, alloc_bytes);
            if (PetscUnlikely(ierr)) goto cleanup;
        } else {
            const size_t active_bytes = n2 * sizeof(PetscScalar);
            for (PetscInt k = 0; k < nmat; ++k) {
                ierr = PetscMemzero(work[k], active_bytes);
                if (PetscUnlikely(ierr)) goto cleanup;
            }
        }

        ierr = IGAElementBeginPoint(element, &point);
        if (PetscUnlikely(ierr)) goto cleanup;
        point_open = PETSC_TRUE;

        while (IGAElementNextPoint(element, point)) {
            ierr = kernel(point, nmat, work, ctx);
            if (PetscUnlikely(ierr)) goto cleanup;
        }

        ierr = IGAElementEndPoint(element, &point);
        point_open = PETSC_FALSE;
        if (PetscUnlikely(ierr)) goto cleanup;

        for (PetscInt k = 0; k < nmat; ++k) {
            ierr = IGAElementAssembleMat(element, work[k], mats[k]);
            if (PetscUnlikely(ierr)) goto cleanup;
        }
    }

cleanup:
    if (point_open) {
        PetscErrorCode ierr2 = IGAElementEndPoint(element, &point);
        if (!ierr) ierr = ierr2;
    }

    if (element_open) {
        PetscErrorCode ierr2 = IGAEndElement(iga, &element);
        if (!ierr) ierr = ierr2;
    }

    {
        PetscErrorCode ierr2 = PetscFree(work_storage);
        if (!ierr) ierr = ierr2;
    }

    PetscCall(ierr);

    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCall(MatAssemblyBegin(mats[k], MAT_FINAL_ASSEMBLY));
    }

    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCall(MatAssemblyEnd(mats[k], MAT_FINAL_ASSEMBLY));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  TDSEZCompOperatorsDirichlet — identical to TDSEZCompOperators EXCEPT it
//  applies the boundary conditions recorded by IGASetBoundaryValue via
//  IGAElementFixSystem (the same step IGAComputeSystem performs). Without this
//  step the Hamiltonian is the free Galerkin (natural/Neumann) operator and the
//  wall is reflecting. With it, homogeneous Dirichlet psi=0 is enforced at the
//  boundary nodes (the infinite-wall / box eigenstate convention).
//
//  This is ADDITIVE: it does not change TDSEZCompOperators behaviour. The
//  production core.cpp call site may opt in by switching to this function.
// ---------------------------------------------------------------------------
PetscErrorCode TDSEZCompOperatorsDirichlet(
    IGA                iga,
    PetscInt           nmat,
    Mat                mats[],
    TDSEZPhysicsKernel kernel,
    void              *ctx)
{
    MPI_Comm comm;

    PetscFunctionBegin;

    PetscCheck(iga,    PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IGA is NULL");
    PetscCheck(mats,   PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "mats[] is NULL");
    PetscCheck(kernel, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "kernel is NULL");
    PetscCheck(nmat >= 1 && nmat <= 12, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "nmat %" PetscInt_FMT " must be in [1,12]", nmat);

    comm = PetscObjectComm((PetscObject)iga);

    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCheck(mats[k], comm, PETSC_ERR_ARG_NULL,
                   "mats[%d] is NULL", k);
        PetscCall(MatSetOption(mats[k], MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
        PetscCall(MatSetOption(mats[k], MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
        PetscCall(MatSetOption(mats[k], MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE));
        PetscCall(MatSetOption(mats[k], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
        PetscCall(MatZeroEntries(mats[k]));
    }

    const size_t size_max = std::numeric_limits<size_t>::max();

    size_t nen_max = 1;
    for (PetscInt d = 0; d < iga->dim; ++d) {
        const size_t q = (size_t)(iga->axis[d]->p + 1);
        PetscCheck(q > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
                   "Invalid IGA degree on axis %" PetscInt_FMT, d);
        PetscCheck(nen_max <= size_max / q, comm, PETSC_ERR_ARG_OUTOFRANGE, "nen_max overflow");
        nen_max *= q;
    }
    PetscCheck(nen_max > 0 && nen_max <= size_max / nen_max, comm,
               PETSC_ERR_ARG_OUTOFRANGE, "nen_max^2 overflow");
    const size_t n2_max = nen_max * nen_max;
    PetscCheck((size_t)nmat <= size_max / n2_max, comm,
               PETSC_ERR_ARG_OUTOFRANGE, "work allocation overflow");

    const size_t alloc_cnt   = (size_t)nmat * n2_max;
    const size_t alloc_bytes = alloc_cnt * sizeof(PetscScalar);

    PetscScalar *work_storage = NULL;
    PetscScalar *work[12];

    IGAElement element = NULL;
    IGAPoint   point   = NULL;
    PetscBool  element_open = PETSC_FALSE;
    PetscBool  point_open   = PETSC_FALSE;
    PetscErrorCode ierr = PETSC_SUCCESS;

    PetscCall(PetscMalloc1(alloc_cnt, &work_storage));
    for (PetscInt k = 0; k < nmat; ++k) work[k] = work_storage + (size_t)k * n2_max;

    ierr = IGABeginElement(iga, &element);
    if (PetscUnlikely(ierr)) goto cleanup;
    element_open = PETSC_TRUE;

    while (IGANextElement(iga, element)) {
        const PetscInt nen = element->nen;
        const size_t   n2  = (size_t)nen * (size_t)nen;

#if defined(PETSC_USE_DEBUG)
        if (PetscUnlikely((size_t)nen > nen_max)) {
            ierr = PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__,
                              PETSC_ERR_ARG_WRONGSTATE, PETSC_ERROR_INITIAL,
                              "element->nen=%" PetscInt_FMT " exceeds computed nen_max=%zu", nen, nen_max);
            goto cleanup;
        }
#endif

        if (PetscLikely(n2 == n2_max)) {
            ierr = PetscMemzero(work_storage, alloc_bytes);
            if (PetscUnlikely(ierr)) goto cleanup;
        } else {
            const size_t active_bytes = n2 * sizeof(PetscScalar);
            for (PetscInt k = 0; k < nmat; ++k) {
                ierr = PetscMemzero(work[k], active_bytes);
                if (PetscUnlikely(ierr)) goto cleanup;
            }
        }

        ierr = IGAElementBeginPoint(element, &point);
        if (PetscUnlikely(ierr)) goto cleanup;
        point_open = PETSC_TRUE;

        while (IGAElementNextPoint(element, point)) {
            ierr = kernel(point, nmat, work, ctx);
            if (PetscUnlikely(ierr)) goto cleanup;
        }

        ierr = IGAElementEndPoint(element, &point);
        point_open = PETSC_FALSE;
        if (PetscUnlikely(ierr)) goto cleanup;

        // ---- Dirichlet enforcement: apply IGASetBoundaryValue via FixSystem ----
        // (this is the step TDSEZCompOperators omits; IGAComputeSystem does it)
        // NOTE: the mass matrix M (slot k==1 in the {H,M} convention) is the
        // L^2 inner product and MUST stay unconstrained — pinning it too would
        // create spurious unit eigenvalues in the generalized eigenproblem
        // H psi = E M psi. Only the operator(s) that multiply psi (H, V, D, Lz)
        // get the homogeneous-Dirichlet pin at the wall.
        {
            PetscScalar *B = NULL;
            ierr = IGAElementGetWorkVec(element, &B);
            if (PetscUnlikely(ierr)) goto cleanup;
            for (PetscInt k = 0; k < nmat; ++k) {
                if (k == 1) continue;   // skip mass matrix M
                ierr = IGAElementFixSystem(element, work[k], B);
                if (PetscUnlikely(ierr)) goto cleanup;
            }
        }

        for (PetscInt k = 0; k < nmat; ++k) {
            ierr = IGAElementAssembleMat(element, work[k], mats[k]);
            if (PetscUnlikely(ierr)) goto cleanup;
        }
    }

cleanup:
    if (point_open) {
        PetscErrorCode ierr2 = IGAElementEndPoint(element, &point);
        if (!ierr) ierr = ierr2;
    }
    if (element_open) {
        PetscErrorCode ierr2 = IGAEndElement(iga, &element);
        if (!ierr) ierr = ierr2;
    }
    {
        PetscErrorCode ierr2 = PetscFree(work_storage);
        if (!ierr) ierr = ierr2;
    }

    PetscCall(ierr);

    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCall(MatAssemblyBegin(mats[k], MAT_FINAL_ASSEMBLY));
    }
    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCall(MatAssemblyEnd(mats[k], MAT_FINAL_ASSEMBLY));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}



PetscErrorCode TDSEZFormHam(
    IGAPoint p,
    PetscInt nmat,
    PetscScalar *M[],
    void *ctx)
{
    PetscFunctionBegin;
    (void)nmat;
    (void)ctx;

    const PetscInt nen = p->nen;
    const PetscInt dim = p->dim;

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));

    const PetscReal x = xyz[0];
    const PetscReal y = xyz[1];
    const PetscReal z = xyz[2];

    const PetscReal w = p->weight[0] * p->detJac[0];
    const PetscReal invMass = (PetscReal)(1.0 / TDSEZParser::MassDist(x, y, z));
    const PetscReal cW = (PetscReal)0.5 * invMass * w;

    const PetscReal *B = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));

    if (__builtin_expect(dim == 3, 1)) {
        const PetscReal (*dB)[3] = NULL;
        PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));

        const PetscReal Vw = TDSEZParser::V(x, y, z) * w;

        enum { STACK_NEN = 512 };
        PetscReal dBx_stack[STACK_NEN];
        PetscReal dBy_stack[STACK_NEN];
        PetscReal dBz_stack[STACK_NEN];

        PetscReal *dBx = dBx_stack;
        PetscReal *dBy = dBy_stack;
        PetscReal *dBz = dBz_stack;

        if (nen > STACK_NEN) {
            PetscCall(PetscMalloc3(nen, &dBx, nen, &dBy, nen, &dBz));
        }

        for (PetscInt i = 0; i < nen; ++i) {
            dBx[i] = dB[i][0];
            dBy[i] = dB[i][1];
            dBz[i] = dB[i][2];
        }

        for (PetscInt a = 0; a < nen; ++a) {
            const PetscReal BaW = B[a] * w;
            const PetscReal pot_a = B[a] * Vw;
            const PetscReal gx_a = dBx[a] * cW;
            const PetscReal gy_a = dBy[a] * cW;
            const PetscReal gz_a = dBz[a] * cW;

            PetscScalar *__restrict__ Hrow = M[0] + a * nen;
            PetscScalar *__restrict__ Mrow = M[1] + a * nen;

            for (PetscInt b = 0; b < nen; ++b) {
                const PetscReal Bb = B[b];

                Hrow[b] += gx_a * dBx[b]
                         + gy_a * dBy[b]
                         + gz_a * dBz[b]
                         + pot_a * Bb;

                Mrow[b] += BaW * Bb;
            }
        }

        if (nen > STACK_NEN) {
            PetscCall(PetscFree3(dBx, dBy, dBz));
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    if (dim == 2) {
        const PetscReal (*dB)[2] = NULL;
        PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));

        const PetscReal Vw = TDSEZParser::V(x, y) * w;

        for (PetscInt a = 0; a < nen; ++a) {
            const PetscReal BaW = B[a] * w;
            const PetscReal pot_a = B[a] * Vw;
            const PetscReal gx_a = dB[a][0] * cW;
            const PetscReal gy_a = dB[a][1] * cW;

            PetscScalar *__restrict__ Hrow = M[0] + a * nen;
            PetscScalar *__restrict__ Mrow = M[1] + a * nen;

            for (PetscInt b = 0; b < nen; ++b) {
                const PetscReal Bb = B[b];

                Hrow[b] += gx_a * dB[b][0]
                         + gy_a * dB[b][1]
                         + pot_a * Bb;

                Mrow[b] += BaW * Bb;
            }
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    if (dim == 1) {
        const PetscReal (*dB)[1] = NULL;
        PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));

        const PetscReal Vw = TDSEZParser::V(x) * w;

        for (PetscInt a = 0; a < nen; ++a) {
            const PetscReal BaW = B[a] * w;
            const PetscReal pot_a = B[a] * Vw;
            const PetscReal gx_a = dB[a][0] * cW;

            PetscScalar *Hrow = M[0] + a * nen;
            PetscScalar *Mrow = M[1] + a * nen;

            for (PetscInt b = 0; b < nen; ++b) {
                const PetscReal Bb = B[b];

                Hrow[b] += gx_a * dB[b][0] + pot_a * Bb;
                Mrow[b] += BaW * Bb;
            }
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


// X-POLARIZATION: [K, V, Md, X, VelX, dVdx]
// static inline __attribute__((always_inline))
// PetscErrorCode TDSEZFormPhyX(
//     IGAPoint p,
//     PetscInt nmat,
//     PetscScalar *M[],
//     void *ctx)
// {
//     PetscFunctionBegin;
//     PetscInt nen = p->nen;
//     PetscInt dim = p->dim;

//     PetscReal xyz[3] = {0.0, 0.0, 0.0};
//     IGAPointFormGeomMap(p, xyz); 
//     PetscReal x = xyz[0], y = xyz[1], z = xyz[2];

//     const PetscReal w = p->weight[0] * p->detJac[0]; 


//     // 0 derivative shape functions
//     alignas(64) const PetscReal *B;
//     const PetscReal invMass = (PetscReal) (1.0 / TDSEZParser::MassDist(x, y, z));
//     const PetscReal coeff = 0.5 * invMass;


//     // Mass grad and laplacian terms
//     auto invMassFunc = [](const std::vector<PetscReal>& xx) {
//         return 1.0/TDSEZParser::MassDist(xx[0], xx[1], xx[2]);
//     };

//     if (dim == 1)
//     {
//         const PetscReal (*dB)[1];
//         IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
//         IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);

//         const PetscReal w = p->weight[0] *   p->detJac[0]; 
//         PetscReal Vxyz = TDSEZParser::V(x);
//         PetscReal dVdx = TDSEZParser::dVx(x);


//         // Compute physics derivatives
//         PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x}, 1e-4);
//         PetscReal f    = invMassFunc({x});
//         PetscReal fx   = ops.first[0];
//         PetscReal fxx  = ops.second[0]; 
//         PetscReal fxxx = ops.third[0];  


//         for (PetscInt a = 0; a < nen; ++a) 
//         {
//             PetscReal Ba = B[a];
//             PetscReal dBa_x = dB[a][0];

//             // pointer to the start of row 'a'
//             PetscScalar *Krow  = &M[0][a * nen];
//             PetscScalar *Vrow  = &M[1][a * nen];
//             PetscScalar *Mdrow = &M[2][a * nen];
//             PetscScalar *Xrow  = &M[3][a * nen];
//             PetscScalar *Pxrow = &M[4][a * nen];
//             PetscScalar *dVdxrow = &M[5][a * nen];

//             for (PetscInt b = 0; b < nen; ++b) {
//                 PetscReal Bb = B[b];
//                 PetscReal dBb_x = dB[b][0];

//                 Krow[b] += coeff * dBa_x * dBb_x * w;
//                 Vrow[b] +=  Ba * Bb * Vxyz * w;
//                 Mdrow[b] += Ba * invMass * Bb * w;
//                 Xrow[b] += Ba * Bb * x * w;
//                 Pxrow[b] += -0.5 * PETSC_i * (Ba*dBb_x - dBa_x*Bb) * w;

//                 // for dVdx row, add contribution from mass gradient terms
//                 // Row 0: -Ba * dVdx * f * Bb
//                 PetscScalar classical = -Ba * dVdx * f * Bb;
//                 // Row 1: 2 * Bi * Bj_x * fx^2
//                 PetscScalar R1 = 2.0 * Ba * dBb_x * (fx * fx);
//                 // Row 3: f * [ 2 * Bi * Bj_x * fxx + Bi * Bb * fxxx ]
//                 PetscScalar R3 = f * (2.0 * Ba * dBb_x * fxx + Ba * Bb * fxxx);
//                 // Row 4: -2 * f * fx * (dBax * dBbx)
//                 PetscScalar R4 = -2.0 * f * fx * (dBa_x * dBb_x);
//                 // Row 5: -2 * Bi * Bj_x * (fx*fx + f*fxx)
//                 PetscScalar R5 = -2.0 * Ba * dBb_x * (fx*fx + f*fxx);
//                 // Row 6: Bi * Bj * fx * fxx
//                 PetscScalar R6 = Ba * Bb * (fx * fxx);
//                 PetscScalar quantum = -0.25 * (R1 + R3 + R4 + R5 + R6);
//                 dVdxrow[b] += (classical + quantum) * w;
//             }
//         }
//     }
//     else if (__builtin_expect(dim == 2, 0))
//     {
//         const PetscReal (*dB)[2], (*ddB)[2][2];
//         IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
//         IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
//         IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);

//         const PetscReal w = p->weight[0] *   p->detJac[0]; 
//         PetscReal Vxyz = TDSEZParser::V(x, y);
//         PetscReal dVdx = TDSEZParser::dVx(x, y);


//         // Mass derivatives (assuming dim=2, so 3rd order tensor size is 2^3=8)
//         PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x, y}, 1e-4);
//         PetscReal f    = invMassFunc({x, y});
//         PetscReal fx   = ops.first[0];
//         PetscReal fy   = ops.first[1];
//         PetscReal fxx  = ops.second[0 * 2 + 0];
//         PetscReal fyy  = ops.second[1 * 2 + 1];
//         PetscReal fxy  = ops.second[0 * 2 + 1]; // fxy = fyx
//         PetscReal fxxx = ops.third[(0 * 2 + 0) * 2 + 0];
//         PetscReal fxyy = ops.third[(0 * 2 + 1) * 2 + 1];



//         for (PetscInt a = 0; a < nen; ++a) 
//         {
//             const PetscReal Ba = B[a];
//             const PetscReal dBa_x = dB[a][0];
//             const PetscReal dBa_y = dB[a][1];
            
//             PetscScalar *Krow  = &M[0][a * nen];
//             PetscScalar *Vrow  = &M[1][a * nen];
//             PetscScalar *Mdrow = &M[2][a * nen];
//             PetscScalar *Xrow  = &M[3][a * nen];
//             PetscScalar *Pxrow = &M[4][a * nen];
//             PetscScalar *dVdxrow = &M[5][a * nen];

//             for (PetscInt b = 0; b < nen; ++b)
//             {
//                 const PetscReal Bb = B[b];
//                 const PetscReal dBb_x = dB[b][0];
//                 const PetscReal dBb_y = dB[b][1];
//                 const PetscScalar dBb_xy = ddB[b][0][1];

//                 Krow[b] += coeff * (dBa_x * dBb_x + dBa_y * dBb_y) * w;
//                 Vrow[b] +=  Ba * Bb * Vxyz * w;
//                 Mdrow[b] += Ba * invMass * Bb * w;
//                 Xrow[b] += Ba * Bb * x * w;
//                 Pxrow[b] += -0.5 * PETSC_i * (Ba*dBb_x - dBa_x*Bb) * w;

//                 // Row 0: -Ba * dVdx * f * Bb
//                 PetscScalar classical = -Ba * dVdx * f * Bb;
                
//                 // Row 1: 2 * Bi * Bj_x * (fx^2 + fy^2)
//                 PetscScalar R1 = 2.0 * Ba * dBb_x * (fx*fx + fy*fy);

//                 // Row 3: f * ( 4*fy*Bi*Bj_xy + 2*Bi*Bj_x*(fyy+fxx) + Bi*Bj*(fxyy+fxxx) )
//                 PetscScalar R3 = f * (4.0 * fy * Ba * dBb_xy + 
//                                       2.0 * Ba * dBb_x * (fyy + fxx) + 
//                                       Ba * Bb * (fxyy + fxxx));

//                 // Row 4: 2 * f * fx * (Bay*Bby - Bax*Bbx)
//                 PetscScalar R4 = 2.0 * f * fx * (dBa_y * dBb_y - dBa_x * dBb_x);

//                 // Row 5: 2 * ( Bi*Bby*(fy*fx + f*fxy) - Bi*Bbx*(fx*fx + f*fxx) )
//                 PetscScalar R5 = 2.0 * (Ba * dBb_y * (fy*fx + f*fxy) - 
//                                         Ba * dBb_x * (fx*fx + f*fxx));

//                 // Row 6: Bi * Bj * (fy*fxy + fx*fxx)
//                 PetscScalar R6 = Ba * Bb * (fy * fxy + fx * fxx);

//                 PetscScalar quantum = -0.25 * (R1 + R3 + R4 + R5 + R6);

//                 dVdxrow[b] += (classical + quantum) * w;
//             }
//         }
//     }
//     else if (dim == 3)
//     {
//         const PetscReal (*dB)[3], (*ddB)[3][3];
        
//         IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
//         IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
//         IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);

//         const PetscReal w = p->weight[0] *   p->detJac[0]; 
//         const PetscReal V_val = TDSEZParser::V(x, y, z);
//         const PetscReal dVdx = TDSEZParser::dVx(x, y, z);


//         // Derivatives of f (Inverse Mass) - Index mapping: i*dim + j
//         PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x, y, z}, 1e-4);
//         PetscReal f    = invMassFunc({x, y, z});
//         PetscReal fx = ops.first[0], fy = ops.first[1], fz = ops.first[2];
        
//         // 2nd Derivatives
//         PetscReal fxx = ops.second[0*3 + 0], fyy = ops.second[1*3 + 1], fzz = ops.second[2*3 + 2];
//         PetscReal fxy = ops.second[0*3 + 1], fxz = ops.second[0*3 + 2], fyz = ops.second[1*3 + 2];
    
//         // 3rd Derivatives - Index mapping: (i*dim + j)*dim + k
//         PetscReal fxxx = ops.third[(0*3 + 0)*3 + 0];
//         PetscReal fxyy = ops.third[(0*3 + 1)*3 + 1];
//         PetscReal fxzz = ops.third[(0*3 + 2)*3 + 2];
    
//         // Row 5 Components: (f*fx)_i = f_i*f_x + f*f_ix
//         PetscReal ffx_x = fx*fx + f*fxx;
//         PetscReal ffx_y = fy*fx + f*fxy;
//         PetscReal ffx_z = fz*fx + f*fxz;

//         // prefetch 
//         PetscReal static_dBx[512], static_dBy[512], static_dBz[512];
//         alignas(64) PetscReal *dBx = static_dBx, *dBy = static_dBy, *dBz = static_dBz;
//         PetscBool heap_used = PETSC_FALSE;
//         if (nen > 512) {
//             PetscMalloc3(nen, &dBx, nen, &dBy, nen, &dBz);
//             heap_used = PETSC_TRUE;
//         }

//         for(PetscInt i=0; i<nen; i++) {
//             dBx[i] = dB[i][0]; dBy[i] = dB[i][1]; dBz[i] = dB[i][2];
//         }

//         for(PetscInt a=0; a<nen; a++)
//         {
//             const PetscReal Ba = B[a];
//             const PetscReal dBa_x = dBx[a];
//             const PetscReal dBa_y = dBy[a];
//             const PetscReal dBa_z = dBz[a];
//             const PetscReal va   = B[a] * V_val;

//             PetscScalar *Krow  = &M[0][a * nen];
//             PetscScalar *Vrow  = &M[1][a * nen];
//             PetscScalar *Mdrow = &M[2][a * nen];
//             PetscScalar *Xrow  = &M[3][a * nen];
//             PetscScalar *Pxrow = &M[4][a * nen];
//             PetscScalar *dVdxrow = &M[5][a * nen];

//             for(PetscInt b=0; b<nen; b++)
//             {
//                 const PetscReal Bb = B[b];
//                 const PetscReal dBb_x = dB[b][0], dBb_y = dB[b][1], dBb_z = dB[b][2];
//                 const PetscScalar dBb_xy = ddB[b][0][1], dBb_xz = ddB[b][0][2];

//                 Krow[b] +=  coeff * (dBa_x * dBb_x + dBa_y * dBb_y + dBa_z * dBb_z) * w;
//                 Vrow[b] +=  va * Bb * w;
//                 Mdrow[b] += Ba * invMass * Bb * w;
//                 Xrow[b] += Ba * Bb * x * w;
//                 Pxrow[b] += -0.5 * PETSC_i * (Ba*dBb_x - dBa_x*Bb) * w;

                    
//                 // Row 0: -Ba * dVdx * f * Bb
//                 PetscScalar classical = -Ba * dVdx * f * Bb;
                    
//                 // Row 1: 2 * Bi * Bj_x * (fx^2 + fy^2 + fz^2)
//                 PetscScalar row1 = 2.0 * Ba * dBb_x * (fx*fx + fy*fy + fz*fz);
    
//                 // Row 2: fz * (4*f*Bi*Bj_xz + Bi*Bj*fxz)
//                 PetscScalar row2 = fz * (4.0 * f * Ba * dBb_xz + Ba * Bb * fxz);
    
//                 // Row 3: f * [4*fy*Bi*Bj_xy + 2*Bi*Bj_x*(fxx+fyy+fzz) + Bi*Bj*(fxxx+fxyy+fxzz)]
//                 PetscScalar row3 = f * (4.0 * fy * Ba * dBb_xy + 
//                                         2.0 * Ba * dBb_x * (fxx + fyy + fzz) + 
//                                         Ba * Bb * (fxxx + fxyy + fxzz));
    
//                 // Row 4: 2*f*fx * (Baz*Bbz + Bay*Bby - Bax*Bbx)
//                 PetscScalar row4 = 2.0 * f * fx * (dBa_z * dBb_z + dBa_y * dBb_y - dBa_x * dBb_x);
    
//                 // Row 5: 2 * [ Ba*dBbz*ffx_z + Ba*dBby*ffx_y - Ba*dBbx*ffx_x ]
//                 PetscScalar row5 = 2.0 * (Ba * dBb_z * ffx_z + Ba * dBb_y * ffx_y - Ba * dBb_x * ffx_x);

//                 // Row 6: Bi * Bj * (fy*fxy + fx*fxx)
//                 PetscScalar row6 = Ba * Bb * (fy * fxy + fx * fxx);
    
//                 PetscScalar quantum = -0.25 * (row1 + row2 + row3 + row4 + row5 + row6);
    
//                 dVdxrow[b] += (classical + quantum) * w;
//             }
//         }

//         if (heap_used) {
//             PetscFree3(dBx, dBy, dBz);
//         }
//     }

//     return 0;
// }




// X-POLARIZATION: [K, V, Md, X, VelX, dVdx]
PetscErrorCode TDSEZFormPhyX(
    IGAPoint  p,
    PetscInt  nmat,
    PetscScalar *M[],
    void     *ctx)
{
    PetscFunctionBegin;
    (void)nmat;
    (void)ctx;
    const PetscInt nen = p->nen;
    const PetscInt dim = p->dim;

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz);
    const PetscReal x = xyz[0], y = xyz[1], z = xyz[2];

    const PetscReal invMass = 1.0 / TDSEZParser::MassDist(x, y, z);
    const PetscReal coeff   = 0.5 * invMass;

    // When mass is constant all spatial derivatives of f=1/m vanish:
    //   fx=fy=fz=0, fxx=...=0, fxxx=...=0
    // => every quantum correction row (R1..R6 / row1..row6) is exactly zero.
    // TDSEZCompDerivative is never called, saving O(dim^2+dim^3) MassDist
    // evaluations per quadrature point — the dominant cost in the variable-mass path.
    const bool massIsConstant = (TDSEZParser::MassIsConstant == PETSC_TRUE);

    // invMassFunc: used only in the variable-mass path.
    // Must take std::vector to match TDSEZCompDerivative's signature.
    // The heap allocation this implies per FD call is unavoidable without
    // changing TDSEZCompDerivative itself — but it only runs when
    // MassIsConstant==0, so constant-mass runs pay none of this cost.
    auto invMassFunc = [](const std::vector<PetscReal>& xx) {
        return 1.0 / TDSEZParser::MassDist(xx[0], xx[1], xx[2]);
    };

    if (dim == 1)
    {
        const PetscReal *B;
        const PetscReal (*dB)[1];
        IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);

        const PetscReal w    = p->weight[0] * p->detJac[0];
        const PetscReal Vxyz = TDSEZParser::V(x);
        const PetscReal dVdx = TDSEZParser::dVx(x);
        const PetscReal neg_dVdx_f = -dVdx * invMass;

        // Variable-mass: compute derivatives once per quadrature point,
        // then precompute all (a,b)-invariant combinations.
        PetscReal two_fx2=0, f_2fxx=0, f_fxxx=0, neg2_f_fx=0, neg2_fx2_fxx=0, fx_fxx=0;
        if (!massIsConstant) {
            PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x, 0.0, 0.0}, 1e-4);
            const PetscReal fx   = ops.first[0];
            const PetscReal fxx  = ops.second[0];
            const PetscReal fxxx = ops.third[0];
            two_fx2      =  2.0 * fx * fx;
            f_2fxx       =  invMass * 2.0 * fxx;
            f_fxxx       =  invMass * fxxx;
            neg2_f_fx    = -2.0 * invMass * fx;
            neg2_fx2_fxx = -2.0 * (fx*fx + invMass*fxx);
            fx_fxx       =  fx * fxx;
        }

        for (PetscInt a = 0; a < nen; ++a)
        {
            const PetscReal Ba    = B[a];
            const PetscReal dBa_x = dB[a][0];

            PetscScalar *Krow    = &M[0][a * nen];
            PetscScalar *Vrow    = &M[1][a * nen];
            PetscScalar *Mdrow   = &M[2][a * nen];
            PetscScalar *Xrow    = &M[3][a * nen];
            PetscScalar *Pxrow   = &M[4][a * nen];
            PetscScalar *dVdxrow = &M[5][a * nen];

            const PetscReal Ba_w            = Ba * w;
            const PetscReal coeff_dBa_x_w   = coeff * dBa_x * w;
            const PetscReal Ba_Vxyz_w       = Ba * Vxyz * w;
            const PetscReal Ba_invMass_w    = Ba * invMass * w;
            const PetscReal Ba_x_w          = Ba * x * w;
            const PetscReal Ba_neg_dVdx_f_w = Ba * neg_dVdx_f * w;

            for (PetscInt b = 0; b < nen; ++b)
            {
                const PetscReal Bb    = B[b];
                const PetscReal dBb_x = dB[b][0];

                Krow[b]  += coeff_dBa_x_w * dBb_x;
                Vrow[b]  += Ba_Vxyz_w * Bb;
                Mdrow[b] += Ba_invMass_w * Bb;
                Xrow[b]  += Ba_x_w * Bb;
                Pxrow[b] += -0.5 * PETSC_i * (Ba_w * dBb_x - dBa_x * Bb * w);

                // Constant mass: quantum correction is exactly zero,
                // dVdx reduces to the classical Ehrenfest term only.
                PetscScalar dv = Ba_neg_dVdx_f_w * Bb;
                if (!massIsConstant) {
                    const PetscScalar R1 = two_fx2      * Ba * dBb_x;
                    const PetscScalar R3 = (f_2fxx * dBb_x + f_fxxx * Bb) * Ba;
                    const PetscScalar R4 = neg2_f_fx    * dBa_x * dBb_x;
                    const PetscScalar R5 = neg2_fx2_fxx * Ba * dBb_x;
                    const PetscScalar R6 = fx_fxx       * Ba * Bb;
                    dv -= 0.25 * (R1 + R3 + R4 + R5 + R6) * w;
                }
                dVdxrow[b] += dv;
            }
        }
    }
    else if (dim == 2)
    {
        const PetscReal *B;
        const PetscReal (*dB)[2];
        const PetscReal (*ddB)[2][2];
        IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
        IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);

        const PetscReal w    = p->weight[0] * p->detJac[0];
        const PetscReal Vxyz = TDSEZParser::V(x, y);
        const PetscReal dVdx = TDSEZParser::dVx(x, y);
        const PetscReal neg_dVdx_f = -dVdx * invMass;

        PetscReal two_fx2_fy2=0, f_4fy=0, f_2_fxx_fyy=0, f_fxxx_fxyy=0;
        PetscReal two_f_fx=0, fx_fxx=0, fy_fxy=0;
        PetscReal fy_fx=0, f_fxy=0, fx2_f_fxx=0;
        if (!massIsConstant) {
            PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x, y, 0.0}, 1e-4);
            const PetscReal fx  = ops.first[0],  fy  = ops.first[1];
            const PetscReal fxx = ops.second[0*2+0], fyy = ops.second[1*2+1];
            const PetscReal fxy = ops.second[0*2+1];
            const PetscReal fxxx = ops.third[(0*2+0)*2+0];
            const PetscReal fxyy = ops.third[(0*2+1)*2+1];
            two_fx2_fy2  =  2.0 * (fx*fx + fy*fy);
            f_4fy        =  4.0 * invMass * fy;
            f_2_fxx_fyy  =  2.0 * invMass * (fxx + fyy);
            f_fxxx_fxyy  =  invMass * (fxxx + fxyy);
            two_f_fx     =  2.0 * invMass * fx;
            fx_fxx       =  fx * fxx;
            fy_fxy       =  fy * fxy;
            fy_fx        =  fy * fx;
            f_fxy        =  invMass * fxy;
            fx2_f_fxx    =  fx*fx + invMass*fxx;
        }

        for (PetscInt a = 0; a < nen; ++a)
        {
            const PetscReal Ba    = B[a];
            const PetscReal dBa_x = dB[a][0];
            const PetscReal dBa_y = dB[a][1];

            PetscScalar *Krow    = &M[0][a * nen];
            PetscScalar *Vrow    = &M[1][a * nen];
            PetscScalar *Mdrow   = &M[2][a * nen];
            PetscScalar *Xrow    = &M[3][a * nen];
            PetscScalar *Pxrow   = &M[4][a * nen];
            PetscScalar *dVdxrow = &M[5][a * nen];

            const PetscReal Ba_w            = Ba * w;
            const PetscReal Ba_Vxyz_w       = Ba * Vxyz * w;
            const PetscReal Ba_invMass_w    = Ba * invMass * w;
            const PetscReal Ba_x_w          = Ba * x * w;
            const PetscReal Ba_neg_dVdx_f_w = Ba * neg_dVdx_f * w;

            for (PetscInt b = 0; b < nen; ++b)
            {
                const PetscReal Bb     = B[b];
                const PetscReal dBb_x  = dB[b][0];
                const PetscReal dBb_y  = dB[b][1];
                const PetscReal dBb_xy = ddB[b][0][1];

                Krow[b]  += coeff * (dBa_x*dBb_x + dBa_y*dBb_y) * w;
                Vrow[b]  += Ba_Vxyz_w * Bb;
                Mdrow[b] += Ba_invMass_w * Bb;
                Xrow[b]  += Ba_x_w * Bb;
                Pxrow[b] += -0.5 * PETSC_i * (Ba_w * dBb_x - dBa_x * Bb * w);

                PetscScalar dv = Ba_neg_dVdx_f_w * Bb;
                if (!massIsConstant) {
                    const PetscScalar R1 = two_fx2_fy2 * Ba * dBb_x;
                    const PetscScalar R3 = f_4fy       * Ba * dBb_xy
                                         + f_2_fxx_fyy * Ba * dBb_x
                                         + f_fxxx_fxyy * Ba * Bb;
                    const PetscScalar R4 = two_f_fx * (dBa_y*dBb_y - dBa_x*dBb_x);
                    const PetscScalar R5 = 2.0 * (Ba*dBb_y*(fy_fx   + f_fxy)
                                                 - Ba*dBb_x*(fx2_f_fxx));
                    const PetscScalar R6 = Ba * Bb * (fy_fxy + fx_fxx);
                    dv -= 0.25 * (R1 + R3 + R4 + R5 + R6) * w;
                }
                dVdxrow[b] += dv;
            }
        }
    }
    else // dim == 3
    {
        const PetscReal *B;
        const PetscReal (*dB)[3];
        const PetscReal (*ddB)[3][3];
        IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
        IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);

        const PetscReal w     = p->weight[0] * p->detJac[0];
        const PetscReal V_val = TDSEZParser::V(x, y, z);
        const PetscReal dVdx  = TDSEZParser::dVx(x, y, z);
        const PetscReal neg_dVdx_f = -dVdx * invMass;

        // All quantum-correction scalars default to zero (constant-mass path).
        PetscReal two_fgrad2=0, f_4fy=0, f_4fz=0, f_2_lap=0, f_mix=0;
        PetscReal two_f_fx=0, ffx_x=0, ffx_y=0, ffx_z=0;
        PetscReal fz_fxz=0, fy_fxy_fx_fxx=0;
        if (!massIsConstant) {
            PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x, y, z}, 1e-4);
            const PetscReal fx = ops.first[0], fy = ops.first[1], fz = ops.first[2];
            const PetscReal fxx = ops.second[0*3+0], fyy = ops.second[1*3+1], fzz = ops.second[2*3+2];
            const PetscReal fxy = ops.second[0*3+1], fxz = ops.second[0*3+2];
            const PetscReal fxxx = ops.third[(0*3+0)*3+0];
            const PetscReal fxyy = ops.third[(0*3+1)*3+1];
            const PetscReal fxzz = ops.third[(0*3+2)*3+2];
            two_fgrad2      =  2.0 * (fx*fx + fy*fy + fz*fz);
            f_4fy           =  4.0 * invMass * fy;
            f_4fz           =  4.0 * invMass * fz;
            f_2_lap         =  2.0 * invMass * (fxx + fyy + fzz);
            f_mix           =  invMass * (fxxx + fxyy + fxzz);
            two_f_fx        =  2.0 * invMass * fx;
            ffx_x           =  fx*fx + invMass*fxx;
            ffx_y           =  fy*fx + invMass*fxy;
            ffx_z           =  fz*fx + invMass*fxz;
            fz_fxz          =  fz * fxz;
            fy_fxy_fx_fxx   =  fy*fxy + fx*fxx;
        }

        // Prefetch dB into flat arrays: eliminates strided dB[i][j] reads
        // from both a-side and b-side of the inner loop.
        PetscReal buf_dBx[512], buf_dBy[512], buf_dBz[512];
        PetscReal *dBx = buf_dBx, *dBy = buf_dBy, *dBz = buf_dBz;
        PetscBool heap_used = PETSC_FALSE;
        if (PetscUnlikely(nen > 512)) {
            PetscMalloc3(nen, &dBx, nen, &dBy, nen, &dBz);
            heap_used = PETSC_TRUE;
        }
        for (PetscInt i = 0; i < nen; i++) {
            dBx[i] = dB[i][0]; dBy[i] = dB[i][1]; dBz[i] = dB[i][2];
        }

        for (PetscInt a = 0; a < nen; a++)
        {
            const PetscReal Ba    = B[a];
            const PetscReal dBa_x = dBx[a];
            const PetscReal dBa_y = dBy[a];
            const PetscReal dBa_z = dBz[a];

            PetscScalar *Krow    = &M[0][a * nen];
            PetscScalar *Vrow    = &M[1][a * nen];
            PetscScalar *Mdrow   = &M[2][a * nen];
            PetscScalar *Xrow    = &M[3][a * nen];
            PetscScalar *Pxrow   = &M[4][a * nen];
            PetscScalar *dVdxrow = &M[5][a * nen];

            const PetscReal Ba_w            = Ba * w;
            const PetscReal Ba_V_w          = Ba * V_val * w;
            const PetscReal Ba_invMass_w    = Ba * invMass * w;
            const PetscReal Ba_x_w          = Ba * x * w;
            const PetscReal Ba_neg_dVdx_f_w = Ba * neg_dVdx_f * w;
            const PetscReal coeff_w         = coeff * w;

            for (PetscInt b = 0; b < nen; b++)
            {
                const PetscReal Bb     = B[b];
                const PetscReal dBb_x  = dBx[b];
                const PetscReal dBb_y  = dBy[b];
                const PetscReal dBb_z  = dBz[b];
                const PetscReal dBb_xy = ddB[b][0][1];
                const PetscReal dBb_xz = ddB[b][0][2];

                Krow[b]  += coeff_w * (dBa_x*dBb_x + dBa_y*dBb_y + dBa_z*dBb_z);
                Vrow[b]  += Ba_V_w * Bb;
                Mdrow[b] += Ba_invMass_w * Bb;
                Xrow[b]  += Ba_x_w * Bb;
                Pxrow[b] += -0.5 * PETSC_i * (Ba_w * dBb_x - dBa_x * Bb * w);

                PetscScalar dv = Ba_neg_dVdx_f_w * Bb;
                if (!massIsConstant) {
                    const PetscScalar row1 = two_fgrad2 * Ba * dBb_x;
                    const PetscScalar row2 = f_4fz * Ba * dBb_xz + fz_fxz * Ba * Bb;
                    const PetscScalar row3 = f_4fy * Ba * dBb_xy
                                           + f_2_lap * Ba * dBb_x
                                           + f_mix   * Ba * Bb;
                    const PetscScalar row4 = two_f_fx * (dBa_z*dBb_z + dBa_y*dBb_y - dBa_x*dBb_x);
                    const PetscScalar row5 = 2.0*(Ba*dBb_z*ffx_z + Ba*dBb_y*ffx_y - Ba*dBb_x*ffx_x);
                    const PetscScalar row6 = fy_fxy_fx_fxx * Ba * Bb;
                    dv -= 0.25 * (row1+row2+row3+row4+row5+row6) * w;
                }
                dVdxrow[b] += dv;
            }
        }

        if (PetscUnlikely(heap_used)) PetscFree3(dBx, dBy, dBz);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}






// Hamiltonian matrix: ∫ (ħ²/2m) ∇B_i · ∇B_j + V(x) B_i B_j dx

// ============================================================================
//  SINGLE-OPERATOR FORM KERNELS (were: form_functions.cpp)
// ============================================================================

#include "tdsez_internal.hpp"

PetscErrorCode TDSEZFormHamiltonian(IGAPoint p, PetscScalar*H, void *ctx)
{
    PetscFunctionBegin;
    (void)ctx;
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz); 
    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = xyz[2];


    // 0 derivative shape functions
    const PetscReal *B;
    const PetscReal invMass = (PetscReal) (1.0 / TDSEZParser::MassDist(x, y, z));
    // const PetscReal coeff = 0.5 * invMass * TDSEZParser::Hbar * TDSEZParser::Hbar;
    const PetscReal coeff = 0.5 * invMass;

    // 1st derivative shape functions + Hamiltonian matrix assembly
    if (dim == 1)
    {
        const PetscReal (*dB)[1];
        IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);

        PetscReal Vxyz = TDSEZParser::V(x);
        for (PetscInt a = 0; a < nen; ++a) 
        {
            PetscReal Na = B[a];
            PetscReal dNa = dB[a][0];


            // pointer to the start of row 'a'
            PetscScalar *Hrow = &H[a * nen];

            #pragma omp simd
            for (PetscInt b = 0; b < nen; ++b)
            {
                PetscReal Nb = B[b];
                PetscReal dNb = dB[b][0];
                PetscReal val = coeff * dNa * dNb + Vxyz * Na * Nb;
                Hrow[b] = val;
            }
        }

    }
    else if (__builtin_expect(dim == 2, 0))
    {
        const PetscReal (*dB)[2];
        IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);

        PetscReal Vxyz = TDSEZParser::V(x, y);

        for (PetscInt a = 0; a < nen; ++a) 
        {
            const PetscReal NaV = B[a] * Vxyz;
            const PetscReal dNaxC = dB[a][0] * coeff;
            const PetscReal dNayC = dB[a][1] * coeff;


            // pointer to the start of row 'a'
            PetscScalar *Hrow = &H[a * nen];

            // enable sim d vectorization
            #pragma omp simd
            for (PetscInt b = 0; b < nen; ++b)
            {
                Hrow[b] = (dNaxC * dB[b][0]) + (dNayC * dB[b][1]) + (NaV * B[b]);
            }
        }
    }
    else if (dim == 3)
    {
        const PetscReal (*dB)[3];
        IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
        const PetscReal V_val = TDSEZParser::V(x, y, z);

        // prefetch 
        PetscReal static_dBx[512], static_dBy[512], static_dBz[512];
        alignas(64) PetscReal *dBx = static_dBx, *dBy = static_dBy, *dBz = static_dBz;
        PetscBool heap_used = PETSC_FALSE;
        if (nen > 512) {
            PetscMalloc3(nen, &dBx, nen, &dBy, nen, &dBz);
            heap_used = PETSC_TRUE;
        }

        for(PetscInt i=0; i<nen; i++) {
            dBx[i] = dB[i][0]; dBy[i] = dB[i][1]; dBz[i] = dB[i][2];
        }

        for(PetscInt a=0; a<nen; a++)
        {
            const PetscReal daxC = dBx[a] * coeff;
            const PetscReal dayC = dBy[a] * coeff;
            const PetscReal dazC = dBz[a] * coeff;
            const PetscReal va   = B[a] * V_val;
            
            PetscScalar * __restrict__ rowA = &H[a * nen];

            #pragma omp simd
            for(PetscInt b=0; b<nen; b++)
            {
                rowA[b] = (daxC * dBx[b] + dayC * dBy[b] + dazC * dBz[b]) + (va * B[b]);
            }
        }

        if (heap_used) {
            PetscFree3(dBx, dBy, dBz);
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}





// Weak form for Lz operator: Lz = -i (x d/dy - y d/dx)
PetscErrorCode TDSEZFormLz(IGAPoint p, PetscScalar *L, void *ctx)
{
    PetscFunctionBegin;
    (void)ctx;

    const PetscInt nen = p->nen;

    if (p->dim != 2) {
        for (PetscInt i = 0; i < nen * nen; ++i) L[i] = 0.0;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz);

    const PetscReal x = xyz[0];
    const PetscReal y = xyz[1];

    const PetscReal *B = NULL;
    IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B);

    const PetscReal (*dB)[2] = NULL;
    IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB);

    enum { LZ_STACK_NEN = 128 };
    PetscReal lzb_stack[LZ_STACK_NEN];
    PetscReal *lzb = lzb_stack;

    if (nen > LZ_STACK_NEN) {
        PetscErrorCode ierr = PetscMalloc1(nen, &lzb);
        CHKERRQ(ierr);
    }

    for (PetscInt b = 0; b < nen; ++b) {
        lzb[b] = x * dB[b][1] - y * dB[b][0];
    }

    for (PetscInt a = 0; a < nen; ++a) {
        const PetscScalar scale = -PETSC_i * B[a];
        PetscScalar *Lrow = L + a * nen;

        for (PetscInt b = 0; b < nen; ++b) {
            Lrow[b] = scale * lzb[b];
        }
    }

    if (lzb != lzb_stack) {
        PetscErrorCode ierr = PetscFree(lzb);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// computing  mat of K, V, Separately to reuse in analysis
PetscErrorCode TDSEZFormKinetic(IGAPoint p, PetscScalar*K, void *ctx)
{
    PetscFunctionBegin;
    (void)ctx;
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz); 
    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = xyz[2];

    // 0 derivative shape functions
    const PetscReal *B;
    (void)B;
    const PetscReal invMass = (PetscReal) (1.0 / TDSEZParser::MassDist(x, y, z));
    const PetscReal coeff = 0.5 * invMass;

    // 1st derivative shape functions + Hamiltonian matrix assembly
    if (dim == 1)
    {
        const PetscReal (*dB)[1];
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);

        for (PetscInt a = 0; a < nen; ++a) 
        {
            PetscReal dBa = dB[a][0];

            // pointer to the start of row 'a'
            PetscScalar *Krow = &K[a * nen];

            #pragma omp simd
            for (PetscInt b = 0; b < nen; ++b)
            {
                PetscReal dBb = dB[b][0];
                PetscReal val = coeff * dBa * dBb;
                Krow[b] = val;
            }
        }

    }
    else if (__builtin_expect(dim == 2, 0))
    {
        const PetscReal (*dB)[2];
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);

        for (PetscInt a = 0; a < nen; ++a) 
        {
            const PetscReal dBa_x = dB[a][0] * coeff;
            const PetscReal dBa_y = dB[a][1] * coeff;

            // pointer to the start of row 'a'
            PetscScalar *Krow = &K[a * nen];

            // enable sim d vectorization
            #pragma omp simd
            for (PetscInt b = 0; b < nen; ++b)
            {
                Krow[b] = (dBa_x * dB[b][0]) + (dBa_y * dB[b][1]);
            }
        }
    }
    else if (dim == 3)
    {
        const PetscReal (*dB)[3];
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
        for(PetscInt a=0; a<nen; a++)
        {
            // hoist terms outside inner loop
            const PetscReal dBa_x = dB[a][0] * coeff;
            const PetscReal dBa_y = dB[a][1] * coeff;
            const PetscReal dBa_z = dB[a][2] * coeff;

            // pointer to the start of row 'a'
            PetscScalar *Krow = &K[a * nen];

            for(PetscInt b=0; b<nen; b++)
            {
                Krow[b] = (dBa_x * dB[b][0] + dBa_y * dB[b][1] + dBa_z * dB[b][2]);
            }
        }
    }

        PetscFunctionReturn(PETSC_SUCCESS);
}




PetscErrorCode TDSEZFormPotential(IGAPoint p, PetscScalar*V, void *ctx)
{
    PetscFunctionBegin;
    (void)ctx;
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;
    (void)dim;

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz); 
    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = xyz[2];


    // 0 derivative shape functions
    const PetscReal *B;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);

    PetscReal Vxyz = TDSEZParser::V(x, y, z);
    for (PetscInt a = 0; a < nen; ++a) 
    {
        PetscReal Ba = B[a] * Vxyz; 

        PetscScalar *Vrow = &V[a * nen];

        for (PetscInt b = 0; b < nen; ++b) 
        {
            PetscReal Bb = B[b];
            Vrow[b] = Ba * Bb;
        }
    }

        PetscFunctionReturn(PETSC_SUCCESS);
}




// Mass matrix: ∫ Ba * Bb dx
PetscErrorCode TDSEZFormMass(IGAPoint p, PetscScalar* __restrict__ M, void *ctx) 
{
    (void)ctx;
    const PetscInt nen = p->nen;
    const PetscReal * __restrict__ B = (const PetscReal *)p->shape[0];
    alignas(64) PetscReal BB[512]; 

    for (PetscInt i = 0; i < nen; i++) {
        BB[i] = B[i];
    }

    for (PetscInt a = 0; a < nen; a++) {   
        const PetscReal Ba = BB[a];
        PetscScalar * __restrict__ Mrow = &M[a * nen];
        #pragma GCC ivdep
        for (PetscInt b = 0; b < nen; b++) {
            Mrow[b] = Ba * BB[b];
        }
    }
    return PETSC_SUCCESS;
}


// form mass distribution matrix:  ∫ (1/m(r)) Ba * Bb dr
PetscErrorCode TDSEZFormMassDist(IGAPoint p, PetscScalar*Md, void *ctx) 
{
    (void)ctx;
    PetscInt nen = p->nen;
    const PetscReal *B;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B); 

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz); 
    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = xyz[2];
    PetscReal invMass = (PetscReal) (1.0 / TDSEZParser::MassDist(x, y, z));
    for (PetscInt a = 0; a < nen; a++) 
    {   
        PetscReal Ba = B[a];
        PetscScalar *Mrow = &Md[a * nen];

        for (PetscInt b = 0; b < nen; b++) 
        {
            Mrow[b] = Ba * B[b] * invMass;
        }
    }

    return 0;
}





PetscErrorCode DipoleX(IGAPoint p, PetscScalar *X, void *ctx) 
{
    (void)ctx;
    PetscInt nen = p->nen;
    const PetscReal *B;
    IGAPointGetShapeFuns(p, 0, &B);

    // only the X component
    PetscReal r[3];          
    IGAPointFormGeomMap(p, r);  
    PetscReal x = r[0]; 

    for (PetscInt a = 0; a < nen; a++) 
    {
        PetscReal val_a = B[a] * x;         
        for (PetscInt b = 0; b < nen; b++) 
        {
            X[a * nen + b] = val_a * B[b];
        }
    }
    return 0;
}




PetscErrorCode DipoleY(IGAPoint p, PetscScalar*Y, void *ctx)
{
    (void)ctx;
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;
    (void)dim;
    const PetscReal *B;
    IGAPointGetShapeFuns(p,0,(const PetscReal**)&B);

    // get physical coordinates of the quadrature point
    PetscReal r[3] = {0.0, 0.0, 0.0};         
    IGAPointFormGeomMap(p, r);  
    PetscReal y = r[1];

    for (PetscInt a = 0; a < nen; a++) 
    {
        for (PetscInt b = 0; b < nen; b++) 
        {
            PetscReal term = B[a] * y * B[b]; 
            Y[a * nen + b] = term;
        }
    }
    return 0;
}



PetscErrorCode DipoleZ(IGAPoint p, PetscScalar*Z, void *ctx) 
{
    (void)ctx;
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;
    (void)dim;
    const PetscReal *B;
    IGAPointGetShapeFuns(p,0,(const PetscReal**)&B);

    // get physical coordinates of the quadrature point
    PetscReal r[3] = {0.0, 0.0, 0.0};         
    IGAPointFormGeomMap(p, r);  
    PetscReal z = r[2]; 

    for (PetscInt a = 0; a < nen; a++) 
    {
        for (PetscInt b = 0; b < nen; b++) 
        {
            PetscReal term = B[a] * z * B[b]; 
            Z[a * nen + b] = term;
        }
    }
    return 0;
}


PetscErrorCode VelocityX(IGAPoint p, PetscScalar *P, void *ctx)
{
    (void)ctx;
    PetscFunctionBegin;
    const PetscInt dir = 0;
    const PetscInt nen = p->nen;
    const PetscInt dim = p->dim;

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz);

    const PetscReal   invMass = 1.0 / TDSEZParser::MassDist(xyz[0], xyz[1], xyz[2]);
    const PetscScalar coeff   = -0.5 * invMass * PETSC_i;

    const PetscReal *B;
    const PetscReal *dB_flat;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
    IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB_flat);

    for (PetscInt a = 0; a < nen; a++)
    for (PetscInt b = 0; b < nen; b++)
    {
        // Anti-symmetric by construction: P_ab = -P_ba
        P[a*nen + b] = coeff * ( B[a]                 * dB_flat[b*dim + dir]
                               - dB_flat[a*dim + dir] * B[b] );
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}




PetscErrorCode VelocityY(IGAPoint p, PetscScalar *P, void *ctx)
{
    (void)ctx;
    PetscFunctionBegin;
    const PetscInt dir = 1; // y-direction
    const PetscInt nen   = p->nen;
    const PetscInt dim   = p->dim;

    // Effective mass at quadrature point
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz);
    const PetscReal   invMass = 1.0 / TDSEZParser::MassDist(xyz[0], xyz[1], xyz[2]);
    const PetscScalar coeff   = -0.5 * invMass * PETSC_i;

    // dB is a flat array of size nen*dim
    const PetscReal *B;
    const PetscReal *dB_flat;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
    IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB_flat);

    for (PetscInt a = 0; a < nen; a++)
    {
        const PetscScalar scaledBa  =  B[a]                 * coeff;
        const PetscScalar scaleddBa =  dB_flat[a*dim + dir] * coeff;
        for (PetscInt b = 0; b < nen; b++)
        {
            P[a*nen + b] = scaledBa * dB_flat[b*dim + dir] - scaleddBa * B[b];
        }
    }


    PetscFunctionReturn(PETSC_SUCCESS);
}




PetscErrorCode VelocityZ(IGAPoint p, PetscScalar *P, void *ctx)
{
    (void)ctx;
    PetscFunctionBegin;
    const PetscInt dir = 2; // z-direction
    const PetscInt nen   = p->nen;
    const PetscInt dim   = p->dim;

    // Effective mass at quadrature point
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz);
    const PetscReal   invMass = 1.0 / TDSEZParser::MassDist(xyz[0], xyz[1], xyz[2]);
    const PetscScalar coeff   = -0.5 * invMass * PETSC_i;

    // dB is a flat array of size nen*dim
    const PetscReal *B;
    const PetscReal *dB_flat;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
    IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB_flat);

    for (PetscInt a = 0; a < nen; a++)
    {
        const PetscScalar scaledBa  =  B[a]                 * coeff;
        const PetscScalar scaleddBa =  dB_flat[a*dim + dir] * coeff;
        for (PetscInt b = 0; b < nen; b++)
        {
            P[a*nen + b] = scaledBa * dB_flat[b*dim + dir] - scaleddBa * B[b];
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}




// momentum matrix: -ihbar∫ Ba * Grad * Bb dx
PetscErrorCode MomentumBase(IGAPoint p, PetscScalar *P, void *ctx) 
{
    // get direction from context
    PetscInt dir = *(PetscInt*)ctx;
    PetscInt nen = p->nen;

    const PetscReal *B;
    const PetscReal (*dB)[3];
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
    IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
    
    PetscArrayzero(P, nen*nen);

    if (dir < 0 || dir > 2)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid direction");

    for (PetscInt a = 0; a < nen; a++) 
    {
        for (PetscInt b = 0; b < nen; b++) 
        {
            P[a * nen + b] =  -PETSC_i * B[a] * dB[b][dir]; 
        }
    }
    return 0;
}







// momentum matrix: -ihbar∫ Ba * Grad * Bb dx
PetscErrorCode TDSEZformPotentialGradX(IGAPoint p, PetscScalar *dVdr, void *ctx) 
{
    (void)ctx;
    // get direction from context
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;

    const PetscReal *B;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
    PetscReal xyz[3] = {0.0,0.0,0.0};
             
    IGAPointFormGeomMap(p, xyz);
    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = xyz[2];


    // Mass grad and laplacian terms
    auto invMassFunc = [](const std::vector<PetscReal>& xx) {
        return 1.0/TDSEZParser::MassDist(xx[0], xx[1], xx[2]);
    };

    if(dim == 1)
    {
        PetscReal dVdx = TDSEZParser::dVx(x);
        const PetscReal (*dB)[1];
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
    
        PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x}, 1e-4);
        PetscReal f    = invMassFunc({x});
        PetscReal fx   = ops.first[0];
        PetscReal fxx  = ops.second[0]; 
        PetscReal fxxx = ops.third[0];  
    
        for (PetscInt a = 0; a < nen; a++) 
        {
            PetscScalar Ba   = B[a];     
            PetscScalar dBax = dB[a][0];
        
            for (PetscInt b = 0; b < nen; b++) 
            {
                PetscScalar Bb   = B[b];
                PetscScalar dBbx = dB[b][0]; 
    
                // Row 0: -Ba * dVdx * f * Bb
                PetscScalar classical = -Ba * dVdx * f * Bb;
                // Row 1: 2 * Bi * Bj_x * fx^2
                PetscScalar R1 = 2.0 * Ba * dBbx * (fx * fx);
                // Row 3: f * [ 2 * Bi * Bj_x * fxx + Bi * Bb * fxxx ]
                PetscScalar R3 = f * (2.0 * Ba * dBbx * fxx + Ba * Bb * fxxx);
                // Row 4: -2 * f * fx * (dBax * dBbx)
                PetscScalar R4 = -2.0 * f * fx * (dBax * dBbx);
                // Row 5: -2 * Bi * Bj_x * (fx*fx + f*fxx)
                PetscScalar R5 = -2.0 * Ba * dBbx * (fx*fx + f*fxx);
                // Row 6: Bi * Bj * fx * fxx
                PetscScalar R6 = Ba * Bb * (fx * fxx);
                PetscScalar quantum = -0.25 * (R1 + R3 + R4 + R5 + R6);

                dVdr[a * nen + b] = classical + quantum;
            }
        }
    }
    else if(dim == 2)
    {
        PetscReal dVdx = TDSEZParser::dVx(x,y);

        const PetscReal (*dB)[2], (*ddB)[2][2];
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
        IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);

        // Mass derivatives (assuming dim=2, so 3rd order tensor size is 2^3=8)
        PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x, y}, 1e-4);
        PetscReal f    = invMassFunc({x, y});
        PetscReal fx   = ops.first[0];
        PetscReal fy   = ops.first[1];
        PetscReal fxx  = ops.second[0 * 2 + 0];
        PetscReal fyy  = ops.second[1 * 2 + 1];
        PetscReal fxy  = ops.second[0 * 2 + 1]; // fxy = fyx
        PetscReal fxxx = ops.third[(0 * 2 + 0) * 2 + 0];
        PetscReal fxyy = ops.third[(0 * 2 + 1) * 2 + 1];

        for (PetscInt a = 0; a < nen; a++) 
        {
            PetscScalar Ba = B[a];     
            PetscScalar dBax = dB[a][0];
            PetscScalar dBay = dB[a][1]; 
    
            for (PetscInt b = 0; b < nen; b++) 
            {
                PetscScalar Bb = B[b];
                PetscScalar dBbx = dB[b][0]; 
                PetscScalar dBby = dB[b][1]; 
                PetscScalar dBbxy = ddB[b][0][1];

                // Row 0: -Ba * dVdx * f * Bb
                PetscScalar classical = -Ba * dVdx * f * Bb;
                
                // Row 1: 2 * Bi * Bj_x * (fx^2 + fy^2)
                PetscScalar R1 = 2.0 * Ba * dBbx * (fx*fx + fy*fy);

                // Row 3: f * ( 4*fy*Bi*Bj_xy + 2*Bi*Bj_x*(fyy+fxx) + Bi*Bj*(fxyy+fxxx) )
                PetscScalar R3 = f * (4.0 * fy * Ba * dBbxy + 
                                      2.0 * Ba * dBbx * (fyy + fxx) + 
                                      Ba * Bb * (fxyy + fxxx));

                // Row 4: 2 * f * fx * (Bay*Bby - Bax*Bbx)
                PetscScalar R4 = 2.0 * f * fx * (dBay * dBby - dBax * dBbx);

                // Row 5: 2 * ( Bi*Bby*(fy*fx + f*fxy) - Bi*Bbx*(fx*fx + f*fxx) )
                PetscScalar R5 = 2.0 * (Ba * dBby * (fy*fx + f*fxy) - 
                                        Ba * dBbx * (fx*fx + f*fxx));

                // Row 6: Bi * Bj * (fy*fxy + fx*fxx)
                PetscScalar R6 = Ba * Bb * (fy * fxy + fx * fxx);

                PetscScalar quantum = -0.25 * (R1 + R3 + R4 + R5 + R6);

                dVdr[a * nen + b] = classical + quantum;
            }
        }
    }
    else if(dim == 3)
    {
        PetscReal dVdx = TDSEZParser::dVx(x, y, z);
        const PetscReal (*dB)[3], (*ddB)[3][3];
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
        IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);
    
        // 1. Derivatives of f (Inverse Mass) - Index mapping: i*dim + j
        PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x, y, z}, 1e-4);
        PetscReal f    = invMassFunc({x, y, z});
        PetscReal fx = ops.first[0], fy = ops.first[1], fz = ops.first[2];
        
        // 2nd Derivatives
        PetscReal fxx = ops.second[0*3 + 0], fyy = ops.second[1*3 + 1], fzz = ops.second[2*3 + 2];
        PetscReal fxy = ops.second[0*3 + 1], fxz = ops.second[0*3 + 2], fyz = ops.second[1*3 + 2];
        (void)fyz;
    
        // 3rd Derivatives - Index mapping: (i*dim + j)*dim + k
        PetscReal fxxx = ops.third[(0*3 + 0)*3 + 0];
        PetscReal fxyy = ops.third[(0*3 + 1)*3 + 1];
        PetscReal fxzz = ops.third[(0*3 + 2)*3 + 2];
    
        // Row 5 Components: (f*fx)_i = f_i*f_x + f*f_ix
        PetscReal ffx_x = fx*fx + f*fxx;
        PetscReal ffx_y = fy*fx + f*fxy;
        PetscReal ffx_z = fz*fx + f*fxz;
    
        for (PetscInt a = 0; a < nen; a++) {
            PetscScalar Ba = B[a], dBax = dB[a][0], dBay = dB[a][1], dBaz = dB[a][2];
    
            for (PetscInt b = 0; b < nen; b++) {
                PetscScalar Bb = B[b], dBbx = dB[b][0], dBby = dB[b][1], dBbz = dB[b][2];
                PetscScalar dBbxy = ddB[b][0][1], dBbxz = ddB[b][0][2];
    
                // Row 0: -Ba * dVdx * f * Bb
                PetscScalar classical = -Ba * dVdx * f * Bb;
                    
                // Row 1: 2 * Bi * Bj_x * (fx^2 + fy^2 + fz^2)
                PetscScalar row1 = 2.0 * Ba * dBbx * (fx*fx + fy*fy + fz*fz);
    
                // Row 2: fz * (4*f*Bi*Bj_xz + Bi*Bj*fxz)
                PetscScalar row2 = fz * (4.0 * f * Ba * dBbxz + Ba * Bb * fxz);
    
                // Row 3: f * [4*fy*Bi*Bj_xy + 2*Bi*Bj_x*(fxx+fyy+fzz) + Bi*Bj*(fxxx+fxyy+fxzz)]
                PetscScalar row3 = f * (4.0 * fy * Ba * dBbxy + 
                                        2.0 * Ba * dBbx * (fxx + fyy + fzz) + 
                                        Ba * Bb * (fxxx + fxyy + fxzz));
    
                // Row 4: 2*f*fx * (Baz*Bbz + Bay*Bby - Bax*Bbx)
                PetscScalar row4 = 2.0 * f * fx * (dBaz * dBbz + dBay * dBby - dBax * dBbx);
    
                // Row 5: 2 * [ Ba*dBbz*ffx_z + Ba*dBby*ffx_y - Ba*dBbx*ffx_x ]
                PetscScalar row5 = 2.0 * (Ba * dBbz * ffx_z + Ba * dBby * ffx_y - Ba * dBbx * ffx_x);

                // Row 6: Bi * Bj * (fy*fxy + fx*fxx)
                PetscScalar row6 = Ba * Bb * (fy * fxy + fx * fxx);
    
                PetscScalar quantum = -0.25 * (row1 + row2 + row3 + row4 + row5 + row6);
    
                dVdr[a * nen + b] = classical + quantum;
            }
        }
    }

    return 0;
}



PetscErrorCode TDSEZformPotentialGradY(IGAPoint p, PetscScalar *dVdr, void *ctx) 
{
    (void)ctx;
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;

    const PetscReal *B;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
    PetscReal xyz[3] = {0.0,0.0,0.0};
    IGAPointFormGeomMap(p, xyz);
    PetscReal x = xyz[0], y = xyz[1], z = xyz[2];

    auto invMassFunc = [](const std::vector<PetscReal>& xx) {
        return 1.0/TDSEZParser::MassDist(xx[0], (xx.size() > 1 ? xx[1] : 0.0), (xx.size() > 2 ? xx[2] : 0.0));
    };

    if(dim == 1) 
    {
        for (PetscInt i = 0; i < nen * nen; i++) dVdr[i] = 0.0;
    }
    else if(dim == 2 || dim == 3) 
    {
        PetscReal dVdy = TDSEZParser::dVy(x, y, z);
        const PetscReal (*dB)[dim], (*ddB)[dim][dim];
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
        IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);

        PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, (dim==2 ? std::vector<PetscReal>{x,y} : std::vector<PetscReal>{x,y,z}), 1e-4);
        
        PetscReal f = invMassFunc((dim==2 ? std::vector<PetscReal>{x,y} : std::vector<PetscReal>{x,y,z}));
        PetscReal fx = ops.first[0], fy = ops.first[1], fz = (dim==3 ? ops.first[2] : 0.0);
        
        PetscReal fxx = ops.second[0*dim+0], fyy = ops.second[1*dim+1], fzz = (dim==3 ? ops.second[2*dim+2] : 0.0);
        PetscReal fxy = ops.second[0*dim+1], fxz = (dim==3 ? ops.second[0*dim+2] : 0.0), fyz = (dim==3 ? ops.second[1*dim+2] : 0.0);
        (void)fxz;

        // 3rd derivatives for Row 3: dy(Laplacian f)
        PetscReal fyyy = ops.third[(1*dim+1)*dim+1];
        PetscReal fxxy = ops.third[(0*dim+0)*dim+1];
        PetscReal fyzz = (dim==3 ? ops.third[(1*dim+2)*dim+2] : 0.0);

        // Row 5 Helpers: (f*fy)_i
        PetscReal ffy_x = fx*fy + f*fxy;
        PetscReal ffy_y = fy*fy + f*fyy;
        PetscReal ffy_z = (dim==3 ? fz*fy + f*fyz : 0.0);

        for (PetscInt a = 0; a < nen; a++) 
        {
            for (PetscInt b = 0; b < nen; b++) 
            {
                PetscScalar Ba = B[a], Bb = B[b];
                PetscScalar dBax = dB[a][0], dBay = dB[a][1], dBaz = (dim==3 ? dB[a][2] : 0.0);
                PetscScalar dBbx = dB[b][0], dBby = dB[b][1], dBbz = (dim==3 ? dB[b][2] : 0.0);
                
                // ddB indices: [basis][deriv1][deriv2]
                PetscScalar dBbxy = ddB[b][0][1];
                PetscScalar dBbyz = (dim==3 ? ddB[b][1][2] : 0.0);

                PetscScalar classical = -Ba * dVdy * f * Bb;

                // Row 1: 2 * Bi * Bj_y * |grad f|^2
                PetscScalar r1 = 2.0 * Ba * dBby * (fx*fx + fy*fy + fz*fz);

                // Row 2: fz * (4*f*Bi*Bj_yz + Bi*Bb*fyz)
                PetscScalar r2 = fz * (4.0 * f * Ba * dBbyz + Ba * Bb * fyz);

                // Row 3: f * (4*fx*Bi*Bj_xy + 2*Bi*Bj_y*Laplacian + Bi*Bb*dy(Laplacian))
                PetscScalar r3 = f * (4.0 * fx * Ba * dBbxy + 
                                      2.0 * Ba * dBby * (fxx + fyy + fzz) + 
                                      Ba * Bb * (fxxy + fyyy + fyzz));

                // Row 4: 2*f*fy * (Baz*Bbz - Bay*Bby + Bax*Bbx)
                PetscScalar r4 = 2.0 * f * fy * (dBaz * dBbz - dBay * dBby + dBax * dBbx);

                // Row 5: 2 * (Bi*Bbz*(ffy)_z - Bi*Bby*(ffy)_y + Bi*Bbx*(ffy)_x)
                PetscScalar r5 = 2.0 * (Ba * dBbz * ffy_z - Ba * dBby * ffy_y + Ba * dBbx * ffy_x);

                // Row 6: Bi * Bb * (fy*fyy + fx*fxy)
                PetscScalar r6 = Ba * Bb * (fy * fyy + fx * fxy);

                PetscScalar quantum = -0.25 * (r1 + r2 + r3 + r4 + r5 + r6);
                dVdr[a * nen + b] = classical + quantum;
            }
        }
    }
    return 0;
}



PetscErrorCode TDSEZformPotentialGradZ(IGAPoint p, PetscScalar *dVdr, void *ctx) 
{
    (void)ctx;
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;

    // dz is only defined for 3D systems
    if(dim < 3) 
    {
        for (PetscInt i = 0; i < nen * nen; i++) dVdr[i] = 0.0;
        return 0;
    }

    const PetscReal *B;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);
    PetscReal xyz[3] = {0.0,0.0,0.0};
    IGAPointFormGeomMap(p, xyz);
    PetscReal x = xyz[0], y = xyz[1], z = xyz[2];

    auto invMassFunc = [](const std::vector<PetscReal>& xx) 
    {
        return 1.0/TDSEZParser::MassDist(xx[0], xx[1], xx[2]);
    };

    PetscReal dVdz = TDSEZParser::dVz(x, y, z);
    const PetscReal (*dB)[3], (*ddB)[3][3];
    IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
    IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);

    PhysicsDerivativesFull ops = TDSEZCompDerivative(invMassFunc, {x, y, z}, 1e-4);
    
    PetscReal f = invMassFunc({x, y, z});
    PetscReal fx = ops.first[0], fy = ops.first[1], fz = ops.first[2];
    PetscReal fxx = ops.second[0*3+0], fyy = ops.second[1*3+1], fzz = ops.second[2*3+2];
    PetscReal fxy = ops.second[0*3+1], fxz = ops.second[0*3+2], fyz = ops.second[1*3+2];
    (void)fxy;

    // 3rd derivatives: dz(Laplacian f)
    PetscReal fzzz = ops.third[(2*3+2)*3+2];
    PetscReal fyyz = ops.third[(1*3+1)*3+2];
    PetscReal fxxz = ops.third[(0*3+0)*3+2];

    // Row 5 Helpers: (f*fz)_i
    PetscReal ffz_x = fx*fz + f*fxz;
    PetscReal ffz_y = fy*fz + f*fyz;
    PetscReal ffz_z = fz*fz + f*fzz;

    for (PetscInt a = 0; a < nen; a++) 
    {
        for (PetscInt b = 0; b < nen; b++) 
        {
            PetscScalar Ba = B[a], Bb = B[b];
            PetscScalar dBax = dB[a][0], dBay = dB[a][1], dBaz = dB[a][2];
            PetscScalar dBbx = dB[b][0], dBby = dB[b][1], dBbz = dB[b][2];
            
            PetscScalar dBbyz = ddB[b][1][2];
            PetscScalar dBbxz = ddB[b][0][2];

            PetscScalar classical = -Ba * dVdz * f * Bb;

            // Row 1: 2 * Bi * Bj_z * |grad f|^2
            PetscScalar r1 = 2.0 * Ba * dBbz * (fx*fx + fy*fy + fz*fz);

            // Row 2 & 3 Combined: f * (4fy*Bi*Bj_yz + 4fx*Bi*Bj_xz + 2*Bi*Bj_z*Laplacian + Bi*Bb*dz(Laplacian))
            PetscScalar r23 = f * (4.0 * fy * Ba * dBbyz + 
                                   4.0 * fx * Ba * dBbxz + 
                                   2.0 * Ba * dBbz * (fxx + fyy + fzz) + 
                                   Ba * Bb * (fzzz + fyyz + fxxz));

            // Row 4: -2*f*fz * (Baz*Bbz - Bay*Bby - Bax*Bbx)
            PetscScalar r4 = -2.0 * f * fz * (dBaz * dBbz - dBay * dBby - dBax * dBbx);

            // Row 5: -2 * (Bi*Bbz*(ffz)_z - Bi*Bby*(ffz)_y - Bi*Bbx*(ffz)_x)
            PetscScalar r5 = -2.0 * (Ba * dBbz * ffz_z - Ba * dBby * ffz_y - Ba * dBbx * ffz_x);

            // Row 6: Bi * Bb * (fz*fzz + fy*fyz + fx*fxz)
            PetscScalar r6 = Ba * Bb * (fz * fzz + fy * fyz + fx * fxz);

            PetscScalar quantum = -0.25 * (r1 + r23 + r4 + r5 + r6);
            dVdr[a * nen + b] = classical + quantum;
        }
    }
    return 0;
}



// Dipole matrix: ∫ Ba * dV/dx * Bb dx
PetscErrorCode TDSEZformPotentialGrad_old(IGAPoint p, PetscScalar*dV, void *ctx) 
{
    (void)ctx;
    PetscInt nen = p->nen;
    PetscInt dim = p->dim;
    
    const PetscReal *B;
    IGAPointGetShapeFuns(p,0,(const PetscReal**)&B);

    PetscReal x[3] = {0.0,0.0,0.0};          
    IGAPointFormGeomMap(p, x);       // x[0..dim-1]
    PetscReal grad[3] = {0.0,0.0,0.0};

#ifdef USE_ANALYTIC
if(__builtin_expect(dim==1, 1)) grad[0] = TDSEZParser::dV(x[0]);
else if(__builtin_expect(dim==2, 1)) 
{
    PetscReal term = TDSEZParser::dV(x[0], x[1]);
    grad[0] = term; 
    grad[1] = term;  
} 
else if (__builtin_expect(dim==3, 1))
{ // dim==3
    PetscReal term = TDSEZParser::dV(x[0], x[1], x[2]);
    grad[0] = term;
    grad[1] = term;
    grad[2] = term;
}
#else
    PetscReal h = 1e-5;
    for (PetscInt i=0; i<dim; ++i) 
    {
        PetscReal x_plus[3]  = {x[0], x[1], x[2]};
        PetscReal x_minus[3] = {x[0], x[1], x[2]};
        PetscReal x_plus2[3] = {x[0], x[1], x[2]};
        PetscReal x_minus2[3]= {x[0], x[1], x[2]};

        x_plus[i]  += h; x_minus[i] -= h;
        x_plus2[i] += 2*h; x_minus2[i]-= 2*h;

        // call TDSEZParser::V with explicit components based on dim
        PetscReal Vpp = 0.0, Vp = 0.0, Vm = 0.0, Vmm = 0.0;

        if(__builtin_expect(dim==1, 1)) 
        {
            Vpp = TDSEZParser::V(x_plus2[0]);
            Vp  = TDSEZParser::V(x_plus[0]);
            Vm  = TDSEZParser::V(x_minus[0]);
            Vmm = TDSEZParser::V(x_minus2[0]);
        } 
        else if((__builtin_expect(dim==2, 1)))
        {
            Vpp = TDSEZParser::V(x_plus2[0], x_plus2[1]);
            Vp  = TDSEZParser::V(x_plus[0], x_plus[1]);
            Vm  = TDSEZParser::V(x_minus[0], x_minus[1]);
            Vmm = TDSEZParser::V(x_minus2[0], x_minus2[1]);
        }
        else if ((__builtin_expect(dim==3, 1)))
        { 
            Vpp = TDSEZParser::V(x_plus2[0], x_plus2[1], x_plus2[2]);
            Vp  = TDSEZParser::V(x_plus[0], x_plus[1], x_plus[2]);
            Vm  = TDSEZParser::V(x_minus[0], x_minus[1], x_minus[2]);
            Vmm = TDSEZParser::V(x_minus2[0], x_minus2[1], x_minus2[2]);
        }

        grad[i] = (-Vpp + 8*Vp - 8*Vm + Vmm) / (12*h);
    }
#endif

    // assemble dV/dx
    for (PetscInt a=0; a<nen; ++a) 
    {
        for (PetscInt b=a; b<nen; ++b) 
        {

            PetscReal Nab = B[a]*B[b];

            PetscReal val = 0.0;
            for (PetscInt i=0; i<dim; ++i) val += Nab * grad[i];
            dV[a*nen + b] = val;
            dV[b*nen + a] = val;
        }
    }

    return 0;
}


// CAP matrix: ∫ Ba * W(x) * Bb dx
PetscErrorCode TDSEZformCap(IGAPoint p, PetscScalar *CAP, void *ctx)
{
    (void)ctx;
    PetscInt nen = p->nen;
    const PetscReal *B;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz);

    // square (2D) or cube (3D)
    PetscReal r_cubic = PetscAbsReal(xyz[0]);
    for (PetscInt i = 1; i < TDSEZParser::Dimension; i++) {
        r_cubic = PetscMax(r_cubic, PetscAbsReal(xyz[i]));
    }

    // setup Boundaries
    PetscReal R_outer = PetscAbsReal(TDSEZParser::LMaxX);
    PetscReal R_inner = 0.90 * R_outer;

    PetscReal Wq = 0.0;
    if (r_cubic > R_inner) {
        // clamp to avoid the 1/(R_outer - r) singularity
        PetscReal r_safe = PetscMin(r_cubic, R_outer - 1e-8);
        Wq = Manolopoulos_CAP_profile(r_safe, R_inner, R_outer, TDSEZParser::CAPKmin);
    }

    const PetscScalar iWq = -PETSC_i * Wq;

    for (PetscInt a = 0; a < nen; a++)
    {
        const PetscScalar row_val = iWq * B[a];
        const PetscInt row_idx = a * nen;

        for (PetscInt b = 0; b < nen; b++)
        {
            CAP[row_idx + b] = row_val * B[b];
        }
    }

    return 0;
}
