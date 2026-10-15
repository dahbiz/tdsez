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
//    - TDSEZFormLz, TDSEZFormKinetic, TDSEZFormPotential,
//      TDSEZFormMassDist   (single-operator form kernels)
//    - DipoleX/Y/Z, VelocityX/Y/Z, TDSEZformPotentialGradX/Y/Z,
//      TDSEZformCap   (form kernels)
//
//  NO logic was rewritten vs the prior split files (form_functions.cpp,
//  physics_kernels.cpp, assembler.cpp, and the assembly parts of
//  diagnostics.cpp). The split was purely organizational.
// ============================================================================

/**
 * @file assembly.cpp
 * @brief All PetIGA form-kernels and the core element-loop assembler for
 *        TDSEZ: turns physics (potential, mass, dipole, velocity, gradient,
 *        CAP) into PETSc Mat/Vec objects for 1D/2D/3D B-spline IGA.
 */

#include "tdsez_internal.hpp"

#include <cstdint>
#include <cstring>
#include <map>

const PhysicsDerivativesFull &TDSEZInvMassDerivatives(const std::vector<PetscReal> &x)
{
    thread_local std::vector<PetscReal> key;
    thread_local PhysicsDerivativesFull last;
    if (x == key) return last;
    const int dim = (int)x.size();
    if (dim > 3) {
        std::map<std::vector<PetscReal>, PetscReal> seen;
        auto invMass = [&seen](const std::vector<PetscReal> &xx) {
            auto it = seen.find(xx);
            if (it != seen.end()) return it->second;
            const PetscReal v = 1.0 / TDSEZParser::MassDist(xx[0], xx[1], xx[2]);
            seen.emplace(xx, v);
            return v;
        };
        last = TDSEZCompDerivative(invMass, x, 1e-4);
        key = x;
        return last;
    }
    // TDSEZCompDerivative(1/MassDist, x, 1e-4) without allocations: the same
    // stencil points built with the same arithmetic, the same formulas, and 1/m
    // evaluated once per distinct point (first point of equal coordinates, as
    // the std::map lookup it replaces), so the values are bitwise identical.
    struct Slot { PetscReal r[3]; PetscReal v; std::uint64_t gen; };
    constexpr unsigned NS = 512;                       // >= 2 x 264 stencil points (3D)
    thread_local Slot table[NS] = {};
    thread_local std::uint64_t gen = 0;
    ++gen;                                             // slots of earlier points are empty
    auto F = [dim](const PetscReal *r) {
        PetscReal c[3] = {0.0, 0.0, 0.0};
        for (int d = 0; d < dim; ++d) c[d] = r[d] + 0.0;   // -0 == +0, as the map compares
        std::uint64_t hsh = 1469598103934665603ull;
        for (int d = 0; d < 3; ++d) {
            std::uint64_t b;
            std::memcpy(&b, &c[d], sizeof b);
            hsh = (hsh ^ b) * 1099511628211ull;
        }
        for (unsigned k = (unsigned)(hsh ^ (hsh >> 29)) & (NS - 1);; k = (k + 1) & (NS - 1)) {
            Slot &s = table[k];
            if (s.gen != gen) {
                s.gen = gen;
                std::memcpy(s.r, r, sizeof(PetscReal) * dim);
                s.v = 1.0 / TDSEZParser::MassDist(r[0], (dim > 1 ? r[1] : 0.0), (dim > 2 ? r[2] : 0.0));
                return s.v;
            }
            bool eq = true;
            for (int d = 0; d < dim; ++d) eq = eq && (s.r[d] == r[d]);
            if (eq) return s.v;
        }
    };
    const PetscReal h = 1e-4;
    const PetscReal inv_h = 1.0 / h;
    const PetscReal inv_12h = inv_h / 12.0;
    const PetscReal inv_4h2 = 1.0 / (4.0 * h * h);
    last.first.resize(dim);
    last.second.resize(dim * dim);
    last.third.resize(dim * dim * dim);
    PetscReal x0[3] = {0.0, 0.0, 0.0};
    for (int d = 0; d < dim; ++d) x0[d] = x[d];
    for (int i = 0; i < dim; ++i) {
        PetscReal p1[3], m1[3], p2[3], m2[3];
        std::memcpy(p1, x0, sizeof x0); std::memcpy(m1, x0, sizeof x0);
        std::memcpy(p2, x0, sizeof x0); std::memcpy(m2, x0, sizeof x0);
        p1[i] = x0[i] + h; m1[i] = x0[i] - h; p2[i] = x0[i] + 2.0 * h; m2[i] = x0[i] - 2.0 * h;
        const PetscReal f_p1 = F(p1), f_m1 = F(m1), f_p2 = F(p2), f_m2 = F(m2);
        last.first[i] = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) * inv_12h;
    }
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            PetscReal pp[3], pm[3], mp[3], mm[3];
            std::memcpy(pp, x0, sizeof x0); std::memcpy(pm, x0, sizeof x0);
            std::memcpy(mp, x0, sizeof x0); std::memcpy(mm, x0, sizeof x0);
            pp[i] += h; pp[j] += h;
            pm[i] += h; pm[j] -= h;
            mp[i] -= h; mp[j] += h;
            mm[i] -= h; mm[j] -= h;
            const PetscReal f_pp = F(pp), f_pm = F(pm), f_mp = F(mp), f_mm = F(mm);
            last.second[i * dim + j] = (f_pp - f_pm - f_mp + f_mm) * inv_4h2;
        }
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            for (int k = 0; k < dim; ++k) {
                PetscReal f[8];
                for (int s = 0; s < 8; ++s) {              // s = 0..7: ppp ppm pmp pmm mpp mpm mmp mmm
                    PetscReal r[3];
                    std::memcpy(r, x0, sizeof x0);
                    if (s & 4) r[i] -= h; else r[i] += h;
                    if (s & 2) r[j] -= h; else r[j] += h;
                    if (s & 1) r[k] -= h; else r[k] += h;
                    f[s] = F(r);
                }
                last.third[(i * dim + j) * dim + k] =
                    (f[0] - f[1] - f[2] + f[3] - f[4] + f[5] + f[6] - f[7]) * inv_h * inv_h * inv_h / 8.0;
            }
    key = x;
    return last;
}

/// 1/m derivatives for the force kernels: the mass-gradient corrections of the
/// force matrices are needed only by the operator-form acceleration. With the
/// discrete acceleration the force matrices are not read; their corrections are
/// skipped (zero derivatives), as in TDSEZFormPhyX and the sum-factorised path.
const PhysicsDerivativesFull &TDSEZForceMassDerivatives(const std::vector<PetscReal> &x)
{
    if (!TDSEZDiscreteAccelerationActive()) return TDSEZInvMassDerivatives(x);
    thread_local PhysicsDerivativesFull zero;
    const size_t d = x.size();
    if (zero.first.size() != d) { zero.first.assign(d, 0.0); zero.second.assign(d * d, 0.0); zero.third.assign(d * d * d, 0.0); }
    return zero;
}
#include <algorithm>
#include <vector>

#include "tdsez_internal.hpp"






/// Manolopoulos complex-absorbing-potential (CAP) profile W(r).
/**
 * Evaluates the smooth CAP from Manolopoulos (2002) that vanishes inside the
 * core region and rises to absorb outgoing flux near the box boundary.
 * The profile uses precomputed constants c, a, b, delta and the minimum
 * momentum k_min to set the absorption onset and strength.
 *
 * @param[in] xq      Spatial coordinate (signed; |xq| is used).
 * @param             (unnamed second parameter, reserved).
 * @param[in] x_max   Outer box boundary.
 * @param[in] ma_kmin Minimum momentum setting the absorption-region width.
 * @return CAP strength W(r) (zero inside the core region).
 */
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




/// Function-pointer type for a PetIGA physics form-kernel callback.
/** @param p     Quadrature point with shape-function data.
 *  @param nmat  Number of operator matrices being assembled.
 *  @param K     Array of nmat element stiffness matrices (nen×nen each).
 *  @param ctx   User context passed through from the assembler.
 *  @return PETSc error code. */
typedef PetscErrorCode (*TDSEZPhysicsKernel)(
    IGAPoint p,
    PetscInt nmat,
    PetscScalar *K[],
    void *ctx);


    

/// Core element-loop assembler: builds nmat PETSc Mat objects from a physics kernel.
/**
 * Iterates over all IGA elements and quadrature points, calling the user
 * kernel at each point to fill per-element work arrays, then assembling them
 * into the global matrices. Allocates a contiguous work buffer sized for the
 * worst-case nen across all dimensions. Performs safe cleanup on error via
 * goto-cleanup. Matrices are zeroed first (IGA uses ADD_VALUES semantics).
 *
 * @param[in]  iga    Isogeometric analysis object.
 * @param[in]  nmat   Number of matrices to assemble (1..12).
 * @param[out] mats   Array of nmat matrices to fill.
 * @param[in]  kernel  Physics form-kernel called per quadrature point.
 * @param[in]  ctx    User context forwarded to the kernel.
 * @return PETSc error code.
 */
PetscErrorCode TDSEZCompOperators(
    IGA                iga,
    PetscInt           nmat,
    Mat                mats[],
    TDSEZPhysicsKernel kernel,
    void              *ctx)
{
    MPI_Comm comm;
    PetscInt blasThreads;
    TDSEZBandAssembler band;

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

    PetscCall(band.Setup(iga, nmat));
    PetscCall(TDSEZBlasSerialBegin(&blasThreads));
    // One element pass per group of matrices (all of them in one pass unless
    // the banded accumulation buffer would exceed its memory budget).
    for (size_t g = 0; g < band.groups.size(); ++g) {
        const unsigned group = band.groups[g];
        const unsigned need  = group;
        ierr = band.BeginGroup(group);
        if (PetscUnlikely(ierr)) goto cleanup;
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

            // Element matrices: sum-factorised where supported, per-point kernel
            // otherwise (TDSEZElementWork, sumfact.cpp).
            ierr = TDSEZElementWork(iga, element, kernel, nmat, work, ctx, need);
            if (PetscUnlikely(ierr)) goto cleanup;

            for (PetscInt k = 0; k < nmat; ++k) {
                if (!((group >> k) & 1u)) continue;
                ierr = band.Add(element, k, work[k], mats[k]);
                if (PetscUnlikely(ierr)) goto cleanup;
            }
        }
        element_open = PETSC_FALSE;
        ierr = IGAEndElement(iga, &element);
        if (PetscUnlikely(ierr)) goto cleanup;
        ierr = band.EndGroup(mats);
        if (PetscUnlikely(ierr)) goto cleanup;
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
    band.Release();
    {
        PetscErrorCode ierr2 = TDSEZBlasSerialEnd(blasThreads);
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
//
//  The wall DoFs are pinned symmetrically in H AND M: their rows and columns
//  are removed and the decoupled diagonal entries give one spurious eigenvalue
//  per pinned DoF near the top of the discrete spectrum. The global indices of the
//  pinned DoFs are recorded so that the time-dependent operators (dipole,
//  velocity, CAP, ...) can be restricted the same way by
//  TDSEZZeroDirichletRows().
// ---------------------------------------------------------------------------

/// Global row indices of the Dirichlet-pinned wall DoFs (all ranks, sorted).
static std::vector<PetscInt> g_dirichletRows;

/// Spurious eigenvalue of a pinned DoF, relative to the largest diagonal
/// Rayleigh quotient H_aa/M_aa over the free DoFs of its element (which lies
/// near the top of the discrete spectrum). Kept at 1: a larger value inflates
/// ||H||, and the shift-invert eigensolver's true residual grows with ||H||.
static const PetscReal kDirichletSpuriousFactor = 1.0;

/// Convert element-local wall indices to global rows and gather them on all ranks.
static PetscErrorCode TDSEZStoreDirichletRows(Mat A, std::vector<PetscInt> &local)
{
    PetscFunctionBeginUser;
    std::sort(local.begin(), local.end());
    local.erase(std::unique(local.begin(), local.end()), local.end());
    std::vector<PetscInt> glob(local.size());
    if (!local.empty()) {
        ISLocalToGlobalMapping rmap = NULL;
        PetscCall(MatGetLocalToGlobalMapping(A, &rmap, NULL));
        PetscCheck(rmap, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE,
                   "matrix has no local-to-global mapping");
        PetscCall(ISLocalToGlobalMappingApply(rmap, (PetscInt)local.size(), local.data(), glob.data()));
    }
    MPI_Comm    comm = PetscObjectComm((PetscObject)A);
    PetscMPIInt size, n = (PetscMPIInt)glob.size();
    PetscCallMPI(MPI_Comm_size(comm, &size));
    std::vector<PetscMPIInt> counts(size), displs(size, 0);
    PetscCallMPI(MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm));
    for (PetscMPIInt r = 1; r < size; ++r) displs[r] = displs[r - 1] + counts[r - 1];
    g_dirichletRows.assign((size_t)(displs[size - 1] + counts[size - 1]), 0);
    PetscCallMPI(MPI_Allgatherv(glob.data(), n, MPIU_INT, g_dirichletRows.data(),
                                counts.data(), displs.data(), MPIU_INT, comm));
    std::sort(g_dirichletRows.begin(), g_dirichletRows.end());
    g_dirichletRows.erase(std::unique(g_dirichletRows.begin(), g_dirichletRows.end()), g_dirichletRows.end());
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// Remove the rows and columns of the Dirichlet wall DoFs from an operator,
/// putting @p diag on the diagonal. No-op when no Dirichlet assembly has run
/// (Neumann boundaries) or when @p A is NULL.
PetscErrorCode TDSEZZeroDirichletRows(Mat A, PetscScalar diag)
{
    PetscFunctionBeginUser;
    if (!A || g_dirichletRows.empty()) PetscFunctionReturn(PETSC_SUCCESS);
    PetscInt rs, re;
    PetscCall(MatGetOwnershipRange(A, &rs, &re));
    std::vector<PetscInt> own;
    for (PetscInt r : g_dirichletRows)
        if (r >= rs && r < re) own.push_back(r);
    PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE));
    PetscCall(MatZeroRowsColumns(A, (PetscInt)own.size(), own.data(), diag, NULL, NULL));
    PetscFunctionReturn(PETSC_SUCCESS);
}
/// Dirichlet-enforcing variant of TDSEZCompOperators.
/**
 * Identical to TDSEZCompOperators except it applies homogeneous Dirichlet
 * boundary conditions (psi=0 at wall nodes) via IGAElementFixSystem after the
 * kernel runs. The mass matrix M (slot k==1) is exempt so the generalized
 * eigenproblem H psi = E M psi stays well-posed.
 *
 * @param[in]  iga    Isogeometric analysis object.
 * @param[in]  nmat   Number of matrices to assemble (1..12).
 * @param[out] mats   Array of nmat matrices to fill.
 * @param[in]  kernel Physics form-kernel called per quadrature point.
 * @param[in]  ctx    User context forwarded to the kernel.
 * @return PETSc error code.
 */
PetscErrorCode TDSEZCompOperatorsDirichlet(
    IGA                iga,
    PetscInt           nmat,
    Mat                mats[],
    TDSEZPhysicsKernel kernel,
    void              *ctx)
{
    MPI_Comm comm;
    PetscInt blasThreads;
    TDSEZBandAssembler band;

    PetscFunctionBegin;

    PetscCheck(iga,    PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IGA is NULL");
    PetscCheck(mats,   PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "mats[] is NULL");
    PetscCheck(kernel, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "kernel is NULL");
    PetscCheck(nmat >= 2 && nmat <= 12, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "nmat %" PetscInt_FMT " must be in [2,12] (mats = {H, M, ...})", nmat);
    PetscCheck(iga->dof == 1, PETSC_COMM_SELF, PETSC_ERR_SUP,
               "Dirichlet assembly supports scalar problems (dof = 1) only");

    comm = PetscObjectComm((PetscObject)iga);

    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCheck(mats[k], comm, PETSC_ERR_ARG_NULL,
                   "mats[%" PetscInt_FMT "] is NULL", k);
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
    std::vector<PetscInt> wallLocal;   // element-local (ghosted) indices of wall DoFs

    PetscCall(PetscMalloc1(alloc_cnt, &work_storage));
    for (PetscInt k = 0; k < nmat; ++k) work[k] = work_storage + (size_t)k * n2_max;

    PetscCall(band.Setup(iga, nmat));
    PetscCall(TDSEZBlasSerialBegin(&blasThreads));
    // One element pass per group of matrices (all of them in one pass unless
    // the banded accumulation buffer would exceed its memory budget).
    for (size_t g = 0; g < band.groups.size(); ++g) {
        const unsigned group = band.groups[g];
        const unsigned need  = group | 3u;   // the wall pinning reads H and M
        ierr = band.BeginGroup(group);
        if (PetscUnlikely(ierr)) goto cleanup;
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

            // Element matrices: sum-factorised where supported, per-point kernel
            // otherwise (TDSEZElementWork, sumfact.cpp).
            ierr = TDSEZElementWork(iga, element, kernel, nmat, work, ctx, need);
            if (PetscUnlikely(ierr)) goto cleanup;

            // ---- Homogeneous Dirichlet: pin the wall DoFs symmetrically ----
            // PetIGA lists the wall DoFs of this element (IGASetBoundaryValue) in
            // element->ifix. Their rows and columns are removed from H and M, so
            // psi is exactly zero there and the remaining problem is the conforming
            // H_0^1 Galerkin problem. Pinning H alone (the previous behaviour) left
            // the wall DoF coupled through M, which shifted the eigenvalues below
            // the exact ones. The decoupled diagonals give each wall DoF the
            // spurious eigenvalue kDirichletSpuriousFactor * max_a(H_aa/M_aa), taken
            // over the free DoFs of the element (the wall DoF's own entry can be
            // huge for potentials singular at the wall), i.e. near the top of the
            // discrete spectrum and away from the shift-invert target.
            ierr = IGAElementBuildFix(element);
            if (PetscUnlikely(ierr)) goto cleanup;
            if (element->nfix > 0) {
                PetscReal mDiag = 0.0, ratio = 1.0;
                for (PetscInt a = 0; a < nen; ++a) {
                    PetscBool isWall = PETSC_FALSE;
                    for (PetscInt f = 0; f < element->nfix; ++f) if (element->ifix[f] == a) isWall = PETSC_TRUE;
                    if (isWall) continue;
                    const PetscReal maa = PetscRealPart(work[1][a*nen + a]);
                    const PetscReal haa = PetscAbsScalar(work[0][a*nen + a]);
                    if (maa > 0.0) { mDiag = PetscMax(mDiag, maa); ratio = PetscMax(ratio, haa / maa); }
                }
                if (mDiag <= 0.0) mDiag = 1.0;
                PetscInt nen_e;
                const PetscInt *rowmap = NULL;   // local (ghosted) DoF indices of the element
                ierr = IGAElementGetClosure(element, &nen_e, &rowmap);
                if (PetscUnlikely(ierr)) goto cleanup;
                for (PetscInt f = 0; f < element->nfix; ++f) {
                    const PetscInt i = element->ifix[f];
                    for (PetscInt k = 0; k < nmat; ++k)
                        for (PetscInt j = 0; j < nen; ++j) {
                            work[k][i*nen + j] = 0.0;
                            work[k][j*nen + i] = 0.0;
                        }
                    work[0][i*nen + i] = kDirichletSpuriousFactor * ratio * mDiag;
                    work[1][i*nen + i] = mDiag;
                    if (g == 0) wallLocal.push_back(rowmap[i]);
                }
            }

            for (PetscInt k = 0; k < nmat; ++k) {
                if (!((group >> k) & 1u)) continue;
                ierr = band.Add(element, k, work[k], mats[k]);
                if (PetscUnlikely(ierr)) goto cleanup;
            }
        }
        element_open = PETSC_FALSE;
        ierr = IGAEndElement(iga, &element);
        if (PetscUnlikely(ierr)) goto cleanup;
        ierr = band.EndGroup(mats);
        if (PetscUnlikely(ierr)) goto cleanup;
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
    band.Release();
    {
        PetscErrorCode ierr2 = TDSEZBlasSerialEnd(blasThreads);
        if (!ierr) ierr = ierr2;
    }

    PetscCall(ierr);

    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCall(MatAssemblyBegin(mats[k], MAT_FINAL_ASSEMBLY));
    }
    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCall(MatAssemblyEnd(mats[k], MAT_FINAL_ASSEMBLY));
    }

    // Record the global wall rows for TDSEZZeroDirichletRows().
    PetscCall(TDSEZStoreDirichletRows(mats[0], wallLocal));

    PetscFunctionReturn(PETSC_SUCCESS);
}



/// Multi-operator form kernel: assembles Hamiltonian H and mass M in one pass.
/**
 * Fills M[0] = H = (1/2m) ∇B·∇B + V B B and M[1] = M = B B (L² inner product)
 * for 1D, 2D, or 3D. Uses a position-dependent inverse mass from
 * TDSEZParser::MassDist. Stack buffers are used for small nen, heap otherwise.
 *
 * @param[in]  p    Quadrature point.
 * @param[in]  nmat Number of matrices (unused, expected 2).
 * @param[out] M    Array of element matrices: M[0]=H, M[1]=mass.
 * @param[in]  ctx  User context (unused).
 * @return PETSc error code.
 */
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

        // H and M are real symmetric. Accumulate only the upper triangle in
        // real per-element buffers (half the work, half the bytes, and the
        // inner loop vectorises) and write the full complex element matrices
        // once, at the element's last quadrature point.
        thread_local std::vector<PetscReal> Hr, Mr;
        const size_t n2 = (size_t)nen * (size_t)nen;
        if (p->index == 0) { Hr.assign(n2, 0.0); Mr.assign(n2, 0.0); }

        for (PetscInt a = 0; a < nen; ++a) {
            const PetscReal BaW = B[a] * w;
            const PetscReal pot_a = B[a] * Vw;
            const PetscReal gx_a = dBx[a] * cW;
            const PetscReal gy_a = dBy[a] * cW;
            const PetscReal gz_a = dBz[a] * cW;

            PetscReal *__restrict__ Hrow = Hr.data() + (size_t)a * nen;
            PetscReal *__restrict__ Mrow = Mr.data() + (size_t)a * nen;

            for (PetscInt b = a; b < nen; ++b) {
                const PetscReal Bb = B[b];
                Hrow[b] += gx_a * dBx[b] + gy_a * dBy[b] + gz_a * dBz[b] + pot_a * Bb;
                Mrow[b] += BaW * Bb;
            }
        }

        if (p->index == p->count - 1) {
            for (PetscInt a = 0; a < nen; ++a)
                for (PetscInt b = a; b < nen; ++b) {
                    const PetscReal h = Hr[(size_t)a * nen + b], m = Mr[(size_t)a * nen + b];
                    M[0][a * nen + b] += h; M[1][a * nen + b] += m;
                    if (b != a) { M[0][b * nen + a] += h; M[1][b * nen + a] += m; }
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

        // H and M are real symmetric. Accumulate only the upper triangle in
        // real per-element buffers (half the work, half the bytes, and the
        // inner loop vectorises) and write the full complex element matrices
        // once, at the element's last quadrature point.
        thread_local std::vector<PetscReal> Hr, Mr;
        const size_t n2 = (size_t)nen * (size_t)nen;
        if (p->index == 0) { Hr.assign(n2, 0.0); Mr.assign(n2, 0.0); }

        for (PetscInt a = 0; a < nen; ++a) {
            const PetscReal BaW = B[a] * w;
            const PetscReal pot_a = B[a] * Vw;
            const PetscReal gx_a = dB[a][0] * cW;
            const PetscReal gy_a = dB[a][1] * cW;

            PetscReal *__restrict__ Hrow = Hr.data() + (size_t)a * nen;
            PetscReal *__restrict__ Mrow = Mr.data() + (size_t)a * nen;

            for (PetscInt b = a; b < nen; ++b) {
                const PetscReal Bb = B[b];
                Hrow[b] += gx_a * dB[b][0] + gy_a * dB[b][1] + pot_a * Bb;
                Mrow[b] += BaW * Bb;
            }
        }

        if (p->index == p->count - 1) {
            for (PetscInt a = 0; a < nen; ++a)
                for (PetscInt b = a; b < nen; ++b) {
                    const PetscReal h = Hr[(size_t)a * nen + b], m = Mr[(size_t)a * nen + b];
                    M[0][a * nen + b] += h; M[1][a * nen + b] += m;
                    if (b != a) { M[0][b * nen + a] += h; M[1][b * nen + a] += m; }
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


/// X-polarization multi-operator batch kernel: [K, V, Md, X, VelX, dVdx].
/**
 * Assembles six operators in a single quadrature-point pass in 1D: kinetic
 * K, potential V, distributed mass Md, dipole X, velocity VelX, and force
 * dVdx (with variable-mass quantum corrections when the operator-form
 * acceleration reads it). 2D and 3D use TDSEZFormPhyAxes.
 *
 * @param[in]  p    Quadrature point.
 * @param[in]  nmat Number of matrices (unused, expected 6).
 * @param[out] M    Element matrices: K, V, Md, X, VelX, dVdx.
 * @param[in]  ctx  User context (unused).
 * @return PETSc error code.
 */
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
    const PetscReal x = xyz[0];
    PetscCheck(dim == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "TDSEZFormPhyX is the 1D kernel (2D/3D use TDSEZFormPhyAxes)");

    const PetscReal invMass = 1.0 / TDSEZParser::MassDist(x, 0.0, 0.0);
    const PetscReal coeff   = 0.5 * invMass;

    // When mass is constant all spatial derivatives of f=1/m vanish:
    //   fx=fy=fz=0, fxx=...=0, fxxx=...=0
    // => every quantum correction row (R1..R6 / row1..row6) is exactly zero.
    // TDSEZCompDerivative is never called, saving O(dim^2+dim^3) MassDist
    // evaluations per quadrature point — the dominant cost in the variable-mass path.
    const bool massIsConstant = (TDSEZParser::MassIsConstant == PETSC_TRUE);
    // Mass-gradient corrections of the force operator dVdx. They are needed
    // only when a(t) uses the operator form; with the discrete acceleration
    // the dVdx matrix is not read (any dimension), so they are skipped.
    const bool corr = !massIsConstant && !TDSEZDiscreteAccelerationActive();


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
        if (corr) {
            const PhysicsDerivativesFull &ops = TDSEZInvMassDerivatives(std::vector<PetscReal>{x, 0.0, 0.0});
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
                Pxrow[b] += -0.5 * PETSC_i * invMass * (Ba_w * dBb_x - dBa_x * Bb * w);

                // Constant mass: quantum correction is exactly zero,
                // dVdx reduces to the classical Ehrenfest term only.
                PetscScalar dv = Ba_neg_dVdx_f_w * Bb;
                if (corr) {
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

    PetscFunctionReturn(PETSC_SUCCESS);
}






/// Hamiltonian matrix: ∫ (ħ²/2m) ∇B_i · ∇B_j + V(x) B_i B_j dx

// ============================================================================
//  SINGLE-OPERATOR FORM KERNELS (were: form_functions.cpp)
// ============================================================================

#include "tdsez_internal.hpp"

/// Weak form for the Lz operator: Lz = -i (x d/dy - y d/dx).
/**
 * Fills the element matrix for the z-component of angular momentum.
 * Only defined for 2D; returns a zero matrix for other dimensions.
 *
 * @param[in]  p   Quadrature point.
 * @param[out] L   Element matrix (nen×nen).
 * @param[in]  ctx User context (unused).
 * @return PETSc error code.
 */
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

/// Form kernel for the kinetic-energy matrix K = (1/2m) ∇B_i · ∇B_j.
/**
 * Assembles the kinetic operator separately (for analysis/reuse) for 1D,
 * 2D, or 3D, using a position-dependent inverse mass. Inner loops are
 * SIMD-annotated where applicable.
 *
 * @param[in]  p   Quadrature point.
 * @param[out] K   Element matrix (nen×nen).
 * @param[in]  ctx User context (unused).
 * @return PETSc error code.
 */
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




/// Form kernel for the potential matrix V_ij = V(r) B_i B_j.
/**
 * Assembles the potential-energy operator separately for 1D/2D/3D.
 *
 * @param[in]  p   Quadrature point.
 * @param[out] V   Element matrix (nen×nen).
 * @param[in]  ctx User context (unused).
 * @return PETSc error code.
 */
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




/// Form kernel for the distributed-mass matrix Md_ij = (1/m(r)) B_i B_j.
/**
 * Assembles the position-dependent inverse-mass (L²-weighted) inner product
 * for use in the generalized eigenproblem and dipole/acceleration observables.
 *
 * @param[in]  p   Quadrature point.
 * @param[out] Md  Element matrix (nen×nen).
 * @param[in]  ctx User context (unused).
 * @return 0 on success.
 */
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





/// Form kernel for the x-dipole operator: X_ij = x B_i B_j.
/** @param[in] p Quadrature point. @param[out] X Element matrix (nen×nen).
 *  @param[in] ctx User context (unused). @return 0 on success. */
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




/// Form kernel for the y-dipole operator: Y_ij = y B_i B_j.
/** @param[in] p Quadrature point. @param[out] Y Element matrix (nen×nen).
 *  @param[in] ctx User context (unused). @return 0 on success. */
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



/// Form kernel for the z-dipole operator: Z_ij = z B_i B_j.
/** @param[in] p Quadrature point. @param[out] Z Element matrix (nen×nen).
 *  @param[in] ctx User context (unused). @return 0 on success. */
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


/// Form kernel for the x-velocity operator: P_x = -(i/2m)(B_a dB_b - dB_a B_b).
/**
 * Anti-symmetric velocity-gauge operator along x, using a position-dependent
 * inverse mass.
 *
 * @param[in]  p   Quadrature point.
 * @param[out] P   Element matrix (nen×nen).
 * @param[in]  ctx User context (unused).
 * @return PETSc error code.
 */
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




/// Form kernel for the y-velocity operator: P_y = -(i/2m)(B_a dB_b - dB_a B_b).
/**
 * Anti-symmetric velocity-gauge operator along y, using a position-dependent
 * inverse mass.
 *
 * @param[in]  p   Quadrature point.
 * @param[out] P   Element matrix (nen×nen).
 * @param[in]  ctx User context (unused).
 * @return PETSc error code.
 */
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




/// Form kernel for the z-velocity operator: P_z = -(i/2m)(B_a dB_b - dB_a B_b).
/**
 * Anti-symmetric velocity-gauge operator along z, using a position-dependent
 * inverse mass.
 *
 * @param[in]  p   Quadrature point.
 * @param[out] P   Element matrix (nen×nen).
 * @param[in]  ctx User context (unused).
 * @return PETSc error code.
 */
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




/// Form kernel for the x-force operator dV/dx with variable-mass corrections.
/**
 * Computes the Ehrenfest force matrix element <B_a|dV/dx|B_b> plus quantum
 * corrections from derivatives of the inverse mass (1/m) for 1D, 2D, and 3D.
 * The classical term is -B_a dV/dx (1/m) B_b; quantum terms involve up to
 * third-order derivatives of f=1/m.
 *
 * @param[in]  p    Quadrature point.
 * @param[out] dVdr Element matrix (nen×nen).
 * @param[in]  ctx  User context (unused).
 * @return 0 on success.
 */
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
        return 1.0/TDSEZParser::MassDist(xx[0], (xx.size() > 1 ? xx[1] : 0.0), (xx.size() > 2 ? xx[2] : 0.0));
    };

    if(dim == 1)
    {
        PetscReal dVdx = TDSEZParser::dVx(x);
        const PetscReal (*dB)[1];
        IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
    
        const PhysicsDerivativesFull &ops = TDSEZForceMassDerivatives(std::vector<PetscReal>{x});
        PetscReal f    = invMassFunc({x});
        PetscReal fx   = ops.first[0];
        PetscReal fxx  = ops.second[0]; 
        PetscReal fxxx = ops.third[0];  
    
        for (PetscInt a = 0; a < nen; a++) 
        {
            PetscReal Ba   = B[a];     
            PetscReal dBax = dB[a][0];
        
            for (PetscInt b = 0; b < nen; b++) 
            {
                PetscReal Bb   = B[b];
                PetscReal dBbx = dB[b][0]; 
    
                // Row 0: -Ba * dVdx * f * Bb
                PetscReal classical = -Ba * dVdx * f * Bb;
                // Row 1: 2 * Bi * Bj_x * fx^2
                PetscReal R1 = 2.0 * Ba * dBbx * (fx * fx);
                // Row 3: f * [ 2 * Bi * Bj_x * fxx + Bi * Bb * fxxx ]
                PetscReal R3 = f * (2.0 * Ba * dBbx * fxx + Ba * Bb * fxxx);
                // Row 4: -2 * f * fx * (dBax * dBbx)
                PetscReal R4 = -2.0 * f * fx * (dBax * dBbx);
                // Row 5: -2 * Bi * Bj_x * (fx*fx + f*fxx)
                PetscReal R5 = -2.0 * Ba * dBbx * (fx*fx + f*fxx);
                // Row 6: Bi * Bj * fx * fxx
                PetscReal R6 = Ba * Bb * (fx * fxx);
                PetscReal quantum = -0.25 * (R1 + R3 + R4 + R5 + R6);

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
        const PhysicsDerivativesFull &ops = TDSEZForceMassDerivatives(std::vector<PetscReal>{x, y});
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
            PetscReal Ba = B[a];     
            PetscReal dBax = dB[a][0];
            PetscReal dBay = dB[a][1]; 
    
            for (PetscInt b = 0; b < nen; b++) 
            {
                PetscReal Bb = B[b];
                PetscReal dBbx = dB[b][0]; 
                PetscReal dBby = dB[b][1]; 
                PetscReal dBbxy = ddB[b][0][1];

                // Row 0: -Ba * dVdx * f * Bb
                PetscReal classical = -Ba * dVdx * f * Bb;
                
                // Row 1: 2 * Bi * Bj_x * (fx^2 + fy^2)
                PetscReal R1 = 2.0 * Ba * dBbx * (fx*fx + fy*fy);

                // Row 3: f * ( 4*fy*Bi*Bj_xy + 2*Bi*Bj_x*(fyy+fxx) + Bi*Bj*(fxyy+fxxx) )
                PetscReal R3 = f * (4.0 * fy * Ba * dBbxy + 
                                      2.0 * Ba * dBbx * (fyy + fxx) + 
                                      Ba * Bb * (fxyy + fxxx));

                // Row 4: 2 * f * fx * (Bay*Bby - Bax*Bbx)
                PetscReal R4 = 2.0 * f * fx * (dBay * dBby - dBax * dBbx);

                // Row 5: 2 * ( Bi*Bby*(fy*fx + f*fxy) - Bi*Bbx*(fx*fx + f*fxx) )
                PetscReal R5 = 2.0 * (Ba * dBby * (fy*fx + f*fxy) - 
                                        Ba * dBbx * (fx*fx + f*fxx));

                // Row 6: Bi * Bj * (fy*fxy + fx*fxx)
                PetscReal R6 = Ba * Bb * (fy * fxy + fx * fxx);

                PetscReal quantum = -0.25 * (R1 + R3 + R4 + R5 + R6);

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
        const PhysicsDerivativesFull &ops = TDSEZForceMassDerivatives(std::vector<PetscReal>{x, y, z});
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
            PetscReal Ba = B[a], dBax = dB[a][0], dBay = dB[a][1], dBaz = dB[a][2];
    
            for (PetscInt b = 0; b < nen; b++) {
                PetscReal Bb = B[b], dBbx = dB[b][0], dBby = dB[b][1], dBbz = dB[b][2];
                PetscReal dBbxy = ddB[b][0][1], dBbxz = ddB[b][0][2];
    
                // Row 0: -Ba * dVdx * f * Bb
                PetscReal classical = -Ba * dVdx * f * Bb;
                    
                // Row 1: 2 * Bi * Bj_x * (fx^2 + fy^2 + fz^2)
                PetscReal row1 = 2.0 * Ba * dBbx * (fx*fx + fy*fy + fz*fz);
    
                // Row 2: fz * (4*f*Bi*Bj_xz + Bi*Bj*fxz)
                PetscReal row2 = fz * (4.0 * f * Ba * dBbxz + Ba * Bb * fxz);
    
                // Row 3: f * [4*fy*Bi*Bj_xy + 2*Bi*Bj_x*(fxx+fyy+fzz) + Bi*Bj*(fxxx+fxyy+fxzz)]
                PetscReal row3 = f * (4.0 * fy * Ba * dBbxy + 
                                        2.0 * Ba * dBbx * (fxx + fyy + fzz) + 
                                        Ba * Bb * (fxxx + fxyy + fxzz));
    
                // Row 4: 2*f*fx * (Baz*Bbz + Bay*Bby - Bax*Bbx)
                PetscReal row4 = 2.0 * f * fx * (dBaz * dBbz + dBay * dBby - dBax * dBbx);
    
                // Row 5: 2 * [ Ba*dBbz*ffx_z + Ba*dBby*ffx_y - Ba*dBbx*ffx_x ]
                PetscReal row5 = 2.0 * (Ba * dBbz * ffx_z + Ba * dBby * ffx_y - Ba * dBbx * ffx_x);

                // Row 6: Bi * Bj * (fy*fxy + fx*fxx)
                PetscReal row6 = Ba * Bb * (fy * fxy + fx * fxx);
    
                PetscReal quantum = -0.25 * (row1 + row2 + row3 + row4 + row5 + row6);
    
                dVdr[a * nen + b] = classical + quantum;
            }
        }
    }

    return 0;
}



/// Form kernel for the y-force operator dV/dy with variable-mass corrections.
/**
 * Computes the Ehrenfest force matrix element <B_a|dV/dy|B_b> plus quantum
 * corrections from derivatives of the inverse mass (1/m) for 2D and 3D.
 * Returns a zero matrix for 1D (dV/dy undefined).
 *
 * @param[in]  p    Quadrature point.
 * @param[out] dVdr Element matrix (nen×nen).
 * @param[in]  ctx  User context (unused).
 * @return 0 on success.
 */
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

        const PhysicsDerivativesFull &ops = TDSEZForceMassDerivatives((dim==2 ? std::vector<PetscReal>{x,y} : std::vector<PetscReal>{x,y,z}));
        
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
                PetscReal Ba = B[a], Bb = B[b];
                PetscReal dBax = dB[a][0], dBay = dB[a][1], dBaz = (dim==3 ? dB[a][2] : 0.0);
                PetscReal dBbx = dB[b][0], dBby = dB[b][1], dBbz = (dim==3 ? dB[b][2] : 0.0);
                
                // ddB indices: [basis][deriv1][deriv2]
                PetscReal dBbxy = ddB[b][0][1];
                PetscReal dBbyz = (dim==3 ? ddB[b][1][2] : 0.0);

                PetscReal classical = -Ba * dVdy * f * Bb;

                // Row 1: 2 * Bi * Bj_y * |grad f|^2
                PetscReal r1 = 2.0 * Ba * dBby * (fx*fx + fy*fy + fz*fz);

                // Row 2: fz * (4*f*Bi*Bj_yz + Bi*Bb*fyz)
                PetscReal r2 = fz * (4.0 * f * Ba * dBbyz + Ba * Bb * fyz);

                // Row 3: f * (4*fx*Bi*Bj_xy + 2*Bi*Bj_y*Laplacian + Bi*Bb*dy(Laplacian))
                PetscReal r3 = f * (4.0 * fx * Ba * dBbxy + 
                                      2.0 * Ba * dBby * (fxx + fyy + fzz) + 
                                      Ba * Bb * (fxxy + fyyy + fyzz));

                // Row 4: 2*f*fy * (Baz*Bbz - Bay*Bby + Bax*Bbx)
                PetscReal r4 = 2.0 * f * fy * (dBaz * dBbz - dBay * dBby + dBax * dBbx);

                // Row 5: 2 * (Bi*Bbz*(ffy)_z - Bi*Bby*(ffy)_y + Bi*Bbx*(ffy)_x)
                PetscReal r5 = 2.0 * (Ba * dBbz * ffy_z - Ba * dBby * ffy_y + Ba * dBbx * ffy_x);

                // Row 6: Bi * Bb * (fy*fyy + fx*fxy)
                PetscReal r6 = Ba * Bb * (fy * fyy + fx * fxy);

                PetscReal quantum = -0.25 * (r1 + r2 + r3 + r4 + r5 + r6);
                dVdr[a * nen + b] = classical + quantum;
            }
        }
    }
    return 0;
}



/// Form kernel for the z-force operator dV/dz with variable-mass corrections.
/**
 * Computes the Ehrenfest force matrix element <B_a|dV/dz|B_b> plus quantum
 * corrections from derivatives of the inverse mass (1/m). Only defined for
 * 3D; returns a zero matrix for lower dimensions.
 *
 * @param[in]  p    Quadrature point.
 * @param[out] dVdr Element matrix (nen×nen).
 * @param[in]  ctx  User context (unused).
 * @return 0 on success.
 */
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
        return 1.0/TDSEZParser::MassDist(xx[0], (xx.size() > 1 ? xx[1] : 0.0), (xx.size() > 2 ? xx[2] : 0.0));
    };

    PetscReal dVdz = TDSEZParser::dVz(x, y, z);
    const PetscReal (*dB)[3], (*ddB)[3][3];
    IGAPointGetShapeFuns(p, 1, (const PetscReal**)&dB);
    IGAPointGetShapeFuns(p, 2, (const PetscReal**)&ddB);

    const PhysicsDerivativesFull &ops = TDSEZForceMassDerivatives(std::vector<PetscReal>{x, y, z});
    
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
            PetscReal Ba = B[a], Bb = B[b];
            PetscReal dBax = dB[a][0], dBay = dB[a][1], dBaz = dB[a][2];
            PetscReal dBbx = dB[b][0], dBby = dB[b][1], dBbz = dB[b][2];
            
            PetscReal dBbyz = ddB[b][1][2];
            PetscReal dBbxz = ddB[b][0][2];

            PetscReal classical = -Ba * dVdz * f * Bb;

            // Row 1: 2 * Bi * Bj_z * |grad f|^2
            PetscReal r1 = 2.0 * Ba * dBbz * (fx*fx + fy*fy + fz*fz);

            // Row 2 & 3 Combined: f * (4fy*Bi*Bj_yz + 4fx*Bi*Bj_xz + 2*Bi*Bj_z*Laplacian + Bi*Bb*dz(Laplacian))
            PetscReal r23 = f * (4.0 * fy * Ba * dBbyz + 
                                   4.0 * fx * Ba * dBbxz + 
                                   2.0 * Ba * dBbz * (fxx + fyy + fzz) + 
                                   Ba * Bb * (fzzz + fyyz + fxxz));

            // Row 4: -2*f*fz * (Baz*Bbz - Bay*Bby - Bax*Bbx)
            PetscReal r4 = -2.0 * f * fz * (dBaz * dBbz - dBay * dBby - dBax * dBbx);

            // Row 5: -2 * (Bi*Bbz*(ffz)_z - Bi*Bby*(ffz)_y - Bi*Bbx*(ffz)_x)
            PetscReal r5 = -2.0 * (Ba * dBbz * ffz_z - Ba * dBby * ffz_y - Ba * dBbx * ffz_x);

            // Row 6: Bi * Bb * (fz*fzz + fy*fyz + fx*fxz)
            PetscReal r6 = Ba * Bb * (fz * fzz + fy * fyz + fx * fxz);

            PetscReal quantum = -0.25 * (r1 + r23 + r4 + r5 + r6);
            dVdr[a * nen + b] = classical + quantum;
        }
    }
    return 0;
}



/// Form kernel for the complex-absorbing-potential (CAP) matrix.
/**
 * Computes CAP_ij = -i W(r) B_i B_j with W the separable Manolopoulos CAP:
 * a reflection-free layer of width c/(2 delta CAPKmin) at both walls of every
 * active axis (see TDSEZformCap below).
 *
 * @param[in]  p   Quadrature point.
 * @param[out] CAP Element matrix (nen×nen).
 * @param[in]  ctx User context (unused).
 * @return 0 on success.
 */
/// Width of the Manolopoulos absorbing layer, w = c / (2 delta kmin).
PetscReal TDSEZCapWidth()
{
    return 2.62206 / (2.0 * 0.2 * TDSEZParser::CAPKmin);
}

PetscErrorCode TDSEZformCap(IGAPoint p, PetscScalar *CAP, void *ctx)
{
    (void)ctx;
    PetscInt nen = p->nen;
    const PetscReal *B;
    IGAPointGetShapeFuns(p, 0, (const PetscReal**)&B);

    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    IGAPointFormGeomMap(p, xyz);

    // Separable Manolopoulos CAP: every active axis gets an absorbing layer of
    // width w = c / (2 delta kmin) at both of its walls, W = sum_d W_d. The
    // profile starts at zero at the inner edge of each layer (no jump: a jump
    // reflects slow electrons back to the core) and uses each axis' own
    // limits, so non-cubic and asymmetric boxes are absorbed on every face.
    // TDSEZValidate checks that the layers leave a free core.
    const PetscReal lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
    const PetscReal hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
    const PetscReal width = TDSEZCapWidth();
    PetscReal Wq = 0.0;
    for (PetscInt d = 0; d < TDSEZParser::Dimension; ++d) {
        const PetscReal s_wall = PetscMin(xyz[d] - lo[d], hi[d] - xyz[d]);   // distance to the nearer wall
        if (s_wall < width) {
            // layer coordinate in [0, width), clamped below the 1/(c - r) pole
            const PetscReal u = PetscMin(width - s_wall, width * (1.0 - 1e-10));
            Wq += Manolopoulos_CAP_profile(u, 0.0, width, TDSEZParser::CAPKmin);
        }
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


// ---------------------------------------------------------------------------
//  TDSEZFormPhyAxes — batched TD operators for y / z / multi-axis polarization
// ---------------------------------------------------------------------------
PetscBool TDSEZPhyAxesOn[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};

/// Per-point kernel for the batch [K, V, Md, (D_d, Vel_d, dV_d) for active d].
/// Each matrix is the corresponding single-operator integrand kernel times
/// weight*detJac, the same contribution IGAComputeMatrix adds per point.
PetscErrorCode TDSEZFormPhyAxes(IGAPoint p, PetscInt nmat, PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)ctx;
    typedef PetscErrorCode (*Integrand)(IGAPoint, PetscScalar *, void *);
    Integrand ops[12];
    PetscInt n = 0;
    ops[n++] = TDSEZFormKinetic; ops[n++] = TDSEZFormPotential; ops[n++] = TDSEZFormMassDist;
    const Integrand D[3]  = {DipoleX, DipoleY, DipoleZ};
    const Integrand Vl[3] = {VelocityX, VelocityY, VelocityZ};
    const Integrand G[3]  = {TDSEZformPotentialGradX, TDSEZformPotentialGradY, TDSEZformPotentialGradZ};
    for (int d = 0; d < 3; ++d)
        if (TDSEZPhyAxesOn[d]) { ops[n++] = D[d]; ops[n++] = Vl[d]; ops[n++] = G[d]; }
    PetscCheck(n == nmat, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "TDSEZFormPhyAxes: %" PetscInt_FMT " matrices expected, %" PetscInt_FMT " given", n, nmat);

    const PetscInt  nen = p->nen;
    const PetscReal w   = p->weight[0] * p->detJac[0];
    thread_local std::vector<PetscScalar> tmp;
    tmp.resize((size_t)nen * nen);
    for (PetscInt k = 0; k < nmat; ++k) {
        PetscCall(ops[k](p, tmp.data(), NULL));
        for (size_t i = 0; i < (size_t)nen * nen; ++i) M[k][i] += w * tmp[i];
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
