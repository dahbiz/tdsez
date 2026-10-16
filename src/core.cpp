#include "tdsez_paths.hpp"
#include "tdsez_internal.hpp"
#include <petscblaslapack.h>

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
    info.header("STATIONARY PROBLEM");
    info.section("ASSEMBLY");
    info.kv("Input file", "%s", inputFile.c_str());
    info.kv("Degrees of freedom", "%" PetscInt_FMT, nm);

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
        info.fatal("unknown BoundaryType '%s' (use Neumann or Dirichlet)", TDSEZParser::BoundaryType.c_str());
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    PetscCall(PetscTime(&t1));
    // Sanity checks
    PetscReal normH, normM;
    PetscCall(MatNorm(H, NORM_INFINITY, &normH));
    PetscCall(MatNorm(M, NORM_INFINITY, &normM));
    info.kv("||H||_inf, ||M||_inf", "%.3e, %.3e", (double)normH, (double)normM);
    {
        // A potential that is infinite or NaN at a quadrature point (a Coulomb
        // singularity on a Gauss-Lobatto point, which includes the element
        // ends) gives non-finite entries; the max-based norm above skips NaN.
        PetscReal fro;
        PetscCall(MatNorm(H, NORM_FROBENIUS, &fro));
        if (PetscIsInfOrNanReal(fro)) {
            info.fatal("the Hamiltonian has infinite or NaN entries: the potential is singular at a "
                "quadrature point. QuadratureRule = lobatto puts points on the knots; with a Coulomb potential "
                "centred on a knot use legendre (default) or reduced, or move the centre off the knots.");
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
    }
    info.kv("Assembly time", "%.3f s", (double)(t1 - t0));

#ifndef NDEBUG
    {
        Mat Ht;
        PetscCall(MatHermitianTranspose(H, MAT_INITIAL_MATRIX, &Ht));
        PetscCall(MatAXPY(Ht, -1.0, H, SAME_NONZERO_PATTERN));
        PetscReal err; PetscCall(MatNorm(Ht, NORM_FROBENIUS, &err));
        info.kv("||H - H^H||_F", "%.2e [%s]", (double)err, err < 1e-10 ? "OK" : "WARN");
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
// ── Deterministic eigenvector gauge for superposition initial states ───────
// SLEPc returns each eigenvector with an arbitrary complex phase that also
// depends on the solver run and the MPI decomposition. A superposition
// sum_k c_k phi_k built from raw eigenvectors therefore has an arbitrary
// relative phase between its terms. The helpers below fix the gauge of every
// state (TDSEZCanonicalEigenpairs): the phase makes phi real (as in
// MakeStatesReal), and the remaining sign is chosen so that the overlap with a
// fixed smooth weight w(r) is positive.

/// @brief Assemble g_I = \int w(r) B_I(r) dr for a separable weight, xhat_d
///        the coordinate scaled to [-1,1]:
///        kind 0: w = prod_d (1 + 0.3 xhat_d), the primary weight. It contains
///                every parity class (1, x, y, z, xy, ...), but only degree <= 1
///                per axis, so a state whose factor along one axis is orthogonal
///                to 1 and x (e.g. a box mode cos(2 pi x / L)) has zero overlap.
///        kind 1: w = prod_d exp(a_d xhat_d), a = (0.37, 0.53, 0.71), the
///                fallback for those states: not a polynomial, so no eigenmode is
///                orthogonal to it.
///        Built from physical coordinates, it is independent of the MPI layout.
static PetscErrorCode TDSEZGaugeWeightVector(IGA iga, Vec *g, int kind = 0)
{
    PetscFunctionBeginUser;
    // w and the B-spline basis are tensor products and so is the Gauss rule:
    // g_I = prod_d \int (1 + 0.3 xhat_d) B_{i_d}(x_d) dx_d, from 1D integrals
    // over each axis (the same quadrature as a full element loop).
    const PetscInt  dim = iga->dim;
    const PetscReal lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
    const PetscReal hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
    std::vector<PetscReal> g1[3];
    for (PetscInt d = 0; d < 3; ++d) {
        const PetscInt n = (d < dim) ? iga->node_lwidth[d] : 1;
        g1[d].assign(n, d < dim ? 0.0 : 1.0);
        if (d >= dim) continue;
        const IGABasis B = iga->basis[d];
        const PetscInt Q = B->nqp, nen = B->nen, i0 = iga->node_lstart[d];
        const PetscReal half = 0.5 * (hi[d] - lo[d]), mid = 0.5 * (hi[d] + lo[d]);
        for (PetscInt el = 0; el < B->nel; ++el) {
            const PetscInt off = B->offset[el];
            if (off + nen <= i0 || off >= i0 + n) continue;      // no owned function
            for (PetscInt q = 0; q < Q; ++q) {
                const PetscReal x    = B->point[el * Q + q];
                const PetscReal xhat = half > 0.0 ? (x - mid) / half : 0.0;
                const PetscReal a[3] = {0.37, 0.53, 0.71};
                const PetscReal wq   = kind == 0 ? 1.0 + 0.3 * xhat : PetscExpReal(a[d] * xhat);
                const PetscReal jw   = B->weight[el * Q + q] * B->detJac[el] * wq;
                for (PetscInt a = 0; a < nen; ++a) {
                    const PetscInt k = off + a - i0;
                    if (k < 0 || k >= n) continue;
                    g1[d][k] += jw * B->value[(((size_t)el * Q + q) * nen + a) * 5];
                }
            }
        }
    }
    PetscCall(IGACreateVec(iga, g));
    PetscScalar *ga;
    PetscInt nloc, idx = 0;
    PetscCall(VecGetLocalSize(*g, &nloc));
    PetscCall(VecGetArray(*g, &ga));
    for (size_t k3 = 0; k3 < g1[2].size(); ++k3)
    for (size_t k2 = 0; k2 < g1[1].size(); ++k2)
    for (size_t k1 = 0; k1 < g1[0].size(); ++k1)
        if (idx < nloc) ga[idx++] = g1[0][k1] * g1[1][k2] * g1[2][k3];
    PetscCall(VecRestoreArray(*g, &ga));
    PetscCheck(idx == nloc, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Gauge weight: node box does not match the vector layout");
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Fix the gauge of an eigenvector in place: rotate its global phase so
///        that c^T M c is real and positive (c real for a nondegenerate state),
///        then flip the sign so that \int w psi = g^T c > 0 (primary weight),
///        or, when that overlap vanishes, \int w2 psi = g2^T c > 0.
/// @param[in,out] c  Eigenvector coefficients.
/// @param[in]     M  Overlap matrix.
/// @param[in]     g  Weight vector from TDSEZGaugeWeightVector (kind 0).
/// @param[in]     g2 Fallback weight vector (kind 1).
/// @param[in]     k  State index (for diagnostics).
static PetscErrorCode TDSEZFixEigenvectorGauge(Vec c, Mat M, Vec g, Vec g2, PetscInt k)
{
    PetscFunctionBeginUser;
    Vec Mc;
    PetscScalar z, ov;
    PetscCall(VecDuplicate(c, &Mc));
    PetscCall(MatMult(M, c, Mc));
    PetscCall(VecTDot(c, Mc, &z));                        // z = c^T M c (no conjugate)
    PetscCall(VecDestroy(&Mc));
    if (PetscAbsScalar(z) > 0.0) {
        const PetscReal theta = 0.5 * PetscAtan2Real(PetscImaginaryPart(z), PetscRealPart(z));
        PetscCall(VecScale(c, PetscExpScalar(-PETSC_i * theta)));
    }
    PetscCall(VecTDot(c, g, &ov));                        // \int w psi
    PetscReal nc, ng;
    PetscCall(VecNorm(c, NORM_2, &nc));
    PetscCall(VecNorm(g, NORM_2, &ng));
    if (PetscAbsReal(PetscRealPart(ov)) < 1e-8 * nc * ng) {
        PetscCall(VecTDot(c, g2, &ov));                   // \int w2 psi
        PetscCall(VecNorm(g2, NORM_2, &ng));
        if (PetscAbsReal(PetscRealPart(ov)) < 1e-8 * nc * ng)
            TDSEZInfo().warn("state %" PetscInt_FMT " has a near-zero gauge overlap; "
                "its sign may not be reproducible.", k);
    }
    if (PetscRealPart(ov) < 0.0) PetscCall(VecScale(c, -1.0));
    PetscFunctionReturn(PETSC_SUCCESS);
}


// ── Canonical eigenpairs ────────────────────────────────────────────────────
// SLEPc's output is not a well-defined basis inside a degenerate eigenspace:
// any rounding change (assembly order, MPI layout, BLAS) rotates the states of
// a shell, which changes per-state populations, currents and saved
// wavefunctions although every basis-independent quantity is unchanged. The
// number of converged pairs also varies from run to run, and a request for
// NBoundStates states can end in the middle of a shell. TDSEZCanonicalEigenpairs
// replaces SLEPc's pairs by a reproducible set:
//   1. exactly NBoundStates states, extended to the end of a degenerate shell
//      when the request cuts through one (a converged state beyond the kept
//      set must confirm the end of the shell; otherwise more pairs are solved);
//   2. inside each degenerate shell, the basis that diagonalises a fixed
//      position weight W (a generic quadratic polynomial of position evaluated
//      at the Greville points), sorted by <W>;
//   3. a deterministic global phase (TDSEZFixEigenvectorGauge).
// The rotation stays inside each shell: energies and spanned eigenspaces are
// unchanged, and all states remain M-orthonormal eigenvectors.
// Degeneracy: |E_i - E_j| <= tol * max(1, |E_i|), tol = -tdsez_degeneracy_tol
// (default 1e-9).

/// Diagonal position weight w_I = f(Greville point of B_I) on the global Vec
/// layout, f a generic quadratic of the scaled coordinates (no symmetry of a
/// box or radial problem leaves its restriction to a shell degenerate).
static PetscErrorCode TDSEZShellWeight(IGA iga, Vec w)
{
    PetscFunctionBeginUser;
    const PetscInt  dim = iga->dim;
    const PetscReal lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
    const PetscReal hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
    const PetscReal a[3]  = {0.137, 0.271, 0.389};          // linear
    const PetscReal b[3]  = {1.000, 0.618, 0.382};          // quadratic
    const PetscReal c[3]  = {0.213, 0.157, 0.101};          // xy, xz, yz
    std::vector<PetscReal> g[3];
    for (PetscInt d = 0; d < 3; ++d) {
        const PetscInt n = (d < dim) ? iga->node_lwidth[d] : 1;
        g[d].assign(n, 0.0);
        if (d >= dim) continue;
        const IGAAxis ax = iga->axis[d];
        const PetscReal half = 0.5 * (hi[d] - lo[d]), mid = 0.5 * (hi[d] + lo[d]);
        for (PetscInt k = 0; k < n; ++k) {
            const PetscInt i = iga->node_lstart[d] + k;
            PetscReal s = 0.0;
            if (ax->p > 0) { for (PetscInt j = 1; j <= ax->p; ++j) s += ax->U[i + j]; s /= ax->p; }
            else s = 0.5 * (ax->U[i] + ax->U[i + 1]);
            g[d][k] = half > 0.0 ? (s - mid) / half : 0.0;
        }
    }
    PetscScalar *wa;
    PetscCall(VecGetArray(w, &wa));
    PetscInt idx = 0;
    for (size_t k3 = 0; k3 < g[2].size(); ++k3)
    for (size_t k2 = 0; k2 < g[1].size(); ++k2)
    for (size_t k1 = 0; k1 < g[0].size(); ++k1) {
        const PetscReal x[3] = {g[0][k1], g[1][k2], g[2][k3]};
        PetscReal f = 0.0;
        for (int d = 0; d < 3; ++d) f += a[d] * x[d] + b[d] * x[d] * x[d];
        f += c[0] * x[0] * x[1] + c[1] * x[0] * x[2] + c[2] * x[1] * x[2];
        wa[idx++] = f;
    }
    PetscCall(VecRestoreArray(w, &wa));
    PetscInt nloc;
    PetscCall(VecGetLocalSize(w, &nloc));
    PetscCheck(nloc == idx, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Shell weight: local size %" PetscInt_FMT " != node box %" PetscInt_FMT, nloc, idx);
    PetscFunctionReturn(PETSC_SUCCESS);
}

const TDSEZFieldScales &TDSEZEstimateFieldScales()
{
    static TDSEZFieldScales f;
    static bool done = false;
    if (done) return f;
    done = true;
    if (!TDSEZParser::EnablePropagation || TDSEZParser::FinalTime <= 0) return f;
    const PetscInt n = 20000;
    const PetscReal T = TDSEZParser::FinalTime;
    int dominant = -1;
    for (int a = 0; a < (int)TDSEZParser::Dimension; ++a) {
        if (!TDSEZParser::Coupled(a)) continue;
        for (PetscInt i = 0; i <= n; ++i) {
            const PetscReal e = PetscAbsReal(TDSEZParser::Field(a, T * i / n));
            if (e > f.E0) { f.E0 = e; dominant = a; }
        }
    }
    if (f.E0 <= 0 || dominant < 0) return f;
    const PetscReal om[3] = {TDSEZParser::Omegax, TDSEZParser::Omegay, TDSEZParser::Omegaz};
    for (int a = 0; a < 3; ++a) if (TDSEZParser::Coupled(a)) f.omega = PetscMax(f.omega, om[a]);
    if (f.omega <= 0) {                             // from the zero crossings of the dominant axis
        // half periods between the first and the last sign change
        PetscInt cross = 0;
        PetscReal prev = TDSEZParser::Field(dominant, 0.0), t0 = 0, t1 = 0;
        for (PetscInt i = 1; i <= n; ++i) {
            const PetscReal t = T * i / n, e = TDSEZParser::Field(dominant, t);
            if ((e > 0 && prev < 0) || (e < 0 && prev > 0)) { if (!cross) t0 = t; t1 = t; ++cross; }
            if (e != 0) prev = e;
        }
        if (cross >= 3) f.omega = PETSC_PI * (cross - 1) / (t1 - t0);
    }
    if (f.omega <= 0) return f;
    f.Up    = f.E0 * f.E0 / (4 * f.omega * f.omega);
    f.Ip    = PetscRealPart(TDSEZParser::TargetEigenvalue) < 0 ? -PetscRealPart(TDSEZParser::TargetEigenvalue) : 0.5;
    f.kmax  = PetscSqrtReal(2 * (3.17 * f.Up + f.Ip));
    f.alpha = f.E0 / (f.omega * f.omega);
    f.h     = PETSC_PI / (2 * f.kmax);
    f.valid = true;
    return f;
}

/// Element width that resolves the strong-field dynamics on axis d:
/// h = pi/(2 k_max) on the axes coupled to the field (Polarization), and
/// TransverseElementFactor * h on the others, where the electron moves with
/// the much smaller transverse momentum spread. Measured on 2D hydrogen-like
/// HHG (800 nm, 1e14 W/cm2, degree 3): h on the laser axis converges the
/// harmonic yields to 1.6 % (median), 2h there misses the cutoff by 4
/// decades; across the laser 2h leaves the plateau unchanged and 3h misses
/// the region beyond the cutoff by 2 decades.
static PetscReal TDSEZAxisResolution(const TDSEZFieldScales &f, int d)
{
    return TDSEZParser::Coupled(d) ? f.h : TDSEZParser::TransverseElementFactor * f.h;
}

/// Largest hydrogenic element width on axis d when HydrogenicHMax = 0: the
/// resolution width of the field (TDSEZAxisResolution), 0 (no cap) without a
/// field. Prints the derivation once.
static PetscReal TDSEZHydrogenicAutoHMax(int d)
{
    const TDSEZFieldScales &f = TDSEZEstimateFieldScales();
    if (!f.valid) return 0.0;
    static bool printed = false;
    if (!printed) {
        printed = true;
        TDSEZInfo().line("HydrogenicHMax from the field (k_max = %.3g): %.3f au on the polarised axes, %.3f au on the "
            "others; set HydrogenicHMax > 0 to choose, < 0 for no cap",
            (double)f.kmax, (double)f.h, (double)(TDSEZParser::TransverseElementFactor * f.h));
    }
    return TDSEZAxisResolution(f, d);
}

/// Spatial resolution of the strong-field dynamics on any knot sequence:
/// compares the widest element of every axis with its resolution width
/// (TDSEZAxisResolution) and warns for the axes that are coarser. Grids are
/// not changed. Silent without a field (nothing to resolve).
static void TDSEZCheckFieldResolution(IGA iga)
{
    const TDSEZFieldScales &f = TDSEZEstimateFieldScales();
    if (!f.valid) return;
    TDSEZInfo info;
    info.section("FIELD RESOLUTION");
    info.kv("E0, omega", "%.4g, %.4g a.u.", (double)f.E0, (double)f.omega);
    info.kv("Up, Ip", "%.4g, %.3g a.u.", (double)f.Up, (double)f.Ip);
    info.kv("k_max", "%.3g a.u.", (double)f.kmax);
    info.kv("Widest element allowed", "%.3f polarised, %.3f transverse (a.u.)", (double)f.h,
            (double)(TDSEZParser::TransverseElementFactor * f.h));
    info.kv("TransverseElementFactor", "%g", (double)TDSEZParser::TransverseElementFactor);
    for (PetscInt d = 0; d < iga->dim; ++d) {
        PetscInt m;
        PetscReal *U, hmax = 0, xmax = 0;
        if (IGAAxisGetKnots(iga->axis[d], &m, &U)) continue;
        for (PetscInt k = 0; k < m; ++k)
            if (U[k + 1] - U[k] > hmax) { hmax = U[k + 1] - U[k]; xmax = 0.5 * (U[k] + U[k + 1]); }
        const PetscReal need = TDSEZAxisResolution(f, (int)d);
        const char *role = TDSEZParser::Coupled((int)d) ? "polarised" : "transverse";
        if (hmax > need * (1 + 1e-9))
            info.warn("%c axis (%s): widest element %.3f au (at %c = %.3g) > %.3f au; the returning "
                "electron and the harmonic cutoff are under-resolved there (refine NSplines/the grading, or use "
                "KnotSequence = hydrogenic, which caps the width)",
                "xyz"[d], role, (double)hmax, "xyz"[d], (double)xmax, (double)need);
        else {
            char lab[48];
            snprintf(lab, sizeof(lab), "%c axis (%s)", "xyz"[d], role);
            info.kv(lab, "widest element %.3f <= %.3f au: resolved", (double)hmax, (double)need);
        }
    }
}

/// Backward error of an eigenpair of the pencil (H, M):
/// ||H v - E M v|| / ((||H|| + |E| ||M||) ||v||), infinity norms of H and M.
/// Unlike ||r|| / (|E| ||v||) it stays meaningful for E -> 0.
static PetscErrorCode TDSEZBackwardError(Mat H, Mat M, PetscReal normH, PetscReal normM, Vec v, PetscReal E,
                                         PetscReal *res)
{
    PetscFunctionBeginUser;
    Vec r, Mv;
    PetscCall(VecDuplicate(v, &r));
    PetscCall(VecDuplicate(v, &Mv));
    PetscCall(MatMult(H, v, r));
    PetscCall(MatMult(M, v, Mv));
    PetscCall(VecAXPY(r, -E, Mv));
    PetscReal nr, nv;
    PetscCall(VecNorm(r, NORM_2, &nr));
    PetscCall(VecNorm(v, NORM_2, &nv));
    *res = nr / PetscMax((normH + PetscAbsReal(E) * normM) * nv, (PetscReal)1e-300);
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&Mv));
    PetscFunctionReturn(PETSC_SUCCESS);
}

namespace {

/// Inner solver of the shift-invert transformation (H - sigma M)^{-1}.
enum class EpsInner { Direct, Iterative, IterativeStrong };

const char *EpsInnerName(EpsInner m)
{
    switch (m) {
    case EpsInner::Direct:          return "direct LU";
    case EpsInner::Iterative:       return "FGMRES + ASM(1)/ILU(0)";
    default:                        return "FGMRES + ASM(2)/ILU(1)";
    }
}

/// Solver package for the direct inner solve, or nullptr if none runs on
/// this communicator.
const char *EpsDirectPackage(Mat A, PetscMPIInt nprocs)
{
    MatType mt;
    if (MatGetType(A, &mt)) return nullptr;
    if (strstr(mt, "cusparse") || strstr(mt, "cuda")) return nprocs == 1 ? MATSOLVERCUSPARSE : nullptr;
#if defined(PETSC_HAVE_MUMPS)
    return MATSOLVERMUMPS;
#elif defined(PETSC_HAVE_SUPERLU_DIST)
    return MATSOLVERSUPERLU_DIST;
#else
    return nprocs == 1 ? MATSOLVERPETSC : nullptr;
#endif
}

struct EpsMonitorCtx {
    PetscInt nev = 0, prev = 0;
};

PetscErrorCode EpsMonitor(EPS, PetscInt its, PetscInt nconv, PetscScalar *, PetscScalar *, PetscReal *, PetscInt,
                          void *ctx)
{
    PetscFunctionBeginUser;
    EpsMonitorCtx *c = static_cast<EpsMonitorCtx *>(ctx);
    TDSEZInfo().row("EPS  it = %-5d converged %4d / %-4d (%5.1f%%)   new %+d", (int)its,
                    (int)nconv, (int)c->nev, 100.0 * nconv / PetscMax(c->nev, (PetscInt)1),
                    (int)(nconv - c->prev));
    c->prev = nconv;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// EPSSetUp or EPSSolve with errors returned instead of aborting: an inner
/// solve that does not converge or a failed factorisation leaves *err != 0
/// and the caller tries another inner solver.
PetscErrorCode EpsTry(PetscErrorCode (*f)(EPS), EPS eps, PetscErrorCode *err)
{
    PetscFunctionBeginUser;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    *err = f(eps);
    PetscCall(PetscPopErrorHandler());
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

static PetscReal TDSEZDegeneracyTol()
{
    PetscReal tol = 1e-9;
    PetscOptionsGetReal(NULL, NULL, "-tdsez_degeneracy_tol", &tol, NULL);
    return tol;
}

static PetscErrorCode TDSEZCanonicalEigenpairs(EPS eps, IGA iga, Mat M, PetscInt nev,
                                               std::vector<Vec> &vecs, std::vector<PetscReal> &vals)
{
    PetscFunctionBeginUser;
    for (auto &v : vecs) PetscCall(VecDestroy(&v));
    vecs.clear(); vals.clear();
    const PetscReal tol = TDSEZDegeneracyTol();
    auto same = [tol](PetscReal x, PetscReal y) {
        return PetscAbsReal(x - y) <= tol * PetscMax((PetscReal)1.0, PetscAbsReal(x));
    };

    // 1. Which states to keep. Pairs are ordered by distance to the target
    //    (SLEPc's selection), ties by energy; the first nev are kept, extended
    //    to the end of the shell of the last one.
    const PetscReal target = PetscRealPart(TDSEZParser::TargetEigenvalue);
    std::vector<std::pair<PetscReal, Vec>> L;                   // (E, vector)
    PetscInt nconv = 0;
    PetscCall(EPSGetConverged(eps, &nconv));
    for (PetscInt i = 0; i < nconv; ++i) {
        PetscScalar e;
        Vec v;
        PetscCall(IGACreateVec(iga, &v));
        PetscCall(EPSGetEigenpair(eps, i, &e, NULL, v, NULL));
        L.emplace_back(PetscRealPart(e), v);
    }
    auto before = [target](const std::pair<PetscReal, Vec> &x, const std::pair<PetscReal, Vec> &y) {
        const PetscReal dx = PetscAbsReal(x.first - target), dy = PetscAbsReal(y.first - target);
        return dx != dy ? dx < dy : x.first < y.first;
    };
    std::stable_sort(L.begin(), L.end(), before);
    auto keepCount = [&]() {
        PetscInt n = PetscMin(nev, (PetscInt)L.size());
        while (n > 0 && n < (PetscInt)L.size() && same(L[n].first, L[n - 1].first)) ++n;
        return n;
    };
    // Krylov solvers can miss copies of a repeated eigenvalue, and when every
    // converged pair is kept the next eigenvalue is unknown. Whenever the kept
    // set contains a degenerate shell or ends at the last converged pair,
    // deflate the kept states and solve for the nearest remaining pair: if it
    // is not farther from the target than the kept states, or has the energy
    // of a kept shell, it was missed and is added; otherwise the kept set is
    // complete. The test needs only the eigenvalue (Hermitian pencil: error
    // ~ residual^2), so it runs at a loose tolerance; a missed pair is
    // recomputed at the full tolerance before it is added.
    {
        PetscReal tolFull; PetscInt itFull;
        PetscCall(EPSGetTolerances(eps, &tolFull, &itFull));
        PetscInt added = 0;
        for (int it = 0; it < 64; ++it) {
            const PetscInt nkeep = keepCount();
            if (nkeep == 0) break;
            bool degenerate = nkeep == (PetscInt)L.size();
            for (PetscInt i = 1; i < nkeep && !degenerate; ++i)
                if (same(L[i].first, L[i - 1].first)) degenerate = true;
            if (!degenerate) break;
            std::vector<Vec> defl(nkeep);
            for (PetscInt i = 0; i < nkeep; ++i) defl[i] = L[i].second;
            PetscCall(EPSSetDeflationSpace(eps, nkeep, defl.data()));
            PetscCall(EPSSetDimensions(eps, 1, PETSC_DETERMINE, PETSC_DETERMINE));
            PetscCall(EPSSetTolerances(eps, PetscMax(tolFull, (PetscReal)1e-6), itFull));
            PetscErrorCode err = PETSC_SUCCESS;
            PetscCall(EpsTry(EPSSolve, eps, &err));
            PetscInt nc = 0;
            if (!err) PetscCall(EPSGetConverged(eps, &nc));
            if (err || nc == 0) {
                TDSEZInfo().warn("degenerate-shell check did not converge; the kept states may miss copies of a degenerate level");
                break;
            }
            PetscScalar e;
            PetscCall(EPSGetEigenvalue(eps, 0, &e, NULL));
            const PetscReal en = PetscRealPart(e);
            bool missed = PetscAbsReal(en - target) <= PetscAbsReal(L[nkeep - 1].first - target);
            for (PetscInt i = 0; i < nkeep && !missed; ++i) if (same(en, L[i].first)) missed = true;
            if (!missed) break;                                   // kept set complete
            // Recompute it at the full tolerance from the loose vector. The
            // deflation and initial spaces do not persist from one EPSSolve to
            // the next; both are set again.
            {
                Vec v0;
                PetscCall(IGACreateVec(iga, &v0));
                PetscCall(EPSGetEigenpair(eps, 0, &e, NULL, v0, NULL));
                PetscCall(EPSSetDeflationSpace(eps, nkeep, defl.data()));
                PetscCall(EPSSetInitialSpace(eps, 1, &v0));
                PetscCall(VecDestroy(&v0));
            }
            PetscCall(EPSSetTolerances(eps, tolFull, itFull));
            PetscCall(EpsTry(EPSSolve, eps, &err));
            if (err) break;
            PetscCall(EPSGetConverged(eps, &nc));
            if (nc == 0) break;
            Vec v;
            PetscCall(IGACreateVec(iga, &v));
            PetscCall(EPSGetEigenpair(eps, 0, &e, NULL, v, NULL));
            L.emplace_back(PetscRealPart(e), v);
            std::stable_sort(L.begin(), L.end(), before);
            ++added;
        }
        PetscCall(EPSSetTolerances(eps, tolFull, itFull));
        if (added)
            TDSEZInfo().status("ⵣ", "Degenerate-shell check: added %" PetscInt_FMT " pair(s) missed by the solver", added);
    }
    const PetscInt nkeep = keepCount();
    for (PetscInt i = 0; i < (PetscInt)L.size(); ++i) {
        if (i < nkeep) { vecs.push_back(L[i].second); vals.push_back(L[i].first); }
        else PetscCall(VecDestroy(&L[i].second));
    }
    std::vector<PetscReal> E(vals);
    if (nkeep > nev)
        TDSEZInfo().status("ⵣ", "NBoundStates = %" PetscInt_FMT " ends inside a degenerate shell; keeping %"
            PetscInt_FMT " states (whole shell)", nev, nkeep);

    // 2. Canonical basis inside each degenerate shell.
    Vec w, Wv;
    PetscCall(IGACreateVec(iga, &w));
    PetscCall(TDSEZShellWeight(iga, w));
    PetscCall(VecDuplicate(w, &Wv));
    for (PetscInt s = 0; s < nkeep; ) {
        PetscInt e = s + 1;
        while (e < nkeep && same(E[e], E[s])) ++e;
        const PetscInt g = e - s;
        if (g > 1) {
            // M-orthonormalise the shell (Gram-Schmidt in the M inner product),
            // then O_ab = v_a^H W v_b.
            Vec Mv;
            PetscCall(VecDuplicate(w, &Mv));
            for (PetscInt a = s; a < e; ++a) {
                for (PetscInt b = s; b < a; ++b) {
                    PetscScalar ov;
                    PetscCall(MatMult(M, vecs[b], Mv));
                    PetscCall(VecDot(vecs[a], Mv, &ov));          // v_b^H M v_a
                    PetscCall(VecAXPY(vecs[a], -ov, vecs[b]));
                }
                PetscScalar nn;
                PetscCall(MatMult(M, vecs[a], Mv));
                PetscCall(VecDot(Mv, vecs[a], &nn));
                PetscCall(VecScale(vecs[a], 1.0 / PetscSqrtReal(PetscRealPart(nn))));
            }
            PetscCall(VecDestroy(&Mv));
            std::vector<PetscScalar> O((size_t)g * g);
            for (PetscInt b = 0; b < g; ++b) {
                PetscCall(VecPointwiseMult(Wv, w, vecs[s + b]));
                for (PetscInt a = 0; a < g; ++a) {
                    PetscScalar o;
                    PetscCall(VecDot(Wv, vecs[s + a], &o));      // v_a^H W v_b
                    O[(size_t)a + (size_t)g * b] = o;             // column-major
                }
            }
            for (PetscInt b = 0; b < g; ++b)                      // exact Hermitian
                for (PetscInt a = 0; a < b; ++a) {
                    const PetscScalar m = 0.5 * (O[a + (size_t)g * b] + PetscConj(O[b + (size_t)g * a]));
                    O[a + (size_t)g * b] = m; O[b + (size_t)g * a] = PetscConj(m);
                }
            PetscBLASInt n = (PetscBLASInt)g, lwork = (PetscBLASInt)(4 * g), info = 0;
            std::vector<PetscReal> ev(g);
            std::vector<PetscScalar> work(4 * g);
#if defined(PETSC_USE_COMPLEX)
            std::vector<PetscReal> rwork(3 * g);
            LAPACKsyev_("V", "U", &n, O.data(), &n, ev.data(), work.data(), &lwork, rwork.data(), &info);
#else
            LAPACKsyev_("V", "U", &n, O.data(), &n, ev.data(), work.data(), &lwork, &info);
#endif
            PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK syev/heev failed: %d", (int)info);
            for (PetscInt k = 1; k < g; ++k)
                if (ev[k] - ev[k - 1] <= 1e-8 * PetscMax((PetscReal)1.0, PetscAbsReal(ev[k])))
                    TDSEZInfo().warn("shell at E = %.8g is not resolved by the canonical weight; "
                        "its basis may not be reproducible", (double)E[s]);
            // u_k = sum_a v_a U_ak (columns of O hold U), sorted by <W> ascending.
            std::vector<Vec> u(g);
            for (PetscInt k = 0; k < g; ++k) {
                PetscCall(VecDuplicate(w, &u[k]));
                PetscCall(VecSet(u[k], 0.0));
                PetscCall(VecMAXPY(u[k], g, &O[(size_t)g * k], &vecs[s]));
            }
            PetscReal Emean = 0.0;                                 // one energy per shell
            for (PetscInt k = 0; k < g; ++k) Emean += E[s + k] / g;
            for (PetscInt k = 0; k < g; ++k) {
                PetscCall(VecDestroy(&vecs[s + k]));
                vecs[s + k] = u[k];
                vals[s + k] = Emean;
            }
        }
        s = e;
    }
    PetscCall(VecDestroy(&Wv));
    PetscCall(VecDestroy(&w));

    // 3. Deterministic phase for every kept state.
    Vec gw = PETSC_NULLPTR, gw2 = PETSC_NULLPTR;
    PetscCall(TDSEZGaugeWeightVector(iga, &gw, 0));
    PetscCall(TDSEZGaugeWeightVector(iga, &gw2, 1));
    for (PetscInt i = 0; i < nkeep; ++i) PetscCall(TDSEZFixEigenvectorGauge(vecs[i], M, gw, gw2, i));
    PetscCall(VecDestroy(&gw));
    PetscCall(VecDestroy(&gw2));
    PetscFunctionReturn(PETSC_SUCCESS);
}



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

    // ── Eigensolver strategy ─────────────────────────────────────────────
    // Krylov-Schur on the shift-invert operator. Inner solves of H - sigma M:
    // direct (exact) in 1D/2D and for small high-degree 3D problems, where
    // the overlapping ASM blocks cover most of the domain; FGMRES with
    // restricted ASM(1)/ILU(0) otherwise. Every attempt is validated (inner
    // solve converged, nev pairs converged, backward errors <= 1e-8); a failed
    // attempt is followed by the next inner solver in the plan. -DirectSolve 1
    // or -tdsez_eps_inner direct|iterative choose the first one; -st_ksp_*,
    // -st_pc_* and -st_sub_* options still apply on top.
    const PetscBool isHuge = (mat_rows > 1000000) ? PETSC_TRUE : PETSC_FALSE;
    const PetscInt  dimension = TDSEZParser::Dimension, degree = TDSEZParser::SplineDegree;
    const char     *directPackage = EpsDirectPackage(H, (PetscMPIInt)nprocs);
    const bool      directFits = directPackage && (dimension < 3 ? mat_rows <= 2000000 : mat_rows <= 300000);
    std::vector<EpsInner> plan;
    {
        char mode[32] = "auto";
        PetscCall(PetscOptionsGetString(NULL, NULL, "-tdsez_eps_inner", mode, sizeof(mode), NULL));
        bool direct;
        if (useDirectSolve == 1 || !strcmp(mode, "direct")) direct = directPackage != nullptr;
        else if (!strcmp(mode, "iterative"))                 direct = false;
        else direct = directFits && (dimension < 3 || (degree >= 6 && mat_rows <= 30000));
        plan.push_back(direct ? EpsInner::Direct : EpsInner::Iterative);
        if (direct) plan.push_back(EpsInner::Iterative);
        else if (directFits) plan.push_back(EpsInner::Direct);
        plan.push_back(EpsInner::IterativeStrong);
    }

    // Krylov-Schur subspace: SLEPc's default ncv = max(2 nev, nev + 15),
    // mpd = ncv; the larger-subspace retry doubles ncv.
    const PetscInt ncv = PetscMax(2 * nev, nev + 15), mpd = ncv;
    const PetscReal epsTol = 1e-12, acceptRes = 1e-8;
    const PetscInt  epsMaxIt = 2000;

    // Problem info
    info.section("PROBLEM");
    {
        auto fmtDom = [](double lo, double hi) -> std::string {
            char b[48];
            snprintf(b, sizeof(b), "[%+.1f, %+.1f]", lo, hi);
            return std::string(b);
        };
        const double lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
        const double hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
        std::string dom;
        for (PetscInt d = 0; d < TDSEZParser::Dimension; ++d) dom += (d ? " x " : "") + fmtDom(lo[d], hi[d]);
        info.kv("Dimension", "%dD", (int)TDSEZParser::Dimension);
        info.kv("Domain", "%s a.u.", dom.c_str());
    }
    info.kv("Boundary",           "%s", TDSEZParser::BoundaryType.c_str());
    info.kv("Degrees of freedom", "%" PetscInt_FMT " (%.3e per rank, %" PetscInt_FMT " ranks)",
            mat_rows, (double)mat_rows / nprocs, (PetscInt)nprocs);
    info.kv("Vector size",        "%.1f MB", (double)mat_rows * sizeof(PetscScalar) / 1048576.0);
    // Initial state info
    if (TDSEZParser::InitialStateMode == 2) {
        std::string desc;
        for (const auto &term : TDSEZParser::InitialStateTerms) {
            if (!desc.empty()) desc += " + ";
            desc += TDSEZ_fmtZ(term.second) + "*state:" +
                     std::to_string(term.first);
        }
        info.kv("Initial state", "%s (%s)", desc.c_str(), TDSEZParser::NormalizeInitialState ? "normalised" : "not normalised");
    } else if (TDSEZParser::InitialStateMode == 3) {
        info.kv("Initial state", "analytic Quesne n=0,1 (M-projected, normalised)");
    } else if (TDSEZParser::InitialStateMode == 4) {
        info.kv("Initial state", "analytic mass-step n=0,1 (M-projected, normalised)");
    } else if (TDSEZParser::InitialStateMode == 5) {
        info.kv("Initial state", "Gaussian wave packet (M-projected, normalised)");
    } else {
        info.kv("Initial state",  "state:%" PetscInt_FMT, TDSEZParser::InitialStateIndex);
    }

    // NBoundStates = 0 (analytic initial states only): no eigenproblem.
    if (nev == 0) {
        nconv = 0;
        info.section("EIGENSOLVER");
        info.status("ⵣ", "NBoundStates = 0: eigensolver skipped (analytic initial state)");
    } else {
        info.section("EIGENSOLVER");
        info.kv("Method", "Krylov-Schur, shift-invert");
        info.kv("Inner linear solver","%s%s", EpsInnerName(plan[0]),
                plan[0] == EpsInner::Direct ? (std::string(" (") + directPackage + ")").c_str() : "");
        info.kv("Target eigenvalue", "%s a.u.", TDSEZ_fmtAu((double)PetscRealPart(targetEV)).c_str());
        info.kv("Pairs requested (nev)", "%" PetscInt_FMT, nev);
        info.kv("Subspace (ncv, mpd)", "%" PetscInt_FMT ", %" PetscInt_FMT, ncv, mpd);
        info.kv("Tolerance, max iterations", "%.0e, %" PetscInt_FMT, (double)epsTol, epsMaxIt);
        info.kv("Accepted backward error", "%.0e", (double)acceptRes);
        info.kv("Krylov restart ratio", "%.2f", isHuge ? 0.3 : 0.5);
        info.kv("Subspace memory", "%.1f MB", (double)ncv * mat_rows * sizeof(PetscScalar) / 1048576.0);
        info.memKV("Memory, rank 0");

        // ── Attempts ─────────────────────────────────────────────────────
        PetscReal normH, normM;
        PetscCall(MatNorm(H, NORM_INFINITY, &normH));
        PetscCall(MatNorm(M, NORM_INFINITY, &normM));
        PetscBool userSubLevels = PETSC_FALSE, userSubShift = PETSC_FALSE;
        PetscCall(PetscOptionsHasName(NULL, NULL, "-st_sub_pc_factor_levels", &userSubLevels));
        PetscCall(PetscOptionsHasName(NULL, NULL, "-st_sub_pc_factor_shift_type", &userSubShift));
        EpsMonitorCtx monitorCtx;              // every solve of eps runs inside Solve()
        monitorCtx.nev = nev;

        // Configure eps for one inner solver; scale > 1 enlarges the subspace.
        auto createEps = [&](EpsInner mode, PetscInt scale) -> PetscErrorCode {
            PetscFunctionBeginUser;
            PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
            PetscCall(EPSSetOperators(eps, H, M));
            PetscCall(EPSSetProblemType(eps, EPS_GHEP));
            // Under shift-invert the eigenvalues nearest the target are the
            // largest of the transformed operator: EPS_TARGET_REAL selects them.
            PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL));
            PetscCall(EPSSetTarget(eps, targetEV));
            PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR));
            PetscCall(EPSSetDimensions(eps, nev, ncv * scale, PETSC_DETERMINE));
            PetscCall(EPSKrylovSchurSetRestart(eps, isHuge ? 0.3 : 0.5));
            PetscCall(EPSSetTolerances(eps, epsTol, epsMaxIt * scale));

            PetscCall(EPSGetST(eps, &st));
            // The shift is offset slightly below the target: inputs often set
            // the target to a known eigenvalue, and a shift on an eigenvalue
            // makes H - sigma M singular to rounding. The selection of the
            // eigenvalues nearest the target is unaffected.
            const PetscReal shiftOffset = 1e-3 * PetscMax((PetscReal)1.0, PetscAbsScalar(targetEV));
            PetscCall(STSetShift(st, targetEV - shiftOffset));
            PetscCall(STSetMatStructure(st, SAME_NONZERO_PATTERN));
            PetscCall(STSetType(st, STSINVERT));
            PetscCall(STGetKSP(st, &ksp));
            PetscCall(KSPGetPC(ksp, &pc));
            if (mode == EpsInner::Direct) {
                PetscCall(KSPSetType(ksp, KSPPREONLY));
                PetscCall(PCSetType(pc, PCLU));
                PetscCall(PCFactorSetMatSolverType(pc, directPackage));
                PetscBool set;
                PetscCall(PetscOptionsHasName(NULL, NULL, "-st_mat_mumps_icntl_14", &set));
                if (!set && !strcmp(directPackage, MATSOLVERMUMPS))
                    PetscCall(PetscOptionsSetValue(NULL, "-st_mat_mumps_icntl_14", "50"));   // workspace +50%
            } else {
                const bool strong = mode == EpsInner::IterativeStrong;
                PetscCall(KSPSetType(ksp, KSPFGMRES));
                PetscCall(KSPGMRESSetRestart(ksp, isHuge ? 50 : 100));
                PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, strong ? 3000 : 500));
                PetscCall(PCSetType(pc, PCASM));
                PetscCall(PCASMSetType(pc, PC_ASM_RESTRICT));
                PetscCall(PCASMSetOverlap(pc, strong ? 2 : 1));
            }
            PetscCall(KSPSetFromOptions(ksp));
            PetscCall(STSetFromOptions(st));
            // The B-spline overlap matrix becomes very ill-conditioned at high
            // degree (3D, p >= 7): x^H M x then carries a rounding-level
            // imaginary part above SLEPc's default definiteness check although
            // M is SPD by construction. 1e-10 relative is still strict; the
            // eigenpairs are validated by their residuals. -bv_definite_tol.
            BV bv;
            PetscCall(EPSGetBV(eps, &bv));
            PetscCall(BVSetDefiniteTolerance(bv, 1e-10));
            monitorCtx.prev = 0;
            PetscCall(EPSMonitorSet(eps, EpsMonitor, &monitorCtx, PETSC_NULLPTR));
            PetscCall(EPSSetFromOptions(eps));
            PetscFunctionReturn(PETSC_SUCCESS);
        };
        // ASM blocks: ILU with the requested fill and in-block shifts against
        // zero pivots of the indefinite H - sigma M (after EPSSetUp has
        // created the blocks, before their factorisation in the first solve).
        auto configureBlocks = [&](EpsInner mode) -> PetscErrorCode {
            PetscFunctionBeginUser;
            PetscBool isASM = PETSC_FALSE;
            PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCASM, &isASM));
            if (!isASM || mode == EpsInner::Direct) PetscFunctionReturn(PETSC_SUCCESS);
            PetscInt nlocal = 0;
            KSP *sub = NULL;
            PetscCall(PCASMGetSubKSP(pc, &nlocal, NULL, &sub));
            for (PetscInt i = 0; i < nlocal; ++i) {
                PC spc;
                PetscCall(KSPGetPC(sub[i], &spc));
                PetscBool isILU = PETSC_FALSE;
                PetscCall(PetscObjectTypeCompare((PetscObject)spc, PCILU, &isILU));
                if (!isILU) continue;
                if (!userSubLevels) PetscCall(PCFactorSetLevels(spc, mode == EpsInner::IterativeStrong ? 1 : 0));
                if (!userSubShift) {
                    PetscCall(PCFactorSetShiftType(spc, MAT_SHIFT_INBLOCKS));
                    PetscCall(PCFactorSetShiftAmount(spc, 1e-12));
                }
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        };

        info.section("SOLVE");
        EPSConvergedReason reason = EPS_CONVERGED_ITERATING;
        PetscInt its = 0;
        PetscReal worst = 0;
        bool accepted = false;
        EpsInner used = plan[0];
        Vec vtmp;
        PetscCall(MatCreateVecs(H, &vtmp, PETSC_NULLPTR));
        for (size_t k = 0; k < plan.size() && !accepted; ++k) {
            for (PetscInt scale = 1; scale <= 2 && !accepted; ++scale) {
                if (eps) PetscCall(EPSDestroy(&eps));
                nconv = 0;
                used = plan[k];
                PetscCall(createEps(plan[k], scale));
                info.status("◆", "EPSSolve, inner solver %s%s", EpsInnerName(plan[k]), scale > 1 ? ", larger subspace" : "");
                PetscErrorCode err = PETSC_SUCCESS;
                PetscLogDouble ta, tb;
                PetscCall(PetscTime(&ta));
                PetscCall(EpsTry(EPSSetUp, eps, &err));
                PetscCall(PetscTime(&tb));
                TDSEZProfile::Get().Sub(TDSEZProfile::Eigensolve, "Solver setup (EPSSetUp)", tb - ta);
                if (!err) {
                    if (k == 0 && scale == 1) info.memKV("Memory after setup, rank 0");
                    PetscCall(configureBlocks(plan[k]));
                    PetscCall(PetscTime(&ta));
                    PetscCall(EpsTry(EPSSolve, eps, &err));
                    PetscCall(PetscTime(&tb));
                    TDSEZProfile::Get().Sub(TDSEZProfile::Eigensolve, "Krylov-Schur iterations (EPSSolve)", tb - ta);
                }
                if (err) {
                    info.warn("inner solve failed (%s, PETSc error %d); trying the next inner solver",
                        EpsInnerName(plan[k]), (int)err);
                    break;
                }
                PetscCall(EPSGetConverged(eps, &nconv));
                PetscCall(EPSGetConvergedReason(eps, &reason));
                PetscCall(EPSGetIterationNumber(eps, &its));
                if (nconv < nev) {
                    info.warn("%" PetscInt_FMT " of %" PetscInt_FMT " pairs converged (%s)%s", nconv, nev,
                        EPSConvergedReasons[reason], scale == 1 ? "; retrying with a larger subspace" : "");
                    continue;
                }
                worst = 0;
                for (PetscInt i = 0; i < nev; ++i) {
                    PetscScalar e;
                    PetscReal r;
                    PetscCall(EPSGetEigenpair(eps, i, &e, NULL, vtmp, NULL));
                    PetscCall(TDSEZBackwardError(H, M, normH, normM, vtmp, PetscRealPart(e), &r));
                    worst = PetscMax(worst, r);
                }
                if (worst > acceptRes) {
                    info.warn("backward error %.1e > %.0e with %s; trying the next inner solver",
                        (double)worst, (double)acceptRes, EpsInnerName(plan[k]));
                    break;
                }
                accepted = true;
            }
        }
        PetscCall(VecDestroy(&vtmp));

        info.memKV("Memory after solve, rank 0");

        // ── Results ──────────────────────────────────────────────────────
        if (eps && !nconv) PetscCall(EPSGetConverged(eps, &nconv));
        info.section("RESULTS");
        info.kv("Inner solver used",   "%s", EpsInnerName(used));
        info.kv("Outer iterations",    "%" PetscInt_FMT, its);
        info.kv("Converged pairs",     "%" PetscInt_FMT " / %" PetscInt_FMT, nconv, nev);
        info.kv("Stop reason",         "%s",             EPSConvergedReasons[reason]);
        if (accepted) info.kv("Max backward error", "%.1e", (double)worst);

        if (nconv == 0) {
            info.status("ⵀ", "No eigenpairs converged with any inner solver (%zu tried).", plan.size());
            info.status("ⵀ", "Check TargetEigenvalue, the box and the grid; -tdsez_eps_inner direct forces an exact solve.");
            info.footer();
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
        if (!accepted)
            info.warn("no inner solver met every check; using the %" PetscInt_FMT " converged pair(s) of the last attempt",
                nconv);

        // Reproducible eigenpairs: whole degenerate shells, canonical basis inside
        // each shell, fixed phases. Every later use reads eigVecs / eigVals.
        {
            PetscLogDouble ta, tb;
            PetscCall(PetscTime(&ta));
            PetscCall(EPSMonitorCancel(eps));   // the shell check below solves again
            PetscCall(TDSEZCanonicalEigenpairs(eps, iga, M, nev, eigVecs, eigVals));
            PetscCall(PetscTime(&tb));
            TDSEZProfile::Get().Sub(TDSEZProfile::Eigensolve, "Degenerate shells and phases", tb - ta);
        }
        nconv = (PetscInt)eigVecs.size();
        eigRes_.clear();

        // Converged eigenvalues
        info.section("EIGENVALUES");
        info.row("%5s %18s %12s %16s", "n", "E (a.u.)", "E (eV)", "backward error");
        for (PetscInt i = 0; i < nconv; i++) {
            PetscReal errest;
            PetscCall(TDSEZBackwardError(H, M, normH, normM, eigVecs[i], eigVals[i], &errest));
            eigRes_.push_back(errest);
            info.row("%5" PetscInt_FMT " %18s %12s %16.1e", i, TDSEZ_fmtAu((double)eigVals[i]).c_str(),
                     TDSEZ_fmtEv((double)eigVals[i]).c_str(), (double)errest);
        }
    }

    // Extract initial state
    //   mode 0/1 -> single converged eigenstate (ground or state:N)
    //   mode 2   -> coherent superposition of up to 3 bound states
    //   mode 3/4 -> exact analytic state, L2-projected into this IGA basis
    info.section("INITIAL STATE");
    PetscInt initIdx = TDSEZParser::InitialStateIndex;
    auto buildFromIndex = [&](PetscInt k, Vec *out) -> PetscErrorCode {
        if (k >= nconv)
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                    "Initial state index %" PetscInt_FMT " >= number of converged eigenstates %"
                    PetscInt_FMT " — increase NBoundStates or check solver convergence.",
                    k, nconv);
        PetscCall(IGACreateVec(iga, out));
        PetscCall(VecCopy(eigVecs[k], *out));
        PetscFunctionReturn(PETSC_SUCCESS);
    };

    if (TDSEZParser::InitialStateMode == 2) {
        // Assemble psi0 = sum_i coeff_i * eigenvector(idx_i)
        PetscCall(IGACreateVec(iga, &initialPsi));
        PetscCall(VecSet(initialPsi, 0.0));
        // eigVecs already carry the fixed gauge (TDSEZCanonicalEigenpairs).
        std::string desc;
        for (const auto &term : TDSEZParser::InitialStateTerms) {
            PetscInt      k = term.first;
            PetscScalar   c = term.second;
            Vec evec = PETSC_NULLPTR;
            PetscCall(buildFromIndex(k, &evec));
            PetscCall(VecAXPY(initialPsi, c, evec));
            PetscCall(VecDestroy(&evec));
            if (!desc.empty()) desc += " + ";
            char cbuf[64];
            snprintf(cbuf, sizeof(cbuf), "%s*state:%" PetscInt_FMT,
                     TDSEZ_fmtZ(c).c_str(), k);
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
    } else if (TDSEZParser::InitialStateMode == 3 ||
               TDSEZParser::InitialStateMode == 4 ||
               TDSEZParser::InitialStateMode == 5) {
        // These states are analytic; neither is constructed from EPS vectors.
        // Assemble b_i = integral N_i(x) psi_exact(x,0) dx, then solve M c=b.
        const PetscBool quesne = TDSEZParser::InitialStateMode == 3;
        const PetscReal a = 2.0, omega = 1.0, alpha = 4.0;
        const PetscReal c0 = PetscPowReal(omega * a * a, (alpha + 1.0) / 2.0) /
                             PetscSqrtReal(a * std::tgamma(alpha + 1.0));
        const PetscReal c1 = PetscPowReal(omega * a * a, (alpha + 1.0) / 2.0) /
                             PetscSqrtReal(a * std::tgamma(alpha + 2.0));
        auto psiExact = [=](PetscReal x) -> PetscScalar {
            if (!quesne) {
                const PetscReal theta0 = PetscAcosReal(1.0 / PetscSqrtReal(6.0));
                const PetscReal theta1 = PETSC_PI - theta0;
                auto stepPhi = [=](PetscReal theta, PetscReal sign) {
                    return x < 0.0
                        ? 2.0 / PetscSqrtReal(5.0) * PetscSinReal(theta * (x + 1.0))
                        : sign * PetscSqrtReal(6.0 / 5.0) *
                          PetscSinReal(2.0 * theta * (1.0 - x));
                };
                return (stepPhi(theta0, 1.0) + stepPhi(theta1, -1.0)) /
                       PetscSqrtReal(2.0);
            }
            if (x <= -a) return 0.0;
            const PetscReal u = 1.0 + x / a;
            const PetscReal z = omega * a * (x + a);
            const PetscReal common = PetscPowReal(u, alpha / 2.0) * PetscExpReal(-z / 2.0);
            const PetscReal phi0 = c0 * common;
            const PetscReal phi1 = c1 * common * (alpha + 1.0 - z);
            return (phi0 + phi1) / PetscSqrtReal(2.0);
        };
        // Gaussian wave packet, normalised in the continuum (InitialState = wavepacket)
        auto wavepacket = [](const PetscReal *x) -> PetscScalar {
            PetscScalar v = 1.0;
            for (PetscInt d = 0; d < TDSEZParser::Dimension; ++d) {
                const PetscReal w = TDSEZParser::WavepacketWidth[d];
                const PetscReal u = x[d] - TDSEZParser::WavepacketCenter[d];
                v *= PetscPowReal(2.0 * PETSC_PI * w * w, -0.25) *
                     PetscExpScalar(PetscCMPLX(-u * u / (4.0 * w * w), TDSEZParser::WavepacketMomentum[d] * x[d]));
            }
            return v;
        };
        Vec rhs = PETSC_NULLPTR;
        PetscCall(IGACreateVec(iga, &rhs));
        PetscCall(VecSet(rhs, 0.0));
        IGAElement element = PETSC_NULLPTR;
        IGAPoint point = PETSC_NULLPTR;
        PetscCall(IGABeginElement(iga, &element));
        while (IGANextElement(iga, element)) {
            PetscScalar *local = PETSC_NULLPTR;
            PetscCall(IGAElementGetWorkVec(element, &local));
            PetscCall(PetscArrayzero(local, element->nen));
            PetscCall(IGAElementBeginPoint(element, &point));
            while (IGAElementNextPoint(element, point)) {
                const PetscScalar value = (TDSEZParser::InitialStateMode == 5)
                                        ? wavepacket(point->point) : psiExact(point->point[0]);
                const PetscReal weight = point->weight[0] * point->detJac[0];
                for (PetscInt i = 0; i < point->nen; ++i)
                    local[i] += weight * point->shape[0][i] * value;
            }
            PetscCall(IGAElementEndPoint(element, &point));
            PetscCall(IGAElementAssembleVec(element, local, rhs));
        }
        PetscCall(IGAEndElement(iga, &element));
        PetscCall(VecAssemblyBegin(rhs));
        PetscCall(VecAssemblyEnd(rhs));
        PetscCall(IGACreateVec(iga, &initialPsi));
        KSP projection = PETSC_NULLPTR;
        PetscCall(KSPCreate(PETSC_COMM_WORLD, &projection));
        PetscCall(KSPSetOperators(projection, M, M));
        PetscCall(KSPSetType(projection, KSPCG));
        PC projectionPC = PETSC_NULLPTR;
        PetscCall(KSPGetPC(projection, &projectionPC));
        PetscCall(PCSetType(projectionPC, PCJACOBI));
        PetscCall(KSPSetTolerances(projection, 1e-13, PETSC_DEFAULT, PETSC_DEFAULT, 10000));
        PetscCall(KSPSolve(projection, rhs, initialPsi));
        KSPConvergedReason projectionReason;
        PetscCall(KSPGetConvergedReason(projection, &projectionReason));
        if (projectionReason < 0)
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
                    "M-projection of analytic initial state did not converge");
        PetscInt projectionIts;
        PetscReal projectionResidual;
        PetscCall(KSPGetIterationNumber(projection, &projectionIts));
        PetscCall(KSPGetResidualNorm(projection, &projectionResidual));
        PetscCall(KSPDestroy(&projection));
        PetscCall(VecDestroy(&rhs));
        Vec Mpsi = PETSC_NULLPTR;
        PetscCall(MatCreateVecs(M, &Mpsi, PETSC_NULLPTR));
        PetscCall(MatMult(M, initialPsi, Mpsi));
        PetscScalar nrm2;
        PetscCall(VecDot(Mpsi, initialPsi, &nrm2));
        const PetscReal nrm = PetscSqrtReal(PetscRealPart(nrm2));
        if (nrm <= 0.0)
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                    "Analytic initial state has zero M-norm after projection");
        PetscCall(VecScale(initialPsi, 1.0 / nrm));
        PetscCall(MatMult(M, initialPsi, Mpsi));
        PetscCall(VecDot(Mpsi, initialPsi, &nrm2));
        PetscCall(VecDestroy(&Mpsi));
        info.status("ⵣ", TDSEZParser::InitialStateMode == 5 ? "Gaussian wave packet projected into spline basis"
                         : quesne ? "Analytic Quesne n=0,1 state projected into spline basis"
                                  : "Analytic mass-step n=0,1 state projected into spline basis");
        info.kv("  projection KSP iterations", "%" PetscInt_FMT, projectionIts);
        info.kv("  projection residual", "%.3e", (double)projectionResidual);
        info.kv("  ||psi||_M (before)", "%.12f", (double)nrm);
        info.kv("  ||psi||_M (after)", "%.12f", (double)PetscSqrtReal(PetscRealPart(nrm2)));
    } else {
        if (initIdx >= nconv)
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                    "Initial state index %" PetscInt_FMT " >= number of converged eigenstates %"
                    PetscInt_FMT " — increase NBoundStates or check solver convergence.",
                    initIdx, nconv);
        Vec epsPsi;
        PetscCall(MatCreateVecs(H, &epsPsi, PETSC_NULLPTR));
        PetscCall(VecCopy(eigVecs[initIdx], epsPsi));
        PetscCall(IGACreateVec(iga, &initialPsi));
        PetscCall(VecCopy(epsPsi, initialPsi));
        PetscCall(VecDestroy(&epsPsi));

        if (initIdx == 0)
            info.status("ⵣ", "Ground state extracted (InitialState = ground)");
        else
            info.status("ⵣ", "Bound state #%" PetscInt_FMT " extracted (InitialState = state:%" PetscInt_FMT ")",
                   initIdx, initIdx);
    }

    // Sanity check: the overlap matrix M must give each computed eigenstate a
    // positive norm psi^H M psi. This does not test current or flux
    // conservation; the length-velocity check (TDSEZCheckLengthVelocity)
    // is the operator-level test of the BDD velocity.
    {
        PetscBool conserved = PETSC_TRUE;
        Vec tmp = PETSC_NULLPTR;
        Vec psi_i = PETSC_NULLPTR;
        PetscCall(MatCreateVecs(H, &tmp, PETSC_NULLPTR));
        PetscCall(MatCreateVecs(H, &psi_i, PETSC_NULLPTR));
        for (PetscInt i = 0; i < nconv; ++i) {
            PetscCall(VecCopy(eigVecs[i], psi_i));
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
               conserved ? "Overlap-matrix norms of the eigenstates: positive" : "Overlap-matrix norms of the eigenstates: NOT positive");
    }

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
        TDSEZInfo().fatal("no eigenvalues converged");
        VecDestroy(&psi);
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // nconv = canonical states kept by Solve(): NBoundStates, extended to the
    // end of a degenerate shell.
    PetscInt  nsave   = nconv;
    PetscBool savewfs = TDSEZParser::NBoundStatesSave;
    PetscInt  nsavePop = nconv;

    // HDF5 setup — only if enabled
    std::string h5file = TDSEZPaths::Static(inputFile, "eigen.h5");
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
        PetscScalar kr = eigVals[i];
        energies.push_back(PetscRealPart(kr));

        VecSetValue(spectrum, i, PetscRealPart(kr), INSERT_VALUES);

        VecCopy(eigVecs[i], psi);

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
            // Keep the structured PetIGA view for plotting, and also export a
            // plain vector in natural ordering, matching PetIGA's MatView
            // permutation for saved matrices. Generic VecLoad cannot interpret
            // a structured 3D DMDA dataset. MPI-global ordering is NOT generally
            // natural ordering (they coincide only for some decompositions).
            Vec coefficients = PETSC_NULLPTR;
            Vec natural = PETSC_NULLPTR; // borrowed workspace owned by IGA
            PetscCall(IGAGetNaturalVec(iga, &natural));
            PetscCall(IGAGlobalToNatural(iga, psi_save, natural));
            PetscInt localSize = 0, globalSize = 0;
            PetscCall(VecGetLocalSize(natural, &localSize));
            PetscCall(VecGetSize(natural, &globalSize));
            PetscCall(VecCreateMPI(PETSC_COMM_WORLD, localSize, globalSize, &coefficients));
            PetscCall(VecCopy(natural, coefficients));
            const std::string coefficientName = "coeff_" + std::to_string(i);
            PetscCall(PetscObjectSetName((PetscObject)coefficients, coefficientName.c_str()));
            PetscCall(VecView(coefficients, viewerH5));
            PetscCall(VecDestroy(&coefficients));
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

        // ── Eigenpair backward errors (computed in Solve) ────────────────
        PetscInt nres = PetscMin(nconv, (PetscInt)eigRes_.size());
        std::vector<double> errs(nres > 0 ? nres : 1, 0.0);
        for (PetscInt i = 0; i < nres; ++i) errs[i] = (double)eigRes_[i];

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
                write_real("Charge",TDSEZParser::Q);
                write_real("TargetEigenvalue", TDSEZParser::TargetEigenvalue);
                write_int ("NBoundStatesSave", (PetscInt)TDSEZParser::NBoundStatesSave);
                write_int ("NBoundStates", TDSEZParser::NBoundStates);
                write_attr("state_format", TDSEZParser::BoundStateFormat);
                write_attr("Mode", TDSEZParser::KHFrame ? "kh" : "td");
                if (TDSEZParser::KHFrame) {           // static Kramers-Henneberger problem
                    const TDSEZKHInfo &kh = TDSEZKHSummary();
                    write_real("KH_window_start", kh.t0);
                    write_real("KH_window_end",   kh.t1);
                    write_real("KH_omega",        kh.omega);
                    write_real("KH_alpha_x", kh.ax); write_real("KH_alpha_y", kh.ay); write_real("KH_alpha_z", kh.az);
                    write_int ("KH_samples", kh.samples);
                }
                // Backward error ||H v - E M v|| / ((||H|| + |E| ||M||) ||v||)
                // per kept state (infinity norms); written by rank 0.
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
    for (auto &v : eigVecs) VecDestroy(&v);
    eigVecs.clear();
    eigVals.clear();
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
                std::filesystem::remove(TDSEZPaths::TD(inputFile, "observables.h5"));
                std::filesystem::remove(TDSEZPaths::TD(inputFile, "wfs.h5"));
                std::filesystem::remove(TDSEZPaths::TD(inputFile, "ts.h5"));
                std::filesystem::remove(TDSEZPaths::Static(inputFile, "eigen.h5"));
            }
            MPI_Barrier(PETSC_COMM_WORLD);

            // Parse the input file (replaces the former PULSE wrapper, which
            // existed only to call PrmReader + initParsers). Validation below
            // relies on the parser state being populated.
            TDSEZParser::PrmReader(input);
            // OffsetX/Y/Z translate the computational box: fold them into the
            // domain bounds once, so the knots, the CAP layers, t-SURFF, the
            // validation and every report use the same shifted box.
            {
                static bool folded = false;
                if (!folded) {
                    TDSEZParser::LMinX += TDSEZParser::OffsetX; TDSEZParser::LMaxX += TDSEZParser::OffsetX;
                    TDSEZParser::LMinY += TDSEZParser::OffsetY; TDSEZParser::LMaxY += TDSEZParser::OffsetY;
                    TDSEZParser::LMinZ += TDSEZParser::OffsetZ; TDSEZParser::LMaxZ += TDSEZParser::OffsetZ;
                    folded = true;
                }
            }
            TDSEZParser::initParsers();

            // Validate the parsed input BEFORE touching PETSc/SLEPc. A bad input
            // (non-positive mass, inverted domain, etc.) would otherwise produce
            // silent garbage or a deep solver crash. Catch the exception and abort
            // with a clear message.
            try {
                TDSEZParser::ValidateOrThrow();
                TDSEZSetupKH();                       // Mode = kh: cycle-averaged potential
            } catch (const std::exception &e) {
                TDSEZInfo().fatal("input validation failed: %s", e.what());
                MPI_Abort(PETSC_COMM_WORLD, 3);
            }
            TDSEZInfo info;
            info.header("GRID AND BASIS");
            info.section("AXES");
            if (TDSEZParser::Q > 0.0)
                info.status("ⵣ", "NOTE: Charge = %+g (hole-like carrier): the Potential and the field term enter "
                    "H = K - q (V + r.E), i.e. with inverted sign. Charge = -1 is the electron.",
                    (double)TDSEZParser::Q);

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
                info.fatal("EnableGPU=1 but PETSc was built without CUDA support.");
                MPI_Abort(PETSC_COMM_WORLD, 4);
#endif
            }

            // 3. Create IGA
            IGACreate(PETSC_COMM_WORLD, &iga);
            IGASetDim(iga, TDSEZParser::Dimension);
            // -iga_* options (processor grid, matrix type, ...). The axes are
            // set explicitly below from the input; PetIGA's default axes
            // (degree 2, 16 elements on [0, 1]) are reset there.
            IGASetFromOptions(iga);   
            IGASetQuadrature(iga, 0, TDSEZParser::NQuadratures);
            IGASetDof(iga, 1);
            IGASetOrder(iga, TDSEZParser::SplineDegree);

            // Domain bounds (OffsetX/Y/Z already folded in, see the validation call).
            PetscReal LminX = TDSEZParser::LMinX, LmaxX = TDSEZParser::LMaxX;
            PetscReal LminY = TDSEZParser::LMinY, LmaxY = TDSEZParser::LMaxY;
            PetscReal LminZ = TDSEZParser::LMinZ, LmaxZ = TDSEZParser::LMaxZ;

            // Grading strength of symexp/symtanu/symtann (alpha or beta in the
            // knot maps) and of symlogtan. KnotAlpha > 0 overrides the defaults.
            const PetscReal alphaDefault = (TDSEZParser::KnotAlpha > 0.0) ? TDSEZParser::KnotAlpha : 0.9;
            const PetscReal alphaLogTan  = (TDSEZParser::KnotAlpha > 0.0) ? TDSEZParser::KnotAlpha : 0.5;
            PetscReal alphax = alphaDefault, alphay = alphaDefault, alphaz = alphaDefault;
            IGAAxis axisx, axisy, axisz;

            // Largest hydrogenic element width per axis: the input value, or
            // (0, default) derived from the field for strong-field propagation.
            auto hydHMax = [&](int d) -> PetscReal {
                const PetscReal v = TDSEZParser::HydrogenicHMax[d];
                if (v > 0) return v;
                if (v < 0) return 0.0;
                return TDSEZHydrogenicAutoHMax(d);
            };

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
                // a fresh axis of the input degree (PetIGA cannot change the
                // degree of an axis that already has knots)
                PetscCallAbort(PETSC_COMM_WORLD, IGAAxisReset(axisx));
                PetscCallAbort(PETSC_COMM_WORLD, IGAAxisSetDegree(axisx, TDSEZParser::SplineDegree));
                // Robustness guard: too few interior knots => basis cannot
                // represent the ground state; warn (not fatal) so the user
                // is not silently handed garbage at N<=~2p.
                {
                    const PetscInt nint = TDSEZParser::Nelements - 1;
                    const PetscInt min_n = 2 * TDSEZParser::SplineDegree + 1;
                    if (nint < min_n && rank == 0)
                        info.warn("x axis has only %" PetscInt_FMT
                            " interior knots (< %" PetscInt_FMT " recommended for p=%"
                            PetscInt_FMT "); result may be unreliable.",
                            nint, min_n, TDSEZParser::SplineDegree);
                }
                if (seq != "uniform")
                {
                if (rank == 0)
                {
                    if (seq == "symexp")
                        knots_x = TDSEZExpSymKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, alphax, TDSEZParser::SplineDegree);
                    else if (seq == "interface" || seq == "interfaces")
                        knots_x = TDSEZInterfaceKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, TDSEZParser::SplineDegree, TDSEZParser::Potential, TDSEZParser::Mass);
                    else if (seq == "symtanu")
                        knots_x = TDSEZTanUSymKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, alphax, TDSEZParser::SplineDegree);
                    else if (seq == "symtann")
                        knots_x = TDSEZTanNSymKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, alphax, TDSEZParser::SplineDegree);
                    else if (seq == "symlogtan")
                        knots_x = TDSEZLogTanKnots(LminX, LmaxX, TDSEZParser::Nelements - 1, alphaLogTan, TDSEZParser::SplineDegree);
                    else if (seq == "hydrogenic")
                        knots_x = TDSEZHydrogenicKnots(LmaxX,
                            TDSEZParser::HydrogenicNLin[0] > 0 ? TDSEZParser::HydrogenicNLin[0] : TDSEZParser::Nelements - 1,
                            TDSEZParser::HydrogenicR1[0] > 0.0 ? TDSEZParser::HydrogenicR1[0] : 0.1,
                            TDSEZParser::HydrogenicNExp[0] > 0 ? TDSEZParser::HydrogenicNExp[0] : 10,
                            TDSEZParser::SplineDegree, hydHMax(0), 'x');
                    else
                    {
                        info.warn("unknown KnotSequence '%s' on the x axis; falling back to uniform.", seq.c_str());
                        PetscCallAbort(PETSC_COMM_WORLD, IGAAxisInitUniform(axisx, TDSEZParser::Nelements, LminX, LmaxX, TDSEZParser::SplineDegree - 1));
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
                        PetscCallAbort(PETSC_COMM_WORLD, IGAAxisSetKnots(axisx, knots_x.size() - 1, knots_x.data()));
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
                    PetscCallAbort(PETSC_COMM_WORLD, IGAAxisInitUniform(axisx, TDSEZParser::Nelements, LminX, LmaxX, TDSEZParser::SplineDegree - 1));
                }
            }


            // ── Y-AXIS KNOTS ─────────────────────────────────────────────────────
            if (TDSEZParser::Dimension >= 2)
            {
                const std::string seq = TDSEZParser::KnotSeq[1];
                IGASetQuadrature(iga, 1, TDSEZParser::NQuadratures);
                IGAGetAxis(iga, 1, &axisy);
                // a fresh axis of the input degree (PetIGA cannot change the
                // degree of an axis that already has knots)
                PetscCallAbort(PETSC_COMM_WORLD, IGAAxisReset(axisy));
                PetscCallAbort(PETSC_COMM_WORLD, IGAAxisSetDegree(axisy, TDSEZParser::SplineDegree));
                if (seq != "uniform")
                {
                    if (rank == 0)
                    {
                        if (seq == "symexp")
                            knots_y = TDSEZExpSymKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, alphay, TDSEZParser::SplineDegree);
                        else if (seq == "interface" || seq == "interfaces")
                            knots_y = TDSEZInterfaceKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, TDSEZParser::SplineDegree, TDSEZParser::Potential, TDSEZParser::Mass);
                        else if (seq == "symtanu")
                            knots_y = TDSEZTanUSymKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, alphay, TDSEZParser::SplineDegree);
                        else if (seq == "symtann")
                            knots_y = TDSEZTanNSymKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, alphay, TDSEZParser::SplineDegree);
                        else if (seq == "symlogtan")
                            knots_y = TDSEZLogTanKnots(LminY, LmaxY, TDSEZParser::Nelements - 1, alphaLogTan, TDSEZParser::SplineDegree);
                        else if (seq == "hydrogenic")
                            knots_y = TDSEZHydrogenicKnots(LmaxY,
                                TDSEZParser::HydrogenicNLin[1] > 0 ? TDSEZParser::HydrogenicNLin[1] : TDSEZParser::Nelements - 1,
                                TDSEZParser::HydrogenicR1[1] > 0.0 ? TDSEZParser::HydrogenicR1[1] : 0.1,
                                TDSEZParser::HydrogenicNExp[1] > 0 ? TDSEZParser::HydrogenicNExp[1] : 10,
                                TDSEZParser::SplineDegree, hydHMax(1), 'y');
                        else
                        {
                            info.warn("unknown KnotSequence '%s' on the y axis; falling back to uniform.", seq.c_str());
                            PetscCallAbort(PETSC_COMM_WORLD, IGAAxisInitUniform(axisy, TDSEZParser::Nelements, LminY, LmaxY, TDSEZParser::SplineDegree - 1));
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
                            PetscCallAbort(PETSC_COMM_WORLD, IGAAxisSetKnots(axisy, knots_y.size() - 1, knots_y.data()));
                    }
                }
                else
                {
                    // Per-axis uniform interval (uses LminY/LmaxY, not the X span).
                    PetscCallAbort(PETSC_COMM_WORLD, IGAAxisInitUniform(axisy, TDSEZParser::Nelements, LminY, LmaxY, TDSEZParser::SplineDegree - 1));
                }
            }


            // ── Z-AXIS KNOTS ─────────────────────────────────────────────────────
            if (TDSEZParser::Dimension == 3)
            {
                const std::string seq = TDSEZParser::KnotSeq[2];
                IGASetQuadrature(iga, 2, TDSEZParser::NQuadratures);
                IGAGetAxis(iga, 2, &axisz);
                // a fresh axis of the input degree (PetIGA cannot change the
                // degree of an axis that already has knots)
                PetscCallAbort(PETSC_COMM_WORLD, IGAAxisReset(axisz));
                PetscCallAbort(PETSC_COMM_WORLD, IGAAxisSetDegree(axisz, TDSEZParser::SplineDegree));
                if (seq != "uniform")
                {
                    if (rank == 0)
                    {
                        if (seq == "symexp")
                            knots_z = TDSEZExpSymKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, alphaz, TDSEZParser::SplineDegree);
                        else if (seq == "interface" || seq == "interfaces")
                            knots_z = TDSEZInterfaceKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, TDSEZParser::SplineDegree, TDSEZParser::Potential, TDSEZParser::Mass);
                        else if (seq == "symtanu")
                            knots_z = TDSEZTanUSymKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, alphaz, TDSEZParser::SplineDegree);
                        else if (seq == "symtann")
                            knots_z = TDSEZTanNSymKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, alphaz, TDSEZParser::SplineDegree);
                        else if (seq == "symlogtan")
                            knots_z = TDSEZLogTanKnots(LminZ, LmaxZ, TDSEZParser::Nelements - 1, alphaLogTan, TDSEZParser::SplineDegree);
                        else if (seq == "hydrogenic")
                            knots_z = TDSEZHydrogenicKnots(LmaxZ,
                                TDSEZParser::HydrogenicNLin[2] > 0 ? TDSEZParser::HydrogenicNLin[2] : TDSEZParser::Nelements - 1,
                                TDSEZParser::HydrogenicR1[2] > 0.0 ? TDSEZParser::HydrogenicR1[2] : 0.1,
                                TDSEZParser::HydrogenicNExp[2] > 0 ? TDSEZParser::HydrogenicNExp[2] : 10,
                                TDSEZParser::SplineDegree, hydHMax(2), 'z');
                        else
                        {
                            info.warn("unknown KnotSequence '%s' on the z axis; falling back to uniform.", seq.c_str());
                            PetscCallAbort(PETSC_COMM_WORLD, IGAAxisInitUniform(axisz, TDSEZParser::Nelements, LminZ, LmaxZ, TDSEZParser::SplineDegree - 1));
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
                            PetscCallAbort(PETSC_COMM_WORLD, IGAAxisSetKnots(axisz, knots_z.size() - 1, knots_z.data()));
                    }
                }
                else
                {
                    // Per-axis uniform interval (uses LminZ/LmaxZ, not the X span).
                    PetscCallAbort(PETSC_COMM_WORLD, IGAAxisInitUniform(axisz, TDSEZParser::Nelements, LminZ, LmaxZ, TDSEZParser::SplineDegree - 1));
                }
            }

            } catch (const std::exception &e) {
                info.fatal("knot-sequence setup failed: %s", e.what());
                MPI_Abort(PETSC_COMM_WORLD, 3);
            }

            // Basis and quadrature rule. The element bases (Bernstein, Lagrange,
            // spectral) are C0: every interior knot is raised to multiplicity p.
            {
                const std::string &b = TDSEZParser::Basis, &q = TDSEZParser::QuadratureRule;
                const IGABasisType bt = b == "bernstein" ? IGA_BASIS_BERNSTEIN
                                      : b == "lagrange"  ? IGA_BASIS_LAGRANGE
                                      : b == "spectral"  ? IGA_BASIS_SPECTRAL : IGA_BASIS_BSPLINE;
                const IGARuleType rt = q == "lobatto" ? IGA_RULE_LOBATTO
                                     : q == "reduced" ? IGA_RULE_REDUCED : IGA_RULE_LEGENDRE;
                const PetscInt p = TDSEZParser::SplineDegree;
                for (PetscInt d = 0; d < TDSEZParser::Dimension; ++d) {
                    if (bt != IGA_BASIS_BSPLINE) {
                        IGAAxis ax;
                        PetscInt m;
                        PetscReal *U;
                        PetscCallAbort(PETSC_COMM_WORLD, IGAGetAxis(iga, d, &ax));
                        PetscCallAbort(PETSC_COMM_WORLD, IGAAxisGetKnots(ax, &m, &U));
                        std::vector<PetscReal> C0(U, U + p + 1);                  // first boundary knots
                        for (PetscInt k = p + 1; k < m - p; ++k)
                            if (U[k] != C0.back()) C0.insert(C0.end(), p, U[k]); // each interior knot p times
                        C0.insert(C0.end(), U + m - p, U + m + 1);               // last boundary knots
                        PetscCallAbort(PETSC_COMM_WORLD, IGAAxisSetKnots(ax, (PetscInt)C0.size() - 1, C0.data()));
                    }
                    PetscCallAbort(PETSC_COMM_WORLD, IGASetBasisType(iga, d, bt));
                    PetscCallAbort(PETSC_COMM_WORLD, IGASetRuleType(iga, d, rt));
                }
            }

            // Boundary Conditions
            for (PetscInt d = 0; d < TDSEZParser::Dimension; d++) {
                IGASetBoundaryValue(iga, d, 0, 0, 0.0);
                IGASetBoundaryValue(iga, d, 1, 0, 0.0);
            }

            // Final Setup
            IGASetUp(iga);
            {
                const char *rule = TDSEZParser::QuadratureRule.empty() ? "legendre" : TDSEZParser::QuadratureRule.c_str();
                info.kv("Basis", "%s, degree %" PetscInt_FMT, TDSEZParser::Basis.empty() ? "bspline" : TDSEZParser::Basis.c_str(),
                        TDSEZParser::SplineDegree);
                info.kv("Quadrature", "%s, %" PetscInt_FMT " points per element and axis", rule, TDSEZParser::NQuadratures);
                PetscInt ndof = 1;
                for (PetscInt d = 0; d < TDSEZParser::Dimension; ++d) {
                    IGAAxis ax;
                    PetscInt m, nel = 0, nb = 0;
                    PetscReal *U, hmin = PETSC_MAX_REAL, hmaxAxis = 0;
                    PetscCallAbort(PETSC_COMM_WORLD, IGAGetAxis(iga, d, &ax));
                    PetscCallAbort(PETSC_COMM_WORLD, IGAAxisGetKnots(ax, &m, &U));
                    PetscCallAbort(PETSC_COMM_WORLD, IGAAxisGetSizes(ax, &nel, &nb));
                    for (PetscInt k = 0; k < m; ++k)
                        if (U[k + 1] > U[k]) { hmin = PetscMin(hmin, U[k + 1] - U[k]); hmaxAxis = PetscMax(hmaxAxis, U[k + 1] - U[k]); }
                    ndof *= nb;
                    char lab[16];
                    snprintf(lab, sizeof(lab), "%c axis", "xyz"[d]);
                    info.kv(lab, "%s, %" PetscInt_FMT " elements, h %.3g-%.3g",
                            TDSEZParser::KnotSeq[d].c_str(), nel, (double)hmin, (double)hmaxAxis);
                }
                info.kv("Degrees of freedom", "%" PetscInt_FMT, ndof);
            }
            TDSEZCheckFieldResolution(iga);
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
