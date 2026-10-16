// ============================================================================
//  sumfact.cpp  —  sum-factorised element assembly for tensor-product B-splines
// ----------------------------------------------------------------------------
//  The element basis is N_a(r) = N_{a1}(x) N_{a2}(y) N_{a3}(z) and the Gauss
//  rule is a tensor product, so every element integral of the form
//
//      E_ab = sum_q c(q) D^alpha N_a(q) D^beta N_b(q)
//
//  can be contracted one axis at a time (sum factorisation):
//
//      T1[a1 b1][q2 q3] = sum_q1 N1^al[a1](q1) N1^be[b1](q1) c(q1,q2,q3)
//      T2[a1 b1 a2 b2][q3] = sum_q2 N2^al[a2](q2) N2^be[b2](q2) T1[a1 b1][q2 q3]
//      E[a b]          = sum_q3 N3^al[a3](q3) N3^be[b3](q3) T2[a1 b1 a2 b2][q3]
//
//  The work per element and term drops from Q^3 (p+1)^6 to about Q (p+1)^6
//  (3D) and from Q^2 (p+1)^4 to Q (p+1)^4 (2D), with the same quadrature
//  points and the same pointwise coefficients: the element matrices are
//  identical up to floating-point rounding. Each axis contraction is a
//  dense matrix product evaluated with the BLAS PETSc was configured with.
//
//  Used for TDSEZFormHam (H, M) and TDSEZFormPhyAxes (2D/3D time-dependent
//  operators), including the mass-gradient corrections of the force matrices
//  (operator-form acceleration with a position-dependent mass).
//
//  Options: -tdsez_sumfact 0        disable (per-point kernels everywhere)
//           -tdsez_sumfact_check    compute every element both ways and print
//                                   the largest relative difference
// ============================================================================

#include "tdsez_internal.hpp"
#include "tdsez_forcecoeffs.hpp"
#include "tdsez_quadobs.hpp"
#include <petscblaslapack.h>
#include <algorithm>
#include <string>
#include <vector>

namespace {

/// 1D tabulation of one element along one axis.
struct Axis1D {
    PetscInt n = 0, Q = 0;          // local basis functions, quadrature points
    std::vector<PetscReal> N[2];    // derivative order 0/1: [a*Q + q]
    std::vector<PetscReal> W;       // weight * detJac: [q]
    std::vector<PetscReal> X;       // physical coordinate: [q]
};

void LoadAxis(IGA iga, IGAElement e, PetscInt d, Axis1D &A)
{
    IGABasis B = iga->basis[d];
    const PetscInt el = e->ID[d];
    A.n = B->nen; A.Q = B->nqp;
    for (int k = 0; k < 2; ++k) A.N[k].assign((size_t)A.n * A.Q, 0.0);
    A.W.assign(A.Q, 0.0); A.X.assign(A.Q, 0.0);
    for (PetscInt q = 0; q < A.Q; ++q) {
        A.W[q] = B->weight[el * A.Q + q] * B->detJac[el];
        // Points of weight 0 (the unused points of PetIGA's reduced rule on
        // interior elements) sit at PETSC_MAX_REAL with uncomputed basis
        // values: keep them out (coordinate at the element start, values 0),
        // or V(x) * 0 = inf * 0 = NaN reaches the matrices.
        const bool used = A.W[q] != 0.0;
        A.X[q] = used ? B->point[el * A.Q + q] : B->point[el * A.Q];
        for (PetscInt a = 0; a < A.n; ++a)
            for (int k = 0; k < 2; ++k)
                A.N[k][(size_t)a * A.Q + q] = used ? B->value[(((size_t)el * A.Q + q) * A.n + a) * 5 + k] : 0.0;
    }
}

/// P[ab*Q + q] = N^al[a](q) N^be[b](q), ab = a*n + b.
void Products(const Axis1D &A, int al, int be, std::vector<PetscReal> &P)
{
    const PetscInt n = A.n, Q = A.Q;
    P.resize((size_t)n * n * Q);
    for (PetscInt a = 0; a < n; ++a)
        for (PetscInt b = 0; b < n; ++b)
            for (PetscInt q = 0; q < Q; ++q)
                P[((size_t)(a * n + b)) * Q + q] = A.N[al][(size_t)a * Q + q] * A.N[be][(size_t)b * Q + q];
}

/// Row-major C (M x N) = A (M x K) * op(B) + beta C, op(B) = B^T with B stored (N x K)
/// row-major. Calls column-major BLAS as C^T = B * A^T through PETSc's
/// mangling macro, so the symbol name, BLAS integer width (32/64-bit) and
/// real precision follow whatever BLAS PETSc was configured with.
void GemmABt(PetscInt M, PetscInt N, PetscInt K, const PetscReal *A, const PetscReal *B, PetscReal *C,
             PetscReal beta = 0.0)
{
    const PetscBLASInt m = (PetscBLASInt)M, n = (PetscBLASInt)N, k = (PetscBLASInt)K;
    const PetscReal one = 1.0;
    BLASREALgemm_("T", "N", &n, &m, &k, &one, B, &k, A, &k, &beta, C, &n);
}

/// Scratch buffers of one thread. Fetched once per element: thread_local
/// lookups inside the loops are expensive where TLS is emulated (macOS gcc).
struct Scratch {
    std::vector<PetscReal> P1, P2, P3, T1, T2, V, E;
    std::vector<size_t> offA, offB;
};

/// Real element matrix  E(a,b) (+)= sum_q c(q) D^al N_a D^be N_b  for a 2D or
/// 3D element, x-fastest point numbering q = q1 + Q1 (q2 + Q2 q3). Each axis
/// contraction is one matrix product (BLAS gemm); the last one overwrites
/// (accumulate = false) or adds to E, so several terms of the same operator
/// share one buffer. E is stored in contraction order, entry (a,b) at
/// offA[a] + offB[b] (see Offsets).
void Contract(const Axis1D *A, PetscInt dim, const int al[3], const int be[3],
              const PetscReal *c, bool accumulate, Scratch &S)
{
    const PetscInt n1 = A[0].n, Q1 = A[0].Q, n2 = A[1].n, Q2 = A[1].Q;
    Products(A[0], al[0], be[0], S.P1);
    Products(A[1], al[1], be[1], S.P2);
    const PetscInt m1 = n1 * n1, m2 = n2 * n2;
    const PetscReal beta = accumulate ? 1.0 : 0.0;

    if (dim == 2) {
        // T1[ab1][q2] = sum_q1 P1[ab1][q1] c[q2][q1]
        S.T1.resize((size_t)m1 * Q2);
        GemmABt(m1, Q2, Q1, S.P1.data(), c, S.T1.data());
        // E[ab1][ab2] (+)= sum_q2 T1[ab1][q2] P2[ab2][q2]
        S.E.resize((size_t)m1 * m2);
        GemmABt(m1, m2, Q2, S.T1.data(), S.P2.data(), S.E.data(), beta);
        return;
    }

    const PetscInt n3 = A[2].n, Q3 = A[2].Q;
    Products(A[2], al[2], be[2], S.P3);
    const PetscInt m3 = n3 * n3;

    // T1[ab1][q3][q2] = sum_q1 P1[ab1][q1] c[q3][q2][q1]
    S.T1.resize((size_t)m1 * Q2 * Q3);
    GemmABt(m1, Q2 * Q3, Q1, S.P1.data(), c, S.T1.data());
    // T2[ab1][q3][ab2] = sum_q2 T1[ab1][q3][q2] P2[ab2][q2]
    S.T2.resize((size_t)m1 * Q3 * m2);
    GemmABt(m1 * Q3, m2, Q2, S.T1.data(), S.P2.data(), S.T2.data());
    // V[ab1][ab2][q3] = T2[ab1][q3][ab2]
    S.V.resize((size_t)m1 * m2 * Q3);
    for (PetscInt ab1 = 0; ab1 < m1; ++ab1) {
        const PetscReal *t = S.T2.data() + (size_t)ab1 * Q3 * m2;
        PetscReal *v = S.V.data() + (size_t)ab1 * m2 * Q3;
        for (PetscInt q3 = 0; q3 < Q3; ++q3)
            for (PetscInt ab2 = 0; ab2 < m2; ++ab2)
                v[(size_t)ab2 * Q3 + q3] = t[(size_t)q3 * m2 + ab2];
    }
    // E[ab1][ab2][ab3] (+)= sum_q3 V[ab1][ab2][q3] P3[ab3][q3]
    S.E.resize((size_t)m1 * m2 * m3);
    GemmABt(m1 * m2, m3, Q3, S.V.data(), S.P3.data(), S.E.data(), beta);
}

/// Position of element-matrix entry (a,b) in Contract's E: offA[a] + offB[b],
/// local numbering a = a1 + n1 (a2 + n2 a3).
void Offsets(const Axis1D *A, PetscInt dim, Scratch &S)
{
    const size_t n1 = A[0].n, n2 = A[1].n, n3 = (dim == 3) ? A[2].n : 1;
    const size_t m2 = n2 * n2, m3 = n3 * n3;
    const size_t nen = n1 * n2 * n3;
    S.offA.resize(nen); S.offB.resize(nen);
    for (size_t a3 = 0; a3 < n3; ++a3)
    for (size_t a2 = 0; a2 < n2; ++a2)
    for (size_t a1 = 0; a1 < n1; ++a1) {
        const size_t a = a1 + n1 * (a2 + n2 * a3);
        S.offA[a] = a1 * n1 * m2 * m3 + a2 * n2 * m3 + a3 * n3;
        S.offB[a] = a1 * m2 * m3 + a2 * m3 + a3;
    }
}

thread_local Scratch scratch;

enum class Sym { Symmetric, AntiSymmetric, General };

/// work += factor * sym(E): symmetric operators get 0.5 (E + E^T), exactly
/// symmetric (the two triangles are contracted in different orders and differ
/// by rounding); the velocity matrix gets E - E^T, exactly antisymmetric.
void Store(const Scratch &S, PetscInt nen, PetscScalar factor, Sym sym, PetscScalar *work)
{
    const PetscReal *E = S.E.data();
    const size_t *oA = S.offA.data(), *oB = S.offB.data();
    const PetscReal h = (sym == Sym::Symmetric) ? 0.5 : 1.0;
    const PetscReal sg = (sym == Sym::Symmetric) ? 1.0 : -1.0;
    for (PetscInt a = 0; a < nen; ++a) {
        PetscScalar *w = work + (size_t)a * nen;
        const size_t ia = oA[a], ja = oB[a];
        if (sym == Sym::General)
            for (PetscInt b = 0; b < nen; ++b) w[b] += factor * E[ia + oB[b]];
        else
            for (PetscInt b = 0; b < nen; ++b) w[b] += factor * (h * (E[ia + oB[b]] + sg * E[oA[b] + ja]));
    }
}

/// Element force matrix from ForceCoeffs (C.BB holds the classical term too).
void ForceElement(const Axis1D *A, PetscInt dim, PetscInt nen, const ForceCoeffs &C, Scratch &S, PetscScalar *work)
{
    const int V0[3] = {0, 0, 0};
    Contract(A, dim, V0, V0, C.BB.data(), false, S);
    for (int d = 0; d < dim; ++d) {
        int e1[3] = {0, 0, 0}; e1[d] = 1;
        Contract(A, dim, V0, e1, C.Bd[d].data(), true, S);
        Contract(A, dim, e1, e1, C.dd[d].data(), true, S);
    }
    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int k = 0; k < 3; ++k) {
        if (pairs[k][1] >= dim) continue;
        int e2[3] = {0, 0, 0}; e2[pairs[k][0]] = 1; e2[pairs[k][1]] = 1;
        Contract(A, dim, V0, e2, C.Bp[k].data(), true, S);
    }
    Store(S, nen, 1.0, Sym::General, work);
}

PetscBool SumFactEnabled()
{
    static PetscBool init = PETSC_FALSE, on = PETSC_TRUE;
    if (!init) {
        PetscOptionsGetBool(NULL, NULL, "-tdsez_sumfact", &on, NULL);
        init = PETSC_TRUE;
    }
    return on;
}

PetscBool SumFactCheck()
{
    static PetscBool init = PETSC_FALSE, on = PETSC_FALSE;
    if (!init) {
        PetscOptionsHasName(NULL, NULL, "-tdsez_sumfact_check", &on);
        init = PETSC_TRUE;
    }
    return on;
}

/// Copy the force coefficients of element e into the store (see ForceCoeffStore).
void KeepForceCoeffs(IGA iga, IGAElement e, PetscInt dim, size_t nq, const ForceCoeffs Fc[3])
{
    ForceCoeffStore &st = TDSEZForceCoeffStore();
    if (st.nq == 0) {
        size_t ne = 1;
        for (PetscInt d = 0; d < dim; ++d) { st.start[d] = iga->elem_start[d]; st.width[d] = iga->elem_width[d]; ne *= (size_t)st.width[d]; }
        st.nq = (PetscInt)nq;
        st.done.assign(ne, 0);
        for (int d = 0; d < 3; ++d) {
            st.axis[d] = TDSEZPhyAxesOn[d] != 0;
            if (st.axis[d]) st.F[d].Resize(ne * nq);
        }
    }
    if ((PetscInt)nq != st.nq) return;
    size_t el = 0;
    for (PetscInt d = dim - 1; d >= 0; --d) el = el * (size_t)st.width[d] + (size_t)(e->ID[d] - st.start[d]);
    for (int d = 0; d < 3; ++d) {
        if (!st.axis[d]) continue;
        auto put = [&](const std::vector<PetscReal> &src, std::vector<PetscReal> &dst) { std::copy(src.begin(), src.begin() + nq, dst.begin() + el * nq); };
        put(Fc[d].BB, st.F[d].BB);
        for (int k = 0; k < 3; ++k) { put(Fc[d].Bd[k], st.F[d].Bd[k]); put(Fc[d].Bp[k], st.F[d].Bp[k]); put(Fc[d].dd[k], st.F[d].dd[k]); }
    }
    if (!st.done[el]) { st.done[el] = 1; ++st.filled; }
}

/// Sum-factorised TDSEZFormPhyAxes: [K, V, Md] + [D_d, Vel_d, dV_d] per active
/// axis. dV_d is the classical term -B_a (dV/dx_d)(1/m) B_b, which is what the
/// per-axis force kernels compute when their mass-gradient corrections vanish:
/// for a constant mass (finite differences of a constant), or when the discrete
/// acceleration is used (the force matrices are then not read, and the kernels
/// skip the corrections). K, Md and Vel take 1/m at every quadrature point, so
/// a position-dependent mass is handled exactly.
bool SumFactAxes(IGA iga, IGAElement e, PetscInt nmat, PetscScalar *work[], unsigned need)
{
    const PetscInt dim = e->dim;
    if (dim < 2 || dim > 3 || e->dof != 1 || e->geometry || e->rational) return false;
    // The force kernels add their mass-gradient corrections when the
    // operator-form acceleration reads them and the mass varies.
    const bool corr = !TDSEZDiscreteAccelerationActive() && MassVaries();
    PetscInt nexp = 3;
    for (int d = 0; d < 3; ++d) if (TDSEZPhyAxesOn[d]) { if (d >= dim) return false; nexp += 3; }
    if (nexp != nmat) return false;

    thread_local Axis1D A[3];
    for (PetscInt d = 0; d < dim; ++d) LoadAxis(iga, e, d, A[d]);
    PetscInt nen = 1;
    for (PetscInt d = 0; d < dim; ++d) nen *= A[d].n;
    if (nen != e->nen) return false;

    const PetscInt Q1 = A[0].Q, Q2 = A[1].Q, Q3 = (dim == 3) ? A[2].Q : 1;
    const size_t nq = (size_t)Q1 * Q2 * Q3;
    thread_local std::vector<PetscReal> cK, cV, cM, cVel, cD[3], cG[3];
    thread_local ForceCoeffs Fc[3];
    cK.resize(nq); cV.resize(nq); cM.resize(nq); cVel.resize(nq);
    for (int d = 0; d < 3; ++d) if (TDSEZPhyAxesOn[d]) { cD[d].resize(nq); cG[d].resize(nq); if (corr) Fc[d].Resize(nq); }

    for (PetscInt q3 = 0; q3 < Q3; ++q3)
    for (PetscInt q2 = 0; q2 < Q2; ++q2)
    for (PetscInt q1 = 0; q1 < Q1; ++q1) {
        const size_t q = (size_t)q1 + (size_t)Q1 * ((size_t)q2 + (size_t)Q2 * q3);
        const PetscReal r[3] = {A[0].X[q1], A[1].X[q2], (dim == 3) ? A[2].X[q3] : 0.0};
        const PetscReal w = A[0].W[q1] * A[1].W[q2] * ((dim == 3) ? A[2].W[q3] : 1.0);
        const PetscReal invMass = (PetscReal)(1.0 / TDSEZParser::MassDist(r[0], r[1], r[2]));
        cK[q]   = 0.5 * invMass * w;
        cV[q]   = TDSEZParser::V(r[0], r[1], r[2]) * w;
        cM[q]   = invMass * w;
        cVel[q] = -0.5 * invMass * w;
        for (int d = 0; d < 3; ++d) {
            if (!TDSEZPhyAxesOn[d]) continue;
            const PetscReal g = (d == 0) ? TDSEZParser::dVx(r[0], r[1], r[2])
                              : (d == 1) ? TDSEZParser::dVy(r[0], r[1], r[2])
                                         : TDSEZParser::dVz(r[0], r[1], r[2]);
            cD[d][q] = r[d] * w;
            cG[d][q] = -g * invMass * w;
        }
        if (corr) {
            const std::vector<PetscReal> pt = dim == 3 ? std::vector<PetscReal>{r[0], r[1], r[2]}
                                                       : std::vector<PetscReal>{r[0], r[1]};
            const PhysicsDerivativesFull &ops = TDSEZInvMassDerivatives(pt);
            for (int d = 0; d < 3; ++d) {
                if (!TDSEZPhyAxesOn[d]) continue;
                Fc[d].BB[q] = cG[d][q];
                AddForceCorrections(d, dim, invMass, ops, w, q, Fc[d]);
            }
        }
    }

    Scratch &S = scratch;
    Offsets(A, dim, S);
    const int V0[3] = {0, 0, 0};
    auto want = [need](PetscInt k) { return ((need >> k) & 1u) != 0; };
    if (want(0)) {
        for (PetscInt d = 0; d < dim; ++d) {
            int ad[3] = {0, 0, 0};
            ad[d] = 1;
            Contract(A, dim, ad, ad, cK.data(), d > 0, S);
        }
        Store(S, nen, 1.0, Sym::Symmetric, work[0]);
    }
    if (want(1)) {
        Contract(A, dim, V0, V0, cV.data(), false, S);
        Store(S, nen, 1.0, Sym::Symmetric, work[1]);
    }
    if (want(2)) {
        Contract(A, dim, V0, V0, cM.data(), false, S);
        Store(S, nen, 1.0, Sym::Symmetric, work[2]);
    }
    if (corr && TDSEZQuadObsEnabled()) KeepForceCoeffs(iga, e, dim, nq, Fc);
    PetscInt k = 3;
    for (int d = 0; d < 3; ++d) {
        if (!TDSEZPhyAxesOn[d]) continue;
        int ad[3] = {0, 0, 0};
        ad[d] = 1;
        if (want(k)) {
            Contract(A, dim, V0, V0, cD[d].data(), false, S);
            Store(S, nen, 1.0, Sym::Symmetric, work[k]);
        }
        if (want(k + 1)) {
            Contract(A, dim, V0, ad, cVel.data(), false, S);
            Store(S, nen, PETSC_i, Sym::AntiSymmetric, work[k + 1]);
        }
        if (want(k + 2)) {
            if (corr) ForceElement(A, dim, nen, Fc[d], S, work[k + 2]);
            else {
                Contract(A, dim, V0, V0, cG[d].data(), false, S);
                Store(S, nen, 1.0, Sym::Symmetric, work[k + 2]);
            }
        }
        k += 3;
    }
    return true;
}

/// Sum-factorised TDSEZFormMatFreeAxes: [D_d] for the dipole axes, then
/// [Vel_d] for the velocity axes (same contractions as SumFactAxes).
bool SumFactMatFree(IGA iga, IGAElement e, PetscInt nmat, PetscScalar *work[], unsigned need)
{
    const PetscInt dim = e->dim;
    PetscInt nexp = 0;
    for (int d = 0; d < 3; ++d) {
        if ((TDSEZMatFreeDipOn[d] || TDSEZMatFreeVelOn[d]) && d >= dim) return false;
        nexp += (TDSEZMatFreeDipOn[d] ? 1 : 0) + (TDSEZMatFreeVelOn[d] ? 1 : 0);
    }
    if (nexp != nmat) return false;
    thread_local Axis1D A[3];
    for (PetscInt d = 0; d < dim; ++d) LoadAxis(iga, e, d, A[d]);
    PetscInt nen = 1;
    for (PetscInt d = 0; d < dim; ++d) nen *= A[d].n;
    if (nen != e->nen) return false;

    const PetscInt Q1 = A[0].Q, Q2 = A[1].Q, Q3 = (dim == 3) ? A[2].Q : 1;
    const size_t nq = (size_t)Q1 * Q2 * Q3;
    thread_local std::vector<PetscReal> cVel, cD[3];
    cVel.resize(nq);
    for (int d = 0; d < 3; ++d) if (TDSEZMatFreeDipOn[d]) cD[d].resize(nq);
    for (PetscInt q3 = 0; q3 < Q3; ++q3)
    for (PetscInt q2 = 0; q2 < Q2; ++q2)
    for (PetscInt q1 = 0; q1 < Q1; ++q1) {
        const size_t q = (size_t)q1 + (size_t)Q1 * ((size_t)q2 + (size_t)Q2 * q3);
        const PetscReal r[3] = {A[0].X[q1], A[1].X[q2], (dim == 3) ? A[2].X[q3] : 0.0};
        const PetscReal w = A[0].W[q1] * A[1].W[q2] * ((dim == 3) ? A[2].W[q3] : 1.0);
        cVel[q] = -0.5 * w / (PetscReal)TDSEZParser::MassDist(r[0], r[1], r[2]);
        for (int d = 0; d < 3; ++d) if (TDSEZMatFreeDipOn[d]) cD[d][q] = r[d] * w;
    }
    Scratch &S = scratch;
    Offsets(A, dim, S);
    const int V0[3] = {0, 0, 0};
    auto want = [need](PetscInt k) { return ((need >> k) & 1u) != 0; };
    PetscInt k = 0;
    for (int d = 0; d < 3; ++d) {
        if (!TDSEZMatFreeDipOn[d]) continue;
        if (want(k)) {
            Contract(A, dim, V0, V0, cD[d].data(), false, S);
            Store(S, nen, 1.0, Sym::Symmetric, work[k]);
        }
        ++k;
    }
    for (int d = 0; d < 3; ++d) {
        if (!TDSEZMatFreeVelOn[d]) continue;
        if (want(k)) {
            int ad[3] = {0, 0, 0};
            ad[d] = 1;
            Contract(A, dim, V0, ad, cVel.data(), false, S);
            Store(S, nen, PETSC_i, Sym::AntiSymmetric, work[k]);
        }
        ++k;
    }
    return true;
}

/// Structured field terms: work[k] = <B_a| w_{i0+k} |B_b>, i0 = TDSEZFieldTermIndex.
bool SumFactFieldTerms(IGA iga, IGAElement e, PetscInt nmat, PetscScalar *work[], unsigned need)
{
    const PetscInt dim = e->dim;
    thread_local Axis1D A[3];
    for (PetscInt d = 0; d < dim; ++d) LoadAxis(iga, e, d, A[d]);
    PetscInt nen = 1;
    for (PetscInt d = 0; d < dim; ++d) nen *= A[d].n;
    if (nen != e->nen) return false;
    const PetscInt Q1 = A[0].Q, Q2 = A[1].Q, Q3 = (dim == 3) ? A[2].Q : 1;
    const size_t nq = (size_t)Q1 * Q2 * Q3;
    thread_local std::vector<PetscReal> c;
    c.resize(nq);
    Scratch &S = scratch;
    Offsets(A, dim, S);
    const int V0[3] = {0, 0, 0};
    for (PetscInt k = 0; k < nmat; ++k) {
        if (!((need >> k) & 1u)) continue;
        for (PetscInt q3 = 0; q3 < Q3; ++q3)
        for (PetscInt q2 = 0; q2 < Q2; ++q2)
        for (PetscInt q1 = 0; q1 < Q1; ++q1) {
            const size_t q = (size_t)q1 + (size_t)Q1 * ((size_t)q2 + (size_t)Q2 * q3);
            const PetscReal z = (dim == 3) ? A[2].X[q3] : 0.0;
            const PetscReal w = A[0].W[q1] * A[1].W[q2] * ((dim == 3) ? A[2].W[q3] : 1.0);
            c[q] = w == 0.0 ? 0.0 : TDSEZParser::FieldProfile(TDSEZFieldTermIndex + k, A[0].X[q1], A[1].X[q2], z) * w;
        }
        Contract(A, dim, V0, V0, c.data(), false, S);
        Store(S, nen, 1.0, Sym::Symmetric, work[k]);
    }
    return true;
}

/// Coefficient arrays on the element's tensor points and the contractions of
/// TDSEZFormHam (H, M) and TDSEZFormPhyAxes. Returns false if the
/// kernel/configuration is not handled (the caller then uses the per-point
/// kernel). TDSEZFormPhyX is the 1D kernel and is never sum-factorised.
bool SumFactElement(IGA iga, IGAElement e, TDSEZPhysicsKernel kernel,
                    PetscInt nmat, PetscScalar *work[], unsigned need)
{
    const PetscInt dim = e->dim;
    if (dim < 2 || dim > 3 || e->dof != 1 || e->geometry || e->rational) return false;
    if (kernel == TDSEZFormPhyAxes) return SumFactAxes(iga, e, nmat, work, need);
    if (kernel == TDSEZFormMatFreeAxes) return SumFactMatFree(iga, e, nmat, work, need);
    if (kernel == TDSEZFormFieldTerms) return SumFactFieldTerms(iga, e, nmat, work, need);
    if (kernel != TDSEZFormHam || nmat < 2) return false;

    thread_local Axis1D A[3];
    for (PetscInt d = 0; d < dim; ++d) LoadAxis(iga, e, d, A[d]);
    PetscInt nen = 1;
    for (PetscInt d = 0; d < dim; ++d) nen *= A[d].n;
    if (nen != e->nen) return false;

    const PetscInt Q1 = A[0].Q, Q2 = A[1].Q, Q3 = (dim == 3) ? A[2].Q : 1;
    const size_t nq = (size_t)Q1 * Q2 * Q3;
    thread_local std::vector<PetscReal> cK, cV, cM;
    cK.resize(nq); cV.resize(nq); cM.resize(nq);

    for (PetscInt q3 = 0; q3 < Q3; ++q3)
    for (PetscInt q2 = 0; q2 < Q2; ++q2)
    for (PetscInt q1 = 0; q1 < Q1; ++q1) {
        const size_t q = (size_t)q1 + (size_t)Q1 * ((size_t)q2 + (size_t)Q2 * q3);
        const PetscReal x = A[0].X[q1], y = A[1].X[q2], z = (dim == 3) ? A[2].X[q3] : 0.0;
        const PetscReal w = A[0].W[q1] * A[1].W[q2] * ((dim == 3) ? A[2].W[q3] : 1.0);
        const PetscReal invMass = (PetscReal)(1.0 / TDSEZParser::MassDist(x, y, z));
        const PetscReal Vxyz = (dim == 3) ? TDSEZParser::V(x, y, z) : TDSEZParser::V(x, y);
        cK[q] = 0.5 * invMass * w;
        cV[q] = Vxyz * w;
        cM[q] = w;
    }

    Scratch &S = scratch;
    Offsets(A, dim, S);
    const int V0[3] = {0, 0, 0};
    auto want = [need](PetscInt k) { return ((need >> k) & 1u) != 0; };
    if (want(0)) {
        for (PetscInt d = 0; d < dim; ++d) {           // kinetic: sum_d (D_d, D_d)
            int ad[3] = {0, 0, 0};
            ad[d] = 1;
            Contract(A, dim, ad, ad, cK.data(), d > 0, S);
        }
        Contract(A, dim, V0, V0, cV.data(), true, S);  // H = K + V
        Store(S, nen, 1.0, Sym::Symmetric, work[0]);
    }
    if (want(1)) {
        Contract(A, dim, V0, V0, cM.data(), false, S); // M
        Store(S, nen, 1.0, Sym::Symmetric, work[1]);
    }
    return true;
}

} // namespace

ForceCoeffStore &TDSEZForceCoeffStore()
{
    static ForceCoeffStore store;
    return store;
}


/// Element matrices of @p kernel for @p element into work[0..nmat-1] (zeroed
/// by the caller): sum-factorised when supported, otherwise via the per-point
/// kernel. Only the matrices flagged in @p need (bit k = work[k]) have to be
/// produced; the sum-factorised path skips the others, the per-point kernels
/// compute all of them. With -tdsez_sumfact_check both paths are run and the
/// needed matrices compared.
PetscErrorCode TDSEZElementWork(IGA iga, IGAElement element, TDSEZPhysicsKernel kernel,
                                PetscInt nmat, PetscScalar *work[], void *ctx, unsigned need)
{
    PetscFunctionBeginUser;
    const PetscBool check = SumFactCheck();
    bool done = false;
    if (SumFactEnabled()) done = SumFactElement(iga, element, kernel, nmat, work, need);

    if (!done || check) {
        const size_t n2 = (size_t)element->nen * element->nen;
        thread_local std::vector<PetscScalar> ref;
        PetscScalar *dst[12];
        if (done) {
            ref.assign((size_t)nmat * n2, 0.0);
            for (PetscInt k = 0; k < nmat; ++k) dst[k] = ref.data() + (size_t)k * n2;
        } else {
            for (PetscInt k = 0; k < nmat; ++k) dst[k] = work[k];
        }
        IGAPoint point = NULL;
        PetscCall(IGAElementBeginPoint(element, &point));
        while (IGAElementNextPoint(element, point)) PetscCall(kernel(point, nmat, dst, ctx));
        PetscCall(IGAElementEndPoint(element, &point));

        if (done) {
            static PetscReal maxRel = 0.0;
            static PetscInt  nEl = 0;
            for (PetscInt k = 0; k < nmat; ++k) {
                if (!((need >> k) & 1u)) continue;
                PetscReal dmax = 0.0, ref_max = 0.0;
                for (size_t i = 0; i < n2; ++i) {
                    dmax    = PetscMax(dmax, PetscAbsScalar(work[k][i] - dst[k][i]));
                    ref_max = PetscMax(ref_max, PetscAbsScalar(dst[k][i]));
                }
                if (ref_max > 0.0) maxRel = PetscMax(maxRel, dmax / ref_max);
            }
            ++nEl;
            if ((nEl & (nEl - 1)) == 0 || nEl == element->count)   // powers of two and last
                TDSEZInfo(PETSC_COMM_SELF).kv("Sum-factorisation check", "%s: %" PetscInt_FMT " elements, max relative difference %.3e",
                    kernel == TDSEZFormHam ? "H,M" : "TD ops", nEl, (double)maxRel);
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZBlasSerialBegin(PetscInt *saved)
{
    PetscFunctionBeginUser;
    *saved = 1;
#if PETSC_VERSION_GE(3, 21, 0)
    PetscCall(PetscBLASGetNumThreads(saved));
    PetscCall(PetscBLASSetNumThreads(1));
#endif
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZBlasSerialEnd(PetscInt saved)
{
    PetscFunctionBeginUser;
#if PETSC_VERSION_GE(3, 21, 0)
    if (saved != 1) PetscCall(PetscBLASSetNumThreads(saved));
#else
    (void)saved;
#endif
    PetscFunctionReturn(PETSC_SUCCESS);
}
