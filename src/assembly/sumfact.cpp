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
//  identical up to floating-point rounding.
//
//  Used for TDSEZFormHam (H, M) and for TDSEZFormPhyX (K, V, Md, X, VelX,
//  dVdx) whenever the mass-gradient corrections of dVdx are not active.
//  All other cases fall back to the per-point kernels.
//
//  Options: -tdsez_sumfact 0        disable (per-point kernels everywhere)
//           -tdsez_sumfact_check    compute every element both ways and print
//                                   the largest relative difference
// ============================================================================

#include "tdsez_internal.hpp"
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
        A.X[q] = B->point[el * A.Q + q];
        for (PetscInt a = 0; a < A.n; ++a)
            for (int k = 0; k < 2; ++k)
                A.N[k][(size_t)a * A.Q + q] = B->value[(((size_t)el * A.Q + q) * A.n + a) * 5 + k];
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

/// Element-matrix contribution  work += factor * sum_q c(q) D^al N_a D^be N_b
/// for a 2D or 3D element with x-fastest local numbering a = a1 + n1 (a2 + n2 a3)
/// and x-fastest point numbering q = q1 + Q1 (q2 + Q2 q3).
void Contract(const Axis1D *A, PetscInt dim, const int al[3], const int be[3],
              const PetscReal *c, PetscScalar factor, PetscScalar *work, PetscInt nen)
{
    thread_local std::vector<PetscReal> P1, P2, P3, T1, T2, E;
    const PetscInt n1 = A[0].n, Q1 = A[0].Q, n2 = A[1].n, Q2 = A[1].Q;
    Products(A[0], al[0], be[0], P1);
    Products(A[1], al[1], be[1], P2);
    const size_t m1 = (size_t)n1 * n1, m2 = (size_t)n2 * n2;

    if (dim == 2) {
        // T1[ab1][q2] = sum_q1 P1[ab1][q1] c[q1 + Q1 q2]
        T1.assign(m1 * Q2, 0.0);
        for (PetscInt q2 = 0; q2 < Q2; ++q2) {
            const PetscReal *cq = c + (size_t)q2 * Q1;
            for (size_t ab1 = 0; ab1 < m1; ++ab1) {
                const PetscReal *p = &P1[ab1 * Q1];
                PetscReal s = 0.0;
                for (PetscInt q1 = 0; q1 < Q1; ++q1) s += p[q1] * cq[q1];
                T1[ab1 * Q2 + q2] = s;
            }
        }
        // E[ab1][ab2] = sum_q2 P2[ab2][q2] T1[ab1][q2]
        for (PetscInt a2 = 0; a2 < n2; ++a2)
        for (PetscInt b2 = 0; b2 < n2; ++b2) {
            const PetscReal *p = &P2[((size_t)(a2 * n2 + b2)) * Q2];
            for (PetscInt a1 = 0; a1 < n1; ++a1)
            for (PetscInt b1 = 0; b1 < n1; ++b1) {
                const PetscReal *t = &T1[((size_t)(a1 * n1 + b1)) * Q2];
                PetscReal s = 0.0;
                for (PetscInt q2 = 0; q2 < Q2; ++q2) s += p[q2] * t[q2];
                const PetscInt a = a1 + n1 * a2, b = b1 + n1 * b2;
                work[(size_t)a * nen + b] += factor * s;
            }
        }
        return;
    }

    const PetscInt n3 = A[2].n, Q3 = A[2].Q;
    Products(A[2], al[2], be[2], P3);
    const size_t m3 = (size_t)n3 * n3;
    const size_t Q23 = (size_t)Q2 * Q3;

    // T1[ab1][q2 + Q2 q3] = sum_q1 P1[ab1][q1] c[q1 + Q1 (q2 + Q2 q3)]
    T1.assign(m1 * Q23, 0.0);
    for (size_t q23 = 0; q23 < Q23; ++q23) {
        const PetscReal *cq = c + q23 * Q1;
        for (size_t ab1 = 0; ab1 < m1; ++ab1) {
            const PetscReal *p = &P1[ab1 * Q1];
            PetscReal s = 0.0;
            for (PetscInt q1 = 0; q1 < Q1; ++q1) s += p[q1] * cq[q1];
            T1[ab1 * Q23 + q23] = s;
        }
    }
    // T2[(ab1 m2 + ab2)][q3] = sum_q2 P2[ab2][q2] T1[ab1][q2 + Q2 q3]
    T2.assign(m1 * m2 * Q3, 0.0);
    for (size_t ab1 = 0; ab1 < m1; ++ab1)
        for (size_t ab2 = 0; ab2 < m2; ++ab2) {
            const PetscReal *p = &P2[ab2 * Q2];
            PetscReal *t2 = &T2[(ab1 * m2 + ab2) * Q3];
            for (PetscInt q3 = 0; q3 < Q3; ++q3) {
                const PetscReal *t1 = &T1[ab1 * Q23 + (size_t)q3 * Q2];
                PetscReal s = 0.0;
                for (PetscInt q2 = 0; q2 < Q2; ++q2) s += p[q2] * t1[q2];
                t2[q3] = s;
            }
        }
    // work[a][b] += factor * sum_q3 P3[ab3][q3] T2[ab1 ab2][q3]
    for (PetscInt a3 = 0; a3 < n3; ++a3)
    for (PetscInt b3 = 0; b3 < n3; ++b3) {
        const PetscReal *p = &P3[((size_t)(a3 * n3 + b3)) * Q3];
        for (PetscInt a2 = 0; a2 < n2; ++a2)
        for (PetscInt b2 = 0; b2 < n2; ++b2) {
            const size_t ab2 = (size_t)(a2 * n2 + b2);
            for (PetscInt a1 = 0; a1 < n1; ++a1)
            for (PetscInt b1 = 0; b1 < n1; ++b1) {
                const PetscReal *t = &T2[((size_t)(a1 * n1 + b1) * m2 + ab2) * Q3];
                PetscReal s = 0.0;
                for (PetscInt q3 = 0; q3 < Q3; ++q3) s += p[q3] * t[q3];
                const PetscInt a = a1 + n1 * (a2 + n2 * a3);
                const PetscInt b = b1 + n1 * (b2 + n2 * b3);
                work[(size_t)a * nen + b] += factor * s;
            }
        }
    }
    (void)m3;
}

/// Make a symmetric element matrix exactly symmetric: the two triangles are
/// contracted in different orders and differ by rounding.
void Symmetrise(PetscScalar *w, PetscInt nen)
{
    for (PetscInt a = 0; a < nen; ++a)
        for (PetscInt b = a + 1; b < nen; ++b) {
            const PetscScalar m = 0.5 * (w[(size_t)a * nen + b] + w[(size_t)b * nen + a]);
            w[(size_t)a * nen + b] = m;
            w[(size_t)b * nen + a] = m;
        }
}

/// w <- w - w^T (exactly antisymmetric); used for the velocity matrix.
void AntiSymmetrise(PetscScalar *w, PetscInt nen)
{
    for (PetscInt a = 0; a < nen; ++a) {
        w[(size_t)a * nen + a] = 0.0;
        for (PetscInt b = a + 1; b < nen; ++b) {
            const PetscScalar d = w[(size_t)a * nen + b] - w[(size_t)b * nen + a];
            w[(size_t)a * nen + b] = d;
            w[(size_t)b * nen + a] = -d;
        }
    }
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

/// Sum-factorised TDSEZFormPhyAxes: [K, V, Md] + [D_d, Vel_d, dV_d] per active
/// axis. Only for a constant mass: the per-axis force kernels then have
/// exactly zero mass-gradient corrections (finite differences of a constant),
/// so dV_d is the classical term -B_a (dV/dx_d)(1/m) B_b.
bool SumFactAxes(IGA iga, IGAElement e, PetscInt nmat, PetscScalar *work[])
{
    const PetscInt dim = e->dim;
    if (dim < 2 || dim > 3 || e->dof != 1 || e->geometry || e->rational) return false;
    // MassIsConstant = 1 in the input, and the mass expression really has no
    // x/y/z dependence (the per-axis force kernels compute the corrections
    // from the expression itself, whatever the flag says).
    if (TDSEZParser::MassIsConstant != PETSC_TRUE) return false;
    try { if (!TDSEZParser::MassExpr.GetUsedVar().empty()) return false; }
    catch (const mu::Parser::exception_type &) { return false; }
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
    cK.resize(nq); cV.resize(nq); cM.resize(nq); cVel.resize(nq);
    for (int d = 0; d < 3; ++d) if (TDSEZPhyAxesOn[d]) { cD[d].resize(nq); cG[d].resize(nq); }

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
    }

    const int V0[3] = {0, 0, 0};
    for (PetscInt d = 0; d < dim; ++d) {
        int ad[3] = {0, 0, 0};
        ad[d] = 1;
        Contract(A, dim, ad, ad, cK.data(), 1.0, work[0], nen);
    }
    Contract(A, dim, V0, V0, cV.data(), 1.0, work[1], nen);
    Contract(A, dim, V0, V0, cM.data(), 1.0, work[2], nen);
    for (int k = 0; k < 3; ++k) Symmetrise(work[k], nen);
    PetscInt k = 3;
    for (int d = 0; d < 3; ++d) {
        if (!TDSEZPhyAxesOn[d]) continue;
        int ad[3] = {0, 0, 0};
        ad[d] = 1;
        Contract(A, dim, V0, V0, cD[d].data(), 1.0, work[k], nen);
        Symmetrise(work[k], nen);
        Contract(A, dim, V0, ad, cVel.data(), PETSC_i, work[k + 1], nen);
        AntiSymmetrise(work[k + 1], nen);
        Contract(A, dim, V0, V0, cG[d].data(), 1.0, work[k + 2], nen);
        Symmetrise(work[k + 2], nen);
        k += 3;
    }
    return true;
}

/// Coefficient arrays on the element's tensor points and the contractions of
/// TDSEZFormHam / TDSEZFormPhyX. Returns false if the kernel/configuration is
/// not handled (the caller then uses the per-point kernel).
bool SumFactElement(IGA iga, IGAElement e, TDSEZPhysicsKernel kernel,
                    PetscInt nmat, PetscScalar *work[])
{
    const PetscInt dim = e->dim;
    if (dim < 2 || dim > 3 || e->dof != 1 || e->geometry || e->rational) return false;

    const bool ham  = (kernel == TDSEZFormHam && nmat >= 2);
    const bool phy  = (kernel == TDSEZFormPhyX && nmat >= 6);
    const bool axes = (kernel == TDSEZFormPhyAxes);
    if (axes) return SumFactAxes(iga, e, nmat, work);
    if (!ham && !phy) return false;
    if (phy) {
        // Same switches as TDSEZFormPhyX: the mass-gradient corrections of
        // dVdx are active in 2D whenever the mass varies, and in 3D only when
        // the operator-form acceleration is used.
        const bool massIsConstant = (TDSEZParser::MassIsConstant == PETSC_TRUE);
        const bool corr = (dim == 2) ? !massIsConstant
                                     : (!massIsConstant && !TDSEZDiscreteAccelerationActive());
        if (corr) return false;
    }

    thread_local Axis1D A[3];
    for (PetscInt d = 0; d < dim; ++d) LoadAxis(iga, e, d, A[d]);
    PetscInt nen = 1;
    for (PetscInt d = 0; d < dim; ++d) nen *= A[d].n;
    if (nen != e->nen) return false;

    const PetscInt Q1 = A[0].Q, Q2 = A[1].Q, Q3 = (dim == 3) ? A[2].Q : 1;
    const size_t nq = (size_t)Q1 * Q2 * Q3;
    thread_local std::vector<PetscReal> cK, cV, cM, cX, cVel, cDV;
    cK.resize(nq); cV.resize(nq); cM.resize(nq);
    if (phy) { cX.resize(nq); cVel.resize(nq); cDV.resize(nq); }

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
        if (ham) {
            cM[q] = w;
        } else {
            const PetscReal dVdx = (dim == 3) ? TDSEZParser::dVx(x, y, z) : TDSEZParser::dVx(x, y);
            cM[q]   = invMass * w;              // Md
            cX[q]   = x * w;
            cVel[q] = -0.5 * invMass * w;
            cDV[q]  = -dVdx * invMass * w;
        }
    }

    const int V0[3] = {0, 0, 0};
    for (PetscInt d = 0; d < dim; ++d) {               // kinetic: sum_d (D_d, D_d)
        int ad[3] = {0, 0, 0};
        ad[d] = 1;
        Contract(A, dim, ad, ad, cK.data(), 1.0, work[0], nen);
    }
    if (ham) {
        Contract(A, dim, V0, V0, cV.data(), 1.0, work[0], nen);   // H += V
        Contract(A, dim, V0, V0, cM.data(), 1.0, work[1], nen);   // M
        Symmetrise(work[0], nen);
        Symmetrise(work[1], nen);
        return true;
    }
    const int Dx[3] = {1, 0, 0};
    Contract(A, dim, V0, V0, cV.data(),  1.0, work[1], nen);      // V
    Contract(A, dim, V0, V0, cM.data(),  1.0, work[2], nen);      // Md
    Contract(A, dim, V0, V0, cX.data(),  1.0, work[3], nen);      // X
    Contract(A, dim, V0, Dx, cVel.data(), PETSC_i, work[4], nen); // i f B_a dB_b/dx
    AntiSymmetrise(work[4], nen);                                 // VelX = i f (B dB_x - dB_x B)
    Contract(A, dim, V0, V0, cDV.data(), 1.0, work[5], nen);      // dVdx (classical)
    for (int k : {0, 1, 2, 3, 5}) Symmetrise(work[k], nen);
    return true;
}

} // namespace


/// Element matrices of @p kernel for @p element into work[0..nmat-1] (zeroed
/// by the caller): sum-factorised when supported, otherwise via the per-point
/// kernel. With -tdsez_sumfact_check both paths are run and compared.
PetscErrorCode TDSEZElementWork(IGA iga, IGAElement element, TDSEZPhysicsKernel kernel,
                                PetscInt nmat, PetscScalar *work[], void *ctx)
{
    PetscFunctionBeginUser;
    const PetscBool check = SumFactCheck();
    bool done = false;
    if (SumFactEnabled()) done = SumFactElement(iga, element, kernel, nmat, work);

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
                PetscReal dmax = 0.0, ref_max = 0.0;
                for (size_t i = 0; i < n2; ++i) {
                    dmax    = PetscMax(dmax, PetscAbsScalar(work[k][i] - dst[k][i]));
                    ref_max = PetscMax(ref_max, PetscAbsScalar(dst[k][i]));
                }
                if (ref_max > 0.0) maxRel = PetscMax(maxRel, dmax / ref_max);
            }
            ++nEl;
            if ((nEl & (nEl - 1)) == 0 || nEl == element->count)   // powers of two and last
                PetscCall(PetscPrintf(PETSC_COMM_SELF,
                    "  [sumfact check] %s: %" PetscInt_FMT " elements, max relative difference %.3e\n",
                    kernel == TDSEZFormHam ? "H,M" : "TD ops", nEl, (double)maxRel));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
