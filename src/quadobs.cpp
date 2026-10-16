#include "tdsez_quadobs.hpp"
#include "tdsez_forcecoeffs.hpp"
#include "tdsez_info.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

/**
 * @file quadobs.cpp
 * @brief Monitor expectation values from psi at the quadrature points
 *        (see tdsez_quadobs.hpp).
 */

namespace {

/// CAP profile W(r) = sum_d W_d(x_d), as TDSEZformCap.
PetscReal CapValue(const PetscReal r[3], PetscInt dim)
{
    const PetscReal lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
    const PetscReal hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
    const PetscReal width = TDSEZCapWidth();
    PetscReal W = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
        const PetscReal s = PetscMin(r[d] - lo[d], hi[d] - r[d]);
        if (s < width) W += Manolopoulos_CAP_profile(PetscMin(width - s, width * (1.0 - 1e-10)), 0.0, width, TDSEZParser::CAPKmin);
    }
    return W;
}

} // namespace

bool TDSEZMatrixFree()
{
    static int on = -1;
    if (on < 0) {
        PetscBool req = PETSC_FALSE, tensor = PETSC_TRUE;
        PetscOptionsGetBool(NULL, NULL, "-tdsez_mat_free", &req, NULL);
        PetscOptionsGetBool(NULL, NULL, "-tdsez_tensor_ops", &tensor, NULL);
        std::string basis = TDSEZParser::Basis;
        std::transform(basis.begin(), basis.end(), basis.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        const bool bspline = basis.empty() || basis == "bspline" || basis == "b-spline";
        const bool ok = TDSEZQuadObsEnabled() && tensor && bspline && !TDSEZParser::EnableGPU;
        on = (req && ok) ? 1 : 0;
        if (req && !ok)
            TDSEZInfo().warn("-tdsez_mat_free needs 2D/3D, no Dirichlet walls, the B-spline basis, a CPU run and the "
                             "factorised and quadrature paths (-tdsez_tensor_ops, -tdsez_quad_obs); assembling all operators");
    }
    return on == 1;
}

bool TDSEZQuadObsEnabled()
{
    static int on = -1;
    if (on < 0) {
        PetscBool opt = PETSC_TRUE;
        PetscOptionsGetBool(NULL, NULL, "-tdsez_quad_obs", &opt, NULL);
        std::string bt = TDSEZParser::BoundaryType;
        std::transform(bt.begin(), bt.end(), bt.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        on = (opt && TDSEZParser::Dimension >= 2 && bt != "dirichlet" && bt != "wall") ? 1 : 0;
    }
    return on == 1;
}

PetscErrorCode TDSEZQuadObs::Setup(IGA iga, unsigned what, bool forceCorrections, unsigned forceAxes)
{
    PetscFunctionBeginUser;
    ready_ = false;
    iga_ = iga;
    if (!forceAxes) what &= ~(unsigned)Force;
    what_ = what;
    forceAxes_ = (what & Force) ? forceAxes : 0u;
    corr_ = forceCorrections && (what & Force);
    dim_ = iga->dim;
    if (dim_ < 2 || dim_ > 3 || iga->dof != 1 || iga->geometry || iga->rational || !what) PetscFunctionReturn(PETSC_SUCCESS);

    std::vector<PetscReal> X[3];
    for (PetscInt d = 0; d < 3; ++d) {
        Axis &A = ax_[d];
        if (d >= dim_) {
            A = Axis(); A.N0 = {1.0}; A.N1 = {0.0}; A.W = {1.0}; A.off = {0}; X[d] = {0.0}; gw_[d] = 1;
            continue;
        }
        const IGABasis B = iga->basis[d];
        const PetscInt es = iga->elem_start[d];
        A.ne = iga->elem_width[d]; A.n = B->nen; A.Q = B->nqp;
        A.N0.assign((size_t)A.ne * A.Q * A.n, 0.0); A.N1 = A.N0;
        A.W.assign((size_t)A.ne * A.Q, 0.0); X[d] = A.W;
        A.off.resize(A.ne);
        for (PetscInt le = 0; le < A.ne; ++le) {
            const PetscInt e = es + le;
            A.off[le] = B->offset[e] - iga->node_gstart[d];
            for (PetscInt q = 0; q < A.Q; ++q) {
                A.W[le * A.Q + q] = B->weight[e * A.Q + q] * B->detJac[e];
                X[d][le * A.Q + q] = B->point[e * A.Q + q];
                for (PetscInt a = 0; a < A.n; ++a) {
                    // weight-0 points (reduced rule) have uncomputed values
                    const PetscReal *v = B->value + (((size_t)e * A.Q + q) * A.n + a) * 5;
                    const bool used = B->weight[e * A.Q + q] != 0.0;
                    A.N0[((size_t)le * A.n + a) * A.Q + q] = used ? v[0] : 0.0;
                    A.N1[((size_t)le * A.n + a) * A.Q + q] = used ? v[1] : 0.0;
                }
            }
        }
        gw_[d] = iga->node_gwidth[d];
    }
    Qe_ = ax_[0].Q * ax_[1].Q * ax_[2].Q;
    const size_t np = (size_t)ax_[0].ne * ax_[1].ne * ax_[2].ne * Qe_;

    // coefficients per point, weights included
    cV_.clear(); cInvM_.clear(); cW_.clear();
    if (what & Pot) cV_.resize(np);
    if (what & (InvMass | Kin | Vel)) cInvM_.resize(np);
    if (what & Cap) cW_.resize(np);
    auto wantF = [this](PetscInt a) { return ((forceAxes_ >> a) & 1u) != 0; };
    // force coefficients kept by the assembly: same points, weights and order
    ForceCoeffStore &st = TDSEZForceCoeffStore();
    bool kept = corr_ && st.Complete() && (size_t)st.nq == (size_t)Qe_ && st.done.size() * (size_t)Qe_ == np;
    for (PetscInt d = 0; d < dim_ && kept; ++d) kept = st.start[d] == iga->elem_start[d] && st.width[d] == iga->elem_width[d];
    for (PetscInt a = 0; a < dim_ && kept; ++a) kept = !wantF(a) || st.axis[a];
    ForceCoeffs fc[3];
    for (PetscInt a = 0; a < dim_; ++a) {
        if (!wantF(a)) continue;
        if (kept) std::swap(fc[a], st.F[a]);
        else { fc[a].Resize(corr_ ? np : 0); fc[a].BB.assign(np, 0.0); }
    }
    const bool tabulateF = (what & Force) && !kept;
    size_t p = 0;
    for (PetscInt e3 = 0; e3 < ax_[2].ne; ++e3)
    for (PetscInt e2 = 0; e2 < ax_[1].ne; ++e2)
    for (PetscInt e1 = 0; e1 < ax_[0].ne; ++e1)
    for (PetscInt q3 = 0; q3 < ax_[2].Q; ++q3)
    for (PetscInt q2 = 0; q2 < ax_[1].Q; ++q2)
    for (PetscInt q1 = 0; q1 < ax_[0].Q; ++q1, ++p) {
        const PetscReal r[3] = {X[0][e1 * ax_[0].Q + q1], X[1][e2 * ax_[1].Q + q2], X[2][e3 * ax_[2].Q + q3]};
        const PetscReal w = ax_[0].W[e1 * ax_[0].Q + q1] * ax_[1].W[e2 * ax_[1].Q + q2] * ax_[2].W[e3 * ax_[2].Q + q3];
        if (w == 0.0) continue;                       // unused points of the reduced interior rules
        const PetscReal invMass = 1.0 / TDSEZParser::MassDist(r[0], r[1], r[2]);
        if (what & Pot) cV_[p] = TDSEZParser::V(r[0], r[1], r[2]) * w;
        if (!cInvM_.empty()) cInvM_[p] = invMass * w;
        if (what & Cap) cW_[p] = CapValue(r, dim_) * w;
        if (tabulateF) {
            for (PetscInt a = 0; a < dim_; ++a) {
                if (!wantF(a)) continue;
                const PetscReal g = (a == 0) ? TDSEZParser::dVx(r[0], r[1], r[2])
                                  : (a == 1) ? TDSEZParser::dVy(r[0], r[1], r[2]) : TDSEZParser::dVz(r[0], r[1], r[2]);
                fc[a].BB[p] = -g * invMass * w;
            }
            if (corr_) {
                const std::vector<PetscReal> pt = dim_ == 3 ? std::vector<PetscReal>{r[0], r[1], r[2]}
                                                            : std::vector<PetscReal>{r[0], r[1]};
                const PhysicsDerivativesFull &ops = TDSEZInvMassDerivatives(pt);
                for (PetscInt a = 0; a < dim_; ++a) if (wantF(a)) AddForceCorrections((int)a, dim_, invMass, ops, w, p, fc[a]);
            }
        }
    }
    for (PetscInt a = 0; a < 3; ++a) {
        cF_[a].swap(fc[a].BB);
        for (int k = 0; k < 3; ++k) { cBd_[a][k].swap(fc[a].Bd[k]); cBp_[a][k].swap(fc[a].Bp[k]); cdd_[a][k].swap(fc[a].dd[k]); }
    }
    for (auto &f : F_) f.resize(Qe_);
    ready_ = true;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZQuadObs::Load(Vec x, std::vector<PetscScalar> &local)
{
    PetscFunctionBeginUser;
    Vec lvec;
    const PetscScalar *arr;
    PetscCall(IGAGetLocalVecArray(iga_, x, &lvec, &arr));
    PetscInt n;
    PetscCall(VecGetLocalSize(lvec, &n));
    local.assign(arr, arr + n);
    PetscCall(IGARestoreLocalVecArray(iga_, x, &lvec, &arr));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// F_[d1 + 2 d2 + 4 d3][q] = D^(d1,d2,d3) psi at the points of element (e1,e2,e3):
/// three 1D contractions of the (p+1)^dim coefficients with the 1D tables.
void TDSEZQuadObs::Evaluate(const PetscScalar *c, PetscInt e1, PetscInt e2, PetscInt e3, bool mixed)
{
    const Axis &X = ax_[0], &Y = ax_[1], &Z = ax_[2];
    const PetscInt n1 = X.n, n2 = Y.n, n3 = Z.n, Q1 = X.Q, Q2 = Y.Q, Q3 = Z.Q;
    const PetscInt o1 = X.off[e1], o2 = Y.off[e2], o3 = Z.off[e3];
    const PetscReal *Nx[2] = {X.N0.data() + (size_t)e1 * Q1 * n1, X.N1.data() + (size_t)e1 * Q1 * n1};
    const PetscReal *Ny[2] = {Y.N0.data() + (size_t)e2 * Q2 * n2, Y.N1.data() + (size_t)e2 * Q2 * n2};
    const PetscReal *Nz[2] = {Z.N0.data() + (size_t)e3 * Q3 * n3, Z.N1.data() + (size_t)e3 * Q3 * n3};
    // stage x: t1[d1][(a3 n2 + a2) Q1 + q1]
    const size_t s1 = (size_t)n3 * n2 * Q1;
    t1_.resize(2 * s1);
    for (int d1 = 0; d1 < 2; ++d1)
        for (PetscInt a3 = 0; a3 < n3; ++a3)
            for (PetscInt a2 = 0; a2 < n2; ++a2) {
                const PetscScalar *cr = c + (o1 + gw_[0] * ((o2 + a2) + gw_[1] * (o3 + a3)));
                PetscScalar *tr = t1_.data() + d1 * s1 + ((size_t)a3 * n2 + a2) * Q1;
                std::fill(tr, tr + Q1, (PetscScalar)0.0);
                for (PetscInt a1 = 0; a1 < n1; ++a1) {
                    const PetscReal *nx = Nx[d1] + (size_t)a1 * Q1;
                    const PetscScalar ca = cr[a1];
                    for (PetscInt q1 = 0; q1 < Q1; ++q1) tr[q1] += nx[q1] * ca;
                }
            }
    // stage y: t2[d1 + 2 d2][(a3 Q2 + q2) Q1 + q1]
    const size_t s2 = (size_t)n3 * Q2 * Q1;
    t2_.resize(4 * s2);
    for (int d = 0; d < 4; ++d) {
        const int d1 = d & 1, d2 = d >> 1;
        if ((d2 && dim_ < 2) || (d == 3 && !mixed)) continue;
        for (PetscInt a3 = 0; a3 < n3; ++a3)
            for (PetscInt q2 = 0; q2 < Q2; ++q2) {
                PetscScalar *tr = t2_.data() + d * s2 + ((size_t)a3 * Q2 + q2) * Q1;
                std::fill(tr, tr + Q1, (PetscScalar)0.0);
                for (PetscInt a2 = 0; a2 < n2; ++a2) {
                    const PetscReal ny = Ny[d2][(size_t)a2 * Q2 + q2];
                    const PetscScalar *src = t1_.data() + d1 * s1 + ((size_t)a3 * n2 + a2) * Q1;
                    for (PetscInt q1 = 0; q1 < Q1; ++q1) tr[q1] += ny * src[q1];
                }
            }
    }
    // stage z: F[d1 + 2 d2 + 4 d3][(q3 Q2 + q2) Q1 + q1]
    const size_t plane = (size_t)Q2 * Q1;
    for (int k = 0; k < 8; ++k) {
        const int d12 = k & 3, d3 = k >> 2;
        const int order = (k & 1) + ((k >> 1) & 1) + d3;
        if ((d3 && dim_ < 3) || (!mixed && order > 1) || order > 2) continue;
        PetscScalar *f = F_[k].data();
        std::fill(f, f + Qe_, (PetscScalar)0.0);
        for (PetscInt q3 = 0; q3 < Q3; ++q3) {
            for (PetscInt a3 = 0; a3 < n3; ++a3) {
                const PetscReal nz = Nz[d3][(size_t)a3 * Q3 + q3];
                const PetscScalar *src = t2_.data() + d12 * s2 + (size_t)a3 * plane;
                PetscScalar *dst = f + (size_t)q3 * plane;
                for (size_t m = 0; m < plane; ++m) dst[m] += nz * src[m];
            }
        }
    }
}

PetscErrorCode TDSEZQuadObs::Expect(Vec psi, Values &v)
{
    PetscFunctionBeginUser;
    PetscCheck(ready_, PetscObjectComm((PetscObject)psi), PETSC_ERR_ARG_WRONGSTATE, "quadrature observables not set up");
    PetscCall(Load(psi, loc_));
    // real arithmetic on (re, im) pairs; pot, invMass, kin, cap: real sums,
    // vel_a = sum c Im(conj psi d_a psi), force_a complex (corrections)
    // per-point accumulators (independent across points, so the loops
    // vectorise without reordering any one sum); summed once at the end
    enum { aV, aM, aW, aK, aVel, aFr = aVel + 3, aFi = aFr + 3, nAcc = aFi + 3 };
    acc_.assign((size_t)nAcc * Qe_, 0.0);
    auto A = [this](int k) { return acc_.data() + (size_t)k * Qe_; };
    PetscReal *sV = A(aV), *sM = A(aM), *sW = A(aW), *sK = A(aK);
    const int dIdx[3] = {1, 2, 4};
    const int pairIdx[3] = {3, 5, 6};                  // (x,y), (x,z), (y,z)
    const bool wRho = (what_ & (Pot | InvMass | Cap | Force)) != 0;
    std::vector<PetscReal> &rho = rho_;
    rho.resize(Qe_);
    auto re = [](const PetscScalar *z) { return reinterpret_cast<const PetscReal *>(z); };
    size_t p0 = 0;
    for (PetscInt e3 = 0; e3 < ax_[2].ne; ++e3)
    for (PetscInt e2 = 0; e2 < ax_[1].ne; ++e2)
    for (PetscInt e1 = 0; e1 < ax_[0].ne; ++e1, p0 += Qe_) {
        Evaluate(loc_.data(), e1, e2, e3, corr_);
        const PetscReal *P = re(F_[0].data());
        if (wRho) for (PetscInt q = 0; q < Qe_; ++q) rho[q] = P[2 * q] * P[2 * q] + P[2 * q + 1] * P[2 * q + 1];
        if (what_ & Pot)     { const PetscReal *c = cV_.data() + p0;    for (PetscInt q = 0; q < Qe_; ++q) sV[q] += c[q] * rho[q]; }
        if (what_ & InvMass) { const PetscReal *c = cInvM_.data() + p0; for (PetscInt q = 0; q < Qe_; ++q) sM[q] += c[q] * rho[q]; }
        if (what_ & Cap)     { const PetscReal *c = cW_.data() + p0;    for (PetscInt q = 0; q < Qe_; ++q) sW[q] += c[q] * rho[q]; }
        for (PetscInt d = 0; d < dim_ && (what_ & (Kin | Vel)); ++d) {
            const PetscReal *D = re(F_[dIdx[d]].data()), *c = cInvM_.data() + p0;
            if (what_ & Kin)
                for (PetscInt q = 0; q < Qe_; ++q) sK[q] += 0.5 * c[q] * (D[2 * q] * D[2 * q] + D[2 * q + 1] * D[2 * q + 1]);
            PetscReal *sv = A(aVel + (int)d);
            if (what_ & Vel)
                for (PetscInt q = 0; q < Qe_; ++q) sv[q] += c[q] * (P[2 * q] * D[2 * q + 1] - P[2 * q + 1] * D[2 * q]);
        }
        if (!(what_ & Force)) continue;
        for (PetscInt a = 0; a < dim_; ++a) {
            if (!((forceAxes_ >> a) & 1u)) continue;
            PetscReal *fr = A(aFr + (int)a), *fi = A(aFi + (int)a);
            { const PetscReal *c = cF_[a].data() + p0; for (PetscInt q = 0; q < Qe_; ++q) fr[q] += c[q] * rho[q]; }
            if (!corr_) continue;
            // conj(psi) g: cBd (g = d_e psi), cBp (g = d_a d_e psi); cdd |d_e psi|^2
            auto cross = [&](const PetscReal *c, const PetscReal *G) {
                for (PetscInt q = 0; q < Qe_; ++q) {
                    fr[q] += c[q] * (P[2 * q] * G[2 * q] + P[2 * q + 1] * G[2 * q + 1]);
                    fi[q] += c[q] * (P[2 * q] * G[2 * q + 1] - P[2 * q + 1] * G[2 * q]);
                }
            };
            for (PetscInt e = 0; e < dim_; ++e) {
                const PetscReal *D = re(F_[dIdx[e]].data());
                cross(cBd_[a][e].data() + p0, D);
                const PetscReal *c = cdd_[a][e].data() + p0;
                for (PetscInt q = 0; q < Qe_; ++q) fr[q] += c[q] * (D[2 * q] * D[2 * q] + D[2 * q + 1] * D[2 * q + 1]);
                if (e != a) {
                    const int k = PairIndex((int)PetscMin(a, e), (int)PetscMax(a, e));
                    cross(cBp_[a][k].data() + p0, re(F_[pairIdx[k]].data()));
                }
            }
        }
    }
    PetscReal sum[nAcc];
    for (int k = 0; k < nAcc; ++k) { sum[k] = 0; for (PetscInt q = 0; q < Qe_; ++q) sum[k] += A(k)[q]; }
    const PetscScalar I = PetscCMPLX(0.0, 1.0);
    PetscScalar acc[10] = {sum[aV], sum[aM], sum[aK], -I * sum[aW], 0, 0, 0, sum[aVel], sum[aVel + 1], sum[aVel + 2]};
    for (int a = 0; a < 3; ++a) acc[4 + a] = PetscCMPLX(sum[aFr + a], sum[aFi + a]);
    PetscScalar tot[10];
    PetscCallMPI(MPI_Allreduce(acc, tot, 10, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)psi)));
    v.pot = tot[0]; v.invMass = tot[1]; v.kin = tot[2]; v.cap = tot[3];
    for (int a = 0; a < 3; ++a) { v.force[a] = tot[4 + a]; v.vel[a] = tot[7 + a]; }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZQuadObs::VelocityBilinear(Vec w, Vec psi, PetscScalar out[3])
{
    PetscFunctionBeginUser;
    PetscCheck(ready_ && !cInvM_.empty(), PetscObjectComm((PetscObject)psi), PETSC_ERR_ARG_WRONGSTATE, "velocity not set up");
    PetscCall(Load(w, loc2_));
    PetscCall(Load(psi, loc_));
    // w^H Vel_a psi = sum_q (-i/2) c z,  z = conj(w) d_a psi - conj(d_a w) psi,  c = w_q / m
    const int dIdx[3] = {1, 2, 4};
    auto re = [](const PetscScalar *z) { return reinterpret_cast<const PetscReal *>(z); };
    for (int k = 0; k < 4; ++k) wq_[k].resize(Qe_);
    acc_.assign((size_t)6 * Qe_, 0.0);                 // per point: [a] c zi, [3 + a] c zr
    size_t p0 = 0;
    for (PetscInt e3 = 0; e3 < ax_[2].ne; ++e3)
    for (PetscInt e2 = 0; e2 < ax_[1].ne; ++e2)
    for (PetscInt e1 = 0; e1 < ax_[0].ne; ++e1, p0 += Qe_) {
        Evaluate(loc2_.data(), e1, e2, e3, false);
        std::copy(F_[0].begin(), F_[0].end(), wq_[0].begin());
        for (PetscInt d = 0; d < dim_; ++d) std::copy(F_[dIdx[d]].begin(), F_[dIdx[d]].end(), wq_[1 + d].begin());
        Evaluate(loc_.data(), e1, e2, e3, false);
        const PetscReal *W0 = re(wq_[0].data()), *P0 = re(F_[0].data()), *c = cInvM_.data() + p0;
        for (PetscInt a = 0; a < dim_; ++a) {
            const PetscReal *Wa = re(wq_[1 + a].data()), *Pa = re(F_[dIdx[a]].data());
            PetscReal *sRe = acc_.data() + (size_t)a * Qe_, *sIm = acc_.data() + (size_t)(3 + a) * Qe_;
            for (PetscInt q = 0; q < Qe_; ++q) {
                // z = conj(w) d_a psi - conj(d_a w) psi
                const PetscReal zr = W0[2 * q] * Pa[2 * q] + W0[2 * q + 1] * Pa[2 * q + 1] - Wa[2 * q] * P0[2 * q] - Wa[2 * q + 1] * P0[2 * q + 1];
                const PetscReal zi = W0[2 * q] * Pa[2 * q + 1] - W0[2 * q + 1] * Pa[2 * q] - Wa[2 * q] * P0[2 * q + 1] + Wa[2 * q + 1] * P0[2 * q];
                sRe[q] += c[q] * zi;
                sIm[q] += c[q] * zr;
            }
        }
    }
    PetscScalar acc[3];
    for (int a = 0; a < 3; ++a) {
        PetscReal zi = 0, zr = 0;
        for (PetscInt q = 0; q < Qe_; ++q) { zi += acc_[(size_t)a * Qe_ + q]; zr += acc_[(size_t)(3 + a) * Qe_ + q]; }
        acc[a] = PetscCMPLX(0.5 * zi, -0.5 * zr);       // (-i/2) c z
    }
    PetscCallMPI(MPI_Allreduce(acc, out, 3, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)psi)));
    PetscFunctionReturn(PETSC_SUCCESS);
}
