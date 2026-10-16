#include "tdsez_tensorops.hpp"
#include <algorithm>
#include <cmath>

/**
 * @file tensorops.cpp
 * @brief Factorised application of the separable operators (see tdsez_tensorops.hpp).
 */

TDSEZTensorOps::~TDSEZTensorOps()
{
    VecScatterDestroy(&halo_);
    VecDestroy(&local_);
}

PetscErrorCode TDSEZTensorOps::Setup(IGA iga, Vec templ, PetscReal invMass)
{
    PetscFunctionBeginUser;
    ready_ = false;
    dim_ = iga->dim;
    invMass_ = invMass;
    if (dim_ < 2 || dim_ > 3 || iga->dof != 1 || iga->geometry || iga->rational) PetscFunctionReturn(PETSC_SUCCESS);
    PetscBool standard = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompareAny((PetscObject)templ, &standard, VECMPI, VECSEQ, VECSTANDARD, ""));
    if (!standard) PetscFunctionReturn(PETSC_SUCCESS);

    for (PetscInt d = 0; d < 3; ++d) {
        if (d >= dim_) {
            n_[d] = 1; p_[d] = 0; bw_[d] = 1;
            M_[d].v = {1.0}; X_[d].v = {0.0}; A_[d].v = {0.0}; S_[d].v = {0.0};
            own0_[d] = h0_[d] = 0; ownw_[d] = hw_[d] = 1;
            continue;
        }
        const IGABasis B = iga->basis[d];
        n_[d] = iga->axis[d]->nnp; p_[d] = iga->axis[d]->p; bw_[d] = 2 * p_[d] + 1;
        const size_t sz = (size_t)n_[d] * bw_[d];
        std::vector<PetscReal> G(sz, 0.0);
        M_[d].v.assign(sz, 0.0); X_[d].v.assign(sz, 0.0); S_[d].v.assign(sz, 0.0); A_[d].v.assign(sz, 0.0);
        const PetscInt nen = B->nen, nqp = B->nqp, p = p_[d], bw = bw_[d];
        for (PetscInt e = 0; e < B->nel; ++e)
            for (PetscInt q = 0; q < nqp; ++q) {
                const PetscReal w = B->weight[e * nqp + q] * B->detJac[e];
                if (w == 0.0) continue;
                const PetscReal x = B->point[e * nqp + q];
                const PetscReal *N = B->value + (size_t)(e * nqp + q) * nen * 5;
                for (PetscInt a = 0; a < nen; ++a)
                    for (PetscInt b = 0; b < nen; ++b) {
                        const size_t k = (size_t)(B->offset[e] + a) * bw + (size_t)(b - a + p);
                        M_[d].v[k] += w * N[a * 5] * N[b * 5];
                        X_[d].v[k] += w * x * N[a * 5] * N[b * 5];
                        G[k]       += w * N[a * 5] * N[b * 5 + 1];
                        S_[d].v[k] += w * N[a * 5 + 1] * N[b * 5 + 1];
                    }
            }
        for (PetscInt i = 0; i < n_[d]; ++i)                       // A = G - G^T
            for (PetscInt j = std::max<PetscInt>(0, i - p); j <= std::min(n_[d] - 1, i + p); ++j)
                A_[d].v[(size_t)i * bw + (j - i + p)] = G[(size_t)i * bw + (j - i + p)] - G[(size_t)j * bw + (i - j + p)];
        own0_[d] = iga->node_lstart[d]; ownw_[d] = iga->node_lwidth[d];
        h0_[d] = std::max<PetscInt>(0, own0_[d] - p);
        hw_[d] = std::min(n_[d], own0_[d] + ownw_[d] + p) - h0_[d];
    }

    // halo box of the owned coefficients, x fastest, as one scatter
    const PetscInt nh = hw_[0] * hw_[1] * hw_[2];
    std::vector<PetscInt> idx(nh);
    PetscInt q = 0;
    for (PetscInt k = 0; k < hw_[2]; ++k)
        for (PetscInt j = 0; j < hw_[1]; ++j)
            for (PetscInt i = 0; i < hw_[0]; ++i, ++q)
                idx[q] = (h0_[0] + i) + n_[0] * ((h0_[1] + j) + n_[1] * (h0_[2] + k));
    PetscCall(AOApplicationToPetsc(iga->ao, nh, idx.data()));
    IS from;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nh, idx.data(), PETSC_COPY_VALUES, &from));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nh, &local_));
    PetscCall(VecScatterCreate(templ, from, local_, NULL, &halo_));
    PetscCall(ISDestroy(&from));
    t1_.resize((size_t)ownw_[0] * hw_[1] * hw_[2]);
    t2_.resize((size_t)ownw_[0] * ownw_[1] * hw_[2]);
    ready_ = true;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZTensorOps::Load(Vec x)
{
    PetscFunctionBeginUser;
    PetscObjectState st;
    PetscCall(PetscObjectStateGet((PetscObject)x, &st));
    if (x == loaded_ && st == loadedState_) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(VecScatterBegin(halo_, x, local_, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(halo_, x, local_, INSERT_VALUES, SCATTER_FORWARD));
    loaded_ = x; loadedState_ = st;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// y (+)= c (A_z (x) A_y (x) A_x) x_loaded on the owned box: three 1D band
/// contractions, x then y then z.
void TDSEZTensorOps::Term(const Band *A[3], PetscScalar c, bool accumulate, PetscScalar *y)
{
    const PetscScalar *X;
    VecGetArrayRead(local_, &X);
    const PetscInt o0 = ownw_[0], o1 = ownw_[1], o2 = ownw_[2], g1 = hw_[1], g2 = hw_[2], g0 = hw_[0];
    // T1[i, j', k'] = sum_i' A_x(I, I') X[i', j', k']
    {
        const PetscInt p = p_[0], bw = bw_[0], n = n_[0];
        const PetscReal *a = A[0]->v.data();
        for (PetscInt r = 0; r < g1 * g2; ++r) {
            const PetscScalar *xr = X + (size_t)r * g0;
            PetscScalar *tr = t1_.data() + (size_t)r * o0;
            for (PetscInt i = 0; i < o0; ++i) {
                const PetscInt I = own0_[0] + i, lo = std::max<PetscInt>(0, I - p), hi = std::min(n - 1, I + p);
                const PetscReal *ar = a + (size_t)I * bw + (p - I);
                PetscScalar s = 0;
                for (PetscInt J = lo; J <= hi; ++J) s += ar[J] * xr[J - h0_[0]];
                tr[i] = s;
            }
        }
    }
    // T2[i, j, k'] = sum_j' A_y(J, J') T1[i, j', k']
    {
        const PetscInt p = p_[1], bw = bw_[1], n = n_[1];
        const PetscReal *a = A[1]->v.data();
        for (PetscInt k = 0; k < g2; ++k)
            for (PetscInt j = 0; j < o1; ++j) {
                const PetscInt Jg = own0_[1] + j, lo = std::max<PetscInt>(0, Jg - p), hi = std::min(n - 1, Jg + p);
                const PetscReal *ar = a + (size_t)Jg * bw + (p - Jg);
                PetscScalar *tr = t2_.data() + ((size_t)k * o1 + j) * o0;
                std::fill(tr, tr + o0, (PetscScalar)0.0);
                for (PetscInt J = lo; J <= hi; ++J) {
                    const PetscReal aj = ar[J];
                    const PetscScalar *src = t1_.data() + ((size_t)k * g1 + (J - h0_[1])) * o0;
                    for (PetscInt i = 0; i < o0; ++i) tr[i] += aj * src[i];
                }
            }
    }
    // y[i, j, k] (+)= c sum_k' A_z(K, K') T2[i, j, k']
    {
        const PetscInt p = p_[2], bw = bw_[2], n = n_[2];
        const PetscReal *a = A[2]->v.data();
        const size_t plane = (size_t)o0 * o1;
        for (PetscInt k = 0; k < o2; ++k) {
            const PetscInt Kg = own0_[2] + k, lo = std::max<PetscInt>(0, Kg - p), hi = std::min(n - 1, Kg + p);
            const PetscReal *ar = a + (size_t)Kg * bw + (p - Kg);
            PetscScalar *yr = y + (size_t)k * plane;
            if (!accumulate) std::fill(yr, yr + plane, (PetscScalar)0.0);
            for (PetscInt K = lo; K <= hi; ++K) {
                const PetscScalar ak = c * ar[K];
                const PetscScalar *src = t2_.data() + (size_t)(K - h0_[2]) * plane;
                for (size_t m = 0; m < plane; ++m) yr[m] += ak * src[m];
            }
        }
    }
    VecRestoreArrayRead(local_, &X);
}

PetscErrorCode TDSEZTensorOps::Apply(Op op, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    PetscCheck(ready_, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_WRONGSTATE, "tensor operators not set up");
    PetscCall(Load(x));
    PetscScalar *Y;
    PetscCall(VecGetArray(y, &Y));
    const Band *T[3] = {&M_[0], &M_[1], &M_[2]};
    switch (op) {
    case Mass: Term(T, 1.0, false, Y); break;
    case DipX: case DipY: case DipZ: {
        const int a = op - DipX;
        T[a] = &X_[a];
        Term(T, 1.0, false, Y);
        break;
    }
    case VelX: case VelY: case VelZ: {
        const int a = op - VelX;
        T[a] = &A_[a];
        Term(T, PetscCMPLX(0.0, -0.5 * invMass_), false, Y);   // -i/(2m) (G - G^T)
        break;
    }
    case Kin:
        for (PetscInt a = 0; a < dim_; ++a) {
            const Band *U[3] = {&M_[0], &M_[1], &M_[2]};
            U[a] = &S_[a];
            Term(U, 0.5 * invMass_, a > 0, Y);
        }
        break;
    default: break;
    }
    PetscCall(VecRestoreArray(y, &Y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

bool TDSEZTensorOps::Available(Op op) const
{
    if (!ready_) return false;
    if ((op == VelX || op == VelY || op == VelZ || op == Kin) && invMass_ == 0.0) return false;
    if (op >= DipX && op <= DipZ && op - DipX >= dim_) return false;
    if (op >= VelX && op <= VelZ && op - VelX >= dim_) return false;
    return true;
}

PetscErrorCode TDSEZTensorOps::Validate(Op op, Mat A, Vec x, PetscReal *relerr)
{
    PetscFunctionBeginUser;
    *relerr = -1;
    usable_[op] = false;
    if (!A || !Available(op)) PetscFunctionReturn(PETSC_SUCCESS);
    Vec y1, y2;
    PetscCall(VecDuplicate(x, &y1));
    PetscCall(VecDuplicate(x, &y2));
    PetscCall(Apply(op, x, y1));
    PetscCall(MatMult(A, x, y2));
    PetscReal ref, diff;
    PetscCall(VecNorm(y2, NORM_INFINITY, &ref));
    PetscCall(VecAXPY(y1, -1.0, y2));
    PetscCall(VecNorm(y1, NORM_INFINITY, &diff));
    PetscCall(VecDestroy(&y1));
    PetscCall(VecDestroy(&y2));
    *relerr = diff / PetscMax(ref, (PetscReal)1e-300);
    usable_[op] = *relerr <= 1e-10;
    PetscFunctionReturn(PETSC_SUCCESS);
}
