// ============================================================================
//  band.cpp  —  banded accumulation of element matrices before insertion
// ----------------------------------------------------------------------------
//  Element-by-element insertion (MatSetValuesLocal per element) adds every
//  global entry once for each element sharing it: (p+1)^{2d} values per
//  element, i.e. up to (p+1)^{2d} / (2p+1)^d insertions per stored entry
//  (78 at p = 7 in 3D), each a search in a row of (2p+1)^d columns plus, for rows owned by
//  another rank, a copy into the communication stash. At high degree this,
//  not the integrals, dominates the assembly.
//
//  On a tensor-product B-spline space, local basis function i = (i1,i2,i3)
//  couples only to j with |j_d - i_d| <= p_d. Each rank therefore sums its
//  element matrices into a dense band over its ghosted node box,
//
//      B[r][ (j1-i1+p1) + w1 ((j2-i2+p2) + w2 (j3-i3+p3)) ],  w_d = 2 p_d + 1,
//
//  (r = ghosted local index of i, PetIGA's numbering) and then inserts each
//  row once with MatSetValuesLocal. Off-rank rows are still communicated by
//  PETSc, but once per entry instead of once per element contribution. The
//  matrices are the same sums of the same element matrices, in a different
//  order: identical up to floating-point rounding.
//
//  Memory: (ghosted nodes) x prod_d (2 p_d + 1) scalars per matrix, about the
//  size of the assembled matrix. Matrices are accumulated in groups that fit
//  in -tdsez_assembly_buffer_mb (default 1024 MB per rank, at least one
//  matrix per group); each group is one element pass.
//
//  Options: -tdsez_band 0               element-by-element insertion
//           -tdsez_assembly_buffer_mb N  buffer budget per rank (MB)
// ============================================================================

#include "tdsez_internal.hpp"
#include <algorithm>

namespace {

PetscBool BandEnabled()
{
    static PetscBool init = PETSC_FALSE, on = PETSC_TRUE;
    if (!init) {
        PetscOptionsGetBool(NULL, NULL, "-tdsez_band", &on, NULL);
        init = PETSC_TRUE;
    }
    return on;
}

PetscReal BandBudgetMB()
{
    static PetscBool init = PETSC_FALSE;
    static PetscReal mb = 1024.0;
    if (!init) {
        PetscOptionsGetReal(NULL, NULL, "-tdsez_assembly_buffer_mb", &mb, NULL);
        init = PETSC_TRUE;
    }
    return mb;
}

} // namespace

PetscErrorCode TDSEZBandAssembler::Setup(IGA iga, PetscInt nmat_)
{
    PetscFunctionBeginUser;
    nmat = nmat_;
    active = false;
    groups.clear();
    const unsigned all = (nmat >= 32) ? ~0u : ((1u << nmat) - 1u);

    bool ok = BandEnabled() && iga->dof == 1 && !iga->collocation &&
              iga->dim >= 1 && iga->dim <= 3 && nmat >= 1 && nmat <= 12;
    if (ok) {
        dim = iga->dim;
        nrows = 1; nband = 1;
        for (int d = 0; d < 3; ++d) {
            if (d < dim) {
                gstart[d] = iga->node_gstart[d];
                gwidth[d] = iga->node_gwidth[d];
                half[d]   = iga->basis[d]->nen - 1;   // element functions are contiguous
            } else {
                gstart[d] = 0; gwidth[d] = 1; half[d] = 0;
            }
            bw[d]  = 2 * half[d] + 1;
            nrows *= (size_t)gwidth[d];
            nband *= (size_t)bw[d];
        }
    }
    if (ok) {
        const PetscReal mbPerMat = (PetscReal)(nrows * nband * sizeof(PetscScalar)) / 1048576.0;
        PetscInt per = (PetscInt)(BandBudgetMB() / PetscMax(mbPerMat, 1e-9));
        per = PetscMax(1, PetscMin(per, nmat));
        for (PetscInt k0 = 0; k0 < nmat; k0 += per) {
            unsigned m = 0;
            for (PetscInt k = k0; k < PetscMin(nmat, k0 + per); ++k) m |= 1u << k;
            groups.push_back(m);
        }
        perGroup = per;
        active = true;
        verified = false;
    } else {
        groups.push_back(all);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZBandAssembler::BeginGroup(unsigned mask)
{
    PetscFunctionBeginUser;
    current = mask;
    slot.assign(12, -1);
    if (!active) PetscFunctionReturn(PETSC_SUCCESS);
    PetscInt ns = 0;
    for (PetscInt k = 0; k < nmat; ++k) if ((mask >> k) & 1u) slot[k] = ns++;
    // assign() on an already-sized vector reuses its storage between groups
    buf.assign((size_t)ns * nrows * nband, 0.0);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZBandAssembler::Add(IGAElement e, PetscInt k, const PetscScalar *K, Mat mat)
{
    PetscFunctionBeginUser;
    if (!active) {
        PetscCall(IGAElementAssembleMat(e, K, mat));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    const PetscInt n1 = e->parent->basis[0]->nen;
    const PetscInt n2 = dim > 1 ? e->parent->basis[1]->nen : 1;
    const PetscInt n3 = dim > 2 ? e->parent->basis[2]->nen : 1;
    PetscInt o[3] = {0, 0, 0};
    for (int d = 0; d < dim; ++d) o[d] = e->parent->basis[d]->offset[e->ID[d]] - gstart[d];

    if (!verified) {
        // The row/column numbering below must be PetIGA's ghosted local
        // numbering (element->mapping); check it on the first element.
        PetscInt nen_e; const PetscInt *map = NULL;
        PetscCall(IGAElementGetClosure(e, &nen_e, &map));
        bool same = (nen_e == n1 * n2 * n3);
        for (PetscInt a3 = 0; same && a3 < n3; ++a3)
        for (PetscInt a2 = 0; same && a2 < n2; ++a2)
        for (PetscInt a1 = 0; same && a1 < n1; ++a1) {
            const PetscInt a = a1 + n1 * (a2 + n2 * a3);
            const PetscInt r = (o[0] + a1) + gwidth[0] * ((o[1] + a2) + gwidth[1] * (o[2] + a3));
            if (map[a] != r) same = false;
        }
        verified = true;
        if (!same) {   // nothing accumulated yet: fall back to element insertion
            PetscCall(PetscInfo(NULL, "Banded assembly disabled: local numbering differs from PetIGA's\n"));
            active = false;
            Release();
            PetscCall(IGAElementAssembleMat(e, K, mat));
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    }

    const PetscInt nen = n1 * n2 * n3;
    PetscScalar *B = buf.data() + (size_t)slot[k] * nrows * nband;
    const size_t s2 = (size_t)bw[0], s3 = (size_t)bw[0] * bw[1];
    for (PetscInt a3 = 0; a3 < n3; ++a3)
    for (PetscInt a2 = 0; a2 < n2; ++a2)
    for (PetscInt a1 = 0; a1 < n1; ++a1) {
        const PetscInt a = a1 + n1 * (a2 + n2 * a3);
        const size_t r = (size_t)(o[0] + a1) + (size_t)gwidth[0] * ((size_t)(o[1] + a2) + (size_t)gwidth[1] * (size_t)(o[2] + a3));
        PetscScalar *row = B + r * nband;
        const PetscScalar *Ka = K + (size_t)a * nen;
        for (PetscInt b3 = 0; b3 < n3; ++b3)
        for (PetscInt b2 = 0; b2 < n2; ++b2) {
            // band offset of (b1 = 0, b2, b3) relative to row a
            PetscScalar *dst = row + (size_t)(half[0] - a1)
                                   + s2 * (size_t)(b2 - a2 + half[1])
                                   + s3 * (size_t)(b3 - a3 + half[2]);
            const PetscScalar *src = Ka + (size_t)n1 * (b2 + (size_t)n2 * b3);
            for (PetscInt b1 = 0; b1 < n1; ++b1) dst[b1] += src[b1];
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZBandAssembler::EndGroup(Mat mats[])
{
    PetscFunctionBeginUser;
    if (!active) PetscFunctionReturn(PETSC_SUCCESS);
    std::vector<PetscInt>    cols(nband);
    std::vector<PetscScalar> vals(nband);
    for (PetscInt k = 0; k < nmat; ++k) {
        if (slot[k] < 0) continue;
        const PetscScalar *B = buf.data() + (size_t)slot[k] * nrows * nband;
        for (PetscInt i3 = 0; i3 < gwidth[2]; ++i3)
        for (PetscInt i2 = 0; i2 < gwidth[1]; ++i2)
        for (PetscInt i1 = 0; i1 < gwidth[0]; ++i1) {
            const PetscInt r = i1 + gwidth[0] * (i2 + gwidth[1] * i3);
            const PetscScalar *row = B + (size_t)r * nband;
            PetscInt nc = 0;
            for (PetscInt d3 = -half[2]; d3 <= half[2]; ++d3) {
                const PetscInt j3 = i3 + d3;
                if (j3 < 0 || j3 >= gwidth[2]) continue;
                for (PetscInt d2 = -half[1]; d2 <= half[1]; ++d2) {
                    const PetscInt j2 = i2 + d2;
                    if (j2 < 0 || j2 >= gwidth[1]) continue;
                    const size_t base = (size_t)(d2 + half[1]) * bw[0] + (size_t)(d3 + half[2]) * bw[0] * bw[1];
                    const PetscInt cbase = gwidth[0] * (j2 + gwidth[1] * j3);
                    const PetscInt lo = PetscMax(-half[0], -i1), hi = PetscMin(half[0], gwidth[0] - 1 - i1);
                    for (PetscInt d1 = lo; d1 <= hi; ++d1) {
                        const PetscScalar v = row[base + (size_t)(d1 + half[0])];
                        if (v == (PetscScalar)0.0) continue;   // untouched or exactly zero
                        cols[nc] = cbase + i1 + d1;
                        vals[nc] = v;
                        ++nc;
                    }
                }
            }
            if (nc) PetscCall(MatSetValuesLocal(mats[k], 1, &r, nc, cols.data(), vals.data(), ADD_VALUES));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

void TDSEZBandAssembler::Release()
{
    std::vector<PetscScalar>().swap(buf);
}
