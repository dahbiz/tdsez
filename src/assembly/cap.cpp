// ============================================================================
//  Separable assembly of the complex absorbing potential
// ============================================================================
//  The CAP is a sum of one-axis profiles, W(r) = sum_d W_d(x_d) (TDSEZformCap),
//  scaled by 1/m for an (effective) mass m (TDSEZCapMassScale); with a
//  position-dependent mass this is separable in 1D only (2D/3D: element kernel),
//  and the quadrature is a tensor Gauss rule on a tensor B-spline basis. Its
//  matrix is therefore exactly
//
//      CAP = -i sum_d ( W_d (x) prod_{e != d} M_e ),
//
//  with the 1D matrices M_e = int B_i B_j and W_d = int W_d B_i B_j built from
//  PetIGA's own 1D quadrature tables (the same points and weights the element
//  loop uses). Rows are filled directly from the 1D bands: no 3D quadrature
//  and no element loop over the free core, where W = 0. Every entry of the
//  operator pattern is inserted (zeros included), so H += CAP keeps
//  SAME_NONZERO_PATTERN.
//
//  -tdsez_cap_check also assembles the CAP with the element kernel and prints
//  the largest difference (diagnostic).
// ============================================================================

#include "tdsez_internal.hpp"
#include <algorithm>
#include <cctype>
#include <vector>

namespace {

/// Value of the one-axis CAP profile, as in TDSEZformCap.
PetscReal AxisProfile(PetscReal u, PetscReal lo, PetscReal hi, PetscReal width)
{
    const PetscReal s = PetscMin(u - lo, hi - u);
    if (!(s < width)) return 0.0;
    return Manolopoulos_CAP_profile(PetscMin(width - s, width * (1.0 - 1e-10)), 0.0, width, TDSEZParser::CAPKmin);
}

} // namespace

PetscErrorCode TDSEZAssembleSeparableCAP(IGA iga, Mat *CAP)
{
    PetscFunctionBeginUser;
    const PetscInt dim = iga->dim;
    PetscCheck(!iga->geometry && !iga->rational && iga->dof == 1, PetscObjectComm((PetscObject)iga), PETSC_ERR_SUP,
               "separable CAP assembly needs scalar Cartesian B-splines");
    PetscCheck(dim == 1 || TDSEZParser::MassIsConstant, PetscObjectComm((PetscObject)iga), PETSC_ERR_SUP,
               "separable CAP assembly with a position-dependent mass needs 1D (1/m(r) is not separable)");
    const PetscReal lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
    const PetscReal hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
    const PetscReal width = TDSEZCapWidth();

    // 1D banded matrices: entry (i, j) at [i * bw + (j - i + p)], bw = 2p + 1
    PetscInt n[3] = {1, 1, 1}, p[3] = {0, 0, 0}, bw[3] = {1, 1, 1};
    std::vector<PetscReal> M1[3], W1[3];
    std::vector<char> T1[3];   // 1 where functions i, j share an element (C0 interface knots break the full band)
    for (PetscInt d = 0; d < 3; ++d) {
        if (d >= dim) { M1[d] = {1.0}; W1[d] = {0.0}; T1[d] = {1}; continue; }
        const IGABasis basis = iga->basis[d];
        n[d] = iga->axis[d]->nnp; p[d] = iga->axis[d]->p; bw[d] = 2 * p[d] + 1;
        M1[d].assign((size_t)n[d] * bw[d], 0.0);
        W1[d].assign((size_t)n[d] * bw[d], 0.0);
        T1[d].assign((size_t)n[d] * bw[d], 0);
        const PetscInt nen = basis->nen, nqp = basis->nqp;
        for (PetscInt e = 0; e < basis->nel; ++e) {
            const PetscReal J = basis->detJac[e];
            for (PetscInt a = 0; a < nen; ++a)       // coupled pairs: all functions of the element
                for (PetscInt b = 0; b < nen; ++b)
                    T1[d][(size_t)(basis->offset[e] + a) * bw[d] + (size_t)(b - a + p[d])] = 1;
            for (PetscInt q = 0; q < nqp; ++q) {
                const PetscReal w = basis->weight[e * nqp + q] * J;
                if (w == 0.0) continue;
                const PetscReal u = basis->point[e * nqp + q];
                PetscReal Wq = AxisProfile(u, lo[d], hi[d], width);
                if (Wq != 0.0) {                      // 1/m: 1D any mass, nD constant mass only
                    const PetscReal r[3] = {u, 0.0, 0.0};
                    Wq *= TDSEZCapMassScale(r);
                }
                const PetscReal *N = basis->value + (size_t)(e * nqp + q) * nen * 5;
                for (PetscInt a = 0; a < nen; ++a)
                    for (PetscInt b = 0; b < nen; ++b) {
                        const size_t k = (size_t)(basis->offset[e] + a) * bw[d] + (size_t)(b - a + p[d]);
                        const PetscReal v = w * N[a * 5] * N[b * 5];
                        M1[d][k] += v;
                        W1[d][k] += Wq * v;
                    }
            }
        }
    }

    // Dirichlet walls: those coefficients stay zero, so the CAP must not couple them
    std::string bt = TDSEZParser::BoundaryType;
    std::transform(bt.begin(), bt.end(), bt.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    const bool dirichlet = (bt == "dirichlet" || bt == "wall");
    auto onWall = [&](const PetscInt idx[3]) {
        if (!dirichlet) return false;
        for (PetscInt d = 0; d < dim; ++d) if (idx[d] == 0 || idx[d] == n[d] - 1) return true;
        return false;
    };

    PetscCall(IGACreateMat(iga, CAP));
    PetscInt w[3] = {1, 1, 1}, s[3] = {0, 0, 0};
    for (PetscInt d = 0; d < dim; ++d) { w[d] = iga->node_lwidth[d]; s[d] = iga->node_lstart[d]; }
    PetscInt rstart;
    PetscCall(PetscLayoutGetRange(iga->map, &rstart, NULL));
    // all owned rows first, then one (collective) ordering map, then insertion
    std::vector<PetscInt> cols, rowptr(1, 0);
    std::vector<PetscScalar> vals;
    PetscInt r = 0;
    for (PetscInt k = 0; k < w[2]; ++k)
        for (PetscInt j = 0; j < w[1]; ++j)
            for (PetscInt i = 0; i < w[0]; ++i, ++r) {
                const PetscInt I[3] = {s[0] + i, s[1] + j, s[2] + k};
                const bool rowWall = onWall(I);
                PetscInt c[3];
                for (c[2] = PetscMax(0, I[2] - p[2]); c[2] <= PetscMin(n[2] - 1, I[2] + p[2]); ++c[2])
                    for (c[1] = PetscMax(0, I[1] - p[1]); c[1] <= PetscMin(n[1] - 1, I[1] + p[1]); ++c[1])
                        for (c[0] = PetscMax(0, I[0] - p[0]); c[0] <= PetscMin(n[0] - 1, I[0] + p[0]); ++c[0]) {
                            PetscReal m[3], wv[3];
                            bool coupled = true;
                            for (PetscInt d = 0; d < 3; ++d) {
                                const size_t kk = (size_t)I[d] * bw[d] + (size_t)(c[d] - I[d] + p[d]);
                                m[d] = M1[d][kk]; wv[d] = W1[d][kk];
                                coupled = coupled && T1[d][kk];
                            }
                            if (!coupled) continue;          // not in the operator pattern (no shared element)
                            PetscReal v = wv[0] * m[1] * m[2];
                            if (dim > 1) v += m[0] * wv[1] * m[2];
                            if (dim > 2) v += m[0] * m[1] * wv[2];
                            if (rowWall || onWall(c)) v = 0.0;
                            cols.push_back(c[0] + n[0] * (c[1] + n[1] * c[2]));
                            vals.push_back(-PETSC_i * v);
                        }
                rowptr.push_back((PetscInt)cols.size());
            }
    PetscCall(AOApplicationToPetsc(iga->ao, (PetscInt)cols.size(), cols.data()));
    for (PetscInt q = 0; q < r; ++q) {
        const PetscInt row = rstart + q;
        PetscCall(MatSetValues(*CAP, 1, &row, rowptr[q + 1] - rowptr[q], cols.data() + rowptr[q],
                               vals.data() + rowptr[q], INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(*CAP, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*CAP, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(PETSC_SUCCESS);
}
