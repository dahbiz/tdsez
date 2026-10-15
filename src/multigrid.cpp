#include "tdsez_multigrid.hpp"
#include "tdsez_parser.hpp"
#include <algorithm>
#include <array>
#include <iterator>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

/**
 * @file multigrid.cpp
 * @brief Geometric B-spline multigrid preconditioner (see tdsez_multigrid.hpp).
 */

namespace {

using Knots = std::vector<PetscReal>;
/// Sparse row of a 1D prolongation: (coarse index, weight).
using Row = std::vector<std::pair<PetscInt, PetscReal>>;

/// Coarse knot vector: drop every other simple interior breakpoint. Returns
/// false when nothing can be dropped.
bool CoarsenKnots(const Knots &U, Knots &C)
{
    std::vector<std::pair<PetscReal, PetscInt>> bp;   // breakpoints with multiplicity
    for (PetscReal u : U) {
        if (!bp.empty() && u == bp.back().first) ++bp.back().second;
        else bp.push_back({u, 1});
    }
    C.clear();
    bool dropped = false;
    for (size_t i = 0; i < bp.size(); ++i) {
        const bool interior = i > 0 && i + 1 < bp.size();
        if (interior && bp[i].second == 1 && i % 2 == 1) { dropped = true; continue; }
        C.insert(C.end(), bp[i].second, bp[i].first);
    }
    return dropped;
}

/// Number of non-empty knot spans.
PetscInt Spans(const Knots &U)
{
    PetscInt n = 0;
    for (size_t i = 0; i + 1 < U.size(); ++i) if (U[i + 1] > U[i]) ++n;
    return n;
}

/// 1D prolongation (fine coefficients from coarse ones) by inserting the knots
/// of F missing from C one at a time (Boehm). Rows are sparse.
std::vector<Row> Prolongation(const Knots &C, const Knots &F, PetscInt p)
{
    Knots U = C;
    const PetscInt nc = (PetscInt)C.size() - p - 1;
    std::vector<Row> P(nc);
    for (PetscInt j = 0; j < nc; ++j) P[j] = {{j, 1.0}};
    // multiset difference F \ C (both sorted)
    Knots add;
    std::set_difference(F.begin(), F.end(), C.begin(), C.end(), std::back_inserter(add));
    for (PetscReal xi : add) {
        // span k: U[k] <= xi < U[k+1], p <= k <= n-1 (n = current #basis)
        const PetscInt n = (PetscInt)U.size() - p - 1;
        PetscInt k = (PetscInt)(std::upper_bound(U.begin(), U.end(), xi) - U.begin()) - 1;
        k = std::min(std::max(k, p), n - 1);
        std::vector<Row> Q(n + 1);
        for (PetscInt i = 0; i <= k - p; ++i) Q[i] = P[i];
        for (PetscInt i = k + 1; i <= n; ++i) Q[i] = P[i - 1];
        for (PetscInt i = k - p + 1; i <= k; ++i) {
            const PetscReal a = (xi - U[i]) / (U[i + p] - U[i]);
            std::map<PetscInt, PetscReal> m;
            for (auto &e : P[i]) m[e.first] += a * e.second;
            for (auto &e : P[i - 1]) m[e.first] += (1 - a) * e.second;
            Q[i].assign(m.begin(), m.end());
        }
        P.swap(Q);
        U.insert(std::upper_bound(U.begin(), U.end(), xi), xi);
    }
    for (auto &r : P) r.erase(std::remove_if(r.begin(), r.end(), [](auto &e) { return std::abs(e.second) < 1e-14; }), r.end());
    return P;
}

/// Greville abscissae.
std::vector<PetscReal> Greville(const Knots &U, PetscInt p)
{
    const PetscInt n = (PetscInt)U.size() - p - 1;
    std::vector<PetscReal> g(n, 0.0);
    for (PetscInt i = 0; i < n; ++i) {
        for (PetscInt j = 1; j <= p; ++j) g[i] += U[i + j];
        g[i] /= std::max<PetscInt>(p, 1);
    }
    return g;
}

/// Create an IGA with the given knots on the processor grid of @p fine (only
/// its ordering and partition are used).
PetscErrorCode CreateLevelIGA(IGA fine, const Knots *U, IGA *out)
{
    PetscFunctionBeginUser;
    IGA iga;
    PetscCall(IGACreate(PetscObjectComm((PetscObject)fine), &iga));
    PetscCall(IGASetDim(iga, fine->dim));
    PetscCall(IGASetDof(iga, 1));
    for (PetscInt d = 0; d < fine->dim; ++d) {
        IGAAxis axis;
        PetscCall(IGASetProcessors(iga, d, fine->proc_sizes[d]));
        PetscCall(IGAGetAxis(iga, d, &axis));
        PetscCall(IGAAxisSetDegree(axis, fine->axis[d]->p));
        PetscCall(IGAAxisSetKnots(axis, (PetscInt)U[d].size() - 1, U[d].data()));
    }
    PetscCall(IGASetUp(iga));
    *out = iga;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// Distributed prolongation from level @p c to level @p f (rows: fine PETSc
/// ordering, columns: coarse PETSc ordering).
PetscErrorCode AssembleProlongation(IGA f, IGA c, const std::vector<Row> P1[3], Mat *P)
{
    PetscFunctionBeginUser;
    const PetscInt dim = f->dim;
    const PetscInt *ls = f->node_lstart, *lw = f->node_lwidth, *Nc = c->node_sizes;
    PetscInt w[3] = {1, 1, 1}, s[3] = {0, 0, 0};
    for (PetscInt d = 0; d < dim; ++d) { w[d] = lw[d]; s[d] = ls[d]; }
    const PetscInt nrows = w[0] * w[1] * w[2];
    PetscInt rstart, cstart, cend, ncols;
    PetscCall(PetscLayoutGetRange(f->map, &rstart, NULL));
    PetscCall(PetscLayoutGetRange(c->map, &cstart, &cend));
    PetscCall(PetscLayoutGetLocalSize(c->map, &ncols));
    // rows in local order (x fastest); columns as coarse natural indices first
    std::vector<PetscInt> rowptr(nrows + 1, 0), cols;
    std::vector<PetscScalar> vals;
    static const Row one = {{0, 1.0}};
    PetscInt r = 0;
    for (PetscInt k = 0; k < w[2]; ++k)
        for (PetscInt j = 0; j < w[1]; ++j)
            for (PetscInt i = 0; i < w[0]; ++i, ++r) {
                const Row &rx = P1[0][s[0] + i];
                const Row &ry = dim > 1 ? P1[1][s[1] + j] : one;
                const Row &rz = dim > 2 ? P1[2][s[2] + k] : one;
                for (auto &ez : rz) for (auto &ey : ry) for (auto &ex : rx) {
                    cols.push_back(ex.first + Nc[0] * (ey.first + (dim > 1 ? Nc[1] : 1) * ez.first));
                    vals.push_back(ex.second * ey.second * ez.second);
                }
                rowptr[r + 1] = (PetscInt)cols.size();
            }
    PetscCall(AOApplicationToPetsc(c->ao, (PetscInt)cols.size(), cols.data()));
    std::vector<PetscInt> dnz(nrows, 0), onz(nrows, 0);
    for (PetscInt q = 0; q < nrows; ++q)
        for (PetscInt e = rowptr[q]; e < rowptr[q + 1]; ++e)
            (cols[e] >= cstart && cols[e] < cend ? dnz : onz)[q]++;
    PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)f), nrows, ncols, PETSC_DETERMINE, PETSC_DETERMINE,
                           0, dnz.data(), 0, onz.data(), P));
    for (PetscInt q = 0; q < nrows; ++q) {
        const PetscInt row = rstart + q;
        PetscCall(MatSetValues(*P, 1, &row, rowptr[q + 1] - rowptr[q], cols.data() + rowptr[q],
                               vals.data() + rowptr[q], INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// Check P on a linear function: coefficients at the Greville points of the
/// coarse level must map to those of the fine level (exact for nested spaces),
/// in the distributed orderings actually used.
PetscErrorCode CheckProlongation(IGA f, IGA c, Mat P, const Knots *Kf, const Knots *Kc, const PetscInt *p)
{
    PetscFunctionBeginUser;
    const PetscInt dim = f->dim;
    auto fill = [&](IGA g, const Knots *K, Vec v) -> PetscErrorCode {
        PetscFunctionBeginUser;
        std::vector<PetscReal> gr[3];
        for (PetscInt d = 0; d < 3; ++d) gr[d] = d < dim ? Greville(K[d], p[d]) : std::vector<PetscReal>{0.0};
        PetscInt w[3] = {1, 1, 1}, s[3] = {0, 0, 0};
        for (PetscInt d = 0; d < dim; ++d) { w[d] = g->node_lwidth[d]; s[d] = g->node_lstart[d]; }
        PetscScalar *a;
        PetscCall(VecGetArray(v, &a));
        PetscInt q = 0;
        for (PetscInt k = 0; k < w[2]; ++k)
            for (PetscInt j = 0; j < w[1]; ++j)
                for (PetscInt i = 0; i < w[0]; ++i, ++q)
                    a[q] = 1.0 + gr[0][s[0] + i] + 2.0 * gr[1][s[1] + j] + 3.0 * gr[2][s[2] + k];
        PetscCall(VecRestoreArray(v, &a));
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    Vec xc, xf, yf;
    PetscCall(MatCreateVecs(P, &xc, &xf));
    PetscCall(VecDuplicate(xf, &yf));
    PetscCall(fill(c, Kc, xc));
    PetscCall(fill(f, Kf, xf));
    PetscCall(MatMult(P, xc, yf));
    PetscCall(VecAXPY(yf, -1.0, xf));
    PetscReal err, scale;
    PetscCall(VecNorm(yf, NORM_INFINITY, &err));
    PetscCall(VecNorm(xf, NORM_INFINITY, &scale));
    PetscCall(VecDestroy(&xc)); PetscCall(VecDestroy(&xf)); PetscCall(VecDestroy(&yf));
    PetscCheck(err <= 1e-10 * scale, PetscObjectComm((PetscObject)P), PETSC_ERR_PLIB,
               "multigrid: distributed prolongation is not exact (error %g)", (double)(err / scale));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode TDSEZSetupMultigrid(IGA iga, PC pc, PetscInt *levels)
{
    PetscFunctionBeginUser;
    MPI_Comm comm = PetscObjectComm((PetscObject)pc);
    const PetscInt dim = iga->dim;
    PetscCheck(!iga->geometry && !iga->rational && iga->dof == 1, comm, PETSC_ERR_SUP,
               "Preconditioner = mg needs scalar non-rational B-splines");

    // Strong fields: the operator M + i tau (H + s E.r) becomes Helmholtz-like
    // once tau max|E.r| exceeds about 1.5, and the coarse levels no longer
    // represent it (2D tests: 3-18 iterations up to 1.5, divergence at 3).
    static bool checked = false;
    if (!checked) {
        checked = true;
        const PetscReal dt = TDSEZParser::TimeStep, T = TDSEZParser::FinalTime;
        const PetscReal tau = TDSEZParser::Propagator == "cn" ? 0.5 * dt
                            : TDSEZParser::Propagator == "pade22" ? dt / std::sqrt(12.0) : dt / (2 * std::sqrt(12.0));
        const PetscReal L[3] = {std::max(std::abs(TDSEZParser::LMinX), std::abs(TDSEZParser::LMaxX)),
                                std::max(std::abs(TDSEZParser::LMinY), std::abs(TDSEZParser::LMaxY)),
                                std::max(std::abs(TDSEZParser::LMinZ), std::abs(TDSEZParser::LMaxZ))};
        PetscReal worst = 0;
        const PetscInt n = (PetscInt)std::ceil(4 * T / dt);
        for (PetscInt i = 0; i <= n; ++i) {
            const PetscReal t = T * i / std::max<PetscInt>(n, 1);
            PetscReal v = 0;
            for (int a = 0; a < (int)dim; ++a) v += std::abs(TDSEZParser::Field(a, t)) * L[a];
            worst = std::max(worst, v);
        }
        if (tau * worst > 1.5)
            PetscCall(PetscPrintf(comm, "      WARNING: tau*max|E.r| = %.2g > 1.5: the field term dominates the propagator "
                                  "operator and multigrid may not converge. Use Preconditioner = asm or a smaller TimeStep.\n",
                                  (double)(tau * worst)));
    }

    // ---- knot hierarchy: finest first
    std::vector<std::array<Knots, 3>> K(1);
    PetscInt p[3] = {0, 0, 0};
    for (PetscInt d = 0; d < dim; ++d) {
        PetscInt m; PetscReal *U;
        PetscCall(IGAAxisGetKnots(iga->axis[d], &m, &U));
        K[0][d].assign(U, U + m + 1);
        p[d] = iga->axis[d]->p;
    }
    const PetscInt maxLevels = TDSEZParser::MGLevels > 0 ? TDSEZParser::MGLevels : 20;
    while ((PetscInt)K.size() < maxLevels) {
        std::array<Knots, 3> C = K.back();
        bool any = false;
        PetscInt dofs = 1;
        for (PetscInt d = 0; d < dim; ++d) {
            Knots Cd;
            // keep at least two spans per rank and p+1 spans on every axis
            const PetscInt minSpans = std::max<PetscInt>(2 * iga->proc_sizes[d], p[d] + 1);
            if (CoarsenKnots(K.back()[d], Cd) && Spans(Cd) >= minSpans) { C[d] = Cd; any = true; }
            dofs *= (PetscInt)C[d].size() - p[d] - 1;
        }
        if (!any) break;
        K.push_back(C);
        if (TDSEZParser::MGLevels == 0 && dofs < 2000) break;   // direct coarse solve is cheap
    }
    const PetscInt L = (PetscInt)K.size();
    *levels = L;
    if (L < 2) {
        PetscCall(PetscInfo(pc, "multigrid: the grid cannot be coarsened, using one level\n"));
    }

    // ---- level IGAs (coarse ones only for ordering and partition)
    std::vector<IGA> igas(L, nullptr);
    igas[0] = iga;
    for (PetscInt l = 1; l < L; ++l) PetscCall(CreateLevelIGA(iga, K[l].data(), &igas[l]));

    PetscCall(PCSetType(pc, PCMG));
    PetscCall(PCMGSetLevels(pc, L, NULL));
    PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
    // PCMG numbers levels 0 = coarsest ... L-1 = finest; K[0] is the finest.
    for (PetscInt l = 1; l < L; ++l) {
        std::vector<Row> P1[3];
        for (PetscInt d = 0; d < dim; ++d) {
            P1[d] = Prolongation(K[l][d], K[l - 1][d], p[d]);
            // exactness checks: partition of unity and linear precision
            const auto gc = Greville(K[l][d], p[d]), gf = Greville(K[l - 1][d], p[d]);
            PetscReal err = 0;
            for (size_t i = 0; i < P1[d].size(); ++i) {
                PetscReal s1 = 0, sx = 0;
                for (auto &e : P1[d][i]) { s1 += e.second; sx += e.second * gc[e.first]; }
                err = std::max({err, std::abs(s1 - 1), std::abs(sx - gf[i]) / (1 + std::abs(gf[i]))});
            }
            PetscCheck(P1[d].size() == gf.size() && err < 1e-10, comm, PETSC_ERR_PLIB,
                       "multigrid: knot-insertion prolongation is not exact (axis %d, error %g)", (int)d, (double)err);
        }
        Mat P;
        PetscCall(AssembleProlongation(igas[l - 1], igas[l], P1, &P));
        PetscCall(CheckProlongation(igas[l - 1], igas[l], P, K[l - 1].data(), K[l].data(), p));
        PetscCall(PCMGSetInterpolation(pc, L - l, P));
        PetscCall(MatDestroy(&P));
    }
    for (PetscInt l = 1; l < L; ++l) PetscCall(IGADestroy(&igas[l]));

    // ---- smoothers and coarse solve (options with the KSP prefix override)
    for (PetscInt l = 1; l < L; ++l) {
        KSP smoother; PC spc;
        PetscCall(PCMGGetSmoother(pc, l, &smoother));
        PetscCall(KSPSetType(smoother, KSPRICHARDSON));
        PetscCall(KSPSetTolerances(smoother, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, TDSEZParser::MGSmooth));
        PetscCall(KSPGetPC(smoother, &spc));
        // ASM with ILU(0) subdomains: Gauss-Seidel/SOR smoothing degrades with
        // the spline degree (14-26 iterations at p = 5 against 1-4 with ILU)
        PetscCall(PCSetType(spc, PCASM));
        PetscCall(PCASMSetType(spc, PC_ASM_RESTRICT));
    }
    {
        KSP coarse; PC cpc;
        PetscCall(PCMGGetCoarseSolve(pc, &coarse));
        PetscCall(KSPSetType(coarse, KSPPREONLY));
        PetscCall(KSPGetPC(coarse, &cpc));
        PetscCall(PCSetType(cpc, PCLU));
#if defined(PETSC_HAVE_MUMPS)
        PetscCall(PCFactorSetMatSolverType(cpc, MATSOLVERMUMPS));
#else
        PetscCall(PCSetType(cpc, PCREDUNDANT));
#endif
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
