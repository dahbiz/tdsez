// Dry-run memory estimate for an input on a given number of MPI ranks.
//
//   tdsez.x -inp run.inp -tdsez_estimate_memory 400 [-tdsez_estimate_ranks_per_node 40]
//           [-tdsez_estimate_node_mem 184] [-tdsez_estimate_base_mb 250] [run options]
//
// Run on one rank, with the options of the production run (-tdsez_mat_free,
// -st_matmode, -pade_stage*_pc_type, ...). The knot vectors are those of the run
// (built by TDSEZCore), and the processor grid and per-rank node boxes are
// PetIGA's (IGA_Partition, element spans), so the rows and nonzeros of every
// rank are exact; nothing is assembled. Memory is the sum of the objects alive
// in each phase (eigensolve, operator assembly, propagation): AIJ matrices with
// the pattern of H, the overlapped ASM blocks and their ILU(0) factors, Krylov
// and subspace vectors, plus a fixed per-rank baseline (MPI, PETSc, IGA). The
// estimate is the maximum over ranks and phases; it is meant to be conservative
// (allocator overhead and freed-but-not-returned memory are not modelled).

#include <petiga.h>
#include <slepceps.h>
#include <algorithm>
#include <string>
#include <vector>
#include "tdsez_parser.hpp"
#include "tdsez_info.hpp"
#include "tdsez_quadobs.hpp"
#include "tdsez_internal.hpp"

PETSC_EXTERN PetscErrorCode IGA_Partition(PetscInt, PetscInt, PetscInt, const PetscInt[], PetscInt[], PetscInt[]);

namespace {

/// One axis: coupling range [lo_i, hi_i] of every basis function (functions
/// sharing a nonzero element), and the per-rank owned node ranges.
struct Axis1D {
    PetscInt p = 0, nel = 0, nnp = 0;
    std::vector<PetscInt> span, lo, hi;

    void Init(IGAAxis ax)
    {
        p = ax->p; nel = ax->nel; nnp = ax->nnp;
        span.assign(ax->span, ax->span + nel);
        lo.assign((size_t)nnp, nnp); hi.assign((size_t)nnp, -1);
        for (PetscInt e = 0; e < nel; ++e)
            for (PetscInt i = span[e] - p; i <= span[e]; ++i) {
                lo[i] = std::min(lo[i], span[e] - p);
                hi[i] = std::max(hi[i], span[e]);
            }
    }
    /// Owned nodes [a, b) of part r of n (PetIGA: element split, then spans).
    void Owned(PetscInt n, PetscInt r, PetscInt *a, PetscInt *b) const
    {
        PetscInt s, w;
        if (TDSEZBalanceEnabled() && n > 1) {          // the run's node-balanced split (partition.cpp)
            std::vector<PetscInt> st;
            TDSEZBalancedElementStarts(p, nel, nnp, span.data(), n, st);
            s = st[(size_t)r]; w = st[(size_t)r + 1] - s;
        } else {                                        // PetIGA's even element split
            w = nel / n + ((nel % n) > r);
            s = r * (nel / n) + (((nel % n) > r) ? r : (nel % n));
        }
        const PetscInt elast = s + w - 1;
        *a = span[s] - p;
        *b = (elast < nel - 1) ? span[elast + 1] - p : span[elast] + 1;
    }
    /// Nonzeros of rows [a, b) restricted to columns [c0, c1).
    double Nnz(PetscInt a, PetscInt b, PetscInt c0, PetscInt c1) const
    {
        double s = 0;
        for (PetscInt i = a; i < b; ++i) s += (double)std::max<PetscInt>(0, std::min(hi[i] + 1, c1) - std::max(lo[i], c0));
        return s;
    }
    /// Range reached from [a, b) in k graph steps (ASM overlap k).
    void Grow(PetscInt k, PetscInt *a, PetscInt *b) const
    {
        for (PetscInt s = 0; s < k; ++s) {
            PetscInt na = *a, nb = *b;
            for (PetscInt i = *a; i < *b; ++i) { na = std::min(na, lo[i]); nb = std::max(nb, hi[i] + 1); }
            *a = na; *b = nb;
        }
    }
};

/// Rows and nonzeros of one rank: owned block, ghosted columns, ASM blocks.
struct RankShape {
    double rows = 1, nnz = 1, ghostCols = 1;
    double subRows[3] = {1, 1, 1}, subNnz[3] = {1, 1, 1};   // ASM overlap 0, 1, 2
};

struct Bytes {
    double S = sizeof(PetscScalar), I = sizeof(PetscInt);
    double Mat(const RankShape &r) const      // MPIAIJ: values + column indices + row data + ghost vector
    { return r.nnz * (S + I) + r.rows * 4 * I + (r.ghostCols - r.rows) * (S + 2 * I); }
    double Block(const RankShape &r, int k) const   // ASM block (SeqAIJ) or its ILU(0) factor
    { return r.subNnz[k] * (S + I) + r.subRows[k] * 4 * I; }
    double Vecs(const RankShape &r, double n) const { return n * r.rows * S; }
    double Values(const RankShape &r) const  // matrix sharing H's index arrays (MAT_SHARE_NONZERO_PATTERN)
    { return r.nnz * S + r.rows * 2 * I; }
};

const char *MB(double b, char *buf) { snprintf(buf, 32, "%.0f MB", b / 1048576.0); return buf; }

} // namespace

/// @brief Print the per-rank memory estimate of this input on nranks ranks and
///        return; called before any assembly when -tdsez_estimate_memory is set.
PetscErrorCode TDSEZEstimateMemory(IGA iga, PetscInt nranks)
{
    PetscFunctionBeginUser;
    const PetscInt dim = (PetscInt)TDSEZParser::Dimension;
    PetscInt  rpn = 0;
    PetscReal nodeGB = 0, baseMB = 250;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-tdsez_estimate_ranks_per_node", &rpn, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-tdsez_estimate_node_mem", &nodeGB, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-tdsez_estimate_base_mb", &baseMB, NULL));

    Axis1D ax[3];
    PetscInt nel[3] = {1, 1, 1}, procs[3] = {0, 0, 0};
    for (PetscInt d = 0; d < dim; ++d) {
        IGAAxis a;
        PetscCall(IGAGetAxis(iga, d, &a));
        ax[d].Init(a);
        nel[d] = ax[d].nel;
    }
    PetscCall(IGA_Partition(nranks, 0, dim, nel, procs, NULL));
    for (PetscInt d = dim; d < 3; ++d) procs[d] = 1;

    // Per-axis, per-part quantities; a rank's values are products over axes.
    struct Part { double rows, nnz, ghost, subRows[3], subNnz[3]; };
    std::vector<Part> parts[3];
    double ndof = 1;
    for (PetscInt d = 0; d < dim; ++d) {
        ndof *= (double)ax[d].nnp;
        for (PetscInt r = 0; r < procs[d]; ++r) {
            Part pt;
            PetscInt a, b;
            ax[d].Owned(procs[d], r, &a, &b);
            pt.rows = (double)(b - a);
            pt.nnz  = ax[d].Nnz(a, b, 0, ax[d].nnp);
            PetscInt ga = a, gb = b;
            ax[d].Grow(1, &ga, &gb);
            pt.ghost = (double)(gb - ga);
            for (int k = 0; k < 3; ++k) {
                PetscInt sa = a, sb = b;
                ax[d].Grow(k, &sa, &sb);
                pt.subRows[k] = (double)(sb - sa);
                pt.subNnz[k]  = ax[d].Nnz(sa, sb, sa, sb);
            }
            parts[d].push_back(pt);
        }
    }

    // ── Objects alive per phase, from the run options ─────────────────────
    const Bytes B;
    PetscBool inplace = PETSC_FALSE, matFree = TDSEZMatrixFree() ? PETSC_TRUE : PETSC_FALSE;
    {
        char mode[32] = "";
        PetscBool set = PETSC_FALSE;
        PetscCall(PetscOptionsGetString(NULL, NULL, "-st_matmode", mode, sizeof(mode), &set));
        inplace = (set && !strcmp(mode, "inplace")) ? PETSC_TRUE : PETSC_FALSE;
    }
    // stage preconditioner: ASM overlap 1 (default) or block Jacobi (no overlap)
    auto stagePCOverlap = [&](const char *prefix, int *ov, bool *none) -> PetscErrorCode {
        PetscFunctionBeginUser;
        char type[32] = "asm";
        PetscInt o = 1;
        PetscBool set;
        std::string key = std::string("-") + prefix + "pc_type";
        PetscCall(PetscOptionsGetString(NULL, NULL, key.c_str(), type, sizeof(type), &set));
        key = std::string("-") + prefix + "pc_asm_overlap";
        PetscCall(PetscOptionsGetInt(NULL, NULL, key.c_str(), &o, &set));
        *none = !strcmp(type, "jacobi") || !strcmp(type, "none");
        *ov = !strcmp(type, "bjacobi") ? 0 : (int)PetscMin(o, (PetscInt)2);
        PetscFunctionReturn(PETSC_SUCCESS);
    };

    const std::string prop = TDSEZParser::Propagator;
    const bool rational = (prop == "pade22" || prop == "magnus4");
    const bool propagate = TDSEZParser::EnablePropagation;
    PetscInt nCoupled = 0, nObserved = 0;
    for (int d = 0; d < dim; ++d) { nCoupled += TDSEZParser::Coupled(d); nObserved += TDSEZParser::Observed(d); }
    const PetscInt nTerms = TDSEZParser::NCouplingTerms();
    const PetscInt nev = TDSEZParser::NBoundStates, ncv = PetscMax(2 * nev, nev + 15);
    const bool directEps = TDSEZParser::UseDirectSolve;

    // operator matrices alive during propagation (besides H and M)
    PetscInt opsAssembled, opsProp;
    if (dim == 1)      { opsAssembled = 6; opsProp = 6; }
    else if (matFree)  { opsAssembled = nCoupled + (TDSEZParser::MassIsConstant ? 0 : nObserved); opsProp = opsAssembled; }
    else {
        // all axis operators are assembled; once the factorised and quadrature
        // monitors are set up the manager frees K, Md, V, dV, the velocities
        // (constant mass) and the uncoupled dipoles
        opsAssembled = 3 + 3 * nObserved;
        opsProp = nCoupled + (TDSEZParser::MassIsConstant ? 0 : nObserved);
    }
    opsAssembled += nTerms; opsProp += nTerms;
    const PetscInt capMats = TDSEZParser::EnableCAP ? 1 : 0;   // merged into H and destroyed at propagation

    int stOv[2] = {1, 1};
    bool stNone[2] = {false, false};
    if (rational) {
        PetscCall(stagePCOverlap("pade_stage1_", &stOv[0], &stNone[0]));
        PetscCall(stagePCOverlap("pade_stage2_", &stOv[1], &stNone[1]));
    } else {
        PetscCall(stagePCOverlap("", &stOv[0], &stNone[0]));
    }
    const PetscInt nStage = rational ? 2 : 1;
    // Astage_[2] (+ the cached bases Base_[2] unless -tdsez_mat_free selects the
    // lean stage update), or the Crank-Nicolson Jacobian
    const PetscInt nStageMats = rational ? (matFree ? 2 : 4) : 1;
    const PetscInt gmresVecs = (ndof > 100000 ? 50 : 30) + 4;  // restart + work, per stage solver

    struct Phase { const char *name; double bytes; };
    auto rankPhases = [&](const RankShape &r, std::vector<Phase> &ph) {
        const double base = baseMB * 1048576.0, mat = B.Mat(r);
        ph.clear();
        // eigensolve: H, M (+ shifted copy), ASM(1) block and ILU(0) (or LU, ~5x), Krylov subspace
        double eig = base + 2 * mat + (inplace ? 0 : mat) + B.Vecs(r, ncv + 110);
        eig += directEps ? 5 * B.Block(r, 0) : (TDSEZEpsILUInPlace() ? 1 : 2) * B.Block(r, 1);
        ph.push_back({"Eigensolve", eig});
        if (!propagate) return;
        // operator assembly: H, M, assembled operators, CAP, bound states
        ph.push_back({"Operator assembly", base + (2 + opsAssembled + capMats) * mat + B.Vecs(r, nev + 10)});
        // propagation: H (with CAP), M, operators, stage matrices, preconditioners, vectors
        // stage matrices (and cached bases) share H's index arrays; the CN Jacobian does not
        double pr = base + (2 + opsProp) * mat + nStageMats * (rational ? B.Values(r) : mat)
                  + B.Vecs(r, nev + 20 + nStage * gmresVecs);
        const bool inplace = rational && TDSEZStageILUInPlace();
        for (PetscInt k = 0; k < nStage; ++k)
            if (!stNone[k]) pr += ((stOv[k] == 0 || inplace) ? 1 : 2) * B.Block(r, stOv[k]);   // [ASM block copy +] ILU(0) factor
        ph.push_back({"Propagation", pr});
    };

    // ── Scan all ranks ─────────────────────────────────────────────────────
    std::vector<Phase> ph, worst, first;
    double peak = 0, rank0 = 0, sumPeak = 0, minPeak = 1e300, minRows = 1e300, maxRows = 0;
    PetscInt worstRank = 0;
    RankShape worstShape;
    for (PetscInt rk = 0; rk < procs[2]; ++rk)
        for (PetscInt rj = 0; rj < procs[1]; ++rj)
            for (PetscInt ri = 0; ri < procs[0]; ++ri) {
                const PetscInt idx[3] = {ri, rj, rk};
                RankShape r;
                for (PetscInt d = 0; d < dim; ++d) {
                    const Part &pt = parts[d][(size_t)idx[d]];
                    r.rows *= pt.rows; r.nnz *= pt.nnz; r.ghostCols *= pt.ghost;
                    for (int k = 0; k < 3; ++k) { r.subRows[k] *= pt.subRows[k]; r.subNnz[k] *= pt.subNnz[k]; }
                }
                rankPhases(r, ph);
                double m = 0;
                for (auto &x : ph) m = std::max(m, x.bytes);
                const PetscInt rank = ri + procs[0] * (rj + procs[1] * rk);
                if (rank == 0) { rank0 = m; first = ph; }
                sumPeak += m; minPeak = std::min(minPeak, m);
                minRows = std::min(minRows, r.rows); maxRows = std::max(maxRows, r.rows);
                if (m > peak) { peak = m; worst = ph; worstRank = rank; worstShape = r; }
            }

    // ── Report ─────────────────────────────────────────────────────────────
    TDSEZInfo info;
    char b1[32], b2[32], b3[32];
    info.header("MEMORY ESTIMATE (dry run, nothing assembled)");
    info.section("LAYOUT");
    info.kv("Degrees of freedom", "%.0f", ndof);
    info.kv("Ranks, processor grid", "%" PetscInt_FMT " = %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT,
            nranks, procs[0], procs[1], procs[2]);
    info.kv("Largest rank", "rank %" PetscInt_FMT ": %.0f rows, %.0f nonzeros (%.0f per row)", worstRank,
            worstShape.rows, worstShape.nnz, worstShape.nnz / worstShape.rows);
    info.kv("ASM blocks of that rank", "%.2fx (overlap 1), %.2fx (overlap 2) the owned nonzeros",
            worstShape.subNnz[1] / worstShape.nnz, worstShape.subNnz[2] / worstShape.nnz);
    info.kv("One matrix (pattern of H)", "%s per rank", MB(B.Mat(worstShape), b1));
    info.section("PHASES (largest rank)");
    info.kv("Partition", "%s", TDSEZBalanceEnabled() ? "node-balanced (-tdsez_balance)" : "PetIGA element split");
    info.kv("Settings", "%s, %s, stage PC %s%s, ST %s, baseline %.0f MB", prop.c_str(),
            matFree ? "matrix-free monitors" : "assembled monitors",
            stNone[0] ? "none" : (stOv[0] == 0 ? "block Jacobi/ILU(0)" : (stOv[0] == 1 ? "ASM(1)/ILU(0)" : "ASM(2)/ILU(0)")),
            (rational && TDSEZStageILUInPlace()) ? " in place" : "", inplace ? "inplace" : "copy", (double)baseMB);
    for (size_t i = 0; i < worst.size(); ++i) info.kv(worst[i].name, "%s (rank 0: %s)", MB(worst[i].bytes, b1), MB(first[i].bytes, b2));
    info.kv("Peak per rank", "%s (rank 0: %s)", MB(peak, b2), MB(rank0, b3));
    info.kv("Balance over ranks", "rows %.0f-%.0f (mean %.0f), peak %s-%s (mean %s)", minRows, maxRows, ndof / (double)nranks,
            MB(minPeak, b1), MB(peak, b2), MB(sumPeak / (double)nranks, b3));
    info.kv("Peak, all ranks", "%.1f GB", peak * (double)nranks / 1073741824.0);
    if (rpn > 0) {
        const double node = peak * (double)rpn / 1073741824.0;
        if (nodeGB > 0) {
            const bool fits = node <= 0.9 * nodeGB;
            info.kv("Per node", "%.1f GB of %.0f GB with %" PetscInt_FMT " ranks per node: %s", node, (double)nodeGB, rpn,
                    fits ? "fits (<= 90%)" : "DOES NOT FIT");
            if (!fits) info.warn("Options, least costly first: -tdsez_stage_ilu_inplace (Pade/Magnus, about 1/4 less "
                                 "propagation memory, ~20%% slower stage solves); -tdsez_mat_free; more nodes; a lower "
                                 "degree or coarser grid; -pade_stage1_pc_type bjacobi -pade_stage2_pc_type bjacobi "
                                 "(much slower).");
        } else {
            info.kv("Per node", "%.1f GB with %" PetscInt_FMT " ranks per node", node, rpn);
        }
    }
    info.footer();
    PetscFunctionReturn(PETSC_SUCCESS);
}
