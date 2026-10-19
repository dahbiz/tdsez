// Node-balanced partition of the IGA.
//
// PetIGA splits the elements of each axis evenly over the processor grid
// (IGA_Dist1D) and derives the owned nodes from the element spans: part r owns
// [span[s_r] - p, span[s_{r+1}] - p), and the last part also gets the p
// trailing basis functions. An axis has nnp = nel + p functions, so the last
// part owns p more rows than the others; in 3D the corner rank (the last on
// every axis) owns (n + p)^3 instead of n^3 rows. At p = 3-5 with ~20 elements
// per part this is 1.25-1.6 times the mean, in memory and in time per step,
// since every step waits for the largest rank.
//
// TDSEZIGASetUp runs PetIGA's set-up in its two stages and, in between, moves
// the element boundaries so that every part owns ~nnp/P nodes per axis; the
// geometry and node ranges are then recomputed exactly as PetIGA's stage 1 does
// from the element ranges, and stage 2 builds the maps and scatters from them.
// The processor grid itself is PetIGA's. -tdsez_balance 0 keeps PetIGA's split.

#include "tdsez_internal.hpp"
#include <algorithm>
#include <cmath>

PETSC_EXTERN PetscErrorCode IGASetUp_Stage1(IGA);
PETSC_EXTERN PetscErrorCode IGASetUp_Stage2(IGA);

PetscReal TDSEZBalanceOverlapWeight();

/// @brief Element starts s_0 = 0 < s_1 < ... < s_P = nel of an axis such that
///        part r owns about nnp/P basis functions (owned nodes start at
///        span[s_r] - p). Every part keeps at least one element.
void TDSEZBalancedElementStarts(PetscInt p, PetscInt nel, PetscInt nnp, const PetscInt *span, PetscInt P,
                                std::vector<PetscInt> &s)
{
    s.assign((size_t)P + 1, 0);
    s[(size_t)P] = nel;
    // Owned widths: interior parts w, the two end parts w + c p, so that
    // w_r + c (overlap of r) is equal for every part (an end part has the
    // ASM overlap / ghost layer on one side only). c = 0 balances the rows.
    const double c = TDSEZBalanceOverlapWeight();
    const double wEnd = P > 2 ? ((double)nnp - (double)(P - 2) * ((double)nnp - 2.0 * c * p) / (double)P) / 2.0
                              : (double)nnp / (double)P;
    const double wInt = P > 2 ? ((double)nnp - 2.0 * wEnd) / (double)(P - 2) : wEnd;
    for (PetscInt r = 1; r < P; ++r) {
        const double target = wEnd + (double)(r - 1) * wInt;
        const PetscInt lo = s[(size_t)r - 1] + 1, hi = nel - (P - r);     // keep >= 1 element per part
        // first element whose owned nodes start at or after the target
        PetscInt e = (PetscInt)(std::lower_bound(span + lo, span + hi + 1, (PetscInt)std::ceil(target) + p) - span);
        if (e > hi) e = hi;
        if (e > lo && std::fabs((double)(span[e - 1] - p) - target) <= std::fabs((double)(span[e] - p) - target)) --e;
        s[(size_t)r] = std::max(lo, std::min(e, hi));
    }
}

/// Weight c of the overlap in the balance (-tdsez_balance_overlap, default 0.5).
PetscReal TDSEZBalanceOverlapWeight()
{
    static PetscBool init = PETSC_FALSE;
    static PetscReal c = 0.5;
    if (!init) {
        PetscOptionsGetReal(NULL, NULL, "-tdsez_balance_overlap", &c, NULL);
        init = PETSC_TRUE;
    }
    return c;
}

PetscBool TDSEZBalanceEnabled()
{
    static PetscBool init = PETSC_FALSE, on = PETSC_TRUE;
    if (!init) {
        PetscOptionsGetBool(NULL, NULL, "-tdsez_balance", &on, NULL);
        init = PETSC_TRUE;
    }
    return on;
}

/// @brief Move the element ranges of stage 1 to the node-balanced split and
///        recompute the geometry and node ranges as PetIGA's stage 1 does.
static PetscErrorCode TDSEZRebalance(IGA iga)
{
    PetscFunctionBeginUser;
    if (iga->collocation) PetscFunctionReturn(PETSC_SUCCESS);
    for (PetscInt i = 0; i < iga->dim; ++i) {
        IGAAxis ax = iga->axis[i];
        const PetscInt P = iga->proc_sizes[i], r = iga->proc_ranks[i];
        if (P < 2 || ax->periodic) continue;
        std::vector<PetscInt> s;
        TDSEZBalancedElementStarts(ax->p, ax->nel, ax->nnp, ax->span, P, s);
        iga->elem_start[i] = s[(size_t)r];
        iga->elem_width[i] = s[(size_t)r + 1] - s[(size_t)r];

        // geometry and node partitioning: PetIGA IGASetUp_Stage1 (non-collocation)
        const PetscInt nel = iga->elem_sizes[i], p = ax->p, *span = ax->span;
        const PetscInt efirst = iga->elem_start[i], elast = efirst + iga->elem_width[i] - 1;
        const PetscInt lstart = span[efirst] - p;
        const PetscInt lend   = (elast < nel - 1) ? span[elast + 1] - p : span[elast] + 1;
        iga->geom_sizes[i]  = span[nel - 1] + 1;
        iga->geom_lstart[i] = lstart;
        iga->geom_lwidth[i] = lend - lstart;
        iga->geom_gstart[i] = span[efirst] - p;
        iga->geom_gwidth[i] = span[elast] + 1 - iga->geom_gstart[i];
        iga->node_shift[i]  = 0;
        iga->node_sizes[i]  = ax->nnp;
        iga->node_lstart[i] = iga->geom_lstart[i];
        iga->node_lwidth[i] = iga->geom_lwidth[i];
        iga->node_gstart[i] = iga->geom_gstart[i];
        iga->node_gwidth[i] = iga->geom_gwidth[i];
        if (r == P - 1) iga->node_lwidth[i] = iga->node_sizes[i] - iga->node_lstart[i];
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief IGASetUp with a node-balanced partition (same steps as PetIGA's
///        IGASetUp; -iga_view is not supported on this path).
PetscErrorCode TDSEZIGASetUp(IGA iga)
{
    PetscFunctionBeginUser;
    if (!TDSEZBalanceEnabled()) {
        PetscCall(IGASetUp(iga));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (iga->setup) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCheck(iga->dim >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call IGASetDim() first");
    iga->setup = PETSC_TRUE;
    PetscCall(IGASetUp_Stage1(iga));
    PetscCall(TDSEZRebalance(iga));
    PetscCall(IGASetUp_Stage2(iga));
    iga->setupstage = 3;
    if (iga->order < 0) {
        for (PetscInt i = 0; i < iga->dim; i++) iga->order = PetscMax(iga->order, iga->axis[i]->p);
        PetscCall(IGASetOrder(iga, iga->order));
    }
    for (PetscInt i = 0; i < 3; i++)
        if (!iga->collocation) {
            if (i >= iga->dim) PetscCall(IGARuleReset(iga->rule[i]));
            if (i >= iga->dim) PetscCall(IGARuleSetType(iga->rule[i], IGA_RULE_LEGENDRE));
            PetscCall(IGABasisInitQuadrature(iga->basis[i], iga->axis[i], iga->rule[i]));
        } else {
            PetscCall(IGARuleReset(iga->rule[i]));
            PetscCall(IGABasisInitCollocation(iga->basis[i], iga->axis[i]));
        }
    PetscCall(IGAElementInit(iga->iterator, iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}
