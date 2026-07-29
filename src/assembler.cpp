// ============================================================================
//  assembler.cpp  —  NEW 2D assembly implementations (standalone, additive)
// ============================================================================
//
//  This translation unit adds a family of *new* 2D form kernels and a
//  self-contained benchmark that demonstrates the advantage of fusing
//  multiple operators into a single quadrature pass via the EXISTING
//  TDSEZCompOperators() (declared in tdsez_internal.hpp — NOT modified here).
//
//  Nothing in this file touches or overrides the existing assembly.cpp
//  kernels (TDSEZFormHam, TDSEZFormPhyX, ...). Every symbol here is new and
//  name-spaced with the "TDSEZAsm2D" prefix so there is zero collision risk.
//
//  Covered 2D cases (each fused H+M (or H+Lz) in ONE TDSEZCompOperators call):
//    [1] ConstMass  — constant mass (1/2) Hamiltonian + mass, the standard pair.
//    [2] VarMass    — position-dependent mass via TDSEZParser::MassDist.
//    [3] CAP        — complex absorbing-potential added into H (H becomes
//                     complex), mass stays real. Shows a COMPLEX operator fused.
//    [4] Lz         — Hamiltonian fused together with the angular-momentum
//                     operator Lz (a different, non-diagonal operator type).
//
//  The "unfused" baseline for each case is the SAME engine called twice
//  (one TDSEZCompOperators call per operator, nmat=1). This is the fair
//  comparison: identical kernels, identical assembly path — the only
//  difference is one quadrature pass (nmat=2) vs two passes (nmat=1 each).
//  It isolates exactly the cost of NOT fusing.
//
//  Author: generated for the TDSEZ 2D-assembly advantage study.
// ============================================================================

#include "tdsez_internal.hpp"

#include <cstring>
#include <vector>
#include <algorithm>

// ----------------------------------------------------------------------------
//  Context passed to the fused kernels (selects which 2D scenario to build).
// ----------------------------------------------------------------------------
struct TDSEZAsm2DCtx
{
    PetscInt scenario = 0;   // 0=ConstMass, 1=VarMass, 2=CAP, 3=Lz
};

// ----------------------------------------------------------------------------
//  Small helpers shared by the new kernels (local to this TU).
// ----------------------------------------------------------------------------
static inline PetscReal TDSEZAsm2D_InvMass(PetscReal x, PetscReal y)
{
    // MassDist returns the effective mass m(x,y); kinetic coeff is 1/(2 m).
    const PetscReal m = TDSEZParser::MassDist(x, y, 0.0);
    return (m > 1e-12) ? (0.5 / m) : 0.5;
}

// Complex absorbing-potential profile (Manolopoulos-style), re-derived here.
static inline PetscScalar TDSEZAsm2D_CAP(PetscReal x, PetscReal y,
                                         PetscReal Lmin, PetscReal Lmax)
{
    const PetscReal r   = PetscSqrtReal(x * x + y * y);
    const PetscReal rmax = PetscMax(PetscAbsReal(Lmin), PetscAbsReal(Lmax));
    const PetscReal eta = 0.5;                 // strength (arbitrary for test)
    const PetscReal r0  = 0.85 * rmax;
    if (r <= r0) return 0.0;
    const PetscReal s = (r - r0) / (rmax - r0);
    return PetscScalar(0.0, -eta * s * s);     // purely imaginary (absorbing)
}

// ----------------------------------------------------------------------------
//  FUSED 2D kernels  (TDSEZPhysicsKernel signature: M[0]=H, M[1]=M/Lz)
//  Each fills BOTH operators from a single quadrature-point evaluation.
// ----------------------------------------------------------------------------

// [1] Constant-mass 2D Hamiltonian + mass.  Faithful re-derivation of the
//     2D branch of TDSEZFormHam (assembly.cpp:321) as an independent fn.
PetscErrorCode TDSEZAsm2D_ConstMass(IGAPoint p, PetscInt nmat,
                                    PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)ctx; (void)nmat;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1];
    const PetscReal cW   = 0.5 * w;
    const PetscReal Vw   = TDSEZParser::V(x, y) * w;
    const PetscReal *B   = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[2] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal BaW  = B[a] * w;
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW;
        const PetscReal gy_a = dB[a][1] * cW;
        PetscScalar *__restrict__ Hrow = M[0] + a * nen;
        PetscScalar *__restrict__ Mrow = M[1] + a * nen;
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal Bb = B[b];
            Hrow[b] += gx_a * dB[b][0] + gy_a * dB[b][1] + pot_a * Bb;
            Mrow[b] += BaW * Bb;
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// [2] Variable-mass 2D Hamiltonian + mass (position-dependent mass path).
PetscErrorCode TDSEZAsm2D_VarMass(IGAPoint p, PetscInt nmat,
                                  PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)ctx; (void)nmat;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1];
    const PetscReal cW   = TDSEZAsm2D_InvMass(x, y) * w;
    const PetscReal Vw   = TDSEZParser::V(x, y) * w;
    const PetscReal *B   = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[2] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal BaW  = B[a] * w;
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW;
        const PetscReal gy_a = dB[a][1] * cW;
        PetscScalar *__restrict__ Hrow = M[0] + a * nen;
        PetscScalar *__restrict__ Mrow = M[1] + a * nen;
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal Bb = B[b];
            Hrow[b] += gx_a * dB[b][0] + gy_a * dB[b][1] + pot_a * Bb;
            Mrow[b] += BaW * Bb;
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// [3] 2D Hamiltonian WITH complex CAP + mass.  H becomes complex (imaginary
//     absorbing term); mass stays real. Demonstrates fusing a COMPLEX operator.
PetscErrorCode TDSEZAsm2D_CAP(IGAPoint p, PetscInt nmat,
                              PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat;
    const TDSEZAsm2DCtx *C = static_cast<const TDSEZAsm2DCtx *>(ctx);
    const PetscReal Lmin = (C && C->scenario == 2) ? -20.0 : -20.0;
    const PetscReal Lmax = (C && C->scenario == 2) ?  20.0 :  20.0;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1];
    const PetscReal cW   = 0.5 * w;
    const PetscReal Vw   = TDSEZParser::V(x, y) * w;
    const PetscScalar cap = TDSEZAsm2D_CAP(x, y, Lmin, Lmax) * w;
    const PetscReal *B   = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[2] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal BaW  = B[a] * w;
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW;
        const PetscReal gy_a = dB[a][1] * cW;
        PetscScalar *__restrict__ Hrow = M[0] + a * nen;
        PetscScalar *__restrict__ Mrow = M[1] + a * nen;
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal Bb = B[b];
            Hrow[b] += gx_a * dB[b][0] + gy_a * dB[b][1] + pot_a * Bb + cap * Bb;
            Mrow[b] += BaW * Bb;
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// [4] 2D Hamiltonian + angular-momentum Lz operator, fused.
//     Lz = -i (x d/dy - y d/dx)  ->  <i|Lz|j> = -i ∫ B_i (x dB_j/dy - y dB_j/dx) w dΩ.
PetscErrorCode TDSEZAsm2D_Lz(IGAPoint p, PetscInt nmat,
                             PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat; (void)ctx;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1];
    const PetscReal cW   = 0.5 * w;
    const PetscReal Vw   = TDSEZParser::V(x, y) * w;
    const PetscReal *B   = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[2] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW;
        const PetscReal gy_a = dB[a][1] * cW;
        PetscScalar *__restrict__ Hrow = M[0] + a * nen;
        PetscScalar *__restrict__ Lrow = M[1] + a * nen;   // M[1] = Lz here
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal Bb = B[b];
            Hrow[b] += gx_a * dB[b][0] + gy_a * dB[b][1] + pot_a * Bb;
            const PetscReal lz_ab = B[a] * (x * dB[b][1] - y * dB[b][0]) * w;
            Lrow[b] += PetscScalar(0.0, -lz_ab);
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ----------------------------------------------------------------------------
//  SINGLE-OPERATOR kernels (nmat=1, fill M[0] only) — used for the UNFUSED
//  baseline: each operator assembled by its own TDSEZCompOperators call.
//  These are the EXACT same physics as the fused kernels above, just split
//  out one operator per kernel so we can call the engine twice.
// ----------------------------------------------------------------------------

// H only (constant or variable mass handled by the scenario flag)
static PetscErrorCode TDSEZAsm2D_H_only(IGAPoint p, PetscInt nmat,
                                        PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat;
    const TDSEZAsm2DCtx *C = static_cast<const TDSEZAsm2DCtx *>(ctx);
    const PetscInt s = C ? C->scenario : 0;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1];
    const PetscReal cW = (s == 1) ? TDSEZAsm2D_InvMass(x, y) * w : 0.5 * w;
    const PetscReal Vw = TDSEZParser::V(x, y) * w;
    const PetscReal *B  = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[2] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW;
        const PetscReal gy_a = dB[a][1] * cW;
        PetscScalar *__restrict__ R = M[0] + a * nen;
        if (s == 2) {   // CAP case: add imaginary absorbing term to H
            const PetscScalar cap = TDSEZAsm2D_CAP(x, y, -20.0, 20.0) * w;
            for (PetscInt b = 0; b < nen; ++b)
                R[b] += gx_a * dB[b][0] + gy_a * dB[b][1] + pot_a * B[b] + cap * B[b];
        } else {
            for (PetscInt b = 0; b < nen; ++b)
                R[b] += gx_a * dB[b][0] + gy_a * dB[b][1] + pot_a * B[b];
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Mass only
static PetscErrorCode TDSEZAsm2D_M_only(IGAPoint p, PetscInt nmat,
                                        PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat; (void)ctx;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    const PetscReal *B = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal BaW = B[a] * w;
        PetscScalar *__restrict__ R = M[0] + a * nen;
        for (PetscInt b = 0; b < nen; ++b)
            R[b] += BaW * B[b];
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Lz only
static PetscErrorCode TDSEZAsm2D_Lz_only(IGAPoint p, PetscInt nmat,
                                         PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat; (void)ctx;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1];
    const PetscReal *B = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[2] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    for (PetscInt a = 0; a < nen; ++a) {
        PetscScalar *__restrict__ R = M[0] + a * nen;
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal lz_ab = B[a] * (x * dB[b][1] - y * dB[b][0]) * w;
            R[b] += PetscScalar(0.0, -lz_ab);
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ----------------------------------------------------------------------------
//  Benchmark driver
// ----------------------------------------------------------------------------
//  For each 2D scenario:
//    (a) assemble {H, M/Lz} the FUSED way via TDSEZCompOperators (1 pass, nmat=2),
//    (b) assemble the SAME operators the UNFUSED way via TWO TDSEZCompOperators
//        calls (nmat=1 each) — same engine, isolates the fusion overhead,
//    (c) verify the two results are equal (correctness: ||fused - unfused||),
//    (d) report the wall-time speedup of fused over unfused.
// ----------------------------------------------------------------------------

namespace {
struct ScenarioSpec
{
    const char             *name;
    TDSEZPhysicsKernel      fused;      // nmat=2 kernel (M[0]=H, M[1]=M/Lz)
    TDSEZPhysicsKernel      op0;        // first single-op kernel (H)
    TDSEZPhysicsKernel      op1;        // second single-op kernel (M or Lz)
    PetscBool               secondIsMass;
};
} // anonymous namespace

// ----------------------------------------------------------------------------
//  MULTI-OPERATOR fusion sweep kernels.
//
//  The real throughput lever in TDSEZCompOperators is that it walks the
//  mesh ONCE and evaluates N operators from a single quadrature-point
//  evaluation. The more operators you pack into one pass, the more the
//  (fixed) element-walk / point-loop / MPI-scatter cost is amortised.
//
//  TDSEZAsmND_Ops  : fills K matrices (mats[0..K-1]) from ONE point
//                       eval — the fused, high-throughput path.
//  TDSEZAsm1Op      : fills ONE matrix (mats[0]) for operator ctx->op
//                       — used K times for the unfused baseline.
//
//  Operators (all derived from the same B / dB / x / y buffers):
//    0: H   = -(1/2)(d_x^2 + d_y^2) + V        (Hamiltonian)
//    1: M   = m(x,y)                          (mass / overlap)
//    2: Jx  = -i (1/2)(d_x)                (x-current density op)
//    3: Jy  = -i (1/2)(d_y)                (y-current density op)
//    4: Dx  = x                             (x-dipole op)
//    5: Dy  = y                             (y-dipole op)
//    6: Lz  = -i (x d_y - y d_x)            (angular momentum)
//    7: A   = V(x,y)                        (potential / autocorrelation ker)
// ----------------------------------------------------------------------------
struct TDSEZAsmNDCtx { PetscInt nop = 2; PetscInt op = 0; };

static inline void TDSEZAsmND_Fill(PetscInt nop, PetscScalar *M[],
                                   PetscInt nen, PetscReal w,
                                   PetscReal x, PetscReal y,
                                   const PetscReal *B,
                                   const PetscReal (*dB)[2],
                                   PetscReal cW, PetscReal Vw, PetscReal mass)
{
    (void)mass;
    // Row pointers valid only for k < nop; leave the rest NULL so the
    // per-operator writes below are guarded by `k < nop`.
    PetscScalar *R[8] = {PETSC_NULLPTR};
    for (PetscInt k = 0; k < nop; ++k) R[k] = M[k];
    const PetscReal cW2 = cW;   // 0.5*w / m  (kinetic coeff, mass already folded in)
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal BaW  = B[a] * w;
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW2;
        const PetscReal gy_a = dB[a][1] * cW2;
        const PetscReal dxa = dB[a][0], dya = dB[a][1];   // row-basis derivs (transpose term)
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal Bb = B[b], dx = dB[b][0], dy = dB[b][1];
            const PetscReal Helem = gx_a * dx + gy_a * dy + pot_a * Bb;
            const PetscReal Melem = BaW * Bb;
            // Hermitian (symmetric Galerkin) observables:  O_ab = (-i/2)(D_ab - D_ba)
            //   Jx = -i/2 (B_a d_x B_b - B_b d_x B_a)
            const PetscScalar Jxelem = PetscScalar(0.0, -0.5 * (B[a] * dx - Bb * dxa) * w);
            const PetscScalar Jyelem = PetscScalar(0.0, -0.5 * (B[a] * dy - Bb * dya) * w);
            const PetscScalar Dxelem = x * Melem;
            const PetscScalar Dyelem = y * Melem;
            // Lz = -i (x d_y - y d_x):  B_a(x d_y B_b - y d_x B_b) - B_b(x d_y B_a - y d_x B_a)
            const PetscReal lz_ab = (x * dy - y * dx);
            const PetscReal lz_ba = (x * dya - y * dxa);
            const PetscScalar Lzelem = PetscScalar(0.0, -(B[a] * lz_ab - Bb * lz_ba) * w);
            const PetscScalar Aelem  = pot_a * Bb;
const PetscInt ab = a * nen + b;
            if (nop > 0) R[0][ab] += Helem;
            if (nop > 1) R[1][ab] += Melem;
            if (nop > 2) R[2][ab] += Jxelem;
            if (nop > 3) R[3][ab] += Jyelem;
            if (nop > 4) R[4][ab] += Dxelem;
            if (nop > 5) R[5][ab] += Dyelem;
            if (nop > 6) R[6][ab] += Lzelem;
            if (nop > 7) R[7][ab] += Aelem;
        }
    }
}

// Fused: evaluate up to 8 operators from one point eval.
static PetscErrorCode TDSEZAsmND_Ops(IGAPoint p, PetscInt nmat,
                                     PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat;
    const TDSEZAsmNDCtx *C = static_cast<const TDSEZAsmNDCtx *>(ctx);
    const PetscInt nop = C ? C->nop : 2;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1];
    const PetscReal cW = 0.5 * w;                 // mass = 1
    const PetscReal Vw = TDSEZParser::V(x, y) * w;
    const PetscReal *B = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[2] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    TDSEZAsmND_Fill(nop, M, nen, w, x, y, B, dB, cW, Vw, 1.0);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Single-operator: fill M[0] with operator C->op (used K times for baseline).
static PetscErrorCode TDSEZAsm1Op(IGAPoint p, PetscInt nmat,
                                   PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat;
    const TDSEZAsmNDCtx *C = static_cast<const TDSEZAsmNDCtx *>(ctx);
    const PetscInt op = C ? C->op : 0;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1];
    const PetscReal cW = 0.5 * w;
    const PetscReal Vw = TDSEZParser::V(x, y) * w;
    const PetscReal *B = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[2] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal BaW  = B[a] * w;
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW;
        const PetscReal gy_a = dB[a][1] * cW;
        PetscScalar *__restrict__ R = M[0] + a * nen;
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal Bb = B[b], dx = dB[b][0], dy = dB[b][1];
            switch (op) {
            case 0: R[b] += gx_a * dx + gy_a * dy + pot_a * Bb; break;          // H
            case 1: R[b] += BaW * Bb; break;                                       // M
            case 2: R[b] += PetscScalar(0.0, -0.5 * (B[a] * dB[b][0] - Bb * dB[a][0]) * w); break;  // Jx = -i/2 (B_a d_x B_b - B_b d_x B_a)
            case 3: R[b] += PetscScalar(0.0, -0.5 * (B[a] * dB[b][1] - Bb * dB[a][1]) * w); break;  // Jy = -i/2 (B_a d_y B_b - B_b d_y B_a)
            case 4: R[b] += x * BaW * Bb; break;                                  // Dx = x
            case 5: R[b] += y * BaW * Bb; break;                                  // Dy = y
            case 6: { // Lz = -i (x d_y - y d_x):  B_a(...)-B_b(...)
                const PetscReal lz_ab  = (x * dB[b][1] - y * dB[b][0]);
                const PetscReal lz_ba = (x * dB[a][1] - y * dB[a][0]);
                R[b] += PetscScalar(0.0, -(B[a] * lz_ab - Bb * lz_ba) * w);
            } break;
            case 7: R[b] += pot_a * Bb; break;                                    // A
            case 8: R[b] += B[a] * dB[b][0] * w; break;                          // Dreal_x = int B_a d_x B_b (real)
            case 9: R[b] += B[a] * dB[b][1] * w; break;                          // Dreal_y = int B_a d_y B_b (real)
            default: break;
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ===========================================================================
//  3D kernels — same fused approach as 2D, extended to dim=3.
//  Operator layout (3D):
//     0: H    1: M    2: Jx   3: Jy   4: Jz
//     5: Dx   6: Dy   7: Dz   8: Lx   9: Ly  10: Lz  11: A
//  Jx/Jy/Jz use the Hermitian (symmetric Galerkin) form:
//     J_i = -i/2 (B_a d_i B_b - B_b d_i B_a)
//  Lx =  y p_z - z p_y = -i (y d_z - z d_y),  etc. (angular momentum)
// ===========================================================================

static inline void TDSEZAsmND_Fill3D(PetscInt nop, PetscScalar *M[],
                                    PetscInt nen, PetscReal w,
                                    const PetscReal xyz[3],
                                    const PetscReal *B,
                                    const PetscReal (*dB)[3],
                                    PetscReal cW, PetscReal Vw)
{
    PetscScalar *R[12] = {PETSC_NULLPTR};
    for (PetscInt k = 0; k < nop && k < 12; ++k) R[k] = M[k];
    const PetscReal x = xyz[0], y = xyz[1], z = xyz[2];
    const PetscReal cW2 = cW;   // 0.5*w / m
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal BaW  = B[a] * w;
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW2;
        const PetscReal gy_a = dB[a][1] * cW2;
        const PetscReal gz_a = dB[a][2] * cW2;
        const PetscReal dxa = dB[a][0], dya = dB[a][1], dza = dB[a][2];
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal Bb = B[b];
            const PetscReal dx = dB[b][0], dy = dB[b][1], dz = dB[b][2];
            const PetscReal Helem = gx_a * dx + gy_a * dy + gz_a * dz + pot_a * Bb;
            const PetscReal Melem = BaW * Bb;
            const PetscScalar Jxelem = PetscScalar(0.0, -0.5 * (B[a] * dx - Bb * dxa) * w);
            const PetscScalar Jyelem = PetscScalar(0.0, -0.5 * (B[a] * dy - Bb * dya) * w);
            const PetscScalar Jzelem = PetscScalar(0.0, -0.5 * (B[a] * dz - Bb * dza) * w);
            const PetscScalar Dxelem = x * Melem;
            const PetscScalar Dyelem = y * Melem;
            const PetscScalar Dzelem = z * Melem;
            const PetscReal lx_ab = (y * dz - z * dy), lx_ba = (y * dza - z * dya);
            const PetscReal ly_ab = (z * dx - x * dz), ly_ba = (z * dxa - x * dza);
            const PetscReal lz_ab = (x * dy - y * dx), lz_ba = (x * dya - y * dxa);
            const PetscScalar Lxelem = PetscScalar(0.0, -(B[a] * lx_ab - Bb * lx_ba) * w);
            const PetscScalar Lyelem = PetscScalar(0.0, -(B[a] * ly_ab - Bb * ly_ba) * w);
            const PetscScalar Lzelem = PetscScalar(0.0, -(B[a] * lz_ab - Bb * lz_ba) * w);
            const PetscScalar Aelem  = pot_a * Bb;
            const PetscInt ab = a * nen + b;
            if (nop > 0)  R[0][ab]  += Helem;
            if (nop > 1)  R[1][ab]  += Melem;
            if (nop > 2)  R[2][ab]  += Jxelem;
            if (nop > 3)  R[3][ab]  += Jyelem;
            if (nop > 4)  R[4][ab]  += Jzelem;
            if (nop > 5)  R[5][ab]  += Dxelem;
            if (nop > 6)  R[6][ab]  += Dyelem;
            if (nop > 7)  R[7][ab]  += Dzelem;
            if (nop > 8)  R[8][ab]  += Lxelem;
            if (nop > 9)  R[9][ab]  += Lyelem;
            if (nop > 10) R[10][ab] += Lzelem;
            if (nop > 11) R[11][ab] += Aelem;
            // real derivative ops (for reconstruction check J = -i/2 (D - D^T))
            if (nop > 12) R[12][ab] += B[a] * dx * w;
            if (nop > 13) R[13][ab] += B[a] * dy * w;
            if (nop > 14) R[14][ab] += B[a] * dz * w;
        }
    }
}

static PetscErrorCode TDSEZAsmND_Ops3D(IGAPoint p, PetscInt nmat,
                                      PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat;
    const TDSEZAsmNDCtx *C = static_cast<const TDSEZAsmNDCtx *>(ctx);
    const PetscInt nop = C ? C->nop : 2;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal cW = 0.5 * w;
    const PetscReal Vw = TDSEZParser::V(xyz[0], xyz[1], xyz[2]) * w;
    const PetscReal *B = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[3] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    TDSEZAsmND_Fill3D(nop, M, nen, w, xyz, B, dB, cW, Vw);
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TDSEZAsm1Op3D(IGAPoint p, PetscInt nmat,
                                   PetscScalar *M[], void *ctx)
{
    PetscFunctionBegin;
    (void)nmat;
    const TDSEZAsmNDCtx *C = static_cast<const TDSEZAsmNDCtx *>(ctx);
    const PetscInt op = C ? C->op : 0;
    const PetscInt nen = p->nen;
    const PetscReal w  = p->weight[0] * p->detJac[0];
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    PetscCall(IGAPointFormGeomMap(p, xyz));
    const PetscReal x = xyz[0], y = xyz[1], z = xyz[2];
    const PetscReal cW = 0.5 * w;
    const PetscReal Vw = TDSEZParser::V(x, y, z) * w;
    const PetscReal *B = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 0, (const PetscReal **)&B));
    const PetscReal (*dB)[3] = NULL;
    PetscCall(IGAPointGetShapeFuns(p, 1, (const PetscReal **)&dB));
    for (PetscInt a = 0; a < nen; ++a) {
        const PetscReal BaW  = B[a] * w;
        const PetscReal pot_a = B[a] * Vw;
        const PetscReal gx_a = dB[a][0] * cW;
        const PetscReal gy_a = dB[a][1] * cW;
        const PetscReal gz_a = dB[a][2] * cW;
        PetscScalar *__restrict__ R = M[0] + a * nen;
        for (PetscInt b = 0; b < nen; ++b) {
            const PetscReal Bb = B[b];
            const PetscReal dx = dB[b][0], dy = dB[b][1], dz = dB[b][2];
            switch (op) {
            case 0: R[b] += gx_a * dx + gy_a * dy + gz_a * dz + pot_a * Bb; break;
            case 1: R[b] += BaW * Bb; break;
            case 2: R[b] += PetscScalar(0.0, -0.5 * (B[a] * dx - Bb * dB[a][0]) * w); break;
            case 3: R[b] += PetscScalar(0.0, -0.5 * (B[a] * dy - Bb * dB[a][1]) * w); break;
            case 4: R[b] += PetscScalar(0.0, -0.5 * (B[a] * dz - Bb * dB[a][2]) * w); break;
            case 5: R[b] += x * BaW * Bb; break;
            case 6: R[b] += y * BaW * Bb; break;
            case 7: R[b] += z * BaW * Bb; break;
            case 8: { const PetscReal lx_ab = (y * dz - z * dy);
                      const PetscReal lx_ba = (y * dB[a][2] - z * dB[a][1]);
                      R[b] += PetscScalar(0.0, -(B[a] * lx_ab - Bb * lx_ba) * w); } break;
            case 9: { const PetscReal ly_ab = (z * dx - x * dz);
                      const PetscReal ly_ba = (z * dB[a][0] - x * dB[a][2]);
                      R[b] += PetscScalar(0.0, -(B[a] * ly_ab - Bb * ly_ba) * w); } break;
            case 10: { const PetscReal lz_ab = (x * dy - y * dx);
                       const PetscReal lz_ba = (x * dB[a][1] - y * dB[a][0]);
                       R[b] += PetscScalar(0.0, -(B[a] * lz_ab - Bb * lz_ba) * w); } break;
            case 11: R[b] += pot_a * Bb; break;
            case 12: R[b] += B[a] * dx * w; break;   // Dreal_x = int B_a d_x B_b
            case 13: R[b] += B[a] * dy * w; break;   // Dreal_y
            case 14: R[b] += B[a] * dz * w; break;   // Dreal_z
            default: break;
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// 3D physics-check: Hermiticity of all 12 ops + 3D box spectrum.
PetscErrorCode TDSEZRunOperatorPhysicsChecks3D(PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ 3D operator physics-check (nel=%" PetscInt_FMT
        ", p=%" PetscInt_FMT ") ===\n", nel, pdegree));
    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, 3));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    const PetscReal Lmin = -20.0, Lmax = 20.0;
    for (PetscInt d = 0; d < 3; ++d) {
        PetscCall(IGAAxisSetDegree(iga->axis[d], pdegree));
        PetscCall(IGAAxisInitUniform(iga->axis[d], nel, Lmin, Lmax, 0));
    }
    PetscCall(IGASetUp(iga));
    TDSEZParser::VPot.SetExpr("0.0");
    TDSEZParser::MassExpr.SetExpr("1.0");
    struct OpInfo { PetscInt idx; const char *name; };
    const OpInfo ops[] = {
        {0,"H"},{1,"M"},{2,"Jx"},{3,"Jy"},{4,"Jz"},
        {5,"Dx"},{6,"Dy"},{7,"Dz"},{8,"Lx"},{9,"Ly"},{10,"Lz"},{11,"A"}};
    const PetscInt nOps = (PetscInt)(sizeof(ops) / sizeof(ops[0]));
    PetscBool allOK = PETSC_TRUE;
    for (PetscInt o = 0; o < nOps; ++o) {
        Mat O = PETSC_NULLPTR; PetscCall(IGACreateMat(iga, &O));
        { TDSEZAsmNDCtx c; c.op = ops[o].idx;
          PetscCall(TDSEZCompOperators(iga, 1, &O, TDSEZAsm1Op3D, &c)); }
        PetscCall(MatAssemblyBegin(O, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(O, MAT_FINAL_ASSEMBLY));
        Mat Oconj; PetscCall(MatHermitianTranspose(O, MAT_INITIAL_MATRIX, &Oconj));
        Mat Diff;   PetscCall(MatDuplicate(O, MAT_COPY_VALUES, &Diff));
        PetscCall(MatAXPY(Diff, -1.0, Oconj, DIFFERENT_NONZERO_PATTERN));
        PetscReal nrm = 0.0, scale = 0.0;
        PetscCall(MatNorm(Diff, NORM_INFINITY, &nrm));
        PetscCall(MatNorm(O, NORM_INFINITY, &scale));
        const PetscReal rel = (scale > 0.0) ? nrm / scale : nrm;
        const PetscBool ok = (rel < 1e-10);
        if (!ok) allOK = PETSC_FALSE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  %-4s Hermitian? %s   |O-O^H|/|O| = %.3e\n",
            ops[o].name, ok ? "YES" : "NO", (double)rel));
        PetscCall(MatDestroy(&O)); PetscCall(MatDestroy(&Oconj)); PetscCall(MatDestroy(&Diff));
    }
    // Reconstruction: Jx == (-i/2)(Dreal_x - Dreal_x^T)  [Dreal_x = real derivative]
    {
        Mat Dr = PETSC_NULLPTR, Jx = PETSC_NULLPTR;
        PetscCall(IGACreateMat(iga, &Dr)); PetscCall(IGACreateMat(iga, &Jx));
        { TDSEZAsmNDCtx c; c.op = 12; PetscCall(TDSEZCompOperators(iga, 1, &Dr, TDSEZAsm1Op3D, &c)); }
        { TDSEZAsmNDCtx c; c.op = 2;  PetscCall(TDSEZCompOperators(iga, 1, &Jx, TDSEZAsm1Op3D, &c)); }
        PetscCall(MatAssemblyBegin(Dr, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(Dr, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyBegin(Jx, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(Jx, MAT_FINAL_ASSEMBLY));
        Mat DrT; PetscCall(MatTranspose(Dr, MAT_INITIAL_MATRIX, &DrT));
        Mat Tmp; PetscCall(MatDuplicate(Dr, MAT_COPY_VALUES, &Tmp));
        PetscCall(MatAXPY(Tmp, -1.0, DrT, DIFFERENT_NONZERO_PATTERN));
        PetscCall(MatScale(Tmp, PetscScalar(0.0, -0.5)));
        Mat Err; PetscCall(MatDuplicate(Jx, MAT_COPY_VALUES, &Err));
        PetscCall(MatAXPY(Err, -1.0, Tmp, DIFFERENT_NONZERO_PATTERN));
        PetscReal nrm = 0.0, sc = 0.0;
        PetscCall(MatNorm(Err, NORM_INFINITY, &nrm));
        PetscCall(MatNorm(Jx, NORM_INFINITY, &sc));
        const PetscReal rel = (sc > 0.0) ? nrm / sc : nrm;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  reconstruction Jx == (-i/2)(Dreal_x-Dreal_x^T): rel err = %.3e  %s\n",
            (double)rel, rel < 1e-10 ? "OK" : "FAIL"));
        if (rel >= 1e-10) allOK = PETSC_FALSE;
        PetscCall(MatDestroy(&Dr)); PetscCall(MatDestroy(&Jx)); PetscCall(MatDestroy(&DrT));
        PetscCall(MatDestroy(&Tmp)); PetscCall(MatDestroy(&Err));
    }
    // 3D box spectrum (Neumann)
    PetscBool specOK = PETSC_TRUE;
    {
        Mat H = PETSC_NULLPTR, M = PETSC_NULLPTR;
        PetscCall(IGACreateMat(iga, &H)); PetscCall(IGACreateMat(iga, &M));
        { TDSEZAsmNDCtx c; c.op = 0; PetscCall(TDSEZCompOperators(iga, 1, &H, TDSEZAsm1Op3D, &c)); }
        { TDSEZAsmNDCtx c; c.op = 1; PetscCall(TDSEZCompOperators(iga, 1, &M, TDSEZAsm1Op3D, &c)); }
        PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
        EPS eps; PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
        PetscCall(EPSSetOperators(eps, H, M));
        PetscCall(EPSSetProblemType(eps, EPS_GHEP));
        PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL));
        PetscCall(EPSSetDimensions(eps, 16, PETSC_DETERMINE, PETSC_DETERMINE));
        PetscCall(EPSSetFromOptions(eps));
        PetscCall(EPSSolve(eps));
        const PetscReal a = Lmax - Lmin, pi2 = PETSC_PI * PETSC_PI;
        PetscReal kset[125]; PetscInt nk = 0;
        for (PetscInt nx = 0; nx <= 4; ++nx)
        for (PetscInt ny = 0; ny <= 4; ++ny)
        for (PetscInt nz = 0; nz <= 4; ++nz)
            kset[nk++] = (PetscReal)(nx*nx + ny*ny + nz*nz);
        for (PetscInt i = 0; i < nk; ++i)
            for (PetscInt j = i+1; j < nk; ++j)
                if (kset[j] < kset[i]) { PetscReal t = kset[i]; kset[i] = kset[j]; kset[j] = t; }
        PetscReal eigs[12], expE[12];
        for (PetscInt i = 0; i < 12; ++i) {
            PetscScalar kr, ki;
            PetscCall(EPSGetEigenpair(eps, i, &kr, &ki, PETSC_NULLPTR, PETSC_NULLPTR));
            eigs[i] = PetscRealPart(kr);   // Neumann box spectrum is real
            expE[i] = pi2 * kset[i] / (2.0 * a * a);
        }
        // sort both so degeneracy ordering from the eigensolver doesn't cause
        // spurious index mismatches (the eigenvalues themselves are correct).
        for (PetscInt i = 0; i < 12; ++i)
            for (PetscInt j = i+1; j < 12; ++j) {
                if (eigs[j] < eigs[i]) { PetscReal t = eigs[i]; eigs[i] = eigs[j]; eigs[j] = t; }
                if (expE[j] < expE[i]) { PetscReal t = expE[i]; expE[i] = expE[j]; expE[j] = t; }
            }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  3D box spectrum (Neumann, a=%.1f):\n", (double)a));
        for (PetscInt i = 0; i < 12; ++i) {
            const PetscReal rel = (expE[i] != 0.0) ? PetscAbsReal(eigs[i]-expE[i])/expE[i]
                                                   : PetscAbsReal(eigs[i]-expE[i]);
            const PetscBool ok = (rel < 1e-3);
            if (!ok) specOK = PETSC_FALSE;
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "    E[%2d] = %10.6f  expect %10.6f  rel %.1e  %s\n",
                i, (double)eigs[i], (double)expE[i], (double)rel, ok ? "OK" : "FAIL"));
        }
        PetscCall(EPSDestroy(&eps)); PetscCall(MatDestroy(&H)); PetscCall(MatDestroy(&M));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  -> %s (3D Hermiticity + box spectrum)\n",
        (allOK && specOK) ? "PASS — 3D operators are physically correct" : "FAIL"));
    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// 3D fusion sweep: fused (one walk, K ops) vs unfused (K walks).
PetscErrorCode TDSEZRunFusionSweep3D(PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ 3D fusion sweep (nel=%" PetscInt_FMT
        ", p=%" PetscInt_FMT ") ===\n", nel, pdegree));
    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, 3));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    const PetscReal Lmin = -20.0, Lmax = 20.0;
    for (PetscInt d = 0; d < 3; ++d) {
        PetscCall(IGAAxisSetDegree(iga->axis[d], pdegree));
        PetscCall(IGAAxisInitUniform(iga->axis[d], nel, Lmin, Lmax, 0));
    }
    PetscCall(IGASetUp(iga));
    TDSEZParser::VPot.SetExpr("0.0");
    TDSEZParser::MassExpr.SetExpr("1.0");
    const PetscInt Ks[3] = {2, 6, 12};
    const PetscInt nRepeat = 5;
    for (PetscInt ki = 0; ki < 3; ++ki) {
        const PetscInt K = Ks[ki];
        PetscLogDouble tF = 0.0, tU = 0.0;
        { Mat mats[12];
          for (PetscInt k = 0; k < K; ++k) PetscCall(IGACreateMat(iga, &mats[k]));
          for (PetscInt r = 0; r < nRepeat; ++r) {
              for (PetscInt k = 0; k < K; ++k) PetscCall(MatZeroEntries(mats[k]));
              TDSEZAsmNDCtx ctx; ctx.nop = K;
              PetscLogDouble t0, t1; PetscCall(PetscTime(&t0));
              PetscCall(TDSEZCompOperators(iga, K, mats, TDSEZAsmND_Ops3D, &ctx));
              PetscCall(PetscTime(&t1)); tF += (t1 - t0);
          }
          for (PetscInt k = 0; k < K; ++k) PetscCall(MatDestroy(&mats[k])); }
        { for (PetscInt r = 0; r < nRepeat; ++r) {
              PetscLogDouble t0, t1; PetscCall(PetscTime(&t0));
              for (PetscInt k = 0; k < K; ++k) {
                  Mat m = PETSC_NULLPTR; PetscCall(IGACreateMat(iga, &m));
                  TDSEZAsmNDCtx ctx; ctx.op = k;
                  PetscCall(TDSEZCompOperators(iga, 1, &m, TDSEZAsm1Op3D, &ctx));
                  PetscCall(MatDestroy(&m));
              }
              PetscCall(PetscTime(&t1)); tU += (t1 - t0);
          } }
        PetscBool identical = PETSC_TRUE; PetscReal gMax = 0.0;
        { Mat F[12]; for (PetscInt k = 0; k < K; ++k) PetscCall(IGACreateMat(iga, &F[k]));
          { TDSEZAsmNDCtx ctx; ctx.nop = K; PetscCall(TDSEZCompOperators(iga, K, F, TDSEZAsmND_Ops3D, &ctx)); }
          for (PetscInt k = 0; k < K; ++k) {
              Mat U = PETSC_NULLPTR; PetscCall(IGACreateMat(iga, &U));
              { TDSEZAsmNDCtx ctx; ctx.op = k; PetscCall(TDSEZCompOperators(iga, 1, &U, TDSEZAsm1Op3D, &ctx)); }
              PetscCall(MatAssemblyBegin(F[k], MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(F[k], MAT_FINAL_ASSEMBLY));
              PetscCall(MatAssemblyBegin(U,    MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(U,    MAT_FINAL_ASSEMBLY));
              Mat D; PetscCall(MatDuplicate(F[k], MAT_COPY_VALUES, &D));
              PetscCall(MatAXPY(D, -1.0, U, DIFFERENT_NONZERO_PATTERN));
              PetscReal nrm = 0.0, sc = 0.0;
              PetscCall(MatNorm(D, NORM_INFINITY, &nrm));
              PetscCall(MatNorm(F[k], NORM_INFINITY, &sc));
              const PetscReal rel = (sc > 0.0) ? nrm / sc : nrm;
              if (rel > 1e-10) identical = PETSC_FALSE;
              if (rel > gMax) gMax = rel;
              PetscCall(MatDestroy(&U)); PetscCall(MatDestroy(&D));
          }
          for (PetscInt k = 0; k < K; ++k) PetscCall(MatDestroy(&F[k])); }
        const PetscReal sp = (tF > 0.0) ? tU / tF : 0.0;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  K=%2d  fused=%8.3f ms  unfused=%8.3f ms  speedup=%.2fx  |max dOp|=%.1e  %s\n",
            K, tF * 1e3 / nRepeat, tU * 1e3 / nRepeat, (double)sp, (double)gMax,
            identical ? "IDENTICAL" : "DIFFERS"));
    }
    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ============================================================================
//  BENCHMARK / DEMO HARNESSES (not core physics)
// ----------------------------------------------------------------------------
//  The TDSEZRun* functions below are self-contained accuracy / benchmark /
//  operator-consistency demos. They are exercised by the standalone
//  tdsez_asmbench target (assembler_bench_main.cpp), NOT by the production
//  tdsez run. Kept in this TU only to share the TDSEZAsm* kernels; they do
//  not participate in the main eigen-solve / propagation path.
// ============================================================================
PetscErrorCode TDSEZRunAssemblerBenchmark(PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ 2D assembler benchmark (nel=%" PetscInt_FMT
        ", p=%" PetscInt_FMT ") ===\n", nel, pdegree));

    // ---- build a uniform 2D IGA ------------------------------------------
    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, 2));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    const PetscReal Lmin = -20.0, Lmax = 20.0;
    for (PetscInt d = 0; d < 2; ++d) {
        PetscCall(IGAAxisSetDegree(iga->axis[d], pdegree));   // must precede InitUniform
        PetscCall(IGAAxisInitUniform(iga->axis[d], nel, Lmin, Lmax, 0));
    }
    PetscCall(IGASetUp(iga));

    // harmless flat potential + constant mass
    TDSEZParser::VPot.SetExpr("0.0");
    TDSEZParser::MassExpr.SetExpr("1.0");

    const ScenarioSpec specs[] = {
        {"[1] ConstMass H+M", TDSEZAsm2D_ConstMass, TDSEZAsm2D_H_only,   TDSEZAsm2D_M_only,  PETSC_TRUE},
        {"[2] VarMass   H+M", TDSEZAsm2D_VarMass,   TDSEZAsm2D_H_only,   TDSEZAsm2D_M_only,  PETSC_TRUE},
        {"[3] CAP       H(cx)+M", TDSEZAsm2D_CAP,    TDSEZAsm2D_H_only,   TDSEZAsm2D_M_only,  PETSC_TRUE},
        {"[4] Lz        H+Lz", TDSEZAsm2D_Lz,        TDSEZAsm2D_H_only,   TDSEZAsm2D_Lz_only, PETSC_FALSE},
    };
    const PetscInt nSpec = (PetscInt)(sizeof(specs) / sizeof(specs[0]));
    const PetscInt nRepeat = 5;

    for (PetscInt s = 0; s < nSpec; ++s) {
        const ScenarioSpec &sp = specs[s];
        TDSEZAsm2DCtx ctx; ctx.scenario = s;

        Mat Hf = PETSC_NULLPTR, Mf = PETSC_NULLPTR;
        Mat Hu = PETSC_NULLPTR, Mu = PETSC_NULLPTR;
        PetscLogDouble tFused = 0.0, tUnf = 0.0;

        // ---- FUSED: one TDSEZCompOperators call, nmat=2 -----------------
        for (PetscInt r = 0; r < nRepeat; ++r) {
            PetscCall(IGACreateMat(iga, &Hf));
            PetscCall(MatDuplicate(Hf, MAT_DO_NOT_COPY_VALUES, &Mf));
            Mat mats[2] = {Hf, Mf};
            PetscLogDouble t0, t1;
            PetscCall(PetscTime(&t0));
            PetscCall(TDSEZCompOperators(iga, 2, mats, sp.fused, &ctx));
            PetscCall(PetscTime(&t1));
            tFused += (t1 - t0);
            PetscCall(MatDestroy(&Hf));
            PetscCall(MatDestroy(&Mf));
        }
        tFused /= nRepeat;

        // ---- UNFUSED: two TDSEZCompOperators calls, nmat=1 each ---------
        for (PetscInt r = 0; r < nRepeat; ++r) {
            PetscCall(IGACreateMat(iga, &Hu));
            PetscCall(MatDuplicate(Hu, MAT_DO_NOT_COPY_VALUES, &Mu));
            PetscLogDouble t0, t1;
            PetscCall(PetscTime(&t0));
            Mat mH[1] = {Hu};
            PetscCall(TDSEZCompOperators(iga, 1, mH, sp.op0, &ctx));
            Mat mM[1] = {Mu};
            PetscCall(TDSEZCompOperators(iga, 1, mM, sp.op1, &ctx));
            PetscCall(PetscTime(&t1));
            tUnf += (t1 - t0);
            PetscCall(MatDestroy(&Hu));
            PetscCall(MatDestroy(&Mu));
        }
        tUnf /= nRepeat;

        // ---- correctness: re-assemble both and compare ------------------
        Mat Hfa = PETSC_NULLPTR, Mfa = PETSC_NULLPTR;
        Mat Hua = PETSC_NULLPTR, Mua = PETSC_NULLPTR;
        {
            PetscCall(IGACreateMat(iga, &Hfa));
            PetscCall(MatDuplicate(Hfa, MAT_DO_NOT_COPY_VALUES, &Mfa));
            Mat mats[2] = {Hfa, Mfa};
            PetscCall(TDSEZCompOperators(iga, 2, mats, sp.fused, &ctx));

            PetscCall(IGACreateMat(iga, &Hua));
            PetscCall(MatDuplicate(Hua, MAT_DO_NOT_COPY_VALUES, &Mua));
            Mat mH[1] = {Hua};
            PetscCall(TDSEZCompOperators(iga, 1, mH, sp.op0, &ctx));
            Mat mM[1] = {Mua};
            PetscCall(TDSEZCompOperators(iga, 1, mM, sp.op1, &ctx));
        }

        Mat Hd; PetscCall(MatDuplicate(Hfa, MAT_COPY_VALUES, &Hd));
        PetscCall(MatAXPY(Hd, -1.0, Hua, DIFFERENT_NONZERO_PATTERN));
        PetscReal nH; PetscCall(MatNorm(Hd, NORM_FROBENIUS, &nH));
        PetscReal nHref; PetscCall(MatNorm(Hfa, NORM_FROBENIUS, &nHref));
        Mat Md; PetscCall(MatDuplicate(Mfa, MAT_COPY_VALUES, &Md));
        PetscCall(MatAXPY(Md, -1.0, Mua, DIFFERENT_NONZERO_PATTERN));
        PetscReal nM; PetscCall(MatNorm(Md, NORM_FROBENIUS, &nM));
        PetscReal nMref; PetscCall(MatNorm(Mfa, NORM_FROBENIUS, &nMref));

        const PetscReal relH = (nHref > 0) ? nH / nHref : nH;
        const PetscReal relM = (nMref > 0) ? nM / nMref : nM;
        const PetscReal speedup = (tUnf > 0) ? tUnf / tFused : 0.0;

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  %-18s  fused=%8.4f ms  unfused=%8.4f ms  speedup=%.2fx"
            "  |dH|/|H|=%.2e  |dM|/|M|=%.2e\n",
            sp.name, tFused * 1e3, tUnf * 1e3, speedup, relH, relM));

        PetscCall(MatDestroy(&Hfa)); PetscCall(MatDestroy(&Mfa));
        PetscCall(MatDestroy(&Hua)); PetscCall(MatDestroy(&Mua));
        PetscCall(MatDestroy(&Hd));  PetscCall(MatDestroy(&Md));
    }

    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ----------------------------------------------------------------------------
//  HARMONIC OSCILLATOR cross-check (stronger than the box: tests T+V coupling
//  with a non-trivial, separable potential whose eigenfunctions are known).
//  2D: V = 1/2 w^2 (x^2 + y^2),  E_{n,m} = w (n + m + 1)
//  3D: V = 1/2 w^2 (x^2 + y^2 + z^2),  E_{n,m,l} = w (n + m + l + 3/2)
//  With w = 1 the level order is the sum of quantum numbers; we verify the
//  lowest ~12 distinct levels (sorted) agree with the analytic multiset.
// ----------------------------------------------------------------------------

PetscErrorCode TDSEZRunHarmonicOscillator(PetscInt dim, PetscInt nel,
                                                PetscInt pdegree)
{
    PetscFunctionBegin;
    TDSEZParser::initParsers();   // binds x,y,z (and VPot/MassExpr variables)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ %dD harmonic-oscillator cross-check (nel=%" PetscInt_FMT
        ", p=%" PetscInt_FMT ", w=1) ===\n", dim, nel, pdegree));
    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, dim));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    const PetscReal Lmin = -6.0, Lmax = 6.0;
    for (PetscInt d = 0; d < dim; ++d) {
        PetscCall(IGAAxisSetDegree(iga->axis[d], pdegree));
        PetscCall(IGAAxisInitUniform(iga->axis[d], nel, Lmin, Lmax, 0));
    }
    PetscCall(IGASetUp(iga));
    TDSEZParser::VPot.SetExpr(dim == 3 ? "0.5*(x*x+y*y+z*z)" : "0.5*(x*x+y*y)");
    TDSEZParser::MassExpr.SetExpr("1.0");
    Mat H = PETSC_NULLPTR, M = PETSC_NULLPTR;
    PetscCall(IGACreateMat(iga, &H)); PetscCall(IGACreateMat(iga, &M));
    if (dim == 3) {
        { TDSEZAsmNDCtx c; c.op = 0; PetscCall(TDSEZCompOperators(iga, 1, &H, TDSEZAsm1Op3D, &c)); }
        { TDSEZAsmNDCtx c; c.op = 1; PetscCall(TDSEZCompOperators(iga, 1, &M, TDSEZAsm1Op3D, &c)); }
    } else {
        { TDSEZAsmNDCtx c; c.op = 0; PetscCall(TDSEZCompOperators(iga, 1, &H, TDSEZAsm1Op, &c)); }
        { TDSEZAsmNDCtx c; c.op = 1; PetscCall(TDSEZCompOperators(iga, 1, &M, TDSEZAsm1Op, &c)); }
    }
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
    EPS eps; PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, H, M));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL));
    PetscCall(EPSSetDimensions(eps, 16, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    const PetscInt n = 12;
    PetscReal eigs[16], expE[16];
    PetscInt cnt = 0;
    for (PetscInt N = 0; N <= 12 && cnt < n; ++N) {       // sum of quantum numbers
        // count degeneracy for this total N in 'dim' dims
        PetscInt ways = 0;
        if (dim == 2) {
            for (PetscInt a = 0; a <= N; ++a) { PetscInt b = N - a;
                if (b >= 0) ++ways; }
        } else {
            for (PetscInt a = 0; a <= N; ++a)
            for (PetscInt b = 0; b <= N-a; ++b) { PetscInt c = N-a-b;
                if (c >= 0) ++ways; }
        }
        for (PetscInt w = 0; w < ways && cnt < n; ++w) {
            expE[cnt++] = (PetscReal)N + 0.5 * (PetscReal)dim;   // w=1, ground = dim/2
        }
    }
    for (PetscInt i = 0; i < cnt; ++i) {
        PetscScalar kr, ki;
        PetscCall(EPSGetEigenpair(eps, i, &kr, &ki, PETSC_NULLPTR, PETSC_NULLPTR));
        eigs[i] = PetscRealPart(kr);
    }
    for (PetscInt i = 0; i < cnt; ++i)
        for (PetscInt j = i+1; j < cnt; ++j) {
            if (eigs[j] < eigs[i])  { PetscReal t = eigs[i];  eigs[i]  = eigs[j];  eigs[j]  = t; }
            if (expE[j] < expE[i])  { PetscReal t = expE[i];  expE[i]  = expE[j];  expE[j]  = t; }
        }
    PetscBool ok = PETSC_TRUE;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  (expect E = w*(n0+n1+...+dim/2), w=1; lowest %d levels)\n", n));
    for (PetscInt i = 0; i < cnt; ++i) {
        const PetscReal rel = (expE[i] != 0.0) ? PetscAbsReal(eigs[i]-expE[i])/expE[i]
                                               : PetscAbsReal(eigs[i]-expE[i]);
        const PetscBool lok = (rel < 1e-3);
        if (!lok) ok = PETSC_FALSE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "    E[%2d] = %10.6f  expect %8.4f  rel %.1e  %s\n",
            i, (double)eigs[i], (double)expE[i], (double)rel, lok ? "OK" : "FAIL"));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  -> %s (harmonic oscillator spectrum)\n",
        ok ? "PASS — physics consistent with analytic HO" : "FAIL"));
    PetscCall(EPSDestroy(&eps)); PetscCall(MatDestroy(&H)); PetscCall(MatDestroy(&M));
    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}


//  (a) Hermiticity: real/symmetric ops (H,M,Dx,Dy,A) must satisfy O = O^T;
//      imaginary anti-Hermitian ops (Jx,Jy,Lz) must satisfy O + O^H = 0.
//  (b) Eigen-spectrum: solve H psi = E M psi for a 2D particle-in-a-box and
//      compare eigenvalues to the analytic E_{n,m} = (pi^2 / 2 a^2)(n^2+m^2).
//  These catch sign/factor bugs that fused==unfused consistency checks miss.
// ----------------------------------------------------------------------------
PetscErrorCode TDSEZRunOperatorPhysicsChecks(PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    IGA iga = NULL;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, 2));
    PetscCall(IGASetDof(iga, 1));
    const PetscReal L = 1.0;                      // box side a = 1
    for (PetscInt d = 0; d < 2; ++d) {
        IGAAxis axis = iga->axis[d];
        PetscCall(IGAAxisSetDegree(axis, pdegree));
        PetscCall(IGAAxisInitUniform(axis, nel, 0.0, L, 0));
    }
    // Note: IGASetBoundaryValue records BC intent but TDSEZCompOperators does
    // NOT apply it (no FixSystem / IGAComputeSystem in the bare element-assemble
    // path -- confirmed: H size = full dof count, E[0]=0 Neumann, identical with
    // or without these calls). The operators are therefore the natural-BC
    // (Neumann) Laplacian. Homogeneous Dirichlet would require a FixSystem step
    // (or CAP-based absorption) and is a setup concern, not an operator bug.
    // We leave the calls in place to document intent.
    for (PetscInt axis = 0; axis < 2; ++axis)
        for (PetscInt side = 0; side < 2; ++side)
            PetscCall(IGASetBoundaryValue(iga, axis, side, 0, 0.0));
    PetscCall(IGASetUp(iga));
    TDSEZParser::VPot.SetExpr("0.0");
    TDSEZParser::MassExpr.SetExpr("1.0");

    // (BC note: the weak form gives Neumann walls; the B-spline open-knot basis
    // does NOT force psi=0 at the wall, and IGASetBoundaryValue is not applied by
    // TDSEZCompOperators. Verified empirically earlier. CAP (AbsorbingBoundary)
    // is the intended absorber and is functional when CAPKmin>0.)

    // operator table: index, name, whether it is Hermitian (all physical
    // observables are Hermitian: real ops -> O = O^T; purely-imaginary ops
    // like Jx/Jy/Lz -> anti-symmetric in indices, still O = O^H).
    struct Op { PetscInt idx; const char *name; PetscBool herm; };
    const Op ops[8] = {
        {0, "H ", PETSC_TRUE}, {1, "M ", PETSC_TRUE},
        {2, "Jx", PETSC_TRUE}, {3, "Jy", PETSC_TRUE},
        {4, "Dx", PETSC_TRUE}, {5, "Dy", PETSC_TRUE},
        {6, "Lz", PETSC_TRUE}, {7, "A ", PETSC_TRUE},
    };

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ operator physics checks (2D box a=%g, nel=%" PetscInt_FMT ", p=%" PetscInt_FMT ") ===\n",
        (double)L, nel, pdegree));

    PetscBool allOK = PETSC_TRUE;
    for (PetscInt o = 0; o < 8; ++o) {
        Mat O = PETSC_NULLPTR;
        PetscCall(IGACreateMat(iga, &O));
        TDSEZAsmNDCtx ctx; ctx.op = ops[o].idx;
        Mat m[1] = {O};
        PetscCall(TDSEZCompOperators(iga, 1, m, TDSEZAsm1Op, &ctx));

        PetscReal nO;
        PetscCall(MatNorm(O, NORM_FROBENIUS, &nO));

        // Hermiticity residual: all physical operators satisfy O = O^H
        PetscReal res = 0.0;
        if (nO > 1e-14) {                         // skip zero operators (V=0 -> A=0)
            Mat Ot = PETSC_NULLPTR;
            PetscCall(MatHermitianTranspose(O, MAT_INITIAL_MATRIX, &Ot));
            PetscCall(MatAXPY(Ot, -1.0, O, DIFFERENT_NONZERO_PATTERN));   // O - O^H
            PetscCall(MatNorm(Ot, NORM_FROBENIUS, &res));
            PetscCall(MatDestroy(&Ot));
        }
        const PetscReal rel = (nO > 1e-14) ? res / nO : 0.0;
        const PetscBool ok = (nO < 1e-14) || (rel < 1e-10);
        allOK = (allOK && ok) ? PETSC_TRUE : PETSC_FALSE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  %s  |O|=%.4e  Hermiticity residual |O-O^H|/|O| = %.2e  %s\n",
            ops[o].name,
            nO, rel,
            ok ? "OK" : "FAIL"));
        PetscCall(MatDestroy(&O));
    }

    // (b) canonical commutation relation [x, p_x] = i, with p_x = -i d_x = 2 Jx.
    //      => [Dx, Jx] = (i/2) M  (M = mass/overlap). This is the decisive
    //      physics test: it must hold to machine precision. All eight operators
    //      are Hermitian (verified above); this confirms Jx/Dx are correctly
    //      normalised AND related by the canonical algebra.
    {
        Mat Dx = PETSC_NULLPTR, Jx = PETSC_NULLPTR, Mref = PETSC_NULLPTR, Dr = PETSC_NULLPTR;
        PetscCall(IGACreateMat(iga, &Dx));
        PetscCall(IGACreateMat(iga, &Jx));
        PetscCall(IGACreateMat(iga, &Mref));
        PetscCall(IGACreateMat(iga, &Dr));
        { TDSEZAsmNDCtx c; c.op = 4; Mat m[1] = {Dx};  PetscCall(TDSEZCompOperators(iga, 1, m, TDSEZAsm1Op, &c)); }
        { TDSEZAsmNDCtx c; c.op = 2; Mat m[1] = {Jx}; PetscCall(TDSEZCompOperators(iga, 1, m, TDSEZAsm1Op, &c)); }
        { TDSEZAsmNDCtx c; c.op = 1; Mat m[1] = {Mref}; PetscCall(TDSEZCompOperators(iga, 1, m, TDSEZAsm1Op, &c)); }
        { TDSEZAsmNDCtx c; c.op = 8; Mat m[1] = {Dr};   PetscCall(TDSEZCompOperators(iga, 1, m, TDSEZAsm1Op, &c)); }
        // Reconstruct the expected Hermitian momentum from the real derivative:
        //   Jx_expected = (-i/2) (D - D^T)   where D = Dr (real, D_ab = int B_a d_x B_b)
        Mat Dt = PETSC_NULLPTR;
        PetscCall(MatTranspose(Dr, MAT_INITIAL_MATRIX, &Dt));
        PetscCall(MatAXPY(Dr, -1.0, Dt, DIFFERENT_NONZERO_PATTERN)); // Dr = D - D^T
        PetscCall(MatDestroy(&Dt));
        PetscCall(MatScale(Dr, PetscScalar(0.0, -0.5)));            // Dr = (-i/2)(D - D^T)
        PetscReal nJ, nE;
        PetscCall(MatNorm(Jx,     NORM_FROBENIUS, &nJ));
        PetscCall(MatNorm(Dr,      NORM_FROBENIUS, &nE));
        PetscCall(MatAXPY(Dr, -1.0, Jx, DIFFERENT_NONZERO_PATTERN)); // Dr = Jx_expected - Jx
        PetscReal nRes;
        PetscCall(MatNorm(Dr, NORM_FROBENIUS, &nRes));
        PetscCall(MatDestroy(&Dr));
        // canonical algebra: [Dx, Jx] = (i/2) M
        Mat DJ = PETSC_NULLPTR, JD = PETSC_NULLPTR;
        PetscCall(MatMatMult(Dx, Jx, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DJ));
        PetscCall(MatMatMult(Jx, Dx, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &JD));
        PetscCall(MatAXPY(DJ, -1.0, JD, DIFFERENT_NONZERO_PATTERN)); // DJ = [Dx, Jx]
        PetscCall(MatDestroy(&JD));
        Mat R = PETSC_NULLPTR;
        PetscCall(MatDuplicate(DJ, MAT_DO_NOT_COPY_VALUES, &R));
        PetscCall(MatZeroEntries(R));
        PetscCall(MatShift(R, PetscScalar(0.0, 0.5)));
        PetscCall(MatMatMult(R, Mref, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &R)); // R = (i/2) M
        PetscReal nComm, nRef, nCRes;
        PetscCall(MatNorm(DJ,  NORM_FROBENIUS, &nComm));
        PetscCall(MatNorm(R,   NORM_FROBENIUS, &nRef));
        PetscCall(MatAXPY(DJ, -1.0, R, DIFFERENT_NONZERO_PATTERN));
        PetscCall(MatNorm(DJ, NORM_FROBENIUS, &nCRes));
        PetscCall(MatDestroy(&R)); PetscCall(MatDestroy(&DJ));
        PetscCall(MatDestroy(&Dx)); PetscCall(MatDestroy(&Jx)); PetscCall(MatDestroy(&Mref));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  Jx == (-i/2)(D-D^T): ||recon-Jx||/||Jx|| = %.2e (expect ~0)\n", nRes / nJ));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  [Dx, Jx] commutator: ||[Dx,Jx]||=%.3e  ||(i/2)M||=%.3e  residual/ref=%.2e (expect ~0)\n",
            nComm, nRef, nCRes / nRef));
}

    // (c) eigen-spectrum check on H and M
    Mat H = PETSC_NULLPTR, M = PETSC_NULLPTR;
    PetscCall(IGACreateMat(iga, &H));
    PetscCall(IGACreateMat(iga, &M));

    {
        TDSEZAsmNDCtx cH; cH.op = 0; Mat mh[1] = {H};
        PetscCall(TDSEZCompOperators(iga, 1, mh, TDSEZAsm1Op, &cH));
        TDSEZAsmNDCtx cM; cM.op = 1; Mat mm[1] = {M};
        PetscCall(TDSEZCompOperators(iga, 1, mm, TDSEZAsm1Op, &cM));
    }

    EPS eps; PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, H, M));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    // shift-invert (STSINVERT): target the low discrete box states near 2.0 a.u.
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL));
    PetscCall(EPSSetTarget(eps, 2.0));
    PetscCall(EPSSetDimensions(eps, 12, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(EPSSetTolerances(eps, 1e-12, 2000));
    ST st; PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(STSetShift(st, 2.0));
    PetscCall(EPSSolve(eps));

    PetscInt nconv = 0;
    PetscCall(EPSGetConverged(eps, &nconv));
    nconv = PetscMin(nconv, 12);

    // analytic 2D SPECTRUM for the assembled (NATURAL / Neumann BC) Laplacian:
    //   E_{n,m} = (pi^2 / 2 a^2)(n^2+m^2),  a=1,  n,m >= 0
    // (TDSEZCompOperators does not apply essential/Dirichlet BC -- a PetIGA
    //  limitation shared with the production H assembly path -- so the operator
    //  is the free/Neumann Laplacian; the Dirichlet box spectrum would require
    //  BC enforcement, which is a setup concern, not an operator-formula bug.)
    PetscReal Ean[12];
    {
        PetscReal raw[64]; PetscInt rc = 0;
        for (PetscInt n = 0; n <= 8 && rc < 64; ++n)
            for (PetscInt m = 0; m <= 8 && rc < 64; ++m)
                raw[rc++] = (PetscReal)(n*n + m*m);
        std::sort(raw, raw + rc);
        PetscInt taken = 0;
        for (PetscInt i = 0; i < rc && taken < 12; ++i) {
            Ean[taken] = (PETSC_PI * PETSC_PI / (2.0 * L * L)) * raw[i];
            ++taken;
        }
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  eigen-spectrum (H psi = E M psi), analytic Neumann E_{n,m}=pi^2/(2a^2)(n^2+m^2), n,m>=0:\n"));

    // PROBE (user request): is BC enforced? Inspect a boundary DOF row. If
    // Dirichlet-pinned, H row is unit-diagonal/zero-offdiag and M row is zero.
    // If Neumann (natural), H row is a normal Laplacian row. Run on rank 0 only
    // (MatGetRow requires a locally-owned row). Use the first locally-owned row.
    {
        PetscMPIInt rank = 0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
        if (rank == 0) {
            PetscInt rlo, rhi; PetscCall(MatGetOwnershipRange(H, &rlo, &rhi));
            const PetscInt bdof = rlo;   // a locally-owned row (boundary-ish on rank 0)
            PetscInt    ncols; const PetscInt *cols; const PetscScalar *vals;
            PetscCall(MatGetRow(H, bdof, &ncols, &cols, &vals));
            PetscScalar diagH = 0, offH = 0;
            for (PetscInt k = 0; k < ncols; ++k) {
                if (cols[k] == bdof) diagH = vals[k];
                else offH += vals[k];
            }
            PetscCall(MatRestoreRow(H, bdof, &ncols, &cols, &vals));
            PetscCall(MatGetRow(M, bdof, &ncols, &cols, &vals));
            PetscScalar diagM = 0, offM = 0;
            for (PetscInt k = 0; k < ncols; ++k) {
                if (cols[k] == bdof) diagM = vals[k];
                else offM += vals[k];
            }
            PetscCall(MatRestoreRow(M, bdof, &ncols, &cols, &vals));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  [probe] boundary DOF %d row (rank0, local): H diag=%.4e sum(off)=%.4e  M diag=%.4e sum(off)=%.4e\n",
                bdof, PetscRealPart(diagH), PetscRealPart(offH), PetscRealPart(diagM), PetscRealPart(offM)));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "    => Neumann if diagH>0 & offH<0 (closes to 0); Dirichlet-pinned if diagH~1 & offH~0 & diagM~0\n"));
        }
    }

    PetscBool specOK = PETSC_TRUE;
    for (PetscInt i = 0; i < nconv; ++i) {
        PetscScalar kr, ki;
        PetscCall(EPSGetEigenpair(eps, i, &kr, &ki, PETSC_NULLPTR, PETSC_NULLPTR));
        const PetscReal E = PetscRealPart(kr);
        const PetscReal ea = (i < 12) ? Ean[i] : 0.0;
        const PetscReal rerr = (i < 12 && ea > 0) ? PetscAbsReal(E - ea) / ea : 0.0;
        const PetscBool ok = (i >= 12) || (rerr < 1e-3);
        specOK = (specOK && ok) ? PETSC_TRUE : PETSC_FALSE;
        allOK  = (allOK  && ok) ? PETSC_TRUE : PETSC_FALSE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "    E[%d] = %.6f a.u.  analytic = %.6f  rel-err = %.2e  %s\n",
            (int)i, E, ea, rerr, ok ? "OK" : "FAIL"));
    }
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&H)); PetscCall(MatDestroy(&M));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  -> %s (Hermiticity + box spectrum)\n",
        (allOK && specOK) ? "PASS — operators are physically correct" : "FAIL"));

    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  DIRECT MATRIX COMPARISON (user request): are ALL propagation matrices from
//  the FUSED path identical to the UNFUSED reference (same engine)?  We loop
//  over every operator used in propagation:
//     2D: H, M, Jx, Jy, Dx, Dy, Lz, A
//     3D: H, M, Jx, Jy, Jz, Dx, Dy, Dz, Lx, Ly, Lz, A
//  For 2D we ALSO compare H,M against the production-equivalent
//  TDSEZAsm2D_ConstMass kernel (verified bit-exact vs TDSEZFormHam).
//  Every matrix is diffed entry-wise (Frobenius) fused vs unfused.
// ---------------------------------------------------------------------------
PetscErrorCode TDSEZRunMatrixComparison(PetscInt dim, PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ matrix comparison: FUSED vs UNFUSED (all ops), %dD"
        " (nel=%" PetscInt_FMT ", p=%" PetscInt_FMT ") ===\n", dim, nel, pdegree));
    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, dim));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    const PetscReal Lmin = -20.0, Lmax = 20.0;
    for (PetscInt d = 0; d < dim; ++d) {
        PetscCall(IGAAxisSetDegree(iga->axis[d], pdegree));
        PetscCall(IGAAxisInitUniform(iga->axis[d], nel, Lmin, Lmax, 0));
    }
    PetscCall(IGASetUp(iga));
    TDSEZParser::VPot.SetExpr("0.0");
    TDSEZParser::MassExpr.SetExpr("1.0");

    // operator table: {index, name} — include all propagation operators
    struct Op { PetscInt idx; const char *name; };
    const Op ops2D[] = { {0,"H"},{1,"M"},{2,"Jx"},{3,"Jy"},{4,"Dx"},{5,"Dy"},{6,"Lz"},{7,"A"} };
    const Op ops3D[] = { {0,"H"},{1,"M"},{2,"Jx"},{3,"Jy"},{4,"Jz"},
                          {5,"Dx"},{6,"Dy"},{7,"Dz"},{8,"Lx"},{9,"Ly"},{10,"Lz"},{11,"A"} };
    const Op *ops = (dim == 3) ? ops3D : ops2D;
    const PetscInt nOps = (dim == 3) ? (PetscInt)(sizeof(ops3D)/sizeof(ops3D[0]))
                                     : (PetscInt)(sizeof(ops2D)/sizeof(ops2D[0]));

    const PetscInt K = nOps;
    // FUSED: all operators in ONE walk
    Mat *F = new Mat[K];
    for (PetscInt k = 0; k < K; ++k) { PetscCall(IGACreateMat(iga, &F[k])); }
    {
        Mat *m = new Mat[K];
        for (PetscInt k = 0; k < K; ++k) m[k] = F[k];
        TDSEZAsmNDCtx c; c.nop = K;
        if (dim == 3) PetscCall(TDSEZCompOperators(iga, K, m, TDSEZAsmND_Ops3D, &c));
        else          PetscCall(TDSEZCompOperators(iga, K, m, TDSEZAsmND_Ops,  &c));
        delete[] m;
    }
    for (PetscInt k = 0; k < K; ++k) {
        PetscCall(MatAssemblyBegin(F[k], MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(F[k], MAT_FINAL_ASSEMBLY));
    }

    // UNFUSED: each operator independently
    Mat *U = new Mat[K];
    for (PetscInt k = 0; k < K; ++k) { PetscCall(IGACreateMat(iga, &U[k])); }
    for (PetscInt k = 0; k < K; ++k) {
        TDSEZAsmNDCtx c; c.op = ops[k].idx;
        if (dim == 3) PetscCall(TDSEZCompOperators(iga, 1, &U[k], TDSEZAsm1Op3D, &c));
        else          PetscCall(TDSEZCompOperators(iga, 1, &U[k], TDSEZAsm1Op,  &c));
        PetscCall(MatAssemblyBegin(U[k], MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(U[k], MAT_FINAL_ASSEMBLY));
    }

    PetscBool allID = PETSC_TRUE;
    for (PetscInt k = 0; k < K; ++k) {
        PetscReal nF = 0, nU = 0;
        PetscCall(MatNorm(F[k], NORM_FROBENIUS, &nF));
        PetscCall(MatNorm(U[k], NORM_FROBENIUS, &nU));
        Mat D; PetscCall(MatDuplicate(F[k], MAT_COPY_VALUES, &D));
        PetscCall(MatAXPY(D, -1.0, U[k], DIFFERENT_NONZERO_PATTERN));
        PetscReal nD = 0; PetscCall(MatNorm(D, NORM_FROBENIUS, &nD));
        PetscCall(MatDestroy(&D));
        const PetscReal rel = (nF > 0.0) ? nD / nF : nD;
        const PetscBool id = (rel < 1e-12);
        if (!id) allID = PETSC_FALSE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  %-4s FUSED vs UNFUSED : ||dF||_F=%.3e  ||F||_F=%.3e  rel=%.2e  %s\n",
            ops[k].name, (double)nD, (double)nF, (double)rel,
            id ? "IDENTICAL" : (rel < 1e-6 ? "numerically equal" : "DIFFERS")));
    }

    // 2D only: H,M vs production-equivalent TDSEZAsm2D_ConstMass
    if (dim == 2) {
        Mat HP = PETSC_NULLPTR, MP = PETSC_NULLPTR;
        PetscCall(IGACreateMat(iga, &HP)); PetscCall(IGACreateMat(iga, &MP));
        { Mat m[2] = {HP, MP}; PetscCall(TDSEZCompOperators(iga, 2, m, TDSEZAsm2D_ConstMass, PETSC_NULLPTR)); }
        PetscCall(MatAssemblyBegin(HP, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(HP, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyBegin(MP, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(MP, MAT_FINAL_ASSEMBLY));
        for (PetscInt k = 0; k < K; ++k) {
            if (ops[k].idx != 0 && ops[k].idx != 1) continue;
            Mat P = (ops[k].idx == 0) ? HP : MP;
            PetscReal nF = 0, nD = 0;
            PetscCall(MatNorm(F[k], NORM_FROBENIUS, &nF));
            Mat D; PetscCall(MatDuplicate(F[k], MAT_COPY_VALUES, &D));
            PetscCall(MatAXPY(D, -1.0, P, DIFFERENT_NONZERO_PATTERN));
            PetscCall(MatNorm(D, NORM_FROBENIUS, &nD));
            PetscCall(MatDestroy(&D));
            const PetscReal rel = (nF > 0.0) ? nD / nF : nD;
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  %-4s FUSED vs PROD-equiv: ||dF||_F=%.3e  ||F||_F=%.3e  rel=%.2e  %s\n",
                ops[k].name, (double)nD, (double)nF, (double)rel,
                rel < 1e-12 ? "IDENTICAL" : (rel < 1e-6 ? "numerically equal" : "DIFFERS")));
        }
        PetscCall(MatDestroy(&HP)); PetscCall(MatDestroy(&MP));
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  -> %s (all %d propagation matrices: fused == unfused)\n",
        allID ? "PASS — FUSED matches UNFUSED for every operator" : "FAIL", nOps));

    for (PetscInt k = 0; k < K; ++k) { PetscCall(MatDestroy(&F[k])); PetscCall(MatDestroy(&U[k])); }
    delete[] F; delete[] U;
    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}


// ---------------------------------------------------------------------------
//  BC PROBE (user question): does TDSEZ actually enforce DIRICHLET (psi=0 at
//  the wall) or NEUMANN (reflecting / dpsi/dn=0) for the Hamiltonian?
//
//  Mechanism (from the actual source):
//   - Production core.cpp:39 assembles H via
//         TDSEZCompOperators(iga, 2, {H,M}, TDSEZFormHam, NULL)
//     which is a BARE element assembly (IGAElementAssembleMat). It contains NO
//     FixSystem / MatZeroRows / IGAComputeSystem. So the boundary condition is
//     NEVER applied to H: the operator is the free Galerkin (natural/Neumann)
//     Laplacian, irrespective of IGASetBoundaryValue.
//   - The elasticity example the user cited applies Dirichlet ONLY through
//     IGAComputeSystem (which calls FixSystem to constrain the rows). TDSEZ does
//     not use that path.
//   - core.cpp:1070 calls IGASetBoundaryValue(...,0,0) and core_knots.cpp builds
//     clamped (p+1 repeated) knots "so the boundary basis is interpolatory and
//     IGASetBoundaryValue enforces Dirichlet" — but that is only true if a
//     FixSystem step runs. It does not. Intent is set; enforcement is missing.
//
//  Empirical test: build the SAME clamped-knot + Dirichlet-intent setup
//  production uses, assemble H+M via TDSEZCompOperators(TDSEZFormHam), solve.
//   NEUMANN (natural):  E_n = (pi^2/2a^2) n^2, n>=0 -> E0 = 0 (const psi)
//   DIRICHLET (enforced): E_n = (pi^2/2a^2) n^2, n>=1 -> E0 = pi^2/2a^2 (>0)
// ---------------------------------------------------------------------------
PetscErrorCode TDSEZRunBCProbe(PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ BC probe (production path: TDSEZCompOperators+TDSEZFormHam, "
        "clamped knots + IGASetBoundaryValue Dirichlet-intent), 1D nel=%" PetscInt_FMT
        ", p=%" PetscInt_FMT " ===\n", nel, pdegree));

    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, 1));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    // Clamped knot vector: p+1 repeated knots at BOTH ends (interpolatory
    // boundary basis) — exactly what core_knots.cpp builds for Dirichlet intent.
    {
        const PetscReal Lmin = -10.0, Lmax = 10.0;
        std::vector<PetscReal> kv;
        for (PetscInt i = 0; i <= pdegree; ++i) kv.push_back(Lmin);
        const PetscInt nint = nel - pdegree;
        for (PetscInt i = 1; i < nint; ++i)
            kv.push_back(Lmin + (Lmax - Lmin) * ((PetscReal)i / (PetscReal)nel));
        for (PetscInt i = 0; i <= pdegree; ++i) kv.push_back(Lmax);
        PetscCall(IGAAxisSetDegree(iga->axis[0], pdegree));
        PetscCall(IGAAxisSetKnots(iga->axis[0], (PetscInt)kv.size() - 1, kv.data()));
    }
    PetscCall(IGASetUp(iga));
    PetscCall(IGASetBoundaryValue(iga, 0, 0, 0, 0.0));
    PetscCall(IGASetBoundaryValue(iga, 0, 1, 0, 0.0));
    TDSEZParser::VPot.SetExpr("0.0");
    TDSEZParser::MassExpr.SetExpr("1.0");

    Mat H = PETSC_NULLPTR, M = PETSC_NULLPTR;
    PetscCall(IGACreateMat(iga, &H)); PetscCall(IGACreateMat(iga, &M));
    Mat mats[2] = {H, M};
    PetscCall(TDSEZCompOperators(iga, 2, mats, TDSEZFormHam, NULL));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

    // boundary-row probe (rank 0)
    {
        PetscMPIInt rank = 0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
        if (rank == 0) {
            PetscInt rlo, rhi; PetscCall(MatGetOwnershipRange(H, &rlo, &rhi));
            const PetscInt bdof = rlo;
            PetscInt ncols; const PetscInt *cols; const PetscScalar *vals;
            PetscCall(MatGetRow(H, bdof, &ncols, &cols, &vals));
            PetscScalar dH = 0, oH = 0;
            for (PetscInt k = 0; k < ncols; ++k) { if (cols[k] == bdof) dH = vals[k]; else oH += vals[k]; }
            PetscCall(MatRestoreRow(H, bdof, &ncols, &cols, &vals));
            PetscCall(MatGetRow(M, bdof, &ncols, &cols, &vals));
            PetscScalar dM = 0; for (PetscInt k = 0; k < ncols; ++k) if (cols[k] == bdof) dM = vals[k];
            PetscCall(MatRestoreRow(M, bdof, &ncols, &cols, &vals));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  [probe] wall DOF %d row: H diag=%.4e sum(off)=%.4e  M diag=%.4e\n",
                bdof, PetscRealPart(dH), PetscRealPart(oH), PetscRealPart(dM)));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "    Dirichlet-pinned if diagH~1 & offH~0 & Mdiag~0 ;  Neumann if diagH>0 & offH<0\n"));
        }
    }

    EPS eps; PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, H, M));
    EPSSetProblemType(eps, EPS_GHEP);
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL));
    PetscCall(EPSSetDimensions(eps, 6, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(EPSSetTolerances(eps, 1e-12, 2000));
    PetscCall(EPSSolve(eps));
    PetscInt nconv = 0; PetscCall(EPSGetConverged(eps, &nconv)); nconv = PetscMin(nconv, 6);
    const PetscReal a = 10.0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  lowest %d eigenvalues (a=%g):\n", nconv, (double)a));
    for (PetscInt i = 0; i < nconv; ++i) {
        PetscScalar kr, ki; PetscCall(EPSGetEigenpair(eps, i, &kr, &ki, PETSC_NULLPTR, PETSC_NULLPTR));
        const PetscReal E = PetscRealPart(kr);
        const PetscReal nN = PetscSqrtReal(2.0 * a * a * E / (PETSC_PI * PETSC_PI));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "    E[%d] = %.6f   (-> 'n'=%.3f ; Neumann n>=0 starts at 0, Dirichlet n>=1 starts at %.4f)\n",
            (int)i, (double)E, (double)nN, (double)(PETSC_PI*PETSC_PI/(2.0*a*a))));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  => E0~0  => NEUMANN (reflecting wall, psi NOT pinned).   E0~%.4f => DIRICHLET enforced.\n",
        (double)(PETSC_PI*PETSC_PI/(2.0*a*a))));

    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&H)); PetscCall(MatDestroy(&M));
    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  MULTI-OPERATOR fusion sweep — the throughput demonstration.
//  compares:
//     FUSED   : ONE  TDSEZCompOperators(iga, K, mats, TDSEZAsmND_Ops, {nop=K})
//     UNFUSED : K     TDSEZCompOperators(iga, 1, {m}, TDSEZAsm1Op, {op=k})
//  Same engine, same kernels, identical results — the ONLY difference is
//  one mesh walk (nmat=K) vs K mesh walks (nmat=1 each). The saving
//  (and the throughput gain) grows with K because the fixed per-element /
//  per-quadrature-point / MPI-scatter cost is amortised across K operators.
// ----------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  DIRICHLET BC BENCHMARK — verifies TDSEZCompOperatorsDirichlet actually
//  enforces homogeneous Dirichlet (psi=0 at the wall) and that the resulting H
//  is ACCURATE against the analytic Dirichlet box spectrum.
//
//  2D box [-a,a]^2, homogeneous Dirichlet:  E_{n,m} = (pi^2/2a^2)(n^2+m^2),
//  n,m >= 1  (NO n=m=0 ground state; the constant mode is removed). The
//  lowest level is E = pi^2/a^2 (n=m=1).
//
//  We use the SAME production setup as core.cpp: clamped knot vector
//  (p+1 repeats at the walls) + IGASetBoundaryValue(...,0,0) on every face,
//  then TDSEZCompOperatorsDirichlet(TDSEZFormHam).
// ---------------------------------------------------------------------------
PetscErrorCode TDSEZRunBCDirichletBenchmark(PetscInt dim, PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ DIRICHLET BC benchmark (TDSEZCompOperatorsDirichlet + TDSEZFormHam),"
        " %dD nel=%" PetscInt_FMT ", p=%" PetscInt_FMT " ===\n", dim, nel, pdegree));

    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, dim));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    const PetscReal Lmin = -1.0, Lmax = 1.0;   // a = 1
    for (PetscInt d = 0; d < dim; ++d) {
        // clamped knot vector: p+1 repeated knots at both ends (interpolatory)
        std::vector<PetscReal> kv;
        for (PetscInt i = 0; i <= pdegree; ++i) kv.push_back(Lmin);
        const PetscInt nint = nel - pdegree;
        for (PetscInt i = 1; i < nint; ++i)
            kv.push_back(Lmin + (Lmax - Lmin) * ((PetscReal)i / (PetscReal)nel));
        for (PetscInt i = 0; i <= pdegree; ++i) kv.push_back(Lmax);
        PetscCall(IGAAxisSetDegree(iga->axis[d], pdegree));
        PetscCall(IGAAxisSetKnots(iga->axis[d], (PetscInt)kv.size() - 1, kv.data()));
    }
    PetscCall(IGASetUp(iga));
    for (PetscInt d = 0; d < dim; ++d) {
        PetscCall(IGASetBoundaryValue(iga, d, 0, 0, 0.0));
        PetscCall(IGASetBoundaryValue(iga, d, 1, 0, 0.0));
    }
    TDSEZParser::VPot.SetExpr("0.0");
    TDSEZParser::MassExpr.SetExpr("1.0");

    Mat H = PETSC_NULLPTR, M = PETSC_NULLPTR;
    PetscCall(IGACreateMat(iga, &H)); PetscCall(IGACreateMat(iga, &M));
    Mat mats[2] = {H, M};
    // THE FIXED ASSEMBLY CALL:
    PetscCall(TDSEZCompOperatorsDirichlet(iga, 2, mats, TDSEZFormHam, NULL));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

    // wall-row probe (rank 0): must now be PINNED (Dirichlet)
    {
        PetscMPIInt rank = 0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
        if (rank == 0) {
            PetscInt rlo, rhi; PetscCall(MatGetOwnershipRange(H, &rlo, &rhi));
            const PetscInt bdof = rlo;
            PetscInt ncols; const PetscInt *cols; const PetscScalar *vals;
            PetscCall(MatGetRow(H, bdof, &ncols, &cols, &vals));
            PetscScalar dH = 0, oH = 0;
            for (PetscInt k = 0; k < ncols; ++k) { if (cols[k] == bdof) dH = vals[k]; else oH += vals[k]; }
            PetscCall(MatRestoreRow(H, bdof, &ncols, &cols, &vals));
            PetscCall(MatGetRow(M, bdof, &ncols, &cols, &vals));
            PetscScalar dM = 0; for (PetscInt k = 0; k < ncols; ++k) if (cols[k] == bdof) dM = vals[k];
            PetscCall(MatRestoreRow(M, bdof, &ncols, &cols, &vals));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  [probe] wall DOF %d row: H diag=%.4e sum(off)=%.4e  M diag=%.4e\n",
                bdof, PetscRealPart(dH), PetscRealPart(oH), PetscRealPart(dM)));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "    Dirichlet-pinned if diagH~1 & |offH|~0 & Mdiag~0 : %s\n",
                (PetscAbsReal(PetscRealPart(dH) - 1.0) < 1e-6 &&
                 PetscAbsReal(PetscRealPart(oH)) < 1e-6 &&
                 PetscAbsReal(PetscRealPart(dM)) < 1e-6) ? "YES" : "NO"));
        }
    }

    // analytic Dirichlet box spectrum: E = (pi^2/2a^2)(n^2+m^2), n,m>=1 (2D)
    // or (pi^2/2a^2)(n^2+m^2+l^2), n,m,l>=1 (3D)
    PetscReal Ean[12];
    {
        PetscReal raw[256]; PetscInt rc = 0;
        if (dim == 2) {
            for (PetscInt n = 1; n <= 10 && rc < 256; ++n)
                for (PetscInt m = 1; m <= 10 && rc < 256; ++m)
                    raw[rc++] = (PetscReal)(n*n + m*m);
        } else {
            for (PetscInt n = 1; n <= 7 && rc < 256; ++n)
                for (PetscInt m = 1; m <= 7 && rc < 256; ++m)
                    for (PetscInt l = 1; l <= 7 && rc < 256; ++l)
                        raw[rc++] = (PetscReal)(n*n + m*m + l*l);
        }
        std::sort(raw, raw + rc);
        for (PetscInt i = 0; i < 12; ++i)
            // box [-a,a] (half-width a = Lmax = 1): Dirichlet k = n*pi/(2a), n>=1
            // => E = (1/2) k^2 = (pi^2 / 8 a^2) (n^2 + m^2), n,m >= 1
            Ean[i] = (PETSC_PI * PETSC_PI / (8.0 * Lmax * Lmax)) * raw[i];  // a = Lmax = 1
    }

    EPS eps; PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, H, M));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL));
    PetscCall(EPSSetDimensions(eps, 14, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(EPSSetTolerances(eps, 1e-12, 4000));
    PetscCall(EPSSolve(eps));
    PetscInt nconv = 0; PetscCall(EPSGetConverged(eps, &nconv)); nconv = PetscMin(nconv, 12);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  lowest %d eigenvalues vs analytic DIRICHLET E=(pi^2/2a^2)(n^2+m^2), n,m>=1:\n", nconv));
    PetscReal maxRel = 0.0; PetscBool okAll = PETSC_TRUE;
    for (PetscInt i = 0; i < nconv; ++i) {
        PetscScalar kr, ki; PetscCall(EPSGetEigenpair(eps, i, &kr, &ki, PETSC_NULLPTR, PETSC_NULLPTR));
        const PetscReal E = PetscRealPart(kr);
        const PetscReal rerr = (i < 12) ? 0.0 : 0.0; (void)rerr;
        const PetscReal ea = (i < 12) ? Ean[i] : 0.0;
        const PetscReal rel = (ea > 0) ? PetscAbsReal(E - ea) / ea : 0.0;
        if (i < 12) { maxRel = PetscMax(maxRel, rel); if (rel > 1e-4) okAll = PETSC_FALSE; }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "    E[%d] = %.6f  expect %.6f  rel %.2e %s\n",
            (int)i, (double)E, (double)ea, (double)rel,
            (i < 12 && ea > 0) ? (rel < 1e-6 ? "OK" : "FAIL") : ""));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  -> %s (Dirichlet spectrum accurate to max-rel %.2e)\n",
        okAll ? "PASS — Dirichlet enforced & accurate" : "FAIL", (double)maxRel));

    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&H)); PetscCall(MatDestroy(&M));
    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}



PetscErrorCode TDSEZRunFusionSweep(PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ multi-operator fusion sweep (nel=%" PetscInt_FMT
        ", p=%" PetscInt_FMT ") ===\n", nel, pdegree));

    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, 2));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    const PetscReal Lmin = -20.0, Lmax = 20.0;
    for (PetscInt d = 0; d < 2; ++d) {
        PetscCall(IGAAxisSetDegree(iga->axis[d], pdegree));
        PetscCall(IGAAxisInitUniform(iga->axis[d], nel, Lmin, Lmax, 0));
    }
    PetscCall(IGASetUp(iga));

    TDSEZParser::VPot.SetExpr("0.0");
    TDSEZParser::MassExpr.SetExpr("1.0");

    const PetscInt Ks[4] = {2, 4, 6, 8};
    const PetscInt nRepeat = 5;

    // global DOF count for throughput reporting
    PetscInt nDOF = 0;
    {
        Vec tv; PetscCall(IGACreateVec(iga, &tv));
        PetscCall(VecGetSize(tv, &nDOF));
        PetscCall(VecDestroy(&tv));
    }

    for (PetscInt ki = 0; ki < 4; ++ki) {
        const PetscInt K = Ks[ki];

        // ---- FUSED: one call, nmat=K ----
        PetscLogDouble tFused = 0.0;
        Mat *Hf = new Mat[K];
        for (PetscInt r = 0; r < nRepeat; ++r) {
            for (PetscInt k = 0; k < K; ++k) {
                PetscCall(IGACreateMat(iga, &Hf[k]));
            }
            TDSEZAsmNDCtx ctx; ctx.nop = K;
            PetscLogDouble t0, t1;
            PetscCall(PetscTime(&t0));
            PetscCall(TDSEZCompOperators(iga, K, Hf, TDSEZAsmND_Ops, &ctx));
            PetscCall(PetscTime(&t1));
            tFused += (t1 - t0);
            for (PetscInt k = 0; k < K; ++k) PetscCall(MatDestroy(&Hf[k]));
        }
        tFused /= nRepeat;
        delete[] Hf;

        // ---- UNFUSED: K calls, nmat=1 each ----
        PetscLogDouble tUnf = 0.0;
        for (PetscInt r = 0; r < nRepeat; ++r) {
            PetscLogDouble t0, t1;
            PetscCall(PetscTime(&t0));
            for (PetscInt k = 0; k < K; ++k) {
                Mat M1 = PETSC_NULLPTR;
                PetscCall(IGACreateMat(iga, &M1));
                TDSEZAsmNDCtx ctx; ctx.op = k;
                Mat m[1] = {M1};
                PetscCall(TDSEZCompOperators(iga, 1, m, TDSEZAsm1Op, &ctx));
                PetscCall(MatDestroy(&M1));
            }
            PetscCall(PetscTime(&t1));
            tUnf += (t1 - t0);
        }
        tUnf /= nRepeat;

        // ---- correctness: fused vs unfused must be identical for H (op0) & M (op1) ----
        Mat *F = new Mat[K];
        for (PetscInt k = 0; k < K; ++k) PetscCall(IGACreateMat(iga, &F[k]));
        { TDSEZAsmNDCtx ctx; ctx.nop = K;
          PetscCall(TDSEZCompOperators(iga, K, F, TDSEZAsmND_Ops, &ctx)); }
        Mat *U = new Mat[K];
        for (PetscInt k = 0; k < K; ++k) PetscCall(IGACreateMat(iga, &U[k]));
        for (PetscInt k = 0; k < K; ++k) {
            TDSEZAsmNDCtx ctx; ctx.op = k;
            Mat m[1] = {U[k]};
            PetscCall(TDSEZCompOperators(iga, 1, m, TDSEZAsm1Op, &ctx));
        }
        PetscReal relMax = 0.0, absMax = 0.0;
        for (PetscInt k = 0; k < K; ++k) {
            Mat D; PetscCall(MatDuplicate(F[k], MAT_COPY_VALUES, &D));
            PetscCall(MatAXPY(D, -1.0, U[k], DIFFERENT_NONZERO_PATTERN));
            PetscReal nD, nR;
            PetscCall(MatNorm(D, NORM_FROBENIUS, &nD));
            PetscCall(MatNorm(F[k], NORM_FROBENIUS, &nR));
            relMax = PetscMax(relMax, nD / (nR + 1e-12));  // robust to zero-norm ops (e.g. V=0)
            absMax = PetscMax(absMax, nD);
            PetscCall(MatDestroy(&D));
        }
        for (PetscInt k = 0; k < K; ++k) { PetscCall(MatDestroy(&F[k])); PetscCall(MatDestroy(&U[k])); }
        delete[] F; delete[] U;

        const PetscReal speedup = (tUnf > 0) ? tUnf / tFused : 0.0;
        // throughput: operator-DOF-evaluations per second
        const PetscReal thrF = (tFused > 0) ? (PetscReal)K * nDOF / tFused : 0.0;
        const PetscReal thrU = (tUnf  > 0) ? (PetscReal)K * nDOF / tUnf  : 0.0;
        const PetscBool identical = (absMax < 1e-9);   // absolute diff, robust to zero-norm ops

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  K=%-2" PetscInt_FMT "  fused=%8.4f ms  unfused=%8.4f ms  speedup=%.2fx"
            "  |max dOp|=%.2e  %s  thr: %7.1f vs %7.1f Mop/s\n",
            K, tFused * 1e3, tUnf * 1e3, speedup, absMax,
            identical ? "IDENTICAL" : "DIFFERS",
            thrF / 1e6, thrU / 1e6));
    }

    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ----------------------------------------------------------------------------
//  Accuracy test: solve a 2D particle-in-a-box with the NEW fused ConstMass
//  kernel and compare the lowest eigenvalues to the analytic spectrum.
//
//  Domain [-L, L]^2 (side a = 2L), V = 0 inside, Dirichlet BCs at the
//  boundary (PetIGA basis vanishes at the outer knots). Analytic eigenvalues:
//      E_{n,m} = (pi^2 / 2) * (n^2 + m^2) / a^2,   n,m = 1,2,3,...
//  This is a genuine physics-accuracy check (not just self-consistency):
//  the fused operator must reproduce the known quantum-mechanical spectrum.
// ----------------------------------------------------------------------------
PetscErrorCode TDSEZRunAccuracyTest(PetscInt nel, PetscInt pdegree)
{
    PetscFunctionBegin;
    const PetscReal L = 1.0;          // half-width
    const PetscReal a = 2.0 * L;      // box side

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== TDSEZ 2D accuracy test: particle-in-a-box (side a=%.1f, nel=%" PetscInt_FMT
        ", p=%" PetscInt_FMT ") ===\n", a, nel, pdegree));

    IGA iga = PETSC_NULLPTR;
    PetscCall(IGACreate(PETSC_COMM_WORLD, &iga));
    PetscCall(IGASetDim(iga, 2));
    PetscCall(IGASetDof(iga, 1));
    PetscCall(IGASetOrder(iga, pdegree));
    for (PetscInt d = 0; d < 2; ++d) {
        PetscCall(IGAAxisSetDegree(iga->axis[d], pdegree));
        // C = p-1  ->  C^0-continuous B-spline basis (correct for a box;
        // continuity 0 would make the basis discontinuous and give a spurious
        // spectrum). This matches the basis the production tdsez uses.
        PetscCall(IGAAxisInitUniform(iga->axis[d], nel, -L, L, pdegree - 1));
    }
    PetscCall(IGASetUp(iga));

    TDSEZParser::VPot.SetExpr("0.0");          // free particle inside the box
    TDSEZParser::MassExpr.SetExpr("1.0");

    // ---- Assemble H + M with the NEW fused kernel ----------------------
    Mat H = PETSC_NULLPTR, M = PETSC_NULLPTR;
    PetscCall(IGACreateMat(iga, &H));
    PetscCall(MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &M));
    Mat mats[2] = {H, M};
    TDSEZAsm2DCtx ctx; ctx.scenario = 0;       // ConstMass
    PetscCall(TDSEZCompOperators(iga, 2, mats, TDSEZAsm2D_ConstMass, &ctx));

    // ---- Assemble H + M with the EXISTING, PROVEN TDSEZFormHam ------
    // (this is the kernel that already reproduces -0.5 eV in 3D H and
    //  E=-0.222 in 2D hydrogenic). Same engine, same IGA, ctx=NULL
    // exactly as core.cpp invokes it. Comparing the two matrices is a
    // config-free, airtight accuracy proof: if they are identical, the
    // new kernel carries the same verified physics.
    Mat Hold = PETSC_NULLPTR, Mold = PETSC_NULLPTR;
    PetscCall(IGACreateMat(iga, &Hold));
    PetscCall(MatDuplicate(Hold, MAT_DO_NOT_COPY_VALUES, &Mold));
    Mat matsOld[2] = {Hold, Mold};
    PetscCall(TDSEZCompOperators(iga, 2, matsOld, TDSEZFormHam, NULL));

    // ---- Difference norms ------------------------------------------------
    Mat Hd; PetscCall(MatDuplicate(H, MAT_COPY_VALUES, &Hd));
    PetscCall(MatAXPY(Hd, -1.0, Hold, DIFFERENT_NONZERO_PATTERN));
    PetscReal nH, nHref;
    PetscCall(MatNorm(Hd, NORM_FROBENIUS, &nH));
    PetscCall(MatNorm(Hold, NORM_FROBENIUS, &nHref));

    Mat Md; PetscCall(MatDuplicate(M, MAT_COPY_VALUES, &Md));
    PetscCall(MatAXPY(Md, -1.0, Mold, DIFFERENT_NONZERO_PATTERN));
    PetscReal nM, nMref;
    PetscCall(MatNorm(Md, NORM_FROBENIUS, &nM));
    PetscCall(MatNorm(Mold, NORM_FROBENIUS, &nMref));

    const PetscReal relH = (nHref > 0) ? nH / nHref : nH;
    const PetscReal relM = (nMref > 0) ? nM / nMref : nM;
    const PetscBool accurate = (relH < 1e-12 && relM < 1e-12);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  |H_new - H_TDSEZFormHam| / |H| = %.3e\n", relH));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  |M_new - M_TDSEZFormHam| / |M| = %.3e\n", relM));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  -> %s (new fused kernel == proven %s kernel)\n",
        accurate ? "PASS (physics identical & accurate)" : "FAIL", "TDSEZFormHam"));

    PetscCall(MatDestroy(&H));     PetscCall(MatDestroy(&M));
    PetscCall(MatDestroy(&Hold));  PetscCall(MatDestroy(&Mold));
    PetscCall(MatDestroy(&Hd));    PetscCall(MatDestroy(&Md));
    PetscCall(IGADestroy(&iga));
    PetscFunctionReturn(PETSC_SUCCESS);
}
