#ifndef TDSEZ_FORCECOEFFS_HPP
#define TDSEZ_FORCECOEFFS_HPP

#include "tdsez_internal.hpp"
#include <algorithm>
#include <string>
#include <vector>

// Shared by the sum-factorised assembly (sumfact.cpp) and the quadrature
// expectation values of the monitors (quadobs.cpp): both must use the same
// coefficients for the force matrices.

/// Coefficients of the force matrix of axis a with the mass-gradient
/// corrections of TDSEZformPotentialGrad{X,Y,Z} (and TDSEZFormPhyX), f = 1/m:
///
///   dV_a(i,j) = sum_q w [ -(dV/dx_a) f B_i B_j - 1/4 sum_t c_t D^al_t B_i D^be_t B_j ],
///
///   (B, B):          sum_e f_e f_ea + f sum_e f_eea
///   (B, d_a B):      2 |grad f|^2 + 2 f lap f - 2 (f_a^2 + f f_aa)
///   (B, d_e B):      2 (f_e f_a + f f_ae)                 e != a
///   (B, d_a d_e B):  4 f f_e                              e != a
///   (d_e B, d_e B):  2 f f_a for e != a, -2 f f_a for e = a
///
/// (third derivatives read with the index order of each kernel). Arrays per
/// point, weights included; pairs (x,y), (x,z), (y,z).
struct ForceCoeffs {
    std::vector<PetscReal> BB, Bd[3], Bp[3], dd[3];
    void Resize(size_t nq)
    {
        BB.assign(nq, 0.0);
        for (int d = 0; d < 3; ++d) { Bd[d].assign(nq, 0.0); Bp[d].assign(nq, 0.0); dd[d].assign(nq, 0.0); }
    }
};

inline int PairIndex(int i, int j) { return i + j - 1; }   // (0,1)->0, (0,2)->1, (1,2)->2 (i < j)

inline void AddForceCorrections(int a, PetscInt dim, PetscReal f, const PhysicsDerivativesFull &ops, PetscReal w,
                         size_t q, ForceCoeffs &C)
{
    const PetscInt n = (PetscInt)ops.first.size();
    auto F1 = [&](int i) { return i < dim ? ops.first[i] : 0.0; };
    auto F2 = [&](int i, int j) {                                  // kernels read (min, max)
        if (i > j) std::swap(i, j);
        return (j < dim) ? ops.second[i * n + j] : 0.0;
    };
    auto F3 = [&](int i, int j, int k) { return (i < dim && j < dim && k < dim) ? ops.third[(i * n + j) * n + k] : 0.0; };
    // sum_e f_eea in the index order of each kernel
    PetscReal t3 = 0.0;
    if (a == 0)      t3 = F3(0, 0, 0) + F3(0, 1, 1) + F3(0, 2, 2);
    else if (a == 1) t3 = F3(0, 0, 1) + F3(1, 1, 1) + F3(1, 2, 2);
    else             t3 = F3(2, 2, 2) + F3(1, 1, 2) + F3(0, 0, 2);
    PetscReal g2 = 0.0, lap = 0.0, mix = 0.0;
    for (int e = 0; e < dim; ++e) { g2 += F1(e) * F1(e); lap += F2(e, e); mix += F1(e) * F2(e, a); }
    const PetscReal s = -0.25 * w;
    C.BB[q] += s * (mix + f * t3);
    for (int e = 0; e < dim; ++e) {
        if (e == a) {
            C.Bd[e][q] += s * (2.0 * g2 + 2.0 * f * lap - 2.0 * (F1(a) * F1(a) + f * F2(a, a)));
            C.dd[e][q] += s * (-2.0 * f * F1(a));
        } else {
            C.Bd[e][q] += s * 2.0 * (F1(e) * F1(a) + f * F2(a, e));
            C.Bp[PairIndex(std::min(a, e), std::max(a, e))][q] += s * 4.0 * f * F1(e);
            C.dd[e][q] += s * 2.0 * f * F1(a);
        }
    }
}

/// Whether the Mass expression depends on x, y or z (cached per expression).
inline bool MassVaries()
{
    static std::string expr;
    static bool varies = true, valid = false;
    if (!valid || expr != TDSEZParser::MassExpr.GetExpr()) {
        try { varies = !TDSEZParser::MassExpr.GetUsedVar().empty(); } catch (const mu::Parser::exception_type &) { varies = true; }
        expr = TDSEZParser::MassExpr.GetExpr(); valid = true;
    }
    return varies;
}


/// Force coefficients of the owned elements, kept by the sum-factorised
/// assembly for the quadrature expectation values of the monitors
/// (tdsez_quadobs.hpp), so the 1/m derivatives are evaluated once per point.
/// Element-major (first axis fastest), then the element's points (q1 fastest).
struct ForceCoeffStore {
    PetscInt start[3] = {0, 0, 0}, width[3] = {1, 1, 1}, nq = 0;
    bool axis[3] = {false, false, false};
    std::vector<char> done;          // per element
    size_t filled = 0;
    ForceCoeffs F[3];
    bool Complete() const { return nq > 0 && filled == done.size(); }
    void Release() { *this = ForceCoeffStore(); }
};
ForceCoeffStore &TDSEZForceCoeffStore();

#endif // TDSEZ_FORCECOEFFS_HPP
