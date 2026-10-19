// ============================================================================
//  coupling.cpp  —  time-dependent coupling terms beyond E.r
// ----------------------------------------------------------------------------
//  H(t) = H + s E(t).r + sum_i c_i(t) G_i. The terms G_i, c_i are
//
//  * FieldTerm lines: G = <B| w_k |B>, c = s f_k(t)            (separable fields)
// ============================================================================

#include "tdsez_internal.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using T = TDSEZCouplingTerm;

bool coupled(int a) { return a < (int)TDSEZParser::Dimension && TDSEZParser::Coupled(a); }

} // namespace

/// Warn when the derivative of order `order` of the field (0: E itself) jumps
/// in [0, FinalTime]: the terms that use the next derivative then contain a
/// delta impulse that no time grid samples correctly. Only a log message.
void TDSEZWarnFieldKinks(int order, const char *terms)
{
    const PetscReal T = TDSEZParser::FinalTime;
    if (!(T > 0)) return;
    PetscReal omega = 0.0;
    const PetscReal om[3] = {TDSEZParser::Omegax, TDSEZParser::Omegay, TDSEZParser::Omegaz};
    for (int a = 0; a < 3; ++a) if (coupled(a)) omega = PetscMax(omega, om[a]);
    const PetscReal h = PetscMin((PetscReal)2e-3, omega > 0 ? 2 * PETSC_PI / omega / 3000 : (PetscReal)2e-3);
    const size_t n = (size_t)std::ceil(T / h) + 1;
    if (n > 4000000) return;
    for (int a = 0; a < 3; ++a) {
        if (!coupled(a)) continue;
        // differences of order+1 of the raw field: O(h^(order+1)) where smooth,
        // O(h^order * jump) at a jump of the order-th derivative
        std::vector<PetscReal> d(n);
        PetscReal big = 0.0;
        for (size_t k = 0; k < n; ++k) {
            d[k] = TDSEZParser::Field(a, PetscMin(T, k * h));
            big = PetscMax(big, std::fabs(d[k]));
        }
        for (int r = 0; r <= order; ++r)
            for (size_t k = 0; k + 1 + r < n; ++k) d[k] = d[k + 1] - d[k];
        const size_t m = n - 1 - (size_t)order;
        const PetscReal floor = 1e-4 * big * std::pow(h, (PetscReal)order);
        for (size_t k = 3; k + 3 < m; ++k) {
            const PetscReal j = std::fabs(d[k]);
            const PetscReal nb = PetscMax(std::fabs(d[k - 3]), std::fabs(d[k + 3]));
            if (j > floor && j > 30 * nb) {
                TDSEZInfo info;
                info.warn("%s jumps near t = %.4g a.u. on %c; the %s then contain an impulse that the time steps do "
                          "not resolve. Use a smoother envelope%s.",
                          order == 0 ? "The laser field" : "The time derivative of the laser field",
                          (double)(k * h), "xyz"[a], terms,
                          TDSEZParser::VelocityGauge ? "" : " or Gauge = velocity");
                break;
            }
        }
    }
}

void TDSEZSetupCouplingTerms()
{
    auto &terms = TDSEZParser::CouplingTerms_;
    terms.clear();

    // separable fields
    for (PetscInt k = 0; k < TDSEZParser::NFieldTerms(); ++k) {
        T t;
        t.label = "FieldTerm " + std::to_string((int)k + 1) + ": " + TDSEZParser::FieldTermProfile[k];
        t.w = [k](const PetscReal *r) { return TDSEZParser::FieldProfile(k, r[0], r[1], r[2]); };
        t.coeff = [k](PetscReal tt) { return TDSEZParser::FieldTermCoupling(k, tt); };
        terms.push_back(t);
    }
    if (TDSEZParser::VelocityGauge) {            // all field terms of the velocity gauge
        if (TDSEZParser::EnablePropagation) TDSEZSetupVelocityGauge();
        return;
    }
}
