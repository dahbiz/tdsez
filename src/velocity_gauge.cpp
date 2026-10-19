// ============================================================================
//  velocity_gauge.cpp  —  Gauge = velocity
// ----------------------------------------------------------------------------
//  H = (p - q A(t))^2 / 2m + V, with A' = -E (A(0) = 0). The length-
//  gauge term s E.r is switched off (TDSEZParser::Coupling returns 0) and the
//  field enters through coupling terms, in the list that also holds the
//  FieldTerms (coupling.cpp), so the propagator, its fused stage updates and
//  the monitors are the ones of the length gauge. Dropping the constant A^2,
//  the coupling is - (q/m) A.p, the Hermitian first-order form of coupling.cpp.
//
//  A(t) = -int_0^t E: 8-point Gauss-Legendre on a uniform table (step
//  <= 0.01 a.u. and <= T/400 of the carrier), cubic Hermite interpolation with
//  the exact slope -E at the nodes (error ~ h^4 |E'''| / 384).
// ============================================================================

#include "tdsez_internal.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace {

const PetscReal gx[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                         0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363};
const PetscReal gw[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
                         0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};

struct Table {
    PetscReal h = 0.0;
    std::vector<PetscReal> A[3];
};
Table &table() { static Table t; return t; }

bool coupled(int a) { return a < (int)TDSEZParser::Dimension && TDSEZParser::Coupled(a); }

void buildTable(PetscReal omega)
{
    Table &tb = table();
    const PetscReal tEnd = TDSEZParser::FinalTime + 20 * TDSEZParser::TimeStep + 1.0;
    PetscReal h = 0.01;
    if (omega > 0) h = PetscMin(h, 2 * PETSC_PI / omega / 400);
    const size_t n = (size_t)std::ceil(tEnd / h) + 1;
    tb.h = tEnd / (PetscReal)(n - 1);
    for (int a = 0; a < 3; ++a) {
        tb.A[a].assign(coupled(a) ? n : 0, 0.0);
        if (!coupled(a)) continue;
        PetscReal acc = 0.0;
        for (size_t k = 1; k < n; ++k) {
            const PetscReal t0 = (k - 1) * tb.h;
            for (int g = 0; g < 8; ++g)
                acc += 0.5 * tb.h * gw[g] * TDSEZParser::Field(a, t0 + 0.5 * tb.h * (gx[g] + 1.0));
            tb.A[a][k] = -acc;
        }
    }
}

} // namespace

PetscReal TDSEZVectorPotential(int a, PetscReal t)
{
    const Table &tb = table();
    if (a < 0 || a > 2 || tb.A[a].empty() || t <= 0.0) return 0.0;
    const std::vector<PetscReal> &A = tb.A[a];
    PetscReal u = t / tb.h;
    size_t k = (size_t)u;
    if (k >= A.size() - 1) k = A.size() - 2;          // beyond the table: extrapolate the last cubic
    const PetscReal s = u - (PetscReal)k, t0 = k * tb.h, t1 = t0 + tb.h;
    const PetscReal d0 = -TDSEZParser::Field(a, t0) * tb.h, d1 = -TDSEZParser::Field(a, t1) * tb.h;
    const PetscReal s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * A[k] + (s3 - 2 * s2 + s) * d0 + (-2 * s3 + 3 * s2) * A[k + 1] + (s3 - s2) * d1;
}

void TDSEZSetupVelocityGauge()
{
    using T = TDSEZCouplingTerm;
    auto &terms = TDSEZParser::CouplingTerms_;

    PetscReal omega = 0.0;
    const PetscReal om[3] = {TDSEZParser::Omegax, TDSEZParser::Omegay, TDSEZParser::Omegaz};
    for (int a = 0; a < 3; ++a) if (coupled(a)) omega = PetscMax(omega, om[a]);
    if (omega <= 0.0) {
        const TDSEZFieldScales &fs = TDSEZEstimateFieldScales();
        if (fs.valid) omega = fs.omega;
    }
    buildTable(omega);
    TDSEZParser::FieldDerivStep_ = 2e-3 / (omega > 0 ? omega : 1.0);

    const PetscReal q = TDSEZParser::Q, m = TDSEZParser::MassDist(0.0, 0.0, 0.0);
    const char *xyz = "xyz";
    auto unit = [](int a) { return [a](const PetscReal *, PetscReal *v) { v[0] = v[1] = v[2] = 0.0; v[a] = 1.0; }; };
    for (int a = 0; a < 3; ++a) {                      // dipole: -(q/m) A_a p_a
        if (!coupled(a)) continue;
        T t;
        t.kind = T::Gradient;
        t.label = std::string("velocity gauge, A_") + xyz[a] + " p_" + xyz[a];
        t.w = [](const PetscReal *) { return 1.0; };
        t.v = unit(a);
        t.coeff = [a, q, m](PetscReal tt) { return -q / m * TDSEZVectorPotential(a, tt); };
        terms.push_back(t);
    }

    TDSEZInfo info;
    info.section("VELOCITY GAUGE");
    info.kv("Hamiltonian", "(p - q A(t))^2 / 2m + V, A' = -E (dipole)");
    info.kv("A(t) table", "%zu nodes, step %.3g a.u. (Gauss-Legendre, Hermite)",
            table().A[0].size() + table().A[1].size() + table().A[2].size(), (double)table().h);
}
