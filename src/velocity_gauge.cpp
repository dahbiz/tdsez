// ============================================================================
//  velocity_gauge.cpp  —  Gauge = velocity
// ----------------------------------------------------------------------------
//  H = (p - q A(t - n.r/c))^2 / 2m + V, with A' = -E (A(0) = 0). The length-
//  gauge term s E.r is switched off (TDSEZParser::Coupling returns 0) and the
//  field enters through coupling terms, in the list that also holds the
//  FieldTerms (coupling.cpp), so the propagator, its fused stage updates and
//  the monitors are the ones of the length gauge. With s = n.r / c,
//  A(t - s) = A + s E - s^2 E'/2 + O(s^3) and, dropping the constant A^2:
//
//    dipole          - (q/m) A.p
//    order 1/c       - (q/m) (n.r/c) E.p          + (q^2/m) (n.r/c) A.E
//    order 1/c^2     + (q/2m) (n.r/c)^2 E'.p      + (q^2/2m) (n.r/c)^2 (E.E - A.E')
//
//  (Multipole = 1, 2, 3). v.p terms are the Hermitian first-order forms of
//  coupling.cpp; n.r commutes with p_a since n is transverse to the field.
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

PetscReal dotn(const PetscReal *r)
{
    const PetscReal *n = TDSEZParser::PropagationDir;
    return n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
}

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
    const PetscReal c = TDSEZParser::SpeedOfLight;
    const int order = (int)TDSEZParser::Multipole;
    const char *xyz = "xyz";
    auto unit = [](int a) { return [a](const PetscReal *, PetscReal *v) { v[0] = v[1] = v[2] = 0.0; v[a] = 1.0; }; };
    auto AE = [](PetscReal t) {                        // A.E and E.E - A.E' over the coupled axes
        PetscReal ae = 0.0, ee = 0.0;
        for (int a = 0; a < 3; ++a) {
            if (!coupled(a)) continue;
            const PetscReal A = TDSEZVectorPotential(a, t), E = TDSEZParser::Field(a, t);
            ae += A * E;
            ee += E * E - A * TDSEZParser::FieldDerivative(a, t, 1);
        }
        return std::make_pair(ae, ee);
    };

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
    if (order >= 2) {                                  // 1/c
        for (int a = 0; a < 3; ++a) {
            if (!coupled(a)) continue;
            T t;
            t.kind = T::Gradient;
            t.label = std::string("velocity gauge 1/c, (n.r) E_") + xyz[a] + " p_" + xyz[a];
            t.w = [](const PetscReal *r) { return dotn(r); };
            t.v = unit(a);
            t.coeff = [a, q, m, c](PetscReal tt) { return -q / (m * c) * TDSEZParser::Field(a, tt); };
            terms.push_back(t);
        }
        T t;
        t.label = "velocity gauge 1/c, (n.r) A.E";
        t.w = [](const PetscReal *r) { return dotn(r); };
        t.coeff = [q, m, c, AE](PetscReal tt) { return q * q / (m * c) * AE(tt).first; };
        terms.push_back(t);
    }
    if (order >= 3) {                                  // 1/c^2
        for (int a = 0; a < 3; ++a) {
            if (!coupled(a)) continue;
            T t;
            t.kind = T::Gradient;
            t.label = std::string("velocity gauge 1/c^2, (n.r)^2 E'_") + xyz[a] + " p_" + xyz[a];
            t.w = [](const PetscReal *r) { const PetscReal u = dotn(r); return u * u; };
            t.v = unit(a);
            t.coeff = [a, q, m, c](PetscReal tt) { return q / (2 * m * c * c) * TDSEZParser::FieldDerivative(a, tt, 1); };
            terms.push_back(t);
        }
        T t;
        t.label = "velocity gauge 1/c^2, (n.r)^2 (E.E - A.E')";
        t.w = [](const PetscReal *r) { const PetscReal u = dotn(r); return u * u; };
        t.coeff = [q, m, c, AE](PetscReal tt) { return q * q / (2 * m * c * c) * AE(tt).second; };
        terms.push_back(t);
    }

    if (order >= 3) TDSEZWarnFieldKinks(0, "1/c^2 velocity-gauge terms (dE/dt)");

    TDSEZInfo info;
    info.section("VELOCITY GAUGE");
    info.kv("Hamiltonian", "(p - q A(t%s))^2 / 2m + V, A' = -E", order >= 2 ? " - n.r/c" : "");
    info.kv("Order", "%s", order == 1 ? "dipole" : (order == 2 ? "1/c" : "1/c^2"));
    if (order >= 2)
        info.kv("n, c", "(%g, %g, %g), %.9g a.u.", (double)TDSEZParser::PropagationDir[0],
                (double)TDSEZParser::PropagationDir[1], (double)TDSEZParser::PropagationDir[2], (double)c);
    info.kv("A(t) table", "%zu nodes, step %.3g a.u. (Gauss-Legendre, Hermite)",
            table().A[0].size() + table().A[1].size() + table().A[2].size(), (double)table().h);
}
