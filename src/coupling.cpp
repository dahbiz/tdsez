// ============================================================================
//  coupling.cpp  —  time-dependent coupling terms beyond the dipole
// ----------------------------------------------------------------------------
//  H(t) = H + s E(t).r + sum_i c_i(t) G_i. The terms G_i, c_i are
//
//  * FieldTerm lines: G = <B| w_k |B>, c = s f_k(t)            (separable fields)
//  * Multipole = 2, 3: the nondipole coupling of a plane wave
//    E(r, t) = E(t - n.r/c), B = n x E / c, in the multipolar (Power-Zienau-
//    Woolley) gauge, expanded about the nucleus in powers of 1/c. With the
//    charge q, the mass m and C(t) = n x E(t):
//
//      order 1/c    electric quadrupole   + q/(2c)   (n.r)(r.E')
//                   magnetic dipole       - q/(2mc)  L.C
//      order 1/c^2  electric octupole     - q/(6c^2) (n.r)^2 (r.E'')
//                   magnetic quadrupole   + q/(3mc^2) {(n.r), L}/2 . C'
//                   diamagnetic           + q^2/(8mc^2) |C x r|^2
//
//    from phi = -int_0^1 r.E(lr) dl, A = -int_0^1 l r x B(lr) dl and
//    E(lr, t) = sum_k (-l n.r/c)^k E^(k)(t) / k!. E' and E'' are 5-point
//    central differences of the laser expressions (step 2e-3/omega:
//    truncation ~1e-11, rounding ~1e-13 relative). L = -i r x grad.
// ============================================================================

#include "tdsez_internal.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

using T = TDSEZCouplingTerm;

PetscReal dotn(const PetscReal *r)
{
    const PetscReal *n = TDSEZParser::PropagationDir;
    return n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
}

/// v of L_b = -i (r x grad)_b: L_x = y d_z - z d_y, L_y = z d_x - x d_z, L_z = x d_y - y d_x.
void lfield(int b, const PetscReal *r, PetscReal *v)
{
    v[0] = v[1] = v[2] = 0.0;
    if (b == 0) { v[1] = -r[2]; v[2] = r[1]; }
    if (b == 1) { v[0] = r[2]; v[2] = -r[0]; }
    if (b == 2) { v[0] = -r[1]; v[1] = r[0]; }
}

bool coupled(int a) { return a < (int)TDSEZParser::Dimension && TDSEZParser::Coupled(a); }

/// (n x e_a)_b: C_b(t) = sum_a (n x e_a)_b E_a(t).
PetscReal nxe(int a, int b)
{
    const PetscReal *n = TDSEZParser::PropagationDir;
    PetscReal e[3] = {0, 0, 0}, c[3];
    e[a] = 1.0;
    c[0] = n[1] * e[2] - n[2] * e[1];
    c[1] = n[2] * e[0] - n[0] * e[2];
    c[2] = n[0] * e[1] - n[1] * e[0];
    return c[b];
}

/// C_b(t) (order 0) or C_b'(t) (order 1).
PetscReal Cb(int b, PetscReal t, int order)
{
    PetscReal s = 0.0;
    for (int a = 0; a < 3; ++a)
        if (coupled(a) && nxe(a, b) != 0.0)
            s += nxe(a, b) * (order == 0 ? TDSEZParser::Field(a, t) : TDSEZParser::FieldDerivative(a, t, 1));
    return s;
}

bool Cactive(int b)
{
    for (int a = 0; a < 3; ++a) if (coupled(a) && nxe(a, b) != 0.0) return true;
    return false;
}

} // namespace

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
    if (TDSEZParser::Multipole < 2 || !TDSEZParser::EnablePropagation) return;

    // 5-point derivative step from the carrier frequency
    PetscReal omega = 0.0;
    const PetscReal om[3] = {TDSEZParser::Omegax, TDSEZParser::Omegay, TDSEZParser::Omegaz};
    for (int a = 0; a < 3; ++a) if (coupled(a)) omega = PetscMax(omega, om[a]);
    if (omega <= 0.0) {
        const TDSEZFieldScales &fs = TDSEZEstimateFieldScales();
        if (fs.valid) omega = fs.omega;
    }
    if (omega <= 0.0) omega = 1.0;
    TDSEZParser::FieldDerivStep_ = 2e-3 / omega;

    const PetscReal q = TDSEZParser::Q, c = TDSEZParser::SpeedOfLight;
    const PetscReal m = TDSEZParser::MassDist(0.0, 0.0, 0.0);
    const bool mag = TDSEZParser::MultipoleMagnetic;
    const char *xyz = "xyz";

    // order 1/c
    for (int a = 0; a < 3; ++a) {
        if (!coupled(a)) continue;
        T t;
        t.label = std::string("electric quadrupole, ") + xyz[a] + " (n.r)";
        t.w = [a](const PetscReal *r) { return r[a] * dotn(r); };
        t.coeff = [a, q, c](PetscReal tt) { return q / (2 * c) * TDSEZParser::FieldDerivative(a, tt, 1); };
        terms.push_back(t);
    }
    for (int b = 0; b < 3 && mag; ++b) {
        if (!Cactive(b)) continue;
        T t;
        t.kind = T::Gradient;
        t.label = std::string("magnetic dipole, L_") + xyz[b];
        t.w = [](const PetscReal *) { return 1.0; };
        t.v = [b](const PetscReal *r, PetscReal *v) { lfield(b, r, v); };
        t.coeff = [b, q, m, c](PetscReal tt) { return -q / (2 * m * c) * Cb(b, tt, 0); };
        terms.push_back(t);
    }
    if (TDSEZParser::Multipole < 3) return;

    // order 1/c^2
    for (int a = 0; a < 3; ++a) {
        if (!coupled(a)) continue;
        T t;
        t.label = std::string("electric octupole, ") + xyz[a] + " (n.r)^2";
        t.w = [a](const PetscReal *r) { const PetscReal u = dotn(r); return r[a] * u * u; };
        t.coeff = [a, q, c](PetscReal tt) { return -q / (6 * c * c) * TDSEZParser::FieldDerivative(a, tt, 2); };
        terms.push_back(t);
    }
    for (int b = 0; b < 3 && mag; ++b) {
        if (!Cactive(b)) continue;
        T t;
        t.kind = T::Gradient;
        t.label = std::string("magnetic quadrupole, {n.r, L_") + xyz[b] + "}/2";
        t.w = [](const PetscReal *r) { return dotn(r); };
        t.v = [b](const PetscReal *r, PetscReal *v) { lfield(b, r, v); };
        t.coeff = [b, q, m, c](PetscReal tt) { return q / (3 * m * c * c) * Cb(b, tt, 1); };
        terms.push_back(t);
    }
    for (int a = 0; a < 3 && mag; ++a)
        for (int b = a; b < 3; ++b) {
            if (!Cactive(a) || !Cactive(b)) continue;
            T t;
            t.label = std::string("diamagnetic, C_") + xyz[a] + " C_" + xyz[b];
            t.w = [a, b](const PetscReal *r) {   // (delta_ab r^2 - x_a x_b), twice off the diagonal
                const PetscReal r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                return a == b ? r2 - r[a] * r[a] : -2 * r[a] * r[b];
            };
            t.coeff = [a, b, q, m, c](PetscReal tt) { return q * q / (8 * m * c * c) * Cb(a, tt, 0) * Cb(b, tt, 0); };
            terms.push_back(t);
        }

    TDSEZInfo info;
    info.section("MULTIPOLE COUPLING");
    info.kv("Order", "%s, plane wave E(t - n.r/c), multipolar gauge",
            TDSEZParser::Multipole == 2 ? "quadrupole (1/c)" : "octupole (1/c^2)");
    info.kv("n, c", "(%g, %g, %g), %.9g a.u.", (double)TDSEZParser::PropagationDir[0],
            (double)TDSEZParser::PropagationDir[1], (double)TDSEZParser::PropagationDir[2], (double)c);
    info.kv("Magnetic terms", "%s", mag ? "included" : "off (MultipoleMagnetic = 0)");
    info.kv("dE/dt step", "%.3g a.u. (5-point)", (double)TDSEZParser::FieldDerivStep_);
}
