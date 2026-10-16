// ============================================================================
//  kh.cpp  —  static Kramers-Henneberger frame (Mode = kh)
// ----------------------------------------------------------------------------
//  In the frame of the free-carrier quiver alpha(t), m alpha'' = q E(t), the
//  Hamiltonian is p^2/2m + s V(r + alpha(t)); its cycle average
//
//      V_KH(r) = (1/P) int_window V(r + alpha(t)) dt
//
//  defines the static KH problem whose eigenstates are the dressed states of
//  high-frequency stabilisation. TDSEZParser::V() returns s V_KH when
//  KHAlpha_ holds the quiver at the composite Gauss-Legendre nodes of the
//  window (weights KHWeight_; exponentially convergent for a smooth quiver,
//  periodic or not).
//
//  The quiver is integrated from the laser expressions:
//    window   one optical cycle centred at the peak of |E(t)| in
//             [0, FinalTime] (period 2 pi/omega from Omegax/y/z, else from
//             the zero crossings of the field), or KHWindow = t0, t1;
//    alpha    within the window the mean velocity (the drift) and the mean
//             position are removed, leaving the pure quiver: for a
//             monochromatic field alpha(t) = -(q E0 / m w^2) sin(w t + phi).
//  With s = t - t0, J_n(s) = int_0^s sigma^n E(sigma) dsigma, Cauchy's formula
//  gives int_0^s int_0^u E = s J_0(s) - J_1(s); the window means follow from
//  J_0, J_1, J_2 at s = P, so alpha(t_k) needs only single integrals of E,
//  evaluated by composite 8-point Gauss-Legendre (accurate to rounding).
//  The number of nodes doubles from 16 until V_KH changes by less than 1e-12
//  (relative) on a test lattice (KHSamples = N fixes it, rounded up to 8).
// ============================================================================

#include "tdsez_internal.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const PetscReal gx[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                         0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363};
const PetscReal gw[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
                         0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};

bool coupled(int a) { return a < (int)TDSEZParser::Dimension && TDSEZParser::Coupled(a); }

PetscReal fieldNorm(PetscReal t)
{
    PetscReal e2 = 0.0;
    for (int a = 0; a < 3; ++a) if (coupled(a)) { const PetscReal e = TDSEZParser::Field(a, t); e2 += e * e; }
    return std::sqrt(e2);
}

/// Window [t0, t0 + P]: KHWindow, or one cycle centred at the peak of |E|.
void window(PetscReal &t0, PetscReal &P, PetscReal &tPeak, PetscReal &omega)
{
    if (TDSEZParser::KHWindow[1] > TDSEZParser::KHWindow[0]) {
        t0 = TDSEZParser::KHWindow[0];
        P = TDSEZParser::KHWindow[1] - t0;
        tPeak = t0 + 0.5 * P;
        omega = 2 * PETSC_PI / P;
        return;
    }
    const PetscReal T = TDSEZParser::FinalTime;
    const PetscInt n = 40000;
    PetscReal emax = 0.0;
    int dom = -1;
    PetscReal dmax = 0.0;
    tPeak = 0.0;
    for (PetscInt i = 0; i <= n; ++i) {
        const PetscReal t = T * i / n, e = fieldNorm(t);
        if (e > emax) { emax = e; tPeak = t; }
    }
    if (!(emax > 0.0)) throw std::runtime_error("Mode = kh: the laser field is zero in [0, FinalTime]");
    for (int a = 0; a < 3; ++a) {
        if (!coupled(a)) continue;
        const PetscReal e = std::fabs(TDSEZParser::Field(a, tPeak));
        if (e > dmax) { dmax = e; dom = a; }
    }
    const PetscReal om[3] = {TDSEZParser::Omegax, TDSEZParser::Omegay, TDSEZParser::Omegaz};
    omega = 0.0;
    for (int a = 0; a < 3; ++a) if (coupled(a)) omega = PetscMax(omega, om[a]);
    if (omega <= 0.0) {   // half periods between the first and the last sign change of the dominant axis
        PetscInt cross = 0;
        PetscReal prev = TDSEZParser::Field(dom, 0.0), tprev = 0.0, c0 = 0, c1 = 0, b0[2] = {0, 0}, b1[2] = {0, 0};
        for (PetscInt i = 1; i <= n; ++i) {
            const PetscReal t = T * i / n, e = TDSEZParser::Field(dom, t);
            if ((e > 0 && prev < 0) || (e < 0 && prev > 0)) {
                if (!cross) { b0[0] = tprev; b0[1] = t; }
                b1[0] = tprev; b1[1] = t;
                ++cross;
            }
            if (e != 0) { prev = e; tprev = t; }
        }
        // the first and last crossings to rounding (bisection on the bracket):
        // the window must be one period exactly for the trapezoid rule to
        // converge exponentially
        auto root = [dom](PetscReal a, PetscReal b) {
            PetscReal fa = TDSEZParser::Field(dom, a);
            for (int it = 0; it < 100 && b - a > 1e-15 * PetscMax((PetscReal)1.0, std::fabs(b)); ++it) {
                const PetscReal m = 0.5 * (a + b), fm = TDSEZParser::Field(dom, m);
                if ((fm > 0) == (fa > 0) && fm != 0) { a = m; fa = fm; } else b = m;
            }
            return 0.5 * (a + b);
        };
        if (cross >= 3) { c0 = root(b0[0], b0[1]); c1 = root(b1[0], b1[1]); }
        if (cross < 3)
            throw std::runtime_error("Mode = kh: cannot find the laser period (fewer than 3 zero crossings in "
                                     "[0, FinalTime]); set Omegax/y/z or KHWindow = t0, t1");
        omega = PETSC_PI * (cross - 1) / (c1 - c0);
    }
    P = 2 * PETSC_PI / omega;
    t0 = tPeak - 0.5 * P;
}

/// Averaging nodes over the window: composite 8-point Gauss-Legendre on nseg
/// equal segments (s in [0, P]); weights normalised to sum to 1.
void nodes(PetscReal P, PetscInt nseg, std::vector<PetscReal> &s, std::vector<PetscReal> &w)
{
    s.clear(); w.clear();
    const PetscReal h = P / nseg;
    for (PetscInt k = 0; k < nseg; ++k)
        for (int g = 0; g < 8; ++g) {
            s.push_back(k * h + 0.5 * h * (gx[g] + 1.0));
            w.push_back(0.5 * h * gw[g] / P);
        }
}

/// 8-point Gauss-Legendre of E_a(t0 + s) s^n, n = 0, 1, 2, over [sa, sb].
void moments(int a, PetscReal t0, PetscReal sa, PetscReal sb, PetscReal j[3])
{
    const PetscReal h = sb - sa;
    j[0] = j[1] = j[2] = 0.0;
    for (int g = 0; g < 8; ++g) {
        const PetscReal s = sa + 0.5 * h * (gx[g] + 1.0), w = 0.5 * h * gw[g];
        const PetscReal e = TDSEZParser::Field(a, t0 + s);
        j[0] += w * e; j[1] += w * s * e; j[2] += w * s * s * e;
    }
}

/// alpha(t0 + s_k) for every coupled axis (s sorted in [0, P]).
std::vector<std::array<PetscReal, 3>> quiver(PetscReal t0, PetscReal P, const std::vector<PetscReal> &s, PetscReal qm)
{
    const size_t N = s.size();
    std::vector<std::array<PetscReal, 3>> alpha(N, std::array<PetscReal, 3>{0.0, 0.0, 0.0});
    for (int a = 0; a < 3; ++a) {
        if (!coupled(a)) continue;
        // cumulative J_0, J_1 at the nodes; the gaps between nodes are short,
        // so one 8-point rule per gap integrates E to rounding
        std::vector<PetscReal> J0(N), J1(N);
        PetscReal c[3] = {0, 0, 0}, m[3], prev = 0.0;
        for (size_t k = 0; k < N; ++k) {
            moments(a, t0, prev, s[k], m);
            for (int i = 0; i < 3; ++i) c[i] += m[i];
            J0[k] = c[0]; J1[k] = c[1];
            prev = s[k];
        }
        moments(a, t0, prev, P, m);
        const PetscReal j0 = c[0] + m[0], j1 = c[1] + m[1], j2 = c[2] + m[2];
        const PetscReal meanV  = j0 - j1 / P;                                  // (1/P) int_0^P J_0
        const PetscReal meanA2 = (0.5 * P * P * j0 - P * j1 + 0.5 * j2) / P;   // (1/P) int_0^P (s J_0 - J_1)
        const PetscReal meanRaw = meanA2 - meanV * 0.5 * P;
        for (size_t k = 0; k < N; ++k) {
            const PetscReal raw = s[k] * J0[k] - J1[k] - meanV * s[k];
            alpha[k][(size_t)a] = qm * (raw - meanRaw);
        }
    }
    return alpha;
}

/// V_KH on a lattice of points in the domain (sign included).
std::vector<PetscReal> sampleVKH()
{
    const PetscReal lo[3] = {TDSEZParser::LMinX, TDSEZParser::LMinY, TDSEZParser::LMinZ};
    const PetscReal hi[3] = {TDSEZParser::LMaxX, TDSEZParser::LMaxY, TDSEZParser::LMaxZ};
    const int n = 9, dim = (int)TDSEZParser::Dimension;
    std::vector<PetscReal> v;
    const int ny = dim >= 2 ? n : 1, nz = dim == 3 ? n : 1;
    for (int i = 0; i < n; ++i) for (int j = 0; j < ny; ++j) for (int l = 0; l < nz; ++l) {
        const int idx[3] = {i, j, l};
        PetscReal r[3] = {0, 0, 0};
        for (int d = 0; d < dim; ++d) r[d] = lo[d] + (hi[d] - lo[d]) * (0.05 + 0.9 * idx[d] / (n - 1.0));
        v.push_back(TDSEZParser::V(r[0], r[1], r[2]));
    }
    return v;
}

} // namespace

void TDSEZSetupKH()
{
    TDSEZParser::KHAlpha_.clear();
    if (!TDSEZParser::KHFrame) return;

    PetscReal t0, P, tPeak, omega;
    window(t0, P, tPeak, omega);
    const PetscReal mass = TDSEZParser::MassDist(0.0, 0.0, 0.0);
    const PetscReal qm = TDSEZParser::Q / mass;   // m alpha'' = q E

    // composite Gauss-Legendre over the window: exponential convergence for a
    // smooth quiver, periodic (CW) or not (pulse envelope across the cycle)
    PetscInt nseg = TDSEZParser::KHSamples > 0 ? (TDSEZParser::KHSamples + 7) / 8 : 2;
    PetscReal change = 0.0;
    bool converged = TDSEZParser::KHSamples > 0;
    std::vector<PetscReal> s, w;
    auto build = [&]() {
        nodes(P, nseg, s, w);
        TDSEZParser::KHAlpha_ = quiver(t0, P, s, qm);
        TDSEZParser::KHWeight_ = w;
    };
    build();
    if (!converged) {
        std::vector<PetscReal> prev = sampleVKH();
        while (nseg < 1024) {
            nseg *= 2;
            build();
            const std::vector<PetscReal> cur = sampleVKH();
            PetscReal scale = 0.0;
            change = 0.0;
            for (size_t i = 0; i < cur.size(); ++i) {
                scale  = PetscMax(scale, std::fabs(cur[i]));
                change = PetscMax(change, std::fabs(cur[i] - prev[i]));
            }
            change /= PetscMax(scale, (PetscReal)1e-300);
            prev = cur;
            if (change <= 1e-12) { converged = true; break; }
        }
    }
    const PetscInt N = (PetscInt)TDSEZParser::KHAlpha_.size();

    PetscReal amp[3] = {0, 0, 0}, rms[3] = {0, 0, 0};
    for (const auto &al : TDSEZParser::KHAlpha_)
        for (int a = 0; a < 3; ++a) amp[a] = PetscMax(amp[a], std::fabs(al[(size_t)a]));
    for (int a = 0; a < 3; ++a) rms[a] = 0.0;
    for (size_t k = 0; k < TDSEZParser::KHAlpha_.size(); ++k)
        for (int a = 0; a < 3; ++a) rms[a] += TDSEZParser::KHWeight_[k] * TDSEZParser::KHAlpha_[k][(size_t)a] * TDSEZParser::KHAlpha_[k][(size_t)a];
    for (int a = 0; a < 3; ++a) rms[a] = std::sqrt(rms[a]);

    TDSEZInfo info;
    info.header("KRAMERS-HENNEBERGER FRAME");
    info.line("Mode = kh: eigenstates of the cycle-averaged potential V_KH(r) = < V(r + alpha(t)) > over the "
              "free-carrier quiver m alpha'' = q E(t); no propagation.");
    info.kv("Window", "t = %.6g .. %.6g a.u. (%s)", (double)t0, (double)(t0 + P),
            TDSEZParser::KHWindow[1] > TDSEZParser::KHWindow[0] ? "KHWindow" : "one cycle at the field peak");
    info.kv("Period, omega", "%.6g a.u., %.6g a.u.", (double)P, (double)omega);
    for (int a = 0; a < (int)TDSEZParser::Dimension; ++a) {
        if (!coupled(a)) continue;
        char lab[32];
        snprintf(lab, sizeof(lab), "Quiver %c", "xyz"[a]);
        info.kv(lab, "amplitude %.6g a.u., rms %.6g a.u.", (double)amp[a], (double)rms[a]);
    }
    if (TDSEZParser::KHSamples > 0)
        info.kv("Samples per cycle", "%d (KHSamples)", (int)N);
    else
        info.kv("Samples per cycle", "%d (V_KH converged to %.1e)", (int)N, (double)change);
    if (!converged)
        info.warn("V_KH did not converge to 1e-12 with %d samples per cycle (last change %.1e): the potential may "
                  "be singular on the quiver path; set KHSamples or smooth the potential", (int)N, (double)change);
    TDSEZKHSummary() = {t0, t0 + P, omega, amp[0], amp[1], amp[2], (PetscInt)TDSEZParser::KHAlpha_.size()};
}

TDSEZKHInfo &TDSEZKHSummary()
{
    static TDSEZKHInfo s;
    return s;
}
