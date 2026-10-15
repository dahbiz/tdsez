#include "tdsez_internal.hpp"
#include <cctype>
#include <cstdlib>

/**
 * @file core_knots.cpp
 * @brief Knot vector generation algorithms for B-spline/IGA basis
 *        (log-tan, exponential-symmetric, tanh N/U-shaped, linear, etc.).
 * @author TDSEZ Project
 */




/// @brief Generate a clamped B-spline knot vector with log-tan graded
///        interior knots, symmetric about the origin. Interior knots are
///        mapped via a scaled tangent function for dense sampling near
///        the Coulomb centre.
/// @param Lmin       Left domain boundary.
/// @param Lmax       Right domain boundary.
/// @param ninterior  Number of interior knots.
/// @param alpha      Grading parameter controlling knot concentration
///                   (larger alpha concentrates more knots near x=0).
/// @param p          Spline degree (boundary multiplicity = p+1).
/// @return Vector of knot values (size ninterior + 2*(p+1)).
std::vector<PetscReal> TDSEZCore::TDSEZLogTanKnots(
    PetscReal Lmin, PetscReal Lmax,
    PetscInt  ninterior,
    PetscReal alpha,
    PetscInt  p)
{
    // Match symexp: nbnd = p+1, total = ninterior + 2*(p+1)
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);

    // Left boundary: p+1 knots at Lmin
    for (PetscInt i = 0; i < nbnd; ++i)
        knots[i] = Lmin;

    // Right boundary: p+1 knots at Lmax
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i)
        knots[i] = Lmax;

    // Interior knots: log-tan graded, symmetric about 0
    // t uniform in (-1,+1), excluding endpoints
    // safe_alpha keeps argument of tan away from ±π/2
    PetscReal safe_alpha = 1.57 * (alpha / (alpha + 1.0));

    for (PetscInt i = 0; i < ninterior; ++i)
    {
        PetscReal t = 2.0 * (i + 1.0) / (PetscReal)(ninterior + 1) - 1.0;
        PetscReal knot_val = Lmax * PetscTanReal(safe_alpha * t)
                                  / PetscTanReal(safe_alpha);
        knots[nbnd + i] = knot_val;
    }

    if (ninterior % 2 == 0)
        PetscPrintf(PETSC_COMM_WORLD,
            "TDSEZLogTanKnots: WARNING ninterior=%d even — "
            "no knot at x=0.\n", (int)ninterior);

    // NO push_back
    return knots;
}


/// @brief Generate a clamped B-spline knot vector with exponentially graded
///        interior knots, symmetric about the origin. Interior knots use an
///        exponential map concentrated near the centre for Coulomb cusp
///        resolution.
/// @param Lmin       Left domain boundary (negative for symmetric domain).
/// @param Lmax       Right domain boundary.
/// @param ninterior  Number of interior knots.
/// @param alpha      Exponential grading strength.
/// @param p          Spline degree (boundary multiplicity = p+1).
/// @return Vector of knot values (size ninterior + 2*(p+1)).
std::vector<PetscReal> TDSEZCore::TDSEZExpSymKnots(
    PetscReal Lmin, PetscReal Lmax,
    PetscInt  ninterior,
    PetscReal alpha,
    PetscInt  p)
{
    // Clamped B-spline knot vector: p+1 repeated at each boundary
    // Total knots = (p+1) + ninterior + (p+1) = ninterior + 2*(p+1)
    const PetscInt nbnd    = p + 1;
    const PetscInt nknots  = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);

    // Left boundary: p+1 knots at Lmin
    for (PetscInt i = 0; i < nbnd; ++i)
        knots[i] = Lmin;

    // Right boundary: p+1 knots at Lmax
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i)
        knots[i] = Lmax;

    // Interior knots: exponentially graded, symmetric about 0
    // t in (-1, +1), s = |t|, mapped exponentially to (0, 1)
    // then scaled to (0, Lmax) or (Lmin, 0)
    for (PetscInt i = 0; i < ninterior; ++i)
    {
        // t uniform in (-1,+1), excluding endpoints
        PetscReal t = 2.0 * (i + 1.0) / (PetscReal)(ninterior + 1) - 1.0;
        PetscReal s = PetscAbsReal(t);

        // Exponential map: 0→0, 1→1, concentrated near 0
        PetscReal mapped = (PetscExpReal(alpha * s) - 1.0)
                         / (PetscExpReal(alpha)     - 1.0);

        knots[nbnd + i] = (t >= 0.0) ? mapped * Lmax : mapped * Lmin;
        // Note: Lmin is negative for symmetric domain, so mapped*Lmin is negative
    }

    // Warn if origin is not represented exactly (even ninterior)
    if (ninterior % 2 == 0)
        PetscPrintf(PETSC_COMM_WORLD,
            "TDSEZExpSymKnots: WARNING ninterior=%d is even — "
            "no knot at x=0. Coulomb cusp under-resolved.\n",
            (int)ninterior);

    return knots;   // NO push_back
}






// N shaped grading: dense at center, sparse at boundaries, but with smoother transition than tanh
/// @brief Generate a clamped B-spline knot vector with tanh N-shaped graded
///        interior knots (dense at centre, sparse at boundaries, smoother
///        transition than tanh). Uses a tanh-based map symmetric about the
///        origin.
/// @param Lmin       Left domain boundary.
/// @param Lmax       Right domain boundary.
/// @param ninterior  Number of interior knots.
/// @param beta       Tanh grading parameter.
/// @param p          Spline degree (boundary multiplicity = p+1).
/// @return Vector of knot values (size ninterior + 2*(p+1)).
std::vector<PetscReal> TDSEZCore::TDSEZTanNSymKnots(
    PetscReal Lmin, PetscReal Lmax,
    PetscInt  ninterior,
    PetscReal beta,
    PetscInt  p)
{
    // Match symexp: nbnd = p+1, total = ninterior + 2*(p+1)
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);

    // Left boundary: p+1 knots at Lmin
    for (PetscInt i = 0; i < nbnd; ++i)
        knots[i] = Lmin;

    // Right boundary: p+1 knots at Lmax
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i)
        knots[i] = Lmax;

    // Interior knots: tanh-graded, symmetric about 0
    for (PetscInt i = 0; i < ninterior; ++i)
    {
        PetscReal t      = 2.0*(i+1)/(ninterior+1) - 1.0;
        PetscReal mapped = 0.5*(1.0 + PetscTanhReal(beta*t)
                                     / PetscTanhReal(beta));
        knots[nbnd + i]  = Lmin + mapped*(Lmax - Lmin);
    }

    // Warn if no knot at origin
    if (ninterior % 2 == 0)
        PetscPrintf(PETSC_COMM_WORLD,
            "TDSEZTanNSymKnots: WARNING ninterior=%d is even — "
            "no knot at x=0.\n", (int)ninterior);

    // NO push_back
    return knots;
}



// U shaped tanh grading: dense at center, sparse at boundaries
/// @brief Generate a clamped B-spline knot vector with tanh U-shaped graded
///        interior knots (dense at centre, sparse at boundaries). Uses a
///        tanh-based map with U-shaped density profile.
/// @param Lmin       Left domain boundary.
/// @param Lmax       Right domain boundary.
/// @param ninterior  Number of interior knots.
/// @param beta       Tanh grading parameter.
/// @param p          Spline degree (boundary multiplicity = p+1).
/// @return Vector of knot values (size ninterior + 2*(p+1)).
std::vector<PetscReal> TDSEZCore::TDSEZTanUSymKnots(
    PetscReal Lmin, PetscReal Lmax,
    PetscInt  ninterior,
    PetscReal beta,
    PetscInt  p)
{
    // Match symexp: nbnd = p+1, total = ninterior + 2*(p+1)
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);

    // Left boundary: p+1 knots at Lmin
    for (PetscInt i = 0; i < nbnd; ++i)
        knots[i] = Lmin;

    // Right boundary: p+1 knots at Lmax
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i)
        knots[i] = Lmax;

    // Interior knots: atanh-graded, dense at center, sparse at boundaries
    const PetscReal tanhBeta = PetscTanhReal(beta);
    for (PetscInt i = 0; i < ninterior; ++i)
    {
        // t uniform in (-1, +1)
        PetscReal t      = 2.0*(i+1)/(ninterior+1) - 1.0;

        // atanh map: compresses points toward center (x=0)
        // mapped in (0,1), with maximum density at mapped=0.5 (x=0)
        PetscReal mapped = 0.5*(1.0 + PetscAtanhReal(t * tanhBeta) / beta);

        knots[nbnd + i]  = Lmin + mapped*(Lmax - Lmin);
    }

    // Warn if no knot at origin
    if (ninterior % 2 == 0)
        PetscPrintf(PETSC_COMM_WORLD,
            "TDSEZTanUSymKnots: WARNING ninterior=%d is even — "
            "no knot at x=0.\n", (int)ninterior);

    // NO push_back
    return knots;
}





// ── Interface-aware knot extraction from a piecewise (?:) expression ──────
// Parses a muParser-style ternary expression such as
//   (abs(x) < 37.794) ? 0.084 : ((abs(x) < 151.176) ? 0.067 : 0.084)
// and returns the x-positions where the piecewise definition changes
// (the material interfaces). Recognised comparisons, with c a numeric
// literal on the right-hand side:
//   x < c, x <= c, x > c, x >= c                 -> breakpoint c
//   abs(x) < c, fabs(x) < c (and <=, >, >=)      -> breakpoints -c, +c
// The operand is the token immediately left of the operator, so mixed
// abs(x)/x branches inside one nested ternary are classified independently.
// Comparisons on any other operand are ignored.
/// @brief Parse a muParser-style piecewise ternary expression and extract
///        the x-positions where the piecewise definition changes (material
///        interfaces). Supports nested ternaries and comparison operators.
/// @param expr  The piecewise expression string (e.g. with ?: operators).
/// @return Sorted, de-duplicated vector of breakpoint x-positions.
std::vector<PetscReal> TDSEZExtractBreakpoints(const std::string& expr)
{
    // Tokenise on whitespace and on the structural characters ? : ( ) < >,
    // keeping <= and >= as single tokens. A leading sign on a number is kept
    // with the number ("x<-188.97" -> "x" "<" "-188.97").
    std::vector<std::string> toks;
    std::string cur;
    auto flush = [&]() { if (!cur.empty()) { toks.push_back(cur); cur.clear(); } };
    for (size_t i = 0; i < expr.size(); ++i) {
        const char ch = expr[i];
        if (std::isspace((unsigned char)ch)) { flush(); continue; }
        if ((ch == '<' || ch == '>') && i + 1 < expr.size() && expr[i + 1] == '=') {
            flush(); toks.push_back(std::string(1, ch) + "="); ++i; continue;
        }
        if (ch == '?' || ch == ':' || ch == '(' || ch == ')' || ch == '<' || ch == '>') {
            flush(); toks.push_back(std::string(1, ch)); continue;
        }
        cur.push_back(ch);
    }
    flush();

    auto to_num = [](const std::string& s, PetscReal& v) -> bool {
        if (s.empty()) return false;
        char* end = nullptr;
        const double d = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0') return false;
        v = (PetscReal)d;
        return true;
    };

    std::vector<PetscReal> bp;
    for (size_t i = 1; i + 1 < toks.size(); ++i) {
        const std::string& op = toks[i];
        if (op != "<" && op != "<=" && op != ">" && op != ">=") continue;
        PetscReal c;
        if (!to_num(toks[i + 1], c)) continue;
        if (toks[i - 1] == "x") {
            bp.push_back(c);
        } else if (toks[i - 1] == ")" && i >= 4 && toks[i - 2] == "x" &&
                   toks[i - 3] == "(" && (toks[i - 4] == "abs" || toks[i - 4] == "fabs")) {
            bp.push_back(-c); bp.push_back(c);
        }
    }
    std::sort(bp.begin(), bp.end());
    std::vector<PetscReal> out;
    for (PetscReal x : bp)
        if (out.empty() || std::fabs(x - out.back()) > 1e-9) out.push_back(x);
    return out;
}


/// @brief Generate a clamped B-spline knot vector whose element boundaries
///        coincide with the material interfaces, each carried as a knot of
///        multiplicity p. The basis is then C0 at every interface, which lets
///        the Galerkin solution carry the BDD kink (psi continuous,
///        (1/m) psi' continuous, psi' discontinuous). Elsewhere the basis is
///        C^{p-1}.
/// @param Lmin       Left domain boundary.
/// @param Lmax       Right domain boundary.
/// @param ninterior  Number of interior knots of the equivalent uniform mesh
///                   (Nelements - 1); sets the total element budget.
/// @param p          Spline degree (boundary multiplicity = p+1).
/// @param expr       Piecewise potential expression string.
/// @param massexpr   Piecewise mass distribution expression string.
/// @return Knot vector. Its length exceeds ninterior + 2(p+1) by (p-1) per
///         interface; the caller passes the actual size to IGAAxisSetKnots.
/// @note Elements are distributed over the segments between interfaces in
///       proportion to segment length (largest-remainder rounding, at least
///       one element per segment) and are uniform within each segment.
std::vector<PetscReal> TDSEZCore::TDSEZInterfaceKnots(
    PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscInt p,
    const std::string& expr, const std::string& massexpr)
{
    const PetscInt nbnd      = p + 1;
    const PetscInt nelements = ninterior + 1;
    const PetscReal tol      = 1e-9 * (Lmax - Lmin);

    // 1) Interfaces from the potential AND the mass expression (union).
    std::vector<PetscReal> iface = TDSEZExtractBreakpoints(expr);
    for (PetscReal x : TDSEZExtractBreakpoints(massexpr)) iface.push_back(x);
    std::sort(iface.begin(), iface.end());
    std::vector<PetscReal> brk;   // Lmin, interfaces strictly inside, Lmax
    brk.push_back(Lmin);
    for (PetscReal x : iface)
        if (x > Lmin + tol && x < Lmax - tol && x > brk.back() + tol) brk.push_back(x);
    brk.push_back(Lmax);
    const PetscInt nseg = (PetscInt)brk.size() - 1;

    if (nseg == 1)
        PetscPrintf(PETSC_COMM_WORLD,
            "TDSEZInterfaceKnots: no breakpoints found in expression; "
            "using uniform interior knots.\n");
    if (nelements < nseg)
        throw std::runtime_error("TDSEZInterfaceKnots: fewer elements than material segments");

    // 2) Elements per segment: proportional to length, >= 1, summing to nelements.
    std::vector<PetscInt>  nel(nseg, 1);
    std::vector<PetscReal> rem(nseg, 0.0);
    PetscInt assigned = 0;
    for (PetscInt s = 0; s < nseg; ++s) {
        const PetscReal share = (PetscReal)nelements * (brk[s + 1] - brk[s]) / (Lmax - Lmin);
        nel[s] = std::max<PetscInt>(1, (PetscInt)std::floor(share));
        rem[s] = share - std::floor(share);
        assigned += nel[s];
    }
    while (assigned < nelements) {          // hand out the remainder
        const PetscInt s = (PetscInt)(std::max_element(rem.begin(), rem.end()) - rem.begin());
        ++nel[s]; rem[s] = -1.0; ++assigned;
    }
    while (assigned > nelements) {          // only possible via the >= 1 floor
        PetscInt s = -1;
        for (PetscInt k = 0; k < nseg; ++k)
            if (nel[k] > 1 && (s < 0 || nel[k] > nel[s])) s = k;
        if (s < 0) break;
        --nel[s]; --assigned;
    }

    // 3) Assemble: clamped ends, uniform knots inside each segment,
    //    multiplicity p at each interior breakpoint.
    std::vector<PetscReal> knots;
    knots.reserve(2 * nbnd + nelements + nseg * p);
    for (PetscInt i = 0; i < nbnd; ++i) knots.push_back(Lmin);
    for (PetscInt s = 0; s < nseg; ++s) {
        const PetscReal a = brk[s], b = brk[s + 1];
        for (PetscInt i = 1; i < nel[s]; ++i) knots.push_back(a + (b - a) * (PetscReal)i / (PetscReal)nel[s]);
        if (s + 1 < nseg)
            for (PetscInt k = 0; k < p; ++k) knots.push_back(b);
    }
    for (PetscInt i = 0; i < nbnd; ++i) knots.push_back(Lmax);

    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZInterfaceKnots: %d C0 interface(s) (multiplicity p=%d) at",
        (int)(nseg - 1), (int)p);
    for (PetscInt s = 1; s < nseg; ++s) PetscPrintf(PETSC_COMM_WORLD, " %.6g", (double)brk[s]);
    PetscPrintf(PETSC_COMM_WORLD, " | %d elements\n", (int)nelements);
    return knots;
}





/// @brief Generate a hydrogenic knot vector: linear near the origin and
///        exponential near the outer boundary. Combines a uniformly spaced
///        inner region (for Coulomb/nuclear cusp resolution) with an
///        exponentially graded outer region.
/// @param Lmax   Outer domain boundary.
/// @param n_lin  Number of linearly spaced knots in the inner region.
/// @param r1     Width of the linear region.
/// @param n_exp  Number of exponentially graded knots in the outer region.
/// @param p      Spline degree (boundary multiplicity = p+1).
/// @param hmax   Largest element width (0: no cap). Where the exponential
///               elements would grow wider, the rest of the half-axis is
///               filled with uniform elements of width <= hmax.
/// @return Vector of knot values.
std::vector<PetscReal> TDSEZCore::TDSEZHydrogenicKnots(
    PetscReal Lmax,
    PetscInt  n_lin,
    PetscReal r1,
    PetscInt  n_exp,
    PetscInt  p,
    PetscReal hmax)
{
    // ── safety checks ───────────────────────────────────────────────
    if (n_lin <= 0 || n_exp <= 0 || p <= 0)
        throw std::runtime_error("Invalid knot parameters");

    PetscReal r_cross = n_lin * r1;

    if (r_cross >= Lmax)
        throw std::runtime_error(
            "TDSEZHydrogenicKnots: r_cross >= Lmax");

    // use standard math log for determinism across PETSc builds
    PetscReal log_ratio = std::log(Lmax / r_cross);

    // optional: deterministic rounding helper (IMPORTANT for MPI stability)
    auto clean = [](PetscReal x) -> PetscReal {
        return std::round(x * 1e12) / 1e12;
    };

    // ── Half-axis [0, Lmax] ─────────────────────────────────────────
    std::vector<PetscReal> half;
    half.reserve(n_lin + n_exp + 1);

    for (PetscInt i = 0; i <= n_lin; ++i)
        half.push_back(clean(i * r1));

    // Exponential grading from r_cross towards Lmax: k = 1..n_exp-1, so each
    // half-axis has exactly n_lin linear and n_exp exponential elements.
    // k = 0 would repeat r_cross (already the last linear knot); the clamp
    // below then turned the duplicate into a 1e-9*Lmax micro-element. k = n_exp
    // would land on Lmax and collide with the p+1 boundary repeats.
    for (PetscInt k = 1; k < n_exp; ++k)
        half.push_back(clean(r_cross * std::exp(k * log_ratio / n_exp)));

    // Width cap: from the first element wider than hmax on, uniform elements
    // of width <= hmax up to Lmax (the exponential tail grows to
    // Lmax (1 - r_cross/Lmax)^(1/n_exp)-wide elements, too coarse for
    // ionised or excited electrons in a large box).
    PetscInt n_capped = 0;
    if (hmax > 0) {
        half.push_back(Lmax);
        for (size_t j = 0; j + 1 < half.size(); ++j) {
            if (half[j + 1] - half[j] <= hmax) continue;
            const PetscReal r  = half[j];
            const PetscInt  nu = (PetscInt)std::ceil((Lmax - r) / hmax - 1e-9);
            half.resize(j + 1);
            for (PetscInt k = 1; k <= nu; ++k) half.push_back(clean(r + k * (Lmax - r) / nu));
            n_capped = nu;
            break;
        }
        half.pop_back();                                 // Lmax is added as the boundary knot
    }

    PetscInt n_half = (PetscInt)half.size();

    // ── Mirror to [-Lmax, Lmax] ─────────────────────────────────────
    std::vector<PetscReal> interior;
    interior.reserve(2 * (n_half - 1) + 1);

    for (PetscInt i = n_half - 1; i >= 1; --i)
        interior.push_back(clean(-half[i]));

    interior.push_back(0.0);

    for (PetscInt i = 1; i < n_half; ++i)
        interior.push_back(clean(half[i]));

    // ── Clamped knot vector ─────────────────────────────────────────
    // Endpoint multiplicity p+1 (NOT p): required so the boundary-span B-spline
    // is interpolatory and IGASetBoundaryValue(...,0,0.0) enforces homogeneous
    // Dirichlet psi=0 at the wall. p repeats would leave the boundary leaky.
    PetscInt ninterior = (PetscInt)interior.size();
    PetscInt nknots    = ninterior + 2 * (p + 1);

    std::vector<PetscReal> knots;
    knots.reserve(nknots);

    for (PetscInt i = 0; i < p + 1; ++i)
        knots.push_back(clean(-Lmax));

    for (auto x : interior)
        knots.push_back(x);

    // ── clamp interior strictly inside (Lmin, Lmax) ───────────────────
    // The exp-graded tail can land exactly on Lmax (or -Lmax), which would
    // collide with the p+1 boundary repeats below and give a knot of
    // multiplicity > p+1 that PetIGA rejects ("multiplicity > degree").
    // Pull any interior knot at/over the boundary strictly inside.
    {
        const PetscInt  nbnd = p + 1;
        const PetscReal eps  = 1e-9 * Lmax;
        for (size_t i = nbnd; i + (size_t)nbnd < knots.size(); ++i)
        {
            if (knots[i] <= -Lmax + eps) knots[i] = -Lmax + eps;
            if (knots[i] >=  Lmax - eps) knots[i] =  Lmax - eps;
        }
        // re-enforce strict increase after clamping
        for (size_t i = nbnd + 1; i + (size_t)nbnd < knots.size(); ++i)
            if (knots[i] <= knots[i - 1]) knots[i] = knots[i - 1] + eps;
    }

    for (PetscInt i = 0; i < p + 1; ++i)
        knots.push_back(clean(Lmax));

    // ── sanity check (tolerance-based, not strict) ───────────────────
    const PetscReal eps = 1e-12;

    for (PetscInt i = 1; i < nknots; ++i)
    {
        if (knots[i] < knots[i - 1] - eps)
        {
            throw std::runtime_error(
                "TDSEZHydrogenicKnots: non-monotone at i="
                + std::to_string(i));
        }
    }

    // ── diagnostics ──────────────────────────────────────────────────
    PetscReal hlargest = 0;
    for (PetscInt i = 1; i < n_half; ++i) hlargest = std::max(hlargest, half[i] - half[i - 1]);
    hlargest = std::max(hlargest, Lmax - half[n_half - 1]);
    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZHydrogenicKnots: %d elems | %d interior | p=%d | "
        "h_min=%.4f au | h_max=%.3f au%s | r_cross=%.2f au | Lmax=%.1f au\n",
        (int)(ninterior + 1),
        (int)ninterior,
        (int)p,
        (double)r1,
        (double)hlargest,
        n_capped ? " (capped)" : "",
        (double)r_cross,
        (double)Lmax);

    return knots;
}













