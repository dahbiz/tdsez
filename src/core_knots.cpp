#include "tdsez_internal.hpp"

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
// (the material interfaces). Supports nested ternaries, parentheses, and the
// comparison/arithmetic operators muParser uses for geometry specs.
// Comparison forms recognised:  x <  c, x <= c, x >  c, x >= c,
//   abs(x) <  c, abs(x) <= c, abs(x) > c, abs(x) >= c,
//   f(x) < c  (generic unary call) — constant pulled from RHS.
/// @brief Parse a muParser-style piecewise ternary expression and extract
///        the x-positions where the piecewise definition changes (material
///        interfaces). Supports nested ternaries and comparison operators.
/// @param expr  The piecewise expression string (e.g. with ?: operators).
/// @return Vector of breakpoint x-positions.
/// @note Full implementation saved in dev/core_knots_implementations.cpp
///       for a future push. Returns empty (falls back to uniform).
std::vector<PetscReal> TDSEZExtractBreakpoints(const std::string& expr)
{
    (void)expr;
    return {};
}


/// @brief Generate a clamped B-spline knot vector with interface-aware knot
///        placement. Inserts breakpoints extracted from a piecewise mass
///        expression, and adds extra knots at material interfaces.
/// @param Lmin       Left domain boundary.
/// @param Lmax       Right domain boundary.
/// @param ninterior  Number of interior knots.
/// @param p          Spline degree (boundary multiplicity = p+1).
/// @param expr       Piecewise potential/geometry expression string.
/// @param massexpr   Piecewise mass distribution expression string.
/// @return Vector of knot values with interface breakpoints included.
/// @note Full implementation saved in dev/core_knots_implementations.cpp
///       for a future push. Falls back to uniform interior knots.
std::vector<PetscReal> TDSEZCore::TDSEZInterfaceKnots(
    PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscInt p,
    const std::string& expr, const std::string& massexpr)
{
    (void)expr; (void)massexpr;
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);
    for (PetscInt i = 0; i < nbnd; ++i) knots[i] = Lmin;
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i) knots[i] = Lmax;
    for (PetscInt i = 0; i < ninterior; ++i)
        knots[nbnd + i] = Lmin + (i + 1.0) / (ninterior + 1.0) * (Lmax - Lmin);
    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZInterfaceKnots: implementation pending — using uniform interior.\n");
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
/// @return Vector of knot values.
std::vector<PetscReal> TDSEZCore::TDSEZHydrogenicKnots(
    PetscReal Lmax,
    PetscInt  n_lin,
    PetscReal r1,
    PetscInt  n_exp,
    PetscInt  p)
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

    // exp grading from r_cross up to (but NOT reaching) Lmax: use k=0..n_exp-1
    // so the largest knot is r_cross*exp((n_exp-1)*log_ratio/n_exp) < Lmax.
    // Reaching Lmax exactly would collide with the p+1 boundary repeats below
    // and trip PetIGA's "multiplicity > degree" check.
    for (PetscInt k = 0; k < n_exp; ++k)
        half.push_back(clean(r_cross * std::exp(k * log_ratio / n_exp)));

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
    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZHydrogenicKnots: %d elems | %d interior | p=%d | "
        "h_min=%.4f au | r_cross=%.2f au | Lmax=%.1f au\n",
        (int)(ninterior + 1),
        (int)ninterior,
        (int)p,
        (double)r1,
        (double)r_cross,
        (double)Lmax);

    return knots;
}



// ── Adaptive (potential-driven, single-pass) knot vector ────────────────────
// Full implementation saved in dev/core_knots_implementations.cpp for a future
// push. Stub falls back to uniform interior knots.
/// @brief Generate an adaptive knot vector driven by the potential gradient.
/// @note Full implementation saved in dev/core_knots_implementations.cpp
///       for a future push. Falls back to uniform interior knots.
std::vector<PetscReal> TDSEZCore::TDSEZAdaptiveKnots(
    PetscReal Lmin, PetscReal Lmax,
    PetscInt  ninterior, PetscInt p,
    PetscInt  axis,            // 0->x, 1->y, 2->z selects dVPotX/Y/Z
    PetscReal kappa,           // gradient-weighting strength (default 1.0)
    PetscReal power)           // sharpening exponent (default 1.0)
{
    (void)axis; (void)kappa; (void)power;
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);
    for (PetscInt i = 0; i < nbnd; ++i) knots[i] = Lmin;
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i) knots[i] = Lmax;
    for (PetscInt i = 0; i < ninterior; ++i)
        knots[nbnd + i] = Lmin + (i + 1.0) / (ninterior + 1.0) * (Lmax - Lmin);
    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZAdaptiveKnots[%d]: implementation pending — using uniform interior.\n",
        (int)axis);
    return knots;
}


// ── Adaptive (density-driven, two-pass) knot vector ──────────────────────────
// Full implementation saved in dev/core_knots_implementations.cpp for a future
// push. Stub falls back to uniform interior knots.
/// @brief Generate an adaptive knot vector from a pre-computed wavefunction
///        density histogram.
/// @note Full implementation saved in dev/core_knots_implementations.cpp
///       for a future push. Falls back to uniform interior knots.
std::vector<PetscReal> TDSEZCore::TDSEZAdaptiveWFKnots(
    PetscReal Lmin, PetscReal Lmax,
    PetscInt  ninterior, PetscInt p,
    const std::vector<PetscReal>& rho_axis)
{
    (void)rho_axis;
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);
    for (PetscInt i = 0; i < nbnd; ++i) knots[i] = Lmin;
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i) knots[i] = Lmax;
    for (PetscInt i = 0; i < ninterior; ++i)
        knots[nbnd + i] = Lmin + (i + 1.0) / (ninterior + 1.0) * (Lmax - Lmin);
    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZAdaptiveWFKnots: implementation pending — using uniform interior.\n");
    return knots;
}










