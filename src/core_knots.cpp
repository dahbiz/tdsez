#include "tdsez_internal.hpp"







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
std::vector<PetscReal> TDSEZExtractBreakpoints(const std::string& expr)
{
    std::vector<PetscReal> bp;
    // Tokenise: split on whitespace AND on the structural characters
    // '?' ':' '(' ')' and comparison operators. muParser specs are typically
    // written without spaces (e.g. "(abs(x)<37.794)?0.084:(...)"), so we must
    // break on operators, not just whitespace.
    std::vector<std::string> toks;
    std::string cur;
    auto flush = [&]() { if (!cur.empty()) { toks.push_back(cur); cur.clear(); } };
    auto is_sep = [](char ch) -> bool {
        return ch == '?' || ch == ':' || ch == '(' || ch == ')' ||
               ch == '<' || ch == '>';
    };
    for (size_t i = 0; i < expr.size(); ++i) {
        char ch = expr[i];
        if (std::isspace((unsigned char)ch)) { flush(); continue; }
        // two-char operators <= and >=
        if ((ch == '<' || ch == '>') && i + 1 < expr.size() && expr[i+1] == '=') {
            flush(); toks.push_back(std::string(1, ch) + "="); ++i; continue;
        }
        if (is_sep(ch)) { flush(); toks.push_back(std::string(1, ch)); continue; }
        cur.push_back(ch);
    }
    flush();

    auto is_num = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        size_t i = 0; bool dot = false;
        if (s[0]=='+'||s[0]=='-') i=1;
        if (i >= s.size()) return false;
        for (; i < s.size(); ++i) {
            if (s[i]=='.') { if (dot) return false; dot=true; }
            else if (!std::isdigit((unsigned char)s[i])) return false;
        }
        return true;
    };
    auto to_num = [](const std::string& s) -> PetscReal {
        try { return std::stod(s); } catch (...) { return 0.0; }
    };
    // strip a trailing (x) so "abs(x)" / "f(x)" -> "abs" / "f"
    auto base = [](const std::string& s) -> std::string {
        if (s.size() >= 3 && s.substr(s.size()-3) == "(x)") return s.substr(0, s.size()-3);
        return s;
    };

    const std::vector<std::string> cmps = {"<=", ">=", "<", ">"};
    for (size_t i = 0; i+1 < toks.size(); ++i) {
        // is toks[i] a comparison operator?
        bool iscmp = false;
        for (auto& c : cmps) if (toks[i] == c) { iscmp = true; break; }
        if (!iscmp) continue;
        // lhs = tokens before the operator, joined and stripped of parens
        std::string lhsjoin;
        for (size_t k = 0; k < i; ++k) {
            if (toks[k] == "(" || toks[k] == ")") continue;
            if (!lhsjoin.empty()) lhsjoin += " ";
            lhsjoin += toks[k];
        }
        // rhs = token immediately after the operator
        if (i+1 >= toks.size()) continue;
        std::string rhs = toks[i+1];
        if (!is_num(rhs)) continue;
        PetscReal c = to_num(rhs);
        // take the head of the lhs (e.g. "abs" from "abs x", or "x")
        std::string lhs = lhsjoin;
        size_t sp = lhs.find(' ');
        if (sp != std::string::npos) lhs = lhs.substr(0, sp);
        lhs = base(lhs);
        bool neg = (lhs == "abs" || lhs == "fabs");
        if (neg) { bp.push_back(-c); bp.push_back(c); }
        else if (lhs == "x") { bp.push_back(c); }
        // other unary f(x): skip (cannot invert generically)
    }
    // de-duplicate (1e-9 tolerance)
    std::sort(bp.begin(), bp.end());
    std::vector<PetscReal> out;
    for (size_t i = 0; i < bp.size(); ++i) {
        if (!out.empty() && std::fabs(bp[i]-out.back()) < 1e-9) continue;
        out.push_back(bp[i]);
    }
    return out;
}


std::vector<PetscReal> TDSEZCore::TDSEZInterfaceKnots(
    PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscInt p,
    const std::string& expr, const std::string& massexpr)
{
    // Clamped (open) knot vector: p+1 repeated knots at each boundary, exactly
    // like the symmetric custom generators (symexp/tanu/tann/logtan) and like
    // IGAAxisInitUniform. Endpoint multiplicity p+1 is REQUIRED so the
    // boundary-span B-spline is interpolatory and IGASetBoundaryValue(...,0,0.0)
    // truly enforces homogeneous Dirichlet psi=0 at the wall. Using p here would
    // leave the boundary non-interpolatory and silently break the BC, raising the
    // ground-state energy (verified with a V=0 particle-in-a-box test).
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);

    for (PetscInt i = 0; i < nbnd; ++i) knots[i] = Lmin;
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i) knots[i] = Lmax;

    // 1) Extract material interfaces from BOTH the potential and the mass
    //    expression (a step can live in either), and union them.
    std::vector<PetscReal> iface = TDSEZExtractBreakpoints(expr);
    {
        std::vector<PetscReal> mi = TDSEZExtractBreakpoints(massexpr);
        for (PetscReal x : mi) iface.push_back(x);
    }
    std::vector<PetscReal> ins;
    for (PetscReal x : iface)
        if (x > Lmin + 1e-9 && x < Lmax - 1e-9) ins.push_back(x);
    std::sort(ins.begin(), ins.end());
    if (ins.empty()) {
        PetscPrintf(PETSC_COMM_WORLD,
            "TDSEZInterfaceKnots: no breakpoints found in expression; "
            "falling back to uniform interior knots.\n");
        for (PetscInt i = 0; i < ninterior; ++i)
            knots[nbnd + i] = Lmin + (i + 1.0) / (ninterior + 1.0) * (Lmax - Lmin);
        return knots;
    }

    // 2) Multiplicity m per interface (<= p), bounded by ninterior budget
    PetscInt m = 1;
    while (true) {
        PetscInt need = (PetscInt)ins.size() * (m + 1);
        if (need > ninterior || m >= p) break;
        ++m;
    }
    while (m * (PetscInt)ins.size() > ninterior && m > 1) --m;

    // 3) Pinned interface knots (sorted, de-duplicated, inside domain)
    std::vector<PetscReal> pp;
    for (PetscReal x : ins)
        for (PetscInt k = 0; k < m; ++k) pp.push_back(x);
    std::sort(pp.begin(), pp.end());
    std::vector<PetscReal> pinned;
    for (PetscReal x : pp) {
        if (x <= Lmin + 1e-9 || x >= Lmax - 1e-9) continue;
        if (!pinned.empty() && std::fabs(x - pinned.back()) < 1e-9) continue;
        pinned.push_back(x);
    }
    PetscInt total_pinned = (PetscInt)pinned.size();

    // 4) Weighted CDF grading for the free knots (cluster at interfaces)
    const PetscInt  Nquad = 100000;
    const PetscReal A     = 6.0;
    const PetscReal width = (Lmax - Lmin) / (PetscReal)(ninterior + 2) * 2.0;
    std::vector<PetscReal> xfine(Nquad+1), cdf(Nquad+1);
    auto sech2 = [](PetscReal x, PetscReal x0, PetscReal w) -> PetscReal {
        PetscReal u = (x - x0) / w;
        PetscReal c = 1.0 / PetscCoshReal(u);
        return c * c;
    };
    auto weight = [&](PetscReal x) -> PetscReal {
        PetscReal w = 1.0;
        for (PetscReal xi : ins) w += A * sech2(x, xi, width);
        return w;
    };
    PetscReal dx = (Lmax - Lmin) / Nquad;
    xfine[0] = Lmin; cdf[0] = 0.0;
    for (PetscInt i = 1; i <= Nquad; ++i) {
        xfine[i] = Lmin + i * dx;
        cdf[i]   = cdf[i-1] + weight(xfine[i]) * dx;
    }
    PetscReal tot = cdf[Nquad];
    for (PetscInt i = 0; i <= Nquad; ++i) cdf[i] /= tot;

    PetscInt nfree = ninterior - total_pinned;
    std::vector<PetscReal> freek;
    freek.reserve(nfree > 0 ? nfree : 0);
    for (PetscInt i = 0; i < nfree; ++i) {
        PetscReal ti = (i + 1.0) / (PetscReal)(nfree + 1);
        PetscInt lo = 0, hi = Nquad;
        while (hi - lo > 1) {
            PetscInt mid = (lo + hi) / 2;
            (cdf[mid] < ti ? lo : hi) = mid;
        }
        PetscReal frac = (ti - cdf[lo]) / (cdf[hi] - cdf[lo] + 1e-30);
        freek.push_back(xfine[lo] + frac * (xfine[hi] - xfine[lo]));
    }

    std::vector<PetscReal> allk = pinned;
    allk.insert(allk.end(), freek.begin(), freek.end());
    std::sort(allk.begin(), allk.end());
    // Build strictly-increasing interior knots inside (Lmin, Lmax).
    const PetscReal eps = 1e-9;
    std::vector<PetscReal> interior;
    interior.reserve(ninterior);
    for (PetscReal x : allk) {
        if (x <= Lmin + eps || x >= Lmax - eps) continue;       // keep strictly inside
        if (!interior.empty() && x <= interior.back() + eps) continue; // enforce increasing & unique
        interior.push_back(x);
    }
    // If we lost knots (dedup/ordering) or gained the wrong count, fill the
    // remainder with evenly spaced knots between the last placed and Lmax-eps,
    // guaranteeing exactly ninterior strictly-increasing interior knots.
    if ((PetscInt)interior.size() != ninterior) {
        interior.clear();
        PetscReal lo = Lmin + eps, hi = Lmax - eps;
        for (PetscInt i = 0; i < ninterior; ++i)
            interior.push_back(lo + (hi - lo) * (i + 1.0) / (ninterior + 1.0));
    }
    // Final safety: guarantee strictly increasing
    for (PetscInt i = 1; i < (PetscInt)interior.size(); ++i)
        if (interior[i] <= interior[i-1]) interior[i] = interior[i-1] + eps;
    for (PetscInt i = 0; i < ninterior; ++i) knots[nbnd + i] = interior[i];

    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZInterfaceKnots: %d interface(s) at", (int)ins.size());
    for (PetscReal xi : ins) PetscPrintf(PETSC_COMM_WORLD, " %.4f", (double)xi);
    PetscPrintf(PETSC_COMM_WORLD, " (mult=%d, pinned=%d/%d)\n",
                (int)m, (int)total_pinned, (int)ninterior);

    return knots;
}





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
// "adaptive" KnotSequence: places interior knots where the POTENTIAL is
// "interesting" — steep gradients (well edges, CAP onset) and deep regions
// (the Coulomb wells) — by inverting the cumulative distribution of a
// potential-derived importance weight. This is the single-pass Tier-1 cousin
// of the two-pass density-driven "adaptive_wf" mode: it needs no prior solve,
// so it slots straight into the existing per-axis KnotSequence dispatch.
//
// Importance weight (bounded, self-scaling so it is potential-independent):
//     w(x) = 1 + kappa * tanh(|dV/daxis| / dVref) + tanh(|V| / Vref)
//   - |dV/daxis| captures steep regions (well flanks, CAP activation) where the
//     basis needs resolution to represent the rapid slope.
//   - |V| captures the deep wells where the wavefunction localises.
//   - tanh compression bounds the Coulomb cusp so knots are NOT all piled onto
//     the nucleus (which would starve the bonding region). Vref/dVref are
//     auto-scaled to the per-axis maxima, so the same kappa works for any
//     potential. An optional power sharpening (AdaptivePower) concentrates
//     knots more aggressively.
//
// The potential and its analytic derivative are evaluated exactly via the
// already-parsed muParser objects TDSEZParser::VPot / dVPotX/Y/Z (no finite
// differences, no oversized grid). Clamped open knot vector: p+1 repeated
// knots at BOTH boundaries (required for interpolatory Dirichlet BCs).
std::vector<PetscReal> TDSEZCore::TDSEZAdaptiveKnots(
    PetscReal Lmin, PetscReal Lmax,
    PetscInt  ninterior, PetscInt p,
    PetscInt  axis,            // 0->x, 1->y, 2->z selects dVPotX/Y/Z
    PetscReal kappa,           // gradient-weighting strength (default 1.0)
    PetscReal power)           // sharpening exponent (default 1.0)
{
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);

    // Left/right boundary: p+1 repeated knots (clamped / interpolatory).
    for (PetscInt i = 0; i < nbnd; ++i) knots[i] = Lmin;
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i) knots[i] = Lmax;

    // ── 1) Sample the importance weight on a modest grid ──────────────
    // 4000 points is ample for an accurate CDF inversion; the prototype's
    // 100k grid was pure waste.
    const PetscInt  Ng     = 4000;
    const PetscReal dx     = (Lmax - Lmin) / (PetscReal)(Ng - 1);
    std::vector<PetscReal> xg(Ng), wg(Ng);

    PetscReal Vmax = 0.0, dVmax = 0.0;
    for (PetscInt i = 0; i < Ng; ++i) {
        PetscReal x = Lmin + i * dx;
        TDSEZParser::setVars(x);
        PetscReal V  = std::fabs(TDSEZParser::VPot.Eval());
        PetscReal dV = 0.0;
        try {
            if (axis == 0)      dV = std::fabs(TDSEZParser::dVPotX.Eval());
            else if (axis == 1) dV = std::fabs(TDSEZParser::dVPotY.Eval());
            else                dV = std::fabs(TDSEZParser::dVPotZ.Eval());
        } catch (const mu::ParserError &e) {
            throw std::runtime_error(
                "TDSEZAdaptiveKnots: failed to evaluate the potential derivative "
                "(dVPot" + std::string(axis == 0 ? "X" : axis == 1 ? "Y" : "Z") +
                "). 'adaptive'/'adaptive_wf' require PotentialDerivative" +
                std::string(axis == 0 ? "X" : axis == 1 ? "Y" : "Z") +
                " to be set (it is empty or invalid). Provide the analytic "
                "derivative of the potential, e.g. PotentialDerivative" +
                std::string(axis == 0 ? "X" : axis == 1 ? "Y" : "Z") + " = -2.0*x.");
        }
        xg[i] = x;  wg[i] = V;  if (V  > Vmax)  Vmax  = V;
        if (dV > dVmax) dVmax = dV;
    }
    // Reference scales (avoid div-by-zero for flat potentials).
    PetscReal Vref  = (Vmax  > 1e-300) ? Vmax  : 1.0;
    PetscReal dVref = (dVmax > 1e-300) ? dVmax : 1.0;

    PetscReal wsum = 0.0;
    for (PetscInt i = 0; i < Ng; ++i) {
        TDSEZParser::setVars(xg[i]);
        PetscReal V  = std::fabs(TDSEZParser::VPot.Eval());
        PetscReal dV = 0.0;
        try {
            if (axis == 0)      dV = std::fabs(TDSEZParser::dVPotX.Eval());
            else if (axis == 1) dV = std::fabs(TDSEZParser::dVPotY.Eval());
            else                dV = std::fabs(TDSEZParser::dVPotZ.Eval());
        } catch (const mu::ParserError &e) {
            throw std::runtime_error(
                "TDSEZAdaptiveKnots: failed to evaluate the potential derivative "
                "(dVPot" + std::string(axis == 0 ? "X" : axis == 1 ? "Y" : "Z") +
                "). 'adaptive'/'adaptive_wf' require PotentialDerivative" +
                std::string(axis == 0 ? "X" : axis == 1 ? "Y" : "Z") +
                " to be set (it is empty or invalid). Provide the analytic "
                "derivative of the potential, e.g. PotentialDerivative" +
                std::string(axis == 0 ? "X" : axis == 1 ? "Y" : "Z") + " = -2.0*x.");
        }
        PetscReal w = 1.0 + kappa * std::tanh(dV / dVref)
                          +        std::tanh(V  / Vref);
        if (power != 1.0) w = std::pow(w, power);
        wg[i] = w;  wsum += w * dx;
    }

    // ── 1b) Robustness guards on the importance weight ────────────────
    // (i) Nearly-flat weight (featureless potential): CDF inversion gives
    //     uniform knots anyway, but guard explicitly for safety.
    // (ii) Edge-peaked weight: for a *confining* potential |V| (and |∇V|)
    //     grow toward the box boundary, so the magnitude-based weight
    //     clusters knots at the edges — far from where the bound state
    //     actually lives. That yields a wrong (too-high) eigenvalue. Such
    //     smooth-confining potentials are better served by uniform knots
    //     (or adaptive_wf); fall back to uniform in that case.
    {
        PetscReal wmean = wsum / (Lmax - Lmin + 1e-30);
        PetscReal wvar  = 0.0;
        for (PetscInt i = 0; i < Ng; ++i) {
            PetscReal d = wg[i] - wmean;
            wvar += d * d * dx;
        }
        wvar = std::sqrt(wvar / (Lmax - Lmin + 1e-30));
        const PetscReal wc = wg[Ng / 2];
        const PetscReal we = 0.5 * (wg[0] + wg[Ng - 1]);
        const PetscBool flat      = (wmean > 1e-300) && (wvar / wmean) < 1e-3;
        const PetscBool edgepeak  = (wc > 1e-300) && (we / wc) > 2.0;
        if (flat || edgepeak) {
            PetscPrintf(PETSC_COMM_WORLD,
                "TDSEZAdaptiveKnots[%d]: weight %s (rel var %.2e, edge/centre %.2e) "
                "-> falling back to uniform interior knots.\n", axis,
                flat ? "nearly flat" : "edge-peaked (confining potential)",
                (double)(wmean > 1e-300 ? wvar / wmean : 0.0),
                (double)(wc > 1e-300 ? we / wc : 1e30));
            for (PetscInt i = 0; i < ninterior; ++i)
                knots[nbnd + i] = Lmin + (i + 1.0) / (ninterior + 1.0) * (Lmax - Lmin);
            return knots;
        }
    }

    // ── 2) Cumulative distribution (normalised) ───────────────────────
    std::vector<PetscReal> cdf(Ng);
    cdf[0] = 0.0;
    for (PetscInt i = 1; i < Ng; ++i)
        cdf[i] = cdf[i - 1] + 0.5 * (wg[i] + wg[i - 1]) * dx;
    PetscReal tot = cdf[Ng - 1];
    if (tot <= 0.0) tot = 1.0;
    for (PetscInt i = 0; i < Ng; ++i) cdf[i] /= tot;

    // ── 3) Invert the CDF to place ninterior knots ────────────────────
    // Knot j sits at the x where CDF(x) = (j+1)/(ninterior+1).
    std::vector<PetscReal> interior;
    interior.reserve(ninterior);
    for (PetscInt j = 0; j < ninterior; ++j) {
        PetscReal tj = (j + 1.0) / (PetscReal)(ninterior + 1);
        // binary search for the bracketing CDF interval
        PetscInt lo = 0, hi = Ng - 1;
        while (hi - lo > 1) {
            PetscInt mid = (lo + hi) / 2;
            if (cdf[mid] < tj) lo = mid; else hi = mid;
        }
        PetscReal frac = (tj - cdf[lo]) / (cdf[hi] - cdf[lo] + 1e-30);
        PetscReal xk = xg[lo] + frac * (xg[hi] - xg[lo]);
        // keep strictly inside the domain (never collide with boundary repeats)
        const PetscReal eps = 1e-9 * (Lmax - Lmin);
        if (xk <= Lmin + eps) xk = Lmin + eps;
        if (xk >= Lmax - eps) xk = Lmax - eps;
        interior.push_back(xk);
    }

    // ── 4) Enforce strictly increasing & de-duplicate ─────────────────
    std::sort(interior.begin(), interior.end());
    std::vector<PetscReal> clean_int;
    clean_int.reserve(ninterior);
    const PetscReal eps2 = 1e-9 * (Lmax - Lmin);
    for (PetscReal x : interior) {
        if (!clean_int.empty() && x <= clean_int.back() + eps2) continue;
        clean_int.push_back(x);
    }
    // If de-dup lost knots (coincident CDF targets), backfill with even spacing.
    while ((PetscInt)clean_int.size() < ninterior) {
        PetscReal lo = (clean_int.empty() ? Lmin : clean_int.back());
        PetscReal hi = Lmax;
        PetscReal x = lo + (hi - lo) / (PetscReal)(ninterior - clean_int.size() + 1);
        if (x <= Lmin + eps2 || x >= Lmax - eps2) break;
        if (!clean_int.empty() && x <= clean_int.back() + eps2) break;
        clean_int.push_back(x);
        std::sort(clean_int.begin(), clean_int.end());
    }
    // Final safety: guarantee strictly increasing.
    for (PetscInt i = 1; i < (PetscInt)clean_int.size(); ++i)
        if (clean_int[i] <= clean_int[i - 1]) clean_int[i] = clean_int[i - 1] + eps2;
    for (PetscInt i = 0; i < ninterior && i < (PetscInt)clean_int.size(); ++i)
        knots[nbnd + i] = clean_int[i];

    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZAdaptiveKnots[%d]: %d interior | p=%d | kappa=%.2f power=%.2f | "
        "wmax/wmin=%.2f\n",
        (int)axis, (int)ninterior, (int)p, (double)kappa, (double)power,
        (double)(*std::max_element(wg.begin(), wg.end()) /
                 (*std::min_element(wg.begin(), wg.end()) + 1e-30)));

    return knots;
}


// ── Adaptive (density-driven, two-pass) knot vector ──────────────────────────
// "adaptive_wf" KnotSequence: places interior knots where the ELECTRON DENSITY
// |psi0|^2 is large, using a 1D marginal supplied by BootstrapDensity(). This is
// the two-pass r-adaptivity bootstrap: a cheap coarse ground-state solve gives
// the density shape, which seeds a fine knot vector that resolves where the
// wavefunction actually lives (Coulomb cusp, nodal structure, bonding region) —
// strictly better per-DOF than the potential-driven "adaptive" mode, because the
// density already folds in everything the physics cares about.
//
// The marginal rho_axis is given on a uniform grid over [Lmin, Lmax] (see
// SampleDensity). We linearly interpolate it, build the cumulative distribution,
// and invert it to place ninterior knots. The density is already smooth and
// positive, so no tanh compression is needed; an optional power sharpening
// (AdaptivePower, shared with the potential-driven mode) concentrates knots more
// tightly on the densest regions. Clamped open knot vector (p+1 at both ends).
std::vector<PetscReal> TDSEZCore::TDSEZAdaptiveWFKnots(
    PetscReal Lmin, PetscReal Lmax,
    PetscInt  ninterior, PetscInt p,
    const std::vector<PetscReal>& rho_axis)
{
    const PetscInt nbnd   = p + 1;
    const PetscInt nknots = ninterior + 2 * nbnd;
    std::vector<PetscReal> knots(nknots);

    for (PetscInt i = 0; i < nbnd; ++i) knots[i] = Lmin;
    for (PetscInt i = nbnd + ninterior; i < nknots; ++i) knots[i] = Lmax;

    if (rho_axis.size() < 2) {
        // Degenerate marginal (e.g. 0D / empty) -> fall back to even spacing.
        for (PetscInt i = 0; i < ninterior; ++i)
            knots[nbnd + i] = Lmin + (i + 1.0) / (ninterior + 1.0) * (Lmax - Lmin);
        PetscPrintf(PETSC_COMM_WORLD,
            "TDSEZAdaptiveWFKnots: empty marginal, using uniform interior.\n");
        return knots;
    }

    // rho_axis is uniformly spaced over [Lmin, Lmax]; build helper to interp.
    const PetscInt  Nr = (PetscInt)rho_axis.size();
    const PetscReal dr = (Lmax - Lmin) / (PetscReal)(Nr - 1);
    auto rho_at = [&](PetscReal x) -> PetscReal {
        if (x <= Lmin) return rho_axis.front();
        if (x >= Lmax) return rho_axis.back();
        PetscReal t = (x - Lmin) / dr;
        PetscInt  i = (PetscInt)t;
        PetscReal f = t - i;
        if (i >= Nr - 1) return rho_axis.back();
        return rho_axis[i] * (1.0 - f) + rho_axis[i + 1] * f;
    };

    const PetscReal power = TDSEZParser::AdaptivePower;

    // ── 1) Sample the (sharpened) density on a fine grid ──────────────
    const PetscInt  Ng = 4000;
    const PetscReal dx = (Lmax - Lmin) / (PetscReal)(Ng - 1);
    std::vector<PetscReal> xg(Ng), wg(Ng);
    PetscReal wmax = 0.0;
    for (PetscInt i = 0; i < Ng; ++i) {
        PetscReal x = Lmin + i * dx;
        PetscReal w = rho_at(x);
        if (power != 1.0) w = std::pow(w, power);
        xg[i] = x; wg[i] = w;
        if (w > wmax) wmax = w;
    }

    // ── 2) Cumulative distribution (trapezoidal, normalised) ──────────
    std::vector<PetscReal> cdf(Ng);
    cdf[0] = 0.0;
    for (PetscInt i = 1; i < Ng; ++i)
        cdf[i] = cdf[i - 1] + 0.5 * (wg[i] + wg[i - 1]) * dx;
    PetscReal tot = cdf[Ng - 1];
    if (tot <= 0.0) tot = 1.0;
    for (PetscInt i = 0; i < Ng; ++i) cdf[i] /= tot;

    // ── 3) Invert the CDF to place ninterior knots ────────────────────
    std::vector<PetscReal> interior;
    interior.reserve(ninterior);
    for (PetscInt j = 0; j < ninterior; ++j) {
        PetscReal tj = (j + 1.0) / (PetscReal)(ninterior + 1);
        PetscInt lo = 0, hi = Ng - 1;
        while (hi - lo > 1) {
            PetscInt mid = (lo + hi) / 2;
            if (cdf[mid] < tj) lo = mid; else hi = mid;
        }
        PetscReal frac = (tj - cdf[lo]) / (cdf[hi] - cdf[lo] + 1e-30);
        PetscReal xk = xg[lo] + frac * (xg[hi] - xg[lo]);
        const PetscReal eps = 1e-9 * (Lmax - Lmin);
        if (xk <= Lmin + eps) xk = Lmin + eps;
        if (xk >= Lmax - eps) xk = Lmax - eps;
        interior.push_back(xk);
    }

    // ── 4) Enforce strictly increasing & de-duplicate ─────────────────
    std::sort(interior.begin(), interior.end());
    std::vector<PetscReal> clean_int;
    clean_int.reserve(ninterior);
    const PetscReal eps2 = 1e-9 * (Lmax - Lmin);
    for (PetscReal x : interior) {
        if (!clean_int.empty() && x <= clean_int.back() + eps2) continue;
        clean_int.push_back(x);
    }
    while ((PetscInt)clean_int.size() < ninterior) {
        PetscReal lo = (clean_int.empty() ? Lmin : clean_int.back());
        PetscReal x = lo + (Lmax - lo) / (PetscReal)(ninterior - clean_int.size() + 1);
        if (x <= Lmin + eps2 || x >= Lmax - eps2) break;
        if (!clean_int.empty() && x <= clean_int.back() + eps2) break;
        clean_int.push_back(x);
        std::sort(clean_int.begin(), clean_int.end());
    }
    for (PetscInt i = 1; i < (PetscInt)clean_int.size(); ++i)
        if (clean_int[i] <= clean_int[i - 1]) clean_int[i] = clean_int[i - 1] + eps2;
    for (PetscInt i = 0; i < ninterior && i < (PetscInt)clean_int.size(); ++i)
        knots[nbnd + i] = clean_int[i];

    PetscPrintf(PETSC_COMM_WORLD,
        "TDSEZAdaptiveWFKnots: %d interior | p=%d | wmax=%.3e (kin-density-driven)\n",
        (int)ninterior, (int)p, (double)wmax);

    return knots;
}










