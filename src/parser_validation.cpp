#include "tdsez_parser.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

/**
 * @file parser_validation.cpp
 * @brief Physical and cross-option validation of parsed input.
 */

namespace {

// The Galerkin volume quadrature never evaluates box endpoints.  Validate
// coefficient fields in that same open domain so a physically valid endpoint
// singularity (e.g. Quesne's m(x)~1/(x+a) at a Dirichlet wall) is not rejected
// before assembly.  Potential validation remains closed-domain because it is
// expected to be finite at the wall as well.
template <typename F>
void sampleOpenDomain(PetscInt dim, const std::array<PetscReal, 3>& lo,
                      const std::array<PetscReal, 3>& hi, F&& evaluate)
{
    constexpr PetscInt samplesPerAxis = 16;
    std::array<PetscReal, 3> center = {0.0, 0.0, 0.0};
    for (PetscInt d = 0; d < dim; ++d) center[d] = (lo[d] + hi[d]) / 2.0;
    for (PetscInt d = 0; d < dim; ++d) {
        for (PetscInt i = 0; i < samplesPerAxis; ++i) {
            auto point = center;
            point[d] = lo[d] + (hi[d] - lo[d]) * (i + 0.5) / samplesPerAxis;
            evaluate(point[0], point[1], point[2]);
        }
    }
}

std::string sampleLocation(PetscReal x, PetscReal y, PetscReal z)
{
    return " at (" + std::to_string(x) + ", " + std::to_string(y) +
           ", " + std::to_string(z) + ")";
}

// Strip one matched outer parenthesis pair: "(0.7+0.7i)" -> "0.7+0.7i".
// Unmatched pairs (e.g. "(1)+(2)") are left untouched.
std::string stripOuterParens(const std::string &x)
{
    if (x.size() < 2 || x.front() != '(' || x.back() != ')')
        return x;
    int depth = 0;
    for (size_t k = 0; k + 1 < x.size(); ++k) {
        if (x[k] == '(') ++depth;
        else if (x[k] == ')') {
            --depth;
            if (depth == 0 && k + 2 != x.size())
                return x;   // closes before the last char: not an outer pair
        }
    }
    return x.substr(1, x.size() - 2);
}

// One superposition coefficient: "a", "a+ib", "a-ib", "a+i", "a-i", "b*i",
// "i", "-i", "(...)" (a,b real decimal parts; "i" may also be written "j").
// In complex builds this covers any complex value with real decimal parts;
// in real builds the imaginary part must be zero.
PetscScalar parseSuperpositionCoefficient(const std::string &raw)
{
    auto trimmed = [](std::string x) {
        size_t a = x.find_first_not_of(" \t");
        size_t b = x.find_last_not_of(" \t");
        return (a == std::string::npos) ? std::string() : x.substr(a, b - a + 1);
    };
    std::string s = trimmed(stripOuterParens(trimmed(raw)));
    if (s.empty())
        throw std::runtime_error("superposition coefficient is empty");

    // Split into +/- parts at operators that are neither leading signs nor
    // exponent separators (1e-3 keeps its '-'). An 'i'/'j' directly before
    // an operator terminates a number, so "2i-1" splits as (2i) - (1).
    std::vector<std::pair<bool /*isPlus*/, std::string>> parts;
    auto pushPart = [&parts, &trimmed](bool isPlus, std::string p) {
        // The sign lives in isPlus; strip it from the token itself so a
        // negative part is not applied twice.
        if (!p.empty() && (p.front() == '+' || p.front() == '-'))
            p.erase(0, 1);
        parts.push_back({isPlus, trimmed(p)});
    };
    {
        std::string cur;
        for (size_t k = 0; k < s.size(); ++k) {
            const char ch = s[k];
            bool isOperator =
                (ch == '+' || ch == '-') &&
                !cur.empty() &&
                !(cur.size() == 1 && (cur == "+" || cur == "-")) &&
                !(s[k - 1] == 'e' || s[k - 1] == 'E');
            if (isOperator) {
                pushPart(cur[0] != '-', cur);
                cur.clear();
            }
            cur += ch;
        }
        // Tail part: empty means the string ended on an operator (caught by
        // the empty-token check below); otherwise cur[0] carries this part's
        // leading sign.
        pushPart(cur.empty() ? true : (cur[0] != '-'), cur);
    }

    PetscReal re = 0.0, im = 0.0;
    for (const auto &part : parts) {
        const bool isPlus = part.first;
        std::string p = part.second;
        // pushPart moved any sign into isPlus; a residual sign means a
        // malformed token such as "1++2" or "1+*".
        if (p.empty() || p == "+" || p == "-" ||
            p.front() == '+' || p.front() == '-')
            throw std::runtime_error("cannot parse superposition coefficient '" + raw + "'");
        bool imaginary = false;
        if (p.back() == 'i' || p.back() == 'j') {
            imaginary = true;
            p.pop_back();
        }
        double v;
        if (p.empty())                v = 1.0;               // "i", "j"
        else {
            size_t consumed = 0;
            try { v = std::stod(p, &consumed); }
            catch (...) {
                throw std::runtime_error(
                    "cannot parse superposition coefficient '" + raw + "'");
            }
            if (consumed != p.size())
                throw std::runtime_error(
                    "cannot parse superposition coefficient '" + raw + "'");
        }
        if (imaginary) im += isPlus ? v : -v;
        else           re += isPlus ? v : -v;
    }
#if !defined(PETSC_USE_COMPLEX)
    if (im != 0.0)
        throw std::runtime_error("nonzero imaginary coefficient '" + raw +
                                 "' requires a complex PETSc build");
#endif
    if (!std::isfinite(re) || !std::isfinite(im))
        throw std::runtime_error("superposition coefficient '" + raw + "' is not finite");
#if defined(PETSC_USE_COMPLEX)
    return PetscScalar(re, im);
#else
    return re;
#endif
}

// Split the sup: body into "coeff*idx" terms at top-level '+'/'-' operators.
// Depth counting leaves parenthesized coefficients ("(0.7+0.7i)*0") intact;
// a '-' separator carries its sign into the following term. An operator is a
// separator only when the preceding non-space character ends a value
// (digit, '.', 'i', 'j' or ')'), so leading signs and exponents do not split.
std::vector<std::string> splitSuperpositionTerms(const std::string &body)
{
    auto endsValue = [](char c) {
        return (c >= '0' && c <= '9') || c == '.' || c == 'i' || c == 'j' || c == ')';
    };
    auto trimmed = [](const std::string &x) {
        size_t a = x.find_first_not_of(" \t");
        size_t b = x.find_last_not_of(" \t");
        return (a == std::string::npos) ? std::string() : x.substr(a, b - a + 1);
    };
    std::vector<std::string> terms;
    std::string cur;
    int depth = 0;
    bool pendingMinus = false;
    for (size_t k = 0; k < body.size(); ++k) {
        const char ch = body[k];
        if (ch == '(') ++depth;
        else if (ch == ')') --depth;
        if (depth == 0 && (ch == '+' || ch == '-')) {
            size_t prev = k;
            while (prev > 0 && (body[prev - 1] == ' ' || body[prev - 1] == '\t'))
                --prev;
            const bool afterValue = prev > 0 && endsValue(body[prev - 1]);
            const bool exponent   = prev > 0 &&
                                    (body[prev - 1] == 'e' || body[prev - 1] == 'E');
            if (afterValue && !exponent) {
                if (!cur.empty()) { terms.push_back(trimmed(cur)); cur.clear(); }
                pendingMinus = (ch == '-');   // sign belongs to the next term
                continue;
            }
        }
        if (pendingMinus) {
            if (ch == ' ' || ch == '\t') continue;   // skip space after '-'
            cur += '-';
            pendingMinus = false;
        }
        cur += ch;
    }
    if (!cur.empty()) terms.push_back(trimmed(cur));
    return terms;
}

} // namespace

/// @brief Validate the parsed parameter set. Throws std::runtime_error on
///        any physically inconsistent or dangerous parameter combination.
///        Called once after parsing is complete.
void TDSEZParser::ValidateOrThrow()
{
    using std::to_string;
    // --- dimension ---
    if (Dimension < 1 || Dimension > 3)
        throw std::runtime_error("ValidateOrThrow: Dimension must be 1, 2 or 3 (got "
                                 + to_string(Dimension) + ")");

    // --- dipole-acceleration method ---
    if (AccelerationMethod != "auto" && AccelerationMethod != "operator" &&
        AccelerationMethod != "discrete")
        throw std::runtime_error("ValidateOrThrow: AccelerationMethod must be 'auto', "
                                 "'operator' or 'discrete' (got '" + AccelerationMethod + "')");

    // --- time propagator ---
    if (Propagator != "cn" && Propagator != "pade22" && Propagator != "magnus4")
        throw std::runtime_error("ValidateOrThrow: Propagator must be 'cn', 'pade22' "
                                 "or 'magnus4' (got '" + Propagator + "')");
    if (Preconditioner != "asm" && Preconditioner != "mg" && Preconditioner != "gamg")
        throw std::runtime_error("ValidateOrThrow: Preconditioner must be 'asm', 'mg' or 'gamg' (got '" + Preconditioner + "')");
    if (!(TransverseElementFactor >= 1.0))
        throw std::runtime_error("ValidateOrThrow: TransverseElementFactor must be >= 1 (got "
                                 + std::to_string(TransverseElementFactor) + ")");
    for (std::string *v : {&Basis, &QuadratureRule})
        std::transform(v->begin(), v->end(), v->begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (Basis != "bspline" && Basis != "bernstein" && Basis != "lagrange" && Basis != "spectral")
        throw std::runtime_error("ValidateOrThrow: Basis must be 'bspline', 'bernstein', 'lagrange' or 'spectral' (got '" + Basis + "')");
    if (QuadratureRule != "legendre" && QuadratureRule != "lobatto" && QuadratureRule != "reduced")
        throw std::runtime_error("ValidateOrThrow: QuadratureRule must be 'legendre', 'lobatto' or 'reduced' (got '" + QuadratureRule + "')");
    if (Basis != "bspline" && Preconditioner == "mg")
        throw std::runtime_error("ValidateOrThrow: Preconditioner = mg needs Basis = bspline (its prolongation is B-spline knot insertion)");
    if (MGLevels < 0 || MGLevels == 1 || MGSmooth < 1)
        throw std::runtime_error("ValidateOrThrow: MGLevels must be 0 (automatic) or >= 2 and MGSmooth >= 1");

    // --- spline degree ---
    if (SplineDegree < 1 || SplineDegree > 14)
        throw std::runtime_error("ValidateOrThrow: SplineDegree must be in [1,14] (got "
                                 + to_string(SplineDegree) + ")");

    // --- domain sanity (per-axis) ---
    auto checkAxis = [](PetscReal lo, PetscReal hi, const char *which) {
        if (!std::isfinite(lo) || !std::isfinite(hi))
            throw std::runtime_error(std::string("ValidateOrThrow: ") + which
                                     + " domain bound is not finite");
        if (hi <= lo)
            throw std::runtime_error(std::string("ValidateOrThrow: ") + which
                                     + " requires LMax > LMin (got ["
                                     + to_string(lo) + "," + to_string(hi) + "])");
    };
    checkAxis(LMinX, LMaxX, "X");
    checkAxis(LMinY, LMaxY, "Y");
    checkAxis(LMinZ, LMaxZ, "Z");

    // --- elements / dof count ---
    if (Nelements < 1)
        throw std::runtime_error("ValidateOrThrow: Nelements must be >= 1 (got "
                                 + to_string(Nelements) + ")");
    PetscInt nfuncs = Nelements + SplineDegree; // PetIGA open-knot convention
    if (nfuncs < 2)
        throw std::runtime_error("ValidateOrThrow: too few basis functions (Nelements + SplineDegree < 2)");

    // --- time ---
    if (EnablePropagation) {
        if (TimeStep <= 0.0 || FinalTime <= 0.0)
            throw std::runtime_error("ValidateOrThrow: TimeStep and FinalTime must be > 0 when propagating");
        if (!std::isfinite(TimeStep) || !std::isfinite(FinalTime))
            throw std::runtime_error("ValidateOrThrow: non-finite TimeStep/FinalTime");
    }

    // --- constant mass: detected from the expression ---
    // A Mass expression without x, y, z is constant; the fast constant-mass
    // assembly (no finite-difference mass derivatives, which vanish exactly
    // then) applies without MassIsConstant = 1 in the input.
    if (!MassIsConstant) {
        try { if (MassExpr.GetUsedVar().empty()) MassIsConstant = PETSC_TRUE; }
        catch (const mu::Parser::exception_type &) { /* reported when evaluated */ }
    }

    // --- requested states ---
    // (0 is allowed with an analytic initial state; checked below)
    if (NBoundStates < 0)
        throw std::runtime_error("ValidateOrThrow: NBoundStates must be >= 0");

    // --- initial state selection ---
    // Parse InitialState into a mode + (index,coeff) terms, and guard indices
    // against NBoundStates. Supported forms:
    //   "ground"              -> mode 0 (== state:0)
    //   "state:N"             -> mode 1, single bound state N
    //   "sup: a*N + b*M [+c*P]"-> mode 2, coherent superposition of <=3 states
    //   "analytic:quesne01"    -> mode 3, projected exact Quesne n=0,1 state
    //   "analytic:massstep01"  -> mode 4, projected exact sharp-interface state
    //   "wavepacket"           -> mode 5, projected Gaussian wave packet
    {
        const std::string &s = InitialState;
        InitialStateTerms.clear();
        InitialStateMode = 0;
        InitialStateIndex = 0;

        if (s == "ground" || s.empty()) {
            InitialStateMode = 0;
            InitialStateIndex = 0;
        } else if (s.rfind("state:", 0) == 0) {
            // "state:N"
            std::string num = s.substr(6);
            trim(num);
            if (num.empty())
                throw std::runtime_error(
                    "ValidateOrThrow: InitialState='" + s + "' missing state index (use 'state:N')");
            PetscInt idx;
            try {
                size_t consumed = 0;
                const auto parsed = std::stoll(num, &consumed);
                if (consumed != num.size() ||
                    parsed < std::numeric_limits<PetscInt>::min() ||
                    parsed > std::numeric_limits<PetscInt>::max())
                    throw std::invalid_argument("invalid state index");
                idx = (PetscInt)parsed;
            }
            catch (...) {
                throw std::runtime_error(
                    "ValidateOrThrow: InitialState='" + s + "' has a non-integer index");
            }
            if (idx < 0)
                throw std::runtime_error(
                    "ValidateOrThrow: InitialState index must be >= 0 (got " + std::to_string(idx) + ")");
            if (idx >= NBoundStates)
                throw std::runtime_error(
                    "ValidateOrThrow: InitialState='state:" + std::to_string(idx) +
                    "' but only NBoundStates=" + std::to_string(NBoundStates) +
                    " bound states are computed. Increase NBoundStates to at least " +
                    std::to_string(idx + 1) + ".");
            InitialStateMode = 1;
            InitialStateIndex = idx;
        } else if (s == "analytic:quesne01") {
            if (Dimension != 1)
                throw std::runtime_error(
                    "ValidateOrThrow: InitialState='analytic:quesne01' is a 1D benchmark");
            InitialStateMode = 3;
        } else if (s == "wavepacket") {
            for (PetscInt d = 0; d < Dimension; ++d)
                if (!(WavepacketWidth[d] > 0.0))
                    throw std::runtime_error("ValidateOrThrow: WavepacketWidth must be > 0 on every axis");
            InitialStateMode = 5;
        } else if (s == "analytic:massstep01") {
            if (Dimension != 1)
                throw std::runtime_error(
                    "ValidateOrThrow: InitialState='analytic:massstep01' is a 1D benchmark");
            InitialStateMode = 4;
        } else if (s.rfind("sup:", 0) == 0) {
            // "sup: c1*N + c2*M [+ c3*P]"  (coeffs real or complex, <=3
            // terms, top-level '+'/'-' separated). Complex coefficients carry
            // internal signs, so multi-term expressions must parenthesize
            // them: "sup: (0.7+0.7i)*0 + 1.0*1"; pure-imaginary shorthand
            // "2i" needs no parens.
            InitialStateMode = 2;
            std::string body = s.substr(4);
            std::vector<std::string> terms = splitSuperpositionTerms(body);
            if (terms.empty() || terms.size() > 3)
                throw std::runtime_error(
                    "ValidateOrThrow: InitialState='" + s + "' must have 1..3 superposition terms");
            for (auto &t : terms) {
                // term is "coeff*idx" (coeff optional -> 1.0)
                size_t star = t.find('*');
                std::string cstr = (star == std::string::npos) ? "1.0" : t.substr(0, star);
                std::string istr = (star == std::string::npos) ? t : t.substr(star + 1);
                // trim whitespace
                auto trim = [](std::string &x){ size_t a=x.find_first_not_of(" \t"); size_t b=x.find_last_not_of(" \t");
                    if(a==std::string::npos){x="";}else{x=x.substr(a,b-a+1);} };
                trim(cstr); trim(istr);
                PetscScalar coeff;
                try {
                    coeff = parseSuperpositionCoefficient(cstr);
                }
                catch (std::runtime_error &e) {
                    throw std::runtime_error(
                        "ValidateOrThrow: InitialState term '" + t + "': " + e.what());
                }
                PetscInt  idx;
                try {
                    size_t consumed = 0;
                    const auto parsed = std::stoll(istr, &consumed);
                    if (consumed != istr.size() ||
                        parsed < std::numeric_limits<PetscInt>::min() ||
                        parsed > std::numeric_limits<PetscInt>::max())
                        throw std::invalid_argument("invalid state index");
                    idx = (PetscInt)parsed;
                }
                catch (...) {
                    throw std::runtime_error(
                        "ValidateOrThrow: InitialState term '" + t + "' has a non-integer state index");
                }
                if (idx < 0)
                    throw std::runtime_error(
                        "ValidateOrThrow: InitialState index must be >= 0 (got " + std::to_string(idx) + ")");
                if (idx >= NBoundStates)
                    throw std::runtime_error(
                        "ValidateOrThrow: InitialState superposition includes state:" + std::to_string(idx) +
                        " but only NBoundStates=" + std::to_string(NBoundStates) +
                        " bound states are computed. Increase NBoundStates to at least " +
                        std::to_string(idx + 1) + ".");
                InitialStateTerms.push_back({idx, coeff});
            }
        } else {
            throw std::runtime_error(
                "ValidateOrThrow: InitialState must be 'ground', 'state:N', \""
                "'sup: c*N + c*M [+ c*P]' (complex coeffs need parens: '(0.7+0.7i)*0'), "
                "'analytic:quesne01', 'analytic:massstep01' or 'wavepacket' (got '" + s + "')");
        }
    }
    if (NBoundStates == 0 && InitialStateMode < 3)
        throw std::runtime_error("ValidateOrThrow: NBoundStates = 0 (no eigenproblem) needs an analytic "
                                 "InitialState (wavepacket, analytic:quesne01, analytic:massstep01)");

    // --- bound-state save format ---
    if (BoundStateFormat != "complex" && BoundStateFormat != "real")
        throw std::runtime_error(
            "ValidateOrThrow: BoundStateFormat must be 'complex' or 'real' (got '" + BoundStateFormat + "')");

    const std::array<PetscReal, 3> boundsLow = {LMinX, LMinY, LMinZ};
    const std::array<PetscReal, 3> boundsHigh = {LMaxX, LMaxY, LMaxZ};

    // A non-positive mass makes the kinetic operator indefinite.  Coefficients
    // are sampled in the open domain: volume quadrature excludes endpoints.
    sampleOpenDomain((PetscInt)Dimension, boundsLow, boundsHigh,
        [&](PetscReal x, PetscReal y, PetscReal z) {
            PetscReal mass;
            try { mass = MassDist(x, y, z); }
            catch (const mu::ParserError& e) {
                throw std::runtime_error("ValidateOrThrow: Mass expression evaluation failed"
                                         + sampleLocation(x, y, z) + ": " + e.GetMsg());
            }
            catch (const std::exception& e) {
                throw std::runtime_error("ValidateOrThrow: Mass expression evaluation failed"
                                         + sampleLocation(x, y, z) + ": " + e.what());
            }
            if (!std::isfinite(mass) || mass <= 0.0)
                throw std::runtime_error("ValidateOrThrow: Mass must be > 0 and finite in the open domain"
                                         + sampleLocation(x, y, z) + " (got "
                                         + to_string(mass) + ")");
        });

    // A non-finite potential in the open domain poisons volume assembly.
    // Endpoint singular walls are admissible with a Dirichlet boundary and are
    // not sampled by PetIGA's Gauss quadrature.
    sampleOpenDomain((PetscInt)Dimension, boundsLow, boundsHigh,
        [&](PetscReal x, PetscReal y, PetscReal z) {
            PetscReal potential;
            try { potential = V(x, y, z); }
            catch (const mu::ParserError& e) {
                throw std::runtime_error("ValidateOrThrow: Potential expression evaluation failed"
                                         + sampleLocation(x, y, z) + ": " + e.GetMsg());
            }
            catch (const std::exception& e) {
                throw std::runtime_error("ValidateOrThrow: Potential expression evaluation failed"
                                         + sampleLocation(x, y, z) + ": " + e.what());
            }
            if (!std::isfinite(potential))
                throw std::runtime_error("ValidateOrThrow: Potential is not finite in the open domain"
                                         + sampleLocation(x, y, z));
        });

    // --- enum normalisation + cross-constraints (footguns caught late before) ---
    auto lower = [](std::string s){
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };

    // Polarization: normalise case + reject invalid axis sets up front
    // (was a cryptic "not compatible with dimension" crash deep in assembly).
    {
        static const std::set<std::string> valid = {
            "", "none", "x", "y", "z", "xy", "xz", "yz", "xyz"};
        std::string p = lower(Polarization);
        if (p == "all") p = "xyz";
        Polarization = p;
        if (valid.find(p) == valid.end())
            throw std::runtime_error(
                "ValidateOrThrow: Polarization '" + Polarization + "' is invalid. "
                "Use one of: none, x, y, z, xy, xz, yz, xyz (case-insensitive).");
        for (int d = (int)Dimension; d < 3; ++d)
            if (Polarization.find("xyz"[d]) != std::string::npos)
                throw std::runtime_error("ValidateOrThrow: Polarization '" + Polarization + "' couples axis " +
                    std::string(1, "xyz"[d]) + ", which does not exist in a " +
                    std::to_string((int)Dimension) + "D problem.");
    }

    // BoundaryType: normalise aliases + reject unknown up front
    // (was a silent default-to-Neumann; now explicit + validated).
    {
        std::string b = lower(BoundaryType);
        if      (b == "natural") b = "neumann";
        else if (b == "wall")    b = "dirichlet";
        if (b != "neumann" && b != "dirichlet")
            throw std::runtime_error(
                "ValidateOrThrow: BoundaryType '" + BoundaryType + "' is unknown. "
                "Use 'Neumann' (reflecting wall) or 'Dirichlet' (psi=0 at wall) "
                "(case-insensitive; 'natural' and 'wall' are aliases).");
        BoundaryType = b;
    }


    // CAPWidth > 0 sets the layer; the Manolopoulos kmin follows from it.
    if (EnableCAP && CAPWidth < 0.0)
        throw std::runtime_error("ValidateOrThrow: CAPWidth must be > 0 (or 0 to use CAPKmin)");
    if (EnableCAP && CAPWidth > 0.0) CAPKmin = 2.62206 / (2.0 * 0.2 * CAPWidth);

    // CAP requires kmin > 0: kmin<=0 makes the Manolopoulos absorb width
    // diverge (absorbs the ENTIRE domain) -> silent TS divergence.
    if (EnableCAP && CAPKmin <= 0.0)
        throw std::runtime_error(
            "ValidateOrThrow: EnableCAP=1 but CAPKmin=" + to_string(CAPKmin) +
            " <= 0. A zero/negative kmin makes the absorbing width diverge and the "
            "time stepper diverges. Set CAPKmin > 0 (e.g. 0.1) or disable CAP.");

    // The absorbing layer at each wall is w = c/(2 delta CAPKmin) = 6.555/CAPKmin
    // wide. It must leave a free core: at most 45% of each axis per wall.
    if (EnableCAP) {
        const PetscReal w = 2.62206 / (2.0 * 0.2 * CAPKmin);
        const PetscReal len[3] = {LMaxX - LMinX, LMaxY - LMinY, LMaxZ - LMinZ};
        for (PetscInt d = 0; d < Dimension; ++d)
            if (w > 0.45 * len[d])
                throw std::runtime_error(
                    "ValidateOrThrow: the CAP layer (width 6.555/CAPKmin = " + std::to_string(w) +
                    " a.u. at each wall) exceeds 45% of the " + std::string(1, "xyz"[d]) +
                    " extent (" + std::to_string(len[d]) + " a.u.). Reduce CAPWidth (or increase CAPKmin; "
                    "electrons slower than ~CAPKmin are then partly reflected) or enlarge the domain.");
    }

    // Propagation: every polarised axis needs a field expression (an empty one
    // makes muParser throw inside the time stepper). Laser = ... has already
    // been copied into the empty LaserX/Y/Z by initParsers().
    if (EnablePropagation)
    {
        if ((Polarization.empty() || Polarization == "none") && NFieldTerms() == 0)
            throw std::runtime_error(
                "ValidateOrThrow: EnablePropagation=1 needs a Polarization (x, y, z, xy, xz, yz "
                "or xyz) or a FieldTerm. For field-free propagation use e.g. Polarization = x, LaserX = 0.");
        if (Polarization.empty()) Polarization = "none";
        const std::string *expr[3] = {&LaserX, &LaserY, &LaserZ};
        for (int d = 0; d < 3; ++d)
            if (Polarization.find("xyz"[d]) != std::string::npos && expr[d]->empty())
                throw std::runtime_error(
                    std::string("ValidateOrThrow: Polarization '") + Polarization + "' includes " +
                    "xyz"[d] + " but Laser" + (char)std::toupper("xyz"[d]) + " (and Laser) is empty. "
                    "Set e.g. Laser" + (char)std::toupper("xyz"[d]) + " = Ampx*sin(Omegax*t)*sin(pi*t/T)^2.");
    }


    // Structured field terms: the profile may use x, y, z only and the time
    // factor t only (muParser reports anything else at the first evaluation,
    // which would be deep inside the assembly or the time stepper).
    for (PetscInt k = 0; k < NFieldTerms(); ++k) {
        const std::string id = "FieldTerm " + std::to_string((int)k + 1);
        try {
            for (const auto &v : FieldProfileExpr[k].GetUsedVar())
                if (v.first != "x" && v.first != "y" && v.first != "z")
                    throw std::runtime_error("ValidateOrThrow: " + id + ": the profile '" + FieldTermProfile[k] +
                        "' uses '" + v.first + "'; it may depend on x, y and z only");
            for (const auto &v : FieldTimeExpr[k].GetUsedVar())
                if (v.first != "t")
                    throw std::runtime_error("ValidateOrThrow: " + id + ": the time function '" + FieldTermTime[k] +
                        "' uses '" + v.first + "'; it may depend on t only");
            (void)FieldProfile(k, 0.0, 0.0, 0.0);
            (void)FieldTime(k, 0.0);
        } catch (const mu::Parser::exception_type &e) {
            throw std::runtime_error("ValidateOrThrow: " + id + ": " + e.GetMsg());
        }
        for (int d = (int)Dimension; d < 3; ++d)
            if (FieldProfileExpr[k].GetUsedVar().count(std::string(1, "xyz"[d])))
                throw std::runtime_error("ValidateOrThrow: " + id + ": the profile uses " + std::string(1, "xyz"[d]) +
                    ", which does not exist in a " + std::to_string((int)Dimension) + "D problem");
    }
    if (NFieldTerms() > 0 && AccelerationMethod == "operator")
        throw std::runtime_error("ValidateOrThrow: AccelerationMethod = operator needs grad w of every FieldTerm; "
            "use auto or discrete (the discrete equation of motion includes the field terms exactly)");

    // The propagation assembles s dV/dx_d on every axis (acceleration output);
    // an empty expression used to abort inside muParser.
    if (EnablePropagation) {
        const std::string *dV[3] = {&PotentialDerivativeX, &PotentialDerivativeY, &PotentialDerivativeZ};
        for (int d = 0; d < (int)Dimension; ++d)
            if (dV[d]->find_first_not_of(" \t") == std::string::npos)
                throw std::runtime_error(std::string("ValidateOrThrow: PotentialDerivative") + (char)std::toupper("xyz"[d]) +
                    " is empty; give dV/d" + "xyz"[d] + " of Potential (0 for a constant potential)");
    }


    // t-SURFF: extraction box |x_d| = R inside every axis and in front of the
    // CAP layers; a momentum grid or a momentum file.
    if (TSurff) {
        if (!EnablePropagation)
            throw std::runtime_error("ValidateOrThrow: TSurff = 1 needs EnablePropagation = 1");
        if (!(SurffRadius > 0.0))
            throw std::runtime_error("ValidateOrThrow: TSurff = 1 needs SurffRadius > 0");
        const PetscReal lo[3] = {LMinX, LMinY, LMinZ}, hi[3] = {LMaxX, LMaxY, LMaxZ};
        const PetscReal w = EnableCAP ? 2.62206 / (2.0 * 0.2 * CAPKmin) : 0.0;
        for (PetscInt d = 0; d < Dimension; ++d) {
            if (!(-SurffRadius > lo[d] && SurffRadius < hi[d]))
                throw std::runtime_error("ValidateOrThrow: the t-SURFF surface |x| = SurffRadius must lie inside the "
                    + std::string(1, "xyz"[d]) + " domain");
            if (EnableCAP && !(-SurffRadius > lo[d] + w && SurffRadius < hi[d] - w))
                throw std::runtime_error("ValidateOrThrow: the t-SURFF surface must lie in front of the CAP layer ("
                    + std::string(1, "xyz"[d]) + " axis: |x| < " + std::to_string(PetscMin(-lo[d], hi[d]) - w) + ")");
        }
        if (SurffMomenta.empty()) {
            for (PetscInt d = 0; d < Dimension; ++d) {
                if (!SurffKminSet) SurffKmin[d] = -SurffKmax[d];
                if (SurffNk[d] < 1 || (SurffNk[d] > 1 && !(SurffKmax[d] > SurffKmin[d])))
                    throw std::runtime_error("ValidateOrThrow: TSurff = 1 needs SurffNk >= 1 and SurffKmax > SurffKmin "
                        "on every axis (or a SurffMomenta file)");
            }
        }
        if (OutputStrideSurff < 1)
            throw std::runtime_error("ValidateOrThrow: OutputStrideSurff must be >= 1");
        if (SurffEndTerm && !EnableCAP)
            throw std::runtime_error("ValidateOrThrow: SurffEndTerm = 1 needs EnableCAP = 1 (the absorber is the "
                                     "outgoing-wave condition of the end term)");
        if (SurffEndTerm && SurffInterior)
            throw std::runtime_error("ValidateOrThrow: SurffEndTerm and SurffInterior both add the part of psi left "
                                     "at FinalTime; use one of them");
        if (SurffEndTermDk < 0.0)
            throw std::runtime_error("ValidateOrThrow: SurffEndTermDk must be >= 0 (0 = automatic)");
        if (SurffQuadrature < 0 || SurffQuadrature > 64)
            throw std::runtime_error("ValidateOrThrow: SurffQuadrature must be in [0, 64] (0 = NQuadratures)");
        // Volkov states outside the surface: a FieldTerm must not act there,
        // i.e. grad w_k must vanish wherever max_d |x_d| >= SurffRadius.
        // Sampled on a lattice (central differences) against max |grad w_k|.
        for (PetscInt k = 0; k < NFieldTerms(); ++k) {
            const int n = Dimension == 3 ? 33 : (Dimension == 2 ? 121 : 2001);
            const int ny = Dimension >= 2 ? n : 1, nz = Dimension == 3 ? n : 1;
            const PetscReal lo3[3] = {LMinX, LMinY, LMinZ}, hi3[3] = {LMaxX, LMaxY, LMaxZ};
            PetscReal gIn = 0.0, gOut = 0.0, xOut[3] = {0, 0, 0};
            for (int i = 0; i < n; ++i) for (int j = 0; j < ny; ++j) for (int l = 0; l < nz; ++l) {
                const int idx[3] = {i, j, l};
                PetscReal r[3] = {0, 0, 0}, g2 = 0.0, rmax = 0.0;
                for (int d = 0; d < (int)Dimension; ++d) {
                    r[d] = lo3[d] + (hi3[d] - lo3[d]) * idx[d] / (n - 1);
                    rmax = PetscMax(rmax, PetscAbsReal(r[d]));
                }
                for (int d = 0; d < (int)Dimension; ++d) {
                    const PetscReal h = 1e-4 * (hi3[d] - lo3[d]);
                    PetscReal rp[3] = {r[0], r[1], r[2]}, rm[3] = {r[0], r[1], r[2]};
                    rp[d] += h; rm[d] -= h;
                    const PetscReal g = (FieldProfile(k, rp[0], rp[1], rp[2]) - FieldProfile(k, rm[0], rm[1], rm[2])) / (2 * h);
                    g2 += g * g;
                }
                const PetscReal g = PetscSqrtReal(g2);
                if (!std::isfinite(g))
                    throw std::runtime_error("ValidateOrThrow: FieldTerm " + std::to_string((int)k + 1) +
                        ": the profile is not finite at (" + std::to_string(r[0]) + ", " + std::to_string(r[1]) +
                        ", " + std::to_string(r[2]) + ")");
                gIn = PetscMax(gIn, g);
                if (rmax >= SurffRadius && g > gOut) { gOut = g; for (int d = 0; d < 3; ++d) xOut[d] = r[d]; }
            }
            if (gOut > 1e-6 * gIn)
                throw std::runtime_error("ValidateOrThrow: TSurff = 1 needs every FieldTerm to vanish outside the "
                    "t-SURFF surface (Volkov states there feel only the uniform LaserX/Y/Z field). FieldTerm " +
                    std::to_string((int)k + 1) + " has |grad w| = " + std::to_string(gOut) + " (" +
                    std::to_string(gOut / PetscMax(gIn, (PetscReal)1e-300)) + " of its maximum) at (" +
                    std::to_string(xOut[0]) + ", " + std::to_string(xOut[1]) + ", " + std::to_string(xOut[2]) +
                    "); make the profile decay inside SurffRadius or enlarge SurffRadius");
        }
    }

    // KnotSequence: every axis must name an implemented sequence (an unknown
    // name used to fall back to uniform knots with only a warning).
    {
        static const std::set<std::string> known = {
            "uniform", "symexp", "symtanu", "symtann", "symlogtan", "hydrogenic",
            "interface", "interfaces"};
        for (PetscInt d = 0; d < Dimension; ++d) {
            const std::string &k = KnotSeq[d];
            if (k == "adaptive" || k == "adaptive_wf")
                throw std::runtime_error("ValidateOrThrow: KnotSequence '" + k + "' has been removed "
                    "(it was a placeholder that produced uniform knots). Use uniform, symexp, symtanu, "
                    "symtann, symlogtan, hydrogenic or interface.");
            if (!known.count(k))
                throw std::runtime_error("ValidateOrThrow: unknown KnotSequence '" + k + "' on axis " +
                    std::string(1, "xyz"[d]) + ". Use uniform, symexp, symtanu, symtann, symlogtan, "
                    "hydrogenic or interface.");
        }
    }
}
