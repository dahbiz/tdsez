#include "tdsez_parser.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

/**
 * @file parser_validation.cpp
 * @brief Physical and cross-option validation of parsed input.
 */

namespace {

// Sample each active axis through the center of the other axes, plus every
// corner. Finite sampling cannot prove a field is valid everywhere, but this
// covers all coordinates without a costly 16^Dimension tensor grid.
template <typename F>
void sampleDomain(PetscInt dim, const std::array<PetscReal, 3>& lo,
                  const std::array<PetscReal, 3>& hi, F&& evaluate)
{
    constexpr PetscInt samplesPerAxis = 16;
    std::array<PetscReal, 3> center = {0.0, 0.0, 0.0};
    for (PetscInt d = 0; d < dim; ++d) center[d] = (lo[d] + hi[d]) / 2.0;

    for (PetscInt d = 0; d < dim; ++d) {
        for (PetscInt i = 0; i < samplesPerAxis; ++i) {
            auto point = center;
            point[d] = lo[d] + (hi[d] - lo[d]) * i / (samplesPerAxis - 1);
            evaluate(point[0], point[1], point[2]);
        }
    }
    for (PetscInt corner = 0; corner < (1 << dim); ++corner) {
        auto point = center;
        for (PetscInt d = 0; d < dim; ++d)
            point[d] = (corner & (1 << d)) ? hi[d] : lo[d];
        evaluate(point[0], point[1], point[2]);
    }
}

std::string sampleLocation(PetscReal x, PetscReal y, PetscReal z)
{
    return " at (" + std::to_string(x) + ", " + std::to_string(y) +
           ", " + std::to_string(z) + ")";
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

    // --- requested states ---
    if (NBoundStates < 1)
        throw std::runtime_error("ValidateOrThrow: NBoundStates must be >= 1");

    // --- initial state selection ---
    // Parse InitialState into a mode + (index,coeff) terms, and guard indices
    // against NBoundStates. Supported forms:
    //   "ground"              -> mode 0 (== state:0)
    //   "state:N"             -> mode 1, single bound state N
    //   "sup: a*N + b*M [+c*P]"-> mode 2, coherent superposition of <=3 states
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
        } else if (s.rfind("sup:", 0) == 0) {
            // "sup: a*N + b*M [+ c*P]"  (real coeffs, <=3 terms, '+'/'-' separated)
            InitialStateMode = 2;
            std::string body = s.substr(4);
            // split on '+' (keep unary '-' inside terms)
            std::vector<std::string> terms;
            {
                std::string cur;
                for (char c : body) {
                    if (c == '+') { if (!cur.empty()) { terms.push_back(cur); cur.clear(); } }
                    else cur += c;
                }
                if (!cur.empty()) terms.push_back(cur);
            }
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
                PetscReal coeff;
                PetscInt  idx;
                try {
                    size_t consumed = 0;
                    coeff = (PetscReal)std::stod(cstr, &consumed);
                    if (consumed != cstr.size())
                        throw std::invalid_argument("invalid coefficient");
                }
                catch (...) {
                    throw std::runtime_error(
                        "ValidateOrThrow: InitialState term '" + t + "' has a non-numeric coefficient");
                }
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
                if (!std::isfinite(coeff))
                    throw std::runtime_error(
                        "ValidateOrThrow: InitialState term '" + t + "' coefficient is not finite");
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
                "ValidateOrThrow: InitialState must be 'ground', 'state:N', or "
                "'sup: a*N + b*M [+ c*P]' (got '" + s + "')");
        }
    }

    // --- bound-state save format ---
    if (BoundStateFormat != "complex" && BoundStateFormat != "real")
        throw std::runtime_error(
            "ValidateOrThrow: BoundStateFormat must be 'complex' or 'real' (got '" + BoundStateFormat + "')");

    const std::array<PetscReal, 3> boundsLow = {LMinX, LMinY, LMinZ};
    const std::array<PetscReal, 3> boundsHigh = {LMaxX, LMaxY, LMaxZ};

    // A non-positive mass makes the kinetic operator indefinite.
    sampleDomain((PetscInt)Dimension, boundsLow, boundsHigh,
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
                throw std::runtime_error("ValidateOrThrow: Mass must be > 0 and finite"
                                         + sampleLocation(x, y, z) + " (got "
                                         + to_string(mass) + ")");
        });

    // A non-finite potential poisons matrix assembly.
    sampleDomain((PetscInt)Dimension, boundsLow, boundsHigh,
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
                throw std::runtime_error("ValidateOrThrow: Potential is not finite"
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
        Polarization = p;
        if (valid.find(p) == valid.end())
            throw std::runtime_error(
                "ValidateOrThrow: Polarization '" + Polarization + "' is invalid. "
                "Use one of: none, x, y, z, xy, xz, yz, xyz (case-insensitive).");
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

    // EnvelopeType: normalise case only (kept permissive).
    EnvelopeType = lower(EnvelopeType);

    // CAP requires kmin > 0: kmin<=0 makes the Manolopoulos absorb width
    // diverge (absorbs the ENTIRE domain) -> silent TS divergence.
    if (EnableCAP && CAPKmin <= 0.0)
        throw std::runtime_error(
            "ValidateOrThrow: EnableCAP=1 but CAPKmin=" + to_string(CAPKmin) +
            " <= 0. A zero/negative kmin makes the absorbing width diverge and the "
            "time stepper diverges. Set CAPKmin > 0 (e.g. 0.1) or disable CAP.");

    // Propagation requires a field expression: an empty Laser/LaserX/Y/Z makes
    // muParser throw on an empty expression inside the TS monitor (after assembly).
    if (EnablePropagation)
    {
        bool hasField = !(LaserX.empty() && LaserY.empty() &&
                          LaserZ.empty() && Laser.empty());
        if (!hasField)
            throw std::runtime_error(
                "ValidateOrThrow: EnablePropagation=1 but no field expression is set "
                "(LaserX/LaserY/LaserZ/Laser are all empty). Set e.g. "
                "LaserX = Amplitude*sin(Omega*t).");
    }

    // The "adaptive" and "adaptive_wf" KnotSequences validation is disabled —
    // implementations are saved in dev/ for a future push. The stubs fall back
    // to uniform knots and do NOT evaluate the potential derivative, so the
    // PotentialDerivativeX requirement no longer applies. Re-enable this block
    // when restoring the full implementations from dev/.
    // {
    //     auto needs = [&](const std::string &seq) {
    //         return seq == "adaptive" || seq == "adaptive_wf";
    //     };
    //     if (needs(KnotSeq[0]) && PotentialDerivativeX.empty())
    //         throw std::runtime_error(
    //             "ValidateOrThrow: KnotSequence='" + KnotSeq[0] + "' (X axis) requires "
    //             "PotentialDerivativeX to be set (it is empty). Provide the analytic "
    //             "derivative of the potential, e.g. PotentialDerivativeX = -2.0*x.");
    //     if (Dimension >= 2 && needs(KnotSeq[1]) && PotentialDerivativeY.empty())
    //         throw std::runtime_error(
    //             "ValidateOrThrow: KnotSequence='" + KnotSeq[1] + "' (Y axis) requires "
    //             "PotentialDerivativeY to be set (it is empty). Provide the analytic "
    //             "derivative of the potential, e.g. PotentialDerivativeY = -2.0*y.");
    //     if (Dimension >= 3 && needs(KnotSeq[2]) && PotentialDerivativeZ.empty())
    //         throw std::runtime_error(
    //             "ValidateOrThrow: KnotSequence='" + KnotSeq[2] + "' (Z axis) requires "
    //             "PotentialDerivativeZ to be set (it is empty). Provide the analytic "
    //             "derivative of the potential, e.g. PotentialDerivativeZ = -2.0*z.");
    // }
}
