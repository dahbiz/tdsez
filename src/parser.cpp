#include "tdsez_parser.hpp"
#include "tdsez_info.hpp"

#include <slepc.h>
#include <petiga.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <filesystem>
#include <petscviewerhdf5.h>
#include <H5Epublic.h>
#include "petscmat.h"
#include <algorithm>
#include <functional>
#include <type_traits>
#include <set>
#include <unordered_map>
#include <petscblaslapack.h>

#include "muParser.h"

/**
 * @file parser.cpp
 * @brief Read input files and initialize muParser expressions.
 * @author TDSEZ Project
 */

/// @brief A number that is the whole value (spaces aside). std::stod/std::stoi read
///        a leading number and drop the rest, so "NSplines = 100, 16, 16" became 100 and
///        "NBoundStates = 4.5" became 4 without a word; these throw instead (the caller
///        reports the key and the value).
static double StrictReal(const std::string &v)
{
    const size_t a = v.find_first_not_of(" \t\r"), b = v.find_last_not_of(" \t\r");
    if (a == std::string::npos) throw std::invalid_argument("empty value, a number is expected");
    const std::string t = v.substr(a, b - a + 1);
    size_t pos = 0;
    const double x = std::stod(t, &pos);
    if (pos != t.size())
        throw std::invalid_argument("'" + t + "' is not a single number" +
                                    (t.find(',') != std::string::npos ? " (this key takes one value, not a list)" : ""));
    return x;
}

/// @brief An integer that is the whole value; an integral real such as "4.0" is accepted.
static long StrictInt(const std::string &v)
{
    const double x = StrictReal(v);
    if (x != std::floor(x) || std::abs(x) > 2e9) throw std::invalid_argument("'" + v + "' is not an integer");
    return (long)x;
}

/// @brief Parse a comma-separated per-axis list "a,b,c" into arr[0..2]. A
///        single token replicates to all axes (as for the hydrogenic keys).
template <typename T>
static void ParseAxisList(const std::string &val, T (&arr)[3])
{
    std::string s = val;
    s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
    size_t p = 0, n = 0;
    while (p < s.size() && n < 3) {
        const size_t c = s.find(',', p);
        const std::string tok = (c == std::string::npos) ? s.substr(p) : s.substr(p, c - p);
        if constexpr (std::is_integral_v<T>) arr[n++] = (T)StrictInt(tok);
        else                                 arr[n++] = (T)StrictReal(tok);
        if (c == std::string::npos) break;
        p = c + 1;
    }
    if (n == 1) for (int i = 1; i < 3; ++i) arr[i] = arr[0];
}

/// @brief Build a handler lambda that parses a comma-separated "a,b,c"
///        per-axis list string into three PetscReal references. A single
///        token replicates to all axes; missing tokens fall back to the
///        first token.
/// @param x  Reference to store the x-axis value.
/// @param y  Reference to store the y-axis value.
/// @param z  Reference to store the z-axis value.
/// @return Lambda accepting a string and populating x, y, z.
static std::function<void(const std::string &)>
axisListHandler(PetscReal &x, PetscReal &y, PetscReal &z)
{
    return [ &x, &y, &z ](const std::string &val) {
        std::string s = val;
        s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
        std::vector<PetscReal> v;
        size_t p = 0;
        while (p < s.size()) {
            size_t c = s.find(',', p);
            std::string tok = (c == std::string::npos) ? s.substr(p) : s.substr(p, c - p);
            v.push_back(StrictReal(tok));
            if (c == std::string::npos) break;
            p = c + 1;
        }
        PetscReal c0 = v.empty() ? 0.0 : v[0];
        PetscReal c1 = v.size() >= 2 ? v[1] : c0;
        PetscReal c2 = v.size() >= 3 ? v[2] : c0;
        x = c0; y = c1; z = c2;
    };
}

/// @brief Read and parse a key=value parameter file, stripping comments.
///        Populates the static TDSEZParser member fields from the .prm input.
/// @param prm  Path to the input parameter file.
/// FieldTerm = w(x,y,z) ; f(t): append one separable field term.
static void AddFieldTerm(const std::string &val)
{
    const size_t semi = val.find(';');
    if (semi == std::string::npos || val.find(';', semi + 1) != std::string::npos)
        throw std::runtime_error("FieldTerm must be 'profile w(x,y,z) ; time function f(t)' (one ';'), got '" + val + "'");
    std::string w = val.substr(0, semi), f = val.substr(semi + 1);
    TDSEZParser::trim(w);
    TDSEZParser::trim(f);
    if (w.empty() || f.empty())
        throw std::runtime_error("FieldTerm needs both a profile w(x,y,z) and a time function f(t), got '" + val + "'");
    TDSEZParser::FieldTermProfile.push_back(w);
    TDSEZParser::FieldTermTime.push_back(f);
}

void TDSEZParser::PrmReader(const std::string &prm)
{
    std::ifstream in(prm);
    if (!in) throw std::runtime_error("Cannot open file: " + prm);
    std::string line;
    std::set<std::string> seenKeys;      // a key given twice: the last value would win silently
    std::vector<std::string> dupKeys;
    FieldTermProfile.clear();            // FieldTerm lines accumulate within one file only
    FieldTermTime.clear();
    while (std::getline(in, line))
    {
        /* To remove comments */
        line = line.substr(0, line.find('#'));
        trim(line);
        if (line.empty()) continue;

        auto delimPos = line.find('=');
        if (delimPos == std::string::npos) continue;
        std::string key = line.substr(0, delimPos);
        std::string val = line.substr(delimPos + 1);
        trim(key); 
        trim(val);
        // Variables and FieldTerm lines accumulate (documented); every other key may appear once.
        if (key != "Variables" && key != "FieldTerm" && key != "StructuredField" && !seenKeys.insert(key).second)
            dupKeys.push_back(key);

        /* ---- Modular O(1) key dispatch ------------------------------------ */
        /* Keys are resolved through a single unordered_map (hash lookup) instead */
        /* of a linear if/else-if chain. Each entry maps a key (and any aliases) */
        /* to a typed handler lambda. To add a new key, register it here — no     */
        /* chain to extend. Unknown keys now emit a warning instead of being      */
        /* silently ignored.                                                       */
        using H = std::function<void(const std::string &)>;
        static const auto &table = []() -> const std::unordered_map<std::string, H>&
        {
            // small helpers to keep entries terse (generic: deduce target type)
            static const auto I  = [](auto &dst){ return H{[&dst](const std::string &v){ dst = StrictInt(v); }}; };
            static const auto R  = [](auto &dst){ return H{[&dst](const std::string &v){ dst = StrictReal(v); }}; };
            static const auto B  = [](auto &dst){ return H{[&dst](const std::string &v){
                std::string l = v; for (auto &c : l) c = (char)std::tolower((unsigned char)c);
                if      (l == "true" || l == "yes" || l == "on")  dst = PETSC_TRUE;
                else if (l == "false" || l == "no" || l == "off") dst = PETSC_FALSE;
                else dst = static_cast<PetscBool>(StrictInt(v) != 0); }}; };
            static const auto S  = [](auto &dst){ return H{[&dst](const std::string &v){ dst = v; }}; };

            static const std::unordered_map<std::string, H> m =
            {
                // --- simulation / grid ---
                {"Verbose",            B(Verbose)},
                {"StrictInput",        B(StrictInput)},
                {"Dimension",          I(Dimension)},
                {"Domain",    H{[](const std::string &val){
                    std::string s = val; s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                    if (s.find('[') != std::string::npos)
                    {
                        std::vector<std::string> pairs; size_t pos = 0;
                        while (pos < s.size()) {
                            size_t ob = s.find('[', pos), cb = s.find(']', ob);
                            if (ob == std::string::npos || cb == std::string::npos) break;
                            pairs.push_back(s.substr(ob + 1, cb - ob - 1)); pos = cb + 1;
                        }
                        auto parsePair = [](const std::string &p, PetscReal &lo, PetscReal &hi){
                            size_t c = p.find(',');
                            if (c != std::string::npos) { lo = StrictReal(p.substr(0, c)); hi = StrictReal(p.substr(c + 1)); } };
                        if (pairs.size() >= 1) parsePair(pairs[0], LMinX, LMaxX);
                        if (pairs.size() >= 2) parsePair(pairs[1], LMinY, LMaxY); else { LMinY = LMinX; LMaxY = LMaxX; }
                        if (pairs.size() >= 3) parsePair(pairs[2], LMinZ, LMaxZ); else { LMinZ = LMinX; LMaxZ = LMaxX; }
                    } else {
                        // flat list: "lo, hi" for every axis, or one pair per
                        // axis "xlo, xhi, ylo, yhi[, zlo, zhi]" (the extra values
                        // used to be ignored silently, giving a cubic box)
                        std::vector<PetscReal> v;
                        size_t p = 0;
                        while (p <= s.size()) {
                            const size_t c = s.find(',', p);
                            const std::string tok = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
                            if (!tok.empty()) v.push_back(StrictReal(tok));
                            if (c == std::string::npos) break;
                            p = c + 1;
                        }
                        if (v.size() != 2 && v.size() != 4 && v.size() != 6)
                            throw std::runtime_error("Domain must be 'lo, hi' (every axis) or one 'lo, hi' pair per "
                                                     "axis, e.g. -50, 50, -20, 20 (got " + std::to_string(v.size()) + " values)");
                        PetscReal *lo[3] = {&LMinX, &LMinY, &LMinZ}, *hi[3] = {&LMaxX, &LMaxY, &LMaxZ};
                        for (size_t a = 0; a < 3; ++a) {
                            const size_t k = (2 * a < v.size()) ? 2 * a : 0;   // missing axes copy x
                            *lo[a] = v[k]; *hi[a] = v[k + 1];
                        }
                    }
                }}},
                {"LMinX",  R(LMinX)}, {"LMinY",  R(LMinY)}, {"LMinZ",  R(LMinZ)},
                {"LMaxX",  R(LMaxX)}, {"LMaxY",  R(LMaxY)}, {"LMaxZ",  R(LMaxZ)},
                {"OffsetX",R(OffsetX)},{"OffsetY",R(OffsetY)},{"OffsetZ",R(OffsetZ)},
                {"NQuadratures", I(NQuadratures)},
                {"SplineDegree", I(SplineDegree)},
                {"Nelements",        I(Nelements)},
                {"NSplines",         I(Nelements)},   // backward-compat alias
                {"NumberOfSplines",  I(Nelements)},   // intuitive alias
                {"ElementCount",     I(Nelements)},   // intuitive alias
                {"NumberOfElements", I(Nelements)},   // basis-agnostic alias
                {"KnotSequence", H{[](const std::string &val){
                    std::string s = val; s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                    // Lowercase so sequence names are case-insensitive
                    // (e.g. "SYMEXP"/"SymExp" match "symexp"). The stored
                    // tokens must already be lowercase so the later string
                    // comparison in core.cpp succeeds.
                    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                    size_t p = 0, n = 0;
                    while (p < s.size() && n < 3) {
                        size_t c = s.find(',', p);
                        std::string tok = (c == std::string::npos) ? s.substr(p) : s.substr(p, c - p);
                        KnotSeq[n++] = tok; if (c == std::string::npos) break; p = c + 1;
                    }
                    if (n == 1) for (int i = 1; i < 3; ++i) KnotSeq[i] = KnotSeq[0];
                    KnotSequence = KnotSeq[0];
                }}},
                // Per-axis integer list (comma-separated, single value replicates
                // to all axes). Used by the hydrogenic knot-sequence tuning keys.
                // Hydrogenic tuning (per-axis lists, single value replicates).
                // Long names and descriptive aliases map to the same storage.
                {"HydrogenicNLin",          H{[](const std::string &v){ ParseAxisList(v, HydrogenicNLin); }}},
                {"HydrogenicInnerElements", H{[](const std::string &v){ ParseAxisList(v, HydrogenicNLin); }}},
                {"InnerElements",           H{[](const std::string &v){ ParseAxisList(v, HydrogenicNLin); }}},
                {"HydrogenicNExp",          H{[](const std::string &v){ ParseAxisList(v, HydrogenicNExp); }}},
                {"HydrogenicOuterElements", H{[](const std::string &v){ ParseAxisList(v, HydrogenicNExp); }}},
                {"OuterElements",           H{[](const std::string &v){ ParseAxisList(v, HydrogenicNExp); }}},
                {"HydrogenicR1",            H{[](const std::string &v){ ParseAxisList(v, HydrogenicR1); }}},
                {"HydrogenicInnerSpacing",  H{[](const std::string &v){ ParseAxisList(v, HydrogenicR1); }}},
                {"InnerSpacing",            H{[](const std::string &v){ ParseAxisList(v, HydrogenicR1); }}},
                {"HydrogenicHMax",          H{[](const std::string &v){ ParseAxisList(v, HydrogenicHMax); }}},
                {"HydrogenicMaxSpacing",    H{[](const std::string &v){ ParseAxisList(v, HydrogenicHMax); }}},
                {"TransverseElementFactor", R(TransverseElementFactor)},
                {"Mass",     S(Mass)},
                {"Charge",    R(Q)},
                {"ParticleCharge", R(Q)},     // intuitive alias
                {"MassIsConstant", B(MassIsConstant)},
                {"ObserveAllAxes", B(ObserveAllAxes)},
                // t-SURFF (only computed with TSurff = 1) and the wave-packet initial state
                {"TSurff",            B(TSurff)},
                {"SurffRadius",       R(SurffRadius)},
                {"SurffKmax",         H{[](const std::string &v){ ParseAxisList(v, SurffKmax); }}},
                {"SurffKmin",         H{[](const std::string &v){ ParseAxisList(v, SurffKmin); SurffKminSet = PETSC_TRUE; }}},
                {"SurffNk",           H{[](const std::string &v){ ParseAxisList(v, SurffNk); }}},
                {"SurffMomenta",      S(SurffMomenta)},
                {"OutputStrideSurff", I(OutputStrideSurff)},
                // restart files
                {"CheckpointStride",   I(CheckpointStride)},
                {"CheckpointWallTime", R(CheckpointWallTime)},
                {"Restart",            B(Restart)},
                {"CheckpointFile",     S(CheckpointFile)},
                {"CheckpointGuess",    B(CheckpointGuess)},
                {"SurffQuadrature",   I(SurffQuadrature)},
                {"SurffMassTolerance",      R(SurffMassTolerance)},
                {"SurffPotentialTolerance", R(SurffPotentialTolerance)},
                {"SurffInterior",     B(SurffInterior)},
                {"SurffEndTerm",      B(SurffEndTerm)},
                {"SurffEndTermDk",    R(SurffEndTermDk)},
                {"WavepacketCenter",   H{[](const std::string &v){ ParseAxisList(v, WavepacketCenter); }}},
                {"WavepacketMomentum", H{[](const std::string &v){ ParseAxisList(v, WavepacketMomentum); }}},
                {"WavepacketWidth",    H{[](const std::string &v){ ParseAxisList(v, WavepacketWidth); }}},

                // --- solver ---
                {"TargetEigenvalue", R(TargetEigenvalue)},
                {"TargetEnergy",      R(TargetEigenvalue)}, // intuitive alias
                {"UseDirectSolve", B(UseDirectSolve)},
                {"DirectSolve",     B(UseDirectSolve)},     // intuitive alias

                // --- time ---
                {"TimeStep",  R(TimeStep)},
                {"TimeStepSize", R(TimeStep)},  // intuitive alias
                {"FinalTime", R(FinalTime)},
                {"TotalTime", R(FinalTime)},    // intuitive alias

                // --- field / envelope (legacy scalar form) ---

                // --- field (per-component generalization) ---
                {"Ampx",R(Ampx)},{"Ampy",R(Ampy)},{"Ampz",R(Ampz)},
                {"CEPx",R(CEPx)},{"CEPy",R(CEPy)},{"CEPz",R(CEPz)},
                {"Omegax",R(Omegax)},{"Omegay",R(Omegay)},{"Omegaz",R(Omegaz)},
                {"LaserX",S(LaserX)},{"LaserY",S(LaserY)},{"LaserZ",S(LaserZ)},
                {"DriverX",S(LaserX)},{"DriverY",S(LaserY)},{"DriverZ",S(LaserZ)}, // driver alias
                // structured (separable) field: FieldTerm = w(x,y,z) ; f(t),
                // one line per term (the lines accumulate)
                {"FieldTerm",       H{AddFieldTerm}},
                // Mode = td (lab frame, default) | kh (static Kramers-Henneberger
                // problem: eigenstates of the cycle-averaged potential)
                {"Mode", H{[](const std::string &v){
                    std::string m = v; for (auto &c : m) c = (char)std::tolower((unsigned char)c);
                    if (m == "td" || m == "lab") KHFrame = PETSC_FALSE;
                    else if (m == "kh") KHFrame = PETSC_TRUE;
                    else throw std::runtime_error("Mode must be td or kh"); }}},
                // Gauge = length (s E.r, default) | velocity ((p - qA)^2 / 2m, velocity_gauge.cpp)
                {"Gauge", H{[](const std::string &v){
                    std::string m = v; for (auto &c : m) c = (char)std::tolower((unsigned char)c);
                    if (m == "length") VelocityGauge = PETSC_FALSE;
                    else if (m == "velocity") VelocityGauge = PETSC_TRUE;
                    else throw std::runtime_error("Gauge must be length or velocity"); }}},
                {"KHWindow",  H{[](const std::string &v){
                    std::string s = v; s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                    const size_t c = s.find(',');
                    if (c == std::string::npos) throw std::runtime_error("KHWindow must be 't0, t1'");
                    KHWindow[0] = StrictReal(s.substr(0, c)); KHWindow[1] = StrictReal(s.substr(c + 1)); }}},
                {"KHSamples",          I(KHSamples)},
                {"StructuredField", H{AddFieldTerm}},   // intuitive alias
                // nondipole coupling of a plane wave E(t - n.r/c)
                {"Multipole",            I(Multipole)},
                {"PropagationDirection", H{[](const std::string &v){ ParseAxisList(v, PropagationDir); }}},
                {"MultipoleMagnetic",    B(MultipoleMagnetic)},
                {"SpeedOfLight",         R(SpeedOfLight)},
                {"Driver", S(Laser)},
                {"Laser",  S(Laser)},

                // --- problem expressions ---
                {"Potential",          S(Potential)},
                {"ExternalPotential",  S(Potential)},  // intuitive alias
                {"PotentialDerivativeX", S(PotentialDerivativeX)},
                {"PotentialDerivativeY", S(PotentialDerivativeY)},
                {"PotentialDerivativeZ", S(PotentialDerivativeZ)},

                // --- general control ---
                {"EnablePropagation", B(EnablePropagation)},
                {"EnableGPU",          B(EnableGPU)},
                {"EnableCAP",          B(EnableCAP)},
                {"AbsorbingBoundary",  B(EnableCAP)},  // intuitive alias
                {"EnableLzDiag",       B(EnableLzDiag)},
                {"LzDiag",             B(EnableLzDiag)}, // intuitive alias
                {"BoundaryType",       S(BoundaryType)},
                {"Boundary",          S(BoundaryType)},   // intuitive alias
                {"OutputStrideWFS",    I(OutputStrideWFS)},
                {"OutputStrideObs",    I(OutputStrideObs)},
                {"ObservableStride",   I(OutputStrideObs)},   // alias
                {"OutputStrideTS",     I(OutputStrideTS)},
                {"HDF5Compress",       B(HDF5Compress)},
                {"HDF5CompressLevel",  I(HDF5CompressLevel)},
                {"CAPKmin",    R(CAPKmin)},
                {"CAPWidth",   R(CAPWidth)},
                {"CAPLayerWidth", R(CAPWidth)},
                {"KnotAlpha",  R(KnotAlpha)},
                {"KnotGrading", R(KnotAlpha)},         // intuitive alias
                {"GradingStrength", R(KnotAlpha)},     // intuitive alias
                {"NBoundStates",      I(NBoundStates)},
                {"NumberOfBoundStates",I(NBoundStates)}, // intuitive alias
                {"NBoundStatesSave",   B(NBoundStatesSave)},
                {"SaveBoundStates",    B(NBoundStatesSave)}, // intuitive alias
                {"SaveDipoleMatrix",   B(SaveDipoleMatrix)}, // write assembled Dx operator to PETSc binary
                {"SaveDipoleAxes",     S(SaveDipoleAxes)},   // which axes: x,xy,xz,xyz,all,none (CLI overrides)
                {"BoundStateFormat",   S(BoundStateFormat)},
                {"SaveStateFormat",    S(BoundStateFormat)}, // intuitive alias
                {"StateFormat",        S(BoundStateFormat)}, // intuitive alias
                {"InitialState",       S(InitialState)},
                {"AccelerationMethod", S(AccelerationMethod)},
                {"Propagator",         S(Propagator)},
                {"Preconditioner",     S(Preconditioner)},
                {"Basis",              S(Basis)},
                {"BasisType",          S(Basis)},
                {"QuadratureRule",     S(QuadratureRule)},
                {"QuadratureType",     S(QuadratureRule)},
                {"MGLevels",           I(MGLevels)},
                {"MGSmooth",           I(MGSmooth)},
                {"StartState",         S(InitialState)},     // intuitive alias
                {"NormalizeInitialState", B(NormalizeInitialState)},
                {"NormalizePsi0",      B(NormalizeInitialState)}, // intuitive alias
                {"PhysicsOutput",      S(PhysicsOutput)},
                {"OutputQuantities",   S(PhysicsOutput)},  // intuitive alias
                {"Polarization",       S(Polarization)},
                {"LaserPolarization",  S(Polarization)},    // intuitive alias
                {"DriverPolarization", S(Polarization)},    // intuitive alias

                // --- user-defined muParser constants ---
                // Comma-separated Name=number pairs; multiple Variables lines
                // accumulate. Each name becomes a muParser constant usable in
                // ANY expression (Potential, Mass, Laser, ...). Example:
                //   Variables = V0=1.2, w=0.5, Rc=4.0
                //   Potential = 0.5*(x*x+y*y) + V0*exp(-w*Rc)
                {"Variables", H{[](const std::string &val){
                    std::string s = val;
                    s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                    if (s.empty()) return;
                    size_t p = 0;
                    while (p < s.size()) {
                        size_t c = s.find(',', p);
                        std::string pair = (c == std::string::npos) ? s.substr(p) : s.substr(p, c - p);
                        size_t eq = pair.find('=');
                        if (eq != std::string::npos) {
                            std::string name = pair.substr(0, eq);
                            std::string num  = pair.substr(eq + 1);
                            if (!name.empty())
                                userConsts_[name] = std::stod(num);
                        }
                        if (c == std::string::npos) break;
                        p = c + 1;
                    }
                }}},


                // --- per-axis LIST forms (Domain / KnotSequence convention) ---
                {"DriverAmplitude",  axisListHandler(Ampx, Ampy, Ampz)},
                {"DriverFrequency",  axisListHandler(Omegax, Omegay, Omegaz)},
                {"DriverCEP",        axisListHandler(CEPx, CEPy, CEPz)},
                {"LaserAmplitude",  axisListHandler(Ampx, Ampy, Ampz)},
                {"LaserFrequency",  axisListHandler(Omegax, Omegay, Omegaz)},
                {"CarrierEnvelopePhase", axisListHandler(CEPx, CEPy, CEPz)},
            };
            return m;
        };

        // Keys of removed features or parameters that had no effect: say why
        // instead of "unknown". A removed key whose value changed nothing
        // (e.g. Planck = 1, EnvelopeType = custom) only warns.
        struct Removed { const char *why; bool (*harmless)(const std::string &); };
        static const auto never = [](const std::string &) { return false; };
        static const auto always = [](const std::string &) { return true; };
        static const auto isOne = [](const std::string &v) {
            try { return std::stod(v) == 1.0; } catch (...) { return false; } };
        static const auto isCustom = [](const std::string &v) {
            std::string l = v; for (auto &c : l) c = (char)std::tolower((unsigned char)c);
            return l.empty() || l == "custom"; };
        static const std::unordered_map<std::string, Removed> removed = {
            {"SurffCouplingSign", {"the coupling sign of t-SURFF follows Charge", never}},
            {"EnvelopeType",      {"the pulse envelope is written in the field expression (LaserX/Y/Z or Laser)", isCustom}},
            {"Envelope",          {"the pulse envelope is written in the field expression (LaserX/Y/Z or Laser)", never}},
            {"Hbar",              {"TDSE-Z works in atomic units (hbar = 1)", isOne}},
            {"Planck",            {"TDSE-Z works in atomic units (hbar = 1)", isOne}},
            {"Amplitude",         {"legacy scalar field parameter; use Ampx/Ampy/Ampz, or Variables = Name=value", never}},
            {"Omega",             {"legacy scalar field parameter; use Omegax/Omegay/Omegaz, or Variables = Name=value", never}},
            {"Phase",             {"legacy scalar field parameter; use CEPx/CEPy/CEPz, or Variables = Name=value", never}},
            {"PulseDuration",     {"name constants for the field expression with Variables = PulseDuration=..., PulseCenter=...", never}},
            {"PulseCenter",       {"name constants for the field expression with Variables = PulseDuration=..., PulseCenter=...", never}},
            {"Gamma",             {"it had no effect; the CAP is set by CAPKmin", never}},
            {"OutputStrideAC",    {"the autocorrelation is the 'autocorrelation' dataset of td/<name>_observables.h5, "
                                   "recorded every step", always}},
            {"CapStrength",       {"use CAPKmin (lowest momentum absorbed by the CAP)", never}},
            {"dinvMassX",         {"it had no effect; mass gradients are computed from Mass", never}},
            {"dinvMassY",         {"it had no effect; mass gradients are computed from Mass", never}},
            {"dinvMassZ",         {"it had no effect; mass gradients are computed from Mass", never}},
            {"LMin",              {"use Domain = lo, hi (or LMinX/LMaxX, ...)", never}},
            {"LMax",              {"use Domain = lo, hi (or LMinX/LMaxX, ...)", never}},
            {"AdaptiveKappa",     {"the adaptive knot sequences have been removed", never}},
            {"AdaptivePower",     {"the adaptive knot sequences have been removed", never}},
            {"AdaptiveWFCoarseN", {"the adaptive knot sequences have been removed", never}},
            {"AdaptiveWFKinLambda", {"the adaptive knot sequences have been removed", never}},
        };
        auto it = table().find(key);
        auto rm = removed.find(key);
        if (it != table().end()) {
            try {
                it->second(val);
            } catch (const std::exception &e) {   // unreadable value (number, FieldTerm, ...)
                TDSEZInfo().fatal("input validation failed: key '%s' = '%s': %s", key.c_str(), val.c_str(), e.what());
                MPI_Abort(PETSC_COMM_WORLD, 3);
            }
        }
        else if (rm != removed.end()) {
            const bool harmless = rm->second.harmless(val);
            const bool fatal = !harmless && TDSEZParser::StrictInput;
            if (fatal) TDSEZInfo().fatal("input key '%s' has been removed: %s.", key.c_str(), rm->second.why);
            else TDSEZInfo().warn("input key '%s' has been removed: %s.%s", key.c_str(), rm->second.why,
                                  harmless ? " This line has no effect; delete it." : "");
            if (fatal) MPI_Abort(PETSC_COMM_WORLD, 2);
        }
        else if (TDSEZParser::StrictInput) {
            // A typo'd key silently changing nothing is a real hazard in a
            // physics code (wrong potential, wrong mass, etc.). With StrictInput
            // (default true) we abort hard so the user sees the mistake.
            TDSEZInfo().fatal("unknown input key '%s'. Set 'StrictInput = 0' in the input file to tolerate "
                              "unknown keys.", key.c_str());
            MPI_Abort(PETSC_COMM_WORLD, 2);
        }
        else {
            TDSEZInfo().warn("unknown key '%s' ignored", key.c_str());
        }
    }

    if (!dupKeys.empty()) {
        std::string list;
        for (const auto &k : dupKeys) list += (list.empty() ? "" : ", ") + k;
        if (TDSEZParser::StrictInput)
            TDSEZInfo().fatal("input key(s) given more than once: %s.", list.c_str());
        else
            TDSEZInfo().warn("input key(s) given more than once: %s (the last value is used).", list.c_str());
        if (TDSEZParser::StrictInput) MPI_Abort(PETSC_COMM_WORLD, 2);
    }

    // Decode PhysicsOutput into per-quantity enable flags.
    // Default ("all" or empty) keeps everything enabled (backward compatible).
    // Otherwise only the listed tokens are enabled; the rest are turned off so
    // their per-step MatMults / reductions / HDF5 writes are skipped.
    {
        std::string s = PhysicsOutput;
        for (auto& c : s) c = (char)std::tolower((unsigned char)c);
        s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
        PhysicsOutputBad.clear();
        if (!(s.empty() || s == "all")) {
            out_dipole = out_accel = out_population = out_energy = out_current = out_autocorr = out_wfs = PETSC_FALSE;
            std::istringstream iss(s);
            std::string tok;
            while (std::getline(iss, tok, ',')) {
                if      (tok == "dipole")         out_dipole    = out_accel = PETSC_TRUE;
                else if (tok == "position")       out_dipole    = PETSC_TRUE;
                else if (tok == "acceleration")   out_accel     = PETSC_TRUE;
                else if (tok == "none" || tok.empty()) {}
                else if (tok == "population")      out_population = PETSC_TRUE;
                else if (tok == "energy")         out_energy    = PETSC_TRUE;
                else if (tok == "current")        out_current   = PETSC_TRUE;
                else if (tok == "autocorrelation" || tok == "ac") out_autocorr = PETSC_TRUE;
                else if (tok == "wfs")            out_wfs       = PETSC_TRUE;
                else PhysicsOutputBad += (PhysicsOutputBad.empty() ? "" : ", ") + tok;
            }
        }
    }

    /* Normalise enum-string members at parse time (case-insensitive) so that
       initParsers() / assembly see a canonical form regardless of how the user
       typed it. The harder cross-constraint checks live in ValidateOrThrow(),
       called later inside core.cpp's try/catch. */
    {
        auto lowerInPlace = [](std::string &s){
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        };
        lowerInPlace(Polarization);
        lowerInPlace(BoundaryType);
        if (BoundaryType == "natural") BoundaryType = "neumann";
        else if (BoundaryType == "wall")    BoundaryType = "dirichlet";
    }
}

/// @brief Initialize muParser instances and bind parser variables (V,
///        MassDist, Ex/Ey/Ez and their derivatives) to the
///        corresponding TDSEZParser static member expressions.
void TDSEZParser::initParsers()
{
    /* bind constant pi */
    for (auto *p : {&VPot, &dVPotX, &dVPotY, &dVPotZ, &LaserXexpr, &LaserYexpr, &LaserZexpr})
    {
        p->DefineConst("pi", PI);
    }

    VPot.DefineVar("x", &varX_);
    VPot.DefineVar("y", &varY_);
    VPot.DefineVar("z", &varZ_);
    // Normalize fabs(->abs( (muParser has abs but not fabs; they are identical
    // for real arguments). Apply to all spatial expressions before SetExpr so a
    // user writing fabs(x) behaves exactly like abs(x).
    auto norm_fabs = [](std::string s){ 
        size_t p=0; while((p=s.find("fabs(",p))!=std::string::npos){ s.replace(p,5,"abs("); p+=4; } return s; };
    VPot.SetExpr(norm_fabs(Potential));

    // Mass(x, y, z)
    MassExpr.DefineVar("x", &varX_);
    MassExpr.DefineVar("y", &varY_);
    MassExpr.DefineVar("z", &varZ_);
    MassExpr.SetExpr(norm_fabs(Mass));

    // dVdx along x
    dVPotX.DefineVar("x", &varX_);
    dVPotX.DefineVar("y", &varY_);
    dVPotX.DefineVar("z", &varZ_);
    dVPotX.SetExpr(norm_fabs(PotentialDerivativeX));

    // dVdx along y 
    dVPotY.DefineVar("x", &varX_);
    dVPotY.DefineVar("y", &varY_);
    dVPotY.DefineVar("z", &varZ_);
    dVPotY.SetExpr(norm_fabs(PotentialDerivativeY));

    // dVdx along z 
    dVPotZ.DefineVar("x", &varX_);
    dVPotZ.DefineVar("y", &varY_);
    dVPotZ.DefineVar("z", &varZ_);
    dVPotZ.SetExpr(norm_fabs(PotentialDerivativeZ));

    std::vector<std::pair<std::string, PetscReal>> scalarConsts;   // also bound on the FieldTerm parsers
    // Bind EVERY scalar input parameter as a muParser constant so users may
    // reference them by name inside expressions, e.g.
    //   Potential = V0*(x<0) : V0 ...  ;  Mass = mbarrier
    // This removes the earlier footgun where referencing an input scalar
    // (V0, BarrierWidth, ...) threw mu::ParserError. Any spatial/temporal
    // parser instance gets the full set. initParsers() runs immediately after
    // PrmReader(), so these are the final parsed values.
    {
        // Snapshot int params as doubles (no extra statics needed).
        const PetscReal dDim     = (PetscReal)Dimension;
        const PetscReal dDeg     = (PetscReal)SplineDegree;
        const PetscReal dNelm    = (PetscReal)Nelements;
        const PetscReal dNq      = (PetscReal)NQuadratures;
        const PetscReal dNbs     = (PetscReal)NBoundStates;
        struct KV { const char *name; PetscReal v; };
        KV scalars[] = {
            {"Q", Q}, {"Charge", Q},
            {"Dimension", dDim}, {"SplineDegree", dDeg},
            {"Nelements", dNelm}, {"NSplines", dNelm},
            {"LMinX", LMinX}, {"LMaxX", LMaxX},
            {"LMinY", LMinY}, {"LMaxY", LMaxY},
            {"LMinZ", LMinZ}, {"LMaxZ", LMaxZ},
            {"OffsetX", OffsetX}, {"OffsetY", OffsetY}, {"OffsetZ", OffsetZ},
            {"Ampx", Ampx}, {"Ampy", Ampy}, {"Ampz", Ampz},
            {"CEPx", CEPx}, {"CEPy", CEPy}, {"CEPz", CEPz},
            {"Omegax", Omegax}, {"Omegay", Omegay}, {"Omegaz", Omegaz},
            {"TargetEigenvalue", TargetEigenvalue},
            {"TimeStep", TimeStep}, {"FinalTime", FinalTime},
            {"CAPKmin", CAPKmin}, {"KnotAlpha", KnotAlpha},
            {"NQuadratures", dNq},
            {"NBoundStates", dNbs},
        };
        mu::Parser *parsers[] = {&VPot, &dVPotX, &dVPotY, &dVPotZ,
                           &MassExpr,
                           &LaserXexpr, &LaserYexpr, &LaserZexpr};
        for (auto *p : parsers) {
            for (auto &kv : scalars) {
                p->DefineConst(kv.name, kv.v);
            }
        }
        for (auto &kv : scalars) scalarConsts.emplace_back(kv.name, kv.v);
        // Bind user-defined constants (Variables key) on every parser instance
        // so they are referenceable inside any expression.
        for (auto *p : parsers) {
            for (auto &uc : userConsts_) {
                p->DefineConst(uc.first, uc.second);
            }
        }
    }

    // Scalar form: Laser = E(t) drives every axis that has no LaserX/Y/Z of its
    // own (documented behaviour; the per-axis expressions override it).
    for (std::string *ax : {&LaserX, &LaserY, &LaserZ})
        if (ax->empty() && !Laser.empty()) *ax = Laser;

    /* laser Ex(t) */
    LaserXexpr.DefineVar("t", &varT_);
    LaserXexpr.SetExpr(LaserX);

    /* laser Ey(t) */
    LaserYexpr.DefineVar("t", &varT_);
    LaserYexpr.SetExpr(LaserY);

    /* laser Ez(t) */
    LaserZexpr.DefineVar("t", &varT_);
    LaserZexpr.SetExpr(LaserZ);

    /* structured field terms: profiles w_k(x, y, z) and time factors f_k(t),
       with the same constants as every other expression */
    const size_t nTerms = FieldTermProfile.size();
    FieldProfileExpr.assign(nTerms, mu::Parser());
    FieldTimeExpr.assign(nTerms, mu::Parser());
    for (size_t k = 0; k < nTerms; ++k) {
        for (mu::Parser *p : {&FieldProfileExpr[k], &FieldTimeExpr[k]}) {
            p->DefineConst("pi", PI);
            for (const auto &kv : scalarConsts) p->DefineConst(kv.first, kv.second);
            for (const auto &uc : userConsts_) p->DefineConst(uc.first, uc.second);
        }
        FieldProfileExpr[k].DefineVar("x", &varX_);
        FieldProfileExpr[k].DefineVar("y", &varY_);
        FieldProfileExpr[k].DefineVar("z", &varZ_);
        FieldProfileExpr[k].SetExpr(norm_fabs(FieldTermProfile[k]));
        FieldTimeExpr[k].DefineVar("t", &varT_);
        FieldTimeExpr[k].SetExpr(norm_fabs(FieldTermTime[k]));
    }
}

/// @brief Trim leading and trailing whitespace from a string in place.
/// @param s  String to trim (modified in place).
void TDSEZParser::trim(std::string &s)
{
    auto isNotSpace = [](char c){ return !std::isspace(static_cast<unsigned char>(c)); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), isNotSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), isNotSpace).base(), s.end());
}








/// @brief Return the code version string for provenance tracking.
/// @return Version string (compiler, date, git hash).
std::string TDSEZParser::VersionString()
{
    // Not a git repo in this deployment -> static tag. If git is ever enabled,
    // shell out to `git describe --tags --dirty` here.
    return std::string("TDSEZ-1.0.0");
}





