#include "tdsez_parser.hpp"

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
#include <set>
#include <unordered_map>
#include <petscblaslapack.h>

#include "muParser.h"

/**
 * @file parser.cpp
 * @brief Read input files and initialize muParser expressions.
 * @author TDSEZ Project
 */

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
            v.push_back(std::stod(tok));
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
void TDSEZParser::PrmReader(const std::string &prm)
{
    std::ifstream in(prm);
    if (!in) throw std::runtime_error("Cannot open file: " + prm);
    std::string line;
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
            static const auto I  = [](auto &dst){ return H{[&dst](const std::string &v){ dst = std::stoi(v); }}; };
            static const auto R  = [](auto &dst){ return H{[&dst](const std::string &v){ dst = std::stod(v); }}; };
            static const auto B  = [](auto &dst){ return H{[&dst](const std::string &v){ dst = static_cast<PetscBool>(std::stoi(v)); }}; };
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
                            if (c != std::string::npos) { lo = std::stod(p.substr(0, c)); hi = std::stod(p.substr(c + 1)); } };
                        if (pairs.size() >= 1) parsePair(pairs[0], LMinX, LMaxX);
                        if (pairs.size() >= 2) parsePair(pairs[1], LMinY, LMaxY); else { LMinY = LMinX; LMaxY = LMaxX; }
                        if (pairs.size() >= 3) parsePair(pairs[2], LMinZ, LMaxZ); else { LMinZ = LMinX; LMaxZ = LMaxX; }
                        LMin = LMinX; LMax = LMaxX;
                    } else {
                        auto commaPos = s.find(',');
                        if (commaPos != std::string::npos) {
                            LMin = std::stod(s.substr(0, commaPos)); LMax = std::stod(s.substr(commaPos + 1));
                            LMinX = LMinY = LMinZ = LMin; LMaxX = LMaxY = LMaxZ = LMax;
                        }
                    }
                }}},
                {"LMin",   R(LMin)},  {"LMax",   R(LMax)},
                {"LMinX",  R(LMinX)}, {"LMinY",  R(LMinY)}, {"LMinZ",  R(LMinZ)},
                {"LMaxX",  R(LMaxX)}, {"LMaxY",  R(LMaxY)}, {"LMaxZ",  R(LMaxZ)},
                {"OffsetX",R(OffsetX)},{"OffsetY",R(OffsetY)},{"OffsetZ",R(OffsetZ)},
                {"NQuadratures", I(NQuadratures)},
                {"Gamma",  R(Gamma)},
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
                {"HydrogenicNLin", H{[](const std::string &val){
                    std::string s = val; s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                    size_t p = 0, n = 0;
                    while (p < s.size() && n < 3) {
                        size_t c = s.find(',', p);
                        std::string tok = (c == std::string::npos) ? s.substr(p) : s.substr(p, c - p);
                        HydrogenicNLin[n++] = std::stoi(tok); if (c == std::string::npos) break; p = c + 1;
                    }
                    if (n == 1) for (int i = 1; i < 3; ++i) HydrogenicNLin[i] = HydrogenicNLin[0];
                }}},
                {"HydrogenicNExp", H{[](const std::string &val){
                    std::string s = val; s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                    size_t p = 0, n = 0;
                    while (p < s.size() && n < 3) {
                        size_t c = s.find(',', p);
                        std::string tok = (c == std::string::npos) ? s.substr(p) : s.substr(p, c - p);
                        HydrogenicNExp[n++] = std::stoi(tok); if (c == std::string::npos) break; p = c + 1;
                    }
                    if (n == 1) for (int i = 1; i < 3; ++i) HydrogenicNExp[i] = HydrogenicNExp[0];
                }}},
                {"HydrogenicR1", H{[](const std::string &val){
                    std::string s = val; s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                    size_t p = 0, n = 0;
                    while (p < s.size() && n < 3) {
                        size_t c = s.find(',', p);
                        std::string tok = (c == std::string::npos) ? s.substr(p) : s.substr(p, c - p);
                        HydrogenicR1[n++] = std::stod(tok); if (c == std::string::npos) break; p = c + 1;
                    }
                    if (n == 1) for (int i = 1; i < 3; ++i) HydrogenicR1[i] = HydrogenicR1[0];
                }}},
                // Adaptive (KnotSequence = adaptive) tuning. Single value applies
                // to all axes; no per-axis form needed (weight is auto-scaled).
                {"AdaptiveKappa", R(AdaptiveKappa)},
                {"AdaptivePower", R(AdaptivePower)},
                {"AdaptiveWFCoarseN", I(AdaptiveWFCoarseN)},
                {"AdaptiveWFKinLambda", R(AdaptiveWFKinLambda)},
                {"Mass",     S(Mass)},
                {"dinvMassX",S(dinvMassX)}, {"dinvMassY",S(dinvMassY)}, {"dinvMassZ",S(dinvMassZ)},
                {"Planck",    R(Hbar)},       // backward-compat alias
                {"Hbar",      R(Hbar)},
                {"Charge",    R(Q)},
                {"ParticleCharge", R(Q)},     // intuitive alias
                {"MassIsConstant", B(MassIsConstant)},

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
                {"Amplitude",     R(Amplitude)},
                {"Phase",         R(Phase)},
                {"Omega",         R(Omega)},
                {"PulseDuration", R(PulseDuration)},
                {"PulseCenter",   R(PulseCenter)},
                {"EnvelopeType",  S(EnvelopeType)},
                {"Envelope",      S(Envelope)},    // muParser envelope expression (was a dead branch)

                // --- field (per-component generalization) ---
                {"Ampx",R(Ampx)},{"Ampy",R(Ampy)},{"Ampz",R(Ampz)},
                {"CEPx",R(CEPx)},{"CEPy",R(CEPy)},{"CEPz",R(CEPz)},
                {"Omegax",R(Omegax)},{"Omegay",R(Omegay)},{"Omegaz",R(Omegaz)},
                {"LaserX",S(LaserX)},{"LaserY",S(LaserY)},{"LaserZ",S(LaserZ)},
                {"DriverX",S(LaserX)},{"DriverY",S(LaserY)},{"DriverZ",S(LaserZ)}, // driver alias
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
                {"OutputStrideTS",     I(OutputStrideTS)},
                {"OutputStrideAC",     I(OutputStrideAC)},
                {"HDF5Compress",       B(HDF5Compress)},
                {"HDF5CompressLevel",  I(HDF5CompressLevel)},
                {"CAPKmin",    R(CAPKmin)},
                {"CapStrength", R(CAPKmin)},           // intuitive alias
                {"KnotAlpha",  R(KnotAlpha)},
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

                // --- t-SURFF ---
                {"TSurff",            B(out_tsurff)},
                {"SurffNk",           I(SurffNk)},
                {"SurffKmax",         R(SurffKmax)},
                {"OutputStrideSurff", I(OutputStrideSurff)},
                {"SurffCouplingSign", R(SurffCouplingSign)},
                {"SurffRadius",       R(SurffRadius)},

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

        auto it = table().find(key);
        if (it != table().end()) it->second(val);
        else if (TDSEZParser::StrictInput) {
            // A typo'd key silently changing nothing is a real hazard in a
            // physics code (wrong potential, wrong mass, etc.). With StrictInput
            // (default true) we abort hard so the user sees the mistake.
            PetscPrintf(PETSC_COMM_WORLD,
                "\nTDSEZParser FATAL: unknown input key '%s'.\n"
                "  Set 'StrictInput = 0' in the input file to tolerate unknown keys.\n",
                key.c_str());
            MPI_Abort(PETSC_COMM_WORLD, 2);
        }
        else {
            PetscPrintf(PETSC_COMM_WORLD,
                "TDSEZParser: WARNING unknown key '%s' ignored\n", key.c_str());
        }
    }

    // Decode PhysicsOutput into per-quantity enable flags.
    // Default ("all" or empty) keeps everything enabled (backward compatible).
    // Otherwise only the listed tokens are enabled; the rest are turned off so
    // their per-step MatMults / reductions / HDF5 writes are skipped.
    {
        std::string s = PhysicsOutput;
        for (auto& c : s) c = (char)std::tolower((unsigned char)c);
        s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
        if (!(s.empty() || s == "all")) {
            out_dipole = out_population = out_energy = out_current = out_autocorr = out_wfs = out_tsurff = PETSC_FALSE;
            std::istringstream iss(s);
            std::string tok;
            while (std::getline(iss, tok, ',')) {
                if      (tok == "dipole")         out_dipole    = PETSC_TRUE;
                else if (tok == "population")      out_population = PETSC_TRUE;
                else if (tok == "energy")         out_energy    = PETSC_TRUE;
                else if (tok == "current")        out_current   = PETSC_TRUE;
                else if (tok == "autocorrelation" || tok == "ac") out_autocorr = PETSC_TRUE;
                else if (tok == "wfs")            out_wfs       = PETSC_TRUE;
                else if (tok == "tsurff")         out_tsurff    = PETSC_TRUE;
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
        lowerInPlace(EnvelopeType);
        if (BoundaryType == "natural") BoundaryType = "neumann";
        else if (BoundaryType == "wall")    BoundaryType = "dirichlet";
    }
}

/// @brief Initialize muParser instances and bind parser variables (V,
///        MassDist, Ex/Ey/Ez, Envelope, and their derivatives) to the
///        corresponding TDSEZParser static member expressions.
void TDSEZParser::initParsers()
{
    /* bind constant pi */
    for (auto *p : {&VPot, &dVPotX, &dVPotY, &dVPotZ, &Field, &LaserXexpr, &LaserYexpr, &LaserZexpr, &Env})
    {
        p->DefineConst("pi", PI);
        p->DefineConst("Amplitude", Amplitude);
        p->DefineConst("Phase", Phase);
        p->DefineConst("Omega", Omega);

        // generalization
        p->DefineConst("Ampx", Ampx);
        p->DefineConst("Ampy", Ampy);
        p->DefineConst("Ampz", Ampz);
        p->DefineConst("CEPx", CEPx);
        p->DefineConst("CEPy", CEPy);
        p->DefineConst("CEPz", CEPz);
        p->DefineConst("Omegax", Omegax);
        p->DefineConst("Omegay", Omegay);
        p->DefineConst("Omegaz", Omegaz);
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

    // dMass/dx
    dinvMassExprx.DefineVar("x", &varX_);
    dinvMassExprx.DefineVar("y", &varY_);
    dinvMassExprx.DefineVar("z", &varZ_);
    dinvMassExprx.SetExpr(norm_fabs(dinvMassX));

    // dMass/dy
    dinvMassExpry.DefineVar("x", &varX_);
    dinvMassExpry.DefineVar("y", &varY_);
    dinvMassExpry.DefineVar("z", &varZ_);
    dinvMassExpry.SetExpr(norm_fabs(dinvMassY));

    // dMass/dz
    dinvMassExprz.DefineVar("x", &varX_);
    dinvMassExprz.DefineVar("y", &varY_);
    dinvMassExprz.DefineVar("z", &varZ_);
    dinvMassExprz.SetExpr(norm_fabs(dinvMassZ));

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

    /* field E(t): in all three dirs */
    Field.DefineVar("t", &varT_);
    Field.SetExpr(Laser);

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
            {"Hbar", Hbar}, {"Q", Q}, {"Planck", Hbar}, {"Charge", Q},
            {"Dimension", dDim}, {"SplineDegree", dDeg},
            {"Nelements", dNelm}, {"NSplines", dNelm},
            {"LMin", LMin}, {"LMax", LMax},
            {"LMinX", LMinX}, {"LMaxX", LMaxX},
            {"LMinY", LMinY}, {"LMaxY", LMaxY},
            {"LMinZ", LMinZ}, {"LMaxZ", LMaxZ},
            {"OffsetX", OffsetX}, {"OffsetY", OffsetY}, {"OffsetZ", OffsetZ},
            {"Amplitude", Amplitude}, {"Omega", Omega}, {"Phase", Phase},
            {"PulseDuration", PulseDuration}, {"PulseCenter", PulseCenter},
            {"Ampx", Ampx}, {"Ampy", Ampy}, {"Ampz", Ampz},
            {"CEPx", CEPx}, {"CEPy", CEPy}, {"CEPz", CEPz},
            {"Omegax", Omegax}, {"Omegay", Omegay}, {"Omegaz", Omegaz},
            {"TargetEigenvalue", TargetEigenvalue},
            {"TimeStep", TimeStep}, {"FinalTime", FinalTime},
            {"CAPKmin", CAPKmin}, {"KnotAlpha", KnotAlpha},
            {"NQuadratures", dNq},
            {"NBoundStates", dNbs},
            {"Gamma", Gamma},
        };
        mu::Parser *parsers[] = {&VPot, &dVPotX, &dVPotY, &dVPotZ,
                           &MassExpr, &dinvMassExprx, &dinvMassExpry, &dinvMassExprz,
                           &Field, &LaserXexpr, &LaserYexpr, &LaserZexpr, &Env};
        for (auto *p : parsers) {
            for (auto &kv : scalars) {
                p->DefineConst(kv.name, kv.v);
            }
        }
        // Bind user-defined constants (Variables key) on every parser instance
        // so they are referenceable inside any expression.
        for (auto *p : parsers) {
            for (auto &uc : userConsts_) {
                p->DefineConst(uc.first, uc.second);
            }
        }
    }

    /* laser Ex(t) */
    LaserXexpr.DefineVar("t", &varT_);
    LaserXexpr.SetExpr(LaserX);

    /* laser Ey(t) */
    LaserYexpr.DefineVar("t", &varT_);
    LaserYexpr.SetExpr(LaserY);

    /* laser Ez(t) */
    LaserZexpr.DefineVar("t", &varT_);
    LaserZexpr.SetExpr(LaserZ);

    /* envelope f(t) */
    Env.DefineVar("t", &varT_);
    Env.SetExpr(Envelope);
}

/// @brief Trim leading and trailing whitespace from a string in place.
/// @param s  String to trim (modified in place).
void TDSEZParser::trim(std::string &s)
{
    auto isNotSpace = [](char c){ return !std::isspace(static_cast<unsigned char>(c)); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), isNotSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), isNotSpace).base(), s.end());
}






// Numerical gradient calculation 
inline std::vector<PetscReal> gradientND(const std::vector<PetscReal>& x)
{
    PetscInt dim = x.size();
    std::vector<PetscReal> grad(dim);
    PetscReal h = 1e-5;

    for (PetscInt i=0; i<dim; ++i) 
    {
        std::vector<PetscReal> x_plus = x;
        std::vector<PetscReal> x_minus = x;
        std::vector<PetscReal> x_plus2 = x;
        std::vector<PetscReal> x_minus2 = x;

        // move along i-th dimension
        x_plus[i]  += h;
        x_minus[i] -= h;
        x_plus2[i] += 2*h;
        x_minus2[i] -= 2*h;

        // evaluate function
        PetscReal Vp  = TDSEZParser::V(x_plus);
        PetscReal Vm  = TDSEZParser::V(x_minus);
        PetscReal Vpp = TDSEZParser::V(x_plus2);
        PetscReal Vmm = TDSEZParser::V(x_minus2);

        grad[i] = (-Vpp + 8*Vp - 8*Vm + Vmm)/(12*h); // 4th-order
    }
    return grad;
}


/// @brief Return the code version string for provenance tracking.
/// @return Version string (compiler, date, git hash).
std::string TDSEZParser::VersionString()
{
    // Not a git repo in this deployment -> static tag. If git is ever enabled,
    // shell out to `git describe --tags --dirty` here.
    return std::string("TDSEZ-1.0.0");
}





