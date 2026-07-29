/usr/bin/bash: /home/zakaria/miniconda3/envs/cudaq-env/lib/libtinfo.so.6: no version information available (required by /usr/bin/bash)
/usr/bin/bash: /home/zakaria/miniconda3/envs/cudaq-env/lib/libtinfo.so.6: no version information available (required by /usr/bin/bash)
#include "tdsez_internal.hpp"

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
#include "debug.hpp"

/* \brief Build a handler that parses "a,b,c" (per-axis list) into 3 refs.
   Single token replicates to all axes; missing tokens fall back to the
   first token (matches Domain / KnotSequence convention). */
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

/* \brief Read key=value file, stripping comments. */
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

/* \\brief Initialize muParser instances and bind variables. */
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

/* \brief Trim whitespace from both ends. */
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


/* \brief Validate the parsed parameter set. Throws std::runtime_error on any
   physically inconsistent / dangerous combination. Called once after parsing. */
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
            if (num.empty())
                throw std::runtime_error(
                    "ValidateOrThrow: InitialState='" + s + "' missing state index (use 'state:N')");
            PetscInt idx;
            try { idx = (PetscInt)std::stoll(num); }
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
                try { coeff = (PetscReal)std::stod(cstr); }
                catch (...) {
                    throw std::runtime_error(
                        "ValidateOrThrow: InitialState term '" + t + "' has a non-numeric coefficient");
                }
                try { idx = (PetscInt)std::stoll(istr); }
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

    // --- mass positivity: sample on a coarse grid. A non-positive mass makes the
    //     kinetic term indefinite -> garbage / solver breakdown. Mass may be a
    //     piecewise expression (DQW material step), so we sample, not assume.
    {
        const PetscInt nsamp = 16;
        auto sampleMass = [&](PetscReal lo, PetscReal hi) {
            for (PetscInt i = 0; i < nsamp; ++i) {
                PetscReal x = lo + (hi - lo) * (PetscReal)i / (nsamp - 1);
                PetscReal m = 1.0;
                try { m = MassDist(x, 0.0, 0.0); } catch (...) { return; }
                if (!std::isfinite(m) || m <= 0.0)
                    throw std::runtime_error(
                        "ValidateOrThrow: Mass must be > 0 and finite everywhere (got "
                        + to_string(m) + " at x=" + to_string(x) + ")");
            }
        };
        sampleMass(LMinX, LMaxX);
        if (Dimension >= 2) sampleMass(LMinY, LMaxY);
        if (Dimension >= 3) sampleMass(LMinZ, LMaxZ);
    }

    // --- potential finiteness: a NaN/Inf potential poisons the assembly.
    {
        const PetscInt nsamp = 16;
        auto sampleV = [&](PetscReal lo, PetscReal hi) {
            for (PetscInt i = 0; i < nsamp; ++i) {
                PetscReal x = lo + (hi - lo) * (PetscReal)i / (nsamp - 1);
                PetscReal v = 0.0;
                try { v = V(x); } catch (...) { return; }
                if (!std::isfinite(v))
                    throw std::runtime_error(
                        "ValidateOrThrow: Potential is not finite at x=" + to_string(x));
            }
        };
        sampleV(LMinX, LMaxX);
        if (Dimension >= 2) sampleV(LMinY, LMaxY);
        if (Dimension >= 3) sampleV(LMinZ, LMaxZ);
    }

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

    // The "adaptive" and "adaptive_wf" KnotSequences need the analytic potential
    // derivative to place knots (TDSEZAdaptiveKnots evaluates dVPotX/Y/Z). An
    // empty PotentialDerivativeX/Y/Z makes muParser throw on an empty expression
    // deep inside knot generation (after assembly starts) -> ugly std::terminate.
    // Catch it here with a clean, actionable FATAL. adaptive_wf needs it too
    // because its coarse bootstrap mesh is built with TDSEZAdaptiveKnots.
    {
        auto needs = [&](const std::string &seq) {
            return seq == "adaptive" || seq == "adaptive_wf";
        };
        if (needs(KnotSeq[0]) && PotentialDerivativeX.empty())
            throw std::runtime_error(
                "ValidateOrThrow: KnotSequence='" + KnotSeq[0] + "' (X axis) requires "
                "PotentialDerivativeX to be set (it is empty). Provide the analytic "
                "derivative of the potential, e.g. PotentialDerivativeX = -2.0*x.");
        if (Dimension >= 2 && needs(KnotSeq[1]) && PotentialDerivativeY.empty())
            throw std::runtime_error(
                "ValidateOrThrow: KnotSequence='" + KnotSeq[1] + "' (Y axis) requires "
                "PotentialDerivativeY to be set (it is empty). Provide the analytic "
                "derivative of the potential, e.g. PotentialDerivativeY = -2.0*y.");
        if (Dimension >= 3 && needs(KnotSeq[2]) && PotentialDerivativeZ.empty())
            throw std::runtime_error(
                "ValidateOrThrow: KnotSequence='" + KnotSeq[2] + "' (Z axis) requires "
                "PotentialDerivativeZ to be set (it is empty). Provide the analytic "
                "derivative of the potential, e.g. PotentialDerivativeZ = -2.0*z.");
    }
}


/* \brief Code version string for provenance. */
std::string TDSEZParser::VersionString()
{
    // Not a git repo in this deployment -> static tag. If git is ever enabled,
    // shell out to `git describe --tags --dirty` here.
    return std::string("TDSEZ-1.0.0");
}






