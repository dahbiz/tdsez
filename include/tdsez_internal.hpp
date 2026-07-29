#ifndef __TDSEZ_INTERNAL_HPP__
#define __TDSEZ_INTERNAL_HPP__

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
#include <cstdio>
#include <filesystem>

// ── TDSE-Z output-formatting helpers (quality bar) ────────────────────────
// fmtSci2 : 2-significant-digit scientific notation, used ONLY for eV.
// fmtAu   : fixed %+02.5e a.u. (NEVER fmtSci2) per quality bar.
// fmtEv   : eV NUMBER via fmtSci2 (2 sig digits); caller appends " eV".
static inline std::string TDSEZ_fmtSci2(double x) {
    char b[32];
    snprintf(b, sizeof(b), "%.2e", x);
    return std::string(b);
}
static inline std::string TDSEZ_fmtAu(double x) {
    char b[48];
    snprintf(b, sizeof(b), "%+02.5e", x);
    return std::string(b);
}
static inline std::string TDSEZ_fmtEv(double au) {
    return TDSEZ_fmtSci2(au * 27.211386245988);
}


// ============================================================================
//  TDSEZInfo  —  centralised terminal-log printer
//  ----------------------------------------------------------------------------
//  Every ═══ rule, ▸ section, KV row, banner, and footer goes through this
//  class so that bar width, indentation, and centering are defined in exactly
//  one place.  To change the box width, edit W below; every call site updates
//  automatically.
//
//  Layout constants
//    INDENT  = 2 spaces before every line
//    W       = 70  (inner content width; rule = INDENT + W ═ chars)
//    Rule    = "  " + 70×═
//
//  Usage
//    TDSEZInfo info;          // default: PETSC_COMM_WORLD
//    info.banner(logoLines);  // print logo + title + credit
//    info.header("EIGENSOLVER SETUP");
//    info.section("PROBLEM");
//    info.kv("DOFs", "%d", n);
//    info.blank();
//    info.footer();
// ============================================================================
class TDSEZInfo
{
public:
    static constexpr int INDENT = 2;
    static constexpr int W      = 70;   // inner width (rule = INDENT + W ═)

    explicit TDSEZInfo(MPI_Comm comm = PETSC_COMM_WORLD)
        : m_comm(comm) {}

    // ── heavy rule: "  " + W×═ + "\n" ──────────────────────────────────
    void rule() const
    {
        PetscPrintf(m_comm, "%*s", INDENT, "");
        for (int i = 0; i < W; ++i) PetscPrintf(m_comm, "═");
        PetscPrintf(m_comm, "\n");
    }

    // ── centered header between two rules ──────────────────────────────
    void header(const char* title) const
    {
        PetscPrintf(m_comm, "\n");
        rule();
        center(title);
        rule();
    }

    // ── ▸ section label + trailing rule (no leading rule) ──────────────
    void section(const char* label) const
    {
        int llen = (int)strlen(label) + 4;   // "  ▸ " prefix
        int pad  = W - llen;
        if (pad < 0) pad = 0;
        PetscPrintf(m_comm, "%*s  ▸ %s%*s\n", INDENT, "", label, pad, "");
        rule();
    }

    // ── KV row: left-aligned label, right-aligned value ────────────────
    //    "  " + 4-space inner indent + label + gap + value + 2 trailing
    void kv(const char* label, const char* fmt, ...) const
    {
        char val[256];
        va_list args;
        va_start(args, fmt);
        vsnprintf(val, sizeof(val), fmt, args);
        va_end(args);

        int llen = (int)strlen(label);
        int vlen = (int)strlen(val);
        int gap  = W - 4 - llen - vlen;
        if (gap < 1) gap = 1;
        PetscPrintf(m_comm, "%*s    %-*s%*s  \n",
                     INDENT, "", llen, label, gap + vlen, val);
    }

    // ── blank line matching box width ─────────────────────────────────
    void blank() const
    {
        PetscPrintf(m_comm, "%*s", INDENT, "");
        for (int i = 0; i < W; ++i) PetscPrintf(m_comm, " ");
        PetscPrintf(m_comm, "\n");
    }

    // ── footer: single rule + blank line ────────────────────────────────
    void footer() const
    {
        rule();
        PetscPrintf(m_comm, "\n");
    }

    // ── status row: icon + message, padded to box width ────────────────
    void status(const char* icon, const char* fmt, ...) const
    {
        char msg[256];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);

        int mlen = (int)strlen(icon) + 1 + (int)strlen(msg);
        int pad  = W - 4 - mlen;
        if (pad < 0) pad = 0;
        PetscPrintf(m_comm, "%*s    %s %s%*s\n",
                     INDENT, "", icon, msg, pad, "");
    }

    // ── memory KV row ──────────────────────────────────────────────────
    void memKV(const char* label) const
    {
        PetscLogDouble mem_curr, mem_max;
        PetscMemoryGetCurrentUsage(&mem_curr);
        PetscMemoryGetMaximumUsage(&mem_max);
        char val[128];
        snprintf(val, sizeof(val), "%.3e MB  (peak %.3e MB)",
                 mem_curr / 1e6, mem_max / 1e6);
        kv(label, "%s", val);
    }

    // ── banner: logo lines (centered) + text lines (each centered) ──────
    //  logoLines: vector of pre-stripped lines (no trailing \n)
    //  textLines: title + credit lines, each centered one under the other
    void banner(const std::vector<std::string>& logoLines,
                const std::vector<std::string>& textLines) const
    {
        rule();
        for (const auto& line : logoLines)
            centerStr(line.c_str(), displayWidth(line));
        PetscPrintf(m_comm, "\n");
        for (const auto& line : textLines)
            centerStr(line.c_str(), (int)line.size());
        PetscPrintf(m_comm, "\n");
    }

    // ── plain printf (for lines that don't fit the box model) ──────────
    void raw(const char* fmt, ...) const
    {
        va_list args;
        va_start(args, fmt);
        va_end(args);
        // PetscVPrintf doesn't exist; format into a buffer and use PetscPrintf
        char buf[1024];
        va_list args2;
        va_start(args2, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args2);
        va_end(args2);
        PetscPrintf(m_comm, "%s", buf);
    }

    // ── center a string in the W-wide field (public for custom blocks) ─
    void center(const char* s) const { centerStr(s, (int)strlen(s)); }

private:
    MPI_Comm m_comm;

    void centerStr(const char* s, int dw) const
    {
        int pad = (dw > 0) ? (W - dw) / 2 : 0;
        if (pad < 0) pad = 0;
        PetscPrintf(m_comm, "%*s%*s%s\n", INDENT, "", pad, "", s);
    }

    // ── UTF-8 display width (code-point count; assumes every glyph = 1 col)
    static int displayWidth(const std::string& s)
    {
        int dw = 0;
        for (size_t i = 0; i < s.size(); ) {
            unsigned char c = (unsigned char)s[i];
            if      (c < 0x80) i += 1;
            else if (c < 0xC0) i += 1;     // continuation byte
            else if (c < 0xE0) i += 2;
            else if (c < 0xF0) i += 3;
            else               i += 4;
            ++dw;
        }
        return dw;
    }
};


#include <petscviewerhdf5.h>
#include <H5Epublic.h>
#include <petscmat.h>
#include <algorithm>
#include <functional>
#include <petscblaslapack.h>

#include "muParser.h"
#include "debug.hpp"

#if defined(__GNUC__) || defined(__clang__)
  #define TDSE_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
  #define TDSE_ALWAYS_INLINE __forceinline
#else
  #define TDSE_ALWAYS_INLINE inline
#endif

// ----------------------------------------------------------------------------
//  Forward declarations
// ----------------------------------------------------------------------------
class TDSEZCore;
class TDSEZAssembler;
class TDSEZPropagator;
class TDSEZManager;

// TS callback forward declarations (used by the propagator)
PetscErrorCode TDSEZIFunctionPolX   (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolY   (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolZ   (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolXY  (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolXZ  (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolYZ  (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolXYZ (TS, PetscReal, Vec, Vec, Vec, void*);

PetscErrorCode TDSEZIJacobianPolX   (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolY   (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolZ   (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolXY  (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolXZ  (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolYZ  (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolXYZ (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);

// ============================================================================
//  TDSEZParser  — input / parameter parser (declaration region verbatim)
// ============================================================================
class TDSEZParser
{
    public:

        TDSEZParser(const std::string &prm)
        {
            PrmReader(prm);

            /* Setup spatial and time grids */
            TotalTimeSteps = static_cast<PetscInt>(std::ceil(FinalTime / TimeStep));

            initParsers();
        }


        // Read parameters from file
        static void PrmReader(const std::string &prm);
        static void initParsers();

        /* Input validation: throws std::runtime_error on any physically
           inconsistent / dangerous parameter combination. Called once after
           the parser has finished reading the input file. */
        static void ValidateOrThrow();

        /* Short code version string (git describe when available, else a
           static tag). Stored in HDF5 provenance for reproducibility. */
        static std::string VersionString();

        // accessors
        template<typename... Args>
        inline static PetscReal V(Args... args)     { setVars(args...) ; return VPot.Eval(); }
        template<typename... Args>
        inline static PetscReal dV(Args... args)    { setVars(args...); return dVPotX.Eval(); }
        template<typename... Args>
        inline static PetscReal dVx(Args... args)      { setVars(args...); return dVPotX.Eval(); }
        template<typename... Args>
        inline static PetscReal dVy(Args... args)      { setVars(args...); return dVPotY.Eval(); }
        template<typename... Args>
        inline static PetscReal dVz(Args... args)      { setVars(args...); return dVPotZ.Eval(); }
        inline static PetscReal E(PetscReal t)      { varT_ = t; return Field.Eval(); }
        inline static PetscReal Ex(PetscReal t)      { varT_ = t; return LaserXexpr.Eval(); }
        inline static PetscReal Ey(PetscReal t)      { varT_ = t; return LaserYexpr.Eval(); }
        inline static PetscReal Ez(PetscReal t)      { varT_ = t; return LaserZexpr.Eval(); }
        inline static PetscReal F(PetscReal t)      { varT_ = t; return Env.Eval(); }
        inline static PetscReal MassDist(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return MassExpr.Eval(); }
        inline static PetscReal dinvMassDistx(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return dinvMassExprx.Eval(); }
        inline static PetscReal dinvMassDisty(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return dinvMassExpry.Eval(); }
        inline static PetscReal dinvMassDistz(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return dinvMassExprz.Eval(); }

        // private:
        static constexpr PetscReal PI     = 3.14159265358979323846;
        inline static std::string Mass    = "1.0";
        inline static std::string dinvMassX  = "0.0";
        inline static std::string dinvMassY  = "0.0";
        inline static std::string dinvMassZ  = "0.0";
        inline static PetscBool MassIsConstant = PETSC_FALSE;


        // inline static PetscReal Mass      = 1.0;
        inline static PetscReal Hbar      = 1.0;
        inline static PetscReal Q         = 1.0;

        /* Spatial grid */
        inline static PetscBool Verbose          = PETSC_TRUE;
        // When PETSC_TRUE (default) an unknown input key is a FATAL error
        // instead of a silent warning — prevents typo'd keys from silently
        // no-op'ing in a physics code. Set StrictInput = 0 to tolerate.
        inline static PetscBool StrictInput      = PETSC_TRUE;
        inline static PetscReal Dimension        = 1; // 1, 2, or 3
        inline static PetscInt  SplineDegree     = 3; // 1 to 14
        inline static PetscInt  Nelements         = 100;
        inline static PetscReal LMin             = -10.0;
        inline static PetscReal LMax             = +10.0;
        // Per-axis bounds for rectangular/asymmetric boxes. LMin*/LMax* default to
        // LMin/LMax when unset, so the symmetric (cube/parallelepiped) case needs
        // only LMin/LMax. Domain = [Lminx,Lmaxx],[Lminy,Lmaxy],[Lminz,Lmaxz] sets these.
        inline static PetscReal LMinX            = -10.0;
        inline static PetscReal LMinY            = -10.0;
        inline static PetscReal LMinZ            = -10.0;
        inline static PetscReal LMaxX            = +10.0;
        inline static PetscReal LMaxY            = +10.0;
        inline static PetscReal LMaxZ            = +10.0;
        inline static PetscReal OffsetX          = 0.0;
        inline static PetscReal OffsetY          = 0.0;
        inline static PetscReal OffsetZ          = 0.0;

        /* Solver grid */
        inline static PetscBool UseDirectSolve    = PETSC_FALSE;
        inline static PetscReal TargetEigenvalue  = 0.0;

        /* Temporal grid */
        inline static  PetscReal TimeStep         = 0.01;
        inline static PetscReal FinalTime         = 100.0;
        inline static PetscInt  TotalTimeSteps    = 0.0;

        /* Field and envelope */
        inline static PetscReal Amplitude         = 0.0;
        inline static PetscReal Omega             = 0.056954; // 800 nm
        inline static PetscReal Phase             = 0.0;
        inline static PetscReal PulseDuration     = 0.0;
        inline static PetscReal PulseCenter       = 0.0;

        // generalization
        inline static PetscReal Ampx              = 0.0;
        inline static PetscReal Ampy              = 0.0;
        inline static PetscReal Ampz              = 0.0;
        inline static PetscReal CEPx              = 0.0;
        inline static PetscReal CEPy              = 0.0;
        inline static PetscReal CEPz              = 0.0;
        inline static PetscReal Omegax            = 0.0;
        inline static PetscReal Omegay            = 0.0;
        inline static PetscReal Omegaz            = 0.0;

        // laser expressions
        inline static std::string LaserX;
        inline static std::string LaserY;
        inline static std::string LaserZ;

        inline static PetscBool             EnablePropagation  = PETSC_TRUE;
        inline static PetscBool             EnableGPU          = PETSC_FALSE;
        inline static PetscBool             EnableCAP          = PETSC_FALSE;
        inline static PetscBool             EnableLzDiag      = PETSC_FALSE;
        inline static PetscReal             CAPKmin           = 0.1;
        inline static PetscInt              OutputStrideWFS       = 100;
        inline static PetscInt              OutputStrideTS       = 0;
        inline static PetscInt              OutputStrideAC       = 0;
        inline static PetscBool             HDF5Compress         = PETSC_FALSE;
        inline static PetscInt              HDF5CompressLevel    = 6;
        inline static PetscInt              NBoundStates       = 1;
        inline static PetscBool             NBoundStatesSave   = PETSC_FALSE;
        // Format of the bound states written to EigenData_*.h5:
        //   "complex" -> raw SLEPc eigenvectors (may carry an arbitrary global phase)
        //   "real"    -> each state phase-rotated to a real canonical form
        //                (TDSEZMakeStateReal2) before saving
        inline static std::string           BoundStateFormat   = "complex";
        // Initial state for propagation / autocorrelation baseline.
        //   "ground"                       -> 0th converged bound state (default)
        //   "state:N"                      -> Nth converged bound state (0-based)
        //   "sup: a*N + b*M [+ c*P]"       -> coherent superposition of up to 3
        //                                     bound states (indices N,M,P; real coeffs)
        inline static std::string           InitialState       = "ground";
        // Parsed in ValidateOrThrow:
        //   0 = ground (== state:0), 1 = single state:InitialStateIndex,
        //   2 = superposition over InitialStateTerms.
        inline static PetscInt              InitialStateMode   = 0;
        inline static PetscInt              InitialStateIndex  = 0;
        // (index, coefficient) pairs for mode 2 (max 3 entries).
        inline static std::vector<std::pair<PetscInt,PetscReal>> InitialStateTerms;
        // Normalize the assembled initial state so <psi|M|psi> = 1 (default on).
        inline static PetscBool             NormalizeInitialState = PETSC_TRUE;
        inline static std::string           PhysicsOutput      = "all";  // declarative enable list: dipole,population,energy,current,autocorrelation,wfs (or "all")
        inline static PetscBool             out_dipole         = PETSC_TRUE;
        inline static PetscBool             out_population     = PETSC_TRUE;
        inline static PetscBool             out_energy         = PETSC_TRUE;
        inline static PetscBool             out_current        = PETSC_TRUE;
        inline static PetscBool             out_autocorr       = PETSC_TRUE;
        inline static PetscBool             out_wfs            = PETSC_TRUE;
        // Save the assembled dipole operator matrix Dx to a PETSc binary file
        // (MatView -> *.bin, reloadable with MatLoad). Off by default. At least
        // for 1D for now; Dx is only non-NULL when the polarisation includes x.
        inline static PetscBool             SaveDipoleMatrix   = PETSC_FALSE;
        // Which dipole operator axes to persist (PETSc binary, one file each).
        // Empty by default: when SaveDipoleMatrix=1 the file-level fallback
        // saves Dx only (1D-friendly). Set from the command line via
        //   -save_dipole x | xy | xz | xyz | all | none
        // to override at runtime. Only axes that were actually assembled
        // (polarisation includes them) are written.
        inline static std::string           SaveDipoleAxes     = "";
        // t-SURFF (time-dependent surface-flux photoelectron momentum spectrum)
        inline static PetscBool             out_tsurff         = PETSC_FALSE;
        inline static PetscInt              SurffNk            = 40;   // momentum grid points per axis in [-SurffKmax,SurffKmax]
        inline static PetscReal             SurffKmax          = 2.0;  // momentum cutoff (a.u.)
        inline static PetscInt              OutputStrideSurff  = 1;    // fold boundary flux into b(k) every N accepted steps (1 = exact)
        inline static PetscReal             SurffCouplingSign  = +1.0; // k_eff = k + s*q*A(t); flip to -1.0 if your field couples as +q*E*r
        inline static PetscReal             SurffRadius        = 0.0;  // if >0, restrict t-SURFF faces to the shell |r|>SurffRadius (a.u.)
        inline static std::string           EnvelopeType       = "custom";
        inline static std::string           KnotSequence       = "uniform";
        // Per-axis knot sequences parsed from the single "KnotSequence" key.
        // "uniform"                  -> all dims uniform
        // "uniform, uniform, uniform"-> all dims uniform
        // "symexp, uniform, symtanu" -> per-dim (X=symexp, Y=uniform, Z=symtanu)
        inline static std::string           KnotSeq[3]          = {"uniform","uniform","uniform"};
        inline static PetscReal             KnotAlpha        = 0.0;
        // Boundary condition at the box wall (declarative input-file option).
        //   "Neumann"   / "natural" -> reflecting wall (free Galerkin, default)
        //   "Dirichlet"  / "wall"    -> homogeneous psi=0 enforced at the wall
        // Chosen in core.cpp: Dirichlet routes H assembly through
        // TDSEZCompOperatorsDirichlet (applies IGASetBoundaryValue).
        inline static std::string           BoundaryType     = "Neumann";

        // Per-axis hydrogenic knot-region counts (only used when an axis uses
        // KnotSequence = hydrogenic). Each region is the half-axis [0, Lmax]:
        //   HydrogenicNLin[d] -> # linear knots in the near-origin region
        //                        (knot spacing r1 = 0.1 a.u.; r_cross = n_lin*r1)
        //   HydrogenicNExp[d] -> # exponential-graded knots from r_cross to Lmax
        // Given as a comma list like "Nelements" (all axes) or "40,40,40".
        // A value of 0 means "use (Nelements-1)" so omitting the key preserves
        // the original behaviour. Defaults match the previous hard-coded values.
        inline static PetscInt              HydrogenicNLin[3]   = {0, 0, 0};
        inline static PetscInt              HydrogenicNExp[3]   = {0, 0, 0};
        // Per-axis near-origin knot spacing (a.u.) for the hydrogenic sequence.
        // r_cross = HydrogenicNLin[d] * HydrogenicR1[d] must stay < Lmax or the
        // generator throws (caught as a clean TDSEZ FATAL). Default 0.1 matches
        // the previous hard-coded value; 0 means "use 0.1".
        inline static PetscReal             HydrogenicR1[3]     = {0.0, 0.0, 0.0};

        // Adaptive (KnotSequence = adaptive) tuning. Single value, all axes.
        //   AdaptiveKappa -> weight of the |grad V| term (default 1.0)
        //   AdaptivePower -> sharpening exponent on the importance weight
        //                   (default 1.0; >1 concentrates knots more tightly
        //                   on the most important regions)
        inline static PetscReal             AdaptiveKappa      = 1.0;
        inline static PetscReal             AdaptivePower      = 1.0;
        // adaptive_wf (two-pass) coarse-bootstrap resolution: knots/axis on the
        // cheap preliminary ground-state solve. Default 20 is plenty to capture
        // the density SHAPE (which is all the re-mesh needs). 0 => auto (=20).
        inline static PetscInt              AdaptiveWFCoarseN   = 20;
        // Weight of the kinetic-energy-density term in the adaptive_wf knot
        // indicator:  w(x) = |psi|^2 + AdaptiveWFKinLambda * |grad psi|^2.
        // Raw |psi|^2 alone starves the Coulomb cusps (density peaks BETWEEN
        // nuclei) and the exponential tails; |grad psi|^2 fixes both because the
        // gradient diverges at the cusps and is non-zero in the decay region.
        // Default 3.0. 0.0 => pure density (legacy, worse for bound states).
        // lambda>=3 eliminates the high-DOF over-clustering regression on
        // multi-centre potentials, making adaptive_wf best at all DOF.
        inline static PetscReal             AdaptiveWFKinLambda = 3.0;

        inline static PetscInt  NQuadratures      = 8;
        inline static PetscReal Gamma             = 5.0;

        /* Expressions */
        inline static std::string Potential;
        inline static std::string PotentialDerivativeX;
        inline static std::string PotentialDerivativeY;
        inline static std::string PotentialDerivativeZ;
        inline static std::string Laser;
        inline static std::string Envelope;

        // read polarization direction
        inline static std::string Polarization;


        /* Parser instances and bound vars */
        inline static mu::Parser VPot, dVPotX, dVPotY, dVPotZ, Field, LaserXexpr, LaserYexpr, LaserZexpr, Env, MassExpr, dinvMassExprx, dinvMassExpry, dinvMassExprz;
        inline static PetscReal  varX_ = 0.0, varY_ = 0.0, varZ_ = 0.0, varT_ = 0.0;

        /* User-defined muParser constants from the input file (Variables key).
           These are bound on every parser instance in initParsers() so they can
           be referenced by name inside any expression, e.g.
             Variables = V0=1.2, w=0.5
             Potential = 0.5*(x*x+y*y) + V0*exp(-w*x)
           This lets users introduce arbitrary named parameters without the
           solver predefining each one. */
        inline static std::unordered_map<std::string, PetscReal> userConsts_;
        static void trim(std::string &s);

        // Coordinate setter for analytic potential/density evaluation. Public so
        // knot-sequence generators (core_knots.cpp) can sample V and dV/daxis.
        inline static void setVars(PetscReal x) { varX_ = x; varY_ = varZ_ = 0.0; }
        inline static void setVars(PetscReal x, PetscReal y) { varX_ = x; varY_ = y; varZ_ = 0.0; }
        inline static void setVars(PetscReal x, PetscReal y, PetscReal z) { varX_ = x; varY_ = y; varZ_ = z; }
        inline static void setVars(const std::vector<PetscReal>& vars) {
            varX_ = (vars.size() > 0) ? vars[0] : 0.0;
            varY_ = (vars.size() > 1) ? vars[1] : 0.0;
            varZ_ = (vars.size() > 2) ? vars[2] : 0.0;
        }
};

// ============================================================================
//  Shared physics structures & types
// ============================================================================
struct PhysicsDerivatives {
    std::vector<PetscReal> grad; // Nabla f
    PetscReal laplacian;         // Nabla^2 f
};

struct PhysicsDerivativesFull {
    std::vector<PetscReal> first;
    std::vector<PetscReal> second;
    std::vector<PetscReal> third;
};

typedef PetscErrorCode (*TDSEZPhysicsKernel)(
    IGAPoint p,
    PetscInt nmat,
    PetscScalar *K[],
    void *ctx);


// ============================================================================
//  TDSEZAssembler  — builds the linear operators of the TDSE Hamiltonian
// ============================================================================
class TDSEZAssembler
{
    private:
        // Private members
        IGA& iga;

    public:
        // Constructors
        TDSEZAssembler(IGA& iga);

        // Destructor
        ~TDSEZAssembler()
        {
            if (Dx)     MatDestroy(&Dx);
            if (Dy)     MatDestroy(&Dy);
            if (Dz)     MatDestroy(&Dz);
            if (VelX)   MatDestroy(&VelX);
            if (VelY)   MatDestroy(&VelY);
            if (VelZ)   MatDestroy(&VelZ);
            if (dVdx)   MatDestroy(&dVdx);
            if (dVdy)   MatDestroy(&dVdy);
            if (dVdz)   MatDestroy(&dVdz);
            if (CAP)    MatDestroy(&CAP);
            if (Md)     MatDestroy(&Md);
            if (K)      MatDestroy(&K);
            if (V)      MatDestroy(&V);
            if (Lz)     MatDestroy(&Lz);
        }


    public:
        // Public references maintain original interface
        Mat Md = PETSC_NULLPTR; // Md matrix
        Mat Dx = PETSC_NULLPTR, Dy = PETSC_NULLPTR, Dz = PETSC_NULLPTR;
        Mat VelX = PETSC_NULLPTR, VelY = PETSC_NULLPTR, VelZ = PETSC_NULLPTR;
        Mat dVdx = PETSC_NULLPTR, dVdy = PETSC_NULLPTR, dVdz = PETSC_NULLPTR, CAP = PETSC_NULLPTR; // CAP matrix
        Mat K = PETSC_NULLPTR;   // Kinetic energy matrix
        Mat V = PETSC_NULLPTR;   // Potential energy matrix
        Mat Lz = PETSC_NULLPTR;  // Angular momentum matrix
};


// ============================================================================
//  TDSEZCore  — IGA setup, ground-state solve, eigenstates
// ============================================================================
class TDSEZCore
{
    friend class TDSEZAssembler;
    struct FormContext {
        TDSEZCore* self;  // Pointer to the class instance
    };


    public:
        TDSEZCore(const std::string &input);

        // Destructor
        ~TDSEZCore();


        // MPI info
        PetscMPIInt rank = 0, size = 1;

        // input file
        std::string inputFile;

        // IGA object
        IGA iga = PETSC_NULLPTR;
        EPS eps = PETSC_NULLPTR;
        PetscInt nconv = 0;
        PetscReal alpha0 = 0.0;

        // methods
        static std::vector<PetscReal> TDSEZExpSymKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscReal alpha, PetscInt p);
        static std::vector<PetscReal> TDSEZTanUSymKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscReal beta, PetscInt p);
        static std::vector<PetscReal> TDSEZTanNSymKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscReal beta, PetscInt p);
        static std::vector<PetscReal> TDSEZLogTanKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscReal alpha, PetscInt p);
        static std::vector<PetscReal> TDSEZInterfaceKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscInt p, const std::string& expr, const std::string& massexpr);
        static std::vector<PetscReal> TDSEZHydrogenicKnots(PetscReal Lmax, PetscInt n_lin, PetscReal r1, PetscInt n_exp, PetscInt p);
        // Adaptive (potential-driven, single-pass) knot vector: clusters knots
        // where the potential is "interesting" (steep gradient + deep wells) by
        // CDF-inverting a tanh-compressed importance weight. axis selects which
        // derivative parser to use (0->x,1->y,2->z); kappa weights the gradient
        // term, power sharpens the weight. See core_knots.cpp for details.
        static std::vector<PetscReal> TDSEZAdaptiveKnots(PetscReal Lmin, PetscReal Lmax,
            PetscInt ninterior, PetscInt p, PetscInt axis,
            PetscReal kappa = 1.0, PetscReal power = 1.0);
        // Adaptive (density-driven, two-pass) knot vector: clusters knots where
        // the ELECTRON DENSITY is large. rho_axis is a precomputed 1D marginal
        // |psi0|^2 (obtained from a cheap coarse ground-state solve — see
        // BootstrapDensity). CDF-inverts the density so knots concentrate where
        // the wavefunction lives. Same clamped p+1 boundary convention.
        static std::vector<PetscReal> TDSEZAdaptiveWFKnots(PetscReal Lmin, PetscReal Lmax,
            PetscInt ninterior, PetscInt p,
            const std::vector<PetscReal>& rho_axis);
        // Auto interface-aligned knots: extract breakpoints from a piecewise
        // (?:) Potential/Mass expression and pin a knot line there, grading the
        // rest by a CDF that clusters resolution at the material boundaries.
        static std::vector<PetscReal> TDSEZInterfaceKnots(
            PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscInt p,
            const std::string& expr);
         PetscErrorCode SolveImaginaryTime();

        PetscErrorCode Assemble();
        PetscErrorCode Solve();
        PetscErrorCode Output();

        // ── adaptive_wf (two-pass density-driven) support ──────────────
        // BootstrapDensity(): if any axis uses KnotSequence = adaptive_wf, run a
        // CHEAP coarse ground-state solve (uniform mesh, AdaptiveWFCoarseN
        // knots/axis), sample the 1D marginal electron density |psi0|^2 onto
        // rho_x_/rho_y_/rho_z_, then tear down the coarse IGA/matrices. The real
        // knot build below then consumes these marginals. No-op otherwise.
        PetscErrorCode BootstrapDensity();
        // Sample |psi|^2 on the coarse volume grid and marginalise to 1D
        // histograms per axis (normalised to unit integral) into rho_*_.
        PetscErrorCode SampleDensity(IGA iga, Vec psi);

        // friend classes
        friend class TDSEZAssembler;

        // stiffness and MASS matrices
        Mat H = PETSC_NULLPTR, M = PETSC_NULLPTR;

        // knots
        std::vector<PetscReal> knots_x, knots_y, knots_z;

        // adaptive_wf: 1D marginal electron densities |psi0|^2, filled by
        // BootstrapDensity() (one vector per axis, normalised to unit integral).
        std::vector<PetscReal> rho_x_, rho_y_, rho_z_;

        // initial state
        Vec initialPsi = PETSC_NULLPTR;

        // bound states and populations
        std::vector<Vec> boundstates;
        std::vector<PetscReal> energies;
};


// ============================================================================
//  TDSEZManager  — orchestrates propagation, diagnostics, HDF5, t-SURFF
// ============================================================================
class TDSEZManager
{
    private:
    TDSEZCore* core_;
    TDSEZAssembler* assembler_;

    public:
        TDSEZManager(TDSEZCore* core, TDSEZAssembler* assembler);

        ~TDSEZManager();

        // disable copy to avoid PetscReal-destroy
        TDSEZManager(const TDSEZManager&) = delete;
        TDSEZManager& operator=(const TDSEZManager&) = delete;
        TDSEZManager(TDSEZManager&&) = delete;
        TDSEZManager& operator=(TDSEZManager&&) = delete;


    // ── Operator / state objects (private: owned by the Manager lifecycle) ──
    // Callers must NOT reach in and null these (it would break the propagator);
    // use the getters below. Naming: m_ prefix = private member.
    Mat m_M  = PETSC_NULLPTR;  // mass
    Mat m_H  = PETSC_NULLPTR;  // Hamiltonian (H = -∇²/2m + V + CAP, post-CAP-apply)
    Mat m_K  = PETSC_NULLPTR;  // kinetic
    Mat m_V  = PETSC_NULLPTR;  // potential
    Mat m_Dx = PETSC_NULLPTR, m_Dy = PETSC_NULLPTR, m_Dz = PETSC_NULLPTR; // dipole (position) ops
    Mat m_VelX = PETSC_NULLPTR, m_VelY = PETSC_NULLPTR, m_VelZ = PETSC_NULLPTR; // velocity-gauge ops
    Mat m_dVdx = PETSC_NULLPTR, m_dVdy = PETSC_NULLPTR, m_dVdz = PETSC_NULLPTR; // -∇V components
    Mat m_Md = PETSC_NULLPTR;  // mass-derivative op (energy/dipole diagnostics)
    Mat m_CAP = PETSC_NULLPTR; // complex absorbing potential (consumed by propagator)
    Mat m_Ht  = PETSC_NULLPTR; // transient H copy used inside the TS step
    Mat m_Lz  = PETSC_NULLPTR; // angular-momentum op (2D degeneracy split; NULL in 3D)
    Vec m_psi0 = PETSC_NULLPTR; // initial state |ψ(0)⟩ (for autocorrelation)

    // Getters (read-only views of the operator/state objects).
    Mat       M()     const { return m_M; }
    Mat       H()     const { return m_H; }
    Mat       K()     const { return m_K; }
    Mat       V()     const { return m_V; }
    Mat       Dx()    const { return m_Dx; }
    Mat       Dy()    const { return m_Dy; }
    Mat       Dz()    const { return m_Dz; }
    Mat       VelX()  const { return m_VelX; }
    Mat       VelY()  const { return m_VelY; }
    Mat       VelZ()  const { return m_VelZ; }
    Mat       dVdx()  const { return m_dVdx; }
    Mat       dVdy()  const { return m_dVdy; }
    Mat       dVdz()  const { return m_dVdz; }
    Mat       Md()    const { return m_Md; }
    Mat       CAP()   const { return m_CAP; }
    Mat       Ht()    const { return m_Ht; }
    Mat       Lz()    const { return m_Lz; }
    Vec       psi0()  const { return m_psi0; }

    // Setters (used only during setup / CAP consumption — not by callers at large).
    void setCAP(Mat c)  { m_CAP  = c; }
    void setPsi0(Vec p) { m_psi0 = p; }
    void setHt(Mat h)   { m_Ht   = h; }

    // Polarization selection
    PetscBool hasX = PETSC_FALSE, hasY = PETSC_FALSE, hasZ = PETSC_FALSE;

    // Buffers (allocated on rank 0 only)
    PetscScalar *dipBuffer = PETSC_NULLPTR; PetscScalar *popBuffer = PETSC_NULLPTR; PetscScalar *energyBuffer = PETSC_NULLPTR, *currBuffer = PETSC_NULLPTR, *acBuffer = PETSC_NULLPTR;

    Vec *IFuncVec = PETSC_NULLPTR;
    PetscReal BerryPhase = 0.0;

    // P vecs
    std::vector<PetscScalar> vMat_x, vMat_y, vMat_z;

    // access vectors
    std::vector<Vec>            boundstates;
    std::vector<Vec>            dipoleVecs;
    std::vector<PetscReal>      population;
    std::vector<PetscReal>      energies;
    std::vector<PetscScalar>    popDots;
    std::vector<PetscScalar>    popDots_prev;
    Vec *tmpVec = PETSC_NULLPTR;
    PetscInt numVecs = 20;
    PetscInt numIFuncVec = 4;
    PetscInt maxAllocatedSteps = 0, recordedSteps = 0, dipWidth = 0, popWidth = 0, energyWidth = 0;

    // other parameters
    PetscInt stride = 2000;
    static constexpr PetscInt FLUSH_INTERVAL = 5000;
    PetscInt lastWrittenSteps = 0;
    std::string inputFile;
    std::string outputFilename;

    // Persistent HDF5 file handle (kept open across flushes for efficiency).
    // -1 == not open.
    hid_t       h5File           = -1;
    bool        h5Compress       = false;
    int         h5CompressLevel  = 6;

    // physical constants
    PetscReal mass = 0.0, invMass = 0.0, hbar = 0.0, charge = 0.0, chargeOvMass = 0.0;
    PetscInt  NPOP = 1, rank = 0;

    // get core and assembler
    TDSEZCore* getCore() { return core_; }
    TDSEZAssembler* getAssembler() { return assembler_; }

    // viewers
    PetscViewer TIMEViewer = PETSC_NULLPTR, WFSViewer = PETSC_NULLPTR, ACViewer = PETSC_NULLPTR;

    // Polarization selection
    PetscErrorCode (*TDSEZIFunctionPtr)(TS,PetscReal,Vec,Vec,Vec,void*) = PETSC_NULLPTR;
    PetscErrorCode (*TDSEZIJacobianPtr)(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*) = PETSC_NULLPTR;
    PetscErrorCode WriteHDF5(const std::string& filename);
    PetscErrorCode CloseHDF5();
    PetscErrorCode PolarizationSelector();

    // ---------------- t-SURFF (time-dependent surface flux) ----------------
    // Faces indexed 0=x-,1=x+,2=y-,3=y+,4=z-,5=z+  i.e. (axis,side)=(f/2,f%2)
    // Only faces consistent with hasX/hasY/hasZ are built/used.
    Mat  Bval_face[6] = {PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR,
                          PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR};
    Mat  Bder_face[6] = {PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR,
                          PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR};
    // MatMult outputs — all rows owned by rank 0 (see BuildFaceOperators), so
    // these come back as naturally rank-0-local data (matches the "diagnostics
    // live on rank 0" convention used for dipBuffer/energyBuffer/etc.).
    Vec  faceValVec[6] = {PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR,
                           PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR};
    Vec  faceDerVec[6] = {PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR,
                           PETSC_NULLPTR,PETSC_NULLPTR,PETSC_NULLPTR};

    // Physical boundary quad-point coords + integration weight (includes the
    // boundary Jacobian, i.e. dS = w_q), gathered onto rank 0 in
    // BuildFaceOperators. faceNquad[f] = total #quad pts on face f (rank 0 only).
    PetscInt faceNquad[6] = {0,0,0,0,0,0};
    // psi gather: all ranks participate in the scatter (collective), rank 0 keeps
    // the sequential copy that the SELF face matrices MatMult against.
    Vec       psiZero       = PETSC_NULLPTR;
    VecScatter surffScatter = PETSC_NULLPTR;
    std::vector<PetscReal> faceX[6], faceY[6], faceZ[6];
    std::vector<PetscReal> faceW[6];
    PetscInt   faceAxis[6]     = {0,0,1,1,2,2};   // 0=x,1=y,2=z
    PetscReal  faceSign[6]     = {-1,+1,-1,+1,-1,+1}; // outward-normal sign
    // Per-quad-point weight mask: 1 if |r_q|>SurffRadius (active shell), else 0.
    std::vector<PetscReal> faceMask[6];

    // Running vector potential A(t) = -int_0^t E(t')dt', trapezoidal update.
    PetscReal Ax = 0.0, Ay = 0.0, Az = 0.0;
    PetscReal surffTprev = 0.0, surffExPrev = 0.0, surffEyPrev = 0.0, surffEzPrev = 0.0;
    PetscBool surffInitialized = PETSC_FALSE;

    // Momentum grid + running accumulators (rank 0 only; size Nk^3).
    PetscInt   surffNk = 0;
    PetscReal  surffKmax = 0.0;
    PetscReal  surffMass = 1.0;        // sampled particle mass (constant-mass approx)
    std::vector<PetscReal>   kAxis;        // Nk values, -Kmax..+Kmax
    std::vector<PetscReal>   surffPhi;     // running Phi(k,t), size Nk^3
    std::vector<PetscScalar> surffAmp;     // running b(k,t) accumulator, size Nk^3
    // Per-face diagnostic PES (only allocated/used when TSURFF_FACE_SPLIT env set).
    // Lets us verify each face family (x/y/z) reconstructs the correct momentum.
    std::vector<PetscScalar> surffAmpFace[6];  // size Nk^3 each, rank 0 only
    PetscBool surffFaceSplit = PETSC_FALSE;

    // Per-fold separable buffers (rank 0). For an active face the boundary flux
    // factorizes as J_f(k) = e^{-i k_d X_d} [ A_f(k_u) + i*sign*k_d * B_f(k_u) ]
    // where (k_u,k_d) split the momentum along the face's tangent/normal axes.
    // A_f/B_f live on the (dim-1)-dimensional tangent k-grid and are rebuilt every
    // fold; this makes the per-fold cost O(Nk^{d-1}*Nq + Nk^d) instead of the
    // naive O(Nk^d*Nq) -- an exact factor-Nk speedup with no approximation.
    PetscInt   surffTangentDim = 0;        // dim-1
    std::vector<PetscScalar> surffAf[6];   // size (Nk)^(d-1) per active face
    std::vector<PetscScalar> surffBf[6];   // size (Nk)^(d-1) per active face

    PetscErrorCode SetupSurff();
    PetscErrorCode BuildFaceOperators();
    PetscErrorCode AccumulateSurff(Vec psi, PetscReal t, PetscReal dt, PetscInt step);
    PetscErrorCode FinalizeSurff();
};


// ============================================================================
//  TDSEZPropagator  — time-stepping driver (PETSc TS)
// ============================================================================
class TDSEZPropagator
{

public:

    // create
    TDSEZPropagator(TDSEZManager &manager);

    // destroy
    ~TDSEZPropagator() {
        PetscFunctionBeginUser;
        // cleanup
        if (J_) MatDestroy(&J_);
        if (ts_) TSDestroy(&ts_);
        PetscFunctionReturnVoid();
    }

    PetscErrorCode Evolve();

private:
    TDSEZCore &tdse_;
    TDSEZManager  &manager_;

    TS  ts_  = PETSC_NULLPTR;
    Mat J_   = PETSC_NULLPTR;
};


// ============================================================================
//  Cross-TU free functions (prototypes)
// ============================================================================

// parser support (defined inline in the header so they are visible in every TU)
inline PhysicsDerivatives compute_physics_ops(
    const std::function<PetscReal(const std::vector<PetscReal>&)>& func,
    const std::vector<PetscReal>& x,
    PetscReal h = 1e-4)
{
    const int dim = x.size();
    PhysicsDerivatives out;
    out.grad.resize(dim);
    out.laplacian = 0.0;

    // Precompute inverse factors
    const PetscReal inv_h = 1.0 / h;
    const PetscReal inv_12h = inv_h / 12.0;

    // Reusable working vectors (avoid heap allocations in loop)
    std::vector<PetscReal> x_p1(dim), x_m1(dim), x_p2(dim), x_m2(dim);

    for (int i = 0; i < dim; ++i)
    {
        // Copy x once (faster than multiple assignments)
        std::copy(x.begin(), x.end(), x_p1.begin());
        std::copy(x.begin(), x.end(), x_m1.begin());
        std::copy(x.begin(), x.end(), x_p2.begin());
        std::copy(x.begin(), x.end(), x_m2.begin());

        // Modify only the i-th component
        x_p1[i] += h;
        x_m1[i] -= h;
        x_p2[i] += 2.0 * h;
        x_m2[i] -= 2.0 * h;

        const PetscReal f_p1 = func(x_p1);
        const PetscReal f_m1 = func(x_m1);
        const PetscReal f_p2 = func(x_p2);
        const PetscReal f_m2 = func(x_m2);

        // 4th-order gradient (∂f/∂xi)
        out.grad[i] = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) * inv_12h;
    }

    return out;
}

inline PhysicsDerivativesFull TDSEZCompDerivative(
    const std::function<PetscReal(const std::vector<PetscReal>&)>& func,
    const std::vector<PetscReal>& x,
    PetscReal h = 1e-4)
{
    const int dim = x.size();
    PhysicsDerivativesFull out;
    out.first.resize(dim);
    out.second.resize(dim * dim);
    out.third.resize(dim * dim * dim);

    // Precompute inverse factors
    const PetscReal inv_h = 1.0 / h;
    const PetscReal inv_12h = inv_h / 12.0;
    const PetscReal inv_4h2 = 1.0 / (4.0 * h * h);

    std::vector<PetscReal> x_p1 = x, x_m1 = x, x_p2 = x, x_m2 = x;

    for (int i = 0; i < dim; ++i)
    {
        x_p1[i] = x[i] + h;
        x_m1[i] = x[i] - h;
        x_p2[i] = x[i] + 2.0 * h;
        x_m2[i] = x[i] - 2.0 * h;

        const PetscReal f_p1 = func(x_p1);
        const PetscReal f_m1 = func(x_m1);
        const PetscReal f_p2 = func(x_p2);
        const PetscReal f_m2 = func(x_m2);

        // 4th-order gradient (∂f/∂xi)
        out.first[i] = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) * inv_12h;

        x_p1[i] = x_m1[i] = x_p2[i] = x_m2[i] = x[i];
    }

    // second derivatives (Hessian)
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            std::vector<PetscReal> x_pp = x, x_pm = x, x_mp = x, x_mm = x;
            x_pp[i] += h; x_pp[j] += h;
            x_pm[i] += h; x_pm[j] -= h;
            x_mp[i] -= h; x_mp[j] += h;
            x_mm[i] -= h; x_mm[j] -= h;

            const PetscReal f_pp = func(x_pp);
            const PetscReal f_pm = func(x_pm);
            const PetscReal f_mp = func(x_mp);
            const PetscReal f_mm = func(x_mm);

            out.second[i * dim + j] = (f_pp - f_pm - f_mp + f_mm) * inv_4h2;
        }
    }

    // third derivatives (∂³f/∂xi∂xj∂xk)
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            for (int k = 0; k < dim; ++k)
            {
                std::vector<PetscReal> x_ppp = x, x_ppm = x, x_pmp = x, x_pmm = x, x_mpp = x, x_mpm = x, x_mmp = x, x_mmm = x;
                x_ppp[i] += h; x_ppp[j] += h; x_ppp[k] += h;
                x_ppm[i] += h; x_ppm[j] += h; x_ppm[k] -= h;
                x_pmp[i] += h; x_pmp[j] -= h; x_pmp[k] += h;
                x_pmm[i] += h; x_pmm[j] -= h; x_pmm[k] -= h;
                x_mpp[i] -= h; x_mpp[j] += h; x_mpp[k] += h;
                x_mpm[i] -= h; x_mpm[j] += h; x_mpm[k] -= h;
                x_mmp[i] -= h; x_mmp[j] -= h; x_mmp[k] += h;
                x_mmm[i] -= h; x_mmm[j] -= h; x_mmm[k] -= h;
                const PetscReal f_ppp = func(x_ppp);
                const PetscReal f_ppm = func(x_ppm);
                const PetscReal f_pmp = func(x_pmp);
                const PetscReal f_pmm = func(x_pmm);
                const PetscReal f_mpp = func(x_mpp);
                const PetscReal f_mpm = func(x_mpm);
                const PetscReal f_mmp = func(x_mmp);
                const PetscReal f_mmm = func(x_mmm);
                out.third[(i * dim + j) * dim + k] = (f_ppp - f_ppm - f_pmp + f_pmm - f_mpp + f_mpm + f_mmp - f_mmm) * inv_h * inv_h * inv_h / 8.0;
            }
        }
    }

    return out;
}

PetscReal Manolopoulos_CAP_profile(const PetscReal xq, const PetscReal, const PetscReal x_max, const PetscReal ma_kmin);

// operator composition
PetscErrorCode TDSEZCompOperators(
    IGA                iga,
    PetscInt           nmat,
    Mat                mats[],
    TDSEZPhysicsKernel kernel,
    void              *ctx);

// Same as TDSEZCompOperators but APPLIES Dirichlet BCs (IGASetBoundaryValue)
// via IGAElementFixSystem. The plain TDSEZCompOperators leaves the boundary
// free (Neumann / reflecting wall); this enforces homogeneous psi=0 at the wall.
PetscErrorCode TDSEZCompOperatorsDirichlet(
    IGA                iga,
    PetscInt           nmat,
    Mat                mats[],
    TDSEZPhysicsKernel kernel,
    void              *ctx);

// physics / form kernels (must be non-static: referenced from other TUs)
PetscErrorCode TDSEZFormHam(IGAPoint p, PetscInt nmat, PetscScalar *M[], void *ctx);
PetscErrorCode TDSEZFormPhyX(IGAPoint p, PetscInt nmat, PetscScalar *M[], void *ctx);
PetscErrorCode TDSEZFormHamiltonian(IGAPoint p, PetscScalar*H, void *ctx);
PetscErrorCode TDSEZFormLz(IGAPoint p, PetscScalar *L, void *ctx);
PetscErrorCode TDSEZFormKinetic(IGAPoint p, PetscScalar*K, void *ctx);
PetscErrorCode TDSEZFormPotential(IGAPoint p, PetscScalar*V, void *ctx);
PetscErrorCode TDSEZFormMass(IGAPoint p, PetscScalar* __restrict__ M, void *ctx);
PetscErrorCode TDSEZFormMassDist(IGAPoint p, PetscScalar*Md, void *ctx);
PetscErrorCode DipoleX(IGAPoint p, PetscScalar *X, void *ctx);
PetscErrorCode DipoleY(IGAPoint p, PetscScalar*Y, void *ctx);
PetscErrorCode DipoleZ(IGAPoint p, PetscScalar*Z, void *ctx);
PetscErrorCode VelocityX(IGAPoint p, PetscScalar *P, void *ctx);
PetscErrorCode VelocityY(IGAPoint p, PetscScalar *P, void *ctx);
PetscErrorCode VelocityZ(IGAPoint p, PetscScalar *P, void *ctx);
PetscErrorCode MomentumBase(IGAPoint p, PetscScalar *P, void *ctx);
PetscErrorCode TDSEZformPotentialGradX(IGAPoint p, PetscScalar *dVdr, void *ctx);
PetscErrorCode TDSEZformPotentialGradY(IGAPoint p, PetscScalar *dVdr, void *ctx);
PetscErrorCode TDSEZformPotentialGradZ(IGAPoint p, PetscScalar *dVdr, void *ctx);
PetscErrorCode TDSEZformPotentialGrad_old(IGAPoint p, PetscScalar*dV, void *ctx);
PetscErrorCode TDSEZformCap(IGAPoint p, PetscScalar *CAP, void *ctx);

// assembler helpers
PetscErrorCode TDSEZAssembleMatrixTimed(IGA iga, Mat *M, IGAFormMatrix form, const char *name);
PetscErrorCode TDSEZAssembleMatBatchTimed(IGA iga, PetscInt n, Mat **mats, TDSEZPhysicsKernel form, const char *name);

// HDF5 monitors
PetscErrorCode TDSEZMonitorHDF5_1D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx);
PetscErrorCode TDSEZMonitorHDF5_2D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx);
PetscErrorCode TDSEZMonitorHDF5_3D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx);

// degeneracy handling
PetscErrorCode TDSEZOrthogonalizeDegenerates(TDSEZManager *TDSEZ);

// state utilities
PetscScalar TDSEZGaussianState(PetscReal x);
PetscScalar TDSEZSneeState(PetscReal x);
PetscErrorCode TDSEZGetCoeffs(IGAPoint p, PetscScalar *V, void *ctx);
PetscErrorCode TDSEZProjState(TDSEZCore &tdse, Vec &projState);
PetscErrorCode TDSEZCompUnifiedDipoleMatrix(
    const std::vector<Vec>& states,
    const std::vector<PetscReal>& energies,
    Mat Dx, Mat Dy, Mat Dz,
    const std::string& fileName);
PetscErrorCode TDSEZSaveOperatorMatrix(Mat mat, const std::string& filename);
PetscErrorCode ComputeCommutator(Mat H, Mat Lz, Mat *C_out);
PetscErrorCode TDSEZMakeStateReal2(Vec v_complex, Vec Vre, Mat M);
PetscErrorCode MakeStatesReal(std::vector<Vec>& states, Mat M);
PetscErrorCode TDSEZComputeCurrents(TDSEZManager *TDSEZ,
                                     Vec psi,
                                     PetscReal t,
                                     PetscScalar *Jx_intra,
                                     PetscScalar *Jy_intra,
                                     PetscScalar *Jx_inter,
                                     PetscScalar *Jy_inter,
                                     PetscScalar *Jx_bc,
                                     PetscScalar *Jy_bc,
                                     PetscScalar *Jx_total,
                                     PetscScalar *Jy_total);
PetscErrorCode TDSEZCheckLengthVelocity(TDSEZManager *TDSEZ);
PetscErrorCode TDSEZPrecomputeMomentumMatrix(TDSEZManager *TDSEZ);
PetscErrorCode DetectAndPrintSymmetry(
    PetscReal (*V2)(PetscReal x, PetscReal y),           // pass nullptr if 3D
    PetscReal (*V3)(PetscReal x, PetscReal y, PetscReal z), // pass nullptr if 2D
    PetscReal xmax, PetscReal ymax, PetscReal zmax,      // zmax ignored in 2D
    PetscInt  Nsample,
    PetscInt  dim);                                       // 2 or 3

// interface-aware knot breakpoints
std::vector<PetscReal> TDSEZExtractBreakpoints(const std::string& expr);

#endif // __TDSEZ_INTERNAL_HPP__
