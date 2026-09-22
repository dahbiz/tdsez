#ifndef TDSEZ_PARSER_HPP
#define TDSEZ_PARSER_HPP

#include <petscsys.h>
#include "muParser.h"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ============================================================================
//  TDSEZParser  — input / parameter parser (declaration region verbatim)
// ============================================================================
/**
 * @brief Input-file and parameter parser for the TDSE-Z solver.
 * @details Reads a PETSc-style parameter file via PrmReader(), initialises the
 * muParser expression instances in initParsers(), and exposes static inline
 * members holding every physics, grid, field, envelope, and output parameter.
 * Accessor templates (V, dV, E, etc.) set coordinates and evaluate the
 * corresponding muParser expression at the current point.
 */
class TDSEZParser
{
    public:

        /**
         * @brief Construct the parser, read the parameter file, and init parsers.
         * @param prm Path to the input parameter file.
         */
        TDSEZParser(const std::string &prm)
        {
            PrmReader(prm);
            initParsers();
        }


        // Read parameters from file
        /**
         * @brief Read parameters from the input file into the static members.
         * @param prm Path to the input parameter file.
         */
        static void PrmReader(const std::string &prm);
        /**
         * @brief Initialise all muParser instances and bind variables/constants.
         */
        static void initParsers();

        /**
         * @brief Validate all parsed parameters for physical consistency.
         * @details Throws std::runtime_error on any physically inconsistent or
         * dangerous parameter combination. Called once after the parser has
         * finished reading the input file.
         * @throws std::runtime_error if a parameter combination is invalid.
         */
        /* Input validation: throws std::runtime_error on any physically
           inconsistent / dangerous parameter combination. Called once after
           the parser has finished reading the input file. */
        static void ValidateOrThrow();

        /**
         * @brief Return a short code version string (git describe or static tag).
         * @return Version string stored in HDF5 provenance for reproducibility.
         */
        /* Short code version string (git describe when available, else a
           static tag). Stored in HDF5 provenance for reproducibility. */
        static std::string VersionString();

        // accessors
        /**
         * @brief Evaluate the scalar potential V at the given coordinates.
         * @tparam Args Coordinate values (1D, 2D, or 3D).
         * @param args Coordinates to set before evaluating the potential.
         * @return V(x[,y[,z]]) in atomic units.
         */
        template<typename... Args>
        inline static PetscReal V(Args... args)     { setVars(args...) ; return VPot.Eval(); }
        /**
         * @brief Evaluate dV/dx (alias of dVx) at the given coordinates.
         * @tparam Args Coordinate values.
         * @param args Coordinates to set before evaluating.
         * @return dV/dx in atomic units.
         */
        template<typename... Args>
        inline static PetscReal dV(Args... args)    { setVars(args...); return dVPotX.Eval(); }
        /**
         * @brief Evaluate dV/dx at the given coordinates.
         * @tparam Args Coordinate values.
         * @param args Coordinates to set before evaluating.
         * @return dV/dx in atomic units.
         */
        template<typename... Args>
        inline static PetscReal dVx(Args... args)      { setVars(args...); return dVPotX.Eval(); }
        /**
         * @brief Evaluate dV/dy at the given coordinates.
         * @tparam Args Coordinate values.
         * @param args Coordinates to set before evaluating.
         * @return dV/dy in atomic units.
         */
        template<typename... Args>
        inline static PetscReal dVy(Args... args)      { setVars(args...); return dVPotY.Eval(); }
        /**
         * @brief Evaluate dV/dz at the given coordinates.
         * @tparam Args Coordinate values.
         * @param args Coordinates to set before evaluating.
         * @return dV/dz in atomic units.
         */
        template<typename... Args>
        inline static PetscReal dVz(Args... args)      { setVars(args...); return dVPotZ.Eval(); }
        /**
         * @brief Evaluate the total laser field E(t) at time @p t.
         * @param t Time in atomic units.
         * @return E(t) in atomic units.
         */
        inline static PetscReal E(PetscReal t)      { varT_ = t; return Field.Eval(); }
        /**
         * @brief Evaluate the x-component of the laser field at time @p t.
         * @param t Time in atomic units.
         * @return Ex(t) in atomic units.
         */
        inline static PetscReal Ex(PetscReal t)      { varT_ = t; return LaserXexpr.Eval(); }
        /**
         * @brief Evaluate the y-component of the laser field at time @p t.
         * @param t Time in atomic units.
         * @return Ey(t) in atomic units.
         */
        inline static PetscReal Ey(PetscReal t)      { varT_ = t; return LaserYexpr.Eval(); }
        /**
         * @brief Evaluate the z-component of the laser field at time @p t.
         * @param t Time in atomic units.
         * @return Ez(t) in atomic units.
         */
        inline static PetscReal Ez(PetscReal t)      { varT_ = t; return LaserZexpr.Eval(); }
        /**
         * @brief Evaluate the envelope function F(t) at time @p t.
         * @param t Time in atomic units.
         * @return Envelope F(t) in atomic units.
         */
        inline static PetscReal F(PetscReal t)      { varT_ = t; return Env.Eval(); }
        /**
         * @brief Evaluate the position-dependent mass distribution M(x,y,z).
         * @param x Cartesian x coordinate.
         * @param y Cartesian y coordinate.
         * @param z Cartesian z coordinate.
         * @return Mass M(x,y,z) in atomic units.
         */
        inline static PetscReal MassDist(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return MassExpr.Eval(); }
        /**
         * @brief Evaluate d(1/M)/dx at the given coordinates.
         * @param x Cartesian x coordinate.
         * @param y Cartesian y coordinate.
         * @param z Cartesian z coordinate.
         * @return d(1/M)/dx in atomic units.
         */
        inline static PetscReal dinvMassDistx(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return dinvMassExprx.Eval(); }
        /**
         * @brief Evaluate d(1/M)/dy at the given coordinates.
         * @param x Cartesian x coordinate.
         * @param y Cartesian y coordinate.
         * @param z Cartesian z coordinate.
         * @return d(1/M)/dy in atomic units.
         */
        inline static PetscReal dinvMassDisty(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return dinvMassExpry.Eval(); }
        /**
         * @brief Evaluate d(1/M)/dz at the given coordinates.
         * @param x Cartesian x coordinate.
         * @param y Cartesian y coordinate.
         * @param z Cartesian z coordinate.
         * @return d(1/M)/dz in atomic units.
         */
        inline static PetscReal dinvMassDistz(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return dinvMassExprz.Eval(); }

        // private:
        /// Mathematical constant pi.
        static constexpr PetscReal PI     = 3.14159265358979323846;
        /// Position-dependent mass expression string (default "1.0").
        inline static std::string Mass    = "1.0";
        /// Inverse-mass x-gradient expression string.
        inline static std::string dinvMassX  = "0.0";
        /// Inverse-mass y-gradient expression string.
        inline static std::string dinvMassY  = "0.0";
        /// Inverse-mass z-gradient expression string.
        inline static std::string dinvMassZ  = "0.0";
        /// Whether the mass is spatially constant (set in ValidateOrThrow).
        inline static PetscBool MassIsConstant = PETSC_FALSE;


        // inline static PetscReal Mass      = 1.0;
        /// Reduced Planck constant hbar (atomic units, default 1.0).
        inline static PetscReal Hbar      = 1.0;
        /// Particle charge q (atomic units, default 1.0).
        inline static PetscReal Q         = 1.0;

        /* Spatial grid */
        /// Whether to print verbose diagnostic output.
        inline static PetscBool Verbose          = PETSC_TRUE;
        // When PETSC_TRUE (default) an unknown input key is a FATAL error
        // instead of a silent warning — prevents typo'd keys from silently
        // no-op'ing in a physics code. Set StrictInput = 0 to tolerate.
        /// Whether unknown input keys are fatal errors (default true).
        inline static PetscBool StrictInput      = PETSC_TRUE;
        /// Spatial dimension of the problem (1, 2, or 3).
        inline static PetscReal Dimension        = 1; // 1, 2, or 3
        /// B-spline polynomial degree (1 to 14).
        inline static PetscInt  SplineDegree     = 3; // 1 to 14
        /// Number of elements per axis (uniform default).
        inline static PetscInt  Nelements         = 100;
        /// Global lower bound of the spatial domain (symmetric default).
        inline static PetscReal LMin             = -10.0;
        /// Global upper bound of the spatial domain (symmetric default).
        inline static PetscReal LMax             = +10.0;
        // Per-axis bounds for rectangular/asymmetric boxes. LMin*/LMax* default to
        // LMin/LMax when unset, so the symmetric (cube/parallelepiped) case needs
        // only LMin/LMax. Domain = [Lminx,Lmaxx],[Lminy,Lmaxy],[Lminz,Lmaxz] sets these.
        /// Lower bound of the x-axis domain.
        inline static PetscReal LMinX            = -10.0;
        /// Lower bound of the y-axis domain.
        inline static PetscReal LMinY            = -10.0;
        /// Lower bound of the z-axis domain.
        inline static PetscReal LMinZ            = -10.0;
        /// Upper bound of the x-axis domain.
        inline static PetscReal LMaxX            = +10.0;
        /// Upper bound of the y-axis domain.
        inline static PetscReal LMaxY            = +10.0;
        /// Upper bound of the z-axis domain.
        inline static PetscReal LMaxZ            = +10.0;
        /// Origin offset along x.
        inline static PetscReal OffsetX          = 0.0;
        /// Origin offset along y.
        inline static PetscReal OffsetY          = 0.0;
        /// Origin offset along z.
        inline static PetscReal OffsetZ          = 0.0;

        /* Solver grid */
        /// Whether to use a direct (factorization) solve instead of iterative.
        inline static PetscBool UseDirectSolve    = PETSC_FALSE;
        /// Target eigenvalue for the SLEPc eigensolver.
        inline static PetscReal TargetEigenvalue  = 0.0;

        /* Temporal grid */
        /// Time step dt (atomic units).
        inline static  PetscReal TimeStep         = 0.01;
        /// Final propagation time (atomic units).
        inline static PetscReal FinalTime         = 100.0;

        /* Field and envelope */
        /// Peak field amplitude (atomic units).
        inline static PetscReal Amplitude         = 0.0;
        /// Carrier frequency omega (atomic units; 0.056954 ~ 800 nm).
        inline static PetscReal Omega             = 0.056954; // 800 nm
        /// Carrier-envelope phase (radians).
        inline static PetscReal Phase             = 0.0;
        /// Pulse duration (atomic units).
        inline static PetscReal PulseDuration     = 0.0;
        /// Pulse center time (atomic units).
        inline static PetscReal PulseCenter       = 0.0;

        // generalization
        /// x-component peak amplitude.
        inline static PetscReal Ampx              = 0.0;
        /// y-component peak amplitude.
        inline static PetscReal Ampy              = 0.0;
        /// z-component peak amplitude.
        inline static PetscReal Ampz              = 0.0;
        /// x-component carrier-envelope phase.
        inline static PetscReal CEPx              = 0.0;
        /// y-component carrier-envelope phase.
        inline static PetscReal CEPy              = 0.0;
        /// z-component carrier-envelope phase.
        inline static PetscReal CEPz              = 0.0;
        /// x-component carrier frequency.
        inline static PetscReal Omegax            = 0.0;
        /// y-component carrier frequency.
        inline static PetscReal Omegay            = 0.0;
        /// z-component carrier frequency.
        inline static PetscReal Omegaz            = 0.0;

        // laser expressions
        /// muParser expression string for the x-component laser field.
        inline static std::string LaserX;
        /// muParser expression string for the y-component laser field.
        inline static std::string LaserY;
        /// muParser expression string for the z-component laser field.
        inline static std::string LaserZ;

        /// Whether to perform time propagation (default true).
        inline static PetscBool             EnablePropagation  = PETSC_TRUE;
        /// Whether to enable GPU acceleration (default false).
        inline static PetscBool             EnableGPU          = PETSC_FALSE;
        /// Whether to enable a complex absorbing potential (default false).
        inline static PetscBool             EnableCAP          = PETSC_FALSE;
        /// Whether to compute Lz angular-momentum diagnostics (default false).
        inline static PetscBool             EnableLzDiag      = PETSC_FALSE;
        /// Inner edge of the CAP region (atomic units).
        inline static PetscReal             CAPKmin           = 0.1;
        /// Output stride (in steps) for wavefunction snapshots.
        inline static PetscInt              OutputStrideWFS       = 100;
        /// Output stride (in steps) for time-series diagnostics.
        inline static PetscInt              OutputStrideTS       = 0;
        /// Output stride (in steps) for autocorrelation diagnostics.
        inline static PetscInt              OutputStrideAC       = 0;
        /// Whether to apply HDF5 chunk compression (default false).
        inline static PetscBool             HDF5Compress         = PETSC_FALSE;
        /// HDF5 compression level (0-9, default 6).
        inline static PetscInt              HDF5CompressLevel    = 6;
        /// Number of bound states to compute.
        inline static PetscInt              NBoundStates       = 1;
        /// Whether to save bound states to disk (default false).
        inline static PetscBool             NBoundStatesSave   = PETSC_FALSE;
        // Format of the bound states written to EigenData_*.h5:
        //   "complex" -> raw SLEPc eigenvectors (may carry an arbitrary global phase)
        //   "real"    -> each state phase-rotated to a real canonical form
        //                (TDSEZMakeStateReal2) before saving
        /// Storage format for bound states: "complex" or "real".
        inline static std::string           BoundStateFormat   = "complex";
        // Initial state for propagation / autocorrelation baseline.
        //   "ground"                       -> 0th converged bound state (default)
        //   "state:N"                      -> Nth converged bound state (0-based)
        //   "sup: a*N + b*M [+ c*P]"       -> coherent superposition of up to 3
        //                                     bound states (indices N,M,P; real coeffs)
        /// Initial-state selector string ("ground", "state:N", or "sup: ...").
        inline static std::string           InitialState       = "ground";
        // Parsed in ValidateOrThrow:
        //   0 = ground (== state:0), 1 = single state:InitialStateIndex,
        //   2 = superposition over InitialStateTerms.
        /// Parsed initial-state mode: 0=ground, 1=single, 2=superposition.
        inline static PetscInt              InitialStateMode   = 0;
        /// Index of the selected bound state (mode 1).
        inline static PetscInt              InitialStateIndex  = 0;
        // (index, coefficient) pairs for mode 2 (max 3 entries).
        /// (index, coefficient) pairs for the superposition (mode 2, max 3).
        inline static std::vector<std::pair<PetscInt,PetscReal>> InitialStateTerms;
        // Normalize the assembled initial state so <psi|M|psi> = 1 (default on).
        /// Whether to normalize the initial state so <psi|M|psi> = 1.
        inline static PetscBool             NormalizeInitialState = PETSC_TRUE;
        /// Declarative output enable list: dipole,population,energy,current,autocorrelation,wfs (or "all").
        inline static std::string           PhysicsOutput      = "all";  // declarative enable list: dipole,population,energy,current,autocorrelation,wfs (or "all")
        /// Whether dipole diagnostics are enabled.
        inline static PetscBool             out_dipole         = PETSC_TRUE;
        /// Whether population diagnostics are enabled.
        inline static PetscBool             out_population     = PETSC_TRUE;
        /// Whether energy diagnostics are enabled.
        inline static PetscBool             out_energy         = PETSC_TRUE;
        /// Whether current diagnostics are enabled.
        inline static PetscBool             out_current        = PETSC_TRUE;
        /// Whether autocorrelation diagnostics are enabled.
        inline static PetscBool             out_autocorr       = PETSC_TRUE;
        /// Whether wavefunction snapshots are enabled.
        inline static PetscBool             out_wfs            = PETSC_TRUE;
        // Save the assembled dipole operator matrix Dx to a PETSc binary file
        // (MatView -> *.bin, reloadable with MatLoad). Off by default. At least
        // for 1D for now; Dx is only non-NULL when the polarisation includes x.
        /// Whether to save the dipole operator matrix to a PETSc binary file.
        inline static PetscBool             SaveDipoleMatrix   = PETSC_FALSE;
        // Which dipole operator axes to persist (PETSc binary, one file each).
        // Empty by default: when SaveDipoleMatrix=1 the file-level fallback
        // saves Dx only (1D-friendly). Set from the command line via
        //   -save_dipole x | xy | xz | xyz | all | none
        // to override at runtime. Only axes that were actually assembled
        // (polarisation includes them) are written.
        /// Which dipole axes to persist ("x", "xy", "xyz", "all", etc.).
        inline static std::string           SaveDipoleAxes     = "";
        // t-SURFF (time-dependent surface-flux photoelectron momentum spectrum)
        /// Whether t-SURFF photoelectron spectrum computation is enabled.
        inline static PetscBool             out_tsurff         = PETSC_FALSE;
        /// Momentum grid points per axis in [-SurffKmax, SurffKmax].
        inline static PetscInt              SurffNk            = 40;   // momentum grid points per axis in [-SurffKmax,SurffKmax]
        /// Momentum cutoff (atomic units).
        inline static PetscReal             SurffKmax          = 2.0;  // momentum cutoff (a.u.)
        /// Fold boundary flux into b(k) every N accepted steps (1 = exact).
        inline static PetscInt              OutputStrideSurff  = 1;    // fold boundary flux into b(k) every N accepted steps (1 = exact)
        /// Sign of the k-A(t) coupling in t-SURFF (+1 or -1).
        inline static PetscReal             SurffCouplingSign  = +1.0; // k_eff = k + s*q*A(t); flip to -1.0 if your field couples as +q*E*r
        /// If >0, restrict t-SURFF faces to the shell |r| > SurffRadius (a.u.).
        inline static PetscReal             SurffRadius        = 0.0;  // if >0, restrict t-SURFF faces to the shell |r|>SurffRadius (a.u.)
        /// Envelope type string ("custom", etc.).
        inline static std::string           EnvelopeType       = "custom";
        /// Knot-sequence type string ("uniform", "symexp", etc.).
        inline static std::string           KnotSequence       = "uniform";
        // Per-axis knot sequences parsed from the single "KnotSequence" key.
        // "uniform"                  -> all dims uniform
        // "uniform, uniform, uniform"-> all dims uniform
        // "symexp, uniform, symtanu" -> per-dim (X=symexp, Y=uniform, Z=symtanu)
        /// Per-axis knot-sequence type strings (one per spatial dimension).
        inline static std::string           KnotSeq[3]          = {"uniform","uniform","uniform"};
        /// Exponential grading parameter for knot sequences.
        inline static PetscReal             KnotAlpha        = 0.0;
        // Boundary condition at the box wall (declarative input-file option).
        //   "Neumann"   / "natural" -> reflecting wall (free Galerkin, default)
        //   "Dirichlet"  / "wall"    -> homogeneous psi=0 enforced at the wall
        // Chosen in core.cpp: Dirichlet routes H assembly through
        // TDSEZCompOperatorsDirichlet (applies IGASetBoundaryValue).
        /// Boundary condition type: "Neumann" (reflecting) or "Dirichlet" (wall).
        inline static std::string           BoundaryType     = "Neumann";

        // Per-axis hydrogenic knot-region counts (only used when an axis uses
        // KnotSequence = hydrogenic). Each region is the half-axis [0, Lmax]:
        //   HydrogenicNLin[d] -> # linear knots in the near-origin region
        //                        (knot spacing r1 = 0.1 a.u.; r_cross = n_lin*r1)
        //   HydrogenicNExp[d] -> # exponential-graded knots from r_cross to Lmax
        // Given as a comma list like "Nelements" (all axes) or "40,40,40".
        // A value of 0 means "use (Nelements-1)" so omitting the key preserves
        // the original behaviour. Defaults match the previous hard-coded values.
        /// Per-axis count of linear knots in the near-origin hydrogenic region.
        inline static PetscInt              HydrogenicNLin[3]   = {0, 0, 0};
        /// Per-axis count of exponential-graded knots from r_cross to Lmax.
        inline static PetscInt              HydrogenicNExp[3]   = {0, 0, 0};
        // Per-axis near-origin knot spacing (a.u.) for the hydrogenic sequence.
        // r_cross = HydrogenicNLin[d] * HydrogenicR1[d] must stay < Lmax or the
        // generator throws (caught as a clean TDSEZ FATAL). Default 0.1 matches
        // the previous hard-coded value; 0 means "use 0.1".
        /// Per-axis near-origin knot spacing for the hydrogenic sequence (a.u.).
        inline static PetscReal             HydrogenicR1[3]     = {0.0, 0.0, 0.0};

        // Adaptive (KnotSequence = adaptive) tuning. Single value, all axes.
        //   AdaptiveKappa -> weight of the |grad V| term (default 1.0)
        //   AdaptivePower -> sharpening exponent on the importance weight
        //                   (default 1.0; >1 concentrates knots more tightly
        //                   on the most important regions)
        /// Weight of the |grad V| term in the adaptive knot indicator.
        inline static PetscReal             AdaptiveKappa      = 1.0;
        /// Sharpening exponent on the adaptive importance weight.
        inline static PetscReal             AdaptivePower      = 1.0;
        // adaptive_wf (two-pass) coarse-bootstrap resolution: knots/axis on the
        // cheap preliminary ground-state solve. Default 20 is plenty to capture
        // the density SHAPE (which is all the re-mesh needs). 0 => auto (=20).
        /// Coarse-bootstrap resolution (knots/axis) for the adaptive_wf two-pass.
        inline static PetscInt              AdaptiveWFCoarseN   = 20;
        // Weight of the kinetic-energy-density term in the adaptive_wf knot
        // indicator:  w(x) = |psi|^2 + AdaptiveWFKinLambda * |grad psi|^2.
        // Raw |psi|^2 alone starves the Coulomb cusps (density peaks BETWEEN
        // nuclei) and the exponential tails; |grad psi|^2 fixes both because the
        // gradient diverges at the cusps and is non-zero in the decay region.
        // Default 3.0. 0.0 => pure density (legacy, worse for bound states).
        // lambda>=3 eliminates the high-DOF over-clustering regression on
        // multi-centre potentials, making adaptive_wf best at all DOF.
        /// Weight of the kinetic-energy-density term in the adaptive_wf indicator.
        inline static PetscReal             AdaptiveWFKinLambda = 3.0;

        /// Number of quadrature points per element.
        inline static PetscInt  NQuadratures      = 8;
        /// CAP strength parameter Gamma.
        inline static PetscReal Gamma             = 5.0;

        /* Expressions */
        /// muParser expression string for the scalar potential V.
        inline static std::string Potential;
        /// muParser expression string for dV/dx.
        inline static std::string PotentialDerivativeX;
        /// muParser expression string for dV/dy.
        inline static std::string PotentialDerivativeY;
        /// muParser expression string for dV/dz.
        inline static std::string PotentialDerivativeZ;
        /// muParser expression string for the total laser field E(t).
        inline static std::string Laser;
        /// muParser expression string for the envelope F(t).
        inline static std::string Envelope;

        /// Polarization direction string ("x", "y", "z", "xy", "xyz", etc.).
        // read polarization direction
        inline static std::string Polarization;


        /* Parser instances and bound vars */
        /// muParser instances for V, dV/dx, dV/dy, dV/dz, E, Ex, Ey, Ez, envelope, mass, and inverse-mass gradients.
        inline static mu::Parser VPot, dVPotX, dVPotY, dVPotZ, Field, LaserXexpr, LaserYexpr, LaserZexpr, Env, MassExpr, dinvMassExprx, dinvMassExpry, dinvMassExprz;
        /// Bound coordinate variables (x, y, z, t) shared across all parser instances.
        inline static PetscReal  varX_ = 0.0, varY_ = 0.0, varZ_ = 0.0, varT_ = 0.0;

        /* User-defined muParser constants from the input file (Variables key).
           These are bound on every parser instance in initParsers() so they can
           be referenced by name inside any expression, e.g.
             Variables = V0=1.2, w=0.5
             Potential = 0.5*(x*x+y*y) + V0*exp(-w*x)
           This lets users introduce arbitrary named parameters without the
           solver predefining each one. */
        /// User-defined muParser constants from the input file (Variables key).
        inline static std::unordered_map<std::string, PetscReal> userConsts_;
        /**
         * @brief Trim leading and trailing whitespace from a string (in place).
         * @param s String to trim.
         */
        static void trim(std::string &s);

        // Coordinate setter for analytic potential/density evaluation. Public so
        // knot-sequence generators (core_knots.cpp) can sample V and dV/daxis.
        /**
         * @brief Set the x coordinate (y, z reset to 0).
         * @param x Cartesian x coordinate.
         */
        inline static void setVars(PetscReal x) { varX_ = x; varY_ = varZ_ = 0.0; }
        /**
         * @brief Set the x and y coordinates (z reset to 0).
         * @param x Cartesian x coordinate.
         * @param y Cartesian y coordinate.
         */
        inline static void setVars(PetscReal x, PetscReal y) { varX_ = x; varY_ = y; varZ_ = 0.0; }
        /**
         * @brief Set the x, y, and z coordinates.
         * @param x Cartesian x coordinate.
         * @param y Cartesian y coordinate.
         * @param z Cartesian z coordinate.
         */
        inline static void setVars(PetscReal x, PetscReal y, PetscReal z) { varX_ = x; varY_ = y; varZ_ = z; }
        /**
         * @brief Set coordinates from a vector (up to 3 components; missing = 0).
         * @param vars Vector of coordinate values.
         */
        inline static void setVars(const std::vector<PetscReal>& vars) {
            varX_ = (vars.size() > 0) ? vars[0] : 0.0;
            varY_ = (vars.size() > 1) ? vars[1] : 0.0;
            varZ_ = (vars.size() > 2) ? vars[2] : 0.0;
        }
};

#endif // TDSEZ_PARSER_HPP
