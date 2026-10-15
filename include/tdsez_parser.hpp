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
         * @brief Carrier sign s = -q of the potential and field terms in H:
         *        H = K + s (V + r.E). q = Charge = -1 (electron, default) gives
         *        s = +1, the Potential and field as written; q = +1 (hole)
         *        inverts both.
         */
        inline static PetscReal CarrierSign() { return -Q; }
        /**
         * @brief Potential energy of the carrier, s V(r), at the given
         *        coordinates (V = Potential expression, s = CarrierSign()).
         * @tparam Args Coordinate values (1D, 2D, or 3D).
         * @param args Coordinates to set before evaluating the potential.
         * @return s V(x[,y[,z]]) in atomic units.
         */
        template<typename... Args>
        inline static PetscReal V(Args... args)     { setVars(args...) ; return CarrierSign() * VPot.Eval(); }
        /**
         * @brief Evaluate dV/dx at the given coordinates.
         * @tparam Args Coordinate values.
         * @param args Coordinates to set before evaluating.
         * @return s dV/dx in atomic units (s = CarrierSign()).
         */
        template<typename... Args>
        inline static PetscReal dVx(Args... args)      { setVars(args...); return CarrierSign() * dVPotX.Eval(); }
        /**
         * @brief Evaluate dV/dy at the given coordinates.
         * @tparam Args Coordinate values.
         * @param args Coordinates to set before evaluating.
         * @return s dV/dy in atomic units (s = CarrierSign()).
         */
        template<typename... Args>
        inline static PetscReal dVy(Args... args)      { setVars(args...); return CarrierSign() * dVPotY.Eval(); }
        /**
         * @brief Evaluate dV/dz at the given coordinates.
         * @tparam Args Coordinate values.
         * @param args Coordinates to set before evaluating.
         * @return s dV/dz in atomic units (s = CarrierSign()).
         */
        template<typename... Args>
        inline static PetscReal dVz(Args... args)      { setVars(args...); return CarrierSign() * dVPotZ.Eval(); }
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
        /// Record dipole/current/acceleration on every spatial axis (default),
        /// or only on the coupled axes (ObserveAllAxes = 0, saves memory).
        inline static PetscBool ObserveAllAxes = PETSC_TRUE;
        /// Whether axis a (0, 1, 2) is observed (its operators are assembled).
        inline static bool Observed(int a) { return a < (int)Dimension && (ObserveAllAxes || Coupled(a)); }
        /// Whether axis a (0, 1, 2) is coupled to the field (in Polarization).
        inline static bool Coupled(int a) { return Polarization.find("xyz"[a]) != std::string::npos; }
        /// Physical field E_a(t) on axis a: the LaserX/Y/Z expression on a
        /// coupled axis, 0 on the others.
        inline static PetscReal Field(int a, PetscReal t)
        {
            if (!Coupled(a)) return 0.0;
            return a == 0 ? Ex(t) : (a == 1 ? Ey(t) : Ez(t));
        }
        /// Field term as it enters H, s E_a(t) (s = CarrierSign()).
        inline static PetscReal Coupling(int a, PetscReal t) { return CarrierSign() * Field(a, t); }
        /**
         * @brief Evaluate the position-dependent mass distribution M(x,y,z).
         * @param x Cartesian x coordinate.
         * @param y Cartesian y coordinate.
         * @param z Cartesian z coordinate.
         * @return Mass M(x,y,z) in atomic units.
         */
        inline static PetscReal MassDist(PetscReal x, PetscReal y, PetscReal z) { setVars(x, y, z); return MassExpr.Eval(); }

        // private:
        /// Mathematical constant pi.
        static constexpr PetscReal PI     = 3.14159265358979323846;
        /// Position-dependent mass expression string (default "1.0").
        inline static std::string Mass    = "1.0";
        /// Whether the mass is spatially constant (set in ValidateOrThrow).
        inline static PetscBool MassIsConstant = PETSC_FALSE;


        /// Carrier charge q in units of e: -1 electron (default), +1 hole.
        /// Scales the reported dipole, current and acceleration, and sets
        /// the sign of the potential and field terms (CarrierSign()).
        inline static PetscReal Q         = -1.0;

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
        // Domain bounds per axis. Domain = lo, hi sets all axes;
        // Domain = [lox,hix],[loy,hiy],[loz,hiz] sets them separately.
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

        /* Field parameters: named constants for the field expressions */
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
        /// Width (a.u.) of the absorbing layer at each wall. > 0 sets the layer
        /// directly and CAPKmin = 2.62206/(2*0.2*CAPWidth) = 6.555/CAPWidth
        /// (the slowest momentum absorbed without reflection) follows from it;
        /// 0 (default): the width follows from CAPKmin.
        inline static PetscReal             CAPWidth          = 0.0;
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
        //   "analytic:quesne01"             -> projected exact Quesne n=0,1 state
        //   "analytic:massstep01"           -> projected exact sharp-interface state
        /// Initial-state selector string ("ground", "state:N", "sup: ...", or analytic benchmark).
        inline static std::string           InitialState       = "ground";
        /// Dipole-acceleration method: "auto" (discrete when the mass is
        /// position dependent, operator otherwise), "operator" (Ehrenfest
        /// force operator, needs smooth m and V), or "discrete"
        /// (a = d<v>/dt from the Galerkin equation of motion).
        inline static std::string           AccelerationMethod = "auto";
        /// Time propagator: "cn" (Crank-Nicolson, Pade(1,1), PETSc TS; default),
        /// "pade22" (exponential midpoint rule with a Pade(2,2) exponential) or
        /// "magnus4" (commutator-free 4th-order Magnus, Pade(2,2) exponentials).
        inline static std::string           Propagator         = "cn";
        /// Preconditioner of the propagator solves: "asm" (additive Schwarz,
        /// ILU(0) subdomains) or "mg" (geometric B-spline multigrid).
        inline static std::string           Preconditioner     = "asm";
        /// Basis of every axis: "bspline" (default, C^{p-1} across simple
        /// knots) or the C0 element bases "bernstein", "lagrange", "spectral"
        /// (Lagrange at Gauss-Lobatto points); for these every interior knot
        /// is raised to multiplicity p.
        inline static std::string           Basis              = "bspline";
        /// Quadrature rule: "legendre" (Gauss, default), "lobatto"
        /// (Gauss-Lobatto, includes the element ends) or "reduced" (PetIGA's
        /// reduced B-spline rule).
        inline static std::string           QuadratureRule     = "legendre";
        /// Multigrid levels (0 = coarsen until about 2000 DOFs) and smoothing
        /// sweeps (Richardson with ASM/ILU(0)) before and after each coarse correction.
        inline static PetscInt              MGLevels           = 0;
        inline static PetscInt              MGSmooth           = 1;
        // Parsed in ValidateOrThrow:
        //   0 = ground (== state:0), 1 = single state:InitialStateIndex,
        //   2 = superposition, 3 = analytic Quesne, 4 = analytic mass step.
        /// Gaussian wave packet (InitialState = wavepacket), per axis:
        /// psi0 = prod_d (2 pi w_d^2)^(-1/4) exp(-(x_d-c_d)^2/(4 w_d^2) + i k_d x_d),
        /// i.e. |psi0|^2 has standard deviation w_d (WavepacketWidth).
        inline static PetscReal             WavepacketCenter[3]   = {0.0, 0.0, 0.0};
        inline static PetscReal             WavepacketMomentum[3] = {0.0, 0.0, 0.0};
        inline static PetscReal             WavepacketWidth[3]    = {1.0, 1.0, 1.0};

        // ---- t-SURFF photoelectron spectra (computed only with TSurff = 1) ----
        /// Enable the t-SURFF observer (output: td/surff_<input>.h5).
        inline static PetscBool             TSurff             = PETSC_FALSE;
        /// Half-width R of the extraction box |x_d| = R.
        inline static PetscReal             SurffRadius        = 0.0;
        /// Momentum grid per axis: SurffNk points in [SurffKmin, SurffKmax]
        /// (SurffKmin defaults to -SurffKmax). Ignored when SurffMomenta is set.
        inline static PetscReal             SurffKmin[3]       = {0.0, 0.0, 0.0};
        inline static PetscReal             SurffKmax[3]       = {0.0, 0.0, 0.0};
        inline static PetscInt              SurffNk[3]         = {0, 0, 0};
        inline static PetscBool             SurffKminSet       = PETSC_FALSE;
        /// Optional text file of momenta (one kx [ky [kz]] row each), instead of the grid.
        inline static std::string           SurffMomenta;
        /// Sample the surface flux every N steps (1 = every step).
        inline static PetscInt              OutputStrideSurff  = 1;
        /// Gauss points per element along the surface (0 = NQuadratures).
        inline static PetscInt              SurffQuadrature    = 0;
        /// Exterior homogeneity tolerances (relative mass, absolute potential in Ha).
        inline static PetscReal             SurffMassTolerance      = 1e-6;
        inline static PetscReal             SurffPotentialTolerance = 1e-6;
        /// Add the projection of psi(FinalTime) inside the box on the Volkov
        /// states (exact for a free carrier at any FinalTime; it contains the
        /// bound states' plane-wave content, so keep 0 for bound systems).
        inline static PetscBool             SurffInterior      = PETSC_FALSE;

        /// Parsed initial-state mode: 0=ground, 1=single, 2=superposition, 3=Quesne, 4=mass step, 5=wave packet.
        inline static PetscInt              InitialStateMode   = 0;
        /// Index of the selected bound state (mode 1).
        inline static PetscInt              InitialStateIndex  = 0;
        // (index, coefficient) pairs for mode 2 (max 3 entries).
        /// (index, coefficient) pairs for the superposition (mode 2, max 3).
        /// Coefficients may be complex: "a", "a+ib", "a-ib", "b*i" (a,b real).
        inline static std::vector<std::pair<PetscInt,PetscScalar>> InitialStateTerms;
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
        /// Per-axis largest element width (a.u.) of the hydrogenic sequence:
        /// where the exponential grading would exceed it, the rest of the
        /// half-axis is filled with uniform elements no wider than it.
        /// 0 (default): derived from the field when propagating with a laser
        /// (pi/(2 k_max)), else no cap; < 0: no cap.
        inline static PetscReal             HydrogenicHMax[3]   = {0.0, 0.0, 0.0};

        /// Number of quadrature points per element.
        inline static PetscInt  NQuadratures      = 8;

        /* Expressions */
        /// muParser expression string for the scalar potential V.
        inline static std::string Potential;
        /// muParser expression string for dV/dx.
        inline static std::string PotentialDerivativeX;
        /// muParser expression string for dV/dy.
        inline static std::string PotentialDerivativeY;
        /// muParser expression string for dV/dz.
        inline static std::string PotentialDerivativeZ;
        /// Field E(t) used for every polarised axis without its own LaserX/Y/Z
        /// (copied into them by initParsers()).
        inline static std::string Laser;

        /// Polarization direction string ("x", "y", "z", "xy", "xyz", etc.).
        // read polarization direction
        inline static std::string Polarization;


        /* Parser instances and bound vars */
        /// muParser instances for V, dV/dx, dV/dy, dV/dz, Ex, Ey, Ez and the mass.
        inline static mu::Parser VPot, dVPotX, dVPotY, dVPotZ, LaserXexpr, LaserYexpr, LaserZexpr, MassExpr;
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
