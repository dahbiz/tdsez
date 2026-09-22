#ifndef TDSEZ_CORE_HPP
#define TDSEZ_CORE_HPP

#include <slepc.h>
#include <petiga.h>
#include <string>
#include <vector>

class TDSEZAssembler;

// ============================================================================
//  TDSEZCore  — IGA setup, ground-state solve, eigenstates
// ============================================================================
/**
 * @brief IGA setup, ground-state solver, and eigenstate computation.
 * @details Creates the IGA context, generates knot vectors (uniform, symmetric
 * exponential/tangent, logarithmic-tangent, interface-aligned, hydrogenic,
 * adaptive potential- or density-driven), assembles and solves the
 * time-independent Schrödinger equation via SLEPc, and stores the resulting
 * bound states and energies. Also supports the two-pass adaptive_wf
 * density-driven mesh via BootstrapDensity/SampleDensity.
 */
class TDSEZCore
{
    /// @cond PRIVATE
    friend class TDSEZAssembler;
    /// Context passed into IGA form kernels, holding a back-pointer to this instance.
    struct FormContext {
        TDSEZCore* self;  // Pointer to the class instance
    };
    /// @endcond


    public:
        /**
         * @brief Construct the core from an input file path.
         * @param input Path to the input parameter file.
         */
        TDSEZCore(const std::string &input);

        /**
         * @brief Destructor; releases IGA, EPS, matrices, and vectors.
         */
        // Destructor
        ~TDSEZCore();


        // MPI info
        /// MPI rank of this process.
        PetscMPIInt rank = 0, size = 1;

        /// Input parameter file path.
        // input file
        std::string inputFile;

        /// Isogeometric analysis context.
        // IGA object
        IGA iga = PETSC_NULLPTR;
        /// SLEPc eigensolver context.
        EPS eps = PETSC_NULLPTR;
        /// Number of converged eigenpairs from the last solve.
        PetscInt nconv = 0;

        // methods
        /**
         * @brief Generate a symmetric exponential knot vector.
         * @param Lmin Lower domain bound.
         * @param Lmax Upper domain bound.
         * @param ninterior Number of interior knots.
         * @param alpha Exponential grading parameter.
         * @param p Spline degree.
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZExpSymKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscReal alpha, PetscInt p);
        /**
         * @brief Generate a symmetric tangent (unbounded) knot vector.
         * @param Lmin Lower domain bound.
         * @param Lmax Upper domain bound.
         * @param ninterior Number of interior knots.
         * @param beta Tangent grading parameter.
         * @param p Spline degree.
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZTanUSymKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscReal beta, PetscInt p);
        /**
         * @brief Generate a symmetric tangent (bounded) knot vector.
         * @param Lmin Lower domain bound.
         * @param Lmax Upper domain bound.
         * @param ninterior Number of interior knots.
         * @param beta Tangent grading parameter.
         * @param p Spline degree.
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZTanNSymKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscReal beta, PetscInt p);
        /**
         * @brief Generate a logarithmic-tangent knot vector.
         * @param Lmin Lower domain bound.
         * @param Lmax Upper domain bound.
         * @param ninterior Number of interior knots.
         * @param alpha Grading parameter.
         * @param p Spline degree.
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZLogTanKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscReal alpha, PetscInt p);
        /**
         * @brief Generate interface-aligned knot vector from potential/mass expressions.
         * @param Lmin Lower domain bound.
         * @param Lmax Upper domain bound.
         * @param ninterior Number of interior knots.
         * @param p Spline degree.
         * @param expr Potential expression string for breakpoint extraction.
         * @param massexpr Mass expression string for breakpoint extraction.
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZInterfaceKnots(PetscReal Lmin, PetscReal Lmax, PetscInt ninterior, PetscInt p, const std::string& expr, const std::string& massexpr);
        /**
         * @brief Generate a hydrogenic knot vector (linear near origin, exponential tail).
         * @param Lmax Upper domain bound (half-axis [0, Lmax]).
         * @param n_lin Number of linear knots near the origin.
         * @param r1 Near-origin knot spacing (a.u.).
         * @param n_exp Number of exponential-graded knots from r_cross to Lmax.
         * @param p Spline degree.
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZHydrogenicKnots(PetscReal Lmax, PetscInt n_lin, PetscReal r1, PetscInt n_exp, PetscInt p);
        // Adaptive (potential-driven, single-pass) knot vector: clusters knots
        // where the potential is "interesting" (steep gradient + deep wells) by
        // CDF-inverting a tanh-compressed importance weight. axis selects which
        // derivative parser to use (0->x,1->y,2->z); kappa weights the gradient
        // term, power sharpens the weight. See core_knots.cpp for details.
        /**
         * @brief Generate an adaptive (potential-driven, single-pass) knot vector.
         * @details Clusters knots where the potential is "interesting" (steep
         * gradient + deep wells) by CDF-inverting a tanh-compressed importance
         * weight.
         * @param Lmin Lower domain bound.
         * @param Lmax Upper domain bound.
         * @param ninterior Number of interior knots.
         * @param p Spline degree.
         * @param axis Which derivative parser to use (0=x, 1=y, 2=z).
         * @param kappa Weight of the |grad V| term (default 1.0).
         * @param power Sharpening exponent on the importance weight (default 1.0).
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZAdaptiveKnots(PetscReal Lmin, PetscReal Lmax,
            PetscInt ninterior, PetscInt p, PetscInt axis,
            PetscReal kappa = 1.0, PetscReal power = 1.0);
        // Adaptive (density-driven, two-pass) knot vector: clusters knots where
        // the ELECTRON DENSITY is large. rho_axis is a precomputed 1D marginal
        // |psi0|^2 (obtained from a cheap coarse ground-state solve — see
        // BootstrapDensity). CDF-inverts the density so knots concentrate where
        // the wavefunction lives. Same clamped p+1 boundary convention.
        /**
         * @brief Generate an adaptive (density-driven, two-pass) knot vector.
         * @details Clusters knots where the electron density |psi0|^2 is large,
         * CDF-inverting a precomputed 1D marginal density.
         * @param Lmin Lower domain bound.
         * @param Lmax Upper domain bound.
         * @param ninterior Number of interior knots.
         * @param p Spline degree.
         * @param rho_axis Precomputed 1D marginal density |psi0|^2.
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZAdaptiveWFKnots(PetscReal Lmin, PetscReal Lmax,
            PetscInt ninterior, PetscInt p,
            const std::vector<PetscReal>& rho_axis);
        // Auto interface-aligned knots: extract breakpoints from a piecewise
        // (?:) Potential/Mass expression and pin a knot line there, grading the
        // rest by a CDF that clusters resolution at the material boundaries.
        // (the 6-arg overload with massexpr is the one actually defined and called)

        /**
         * @brief Assemble the Hamiltonian and mass matrices on the IGA.
         * @return PetscErrorCode (0 on success).
         */
        PetscErrorCode Assemble();
        /**
         * @brief Solve the time-independent Schrödinger equation for bound states.
         * @return PetscErrorCode (0 on success).
         */
        PetscErrorCode Solve();
        /**
         * @brief Output eigenstates and energies to disk.
         * @return PetscErrorCode (0 on success).
         */
        PetscErrorCode Output();

        // ── adaptive_wf (two-pass density-driven) support ──────────────
        // BootstrapDensity(): if any axis uses KnotSequence = adaptive_wf, run a
        // CHEAP coarse ground-state solve (uniform mesh, AdaptiveWFCoarseN
        // knots/axis), sample the 1D marginal electron density |psi0|^2 onto
        // rho_x_/rho_y_/rho_z_, then tear down the coarse IGA/matrices. The real
        // knot build below then consumes these marginals. No-op otherwise.
        /**
         * @brief Bootstrap the 1D marginal density for adaptive_wf meshing.
         * @details If any axis uses KnotSequence = adaptive_wf, runs a cheap
         * coarse ground-state solve, samples |psi0|^2 onto rho_x_/rho_y_/rho_z_,
         * then tears down the coarse IGA/matrices. No-op otherwise.
         * @return PetscErrorCode (0 on success).
         */
        PetscErrorCode BootstrapDensity();
        // Sample |psi|^2 on the coarse volume grid and marginalise to 1D
        // histograms per axis (normalised to unit integral) into rho_*_.
        /**
         * @brief Sample |psi|^2 and marginalise to 1D per-axis density histograms.
         * @param iga Coarse IGA context to sample on.
         * @param psi Wavefunction vector to sample.
         * @return PetscErrorCode (0 on success).
         */
        PetscErrorCode SampleDensity(IGA iga, Vec psi);

        // friend classes
        /// @cond PRIVATE
        friend class TDSEZAssembler;
        /// @endcond

        /// Hamiltonian (stiffness) matrix H and mass matrix M.
        // stiffness and MASS matrices
        Mat H = PETSC_NULLPTR, M = PETSC_NULLPTR;

        // knots
        /// Knot vectors along x, y, and z.
        std::vector<PetscReal> knots_x, knots_y, knots_z;

        // adaptive_wf: 1D marginal electron densities |psi0|^2, filled by
        // BootstrapDensity() (one vector per axis, normalised to unit integral).
        /// 1D marginal electron densities |psi0|^2 per axis (adaptive_wf).
        std::vector<PetscReal> rho_x_, rho_y_, rho_z_;

        /// Initial state vector psi(0).
        // initial state
        Vec initialPsi = PETSC_NULLPTR;

        /// Converged bound-state vectors.
        // bound states and populations
        std::vector<Vec> boundstates;
        /// Corresponding eigen-energies (atomic units).
        std::vector<PetscReal> energies;
};


#endif // TDSEZ_CORE_HPP
