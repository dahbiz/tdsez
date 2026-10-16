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
 * exponential/tangent, logarithmic-tangent, interface-aligned, hydrogenic),
 * assembles and solves the
 * time-independent Schrödinger equation via SLEPc, and stores the resulting
 * bound states and energies.
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
        /// Number of kept eigenpairs (NBoundStates, extended to the end of a
        /// degenerate shell) from the last solve.
        PetscInt nconv = 0;
        /// Canonical eigenpairs (TDSEZCanonicalEigenpairs in core.cpp); every
        /// consumer reads these instead of the raw SLEPc vectors.
        std::vector<Vec> eigVecs;
        std::vector<PetscReal> eigVals;
        std::vector<PetscReal> eigRes_;   ///< backward error of each kept eigenpair

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
         * @param axis Axis letter for the log.
         * @return Vector of knot coordinates.
         */
        static std::vector<PetscReal> TDSEZHydrogenicKnots(PetscReal Lmax, PetscInt n_lin, PetscReal r1, PetscInt n_exp, PetscInt p,
                                                           PetscReal hmax = 0.0, char axis = 'x');
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
