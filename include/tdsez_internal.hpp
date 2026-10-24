#ifndef TDSEZ_INTERNAL_HPP
#define TDSEZ_INTERNAL_HPP

/**
 * @file tdsez_internal.hpp
 * @brief Shared solver types and cross-module function declarations.
 * @author TDSEZ Project
 *
 * @details Includes the parser, printer, assembler, core, manager, and
 * propagator interfaces. The declarations below are shared by assembly,
 * diagnostics, HDF5 monitoring, and time propagation.
 */

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

#include "tdsez_info.hpp"

#include <petscviewerhdf5.h>
#include <H5Epublic.h>
#include <petscmat.h>
#include <algorithm>
#include <functional>
#include <petscblaslapack.h>

#include "muParser.h"

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
/// Forward declaration of the IGA core / ground-state solver class.
class TDSEZCore;
/// Forward declaration of the linear-operator assembler class.
class TDSEZAssembler;
/// Forward declaration of the time-stepping propagator class.
class TDSEZPropagator;
/// Forward declaration of the propagation/diagnostics orchestrator class.
class TDSEZManager;

// TS callback forward declarations (used by the propagator)
/// @name TS implicit-function callbacks (polarization-specific).
/// Each computes F(t,psi,psi_dot) = M*psi_dot - H(t)*psi for a given polarization.
///@{
PetscErrorCode TDSEZIFunctionPolX   (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolY   (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolZ   (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolXY  (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolXZ  (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolYZ  (TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode TDSEZIFunctionPolXYZ (TS, PetscReal, Vec, Vec, Vec, void*);
///@}

/// @name TS implicit-Jacobian callbacks (polarization-specific).
/// Each computes the Jacobian J = shift*M - dF/dpsi for a given polarization.
///@{
PetscErrorCode TDSEZIJacobianPolX   (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolY   (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolZ   (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolXY  (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolXZ  (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolYZ  (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIJacobianPolXYZ (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode TDSEZIFunctionTerms (TS, PetscReal, Vec, Vec, Vec, void*);                       ///< with FieldTerms
PetscErrorCode TDSEZIJacobianTerms (TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);       ///< with FieldTerms
///@}

#include "tdsez_parser.hpp"

// ============================================================================
//  Shared physics structures & types
// ============================================================================
/**
 * @brief Container for first-, second-, and third-order partial derivatives.
 * @details Used by TDSEZCompDerivative to return all derivative orders of a
 * scalar function in one call. Arrays are flat (row-major) with dimension
 * implied by the input point size.
 */
struct PhysicsDerivativesFull {
    /// First-order partial derivatives (gradient), length = dim.
    std::vector<PetscReal> first;
    /// Second-order partial derivatives (Hessian), length = dim*dim.
    std::vector<PetscReal> second;
    /// Third-order partial derivatives, length = dim*dim*dim.
    std::vector<PetscReal> third;
};

/// Function pointer type for IGA point-wise physics form kernels.
typedef PetscErrorCode (*TDSEZPhysicsKernel)(
    IGAPoint p,
    PetscInt nmat,
    PetscScalar *K[],
    void *ctx);


#include "tdsez_assembler.hpp"

#include "tdsez_core.hpp"

#include "tdsez_manager.hpp"

#include "tdsez_propagator.hpp"

/**
 * @brief Run-time profile: phase wall times, PETSc log stages for the
 *        propagation breakdown, and the end-of-run timing report.
 */
struct TDSEZProfile
{
    /// Top-level phases of a run, in execution order.
    enum Phase { Setup, Assembly, Eigensolve, StaticOutput, OperatorAssembly,
                 TDSetup, Propagation, FinalOutput, NPHASES };
    PetscLogDouble phase[NPHASES] = {0};
    PetscLogDouble tStart = 0.0;
    PetscLogStage  stagePropagation = -1, stageObservables = -1;
    PetscLogEvent  eventHDF5Write = -1, eventStageUpdate = -1;
    PetscInt       steps = 0;               ///< time steps taken
    PetscBool      initialised = PETSC_FALSE;
    /// Named parts of a phase, in first-call order (same order on every rank).
    struct Part { Phase phase; std::string label; PetscLogDouble seconds; };
    std::vector<Part> parts;

    static TDSEZProfile& Get();
    /// Enable PETSc logging if needed, register stages/events, start the clock.
    PetscErrorCode Init();
    /// Accumulate wall time into a phase.
    void Add(Phase p, PetscLogDouble seconds);
    /// Accumulate wall time into a named part of a phase (collective order).
    void Sub(Phase p, const std::string &label, PetscLogDouble seconds);
    /// Print the timing table and performance summary (collective).
    PetscErrorCode Report(PetscInt dofs, PetscBool propagated);
};

// ============================================================================
//  Cross-TU free functions (prototypes)
// ============================================================================


/**
 * @brief Compute first, second, and third derivatives via 4th-order finite differences.
 * @param func Scalar function of a vector argument.
 * @param x Point at which to evaluate derivatives.
 * @param h Step size (default 1e-4).
 * @return PhysicsDerivativesFull containing gradient, Hessian, and third-derivative tensor.
 */
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

/**
 * @brief Derivatives of 1/m at x (x as the kernels pass it: 1, 2 or 3
 *        coordinates): TDSEZCompDerivative(1/MassDist, x, 1e-4), identical values.
 * @details The batched force kernels of the three axes ask for the same point
 *          one after another, so the last result is kept per thread; within one
 *          evaluation 1/m is computed once per distinct stencil point (the
 *          stencils of permuted index triples share their points).
 */
const PhysicsDerivativesFull &TDSEZInvMassDerivatives(const std::vector<PetscReal> &x);
/// As TDSEZInvMassDerivatives, but zero when the discrete acceleration is used
/// (the force matrices, the only users, are then not read).
const PhysicsDerivativesFull &TDSEZForceMassDerivatives(const std::vector<PetscReal> &x);

/// @brief Assemble the CAP from 1D matrices (W = sum_d W_d is separable; see
///        src/assembly/cap.cpp). Same matrix as the element kernel TDSEZformCap.
PetscErrorCode TDSEZAssembleSeparableCAP(IGA iga, Mat *CAP);
/// CAP scale 1/m(r) for an (effective) mass m (1 for m = 1); see TDSEZformCap.
PetscReal TDSEZCapMassScale(const PetscReal r[3]);

/// Dry-run per-rank memory estimate on nranks ranks (-tdsez_estimate_memory N); see memory_estimate.cpp.
PetscErrorCode TDSEZEstimateMemory(IGA iga, PetscInt nranks);

/// IGASetUp with a node-balanced element split (partition.cpp; -tdsez_balance 0: PetIGA's split).
PetscErrorCode TDSEZIGASetUp(IGA iga);
PetscBool      TDSEZBalanceEnabled();
PetscBool      TDSEZStageILUInPlace();     // -tdsez_stage_ilu_inplace (propagator.cpp)
PetscBool      TDSEZEpsILUInPlace();       // -tdsez_eps_ilu_inplace (propagator.cpp)
/// Element starts s_0..s_P of an axis with ~nnp/P owned nodes per part.
void TDSEZBalancedElementStarts(PetscInt p, PetscInt nel, PetscInt nnp, const PetscInt *span, PetscInt P,
                                std::vector<PetscInt> &s);

/**
 * @brief Manolopoulos complex absorbing potential profile.
 * @details Smooth CAP that is zero inside [0, x_max - Δ] and ramps up
 * smoothly to a large value at x_max, avoiding reflections.
 * @param xq Quadrature point coordinate.
 * @param x0 Unused (kept for API compatibility).
 * @param x_max Outer boundary of the CAP region.
 * @param ma_kmin CAP strength parameter.
 * @return CAP value at xq.
 */
PetscReal Manolopoulos_CAP_profile(const PetscReal xq, const PetscReal, const PetscReal x_max, const PetscReal ma_kmin);

/// @brief Assemble multiple IGA matrices from a batch form kernel (free BC).
/// @param iga IGA context.
/// @param nmat Number of matrices to assemble.
/// @param mats Output array of Mat objects (length nmat).
/// @param kernel Batch form kernel callback.
/// @param ctx User context passed to the kernel.
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZCompOperators(
    IGA                iga,
    PetscInt           nmat,
    Mat                mats[],
    TDSEZPhysicsKernel kernel,
    void              *ctx);

/**
 * @brief Assemble multiple IGA matrices with homogeneous Dirichlet BCs.
 * @details Same as TDSEZCompOperators but applies IGASetBoundaryValue via
 * IGAElementFixSystem, enforcing psi=0 at the domain boundary.
 * @param iga IGA context.
 * @param nmat Number of matrices to assemble.
 * @param mats Output array of Mat objects (length nmat).
 * @param kernel Batch form kernel callback.
 * @param ctx User context passed to the kernel.
 * @return PetscErrorCode (0 on success).
 */
PetscErrorCode TDSEZCompOperatorsDirichlet(
    IGA                iga,
    PetscInt           nmat,
    Mat                mats[],
    TDSEZPhysicsKernel kernel,
    void              *ctx);

/// Element matrices of a physics kernel for one element (work zeroed by the
/// caller): sum-factorised for tensor-product elements where supported,
/// per-point kernel otherwise. See src/assembly/sumfact.cpp.
PetscErrorCode TDSEZElementWork(IGA iga, IGAElement element, TDSEZPhysicsKernel kernel,
                                PetscInt nmat, PetscScalar *work[], void *ctx,
                                unsigned need = ~0u);

/// Banded accumulation of element matrices on the ghosted tensor node box,
/// inserted row by row (see src/assembly/band.cpp). Falls back to
/// element-by-element insertion when not applicable or with -tdsez_band 0.
struct TDSEZBandAssembler {
    PetscErrorCode Setup(IGA iga, PetscInt nmat);
    PetscErrorCode BeginGroup(unsigned mask);
    PetscErrorCode Add(IGAElement e, PetscInt k, const PetscScalar *K, Mat mat);
    PetscErrorCode EndGroup(Mat mats[]);
    void Release();

    std::vector<unsigned> groups;   // matrix masks, one element pass each
    bool active = false;

  private:
    bool verified = false;
    PetscInt dim = 0, nmat = 0, perGroup = 0;
    PetscInt gstart[3] = {0, 0, 0}, gwidth[3] = {1, 1, 1}, half[3] = {0, 0, 0}, bw[3] = {1, 1, 1};
    size_t nrows = 0, nband = 0;
    unsigned current = 0;
    std::vector<PetscInt> slot;
    std::vector<PetscScalar> buf;
};

/// Run BLAS single-threaded during an element loop and restore PETSc's BLAS
/// thread count afterwards. Every MPI rank calls BLAS on small element-sized
/// products; a threaded BLAS (OpenBLAS/MKL/BLIS) left at its default would
/// start one thread per core in every rank. No-op before PETSc 3.21.
PetscErrorCode TDSEZBlasSerialBegin(PetscInt *saved);
PetscErrorCode TDSEZBlasSerialEnd(PetscInt saved);

/// Remove the rows and columns of the Dirichlet wall DoFs (recorded by
/// TDSEZCompOperatorsDirichlet) from an operator, with @p diag on the
/// diagonal. No-op for Neumann boundaries or a NULL matrix.
PetscErrorCode TDSEZZeroDirichletRows(Mat A, PetscScalar diag);

/// @name IGA form kernels (pointwise callback functions)
/// @brief These are the per-quadrature-point form kernels used by the
/// IGA assembly.  Each computes a local matrix contribution at a single
/// IGAPoint.  They must be non-static (referenced from other translation units).
/// @{
PetscErrorCode TDSEZFormHam(IGAPoint p, PetscInt nmat, PetscScalar *M[], void *ctx);       ///< Hamiltonian H = K + V
PetscErrorCode TDSEZFormPhyX(IGAPoint p, PetscInt nmat, PetscScalar *M[], void *ctx);      ///< physical-position operator
/// Batched TD operators for a general polarization: [K, V, Md] followed by
/// [D_d, Vel_d, dV_d] for every active axis d (TDSEZPhyAxesOn). Per point it
/// accumulates the single-operator integrand kernels times weight*detJac,
/// i.e. exactly what separate IGAComputeMatrix passes assemble.
PetscErrorCode TDSEZFormPhyAxes(IGAPoint p, PetscInt nmat, PetscScalar *M[], void *ctx);
/// Active axes of TDSEZFormPhyAxes (set by the assembler before the batch).
extern PetscBool TDSEZPhyAxesOn[3];
/// -tdsez_mat_free batch: [D_d for TDSEZMatFreeDipOn[d]] + [Vel_d for
/// TDSEZMatFreeVelOn[d]], the only operators kept assembled besides H, M and
/// the CAP (sum-factorised in 2D/3D like TDSEZFormPhyAxes).
PetscErrorCode TDSEZFormMatFreeAxes(IGAPoint p, PetscInt nmat, PetscScalar *M[], void *ctx);
extern PetscBool TDSEZMatFreeDipOn[3], TDSEZMatFreeVelOn[3];
PetscErrorCode TDSEZFormLz(IGAPoint p, PetscScalar *L, void *ctx);                        ///< angular momentum Lz
PetscErrorCode TDSEZFormKinetic(IGAPoint p, PetscScalar*K, void *ctx);                    ///< kinetic energy -∇²/2m
PetscErrorCode TDSEZFormPotential(IGAPoint p, PetscScalar*V, void *ctx);                  ///< potential V(x)
PetscErrorCode TDSEZFormMassDist(IGAPoint p, PetscScalar*Md, void *ctx);                   ///< mass distribution
PetscErrorCode DipoleX(IGAPoint p, PetscScalar *X, void *ctx);                           ///< x-position (dipole) operator
PetscErrorCode DipoleY(IGAPoint p, PetscScalar*Y, void *ctx);                             ///< y-position (dipole) operator
PetscErrorCode DipoleZ(IGAPoint p, PetscScalar*Z, void *ctx);                             ///< z-position (dipole) operator
PetscErrorCode VelocityX(IGAPoint p, PetscScalar *P, void *ctx);                         ///< x-velocity (momentum) operator
PetscErrorCode VelocityY(IGAPoint p, PetscScalar *P, void *ctx);                         ///< y-velocity (momentum) operator
PetscErrorCode VelocityZ(IGAPoint p, PetscScalar *P, void *ctx);                         ///< z-velocity (momentum) operator
PetscErrorCode TDSEZformPotentialGradX(IGAPoint p, PetscScalar *dVdr, void *ctx);         ///< -dV/dx
PetscErrorCode TDSEZformPotentialGradY(IGAPoint p, PetscScalar *dVdr, void *ctx);         ///< -dV/dy
PetscErrorCode TDSEZformPotentialGradZ(IGAPoint p, PetscScalar *dVdr, void *ctx);         ///< -dV/dz
PetscErrorCode TDSEZFormCouplingTerms(IGAPoint p, PetscInt nmat, PetscScalar *M[], void *ctx); ///< coupling terms G_i
extern PetscInt TDSEZCouplingIndex;                                                     ///< first term of the batch
/// Build TDSEZParser::CouplingTerms_ (FieldTerms, multipoles; src/coupling.cpp).
void TDSEZSetupCouplingTerms();
/// Log a warning where the order-th derivative of the laser (0: E) jumps (impulse in the next one).
void TDSEZWarnFieldKinks(int order, const char *terms);
/// Gauge = velocity: A(t) table and the (p - qA(t - n.r/c))^2 / 2m terms (velocity_gauge.cpp).
void TDSEZSetupVelocityGauge();
/// Gauge = velocity: A_a(t) = -int_0^t E_a (0 on uncoupled axes).
PetscReal TDSEZVectorPotential(int a, PetscReal t);
PetscErrorCode TDSEZformCap(IGAPoint p, PetscScalar *CAP, void *ctx);                     ///< CAP form kernel
/// @}

/// @name Assembler helpers
/// @brief Timed wrappers around IGA matrix assembly.
/// @{
/// @brief Assemble a single IGA matrix with timing instrumentation.
/// @param iga IGA context.
/// @param M Pointer to the output Mat (allocated inside).
/// @param form Form matrix callback.
/// @param name Human-readable name for logging.
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZAssembleMatrixTimed(IGA iga, Mat *M, IGAFormMatrix form, const char *name);
/// @brief Assemble a batch of IGA matrices with timing instrumentation.
/// @param iga IGA context.
/// @param n Number of matrices in the batch.
/// @param mats Pointer to array of Mat pointers (allocated inside).
/// @param form Batch form kernel callback.
/// @param name Human-readable name for logging.
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZAssembleMatBatchTimed(IGA iga, PetscInt n, Mat **mats, TDSEZPhysicsKernel form, const char *name);
/// @}

/// @name HDF5 monitor callbacks
/// @brief Called by PETSc TS at each diagnostic stride to write the
/// current wavefunction and diagnostic data to HDF5.
/// @{
PetscErrorCode TDSEZMonitorHDF5_1D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx); ///< 1D monitor
PetscErrorCode TDSEZMonitorHDF5_2D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx); ///< 2D monitor
PetscErrorCode TDSEZMonitorHDF5_3D(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx); ///< 3D monitor
/// Monitor of every step: the observers of the dimension, then the restart files (checkpoint.cpp).
PetscErrorCode TDSEZMonitorStep(TS ts, PetscInt step, PetscReal t, Vec psi, void *ctx);
/// @}

/// @brief Orthogonalize degenerate eigenstates using Lz (2D) or Gram-Schmidt.
/// @param TDSEZ Pointer to the TDSEZManager owning the bound states.
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZOrthogonalizeDegenerates(TDSEZManager *TDSEZ);

/// @brief Compute and save the unified transition dipole matrix (TDM).
/// @param states Converged bound-state vectors.
/// @param energies Corresponding eigen-energies.
/// @param mgr Manager: the dipole operators (assembled or factorised).
/// @param fileName Output filename for the TDM.
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZCompUnifiedDipoleMatrix(
    const std::vector<Vec>& states,
    const std::vector<PetscReal>& energies,
    TDSEZManager *mgr,
    const std::string& fileName);

/// @brief Save a PETSc Mat to a file (dense or sparse format).
/// @param mat Matrix to save.
/// @param filename Output filename.
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZSaveOperatorMatrix(Mat mat, const std::string& filename);

/// @brief Convert a complex state vector to real form via a unitary rotation.
/// @param v_complex Complex input vector.
/// @param Vre Output real vector.
/// @param M Mass matrix (for projection).
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZMakeStateReal2(Vec v_complex, Vec Vre, Mat M);

/// @brief Verify the length-velocity gauge consistency for dipole/velocity operators.
/// @param TDSEZ Pointer to the TDSEZManager.
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZCheckLengthVelocity(TDSEZManager *TDSEZ);

/// Width of the Manolopoulos CAP layer at each wall, c / (2 delta CAPKmin).
PetscReal TDSEZCapWidth();

/// Strong-field scales of the input field over [0, FinalTime] (propagation
/// with a coupled field only): peak field E0, angular frequency omega
/// (Omegax/y/z or the field's zero crossings), Up = E0^2/(4 omega^2), Ip
/// (|TargetEigenvalue| if negative, else 0.5), k_max = sqrt(2 (3.17 Up + Ip)),
/// quiver amplitude alpha = E0/omega^2 and the element width h = pi/(2 k_max)
/// that resolves the returning electron (about 4 elements per wavelength;
/// measured on 2D hydrogen-like HHG, see Knot Sequences). valid = false when
/// there is no field or omega cannot be estimated. Computed once.
struct TDSEZFieldScales {
    bool valid = false;
    PetscReal E0 = 0, omega = 0, Up = 0, Ip = 0, kmax = 0, alpha = 0, h = 0;
};
const TDSEZFieldScales &TDSEZEstimateFieldScales();

/// Static Kramers-Henneberger frame (Mode = kh, src/kh.cpp): integrate the
/// quiver of the laser and fill TDSEZParser::KHAlpha_; throws on bad input.
void TDSEZSetupKH();
/// Window, frequency, quiver amplitudes and samples of the KH average (for the output metadata).
struct TDSEZKHInfo { PetscReal t0 = 0, t1 = 0, omega = 0, ax = 0, ay = 0, az = 0; PetscInt samples = 0; };
TDSEZKHInfo &TDSEZKHSummary();

/// @brief Precompute the velocity (momentum) matrix elements between bound states.
/// @param TDSEZ Pointer to the TDSEZManager.
/// @return PetscErrorCode (0 on success).
PetscErrorCode TDSEZPrecomputeMomentumMatrix(TDSEZManager *TDSEZ);

/// @brief Extract breakpoint positions from a piecewise (?:) expression string.
/// @param expr muParser expression potentially containing ternary operators.
/// @return Vector of breakpoint coordinates.
std::vector<PetscReal> TDSEZExtractBreakpoints(const std::string& expr);
/// True when a(t) comes from the discrete equation of motion (AccelerationMethod);
/// the force-operator mass corrections are then unused and need not be assembled.
PetscBool TDSEZDiscreteAccelerationActive();

#endif // TDSEZ_INTERNAL_HPP
