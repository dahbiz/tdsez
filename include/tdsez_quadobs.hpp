#ifndef TDSEZ_QUADOBS_HPP
#define TDSEZ_QUADOBS_HPP

#include <petiga.h>
#include <vector>

/**
 * @brief Expectation values of the monitors from psi at the quadrature points.
 *
 * Every monitor operator is A_ij = sum_q c(q) D^al B_i(q) D^be B_j(q) on the
 * assembly quadrature, so <psi|A|psi> = sum_q c(q) conj(D^al psi(q)) D^be psi(q)
 * gives the same number as the assembled matrix without storing it. psi and
 * its derivatives (first order per axis, mixed second order for the
 * mass-gradient force corrections) are evaluated element by element by sum
 * factorisation from the ghosted coefficients; the sums are reduced across
 * ranks once per call.
 *
 * Quantities: <V>, <1/m> (Md), <K>, <W> of the CAP, the forces dV_a with or
 * without the mass-gradient corrections, the velocities, and w^H Vel_a psi for
 * the discrete acceleration. Each is used only after the caller has checked
 * it against the assembled matrix.
 */
/// Monitor expectation values from the quadrature points are requested
/// (-tdsez_quad_obs, default on) and possible (2D/3D, no Dirichlet walls).
bool TDSEZQuadObsEnabled();
/// -tdsez_mat_free and the run allows it (TDSEZQuadObsEnabled, factorised
/// operators on, B-spline basis, CPU): the monitor and diagnostic operators
/// (K, V, Md, dV, the velocities for a constant mass, the dipoles of uncoupled
/// axes) are never assembled; their factorised and quadrature forms are used
/// directly. H, M, the CAP and the coupled dipoles stay assembled.
bool TDSEZMatrixFree();

class TDSEZQuadObs {
public:
    enum What : unsigned { Pot = 1, InvMass = 2, Kin = 4, Cap = 8, Force = 16, Vel = 32 };
    struct Values {
        PetscScalar pot = 0, invMass = 0, kin = 0, cap = 0, force[3] = {0, 0, 0}, vel[3] = {0, 0, 0};
    };

    /// Tabulate the owned elements and the coefficients of the quantities in
    /// @p what (dim 2 or 3, scalar Cartesian B-splines). @p forceCorrections:
    /// add the mass-gradient terms of the force matrices; @p forceAxes: bit a
    /// set for the forces wanted. The force coefficients are taken from the
    /// sum-factorised assembly (TDSEZForceCoeffStore) when it kept them all.
    PetscErrorCode Setup(IGA iga, unsigned what, bool forceCorrections, unsigned forceAxes);
    bool Ready() const { return ready_; }
    /// All quantities of Setup for psi (collective).
    PetscErrorCode Expect(Vec psi, Values &v);
    /// out[a] = w^H Vel_a psi (collective).
    PetscErrorCode VelocityBilinear(Vec w, Vec psi, PetscScalar out[3]);

private:
    struct Axis {                      // owned elements along one axis
        PetscInt ne = 1, n = 1, Q = 1;
        std::vector<PetscReal> N0, N1; // [(e*n + a)*Q + q]
        std::vector<PetscReal> W;      // weight * detJac: [e*Q + q]
        std::vector<PetscInt> off;     // first basis function of element e in the ghosted box
    };
    // field values at the points of one element: idx (d1,d2,d3) in {0,1}^3 -> [q]
    void Evaluate(const PetscScalar *c, PetscInt e1, PetscInt e2, PetscInt e3, bool mixed);
    PetscErrorCode Load(Vec x, std::vector<PetscScalar> &local);

    IGA iga_ = nullptr;
    bool ready_ = false, corr_ = false;
    unsigned what_ = 0, forceAxes_ = 0;
    PetscInt dim_ = 0, Qe_ = 0, gw_[3] = {1, 1, 1};
    Axis ax_[3];
    // per point (element-major, then q1 fastest), weights included
    std::vector<PetscReal> cV_, cInvM_, cW_, cF_[3];   // cF_[a]: classical force or (B,B) term
    std::vector<PetscReal> cBd_[3][3], cBp_[3][3], cdd_[3][3];   // [axis a][e or pair]
    std::vector<PetscScalar> F_[8], t1_, t2_;          // field values, work
    std::vector<PetscScalar> loc_, loc2_;
    std::vector<PetscReal> rho_, acc_;
    std::vector<PetscScalar> wq_[4];
};

#endif // TDSEZ_QUADOBS_HPP
