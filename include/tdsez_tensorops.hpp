#ifndef TDSEZ_TENSOROPS_HPP
#define TDSEZ_TENSOROPS_HPP

#include <petiga.h>
#include <vector>

/**
 * @brief Factorised (Kronecker) application of the separable operators.
 *
 * On a tensor B-spline basis with a tensor Gauss rule, the operators the
 * monitors apply every step are sums of Kronecker products of 1D banded
 * matrices (x fastest):
 *
 *   M    = M_z (x) M_y (x) M_x
 *   D_a  = X_a on axis a, M on the others            (dipole x_a)
 *   Vel_a = -i/(2m) (G - G^T)_a, M on the others     (constant mass m)
 *   K    = 1/(2m) sum_a S_a, M on the others         (constant mass m)
 *
 * with M = int B_i B_j, X = int x B_i B_j, G = int B_i B_j', S = int B_i' B_j'
 * built from PetIGA's 1D quadrature tables. Applying one term to psi takes one
 * halo exchange of psi (shared by all operators) and three 1D band
 * contractions: about 6 d (2p+1) flops per coefficient instead of 8 (2p+1)^d
 * for the assembled complex matrix, and no matrix in memory.
 *
 * Each operator is used only after Validate() found it equal to the assembled
 * matrix on a state to 1e-10 relative; otherwise the caller keeps MatMult.
 * Matrix-free runs (-tdsez_mat_free) assemble no such matrix and use Enable().
 */
class TDSEZTensorOps {
public:
    enum Op { Mass = 0, DipX, DipY, DipZ, VelX, VelY, VelZ, Kin, NOps };

    ~TDSEZTensorOps();
    /// Build the 1D bands and the halo scatter (dim 2 or 3, scalar Cartesian
    /// B-splines). invMass = 1/m for a constant mass, 0 if the mass varies (then
    /// Vel and K are not separable and stay unavailable).
    PetscErrorCode Setup(IGA iga, Vec templ, PetscReal invMass);
    /// Compare op x against A x on x; marks op usable when they agree.
    PetscErrorCode Validate(Op op, Mat A, Vec x, PetscReal *relerr);
    /// Matrix-free runs (-tdsez_mat_free): mark op usable without an
    /// assembled matrix to compare with, if it is separable here.
    bool Enable(Op op) { usable_[op] = Available(op); return usable_[op]; }
    bool Usable(Op op) const { return ready_ && usable_[op]; }
    /// y = op x.
    PetscErrorCode Apply(Op op, Vec x, Vec y);

private:
    /// op is separable on this basis (dimension, constant mass for Vel and K).
    bool Available(Op op) const;
    struct Band { std::vector<PetscReal> v; };   // n x (2p+1): (i, j) at i*bw + j-i+p
    PetscErrorCode Load(Vec x);
    void Term(const Band *A[3], PetscScalar c, bool accumulate, PetscScalar *y);

    bool ready_ = false, usable_[NOps] = {};
    PetscInt dim_ = 0, n_[3] = {1, 1, 1}, p_[3] = {0, 0, 0}, bw_[3] = {1, 1, 1};
    PetscInt own0_[3] = {0, 0, 0}, ownw_[3] = {1, 1, 1}, h0_[3] = {0, 0, 0}, hw_[3] = {1, 1, 1};
    PetscReal invMass_ = 0;
    Band M_[3], X_[3], A_[3], S_[3], I_;          // A = G - G^T; I_ = identity of size 1
    VecScatter halo_ = nullptr;
    Vec local_ = nullptr;                          // halo box of the loaded vector
    Vec loaded_ = nullptr;
    PetscObjectState loadedState_ = -1;
    std::vector<PetscScalar> t1_, t2_;
};

#endif // TDSEZ_TENSOROPS_HPP
