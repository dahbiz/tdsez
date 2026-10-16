#ifndef TDSEZ_TSURFF_HPP
#define TDSEZ_TSURFF_HPP

#include <petiga.h>
#include <array>
#include <string>
#include <vector>

/**
 * @brief t-SURFF photoelectron momentum spectra (time-dependent surface flux).
 *
 * Passive observer: reads psi at the sampled steps and never changes the
 * propagation. Outside the box |x_d| < R (SurffRadius) the carrier must be free:
 * constant mass m and constant potential V0, Hamiltonian
 * H = -(1/2m) Delta + V0 + s E(t).r (s = -Charge). The amplitude of the
 * length-gauge Volkov state
 *     chi_k(r,t) = (2 pi)^{-d/2} exp(i (k + A(t)).r - i Phi_k(t)),
 *     A' = -s E,   Phi_k' = |k + A|^2 / 2m + V0,
 * is accumulated from the surface flux,
 *     b(k) = -(i/2m) int_0^T dt  oint dS (chi_k^* d_n psi - psi d_n chi_k^*),
 * with the trapezoidal rule over the samples (second order in the sampling
 * interval). |b(k)|^2 = dP/d^d k for the canonical momentum k; the final
 * kinetic momentum is k + A(T).
 *
 * End term (SurffEndTerm = 1): after FinalTime T the field is off, the Volkov
 * state only gains the phase exp(i E_k (t - T)), E_k = |k + A(T)|^2/2m + V0, and
 * psi evolves with the field-free H (CAP included). The flux still to come,
 * int_T^inf dt, is therefore the flux functional at T applied to
 *     phi_E = int_0^inf e^{i E s} psi(T + s) ds,   (E M - H) phi_E = i M psi(T),
 * the CAP being the outgoing-wave condition (+i0). phi_E is solved on energy
 * nodes uniform in |k + A| and its flux projections are combined with 4-point
 * Lagrange weights in |k + A| for every momentum, so the cost is one linear
 * solve and one flux evaluation per node, not per momentum.
 *
 * Efficiency: psi and (1/m) d_n psi at the surface Gauss points come from one
 * sparse product with a precomputed trace matrix. On a momentum grid the
 * Volkov factors separate by axis, so each face costs two small dense
 * products per sample instead of (#momenta x #surface points) complex
 * exponentials. Every rank works only on the surface points it owns; the
 * amplitudes are reduced once at the end.
 */
class TDSEZTSurff {
public:
    ~TDSEZTSurff();
    /// Build the surface, trace matrix and momentum tables from the TDSEZParser keys.
    PetscErrorCode Setup(IGA iga, Vec psi);
    /// Add the flux of psi at time t on every OutputStrideSurff-th step and at
    /// FinalTime (sample times must increase).
    PetscErrorCode Sample(PetscInt step, PetscReal t, Vec psi);
    /// Sample (t, psi) if it was not the last sample, add the end term (with
    /// SurffEndTerm; H is the field-free Hamiltonian with the CAP, M the
    /// overlap), reduce the amplitudes and write the separate HDF5 file (rank 0).
    PetscErrorCode Finish(const std::string &path, PetscReal t, Vec psi, Mat H = nullptr, Mat M = nullptr);

private:
    struct Point {
        PetscReal x[3] = {0, 0, 0};
        PetscReal w = 1, invMass = 1;
        int face = 0;          // 2*axis + (sign > 0)
        int it[2] = {0, 0};    // indices of the tangential coordinates (axis order)
    };
    PetscErrorCode AddFlux(PetscReal t, Vec psi);
    PetscErrorCode FluxGrid(const PetscScalar *v);
    PetscErrorCode FluxList(const PetscScalar *v);
    PetscErrorCode AuditTraces(IGA iga, Vec psi);
    void AdvanceVolkov(PetscReal t);
    PetscErrorCode Interior(Vec psi, std::vector<PetscScalar> &out);
    PetscErrorCode EndTerm(Vec psi, Mat H, Mat M);
    PetscErrorCode Project(const PetscScalar *c, const PetscInt n[3], const std::vector<PetscReal> k[3], PetscScalar *out);

    MPI_Comm comm_ = MPI_COMM_NULL;
    PetscMPIInt rank_ = 0;
    PetscInt dim_ = 1;
    PetscReal R_ = 0, mass_ = 1, V0_ = 0;
    // surface
    std::vector<PetscReal> nodes_[3], weights_[3];   // tangential Gauss nodes in [-R, R]
    std::vector<Point> points_;                      // owned surface points
    std::vector<int> facePoints_[6];                 // owned point indices per face
    Mat trace_ = nullptr;                            // rows 2q, 2q+1: psi, (1/m) d_n psi
    Vec surface_ = nullptr;
    // momenta
    bool grid_ = true;
    PetscInt n_[3] = {1, 1, 1};                      // grid sizes (1 on unused axes)
    std::vector<PetscReal> k_[3];                    // grid axes
    std::vector<std::array<PetscReal, 3>> list_;     // momentum list (list mode)
    std::vector<PetscScalar> ptan_[3];               // [k][q] = exp(-i k x_q)   (n_d x Q_d)
    std::vector<PetscScalar> pnorm_[3][2];           // [k]    = exp(-i k sigma R)
    PetscInt nk_ = 0;
    std::vector<PetscScalar> amp_, prev_, cur_;
    std::vector<PetscScalar> h1_, t1_;                // GEMM work arrays
    // Volkov integrals from t = 0 (RK4 substeps between samples):
    // A' = -s E, B = int A, C = int (|A|^2/2m + V0)
    PetscReal A_[3] = {0, 0, 0}, B_[3] = {0, 0, 0}, C_ = 0, volkovT_ = 0;
    PetscReal prevT_ = 0;
    bool havePrev_ = false, finished_ = false;
    PetscInt nSamples_ = 0;
    PetscLogDouble seconds_ = 0;
    PetscInt nGlobalPoints_ = 0;
    // end term statistics
    PetscInt endNodes_ = 0, endBuilds_ = 0, endIts_ = 0;
    PetscReal endDp_ = 0;
    PetscLogDouble endSeconds_ = 0;
    // interior projection (SurffInterior): spline knots and owned coefficient box
    std::vector<PetscReal> knots_[3];
    PetscInt p_[3] = {0, 0, 0}, nnp_[3] = {1, 1, 1}, lstart_[3] = {0, 0, 0}, lwidth_[3] = {1, 1, 1};
};

#endif // TDSEZ_TSURFF_HPP
