#include "tdsez_tsurff.hpp"
#include "tdsez_info.hpp"
#include "tdsez_parser.hpp"
#include <petscblaslapack.h>
#include <hdf5.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <sstream>

/**
 * @file tsurff.cpp
 * @brief t-SURFF photoelectron spectra (see tdsez_tsurff.hpp).
 */

namespace {

/// Gauss-Legendre nodes and weights on [-1, 1].
void GaussLegendre(PetscInt n, std::vector<PetscReal> &x, std::vector<PetscReal> &w)
{
    x.resize(n); w.resize(n);
    const PetscReal pi = std::acos(-1.0);
    for (PetscInt j = 0; j < n; ++j) {
        PetscReal z = std::cos(pi * (j + 0.75) / (n + 0.5)), der = 1;
        for (int it = 0; it < 100; ++it) {
            PetscReal p = 1, p0 = 0;
            for (PetscInt k = 1; k <= n; ++k) { const PetscReal old = p; p = ((2 * k - 1) * z * p - (k - 1) * p0) / k; p0 = old; }
            der = n * (z * p - p0) / (z * z - 1);
            const PetscReal dz = p / der;
            z -= dz;
            if (std::abs(dz) < 1e-15) break;
        }
        PetscReal p = 1, p0 = 0;
        for (PetscInt k = 1; k <= n; ++k) { const PetscReal old = p; p = ((2 * k - 1) * z * p - (k - 1) * p0) / k; p0 = old; }
        der = n * (z * p - p0) / (z * z - 1);
        x[j] = z;
        w[j] = 2 / ((1 - z * z) * der * der);
    }
}

/// B-spline basis functions N_{s-p..s}(x) of knot span s (Piegl-Tiller A2.2).
PetscInt BasisFuns(const std::vector<PetscReal> &U, PetscInt p, PetscInt nnp, PetscReal x, PetscReal *N)
{
    PetscInt lo = p, hi = nnp;                         // span s in [p, nnp-1]: U[s] <= x < U[s+1]
    if (x >= U[nnp]) lo = nnp - 1;
    else while (hi - lo > 1) { const PetscInt mid = (lo + hi) / 2; if (x < U[mid]) hi = mid; else lo = mid; }
    const PetscInt s = lo;
    std::vector<PetscReal> left(p + 1), right(p + 1);
    N[0] = 1;
    for (PetscInt j = 1; j <= p; ++j) {
        left[j] = x - U[s + 1 - j]; right[j] = U[s + j] - x;
        PetscReal saved = 0;
        for (PetscInt r = 0; r < j; ++r) {
            const PetscReal t = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * t;
            saved = left[j - r] * t;
        }
        N[j] = saved;
    }
    return s;
}

inline PetscScalar Phase(PetscReal angle) { return PetscCMPLX(std::cos(angle), std::sin(angle)); }

/// The two axes tangential to face axis a, in increasing order. Axes >= dim
/// are dummies with one node at 0 and one momentum k = 0.
inline void Tangential(int a, int &b, int &c) { b = (a == 0) ? 1 : 0; c = (a == 2) ? 1 : 2; }

herr_t WriteReal(hid_t file, const char *name, int rank, const hsize_t *dims, const PetscReal *data)
{
    const hid_t space = H5Screate_simple(rank, dims, nullptr);
    const hid_t set = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = (space < 0 || set < 0) ? -1 : H5Dwrite(set, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    if (set >= 0) H5Dclose(set);
    if (space >= 0) H5Sclose(space);
    return status;
}

herr_t WriteAttribute(hid_t file, const char *name, hsize_t n, const PetscReal *data)
{
    const hid_t space = H5Screate_simple(1, &n, nullptr);
    const hid_t attr = H5Acreate2(file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = (space < 0 || attr < 0) ? -1 : H5Awrite(attr, H5T_NATIVE_DOUBLE, data);
    if (attr >= 0) H5Aclose(attr);
    if (space >= 0) H5Sclose(space);
    return status;
}

} // namespace

TDSEZTSurff::~TDSEZTSurff()
{
    VecDestroy(&surface_);
    MatDestroy(&trace_);
}

PetscErrorCode TDSEZTSurff::Setup(IGA iga, Vec psi)
{
    PetscFunctionBeginUser;
    comm_ = PetscObjectComm((PetscObject)psi);
    PetscCallMPI(MPI_Comm_rank(comm_, &rank_));
    dim_ = TDSEZParser::Dimension;
    R_ = TDSEZParser::SurffRadius;
    PetscCheck(!iga->geometry && !iga->rational && iga->dof == 1, comm_, PETSC_ERR_SUP,
               "t-SURFF needs scalar Cartesian non-rational B-splines");

    // ---- surface quadrature: Gauss points of every knot span inside [-R, R]
    const PetscInt nq = TDSEZParser::SurffQuadrature > 0 ? TDSEZParser::SurffQuadrature : TDSEZParser::NQuadratures;
    std::vector<PetscReal> gx, gw;
    GaussLegendre(nq, gx, gw);
    PetscReal axisEnd[3][2] = {};
    for (int d = 0; d < 3; ++d) {
        nodes_[d].clear(); weights_[d].clear();
        if (d >= dim_) { nodes_[d] = {0}; weights_[d] = {1}; continue; }
        IGAAxis axis; PetscInt m; PetscReal *U;
        PetscCall(IGAGetAxis(iga, d, &axis));
        PetscCall(IGAAxisGetKnots(axis, &m, &U));
        axisEnd[d][0] = U[0]; axisEnd[d][1] = U[m];
        knots_[d].assign(U, U + m + 1);
        p_[d] = axis->p; nnp_[d] = axis->nnp;
        lstart_[d] = iga->node_lstart[d]; lwidth_[d] = iga->node_lwidth[d];
        PetscCheck(U[0] < -R_ && U[m] > R_, comm_, PETSC_ERR_ARG_OUTOFRANGE,
                   "t-SURFF surface |x| = %g is not inside axis %d", (double)R_, d);
        for (PetscInt j = 0; j < m; ++j) {
            const PetscReal lo = PetscMax(U[j], -R_), hi = PetscMin(U[j + 1], R_);
            if (hi <= lo) continue;
            for (PetscInt k = 0; k < nq; ++k) {
                nodes_[d].push_back(0.5 * (lo + hi) + 0.5 * (hi - lo) * gx[k]);
                weights_[d].push_back(0.5 * (hi - lo) * gw[k]);
            }
        }
    }

    // ---- exterior reference: mass and carrier potential at x = (R, 0, 0)
    mass_ = TDSEZParser::MassDist(R_, 0, 0);
    V0_ = TDSEZParser::V(R_, (PetscReal)0, (PetscReal)0);
    PetscCheck(std::isfinite(mass_) && mass_ > 0 && std::isfinite(V0_), comm_, PETSC_ERR_USER_INPUT,
               "t-SURFF: invalid mass or potential at the surface");

    // ---- candidate surface points; keep the ones whose element this rank owns
    struct Row { Point p; std::vector<PetscInt> cols; std::vector<PetscScalar> value, normal; };
    std::vector<Row> rows;
    std::vector<int> candidate;
    IGAProbe probe = nullptr;
    PetscCall(IGAProbeCreate(iga, psi, &probe));
    PetscCall(IGAProbeSetOrder(probe, 1));
    PetscCall(IGAProbeSetCollective(probe, PETSC_FALSE));
    for (int a = 0; a < dim_; ++a) for (int s = 0; s < 2; ++s) {
        const PetscReal sigma = s ? 1.0 : -1.0;
        int b, c; Tangential(a, b, c);
        for (size_t ib = 0; ib < nodes_[b].size(); ++ib) for (size_t ic = 0; ic < nodes_[c].size(); ++ic) {
            Point p;
            p.x[a] = sigma * R_; p.x[b] = nodes_[b][ib]; p.x[c] = nodes_[c][ic];
            p.w = weights_[b][ib] * weights_[c][ic];
            p.face = 2 * a + s; p.it[0] = (int)ib; p.it[1] = (int)ic;
            PetscCall(IGAProbeSetPoint(probe, p.x));
            candidate.push_back(probe->offprocess ? INT_MAX : (int)rank_);
            if (probe->offprocess) continue;
            Row row; row.p = p;
            row.cols.resize(probe->nen); row.value.resize(probe->nen); row.normal.resize(probe->nen);
            PetscCall(ISLocalToGlobalMappingApply(iga->lgmap, probe->nen, probe->map, row.cols.data()));
            for (PetscInt e = 0; e < probe->nen; ++e) {
                row.value[e] = probe->basis[0][e];
                row.normal[e] = sigma * probe->basis[1][e * dim_ + a];
            }
            rows.push_back(std::move(row));
        }
    }
    PetscCall(IGAProbeDestroy(&probe));
    // One reduction decides the owner of every point (the lowest candidate rank).
    std::vector<int> owner(candidate.size());
    PetscCallMPI(MPI_Allreduce(candidate.data(), owner.data(), (int)candidate.size(), MPI_INT, MPI_MIN, comm_));
    for (int o : owner) PetscCheck(o != INT_MAX, comm_, PETSC_ERR_PLIB, "t-SURFF: no rank owns a surface point");
    nGlobalPoints_ = (PetscInt)candidate.size();

    // ---- owned points: exterior checks along the outward ray, trace rows
    PetscReal massErr = 0, potErr = 0;
    PetscBool valid = PETSC_TRUE;
    std::vector<Row> owned;
    {
        size_t r = 0;
        for (size_t i = 0; i < candidate.size(); ++i) {
            if (candidate[i] != (int)rank_) continue;
            Row &row = rows[r++];
            if (owner[i] != (int)rank_) continue;
            const int a = row.p.face / 2, s = row.p.face % 2;
            for (int check = 0; check <= 4; ++check) {
                PetscReal x[3] = {row.p.x[0], row.p.x[1], row.p.x[2]};
                x[a] += (axisEnd[a][s] - x[a]) * check / 4.0;
                PetscReal m = 0, v = 0;
                try { m = TDSEZParser::MassDist(x[0], x[1], x[2]); v = TDSEZParser::V(x[0], x[1], x[2]); }
                catch (...) { valid = PETSC_FALSE; }
                if (!(std::isfinite(m) && m > 0 && std::isfinite(v))) { valid = PETSC_FALSE; continue; }
                massErr = PetscMax(massErr, std::abs(m - mass_) / mass_);
                potErr = PetscMax(potErr, std::abs(v - V0_));
                if (check == 0) row.p.invMass = 1.0 / m;
            }
            for (auto &n : row.normal) n *= row.p.invMass;
            owned.push_back(std::move(row));
        }
    }
    PetscReal local[3] = {massErr, potErr, valid ? 0.0 : 1.0}, global[3];
    PetscCallMPI(MPI_Allreduce(local, global, 3, MPIU_REAL, MPIU_MAX, comm_));
    PetscCheck(global[2] == 0, comm_, PETSC_ERR_USER_INPUT, "t-SURFF: invalid mass or potential outside the surface");
    TDSEZInfo(comm_).kv("t-SURFF exterior", "mass %g, potential %g Ha; max deviation outside |x| = %g: "
                          "mass %.2e (relative), potential %.2e Ha",
                          (double)mass_, (double)V0_, (double)R_, (double)global[0], (double)global[1]);
    PetscCheck(global[0] <= TDSEZParser::SurffMassTolerance && global[1] <= TDSEZParser::SurffPotentialTolerance,
               comm_, PETSC_ERR_USER_INPUT,
               "t-SURFF needs a free carrier outside the surface: the mass deviates by %g (SurffMassTolerance %g) and the "
               "potential by %g Ha (SurffPotentialTolerance %g). Move SurffRadius out, cut the potential off, or raise "
               "the tolerances", (double)global[0], (double)TDSEZParser::SurffMassTolerance, (double)global[1],
               (double)TDSEZParser::SurffPotentialTolerance);

    // ---- trace matrix: row 2q = psi(x_q), row 2q+1 = (1/m) d_n psi(x_q)
    PetscInt localCols, globalCols, nen = 1;
    PetscCall(VecGetLocalSize(psi, &localCols));
    PetscCall(VecGetSize(psi, &globalCols));
    for (int d = 0; d < dim_; ++d) nen *= TDSEZParser::SplineDegree + 1;
    PetscCall(MatCreateAIJ(comm_, 2 * (PetscInt)owned.size(), localCols, PETSC_DETERMINE, globalCols,
                           nen, nullptr, nen, nullptr, &trace_));
    PetscInt begin;
    PetscCall(MatGetOwnershipRange(trace_, &begin, nullptr));
    points_.clear();
    for (auto &f : facePoints_) f.clear();
    for (size_t q = 0; q < owned.size(); ++q) {
        const Row &row = owned[q];
        PetscInt r = begin + 2 * (PetscInt)q;
        const PetscInt n = (PetscInt)row.cols.size();
        PetscCall(MatSetValues(trace_, 1, &r, n, row.cols.data(), row.value.data(), INSERT_VALUES)); ++r;
        PetscCall(MatSetValues(trace_, 1, &r, n, row.cols.data(), row.normal.data(), INSERT_VALUES));
        facePoints_[row.p.face].push_back((int)q);
        points_.push_back(row.p);
    }
    PetscCall(MatAssemblyBegin(trace_, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(trace_, MAT_FINAL_ASSEMBLY));
    PetscCall(MatCreateVecs(trace_, nullptr, &surface_));
    PetscCall(AuditTraces(iga, psi));

    // ---- momenta
    grid_ = TDSEZParser::SurffMomenta.empty();
    if (grid_) {
        nk_ = 1;
        for (int d = 0; d < 3; ++d) {
            n_[d] = d < dim_ ? TDSEZParser::SurffNk[d] : 1;
            k_[d].resize(n_[d]);
            for (PetscInt i = 0; i < n_[d]; ++i)
                k_[d][i] = d >= dim_ ? 0.0 : (n_[d] == 1 ? TDSEZParser::SurffKmin[d]
                           : TDSEZParser::SurffKmin[d] + i * (TDSEZParser::SurffKmax[d] - TDSEZParser::SurffKmin[d]) / (n_[d] - 1));
            nk_ *= n_[d];
            // ptan_[d][k + n_d q] = exp(-i k x_q): left GEMM operand (n_d x Q_d,
            // column major) and, read by column, the contiguous axpy row of x_q.
            const size_t Q = nodes_[d].size();
            ptan_[d].resize(n_[d] * Q);
            for (size_t q = 0; q < Q; ++q)
                for (PetscInt i = 0; i < n_[d]; ++i) ptan_[d][i + n_[d] * q] = Phase(-k_[d][i] * nodes_[d][q]);
            for (int s = 0; s < 2; ++s) {
                pnorm_[d][s].resize(n_[d]);
                for (PetscInt i = 0; i < n_[d]; ++i) pnorm_[d][s][i] = Phase(-k_[d][i] * (s ? R_ : -R_));
            }
        }
        size_t hsize = 0, tsize = 0;
        for (int a = 0; a < dim_; ++a) {
            int b, c; Tangential(a, b, c);
            hsize = std::max(hsize, (size_t)(2 * n_[c]) * nodes_[b].size());
            tsize = std::max(tsize, (size_t)(2 * n_[b] * n_[c]));
        }
        h1_.resize(hsize); t1_.resize(tsize);
    } else {
        PetscInt n = 0;
        if (!rank_) {
            std::ifstream in(TDSEZParser::SurffMomenta);
            std::string line;
            while (in && std::getline(in, line)) {
                line = line.substr(0, line.find('#'));
                std::istringstream row(line);
                std::array<PetscReal, 3> k{};
                if (!(row >> k[0])) continue;
                bool ok = true;
                for (int d = 1; d < dim_; ++d) ok = ok && (bool)(row >> k[d]);
                if (!ok) { list_.clear(); break; }
                list_.push_back(k);
            }
            n = (PetscInt)list_.size();
        }
        PetscCallMPI(MPI_Bcast(&n, 1, MPIU_INT, 0, comm_));
        PetscCheck(n > 0, comm_, PETSC_ERR_USER_INPUT,
                   "t-SURFF: SurffMomenta file '%s' is missing, empty or has a row with fewer than %d values",
                   TDSEZParser::SurffMomenta.c_str(), (int)dim_);
        list_.resize(n);
        PetscCallMPI(MPI_Bcast(list_.data(), 3 * (int)n, MPIU_REAL, 0, comm_));
        nk_ = n;
    }
    amp_.assign(nk_, 0); prev_.assign(nk_, 0); cur_.assign(nk_, 0);

    TDSEZInfo(comm_).kv("t-SURFF surface", "|x_d| = %g, %" PetscInt_FMT " points (%" PetscInt_FMT
                          " per span), %" PetscInt_FMT " momenta (%s), stride %" PetscInt_FMT,
                          (double)R_, nGlobalPoints_, nq, nk_, grid_ ? "grid" : "list", TDSEZParser::OutputStrideSurff);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZTSurff::AuditTraces(IGA iga, Vec psi)
{
    PetscFunctionBeginUser;
    PetscCall(MatMult(trace_, psi, surface_));
    IGAProbe probe = nullptr;
    PetscCall(IGAProbeCreate(iga, psi, &probe));
    PetscCall(IGAProbeSetOrder(probe, 1));
    PetscCall(IGAProbeSetCollective(probe, PETSC_FALSE));
    const PetscScalar *v;
    PetscCall(VecGetArrayRead(surface_, &v));
    PetscReal err[2] = {0, 0}, global[2];
    for (size_t q = 0; q < points_.size(); ++q) {
        const Point &p = points_[q];
        const int a = p.face / 2;
        const PetscReal sigma = (p.face % 2) ? 1.0 : -1.0;
        PetscScalar value = 0, grad[3] = {0, 0, 0};
        PetscCall(IGAProbeSetPoint(probe, p.x));
        PetscCall(IGAProbeFormValue(probe, &value));
        PetscCall(IGAProbeFormGrad(probe, grad));
        const PetscScalar normal = p.invMass * sigma * grad[a];
        err[0] = PetscMax(err[0], PetscMax(PetscAbsScalar(v[2 * q] - value), PetscAbsScalar(v[2 * q + 1] - normal)));
        err[1] = PetscMax(err[1], PetscMax(PetscAbsScalar(value), PetscAbsScalar(normal)));
    }
    PetscCall(VecRestoreArrayRead(surface_, &v));
    PetscCall(IGAProbeDestroy(&probe));
    PetscCallMPI(MPI_Allreduce(err, global, 2, MPIU_REAL, MPIU_MAX, comm_));
    PetscCheck(global[0] <= 1e-10 * PetscMax(1.0, global[1]), comm_, PETSC_ERR_PLIB,
               "t-SURFF: surface trace differs from direct evaluation by %g", (double)global[0]);
    PetscFunctionReturn(PETSC_SUCCESS);
}

void TDSEZTSurff::AdvanceVolkov(PetscReal t)
{
    // y = (A, B, C): A' = -s E, B' = A, C' = |A|^2/2m + V0; classical RK4 on
    // substeps of at most TimeStep/2, so the Volkov phase does not depend on
    // the sampling stride.
    const PetscReal h = t - volkovT_;
    if (h <= 0) return;
    const PetscInt n = PetscMax((PetscInt)1, (PetscInt)std::ceil(h / (0.5 * TDSEZParser::TimeStep) - 1e-9));
    const PetscReal dt = h / n;
    auto f = [&](PetscReal tt, const PetscReal y[7], PetscReal dy[7]) {
        PetscReal a2 = 0;
        for (int d = 0; d < 3; ++d) {
            dy[d] = d < dim_ ? -TDSEZParser::Coupling(d, tt) : 0.0;
            dy[3 + d] = y[d];
            a2 += y[d] * y[d];
        }
        dy[6] = a2 / (2 * mass_) + V0_;
    };
    PetscReal y[7] = {A_[0], A_[1], A_[2], B_[0], B_[1], B_[2], C_}, k1[7], k2[7], k3[7], k4[7], z[7];
    for (PetscInt i = 0; i < n; ++i) {
        const PetscReal t0 = volkovT_ + i * dt;
        f(t0, y, k1);
        for (int j = 0; j < 7; ++j) z[j] = y[j] + 0.5 * dt * k1[j];
        f(t0 + 0.5 * dt, z, k2);
        for (int j = 0; j < 7; ++j) z[j] = y[j] + 0.5 * dt * k2[j];
        f(t0 + 0.5 * dt, z, k3);
        for (int j = 0; j < 7; ++j) z[j] = y[j] + dt * k3[j];
        f(t0 + dt, z, k4);
        for (int j = 0; j < 7; ++j) y[j] += dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
    }
    for (int d = 0; d < 3; ++d) { A_[d] = y[d]; B_[d] = y[3 + d]; }
    C_ = y[6];
    volkovT_ = t;
}

PetscErrorCode TDSEZTSurff::Sample(PetscInt step, PetscReal t, Vec psi)
{
    PetscFunctionBeginUser;
    const PetscReal T = TDSEZParser::FinalTime;
    const bool last = t >= T - 1e-12 * PetscMax((PetscReal)1, T);
    if (step % TDSEZParser::OutputStrideSurff == 0 || last) PetscCall(AddFlux(t, psi));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZTSurff::AddFlux(PetscReal t, Vec psi)
{
    PetscFunctionBeginUser;
    PetscCheck(!finished_, comm_, PETSC_ERR_ARG_WRONGSTATE, "t-SURFF already finished");
    if (havePrev_ && t == prevT_) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCheck(!havePrev_ || t > prevT_, comm_, PETSC_ERR_ARG_OUTOFRANGE, "t-SURFF sample times must increase");
    PetscLogDouble t0;
    PetscCall(PetscTime(&t0));
    AdvanceVolkov(t);
    PetscCall(MatMult(trace_, psi, surface_));
    const PetscScalar *v;
    PetscCall(VecGetArrayRead(surface_, &v));
    std::fill(cur_.begin(), cur_.end(), 0.0);
    if (!points_.empty()) {
        if (grid_) PetscCall(FluxGrid(v));
        else PetscCall(FluxList(v));
    }
    PetscCall(VecRestoreArrayRead(surface_, &v));
    if (havePrev_) {
        const PetscReal h = 0.5 * (t - prevT_);
        for (PetscInt k = 0; k < nk_; ++k) amp_[k] += h * (prev_[k] + cur_[k]);
    }
    prev_.swap(cur_);
    prevT_ = t;
    havePrev_ = true;
    ++nSamples_;
    PetscLogDouble t1;
    PetscCall(PetscTime(&t1));
    seconds_ += t1 - t0;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// flux(k) = (i/2) sum_q w_q chi_k^*(x_q) [-(1/m) d_n psi - i sigma (k_a + A_a) psi / m_q],
// chi_k^* = (2 pi)^{-d/2} exp(i Phi_k - i (k + A).x). On a face x_a = sigma R with
// tangential axes b, c the sum over the face's Gauss grid factorises:
//   H[(row, ib)]  = sum_ic g_row(ib, ic) exp(-i k_c x_ic)        (axpy per point)
//   T[(kb, row)]  = sum_ib exp(-i k_b x_ib) H[(row, ib)]          (one GEMM)
// with g_1 = -w e^{-i A_t.x_t} (1/m) d_n psi, g_2 = -i sigma w e^{-i A_t.x_t} psi / m_q
// and row = (1 or 2, k_c); then flux += e^{-i (k_a + A_a) sigma R} (T_1 + (k_a + A_a) T_2).
PetscErrorCode TDSEZTSurff::FluxGrid(const PetscScalar *v)
{
    PetscFunctionBeginUser;
    const PetscScalar I = PetscCMPLX(0.0, 1.0);
    for (int f = 0; f < 2 * dim_; ++f) {
        if (facePoints_[f].empty()) continue;
        const int a = f / 2, s = f % 2;
        const PetscReal sigma = s ? 1.0 : -1.0;
        int b, c; Tangential(a, b, c);
        const PetscInt nb = n_[b], nc = n_[c], Qb = (PetscInt)nodes_[b].size();
        const PetscScalar *Pb = ptan_[b].data(), *Pc = ptan_[c].data();
        PetscScalar *H = h1_.data(), *T = t1_.data();
        std::fill(H, H + 2 * nc * Qb, 0.0);
        for (int q : facePoints_[f]) {
            const Point &p = points_[q];
            const PetscScalar e = p.w * Phase(-(A_[b] * p.x[b] + A_[c] * p.x[c]));
            const PetscScalar g1 = -e * v[2 * q + 1], g2 = -I * sigma * p.invMass * e * v[2 * q];
            PetscScalar *h = H + 2 * nc * p.it[0];
            const PetscScalar *pc = Pc + nc * p.it[1];
            for (PetscInt kc = 0; kc < nc; ++kc) { h[kc] += g1 * pc[kc]; h[nc + kc] += g2 * pc[kc]; }
        }
        PetscBLASInt m, n, k, ldh;
        PetscCall(PetscBLASIntCast(nb, &m));
        PetscCall(PetscBLASIntCast(2 * nc, &n));
        PetscCall(PetscBLASIntCast(Qb, &k));
        ldh = n;
        const PetscScalar one = 1.0, zero = 0.0;
        PetscCallBLAS("BLASgemm", BLASgemm_("N", "T", &m, &n, &k, &one, Pb, &m, H, &ldh, &zero, T, &m));
        const PetscReal Aa = A_[a];
        const PetscScalar ea = Phase(-Aa * sigma * R_);
        const PetscScalar *pn = pnorm_[a][s].data();
        PetscInt kk[3], idx = 0;
        for (kk[0] = 0; kk[0] < n_[0]; ++kk[0])
            for (kk[1] = 0; kk[1] < n_[1]; ++kk[1])
                for (kk[2] = 0; kk[2] < n_[2]; ++kk[2], ++idx) {
                    const PetscInt ka = kk[a], kb = kk[b], kc = kk[c];
                    cur_[idx] += pn[ka] * ea * (T[kb + nb * kc] + (k_[a][ka] + Aa) * T[kb + nb * (nc + kc)]);
                }
    }
    // Volkov phase exp(i Phi_k) = exp(i C) prod_d exp(i (t k_d^2 / 2m + k_d B_d / m)).
    std::vector<PetscScalar> ph[3];
    for (int d = 0; d < 3; ++d) {
        ph[d].resize(n_[d]);
        for (PetscInt i = 0; i < n_[d]; ++i)
            ph[d][i] = Phase(volkovT_ * k_[d][i] * k_[d][i] / (2 * mass_) + k_[d][i] * B_[d] / mass_);
    }
    const PetscScalar pre = 0.5 * I * std::pow(2 * PETSC_PI, -0.5 * dim_) * Phase(C_);
    PetscInt idx = 0;
    for (PetscInt i = 0; i < n_[0]; ++i) {
        const PetscScalar p0 = pre * ph[0][i];
        for (PetscInt j = 0; j < n_[1]; ++j) {
            const PetscScalar p1 = p0 * ph[1][j];
            for (PetscInt l = 0; l < n_[2]; ++l, ++idx) cur_[idx] *= p1 * ph[2][l];
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZTSurff::FluxList(const PetscScalar *v)
{
    PetscFunctionBeginUser;
    const PetscScalar I = PetscCMPLX(0.0, 1.0);
    const PetscReal norm = std::pow(2 * PETSC_PI, -0.5 * dim_);
    for (PetscInt k = 0; k < nk_; ++k) {
        const auto &kv = list_[k];
        PetscReal phi = C_, kA[3];
        for (int d = 0; d < 3; ++d) {
            phi += volkovT_ * kv[d] * kv[d] / (2 * mass_) + kv[d] * B_[d] / mass_;
            kA[d] = kv[d] + A_[d];
        }
        PetscScalar sum = 0;
        for (size_t q = 0; q < points_.size(); ++q) {
            const Point &p = points_[q];
            const int a = p.face / 2;
            const PetscReal sigma = (p.face % 2) ? 1.0 : -1.0;
            const PetscScalar chi = p.w * Phase(phi - (kA[0] * p.x[0] + kA[1] * p.x[1] + kA[2] * p.x[2]));
            sum += chi * (-v[2 * q + 1] - I * sigma * p.invMass * kA[a] * v[2 * q]);
        }
        cur_[k] = 0.5 * I * norm * sum;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// <chi_k(T)| theta psi(T)> over the box |x_d| < R: with F_d[k][i] = int_{-R}^{R}
// exp(-i (k + A_d) x) B_i(x) dx the overlap is the tensor contraction
// sum_{ijl} c_ijl F_x[kx][i] F_y[ky][j] F_z[kz][l] of the owned spline
// coefficients (x fastest in the PETSc local array), done as three GEMMs.
PetscErrorCode TDSEZTSurff::Project(const PetscScalar *c, const PetscInt n[3], const std::vector<PetscReal> k[3], PetscScalar *out)
{
    PetscFunctionBeginUser;
    std::vector<PetscScalar> F[3];
    for (int d = 0; d < 3; ++d) {
        if (d >= dim_) { F[d].assign(n[d], 1.0); continue; }
        F[d].assign(n[d] * lwidth_[d], 0.0);
        std::vector<PetscReal> N(p_[d] + 1);
        for (size_t q = 0; q < nodes_[d].size(); ++q) {
            const PetscReal x = nodes_[d][q];
            const PetscInt s = BasisFuns(knots_[d], p_[d], nnp_[d], x, N.data());
            for (PetscInt b = 0; b <= p_[d]; ++b) {
                const PetscInt i = s - p_[d] + b - lstart_[d];
                if (i < 0 || i >= lwidth_[d]) continue;
                const PetscReal wb = weights_[d][q] * N[b];
                for (PetscInt kk = 0; kk < n[d]; ++kk) F[d][kk + n[d] * i] += wb * Phase(-(k[d][kk] + A_[d]) * x);
            }
        }
    }
    const PetscInt L0 = lwidth_[0], L1 = lwidth_[1], L2 = lwidth_[2], n0 = n[0], n1 = n[1], n2 = n[2];
    std::vector<PetscScalar> G1(n0 * L1 * L2), G2(n0 * n1 * L2), G3(n0 * n1 * n2);
    const PetscScalar one = 1.0, zero = 0.0;
    PetscBLASInt bn0, bn1, bn2, bL0, bL1, bL2, b01, b12;
    PetscCall(PetscBLASIntCast(n0, &bn0)); PetscCall(PetscBLASIntCast(n1, &bn1)); PetscCall(PetscBLASIntCast(n2, &bn2));
    PetscCall(PetscBLASIntCast(L0, &bL0)); PetscCall(PetscBLASIntCast(L1, &bL1)); PetscCall(PetscBLASIntCast(L2, &bL2));
    PetscCall(PetscBLASIntCast(n0 * n1, &b01)); PetscCall(PetscBLASIntCast(L1 * L2, &b12));
    // G1 (n0 x L1 L2) = F_x (n0 x L0) C (L0 x L1 L2)
    PetscCallBLAS("BLASgemm", BLASgemm_("N", "N", &bn0, &b12, &bL0, &one, F[0].data(), &bn0, c, &bL0, &zero, G1.data(), &bn0));
    // G2_l (n0 x n1) = G1_l (n0 x L1) F_y^T
    for (PetscInt l = 0; l < L2; ++l)
        PetscCallBLAS("BLASgemm", BLASgemm_("N", "T", &bn0, &bn1, &bL1, &one, G1.data() + n0 * L1 * l, &bn0,
                                            F[1].data(), &bn1, &zero, G2.data() + n0 * n1 * l, &bn0));
    // G3 (n0 n1 x n2) = G2 (n0 n1 x L2) F_z^T
    PetscCallBLAS("BLASgemm", BLASgemm_("N", "T", &b01, &bn2, &bL2, &one, G2.data(), &b01, F[2].data(), &bn2, &zero, G3.data(), &b01));
    for (PetscInt i = 0; i < n0; ++i)
        for (PetscInt j = 0; j < n1; ++j)
            for (PetscInt l = 0; l < n2; ++l) out[(i * n1 + j) * n2 + l] = G3[i + n0 * (j + n1 * l)];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZTSurff::Interior(Vec psi, std::vector<PetscScalar> &out)
{
    PetscFunctionBeginUser;
    PetscInt nloc;
    PetscCall(VecGetLocalSize(psi, &nloc));
    PetscCheck(nloc == lwidth_[0] * lwidth_[1] * lwidth_[2], comm_, PETSC_ERR_PLIB, "t-SURFF: unexpected local layout");
    const PetscScalar *c;
    PetscCall(VecGetArrayRead(psi, &c));
    out.assign(nk_, 0.0);
    const PetscReal norm = std::pow(2 * PETSC_PI, -0.5 * dim_);
    auto phi = [&](const PetscReal kv[3]) {
        PetscReal f = C_;
        for (int d = 0; d < 3; ++d) f += volkovT_ * kv[d] * kv[d] / (2 * mass_) + kv[d] * B_[d] / mass_;
        return norm * Phase(f);
    };
    if (grid_) {
        PetscCall(Project(c, n_, k_, out.data()));
        PetscInt idx = 0;
        for (PetscInt i = 0; i < n_[0]; ++i)
            for (PetscInt j = 0; j < n_[1]; ++j)
                for (PetscInt l = 0; l < n_[2]; ++l, ++idx) {
                    const PetscReal kv[3] = {k_[0][i], k_[1][j], k_[2][l]};
                    out[idx] *= phi(kv);
                }
    } else {
        const PetscInt one[3] = {1, 1, 1};
        for (PetscInt q = 0; q < nk_; ++q) {
            const std::vector<PetscReal> kq[3] = {{list_[q][0]}, {list_[q][1]}, {list_[q][2]}};
            PetscCall(Project(c, one, kq, &out[q]));
            out[q] *= phi(list_[q].data());
        }
    }
    PetscCall(VecRestoreArrayRead(psi, &c));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZTSurff::Finish(const std::string &path, PetscReal t, Vec psi)
{
    PetscFunctionBeginUser;
    if (finished_) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(AddFlux(t, psi));
    finished_ = true;
    std::vector<PetscScalar> total(rank_ ? 0 : nk_), inner, innerTotal(rank_ ? 0 : nk_);
    PetscCallMPI(MPI_Reduce(amp_.data(), rank_ ? nullptr : total.data(), (int)nk_, MPIU_SCALAR, MPIU_SUM, 0, comm_));
    const bool interior = TDSEZParser::SurffInterior;
    if (interior) {
        PetscCall(Interior(psi, inner));
        PetscCallMPI(MPI_Reduce(inner.data(), rank_ ? nullptr : innerTotal.data(), (int)nk_, MPIU_SCALAR, MPIU_SUM, 0, comm_));
    }
    PetscLogDouble maxSeconds = 0;
    PetscCallMPI(MPI_Reduce(&seconds_, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm_));

    int status = 0;
    if (!rank_) {
        const hsize_t nk = (hsize_t)nk_, d = (hsize_t)dim_;
        std::vector<PetscReal> momenta(nk * d), amplitude(2 * nk), flux(interior ? 2 * nk : 0), pes(nk);
        for (hsize_t k = 0; k < nk; ++k) {
            if (grid_) {
                const PetscInt i = (PetscInt)k / (n_[1] * n_[2]), j = ((PetscInt)k / n_[2]) % n_[1], l = (PetscInt)k % n_[2];
                const PetscReal kv[3] = {k_[0][i], k_[1][j], k_[2][l]};
                for (hsize_t e = 0; e < d; ++e) momenta[k * d + e] = kv[e];
            } else {
                for (hsize_t e = 0; e < d; ++e) momenta[k * d + e] = list_[k][e];
            }
            if (interior) {
                flux[2 * k] = PetscRealPart(total[k]);
                flux[2 * k + 1] = PetscImaginaryPart(total[k]);
                total[k] += innerTotal[k];
            }
            amplitude[2 * k] = PetscRealPart(total[k]);
            amplitude[2 * k + 1] = PetscImaginaryPart(total[k]);
            pes[k] = std::norm(total[k]);
        }
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file < 0) status = -1;
        else {
            const hsize_t dm[2] = {nk, d}, da[2] = {nk, 2};
            status |= WriteReal(file, "momenta", 2, dm, momenta.data());
            status |= WriteReal(file, "amplitude", 2, da, amplitude.data());
            status |= WriteReal(file, "pes", 1, &nk, pes.data());
            if (interior) status |= WriteReal(file, "amplitude_flux", 2, da, flux.data());
            if (grid_) {
                const char *axes[3] = {"k_x", "k_y", "k_z"};
                hsize_t shape[3];
                for (int e = 0; e < dim_; ++e) {
                    shape[e] = (hsize_t)n_[e];
                    status |= WriteReal(file, axes[e], 1, &shape[e], k_[e].data());
                }
                status |= WriteReal(file, "pes_grid", (int)dim_, shape, pes.data());
            }
            const PetscReal scalars[][1] = {{prevT_}, {R_}, {mass_}, {V0_}, {TDSEZParser::Q},
                                            {(PetscReal)TDSEZParser::OutputStrideSurff}, {(PetscReal)nGlobalPoints_},
                                            {(PetscReal)nSamples_}, {interior ? 1.0 : 0.0}};
            const char *names[] = {"final_time", "radius", "exterior_mass", "exterior_potential", "charge",
                                   "stride", "n_surface_points", "n_samples", "interior"};
            for (int e = 0; e < 9; ++e) status |= WriteAttribute(file, names[e], 1, scalars[e]);
            status |= WriteAttribute(file, "A_final", d, A_);
            if (H5Fclose(file) < 0) status = -1;
        }
    }
    PetscCallMPI(MPI_Bcast(&status, 1, MPI_INT, 0, comm_));
    PetscCheck(status == 0, comm_, PETSC_ERR_FILE_WRITE, "t-SURFF: cannot write %s", path.c_str());
    TDSEZInfo(comm_).kv("t-SURFF", "%" PetscInt_FMT " samples, %.3f s (%.3g ms per sample) -> %s",
                          nSamples_, (double)maxSeconds, 1e3 * maxSeconds / PetscMax((PetscInt)1, nSamples_), path.c_str());
    PetscFunctionReturn(PETSC_SUCCESS);
}
