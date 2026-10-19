#include "tdsez_paths.hpp"
#include "tdsez_internal.hpp"

/**
 * @file diagnostics.cpp
 * @brief Runtime diagnostics: transition dipole matrix computation,
 *        momentum matrix precomputation, and length-velocity gauge
 *        consistency verification.
 * @author TDSEZ Project
 */

/// @brief Compute and print the unified transition dipole matrix
///        d_ij = <psi_i | D | psi_j> for all active dipole operators
///        (Dx, Dy, Dz). Explicitly conjugates the bra since this PETSc
///        build's VecTDot does not conjugate either argument. Caches
///        the upper triangle and fills the lower by Hermitian conjugation.
/// @param states    Vector of bound-state eigenvectors.
/// @param energies  Corresponding energy eigenvalues.
/// @param mgr       Manager holding the dipole operators (assembled or
///                  factorised); an axis without either is skipped.
/// @param fileName  Base filename for text output.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZCompUnifiedDipoleMatrix(
    const std::vector<Vec>& states,
    const std::vector<PetscReal>& energies,
    TDSEZManager *mgr,
    const std::string& fileName)
{
    PetscFunctionBeginUser;
    if (states.empty()) PetscFunctionReturn(PETSC_SUCCESS);

    const char* RED    = "\033[1;31m";
    const char* GREEN  = "\033[1;32m";
    const char* YELLOW = "\033[1;33m";
    const char* RESET  = "\033[0m";

    const char* fmt_header = "    %-13s %-19s %-21s %-15s %-13s %-7s\n";
    const char* fmt_data   = "    |%1" PetscInt_FMT "⟩   %14.4f   %18.8f   %16.6f   %14.2f   %+6" PetscInt_FMT "\n";
    const char* separator  = "    ─────────────────────────────────────────────────────────────────────────────────────────";

    PetscInt nStates = static_cast<PetscInt>(states.size());
    const bool Dx = mgr->HasOp(TDSEZTensorOps::DipX, mgr->Dx()), Dy = mgr->HasOp(TDSEZTensorOps::DipY, mgr->Dy()),
               Dz = mgr->HasOp(TDSEZTensorOps::DipZ, mgr->Dz());
    int activeDim = (Dx ? 1 : 0) + (Dy ? 1 : 0) + (Dz ? 1 : 0);

    // Precompute D|j> 
    std::vector<Vec> Dx_s(nStates, NULL), Dy_s(nStates, NULL), Dz_s(nStates, NULL);
    for (PetscInt j = 0; j < nStates; j++) {
        if (Dx) { PetscCall(VecDuplicate(states[j], &Dx_s[j])); PetscCall(mgr->ApplyOp(TDSEZTensorOps::DipX, mgr->Dx(), states[j], Dx_s[j])); }
        if (Dy) { PetscCall(VecDuplicate(states[j], &Dy_s[j])); PetscCall(mgr->ApplyOp(TDSEZTensorOps::DipY, mgr->Dy(), states[j], Dy_s[j])); }
        if (Dz) { PetscCall(VecDuplicate(states[j], &Dz_s[j])); PetscCall(mgr->ApplyOp(TDSEZTensorOps::DipZ, mgr->Dz(), states[j], Dz_s[j])); }
    }

    // Pre-conjugate the bra states ONCE: <psi_i|D|psi_j> = VecTDot(conj(psi_i), D psi_j).
    // VecTDot(x, y) = y^T x does not conjugate (VecDot(x, y) = y^H x would),
    // so the bra must be conjugated explicitly or complex eigenstates give the
    // wrong phase. (For real states conj is identity, so no change.)
    std::vector<Vec> statesConj(nStates, NULL);
    for (PetscInt i = 0; i < nStates; i++) {
        PetscCall(VecDuplicate(states[i], &statesConj[i]));
        PetscCall(VecCopy(states[i], statesConj[i]));
        PetscCall(VecConjugate(statesConj[i]));
    }

    // --- Cache all matrix elements (upper triangle + diagonal) ---
    // Layout: mat_x[i*nStates + j] for i <= j
    // For i > j, use conj(mat_x[j*nStates + i])
    std::vector<PetscScalar> mat_x(nStates * nStates, 0.0);
    std::vector<PetscScalar> mat_y(nStates * nStates, 0.0);
    std::vector<PetscScalar> mat_z(nStates * nStates, 0.0);

    for (PetscInt i = 0; i < nStates; i++) {
        for (PetscInt j = i; j < nStates; j++) {  // upper triangle only
            // <psi_i | D | psi_j> = VecTDot( conj(psi_i), D psi_j )
            // (bra explicitly conjugated; VecTDot does NOT conjugate
            //  the first arg in this PETSc build).
            if (Dx) PetscCall(VecTDot(statesConj[i], Dx_s[j], &mat_x[i*nStates + j]));
            if (Dy) PetscCall(VecTDot(statesConj[i], Dy_s[j], &mat_y[i*nStates + j]));
            if (Dz) PetscCall(VecTDot(statesConj[i], Dz_s[j], &mat_z[i*nStates + j]));
        }
    }

    // Helper to retrieve ⟨i|D|j⟩ using Hermitian symmetry
    auto getMx = [&](PetscInt i, PetscInt j) -> PetscScalar {
        return (i <= j) ? mat_x[i*nStates + j] : PetscConj(mat_x[j*nStates + i]);
    };
    auto getMy = [&](PetscInt i, PetscInt j) -> PetscScalar {
        return (i <= j) ? mat_y[i*nStates + j] : PetscConj(mat_y[j*nStates + i]);
    };
    auto getMz = [&](PetscInt i, PetscInt j) -> PetscScalar {
        return (i <= j) ? mat_z[i*nStates + j] : PetscConj(mat_z[j*nStates + i]);
    };

    // Build TDM summary from cached upper triangle
    struct TDMTableEntry {
        PetscInt i, j;
        PetscScalar mx, my, mz;
        PetscReal mag;
    };

    std::vector<TDMTableEntry> tdmSummary;
    tdmSummary.reserve(nStates * (nStates - 1) / 2);
    for (PetscInt i = 0; i < nStates; i++) {
        for (PetscInt j = i + 1; j < nStates; j++) {
            PetscScalar mx = getMx(i,j), my = getMy(i,j), mz = getMz(i,j);
            PetscReal m_sq = PetscRealPart(mx*PetscConj(mx) + my*PetscConj(my) + mz*PetscConj(mz));
            PetscReal m_abs = PetscSqrtReal(m_sq);
            if (m_abs < 1e-5) continue;
            tdmSummary.push_back({i, j, mx, my, mz, m_abs});
        }
    }

    //  Precompute total |f_ij| per state for branching ratio denominator 
    // BR(i->j) = |f_ij| / sum_k(|f_ik|, k!=i) * 100
    std::vector<PetscReal> fTotalPerState(nStates, 0.0);
    for (PetscInt i = 0; i < nStates; i++) {
        for (PetscInt j = 0; j < nStates; j++) {
            if (i == j) continue;
            PetscScalar mx = getMx(i,j), my = getMy(i,j), mz = getMz(i,j);
            PetscReal m_sq = PetscRealPart(mx*PetscConj(mx) + my*PetscConj(my) + mz*PetscConj(mz));
            if (PetscSqrtReal(m_sq) < 1e-6) continue;
            PetscReal dE = energies[j] - energies[i];
            PetscReal f  = (2.0 / (PetscReal)activeDim) * dE * m_sq;
            fTotalPerState[i] += std::abs(f);
        }
    }

    // Output destination: static/<name>_tdm.txt (name = the input file name
    // without extension; fileName is "DipoleMatrix_<input>"). Create the dir if needed.
    std::error_code tdm_ec;
    std::filesystem::create_directories("static", tdm_ec);
    std::string tdmPrefix = fileName;
    const std::string tdmTag = "DipoleMatrix_";
    if (tdmPrefix.rfind(tdmTag, 0) == 0) tdmPrefix = tdmPrefix.substr(tdmTag.size());
    const std::string tdmPath = TDSEZPaths::Static(tdmPrefix, "tdm.txt");

    PetscViewer log;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, tdmPath.c_str(), &log));

    PetscCall(PetscViewerASCIIPrintf(log, "┌────────────────────────────────────────────────────────────────────────────────────────┐\n"));
    PetscCall(PetscViewerASCIIPrintf(log, "│ SPECTROSCOPIC ANALYSIS: %dD-AB INITIO EIGENBASIS CHARACTERIZATION                      │\n", activeDim));
    PetscCall(PetscViewerASCIIPrintf(log, "└────────────────────────────────────────────────────────────────────────────────────────┘\n"));
    PetscCall(PetscViewerASCIIPrintf(log, "  System : Arbitrary Potential | Basis States: %-2" PetscInt_FMT "\n", nStates));
    PetscCall(PetscViewerASCIIPrintf(log, "──────────────────────────────────────────────────────────────────────────────────────────\n\n"));

    std::map<PetscInt, bool> dominantTransitions;

    for (PetscInt i = 0; i < nStates; i++) {
        // Diagonal expectation values from cache (i == j)
        PetscScalar ex_x = getMx(i,i), ex_y = getMy(i,i), ex_z = getMz(i,i);
        PetscReal symmetryErr = PetscSqrtReal(PetscRealPart(
            ex_x*PetscConj(ex_x) + ex_y*PetscConj(ex_y) + ex_z*PetscConj(ex_z)));

        std::string pRaw = (i % 2 == 0) ? "EVEN (+)" : "ODD  (-)";
        std::string parityLabel = (symmetryErr > 1e-7)
            ? (std::string(RED) + "ASYM (?)" + RESET)
            : (std::string(GREEN) + pRaw + RESET);

        PetscCall(PetscViewerASCIIPrintf(log, " STATE |%1" PetscInt_FMT "⟩  E = %s a.u.  [%s]\n", i, TDSEZ_fmtAu(energies[i]).c_str(), parityLabel.c_str()));
        PetscCall(PetscViewerASCIIPrintf(log, " └─ Centroid ⟨r⟩: (%8.1e, %8.1e, %8.1e) | Tol: %s\n",
            PetscRealPart(ex_x), PetscRealPart(ex_y), PetscRealPart(ex_z),
            (symmetryErr < 1e-9 ? "PASSED" : "WARN")));

        PetscCall(PetscViewerASCIIPrintf(log, fmt_header, "Target", "ΔE (au)", "|μ_ij|", "f_ij", "BR (%)", "Δn"));
        PetscCall(PetscViewerASCIIPrintf(log, "%s\n", separator));

        PetscReal M0 = 0;
        for (PetscInt j = 0; j < nStates; j++) {
            if (i == j) continue;
            PetscScalar mx = getMx(i,j), my = getMy(i,j), mz = getMz(i,j);
            PetscReal m_sq = PetscRealPart(mx*PetscConj(mx) + my*PetscConj(my) + mz*PetscConj(mz));
            PetscReal m_abs = PetscSqrtReal(m_sq);
            if (m_abs < 1e-5) continue;

            PetscReal dE = energies[j] - energies[i];
            PetscReal f = (2.0 / (PetscReal)activeDim) * dE * m_sq;
            M0 += f;

            PetscReal br = (fTotalPerState[i] > 1e-30)
                ? (std::abs(f) / fTotalPerState[i]) * 100.0
                : 0.0;
            if (br > 1.0) dominantTransitions[std::abs(j - i)] = true;

            PetscCall(PetscViewerASCIIPrintf(log, fmt_data, j, dE, m_abs, f, br, j - i));
        }

        PetscBool isTrusted = (i < (PetscInt)(nStates * 0.75));
        PetscReal dev = PetscAbsReal(M0 - 1.0);
        const char* color = GREEN;
        const char* status = "PASSED";
        if (isTrusted) {
            if (dev > 0.01) { color = RED; status = "FAILED (Incomplete Basis)"; }
        } else {
            color = YELLOW; status = "BUFFER (Truncated Basis)";
        }

        PetscCall(PetscViewerASCIIPrintf(log, "%s\n", separator));
        PetscCall(PetscViewerASCIIPrintf(log, "    [Sum Rule] %sTRK Σf_ij = %9.6f%s | Status: %s\n\n",
            color, M0, RESET, status));
    }

    // Spectral Summary
    PetscCall(PetscViewerASCIIPrintf(log, "──────────────────────────────────────────────────────────────────────────────────────────\n"));
    PetscCall(PetscViewerASCIIPrintf(log, " SPECTRAL SUMMARY | Main Resonances: Δn ∈ { "));
    for (auto const& [jump, _] : dominantTransitions) { PetscCall(PetscViewerASCIIPrintf(log, "±%" PetscInt_FMT " ", jump)); }
    PetscCall(PetscViewerASCIIPrintf(log, "}\n──────────────────────────────────────────────────────────────────────────────────────────\n"));

    // TDM Table
    PetscCall(PetscViewerASCIIPrintf(log, "\n UNIFIED SPECTROSCOPIC ANALYSIS: Transition Dipole Moments (TDM)\n"));
    PetscCall(PetscViewerASCIIPrintf(log, "========================================================================================\n"));
    for (const auto& t : tdmSummary) {
        PetscCall(PetscViewerASCIIPrintf(log, "⟨%02" PetscInt_FMT "|d|%02" PetscInt_FMT "⟩ | ", t.i, t.j));
        PetscCall(PetscViewerASCIIPrintf(log, "X:(%6.3f, %6.3f) | ", (double)PetscRealPart(t.mx), (double)PetscImaginaryPart(t.mx)));
        PetscCall(PetscViewerASCIIPrintf(log, "Y:(%6.3f, %6.3f) | ", (double)PetscRealPart(t.my), (double)PetscImaginaryPart(t.my)));
        PetscCall(PetscViewerASCIIPrintf(log, "Z:(%6.3f, %6.3f) | ", (double)PetscRealPart(t.mz), (double)PetscImaginaryPart(t.mz)));
        PetscCall(PetscViewerASCIIPrintf(log, "|μ|= %7.4f\n", (double)t.mag));
    }
    PetscCall(PetscViewerASCIIPrintf(log, "========================================================================================\n"));

    // Cleanup
    for (PetscInt i = 0; i < nStates; i++) {
        if (Dx_s[i]) PetscCall(VecDestroy(&Dx_s[i]));
        if (Dy_s[i]) PetscCall(VecDestroy(&Dy_s[i]));
        if (Dz_s[i]) PetscCall(VecDestroy(&Dz_s[i]));
        if (statesConj[i]) PetscCall(VecDestroy(&statesConj[i]));
    }
    PetscCall(PetscViewerDestroy(&log));
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief Save a single assembled operator matrix to a PETSc binary file,
///        reloadable later with MatLoad. No-op if mat is NULL. MatView is
///        collective, so every rank must call it.
/// @param mat      Operator matrix to save (may be NULL).
/// @param filename Output binary file path.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZSaveOperatorMatrix(Mat mat, const std::string& filename)
{
    PetscFunctionBeginUser;
    if (!mat) PetscFunctionReturn(PETSC_SUCCESS);

    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(),
                                    FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(mat, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    TDSEZInfo().kv("Operator matrix written", "%s", filename.c_str());
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief Make a single complex eigenstate real-valued by projecting out
///        its imaginary part under the mass matrix M. Computes the
///        expectation <psi|M|Im(psi)> and subtracts the corresponding
///        real state to produce a real eigenvector (up to numerical
///        precision).
/// @param v_complex  Complex eigenvector (modified in place).
/// @param Vre        Work vector for the real part.
/// @param M          Mass (overlap) matrix.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZMakeStateReal2(Vec v_complex, Vec Vre, Mat M)
{
    PetscFunctionBegin;
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)v_complex,&comm);CHKERRQ(ierr);
    MPI_Comm_rank(comm, &rank);

    // Copy complex vector
    ierr = VecCopy(v_complex, Vre);CHKERRQ(ierr);

    // Get local ownership
    PetscInt start,end;
    ierr = VecGetOwnershipRange(v_complex,&start,&end);CHKERRQ(ierr);

    const PetscScalar *array;
    ierr = VecGetArrayRead(v_complex,&array);CHKERRQ(ierr);

    PetscReal local_phase = 0.0;
    PetscBool found = PETSC_FALSE;
    for (PetscInt i=0;i<end-start;++i) 
    {
        if (PetscAbsScalar(array[i])>1e-14) 
        {
            local_phase = std::atan2(PetscImaginaryPart(array[i]),PetscRealPart(array[i]));
            found = PETSC_TRUE;
            break;
        }
    }
    ierr = VecRestoreArrayRead(v_complex,&array);CHKERRQ(ierr);

    // Determine global phase
    struct {PetscReal phase; int found;} local = {local_phase, found ? 1 : 0}, global;
    MPI_Allreduce(&local,&global,1,MPI_DOUBLE_INT,MPI_MINLOC,comm);
    PetscReal phase = global.phase;

    // Rotate vector by exp(-i*phase)
    PetscScalar *xarr;
    ierr = VecGetArray(Vre,&xarr);CHKERRQ(ierr);
    ierr = VecGetArrayRead(v_complex,&array);CHKERRQ(ierr);
    for (PetscInt i=0;i<end-start;i++) {
        PetscScalar z = array[i] * PetscExpComplex(-PETSC_i * phase);
        xarr[i] = PetscRealPart(z);
    }
    ierr = VecRestoreArrayRead(v_complex,&array);CHKERRQ(ierr);
    ierr = VecRestoreArray(Vre,&xarr);CHKERRQ(ierr);

    // Assemble
    ierr = VecAssemblyBegin(Vre);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Vre);CHKERRQ(ierr);

    // Normalize
    Vec tmp;
    ierr = VecDuplicate(Vre,&tmp);CHKERRQ(ierr);
    ierr = MatMult(M,Vre,tmp);CHKERRQ(ierr);
    PetscScalar norm;
    ierr = VecDot(tmp,Vre,&norm);CHKERRQ(ierr);
    norm = PetscAbsScalar(norm);
    ierr = VecScale(Vre,1.0/PetscSqrtReal(norm));CHKERRQ(ierr);
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


/// @brief Verify length-velocity gauge consistency: checks that the
///        momentum matrix v_mn satisfies [D, H] = i*omega*v (the
///        commutator relation linking the length and velocity gauges).
///        Reports per-element residuals and overall pass/fail status.
/// @param TDSEZ  Pointer to the TDSEZManager holding bound states and
///               operators.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZCheckLengthVelocity(TDSEZManager *TDSEZ)
{
    PetscFunctionBeginUser;

    PetscCheck(TDSEZ, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "TDSEZ is NULL");

    const PetscInt N = TDSEZ->NPOP;
    MPI_Comm comm = PETSC_COMM_WORLD;

    const PetscBool doX = (PetscBool)(TDSEZ->hasX && TDSEZ->HasOp(TDSEZTensorOps::DipX, TDSEZ->Dx()) &&
                                      !TDSEZ->vMat_x.empty());
    const PetscBool doY = (PetscBool)(TDSEZ->hasY && TDSEZ->HasOp(TDSEZTensorOps::DipY, TDSEZ->Dy()) &&
                                      !TDSEZ->vMat_y.empty());
    const PetscBool doZ = (PetscBool)(TDSEZ->hasZ && TDSEZ->HasOp(TDSEZTensorOps::DipZ, TDSEZ->Dz()) &&
                                      !TDSEZ->vMat_z.empty());

    if (!doX && !doY && !doZ) {
        TDSEZInfo info;
        info.section("LENGTH-VELOCITY GAUGE CHECK");
        info.kv("Status", "skipped (no dipole or velocity operator)");
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    std::vector<Vec> states(N);
    for (PetscInt i = 0; i < N; ++i) {
        states[i] = TDSEZ->boundstates[i];
    }

    std::vector<PetscScalar> dMat_x((size_t)N * N, 0.0);
    std::vector<PetscScalar> dMat_y((size_t)N * N, 0.0);
    std::vector<PetscScalar> dMat_z((size_t)N * N, 0.0);
    std::vector<PetscScalar> dots(N, 0.0);

    Vec tmp;
    PetscCall(VecDuplicate(TDSEZ->boundstates[0], &tmp));

    for (PetscInt n = 0; n < N; ++n) {
        if (doX) {
            PetscCall(TDSEZ->ApplyOp(TDSEZTensorOps::DipX, TDSEZ->Dx(), TDSEZ->boundstates[n], tmp));
            PetscCall(VecMDot(tmp, N, states.data(), dots.data()));
            for (PetscInt m = 0; m < N; ++m) {
                dMat_x[(size_t)m * N + n] = dots[m];   // d_mn = <phi_m|D|phi_n>
            }
        }

        if (doY) {
            PetscCall(TDSEZ->ApplyOp(TDSEZTensorOps::DipY, TDSEZ->Dy(), TDSEZ->boundstates[n], tmp));
            PetscCall(VecMDot(tmp, N, states.data(), dots.data()));
            for (PetscInt m = 0; m < N; ++m) {
                dMat_y[(size_t)m * N + n] = dots[m];   // d_mn = <phi_m|D|phi_n>
            }
        }

        if (doZ) {
            PetscCall(TDSEZ->ApplyOp(TDSEZTensorOps::DipZ, TDSEZ->Dz(), TDSEZ->boundstates[n], tmp));
            PetscCall(VecMDot(tmp, N, states.data(), dots.data()));
            for (PetscInt m = 0; m < N; ++m) {
                dMat_z[(size_t)m * N + n] = dots[m];   // d_mn = <phi_m|D|phi_n>
            }
        }
    }

    std::vector<PetscReal> eps(N, 0.0);

    for (PetscInt n = 0; n < N; ++n) {
        PetscScalar en, norm;

        PetscCall(MatMult(TDSEZ->H(), TDSEZ->boundstates[n], tmp));
        PetscCall(VecDot(tmp, TDSEZ->boundstates[n], &en));

        PetscCall(MatMult(TDSEZ->M(), TDSEZ->boundstates[n], tmp));
        PetscCall(VecDot(tmp, TDSEZ->boundstates[n], &norm));

        const PetscReal nrm = PetscRealPart(norm);

        PetscCheck(nrm > PETSC_SMALL, comm, PETSC_ERR_ARG_WRONGSTATE,
                   "State %" PetscInt_FMT " has near-zero M norm", n);

        eps[n] = PetscRealPart(en) / nrm;
    }

    PetscCall(VecDestroy(&tmp));

    if (TDSEZ->rank == 0) {
        TDSEZInfo info(PETSC_COMM_SELF);
        info.section("LENGTH-VELOCITY GAUGE CHECK");
        info.kv("Axes", "%s%s%s", doX ? "x" : "", doY ? "y" : "", doZ ? "z" : "");

        PetscBool lv_ok = PETSC_TRUE;
        PetscReal max_lv = 0.0;
        PetscInt checked = 0;
        PetscInt skipped = 0;

        const PetscReal omega_abs_tol = 1e-4;
        const PetscReal lv_abs_tol    = 1e-8;
        const PetscReal lv_rel_tol    = 1e-4;

        for (PetscInt m = 0; m < N; ++m) {
            for (PetscInt n = 0; n < N; ++n) {
                if (m == n) continue;

                // v_mn = <m|i[H,r]|n> = i (eps_m - eps_n) d_mn
                const PetscReal omega_mn = eps[m] - eps[n];

                if (PetscAbsReal(omega_mn) < omega_abs_tol) {
                    ++skipped;
                    continue;
                }

                PetscBool warn = PETSC_FALSE;
                PetscReal ex = 0.0, ey = 0.0, ez = 0.0;
                PetscReal tx = 0.0, ty = 0.0, tz = 0.0;

                if (doX) {
                    const PetscScalar lhs = TDSEZ->vMat_x[(size_t)m * N + n];
                    const PetscScalar rhs = PETSC_i * omega_mn * dMat_x[(size_t)m * N + n];
                    const PetscScalar res = lhs - rhs;

                    const PetscReal scale =
                        PetscMax((PetscReal)1.0,
                                 PetscMax(PetscAbsScalar(lhs), PetscAbsScalar(rhs)));

                    tx = lv_abs_tol + lv_rel_tol * scale;
                    ex = PetscAbsScalar(res);

                    max_lv = PetscMax(max_lv, ex);
                    if (ex > tx) warn = PETSC_TRUE;
                }

                if (doY) {
                    const PetscScalar lhs = TDSEZ->vMat_y[(size_t)m * N + n];
                    const PetscScalar rhs = PETSC_i * omega_mn * dMat_y[(size_t)m * N + n];
                    const PetscScalar res = lhs - rhs;

                    const PetscReal scale =
                        PetscMax((PetscReal)1.0,
                                 PetscMax(PetscAbsScalar(lhs), PetscAbsScalar(rhs)));

                    ty = lv_abs_tol + lv_rel_tol * scale;
                    ey = PetscAbsScalar(res);

                    max_lv = PetscMax(max_lv, ey);
                    if (ey > ty) warn = PETSC_TRUE;
                }

                if (doZ) {
                    const PetscScalar lhs = TDSEZ->vMat_z[(size_t)m * N + n];
                    const PetscScalar rhs = PETSC_i * omega_mn * dMat_z[(size_t)m * N + n];
                    const PetscScalar res = lhs - rhs;

                    const PetscReal scale =
                        PetscMax((PetscReal)1.0,
                                 PetscMax(PetscAbsScalar(lhs), PetscAbsScalar(rhs)));

                    tz = lv_abs_tol + lv_rel_tol * scale;
                    ez = PetscAbsScalar(res);

                    max_lv = PetscMax(max_lv, ez);
                    if (ez > tz) warn = PETSC_TRUE;
                }

                ++checked;

                if (warn) {
                    std::string parts;
                    char b[64];
                    if (doX) { snprintf(b, sizeof(b), ", x %.3e (tol %.3e)", (double)ex, (double)tx); parts += b; }
                    if (doY) { snprintf(b, sizeof(b), ", y %.3e (tol %.3e)", (double)ey, (double)ty); parts += b; }
                    if (doZ) { snprintf(b, sizeof(b), ", z %.3e (tol %.3e)", (double)ez, (double)tz); parts += b; }
                    info.warn("length-velocity residual for |%" PetscInt_FMT "> <- |%" PetscInt_FMT ">, omega = %+.6e%s",
                              n, m, (double)omega_mn, parts.c_str());
                    lv_ok = PETSC_FALSE;
                }
            }
        }

        info.kv("Pairs checked, skipped", "%" PetscInt_FMT ", %" PetscInt_FMT " (near-degenerate)", checked, skipped);
        info.kv("Max length-vel residual", "%.2e  %s", (double)max_lv, lv_ok ? "passed" : "FAILED");
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief Precompute the momentum matrix v_mn = <psi_m | Vel | psi_n> for
///        all active velocity operators (VelX, VelY, VelZ). Performs
///        Hermiticity and diagonal-reality checks, and reports global
///        operator assembly norms ||Vel - Vel^H||.
/// @param TDSEZ  Pointer to the TDSEZManager holding bound states and
///               velocity operators.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZPrecomputeMomentumMatrix(TDSEZManager *TDSEZ)
{
    PetscFunctionBeginUser;

    const PetscInt N = TDSEZ->NPOP;
    TDSEZ->vMat_x.assign(N * N, 0.0);
    TDSEZ->vMat_y.assign(N * N, 0.0);
    TDSEZ->vMat_z.assign(N * N, 0.0);
    if (N == 0) PetscFunctionReturn(PETSC_SUCCESS);  // NBoundStates = 0

    Vec tmp;
    PetscCall(VecDuplicate(TDSEZ->boundstates[0], &tmp));

    for (PetscInt n = 0; n < N; n++)
    {
        if (TDSEZ->hasX && TDSEZ->HasOp(TDSEZTensorOps::VelX, TDSEZ->VelX())) {
            PetscCall(TDSEZ->ApplyOp(TDSEZTensorOps::VelX, TDSEZ->VelX(), TDSEZ->boundstates[n], tmp));
            for (PetscInt m = 0; m < N; m++) {
                PetscScalar val;
                // VecDot(x, y) = y^H x, so this is <phi_m|Vel|phi_n> = v_mn
                PetscCall(VecDot(tmp, TDSEZ->boundstates[m], &val));
                TDSEZ->vMat_x[m*N+n] = val;
            }
        }
        if (TDSEZ->hasY && TDSEZ->HasOp(TDSEZTensorOps::VelY, TDSEZ->VelY())) {
            PetscCall(TDSEZ->ApplyOp(TDSEZTensorOps::VelY, TDSEZ->VelY(), TDSEZ->boundstates[n], tmp));
            for (PetscInt m = 0; m < N; m++) {
                PetscScalar val;
                // VecDot(x, y) = y^H x, so this is <phi_m|Vel|phi_n> = v_mn
                PetscCall(VecDot(tmp, TDSEZ->boundstates[m], &val));
                TDSEZ->vMat_y[m*N+n] = val;
            }
        }
        if (TDSEZ->hasZ && TDSEZ->HasOp(TDSEZTensorOps::VelZ, TDSEZ->VelZ())) {
            PetscCall(TDSEZ->ApplyOp(TDSEZTensorOps::VelZ, TDSEZ->VelZ(), TDSEZ->boundstates[n], tmp));
            for (PetscInt m = 0; m < N; m++) {
                PetscScalar val;
                // VecDot(x, y) = y^H x, so this is <phi_m|Vel|phi_n> = v_mn
                PetscCall(VecDot(tmp, TDSEZ->boundstates[m], &val));
                TDSEZ->vMat_z[m*N+n] = val;
            }
        }
    }

    PetscCall(VecDestroy(&tmp));

    // Checks of the bound-state momentum matrix and of the velocity operators.
    TDSEZInfo info;
    info.section("MOMENTUM MATRIX CHECKS");
    const bool on[3] = {TDSEZ->hasX && TDSEZ->HasOp(TDSEZTensorOps::VelX, TDSEZ->VelX()),
                        TDSEZ->hasY && TDSEZ->HasOp(TDSEZTensorOps::VelY, TDSEZ->VelY()),
                        TDSEZ->hasZ && TDSEZ->HasOp(TDSEZTensorOps::VelZ, TDSEZ->VelZ())};
    const std::vector<PetscScalar> *vm[3] = {&TDSEZ->vMat_x, &TDSEZ->vMat_y, &TDSEZ->vMat_z};
    const PetscReal tol = 1e-10;

    // 1. Real diagonal: v_nn = <n|Vel|n> of a real bound state.
    PetscReal maxDiag = 0.0;
    for (int a = 0; a < 3; ++a) {
        if (!on[a]) continue;
        for (PetscInt n = 0; n < N; n++) {
            const PetscReal e = PetscAbsReal(PetscImaginaryPart((*vm[a])[n * N + n]));
            maxDiag = PetscMax(maxDiag, e);
            if (e > tol) info.warn("|Im v_nn| = %.2e on the %c axis for n = %" PetscInt_FMT, (double)e, "xyz"[a], n);
        }
    }
    info.kv("max |Im v_nn|", "%.2e  %s", (double)maxDiag, maxDiag <= tol ? "passed" : "FAILED");

    // 2. Hermiticity of the matrix elements.
    PetscReal maxHerm = 0.0;
    for (int a = 0; a < 3; ++a) {
        if (!on[a]) continue;
        for (PetscInt m = 0; m < N; m++)
            for (PetscInt n = m + 1; n < N; n++) {
                const PetscReal e = PetscAbsScalar((*vm[a])[m * N + n] - PetscConj((*vm[a])[n * N + m]));
                maxHerm = PetscMax(maxHerm, e);
                if (e > tol)
                    info.warn("|v_mn - conj(v_nm)| = %.2e on the %c axis for (m, n) = (%" PetscInt_FMT ", %" PetscInt_FMT ")",
                              (double)e, "xyz"[a], m, n);
            }
    }
    info.kv("max |v_mn - conj(v_nm)|", "%.2e  %s", (double)maxHerm, maxHerm <= tol ? "passed" : "FAILED");

    // 3. The assembled velocity operators are Hermitian (a factorised velocity,
    //    -i/(2m) (G - G^T) with M on the other axes, is Hermitian by construction).
    Mat vel[3] = {TDSEZ->VelX(), TDSEZ->VelY(), TDSEZ->VelZ()};
    for (int a = 0; a < 3; ++a) {
        if (!on[a] || !vel[a]) continue;
        Mat VH;
        PetscCall(MatHermitianTranspose(vel[a], MAT_INITIAL_MATRIX, &VH));
        PetscCall(MatAXPY(VH, -1.0, vel[a], DIFFERENT_NONZERO_PATTERN));
        PetscReal nrm;
        PetscCall(MatNorm(VH, NORM_FROBENIUS, &nrm));
        PetscCall(MatDestroy(&VH));
        char lab[40];
        snprintf(lab, sizeof(lab), "||Vel%c - Vel%c^H||_F", "XYZ"[a], "XYZ"[a]);
        info.kv(lab, "%.2e  %s", (double)nrm, nrm < 1e-8 ? "passed" : "FAILED");
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}