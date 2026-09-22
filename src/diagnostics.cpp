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
///        build's VecDot does not conjugate the first argument. Caches
///        the upper triangle and fills the lower by Hermitian conjugation.
/// @param states    Vector of bound-state eigenvectors.
/// @param energies  Corresponding energy eigenvalues.
/// @param Dx        X dipole operator (may be NULL if inactive).
/// @param Dy        Y dipole operator (may be NULL if inactive).
/// @param Dz        Z dipole operator (may be NULL if inactive).
/// @param fileName  Base filename for text output.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZCompUnifiedDipoleMatrix(
    const std::vector<Vec>& states,
    const std::vector<PetscReal>& energies,
    Mat Dx, Mat Dy, Mat Dz,


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
    int activeDim = (Dx ? 1 : 0) + (Dy ? 1 : 0) + (Dz ? 1 : 0);

    // Precompute D|j> 
    std::vector<Vec> Dx_s(nStates, NULL), Dy_s(nStates, NULL), Dz_s(nStates, NULL);
    for (PetscInt j = 0; j < nStates; j++) {
        if (Dx) { PetscCall(VecDuplicate(states[j], &Dx_s[j])); PetscCall(MatMult(Dx, states[j], Dx_s[j])); }
        if (Dy) { PetscCall(VecDuplicate(states[j], &Dy_s[j])); PetscCall(MatMult(Dy, states[j], Dy_s[j])); }
        if (Dz) { PetscCall(VecDuplicate(states[j], &Dz_s[j])); PetscCall(MatMult(Dz, states[j], Dz_s[j])); }
    }

    // Pre-conjugate the bra states ONCE: <psi_i|D|psi_j> = VecDot(conj(psi_i), D psi_j).
    // In this PETSc build VecDot/VecTDot do NOT conjugate the first argument,
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
            // <psi_i | D | psi_j> = VecDot( conj(psi_i), D psi_j )
            // (bra explicitly conjugated; VecDot/VecTDot do NOT conjugate
            //  the first arg in this PETSc build).
            if (Dx) PetscCall(VecDot(statesConj[i], Dx_s[j], &mat_x[i*nStates + j]));
            if (Dy) PetscCall(VecDot(statesConj[i], Dy_s[j], &mat_y[i*nStates + j]));
            if (Dz) PetscCall(VecDot(statesConj[i], Dz_s[j], &mat_z[i*nStates + j]));
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

    // Output destination: write the TDM info into a "static" sub-directory,
    // named TDMInfo_<prefix>.txt (prefix = the input-file stem, as in the
    // legacy "DipoleMatrix_<prefix>" convention). Create the dir if needed.
    std::error_code tdm_ec;
    std::filesystem::create_directories("static", tdm_ec);
    std::string tdmPrefix = fileName;
    const std::string tdmTag = "DipoleMatrix_";
    if (tdmPrefix.rfind(tdmTag, 0) == 0) tdmPrefix = tdmPrefix.substr(tdmTag.size());
    const std::string tdmPath = "static/TDMInfo_" + tdmPrefix + ".txt";

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

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "      [SaveDipoleMatrix] wrote operator matrix to %s\n",
        filename.c_str()));
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


/// @brief Make all bound-state eigenstates real-valued by applying
///        TDSEZMakeStateReal2 to each state. Real states simplify the
///        dipole matrix computation and are required for certain
///        post-processing tools.
/// @param states  Vector of bound-state eigenvectors (modified in place).
/// @param M       Mass (overlap) matrix.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode MakeStatesReal(std::vector<Vec>& states, Mat M)
{
    PetscFunctionBeginUser;

    Vec Mpsi;
    PetscCall(VecDuplicate(states[0], &Mpsi));

    for (auto& psi : states) {

        // 1. Find global phase via <psi_conj|M|psi_conj> = sum conj(psi_i)^2 * M_ij
        Vec psi_conj;
        PetscCall(VecDuplicate(psi, &psi_conj));
        PetscCall(VecCopy(psi, psi_conj));
        PetscCall(VecConjugate(psi_conj));

        // M|psi_conj>
        PetscCall(MatMult(M, psi_conj, Mpsi));

        // <psi_conj|M|psi_conj>
        PetscScalar dot;
        PetscCall(VecDot(Mpsi, psi_conj, &dot));

        // Extract global phase and fix
        PetscReal dotMag = PetscAbsScalar(dot);
        if (dotMag > 1e-12) {
            PetscReal   angle = 0.5 * PetscAtan2Real(
                                    PetscImaginaryPart(PetscConj(dot / (PetscScalar)dotMag)),
                                    PetscRealPart(PetscConj(dot / (PetscScalar)dotMag)));
            PetscScalar fix   = PetscCMPLX(PetscCosReal(angle), PetscSinReal(angle));
            PetscCall(VecScale(psi, fix));
        }
        PetscCall(VecDestroy(&psi_conj));

        // Zero imaginary part
        PetscCall(VecRealPart(psi));

        // Renormalize with M: norm = sqrt(<psi|M|psi>)
        PetscCall(MatMult(M, psi, Mpsi));
        PetscScalar norm_sq;
        PetscCall(VecDot(Mpsi, psi, &norm_sq));
        PetscReal norm = PetscSqrtReal(PetscRealPart(norm_sq));
        if (norm > 1e-12)
            PetscCall(VecScale(psi, 1.0 / (PetscScalar)norm));
    }

    PetscCall(VecDestroy(&Mpsi));
    PetscFunctionReturn(PETSC_SUCCESS);
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

    const PetscBool doX = (PetscBool)(TDSEZ->hasX && TDSEZ->Dx() &&
                                      TDSEZ->VelX() && !TDSEZ->vMat_x.empty());
    const PetscBool doY = (PetscBool)(TDSEZ->hasY && TDSEZ->Dy() &&
                                      TDSEZ->VelY() && !TDSEZ->vMat_y.empty());
    const PetscBool doZ = (PetscBool)(TDSEZ->hasZ && TDSEZ->Dz() &&
                                      TDSEZ->VelZ() && !TDSEZ->vMat_z.empty());

    if (!doX && !doY && !doZ) {
        if (TDSEZ->rank == 0) {
            PetscPrintf(PETSC_COMM_SELF, "  ══════════════════════════════════════════════════════════════════════\n");
            PetscPrintf(PETSC_COMM_SELF, "    ▸ GAUGE INVARIANCE CONSISTENCY CHECK\n");
            PetscPrintf(PETSC_COMM_SELF, "  ══════════════════════════════════════════════════════════════════════\n");
            PetscPrintf(PETSC_COMM_SELF, "      Status: SKIPPED (No Dx/Dy/Dz or VelX/VelY/VelZ available)\n");
            PetscPrintf(PETSC_COMM_SELF, "  ══════════════════════════════════════════════════════════════════════\n");
        }
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
            PetscCall(MatMult(TDSEZ->Dx(), TDSEZ->boundstates[n], tmp));
            PetscCall(VecMDot(tmp, N, states.data(), dots.data()));
            for (PetscInt m = 0; m < N; ++m) {
                dMat_x[(size_t)m * N + n] = PetscConj(dots[m]);
            }
        }

        if (doY) {
            PetscCall(MatMult(TDSEZ->Dy(), TDSEZ->boundstates[n], tmp));
            PetscCall(VecMDot(tmp, N, states.data(), dots.data()));
            for (PetscInt m = 0; m < N; ++m) {
                dMat_y[(size_t)m * N + n] = PetscConj(dots[m]);
            }
        }

        if (doZ) {
            PetscCall(MatMult(TDSEZ->Dz(), TDSEZ->boundstates[n], tmp));
            PetscCall(VecMDot(tmp, N, states.data(), dots.data()));
            for (PetscInt m = 0; m < N; ++m) {
                dMat_z[(size_t)m * N + n] = PetscConj(dots[m]);
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
        PetscPrintf(PETSC_COMM_SELF, "  ══════════════════════════════════════════════════════════════════════\n");
        PetscPrintf(PETSC_COMM_SELF, "    ▸ LENGTH-VELOCITY GAUGE CONSISTENCY CHECK [%s%s%s%s%s]\n",
            doX ? "X" : "", (doX && (doY||doZ)) ? "," : "", doY ? "Y" : "",
            (doX||doY) && doZ ? "," : "", doZ ? "Z" : "");
        PetscPrintf(PETSC_COMM_SELF, "  ══════════════════════════════════════════════════════════════════════\n\n");

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

                const PetscReal omega_mn = eps[n] - eps[m];

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
                    PetscPrintf(PETSC_COMM_SELF,
                        "      |%" PetscInt_FMT "> <- |%" PetscInt_FMT ">  omega = %+.6e",
                        n, m, (double)omega_mn);

                    if (doX) {
                        PetscPrintf(PETSC_COMM_SELF,
                            "  Dx = %.3e (tol=%.3e)", (double)ex, (double)tx);
                    }

                    if (doY) {
                        PetscPrintf(PETSC_COMM_SELF,
                            "  Dy = %.3e (tol=%.3e)", (double)ey, (double)ty);
                    }

                    if (doZ) {
                        PetscPrintf(PETSC_COMM_SELF,
                            "  Dz = %.3e (tol=%.3e)", (double)ez, (double)tz);
                    }

                    PetscPrintf(PETSC_COMM_SELF, " <-- WARNING\n");
                    lv_ok = PETSC_FALSE;
                }
            }
        }

        PetscPrintf(PETSC_COMM_SELF,
            "      Checked pairs            = %" PetscInt_FMT "\n"
            "      Skipped near-degenerate  = %" PetscInt_FMT "\n",
            checked, skipped);
        PetscPrintf(PETSC_COMM_SELF, "      Max length-vel residual  = %.2e\n", (double)max_lv);
        PetscPrintf(PETSC_COMM_SELF, "\n      Status: %s\n", lv_ok ? "PASSED" : "FAILED");
        PetscPrintf(PETSC_COMM_SELF, "  ══════════════════════════════════════════════════════════════════════\n");
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

    Vec tmp;
    PetscCall(VecDuplicate(TDSEZ->boundstates[0], &tmp));

    for (PetscInt n = 0; n < N; n++)
    {
        if (TDSEZ->hasX && TDSEZ->VelX() != PETSC_NULLPTR) {
            PetscCall(MatMult(TDSEZ->VelX(), TDSEZ->boundstates[n], tmp));
            for (PetscInt m = 0; m < N; m++) {
                PetscScalar val;
                PetscCall(VecDot(TDSEZ->boundstates[m], tmp, &val));
                TDSEZ->vMat_x[m*N+n] = val;
            }
        }
        if (TDSEZ->hasY && TDSEZ->VelY() != PETSC_NULLPTR) {
            PetscCall(MatMult(TDSEZ->VelY(), TDSEZ->boundstates[n], tmp));
            for (PetscInt m = 0; m < N; m++) {
                PetscScalar val;
                PetscCall(VecDot(TDSEZ->boundstates[m], tmp, &val));
                TDSEZ->vMat_y[m*N+n] = val;
            }
        }
        if (TDSEZ->hasZ && TDSEZ->VelZ() != PETSC_NULLPTR) {
            PetscCall(MatMult(TDSEZ->VelZ(), TDSEZ->boundstates[n], tmp));
            for (PetscInt m = 0; m < N; m++) {
                PetscScalar val;
                PetscCall(VecDot(TDSEZ->boundstates[m], tmp, &val));
                TDSEZ->vMat_z[m*N+n] = val;
            }
        }
    }

    PetscCall(VecDestroy(&tmp));

    /* ----------------------------------------------------------------
       Check 1: Diagonal Purely Real (Hermitian)
       ---------------------------------------------------------------- */
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");
    PetscPrintf(PETSC_COMM_WORLD, "    ▸ MOMENTUM MATRIX DIAGONAL CHECK (Expect < 1e-10) \n");
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n\n");
    
    PetscBool diag_ok = PETSC_TRUE;
    for (PetscInt n = 0; n < N; n++)
    {
        PetscReal dx = TDSEZ->hasX && TDSEZ->VelX() != PETSC_NULLPTR
                     ? PetscAbsReal(PetscImaginaryPart(TDSEZ->vMat_x[n*N+n])) : 0.0;
        PetscReal dy = TDSEZ->hasY && TDSEZ->VelY() != PETSC_NULLPTR
                     ? PetscAbsReal(PetscImaginaryPart(TDSEZ->vMat_y[n*N+n])) : 0.0;
        PetscPrintf(PETSC_COMM_WORLD,
            "      n=%-2d  |Im(v_nn^x)| = %-11.2e  |Im(v_nn^y)| = %-11.2e %s\n",
            (int)n, dx, dy,
            (dx > 1e-10 || dy > 1e-10) ? " <-- WARNING" : "");
        if (dx > 1e-10 || dy > 1e-10) diag_ok = PETSC_FALSE;
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n      Status: %s\n", diag_ok ? "PASSED" : "FAILED");

    /* ----------------------------------------------------------------
       Check 2: Hermiticity |v_mn - conj(v_nm)|
       ---------------------------------------------------------------- */
    PetscPrintf(PETSC_COMM_WORLD, "\n  ══════════════════════════════════════════════════════════════════════\n");
    PetscPrintf(PETSC_COMM_WORLD, "    ▸ STATE HERMITICITY ELEMENT CHECK (Expect < 1e-10) \n");
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n\n");
    
    PetscBool herm_ok  = PETSC_TRUE;
    PetscReal max_herr = 0.0;
    for (PetscInt m = 0; m < N; m++)
    for (PetscInt n = m+1; n < N; n++)
    {
        PetscReal ex = 0.0, ey = 0.0;
        if (TDSEZ->hasX && TDSEZ->VelX() != PETSC_NULLPTR) {
            PetscScalar hx = TDSEZ->vMat_x[m*N+n] - PetscConj(TDSEZ->vMat_x[n*N+m]);
            ex = PetscAbsScalar(hx);
            if (ex > max_herr) max_herr = ex;
        }
        if (TDSEZ->hasY && TDSEZ->VelY() != PETSC_NULLPTR) {
            PetscScalar hy = TDSEZ->vMat_y[m*N+n] - PetscConj(TDSEZ->vMat_y[n*N+m]);
            ey = PetscAbsScalar(hy);
            if (ey > max_herr) max_herr = ey;
        }
        if (ex > 1e-10 || ey > 1e-10) {
            PetscPrintf(PETSC_COMM_WORLD,
                "      [%d,%d]  x = %11.2e  y = %11.2e  <-- WARNING\n",
                (int)m, (int)n, ex, ey);
            herm_ok = PETSC_FALSE;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, "      Max hermitian error = %.2e\n", max_herr);
    PetscPrintf(PETSC_COMM_WORLD, "      Status: %s\n", herm_ok ? "PASSED" : "FAILED");

    /* ----------------------------------------------------------------
       Check 3: Full Operator Identity Norms ||Vel - Vel^H||
       ---------------------------------------------------------------- */
    PetscPrintf(PETSC_COMM_WORLD, "\n  ══════════════════════════════════════════════════════════════════════\n");
    PetscPrintf(PETSC_COMM_WORLD, "    ▸ GLOBAL OPERATOR ASSEMBLY NORM CHECK ||Vel - Vel^H|| \n");
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n\n");

    if (TDSEZ->hasX && TDSEZ->VelX() != PETSC_NULLPTR) {
        Mat PxH;
        PetscCall(MatHermitianTranspose(TDSEZ->VelX(), MAT_INITIAL_MATRIX, &PxH));
        PetscCall(MatAXPY(PxH, -1.0, TDSEZ->VelX(), DIFFERENT_NONZERO_PATTERN));
        PetscReal norm_x;
        PetscCall(MatNorm(PxH, NORM_FROBENIUS, &norm_x));
        PetscPrintf(PETSC_COMM_WORLD,
            "      ||VelX - VelX^H|| = %-11.2e  Status: %s\n",
            norm_x, norm_x < 1e-8 ? "PASSED" : "FAILED");
        PetscCall(MatDestroy(&PxH));
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "      ⵣ VelX Operator : SKIPPED (not available)\n");
    }

    if (TDSEZ->hasY && TDSEZ->VelY() != PETSC_NULLPTR) 
    {
        Mat PyH;
        PetscCall(MatHermitianTranspose(TDSEZ->VelY(), MAT_INITIAL_MATRIX, &PyH));
        PetscCall(MatAXPY(PyH, -1.0, TDSEZ->VelY(), DIFFERENT_NONZERO_PATTERN));
        PetscReal norm_y;
        PetscCall(MatNorm(PyH, NORM_FROBENIUS, &norm_y));
        PetscPrintf(PETSC_COMM_WORLD,
            "      ||VelY - VelY^H|| = %-11.2e  Status: %s\n",
            norm_y, norm_y < 1e-8 ? "PASSED" : "FAILED");
        PetscCall(MatDestroy(&PyH));
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "      ⵣ VelY Operator : SKIPPED (not available)\n");
    }
    
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");

    PetscFunctionReturn(PETSC_SUCCESS);
}