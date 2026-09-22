#include "tdsez_internal.hpp"

/**
 * @file tdsez.cpp
 * @brief Main entry point for the TDSE-Z solver: parses CLI options,
 *        assembles operators, solves the TISE, optionally propagates the
 *        TDSE, and writes HDF5 output.
 */

/**
 * @brief Entry point: orchestrates CLI parsing, assembly, eigensolve,
 *        dipole-matrix save, and time propagation.
 *
 * Strips custom CLI flags (-enable_gpu, -save_dipole, -mat_type, -vec_type)
 * before SLEPc/PETSc initialize, then re-injects -mat_type / -vec_type as
 * PetIGA -iga_mat_type / -iga_vec_type so device types reach the IGA matrices.
 *
 * @param argc  Number of command-line arguments.
 * @param argv  Command-line argument strings.
 * @return 0 on success, 1 on uncaught exception.
 */
int main(int argc, char *argv[])
{
    // Filter custom CLI flags before SLEPc/PETSc processes argv.
    //
    // NOTE: plain -mat_type / -vec_type are NOT understood by PetIGA's
    // IGASetFromOptions (it only reads -iga_mat_type / -iga_vec_type). We strip
    // them from argv here and re-inject the values as -iga_mat_type /
    // -iga_vec_type into the options database AFTER SlepcInitialize, so the
    // IGA matrices/vectors actually get the requested device type. If we left
    // -mat_type in argv with a value PETSc's MatSetFromOptions would consume it
    // and, in the corrupted-argv case below, could even receive an EMPTY value
    // and abort with "Unknown Mat type given: ".
    //
    // The argument-removal below uses a correct memmove that shifts the *entire*
    // remainder down by 2 (dropping both the flag and its value), then only
    // decrements i by 1 so the loop's i++ re-scans the element that shifted
    // into slot i. The earlier code did `argv[i-1]=argv[i]; memmove(...); i-=2;`
    // which shifted by only 1 and promoted a dangling duplicate of the next
    // flag's value into an option position — corrupting the options database.
    std::string igaMatType, igaVecType;
    auto consume_value_flag = [&](const char *flag) -> const char * {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], flag) == 0 && i + 1 < argc) {
                const char *val = argv[i + 1];
                // remove argv[i] and argv[i+1], shift the rest down by 2
                memmove(&argv[i], &argv[i + 2], (argc - i - 2) * sizeof(char *));
                argc -= 2;
                --i; // loop's i++ will re-scan the shifted element
                return val;
            }
        }
        return (const char *)nullptr;
    };

    {
        const char *v = consume_value_flag("-enable_gpu");
        if (v) TDSEZParser::EnableGPU = std::stoi(v) ? PETSC_TRUE : PETSC_FALSE;
    }
    {
        const char *v = consume_value_flag("-save_dipole");
        if (v) TDSEZParser::SaveDipoleAxes = v; // x/y/z combos (xy, xz, xyz, all, none)
    }
    {
        const char *v = consume_value_flag("-mat_type");
        if (v) igaMatType = v;
    }
    {
        const char *v = consume_value_flag("-vec_type");
        if (v) igaVecType = v;
    }

    SlepcInitialize(&argc, &argv, PETSC_NULLPTR, PETSC_NULLPTR);

    // --- Branding banner (TDSE-Z logo) ---------------------------------
    // Printed once on rank 0 from the checked-in TDSE-Z_logo.txt (the
    // official Unicode ASCII-art banner), followed by the credit line.
    // The logo path is resolved against the CWD (mpirun launches with a bare
    // "./tdsez" program name, so the executable dir is NOT reliable) by
    // searching a list of candidate locations.
    /// Print the TDSE-Z ASCII-art banner and credit line on rank 0.
    {
        PetscMPIInt rank;
        PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
        if (rank == 0) {
            const char *cands[] = {
                "src/TDSE-Z_logo.txt",   // CWD is repo root
                "TDSE-Z_logo.txt",       // CWD is src/
                "../src/TDSE-Z_logo.txt", // CWD is build/
                nullptr
            };
            const char *logo = nullptr;
            for (int i = 0; cands[i]; ++i) {
                FILE *pf = fopen(cands[i], "r");
                if (pf) { fclose(pf); logo = cands[i]; break; }
            }
            if (logo) {
                // Load logo lines into a vector for TDSEZInfo::banner()
                std::vector<std::string> logoLines;
                FILE *lf = fopen(logo, "r");
                if (lf) {
                    char buf[512];
                    while (fgets(buf, sizeof(buf), lf)) {
                        size_t n = strlen(buf);
                        while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r'))
                            buf[--n] = '\0';
                        logoLines.push_back(std::string(buf));
                    }
                    fclose(lf);
                }
                TDSEZInfo infoSelf(PETSC_COMM_SELF);
                infoSelf.banner(logoLines, {
                    "TDSE-Z - Time-Dependent Schrodinger Equation solver (B-spline / IGA)",
                    "Dr. Zakaria Dahbi  |  zdahbi@outlook.es",
                    "Attosecond Quantum Physics Lab, King's College London, UK"
                });
            }
        }
    }

    // Output-formatting helpers live in tdsez_internal.hpp (TDSEZ_fmtSci2 /
    // TDSEZ_fmtAu / TDSEZ_fmtEv) so all translation units share one impl.

    /// Re-inject -mat_type / -vec_type as PetIGA -iga_mat_type / -iga_vec_type.
    /// This routes them into iga->mattype / iga->vectype — the only types
    /// IGACreateMat / IGACreateVec honor. Without this, -mat_type/-vec_type
    /// are ignored by PetIGA or fed an empty value to PETSc's
    /// MatSetFromOptions, crashing with "Unknown Mat type given: ".
    if (!igaMatType.empty())
        PetscCallAbort(PETSC_COMM_WORLD,
            PetscOptionsSetValue(PETSC_NULLPTR, "-iga_mat_type", igaMatType.c_str()));
    if (!igaVecType.empty())
        PetscCallAbort(PETSC_COMM_WORLD,
            PetscOptionsSetValue(PETSC_NULLPTR, "-iga_vec_type", igaVecType.c_str()));
    // Dedicated, clean timing metrics
    PetscLogDouble t_global_start, t_global_end;
    PetscLogDouble t_assemble_start, t_assemble_end;
    PetscLogDouble t_solve_start, t_solve_end;
    PetscLogDouble t_data_write_start = 0.0, t_data_write_end = 0.0;
    PetscLogDouble t_prop_start = 0.0, t_prop_end = 0.0;
    PetscTime(&t_global_start);

    /// Main workflow: read input, assemble, solve, and optionally propagate.
    try {
        // READ INPUT PARAMETERS
        std::string input  = IGAGetOptString(PETSC_NULLPTR, "-inp", "tdse.prm");
        (void)TDSEZParser::Dimension;

        TDSEZCore TDSEZ(input);

        // 1. Matrix Assembly Timing
        PetscTime(&t_assemble_start);
        PetscCallAbort(PETSC_COMM_WORLD, TDSEZ.Assemble());
        PetscTime(&t_assemble_end);
        // 2. Eigensolver Timing
        PetscTime(&t_solve_start);
        PetscCallAbort(PETSC_COMM_WORLD, TDSEZ.Solve());
        PetscTime(&t_solve_end);

        PetscCallAbort(PETSC_COMM_WORLD, TDSEZ.Output());

        // Decide whether the dipole operator assembler must be built. It is
        // needed ONLY when (a) a dipole-matrix save was explicitly requested
        // (CLI -save_dipole / file SaveDipoleMatrix), or (b) time propagation
        // is enabled. A plain static solve with no save request does NOT build
        // Dx/Dy/Dz — matching the original "assemble dipole only if propagation
        // is on" behaviour, while still allowing the save path to opt in.
        std::string axes = TDSEZParser::SaveDipoleAxes; // CLI override
        bool wantX = false, wantY = false, wantZ = false;
        if (!axes.empty()) {
            for (auto& c : axes) c = (char)std::tolower((unsigned char)c);
            if (axes == "none") { /* nothing */ }
            else if (axes == "all" || axes == "xyz") axes = "xyz";
            wantX = axes.find('x') != std::string::npos;
            wantY = axes.find('y') != std::string::npos;
            wantZ = axes.find('z') != std::string::npos;
        } else if (TDSEZParser::SaveDipoleMatrix) {
            wantX = true; // file-level fallback: Dx only (1D-friendly)
        }
        const bool saveRequested = wantX || wantY || wantZ;

        TDSEZAssembler* pDipole = nullptr;
        if (saveRequested || TDSEZParser::EnablePropagation)
            pDipole = new TDSEZAssembler(TDSEZ.iga);

        // Persist the requested dipole operator matrices to PETSc binary files
        // (reloadable with MatLoad). Runs for both static-only (save opt-in)
        // and propagation runs. Only axes actually assembled are written.
        if (pDipole && saveRequested) {
            std::string dipPrefix =
                std::filesystem::path(input).filename().string();
            std::error_code ec;
            std::filesystem::create_directories("static", ec);
            if (wantX && pDipole->Dx)
                PetscCall(TDSEZSaveOperatorMatrix(
                    pDipole->Dx, "static/Dx_" + dipPrefix + ".bin"));
            if (wantY && pDipole->Dy)
                PetscCall(TDSEZSaveOperatorMatrix(
                    pDipole->Dy, "static/Dy_" + dipPrefix + ".bin"));
            if (wantZ && pDipole->Dz)
                PetscCall(TDSEZSaveOperatorMatrix(
                    pDipole->Dz, "static/Dz_" + dipPrefix + ".bin"));
        }

        if (!TDSEZParser::EnablePropagation)
        {
            TDSEZInfo infoWorld;
            infoWorld.rule();
            infoWorld.center("Time propagation disabled");
            infoWorld.center("Set EnablePropagation = 1 to propagate");
            infoWorld.rule();
        }
        else
        {
            // Intermediate Setup Timing (Local Scope Variables)
            PetscLogDouble t_setup_start;
            (void)t_setup_start;
            if (!pDipole) pDipole = new TDSEZAssembler(TDSEZ.iga); // safety
            TDSEZManager manager(&TDSEZ, pDipole);

            // Degenerate-shell splitting by angular momentum (Lz^2 diagonalisation
            // + per-|m|-block H diagonalisation) is opt-in: set EnableLzDiag = 1
            // in the input file.  Only meaningful in 2D (Lz operator exists).
            if (manager.Lz() && TDSEZParser::EnableLzDiag)
            {
                PetscCall(TDSEZOrthogonalizeDegenerates(&manager));
            }

            // Make the states real-valued for the dipole matrix computation
            PetscCallAbort(PETSC_COMM_WORLD,
                           MakeStatesReal(manager.boundstates, manager.M()));

            std::string dipPrefix = std::filesystem::path(manager.inputFile).filename().string();
            PetscCall(TDSEZCompUnifiedDipoleMatrix(manager.boundstates, manager.energies, manager.Dx(), manager.Dy(), manager.Dz(), "DipoleMatrix_" + dipPrefix));
            PetscCall(TDSEZPrecomputeMomentumMatrix(&manager));
            PetscCall(TDSEZCheckLengthVelocity(&manager));

            // t-SURFF: build boundary evaluation operators + momentum grid.
            PetscCall(manager.SetupSurff());

            Vec psi0Local = PETSC_NULLPTR;
            PetscCallAbort(PETSC_COMM_WORLD,
                           VecDuplicate(TDSEZ.initialPsi, &psi0Local));
            PetscCallAbort(PETSC_COMM_WORLD,
                           VecCopy(TDSEZ.initialPsi, psi0Local));
            if (TDSEZParser::EnableGPU) {
              Vec gpupsi = PETSC_NULLPTR;
              PetscCall(VecDuplicate(psi0Local, &gpupsi));
              PetscCall(VecCopy(psi0Local, gpupsi));
              PetscCall(VecDestroy(&psi0Local));
              psi0Local = gpupsi;
            }
            manager.setPsi0(psi0Local);

            PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");
            PetscPrintf(PETSC_COMM_WORLD, "                       PROPAGATING THE QUANTUM SYSTEM                  \n");
            PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");

            // 3. Time Propagation Timing (Dedicated Scope Variables)
            PetscTime(&t_prop_start);
            manager.outputFilename = "td/TimeEvolutionData_" + dipPrefix + ".h5";
            TDSEZPropagator propagator(manager);
            PetscCallAbort(PETSC_COMM_WORLD, propagator.Evolve());
            PetscTime(&t_prop_end);

            // t-SURFF: finalize accumulators and write the PES to its own HDF5.
            PetscCall(manager.FinalizeSurff());

            // write HDF5 output (incremental: only records any remaining unflushed rows),
            // then flush and close the persistent handle so the file is always finalized.
            PetscTime(&t_data_write_start);
            PetscCallAbort(PETSC_COMM_WORLD, manager.CloseHDF5());
            PetscTime(&t_data_write_end);
        }

        // Assembler (if built) is no longer needed once the manager is gone.
        delete pDipole;
    }
    catch (const std::exception& e) {
        PetscPrintf(PETSC_COMM_WORLD, "\n  ERROR: %s\n\n", e.what());
        SlepcFinalize();
        return 1;
    }

    // Capture global termination timestamp
    PetscTime(&t_global_end);

    // Print out the distinct, accurately mapped parameters
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");
    PetscPrintf(PETSC_COMM_WORLD, "         TISE assembly took : %2.4g seconds.\n", t_assemble_end - t_assemble_start);
    PetscPrintf(PETSC_COMM_WORLD, "         Eigensolve took    : %2.4g seconds.\n", t_solve_end - t_solve_start);
    if (TDSEZParser::EnablePropagation) {
        PetscPrintf(PETSC_COMM_WORLD, "         Propagation took : %2.4g seconds.\n", t_prop_end - t_prop_start);
        PetscPrintf(PETSC_COMM_WORLD, "         Data write took  : %2.4g seconds.\n", t_data_write_end - t_data_write_start);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "         Propagation took : SKIPPED (0.0 seconds).\n");
        PetscPrintf(PETSC_COMM_WORLD, "         Data write took  : SKIPPED (0.0 seconds).\n");
    }
    PetscPrintf(PETSC_COMM_WORLD, "         Total Wall time  : %2.4g seconds.\n", t_global_end - t_global_start);
    PetscPrintf(PETSC_COMM_WORLD, "  ══════════════════════════════════════════════════════════════════════\n");

    SlepcFinalize();
    return 0;
}
