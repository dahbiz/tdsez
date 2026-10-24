#include "tdsez_paths.hpp"
#include "tdsez_logo.hpp"
#include "tdsez_internal.hpp"
#include "tdsez_tsurff.hpp"
#include "tdsez_checkpoint.hpp"
#include <memory>

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
/// Thrown after -tdsez_estimate_memory has printed its report.
struct TDSEZEstimateDone {};

static int RunTDSEZ(int argc, char *argv[])
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

    // One BLAS thread per MPI rank unless the user chose otherwise: a threaded OpenBLAS or MKL
    // otherwise starts one thread per core in every rank (40 ranks x 40 threads on a 40-core
    // node), and with ranks bound to cores those threads share one core.
#if PETSC_VERSION_GE(3, 20, 0)
    {
        PetscBool set = PETSC_FALSE;
        PetscCallAbort(PETSC_COMM_WORLD, PetscOptionsHasName(NULL, NULL, "-blas_num_threads", &set));
        if (!set && !getenv("OMP_NUM_THREADS") && !getenv("OPENBLAS_NUM_THREADS") && !getenv("MKL_NUM_THREADS") &&
            !getenv("BLIS_NUM_THREADS"))
            PetscCallAbort(PETSC_COMM_WORLD, PetscBLASSetNumThreads(1));
    }
#endif

    // --- Branding banner (TDSE-Z logo) ---------------------------------
    // Printed once on rank 0 at launch. The logo (src/TDSE-Z_logo.txt) is
    // embedded at build time (generated tdsez_logo.hpp), so it appears
    // whatever the working directory of the run.
    /// Print the TDSE-Z ASCII-art banner and credit line on rank 0.
    {
        PetscMPIInt rank;
        PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
        if (rank == 0) {
            std::vector<std::string> logoLines;
            std::string line;
            for (const char *c = TDSEZ_LOGO; *c; ++c) {
                if (*c == '\n') { logoLines.push_back(line); line.clear(); }
                else if (*c != '\r') line.push_back(*c);
            }
            if (!line.empty()) logoLines.push_back(line);
            TDSEZInfo infoSelf(PETSC_COMM_SELF);
            infoSelf.banner(logoLines, {
                {"Z is not the author's initial. Z is not the third axis.",
                 "Z is ⵣ (yaz), the Amazigh sign of the free man and the free people."},
                {"TDSE-Z  |  Time-Dependent Schrödinger Equation solver (B-spline / IGA)",
                 "Dr. Zakaria Dahbi  |  drdahbi@outlook.com",
                 "Attosecond Quantum Physics Lab, King's College London, UK"}
            });
        }
    }


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
    // Run profile: phase wall times plus PETSc log stages for the propagation
    // breakdown; reported in the timing summary at the end of the run.
    TDSEZProfile &profile = TDSEZProfile::Get();
    PetscCallAbort(PETSC_COMM_WORLD, profile.Init());
    PetscLogDouble t_phase = 0.0, t_now = 0.0;
    PetscTime(&t_phase);
    // Close the current phase: charge the time since the last mark to p.
    auto markPhase = [&](TDSEZProfile::Phase p) {
        PetscTime(&t_now);
        profile.Add(p, t_now - t_phase);
        t_phase = t_now;
    };
    PetscInt  reportDofs = 0;
    PetscBool propagated = PETSC_FALSE;

    /// Main workflow: read input, assemble, solve, and optionally propagate.
    try {
        // READ INPUT PARAMETERS
        std::string input  = IGAGetOptString(PETSC_NULLPTR, "-inp", "tdse.prm");
        (void)TDSEZParser::Dimension;

        TDSEZCore TDSEZ(input);
        markPhase(TDSEZProfile::Setup);

        // -tdsez_estimate_memory N: predict the per-rank memory on N ranks from
        // the knots and the PetIGA partition, then stop (nothing is assembled).
        {
            PetscInt  estRanks = 0;
            PetscBool est = PETSC_FALSE;
            PetscCallAbort(PETSC_COMM_WORLD, PetscOptionsGetInt(NULL, NULL, "-tdsez_estimate_memory", &estRanks, &est));
            if (est) {
                PetscMPIInt size;
                PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
                PetscCallAbort(PETSC_COMM_WORLD, TDSEZEstimateMemory(TDSEZ.iga, estRanks > 0 ? estRanks : (PetscInt)size));
                throw TDSEZEstimateDone{};   // unwinds TDSEZ (IGA) before SlepcFinalize
            }
        }

        PetscCallAbort(PETSC_COMM_WORLD, TDSEZ.Assemble());
        markPhase(TDSEZProfile::Assembly);
        PetscCallAbort(PETSC_COMM_WORLD, TDSEZ.Solve());
        markPhase(TDSEZProfile::Eigensolve);
        if (TDSEZ.initialPsi) PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(TDSEZ.initialPsi, &reportDofs));

        if (TDSEZParser::NBoundStates > 0) PetscCallAbort(PETSC_COMM_WORLD, TDSEZ.Output());
        markPhase(TDSEZProfile::StaticOutput);

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
                    pDipole->Dx, TDSEZPaths::Static(dipPrefix, "Dx.bin")));
            if (wantY && pDipole->Dy)
                PetscCall(TDSEZSaveOperatorMatrix(
                    pDipole->Dy, TDSEZPaths::Static(dipPrefix, "Dy.bin")));
            if (wantZ && pDipole->Dz)
                PetscCall(TDSEZSaveOperatorMatrix(
                    pDipole->Dz, TDSEZPaths::Static(dipPrefix, "Dz.bin")));
        }
        markPhase(TDSEZProfile::OperatorAssembly);

        if (!TDSEZParser::EnablePropagation)
        {
            TDSEZInfo infoWorld;
            infoWorld.header("PROPAGATION");
            infoWorld.blank();
            infoWorld.center("disabled (EnablePropagation = 0)");
        }
        else
        {
            if (!pDipole) pDipole = new TDSEZAssembler(TDSEZ.iga); // safety
            TDSEZManager manager(&TDSEZ, pDipole);

            // Degenerate-shell splitting by angular momentum (Lz^2 diagonalisation
            // + per-|m|-block H diagonalisation) is opt-in: set EnableLzDiag = 1
            // in the input file.  Only meaningful in 2D (Lz operator exists).
            if (manager.Lz() && TDSEZParser::EnableLzDiag)
            {
                PetscCall(TDSEZOrthogonalizeDegenerates(&manager));
            }

            // The bound states are real with a fixed phase (TDSEZCanonicalEigenpairs).
            std::string dipPrefix = std::filesystem::path(manager.inputFile).filename().string();
            {
                PetscLogDouble a, b;
                PetscTime(&a);
                if (!manager.boundstates.empty()) PetscCall(manager.SetupMatrixFree(manager.boundstates[0]));
                else                              PetscCall(manager.SetupMatrixFree(TDSEZ.initialPsi));
                PetscCall(TDSEZCompUnifiedDipoleMatrix(manager.boundstates, manager.energies, &manager, "DipoleMatrix_" + dipPrefix));
                PetscTime(&b); profile.Sub(TDSEZProfile::TDSetup, "Bound-state dipole matrix", b - a); a = b;
                PetscCall(TDSEZPrecomputeMomentumMatrix(&manager));
                PetscTime(&b); profile.Sub(TDSEZProfile::TDSetup, "Bound-state momentum matrix", b - a); a = b;
                PetscCall(TDSEZCheckLengthVelocity(&manager));
                PetscTime(&b); profile.Sub(TDSEZProfile::TDSetup, "Length-velocity gauge check", b - a);
            }

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

            TDSEZInfo propagationInfo(PETSC_COMM_WORLD);
            propagationInfo.header("PROPAGATION");
            propagationInfo.section("SOLVER");
            markPhase(TDSEZProfile::TDSetup);

            // Propagation runs in its own log stage; the monitors push the nested
            // "TDSEZ Observables" stage, so solver events in this stage are
            // purely time-stepping work.
            PetscCall(PetscLogStagePush(profile.stagePropagation));
            manager.outputFilename = TDSEZPaths::TD(dipPrefix, "observables.h5");
            auto propagator = std::make_unique<TDSEZPropagator>(manager);
            // t-SURFF (TSurff = 1): passive surface-flux observer, separate output file.
            std::unique_ptr<TDSEZTSurff> tsurff;
            if (TDSEZParser::TSurff) {
                tsurff = std::make_unique<TDSEZTSurff>();
                PetscCall(tsurff->Setup(TDSEZ.iga, TDSEZ.initialPsi));
                manager.tsurff = tsurff.get();
            }
            PetscCallAbort(PETSC_COMM_WORLD, propagator->Evolve());
            // the stage matrices and their preconditioners are not needed any
            // more: free them before the t-SURFF end term factorises E M - H
            propagator.reset();
            if (TDSEZCheckpointer::Get().Stopped()) {      // CheckpointWallTime: the spectrum needs the end of the run
                TDSEZInfo().kv("Stopped for a restart", "continue with Restart = 1 (%s)", TDSEZCheckpointer::Get().Path().c_str());
                manager.tsurff = PETSC_NULLPTR;
            } else if (tsurff) {
                // Both propagators evolve TDSEZ.initialPsi in place up to FinalTime.
                PetscLogDouble a, b;
                PetscTime(&a);
                PetscCall(tsurff->Finish(TDSEZPaths::TD(dipPrefix, "surff.h5"), TDSEZParser::FinalTime, TDSEZ.initialPsi,
                                         manager.H(), manager.M()));
                PetscTime(&b);
                if (TDSEZParser::SurffEndTerm) profile.Sub(TDSEZProfile::Propagation, "t-SURFF end term", b - a);
                manager.tsurff = PETSC_NULLPTR;
            }
            PetscCall(PetscLogStagePop());
            markPhase(TDSEZProfile::Propagation);
            propagated = PETSC_TRUE;

            // write HDF5 output (incremental: only records any remaining unflushed rows),
            // then flush and close the persistent handle so the file is always finalized.
            PetscCallAbort(PETSC_COMM_WORLD, manager.CloseHDF5());
            markPhase(TDSEZProfile::FinalOutput);
        }

        // Assembler (if built) is no longer needed once the manager is gone.
        delete pDipole;
    }
    catch (const TDSEZEstimateDone &) {
        SlepcFinalize();
        return 0;
    }
    catch (const std::exception& e) {
        // often thrown on one rank only (std::bad_alloc): SlepcFinalize would wait for the others
        // forever, so the message goes out from this rank and the whole job stops
        PetscMPIInt rank = 0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        if (rank == 0) TDSEZInfo(PETSC_COMM_SELF).fatal("%s", e.what());
        else fprintf(stderr, "[%d] TDSEZ FATAL: %s\n", (int)rank, e.what());
        fflush(stdout); fflush(stderr);
        MPI_Abort(PETSC_COMM_WORLD, 1);
        return 1;
    }

    // End-of-run timing summary (collective).
    PetscCallAbort(PETSC_COMM_WORLD, profile.Report(reportDofs, propagated));

    SlepcFinalize();
    return 0;
}

/// An error returned on one rank (a PETSc error raised on PETSC_COMM_SELF, a rank out of memory)
/// would leave the others waiting in their next collective until the walltime ends (srun keeps
/// the remaining tasks unless --kill-on-bad-exit): stop them all.
int main(int argc, char *argv[])
{
    const int rc = RunTDSEZ(argc, argv);
    int init = 0, fin = 0;
    MPI_Initialized(&init);
    MPI_Finalized(&fin);
    if (rc && init && !fin) {
        fflush(stdout); fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    return rc;
}
