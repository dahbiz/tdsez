#include "tdsez_internal.hpp"

/**
 * @file timing.cpp
 * @brief Run-time profile and the end-of-run timing report.
 *
 * Phase wall times are measured with PetscTime and reduced over MPI ranks
 * (max, min). The propagation breakdown is read from the PETSc event log:
 * the propagation work runs in the log stage "TDSEZ Propagation" and every
 * monitor call in the nested stage "TDSEZ Observables". PETSc charges time to
 * the innermost active stage, so events read from the propagation stage
 * (KSPSolve, PCSetUp, ...) contain only the time-stepping work.
 */

TDSEZProfile& TDSEZProfile::Get()
{
    static TDSEZProfile profile;
    return profile;
}

PetscErrorCode TDSEZProfile::Init()
{
    PetscFunctionBeginUser;
    PetscBool active = PETSC_FALSE;
    PetscCall(PetscLogIsActive(&active));
    if (!active) PetscCall(PetscLogDefaultBegin());
    PetscCall(PetscLogStageRegister("TDSEZ Propagation", &stagePropagation));
    PetscCall(PetscLogStageRegister("TDSEZ Observables", &stageObservables));
    PetscClassId classid = 0;
    PetscCall(PetscClassIdRegister("TDSE-Z", &classid));
    PetscCall(PetscLogEventRegister("TDSEZHDF5Write",   classid, &eventHDF5Write));
    PetscCall(PetscLogEventRegister("TDSEZStageUpdate", classid, &eventStageUpdate));
    PetscCall(PetscMemorySetGetMaximumUsage());
    PetscCall(PetscTime(&tStart));
    initialised = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
}

void TDSEZProfile::Add(Phase p, PetscLogDouble seconds)
{
    phase[p] += seconds;
}

/// Max and min of a local value over PETSC_COMM_WORLD.
static void TDSEZReduceMaxMin(PetscLogDouble v, PetscLogDouble *vmax, PetscLogDouble *vmin)
{
    double in = (double)v, mx = 0.0, mn = 0.0;
    MPI_Allreduce(&in, &mx, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    MPI_Allreduce(&in, &mn, 1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);
    *vmax = mx; *vmin = mn;
}

/// Time (rank max) and count of a PETSc event inside a log stage.
static void TDSEZEventInStage(PetscLogStage stage, const char *name, PetscLogDouble *time, PetscInt *count)
{
    *time = 0.0; *count = 0;
    PetscLogEvent ev = -1;
    if (PetscLogEventGetId(name, &ev) != PETSC_SUCCESS || ev < 0) return;
    PetscEventPerfInfo info{};
    if (PetscLogEventGetPerfInfo(stage, ev, &info) != PETSC_SUCCESS) return;
    PetscLogDouble mn;
    TDSEZReduceMaxMin(info.time, time, &mn);
    *count = info.count;
}

static void TDSEZEventInStage(PetscLogStage stage, PetscLogEvent ev, PetscLogDouble *time, PetscInt *count)
{
    *time = 0.0; *count = 0;
    if (ev < 0) return;
    PetscEventPerfInfo info{};
    if (PetscLogEventGetPerfInfo(stage, ev, &info) != PETSC_SUCCESS) return;
    PetscLogDouble mn;
    TDSEZReduceMaxMin(info.time, time, &mn);
    *count = info.count;
}

/// Left-justify a UTF-8 label to a display width (box-drawing glyphs are
/// multi-byte, so printf's byte-based %-*s would misalign the columns).
static std::string TDSEZPadLabel(const char *label, int width)
{
    std::string out(label);
    int cols = 0;
    for (const unsigned char *c = (const unsigned char *)label; *c; ++c)
        if ((*c & 0xC0) != 0x80) ++cols;
    if (cols < width) out.append((size_t)(width - cols), ' ');
    return out;
}

/// Print one table row: label, wall time, share of the total, rank imbalance.
static void TDSEZTimingRow(const char *label, PetscLogDouble tmax, PetscLogDouble tmin,
                           PetscLogDouble total, const char *note = "")
{
    const double share = total > 0.0 ? 100.0 * tmax / total : 0.0;
    char imb[16];
    if (tmin > 1e-9) snprintf(imb, sizeof(imb), "%6.2f", tmax / tmin);
    else             snprintf(imb, sizeof(imb), "%6s", "-");
    PetscPrintf(PETSC_COMM_WORLD, "      %s %11.3f  %6.1f%%  %s  %s\n",
                TDSEZPadLabel(label, 34).c_str(), (double)tmax, share, imb, note);
}

static void TDSEZTimingSubRow(const char *label, PetscLogDouble t, PetscLogDouble total, const char *note)
{
    const double share = total > 0.0 ? 100.0 * t / total : 0.0;
    PetscPrintf(PETSC_COMM_WORLD, "      %s %11.3f  %6.1f%%  %6s  %s\n",
                TDSEZPadLabel(label, 34).c_str(), (double)t, share, "", note);
}

static std::string TDSEZFormatSeconds(double s)
{
    char b[64];
    if (s >= 3600.0)   snprintf(b, sizeof(b), "%.0f h %02.0f min %04.1f s", std::floor(s / 3600.0), std::floor(std::fmod(s, 3600.0) / 60.0), std::fmod(s, 60.0));
    else if (s >= 60.0) snprintf(b, sizeof(b), "%.0f min %04.1f s", std::floor(s / 60.0), std::fmod(s, 60.0));
    else               snprintf(b, sizeof(b), "%.3f s", s);
    return std::string(b);
}

PetscErrorCode TDSEZProfile::Report(PetscInt dofs, PetscBool propagated)
{
    PetscFunctionBeginUser;
    PetscLogDouble tEnd;
    PetscCall(PetscTime(&tEnd));
    PetscLogDouble totalMax, totalMin;
    TDSEZReduceMaxMin(tEnd - tStart, &totalMax, &totalMin);

    PetscMPIInt size = 1;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    TDSEZInfo info(PETSC_COMM_WORLD);
    info.rule();
    info.center("TIMING SUMMARY");
    info.rule();
    PetscPrintf(PETSC_COMM_WORLD, "      %-34s %11s  %7s  %6s\n", "Phase", "Wall (s)", "Share", "Max/min");
    PetscPrintf(PETSC_COMM_WORLD, "      %-34s %11s  %7s  %6s\n", "", "(max rank)", "", "rank");
    info.raw("      ──────────────────────────────────────────────────────────────\n");

    static const char *names[NPHASES] = {
        "Input and IGA setup",
        "TISE assembly (H, M)",
        "Eigensolve (SLEPc)",
        "Static output",
        "Dipole/TD operator assembly",
        "Time-dependent setup",
        "Propagation",
        "Final output (HDF5, t-SURFF)"};
    PetscLogDouble accounted = 0.0;
    for (int p = 0; p < NPHASES; ++p) {
        PetscLogDouble mx, mn;
        TDSEZReduceMaxMin(phase[p], &mx, &mn);
        if (!propagated && p >= TDSetup) {
            PetscPrintf(PETSC_COMM_WORLD, "      %-34s %11s\n", names[p], "skipped");
            continue;
        }
        if (p == OperatorAssembly && mx <= 0.0) continue;
        TDSEZTimingRow(names[p], mx, mn, totalMax);
        accounted += mx;

        if (p == Propagation && propagated) {
            PetscBool logging = PETSC_FALSE;
            PetscCall(PetscLogIsActive(&logging));
            if (!logging || stagePropagation < 0) {
                PetscPrintf(PETSC_COMM_WORLD, "        (breakdown unavailable: PETSc built without logging)\n");
                continue;
            }
            PetscEventPerfInfo sp{}, so{};
            PetscCall(PetscLogStageGetPerfInfo(stagePropagation, &sp));
            PetscCall(PetscLogStageGetPerfInfo(stageObservables, &so));
            // Stage times are exclusive: pushing the observables stage pauses the
            // propagation stage, so the two add up to the propagation phase.
            PetscLogDouble stepT, stepMin, obsT, obsMin;
            TDSEZReduceMaxMin(sp.time, &stepT, &stepMin);
            TDSEZReduceMaxMin(so.time, &obsT, &obsMin);

            PetscLogDouble tKSP, tPC, tPCApply, tMult, tJac, tStage, tFun, tH5;
            PetscInt       nKSP, nPC, nPCApply, nMult, nJac, nStage, nFun, nH5;
            TDSEZEventInStage(stagePropagation, "KSPSolve",       &tKSP,     &nKSP);
            TDSEZEventInStage(stagePropagation, "PCSetUp",        &tPC,      &nPC);
            TDSEZEventInStage(stagePropagation, "PCApply",        &tPCApply, &nPCApply);
            TDSEZEventInStage(stagePropagation, "MatMult",        &tMult,    &nMult);
            TDSEZEventInStage(stagePropagation, "TSJacobianEval", &tJac,     &nJac);
            TDSEZEventInStage(stagePropagation, "TSFunctionEval", &tFun,     &nFun);
            TDSEZEventInStage(stagePropagation, eventStageUpdate, &tStage,   &nStage);
            TDSEZEventInStage(stageObservables, eventHDF5Write,   &tH5,      &nH5);

            char note[96];
            snprintf(note, sizeof(note), "%" PetscInt_FMT " steps", steps);
            TDSEZTimingSubRow("  ├ Time stepping", stepT, totalMax, note);
            snprintf(note, sizeof(note), "%" PetscInt_FMT " solves, %.1f PC apps/solve, %" PetscInt_FMT " mat-vecs",
                     nKSP, nKSP ? (double)nPCApply / (double)nKSP : 0.0, nMult);
            TDSEZTimingSubRow("  │   ├ Linear solves (KSPSolve)", tKSP, totalMax, note);
            snprintf(note, sizeof(note), "%" PetscInt_FMT " calls (factorizations reused)", nPC);
            TDSEZTimingSubRow("  │   ├ Preconditioner setup", tPC, totalMax, note);
            snprintf(note, sizeof(note), "%" PetscInt_FMT " updates", nJac + nStage);
            if (nFun > 0) {
                TDSEZTimingSubRow("  │   ├ Operator updates", tJac + tStage, totalMax, note);
                snprintf(note, sizeof(note), "%" PetscInt_FMT " evaluations", nFun);
                TDSEZTimingSubRow("  │   └ Residual evaluations", tFun, totalMax, note);
            } else {
                TDSEZTimingSubRow("  │   └ Operator updates", tJac + tStage, totalMax, note);
            }
            snprintf(note, sizeof(note), "%d calls", so.count);
            TDSEZTimingSubRow("  └ Observables and monitoring", obsT, totalMax, note);
            snprintf(note, sizeof(note), "%" PetscInt_FMT " flushes (rank 0)", nH5);
            TDSEZTimingSubRow("      └ HDF5 time-series writes", tH5, totalMax, note);
        }
    }
    info.raw("      ──────────────────────────────────────────────────────────────\n");
    if (propagated)
        PetscPrintf(PETSC_COMM_WORLD, "      Sub-rows are inclusive PETSc event times and may overlap.\n");
    TDSEZTimingRow("Other (outside listed phases)", PetscMax(0.0, totalMax - accounted), 0.0, totalMax);
    TDSEZTimingRow("Total wall time", totalMax, totalMin, totalMax,
                   ("(" + TDSEZFormatSeconds((double)totalMax) + ")").c_str());

    info.blank();
    info.section("PERFORMANCE");
    info.kv("Degrees of freedom", "%" PetscInt_FMT, dofs);
    info.kv("MPI ranks", "%d", (int)size);
    info.kv("DOFs per rank", "%.3e", size > 0 ? (double)dofs / (double)size : 0.0);
    if (propagated) {
        PetscLogDouble pmax, pmin;
        TDSEZReduceMaxMin(phase[Propagation], &pmax, &pmin);
        info.kv("Propagator", "%s", TDSEZParser::Propagator.c_str());
        info.kv("Time step / final time", "%g / %g a.u.", (double)TDSEZParser::TimeStep, (double)TDSEZParser::FinalTime);
        info.kv("Time steps", "%" PetscInt_FMT, steps);
        if (steps > 0) {
            info.kv("Wall time per step", "%.3e s", (double)pmax / (double)steps);
            info.kv("Steps per second", "%.2f", pmax > 0.0 ? (double)steps / (double)pmax : 0.0);
            info.kv("Simulated time per wall s", "%.3e a.u.", pmax > 0.0 ? (double)TDSEZParser::FinalTime / (double)pmax : 0.0);
        }
    }
    PetscLogDouble mem = 0.0, memMax = 0.0, memMin = 0.0;
    if (PetscMemoryGetMaximumUsage(&mem) == PETSC_SUCCESS && mem > 0.0) {
        TDSEZReduceMaxMin(mem, &memMax, &memMin);
        info.kv("Peak memory per rank", "%.1f MB (max), %.1f MB (min)", (double)memMax / 1048576.0, (double)memMin / 1048576.0);
        info.kv("Peak memory, all ranks", "~%.1f MB", (double)memMax * size / 1048576.0);
    }
    info.rule();
    PetscFunctionReturn(PETSC_SUCCESS);
}
