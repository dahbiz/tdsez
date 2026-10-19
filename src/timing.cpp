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

void TDSEZProfile::Sub(Phase p, const std::string &label, PetscLogDouble seconds)
{
    for (auto &q : parts)
        if (q.phase == p && q.label == label) { q.seconds += seconds; return; }
    parts.push_back({p, label, seconds});
}

namespace {
/// One row of the timing table: tree depth, label, rank-max and rank-min time
/// (tmin < 0: no imbalance column).
struct TimingRow { int depth; std::string label; PetscLogDouble tmax, tmin; };

std::string FormatSeconds(double s)
{
    char b[64];
    if (s >= 3600.0)   snprintf(b, sizeof(b), "%.0f h %02.0f min %04.1f s", std::floor(s / 3600.0), std::floor(std::fmod(s, 3600.0) / 60.0), std::fmod(s, 60.0));
    else if (s >= 60.0) snprintf(b, sizeof(b), "%.0f min %04.1f s", std::floor(s / 60.0), std::fmod(s, 60.0));
    else               snprintf(b, sizeof(b), "%.3f s", s);
    return std::string(b);
}

constexpr int kLabel = 36;

/// Label, wall time, share and imbalance in fixed columns.
void PrintRow(const TDSEZInfo &info, const std::string &label, double t, double tmin, double total)
{
    std::string l = label;
    const int w = TDSEZInfo::displayWidth(l);
    if (w < kLabel) l.append((size_t)(kLabel - w), ' ');
    char imb[16] = "";
    if (tmin > 1e-9) snprintf(imb, sizeof(imb), "%8.2f", t / tmin);
    else if (tmin >= 0) snprintf(imb, sizeof(imb), "%8s", "-");
    info.row("%s%9.3f%8.1f%%%s", l.c_str(), t, total > 0 ? 100.0 * t / total : 0.0, imb);
}

/// Rows with tree glyphs: a row at depth d gets "├ " or "└ " depending on
/// whether a later sibling follows, and "│ " for each open ancestor.
void PrintTree(const TDSEZInfo &info, const std::vector<TimingRow> &rows, double total)
{
    for (size_t i = 0; i < rows.size(); ++i) {
        std::string prefix;
        for (int d = 1; d <= rows[i].depth; ++d) {
            bool more = false;                       // another row at depth d before the branch closes
            for (size_t j = i + 1; j < rows.size() && rows[j].depth >= d; ++j)
                if (rows[j].depth == d) { more = true; break; }
            if (d < rows[i].depth) prefix += more ? "│ " : "  ";
            else prefix += more ? "├ " : "└ ";
        }
        PrintRow(info, prefix + rows[i].label, rows[i].tmax, rows[i].tmin, total);
    }
}
} // namespace

PetscErrorCode TDSEZProfile::Report(PetscInt dofs, PetscBool propagated)
{
    PetscFunctionBeginUser;
    PetscLogDouble tEnd;
    PetscCall(PetscTime(&tEnd));
    PetscLogDouble totalMax, totalMin;
    TDSEZReduceMaxMin(tEnd - tStart, &totalMax, &totalMin);

    PetscMPIInt size = 1;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    static const char *names[NPHASES] = {
        "Input, grid and IGA setup",
        "Stationary assembly (H, M)",
        "Eigensolve (SLEPc)",
        "Static output (HDF5)",
        "Time-dependent operators",
        "Time-dependent setup",
        "Propagation",
        "Final output (HDF5)"};

    // Propagation breakdown from the PETSc log (collective reductions).
    PetscBool logging = PETSC_FALSE;
    PetscCall(PetscLogIsActive(&logging));
    const bool breakdown = propagated && logging && stagePropagation >= 0;
    PetscLogDouble stepT = 0, obsT = 0, tKSP = 0, tPC = 0, tJac = 0, tStage = 0, tFun = 0, tH5 = 0, tPCApply = 0, tMult = 0;
    PetscInt nKSP = 0, nPC = 0, nJac = 0, nStage = 0, nFun = 0, nH5 = 0, nPCApply = 0, nMult = 0, nObs = 0;
    if (breakdown) {
        PetscEventPerfInfo sp{}, so{};
        PetscCall(PetscLogStageGetPerfInfo(stagePropagation, &sp));
        PetscCall(PetscLogStageGetPerfInfo(stageObservables, &so));
        // Stage times are exclusive: pushing the observables stage pauses the
        // propagation stage, so the two add up to the propagation phase.
        PetscLogDouble mn;
        TDSEZReduceMaxMin(sp.time, &stepT, &mn);
        TDSEZReduceMaxMin(so.time, &obsT, &mn);
        nObs = so.count;
        TDSEZEventInStage(stagePropagation, "KSPSolve",       &tKSP,     &nKSP);
        TDSEZEventInStage(stagePropagation, "PCSetUp",        &tPC,      &nPC);
        TDSEZEventInStage(stagePropagation, "PCApply",        &tPCApply, &nPCApply);
        TDSEZEventInStage(stagePropagation, "MatMult",        &tMult,    &nMult);
        TDSEZEventInStage(stagePropagation, "TSJacobianEval", &tJac,     &nJac);
        TDSEZEventInStage(stagePropagation, "TSFunctionEval", &tFun,     &nFun);
        TDSEZEventInStage(stagePropagation, eventStageUpdate, &tStage,   &nStage);
        TDSEZEventInStage(stageObservables, eventHDF5Write,   &tH5,      &nH5);
    }

    std::vector<TimingRow> rows;
    PetscLogDouble accounted = 0.0;
    for (int p = 0; p < NPHASES; ++p) {
        PetscLogDouble mx, mn;
        TDSEZReduceMaxMin(phase[p], &mx, &mn);
        if (!propagated && p >= TDSetup) continue;
        if (p == OperatorAssembly && mx <= 0.0) continue;
        rows.push_back({0, names[p], mx, mn});
        accounted += mx;
        for (const auto &q : parts) {
            if (q.phase != p) continue;
            PetscLogDouble qmx, qmn;
            TDSEZReduceMaxMin(q.seconds, &qmx, &qmn);
            rows.push_back({1, q.label, qmx, qmn});
        }
        if (p == Propagation && breakdown) {
            rows.push_back({1, "Time stepping", stepT, -1});
            rows.push_back({2, "Linear solves (KSPSolve)", tKSP, -1});
            rows.push_back({2, "Preconditioner setup", tPC, -1});
            rows.push_back({2, "Operator updates", tJac + tStage, -1});
            if (nFun > 0) rows.push_back({2, "Residual evaluations", tFun, -1});
            rows.push_back({1, "Observables and monitoring", obsT, -1});
            rows.push_back({2, "HDF5 time-series writes", tH5, -1});
        }
    }

    TDSEZInfo info(PETSC_COMM_WORLD);
    info.header("TIMING SUMMARY");
    info.blank();
    {
        std::string h = "Component";
        h.append((size_t)(kLabel - (int)h.size()), ' ');
        info.row("%s%9s%9s%8s", h.c_str(), "Wall (s)", "Share", "Imbal.");
    }
    info.thin();
    PrintTree(info, rows, totalMax);
    info.thin();
    PrintRow(info, "Other (outside the listed phases)", PetscMax(0.0, totalMax - accounted), -1, totalMax);
    PrintRow(info, "Total wall time", totalMax, totalMin, totalMax);
    info.blank();
    info.line("Wall: slowest rank. Imbal.: slowest / fastest rank. Indented rows are parts of the row above%s",
              breakdown ? "; the propagation parts are PETSc event times and may overlap." : ".");

    info.section("PERFORMANCE");
    info.kv("Total wall time", "%s", FormatSeconds((double)totalMax).c_str());
    info.kv("Degrees of freedom", "%" PetscInt_FMT, dofs);
    info.kv("MPI ranks", "%d (%.3e DOFs per rank)", (int)size, size > 0 ? (double)dofs / (double)size : 0.0);
    if (propagated) {
        PetscLogDouble pmax, pmin;
        TDSEZReduceMaxMin(phase[Propagation], &pmax, &pmin);
        info.kv("Propagator", "%s", TDSEZParser::Propagator.c_str());
        info.kv("Time step / final time", "%g / %g a.u.", (double)TDSEZParser::TimeStep, (double)TDSEZParser::FinalTime);
        info.kv("Time steps", "%" PetscInt_FMT, steps);
        if (steps > 0) {
            info.kv("Wall time per step", "%.3e s", (double)pmax / (double)steps);
            info.kv("Steps per second", "%.2f", pmax > 0.0 ? (double)steps / (double)pmax : 0.0);
            info.kv("Simulated time per wall second", "%.3e a.u.", pmax > 0.0 ? (double)TDSEZParser::FinalTime / (double)pmax : 0.0);
        }
        if (breakdown) {
            info.kv("Linear solves", "%" PetscInt_FMT "; %.1f PC applies, %.1f mat-vecs each",
                    nKSP, nKSP ? (double)nPCApply / (double)nKSP : 0.0, nKSP ? (double)nMult / (double)nKSP : 0.0);
            info.kv("Preconditioner setups", "%" PetscInt_FMT, nPC);
            info.kv("Operator updates", "%" PetscInt_FMT, nJac + nStage);
            if (nFun > 0) info.kv("Residual evaluations", "%" PetscInt_FMT, nFun);
            info.kv("Observable evaluations", "%" PetscInt_FMT, nObs);
            info.kv("HDF5 flushes (rank 0)", "%" PetscInt_FMT, nH5);
        }
    }
    PetscLogDouble mem = 0.0, memMax = 0.0, memMin = 0.0;
    if (PetscMemoryGetMaximumUsage(&mem) == PETSC_SUCCESS && mem > 0.0) {
        TDSEZReduceMaxMin(mem, &memMax, &memMin);
        info.kv("Peak memory per rank", "%.1f MB (max), %.1f MB (min)", (double)memMax / 1048576.0, (double)memMin / 1048576.0);
        info.kv("Peak memory, all ranks", "about %.1f MB", (double)memMax * size / 1048576.0);
    }
    info.blank();
    info.footer();
    PetscFunctionReturn(PETSC_SUCCESS);
}
