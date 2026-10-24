// ============================================================================
//  checkpoint.cpp  —  restart files (CheckpointStride, CheckpointWallTime,
//                     Restart, CheckpointFile, CheckpointGuess)
// ----------------------------------------------------------------------------
//  See include/tdsez_checkpoint.hpp. The wavefunction is stored in the natural
//  ordering of the IGA (the application ordering of iga->ao), so a checkpoint
//  written on N ranks is read on any number of ranks.
// ============================================================================

#include "tdsez_checkpoint.hpp"
#include "tdsez_internal.hpp"
#include "tdsez_paths.hpp"

#include <petscviewerhdf5.h>
#include <cmath>
#include <filesystem>
#include <vector>

namespace {
const PetscInt kVersion = 1;

PetscInt PropagatorCode()
{
    const std::string &p = TDSEZParser::Propagator;
    return p == "cn" ? 0 : p == "pade22" ? 1 : 2;
}
} // namespace

TDSEZCheckpointer &TDSEZCheckpointer::Get()
{
    static TDSEZCheckpointer c;
    return c;
}

bool TDSEZCheckpointer::Restarting() const { return TDSEZParser::Restart; }

PetscErrorCode TDSEZCheckpointer::Setup(IGA iga, Vec templ, const std::string &inputFile, PetscReal dt)
{
    PetscFunctionBeginUser;
    iga_ = iga;
    dt_ = dt;
    stride_ = TDSEZParser::CheckpointStride;
    wallTime_ = TDSEZParser::CheckpointWallTime;
    sampleStride_ = TDSEZParser::TSurff ? TDSEZParser::OutputStrideSurff : 1;   // checkpoints at t-SURFF samples
    path_ = TDSEZParser::CheckpointFile.empty()
                ? TDSEZPaths::TD(std::filesystem::path(inputFile).filename().string(), "checkpoint.h5")
                : TDSEZParser::CheckpointFile;
    if (Writing() || Restarting()) {
        PetscCall(Scatter(templ));
        TDSEZInfo info;
        info.section("RESTART FILES");
        info.kv("Checkpoint file", "%s", path_.c_str());
        if (stride_ > 0) info.kv("Checkpoint stride", "%" PetscInt_FMT " steps", stride_);
        if (wallTime_ > 0) info.kv("Wall-time budget", "%.0f s: checkpoint and stop when it is used up", (double)wallTime_);
        if (sampleStride_ > 1) info.kv("", "checkpoints at multiples of OutputStrideSurff = %" PetscInt_FMT, sampleStride_);
        if (Restarting()) info.kv("Restart", "from %s", path_.c_str());
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// Scatter from the PETSc ordering of templ to a natural-ordering copy.
PetscErrorCode TDSEZCheckpointer::Scatter(Vec templ)
{
    PetscFunctionBeginUser;
    if (toNat_) PetscFunctionReturn(PETSC_SUCCESS);
    PetscInt r0, r1, N;
    PetscCall(VecGetOwnershipRange(templ, &r0, &r1));
    PetscCall(VecGetSize(templ, &N));
    std::vector<PetscInt> nat((size_t)(r1 - r0));
    for (PetscInt i = r0; i < r1; ++i) nat[(size_t)(i - r0)] = i;
    PetscCall(AOPetscToApplication(iga_->ao, r1 - r0, nat.data()));   // dof = 1: node index = row
    IS from, to;
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)templ), r1 - r0, r0, 1, &from));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)templ), r1 - r0, nat.data(), PETSC_COPY_VALUES, &to));
    PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)templ), PETSC_DECIDE, N, &nat_));
    PetscCall(VecScatterCreate(templ, from, nat_, to, &toNat_));
    PetscCall(ISDestroy(&from));
    PetscCall(ISDestroy(&to));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZCheckpointer::WriteVec(PetscViewer v, const char *name, Vec x)
{
    PetscFunctionBeginUser;
    PetscCall(VecScatterBegin(toNat_, x, nat_, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(toNat_, x, nat_, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(PetscObjectSetName((PetscObject)nat_, name));
    PetscCall(VecView(nat_, v));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZCheckpointer::ReadVec(PetscViewer v, const char *name, Vec x, PetscBool *found)
{
    PetscFunctionBeginUser;
    PetscCall(PetscObjectSetName((PetscObject)nat_, name));
    PetscCall(PetscViewerHDF5HasObject(v, (PetscObject)nat_, found));
    if (!*found) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(VecLoad(nat_, v));
    PetscCall(VecScatterBegin(toNat_, nat_, x, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(toNat_, nat_, x, INSERT_VALUES, SCATTER_REVERSE));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZCheckpointer::Write(PetscInt step, PetscReal t, Vec psi)
{
    PetscFunctionBeginUser;
    PetscLogDouble t0, t1;
    PetscCall(PetscTime(&t0));
    MPI_Comm comm = PetscObjectComm((PetscObject)psi);
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    const std::string tmp = path_ + ".tmp";
    if (rank == 0) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec);
    }
    PetscCallMPI(MPI_Barrier(comm));
    PetscViewer v;
    PetscCall(PetscViewerHDF5Open(comm, tmp.c_str(), FILE_MODE_WRITE, &v));
    PetscInt N, prop = PropagatorCode(), version = kVersion;
    PetscCall(VecGetSize(psi, &N));
    PetscCall(PetscViewerHDF5WriteAttribute(v, "/", "version", PETSC_INT, &version));
    PetscCall(PetscViewerHDF5WriteAttribute(v, "/", "step", PETSC_INT, &step));
    PetscCall(PetscViewerHDF5WriteAttribute(v, "/", "time", PETSC_REAL, &t));
    PetscCall(PetscViewerHDF5WriteAttribute(v, "/", "time_step", PETSC_REAL, &dt_));
    PetscCall(PetscViewerHDF5WriteAttribute(v, "/", "dofs", PETSC_INT, &N));
    PetscCall(PetscViewerHDF5WriteAttribute(v, "/", "propagator", PETSC_INT, &prop));
    PetscCall(WriteVec(v, "psi", psi));
    if (TDSEZParser::CheckpointGuess && savePropagator) PetscCall(savePropagator(v));
    if (saveTSurff) PetscCall(saveTSurff(v));
    PetscCall(PetscViewerDestroy(&v));
    PetscCallMPI(MPI_Barrier(comm));
    if (rank == 0) {                                   // atomic replacement of the previous checkpoint
        std::error_code ec;
        std::filesystem::rename(tmp, path_, ec);
        PetscCheck(!ec, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "checkpoint: could not rename %s to %s: %s",
                   tmp.c_str(), path_.c_str(), ec.message().c_str());
    }
    PetscCallMPI(MPI_Barrier(comm));
    PetscCall(PetscTime(&t1));
    ++writes_;
    writeSeconds_ += t1 - t0;
    double mb = 0;
    if (rank == 0) { std::error_code ec; mb = (double)std::filesystem::file_size(path_, ec) / 1048576.0; }
    PetscCall(PetscPrintf(comm, "      checkpoint: step %" PetscInt_FMT ", t = %.6g a.u. -> %s (%.1f MB, %.2f s)\n", step,
                          (double)t, path_.c_str(), mb, (double)(t1 - t0)));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZCheckpointer::AfterStep(TS ts, PetscInt step, PetscReal t, Vec psi)
{
    PetscFunctionBeginUser;
    if (!Writing() || step == 0 || step == restartStep_ || step % sampleStride_ != 0) PetscFunctionReturn(PETSC_SUCCESS);
    const PetscReal T = TDSEZParser::FinalTime;
    if (t >= T - 1e-12 * PetscMax((PetscReal)1, T)) PetscFunctionReturn(PETSC_SUCCESS);   // the run ends here anyway
    bool stop = false;
    if (wallTime_ > 0) {                               // one decision for all ranks
        PetscLogDouble now, elapsed, maxElapsed;
        PetscCall(PetscTime(&now));
        elapsed = now - TDSEZProfile::Get().tStart;
        PetscCallMPI(MPIU_Allreduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, PetscObjectComm((PetscObject)psi)));
        stop = maxElapsed >= wallTime_;
    }
    const bool due = stop || (stride_ > 0 && step % stride_ == 0);
    if (!due) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(Write(step, t, psi));
    if (stop) {
        stopped_ = true;
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)psi),
                              "      wall-time budget (CheckpointWallTime = %.0f s) used: stopping at step %" PetscInt_FMT
                              "; continue with Restart = 1\n", (double)wallTime_, step));
        if (ts) PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_USER));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZCheckpointer::Read(Vec psi, PetscInt *step, PetscReal *t)
{
    PetscFunctionBeginUser;
    MPI_Comm comm = PetscObjectComm((PetscObject)psi);
    PetscLogDouble r0, r1;
    PetscCall(PetscTime(&r0));
    PetscBool exists = std::filesystem::exists(path_) ? PETSC_TRUE : PETSC_FALSE;
    PetscCheck(exists, comm, PETSC_ERR_FILE_OPEN, "Restart = 1: no checkpoint file %s", path_.c_str());
    PetscViewer v;
    PetscCall(PetscViewerHDF5Open(comm, path_.c_str(), FILE_MODE_READ, &v));
    PetscInt N, n = -1, prop = -1, version = 0;
    PetscReal dt = 0;
    PetscCall(VecGetSize(psi, &N));
    PetscCall(PetscViewerHDF5ReadAttribute(v, "/", "version", PETSC_INT, NULL, &version));
    PetscCall(PetscViewerHDF5ReadAttribute(v, "/", "step", PETSC_INT, NULL, step));
    PetscCall(PetscViewerHDF5ReadAttribute(v, "/", "time", PETSC_REAL, NULL, t));
    PetscCall(PetscViewerHDF5ReadAttribute(v, "/", "time_step", PETSC_REAL, NULL, &dt));
    PetscCall(PetscViewerHDF5ReadAttribute(v, "/", "dofs", PETSC_INT, NULL, &n));
    PetscCall(PetscViewerHDF5ReadAttribute(v, "/", "propagator", PETSC_INT, NULL, &prop));
    PetscCheck(version == kVersion, comm, PETSC_ERR_FILE_UNEXPECTED, "checkpoint %s: version %d, expected %d",
               path_.c_str(), (int)version, (int)kVersion);
    PetscCheck(n == N, comm, PETSC_ERR_FILE_UNEXPECTED,
               "checkpoint %s has %" PetscInt_FMT " DOFs, this grid %" PetscInt_FMT ": restart with the same grid",
               path_.c_str(), n, N);
    PetscCheck(PetscAbsReal(dt - dt_) <= 1e-12 * dt_, comm, PETSC_ERR_FILE_UNEXPECTED,
               "checkpoint %s was written with TimeStep = %g, this input has %g", path_.c_str(), (double)dt, (double)dt_);
    PetscCheck(prop == PropagatorCode(), comm, PETSC_ERR_FILE_UNEXPECTED,
               "checkpoint %s was written by another Propagator", path_.c_str());
    PetscBool found;
    PetscCall(ReadVec(v, "psi", psi, &found));
    PetscCheck(found, comm, PETSC_ERR_FILE_UNEXPECTED, "checkpoint %s has no wavefunction", path_.c_str());
    if (loadPropagator) PetscCall(loadPropagator(v));
    if (loadTSurff) PetscCall(loadTSurff(v));
    PetscCall(PetscViewerDestroy(&v));
    restartStep_ = *step;
    PetscCall(PetscTime(&r1));
    TDSEZInfo().kv("Restarted", "from step %" PetscInt_FMT ", t = %.6g a.u. (%s, read in %.2f s)", *step, (double)*t,
                   path_.c_str(), (double)(r1 - r0));
    PetscFunctionReturn(PETSC_SUCCESS);
}
