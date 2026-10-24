#ifndef TDSEZ_CHECKPOINT_HPP
#define TDSEZ_CHECKPOINT_HPP

#include <petiga.h>
#include <functional>
#include <string>

/**
 * @brief Restart files: write the propagation state during a run and resume
 *        from it (input keys CheckpointStride, CheckpointWallTime, Restart,
 *        CheckpointFile, CheckpointGuess).
 *
 * A checkpoint is one HDF5 file (default td/<name>_checkpoint.h5) holding the
 * wavefunction in the natural (rank-independent) ordering of the IGA, the
 * step index and time, the grid size, the time step and the propagator, and
 * the state of the observers that integrate over time (the t-SURFF amplitudes,
 * summed over ranks). With CheckpointGuess = 1 the initial-guess history of
 * the Pade/Magnus stage solves is stored as well. A run restarted from it on
 * any number of ranks continues the uninterrupted run to the tolerance of the
 * stage solves; on the same ranks with CheckpointGuess = 1 it is identical.
 *
 * The file is written to <file>.tmp and renamed, so an interrupted write
 * never replaces the previous checkpoint. Checkpoints are written after the
 * observers of a step (the monitor), at steps that are multiples of
 * CheckpointStride and of the t-SURFF sampling stride. With
 * CheckpointWallTime the run writes a checkpoint and stops once that many
 * seconds have passed since the program started (on the next eligible step).
 */
class TDSEZCheckpointer {
public:
    static TDSEZCheckpointer &Get();

    /// Path, the IGA (natural ordering) and the run's time step; reads the input keys.
    PetscErrorCode Setup(IGA iga, Vec templ, const std::string &inputFile, PetscReal dt);
    bool Writing() const { return stride_ > 0 || wallTime_ > 0; }
    bool Restarting() const;
    const std::string &Path() const { return path_; }

    /// Called after the monitor of every step: writes a checkpoint when it is
    /// due and stops the run (TS reason or the stopped() flag) when the wall
    /// time is used up. ts may be NULL (rational propagators).
    PetscErrorCode AfterStep(TS ts, PetscInt step, PetscReal t, Vec psi);
    /// The run stopped early to be restarted (wall time).
    bool Stopped() const { return stopped_; }

    /// Load the wavefunction, step and time of the checkpoint (Restart = 1).
    PetscErrorCode Read(Vec psi, PetscInt *step, PetscReal *t);

    /// Vectors in the natural ordering (also used by the state hooks below).
    PetscErrorCode WriteVec(PetscViewer v, const char *name, Vec x);
    PetscErrorCode ReadVec(PetscViewer v, const char *name, Vec x, PetscBool *found);

    /// Extra state (propagator guess history, t-SURFF): save into / load from
    /// the open checkpoint viewer. Set by the objects that own the state.
    std::function<PetscErrorCode(PetscViewer)> savePropagator, loadPropagator;
    std::function<PetscErrorCode(PetscViewer)> saveTSurff, loadTSurff;

    PetscInt Writes() const { return writes_; }
    double WriteSeconds() const { return writeSeconds_; }

private:
    PetscErrorCode Write(PetscInt step, PetscReal t, Vec psi);
    PetscErrorCode Scatter(Vec templ);

    IGA iga_ = nullptr;
    std::string path_;
    PetscReal dt_ = 0;
    PetscInt stride_ = 0, sampleStride_ = 1;
    PetscReal wallTime_ = 0;
    bool stopped_ = false;
    PetscInt restartStep_ = -1;        // the step read on restart (not written again)
    Vec nat_ = nullptr;                 // natural-ordering copy
    VecScatter toNat_ = nullptr;        // PETSc -> natural ordering
    PetscInt writes_ = 0;
    double writeSeconds_ = 0;
};

#endif // TDSEZ_CHECKPOINT_HPP
