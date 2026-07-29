// ============================================================================
//  assembler_bench_main.cpp  —  standalone entry point for the 2D assembler
//  advantage benchmark defined in assembler.cpp.
//
//  Built as the separate executable `tdsez_asmbench` (see CMakeLists.txt).
//  It does NOT link into the main `tdsez` binary's control flow; it only
//  exercises the NEW assembler.cpp kernels so the existing code is untouched.
// ============================================================================

#include "tdsez_internal.hpp"

// Forward declarations of the drivers defined in assembler.cpp.
PetscErrorCode TDSEZRunAssemblerBenchmark(PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunFusionSweep(PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunOperatorPhysicsChecks(PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunOperatorPhysicsChecks3D(PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunFusionSweep3D(PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunHarmonicOscillator(PetscInt dim, PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunAccuracyTest(PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunMatrixComparison(PetscInt dim, PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunBCProbe(PetscInt nel, PetscInt pdegree);
PetscErrorCode TDSEZRunBCDirichletBenchmark(PetscInt dim, PetscInt nel, PetscInt pdegree);

int main(int argc, char **argv)
{
    PetscCall(SlepcInitialize(&argc, &argv, (char *)0, "TDSEZ 2D assembler benchmark"));

    // Sweep a few 2D grid resolutions to show the speedup scales.
    const PetscInt grids[][2] = {
        {20, 3},   // nel=20, p=3
        {40, 3},   // nel=40, p=3
        {80, 5},   // nel=80, p=5
        {200, 5},  // nel=200, p=5  (production-class 2D grid)
    };
    const PetscInt nGrids = (PetscInt)(sizeof(grids) / sizeof(grids[0]));

    for (PetscInt g = 0; g < nGrids; ++g)
        PetscCall(TDSEZRunAssemblerBenchmark(grids[g][0], grids[g][1]));

    // Physics correctness: independent checks (Hermiticity + analytic 2D box
    // spectrum) on the smallest grid — proves the operators are physically right,
    // not just self-consistent.
    PetscCall(TDSEZRunOperatorPhysicsChecks(grids[0][0], grids[0][1]));

    for (PetscInt g = 0; g < nGrids; ++g)
        PetscCall(TDSEZRunFusionSweep(grids[g][0], grids[g][1]));

    // Physics-accuracy check: 2D particle-in-a-box vs analytic spectrum.
    PetscCall(TDSEZRunAccuracyTest(20, 3));
    PetscCall(TDSEZRunAccuracyTest(40, 5));

    // Direct matrix comparison: fused vs unfused vs production-equivalent.
    PetscCall(TDSEZRunMatrixComparison(2, 20, 3));
    PetscCall(TDSEZRunMatrixComparison(2, 40, 5));
    PetscCall(TDSEZRunMatrixComparison(3, 12, 2));
    PetscCall(TDSEZRunMatrixComparison(3, 10, 2));

    // BC probe: does production enforce Dirichlet or Neumann? (uses TDSEZFormHam)
    PetscCall(TDSEZRunBCProbe(20, 3));
    // Dirichlet fix benchmark: TDSEZCompOperatorsDirichlet + accuracy vs analytic
    PetscCall(TDSEZRunBCDirichletBenchmark(2, 40, 3));
    PetscCall(TDSEZRunBCDirichletBenchmark(3, 8, 2));

    // ---- 3D polarisation cases: same fused approach + physics verification ----
    const PetscInt grids3D[][2] = {{10, 2}, {12, 2}, {25, 3}, {40, 2}};   // 3D dof grows fast
    const PetscInt nGrids3D = (PetscInt)(sizeof(grids3D) / sizeof(grids3D[0]));
    for (PetscInt g = 0; g < nGrids3D; ++g)
        PetscCall(TDSEZRunOperatorPhysicsChecks3D(grids3D[g][0], grids3D[g][1]));
    for (PetscInt g = 0; g < nGrids3D; ++g)
        PetscCall(TDSEZRunFusionSweep3D(grids3D[g][0], grids3D[g][1]));

    // ---- Harmonic-oscillator cross-check (2D + 3D): analytic spectrum ----
    PetscCall(TDSEZRunHarmonicOscillator(2, 40, 3));
    PetscCall(TDSEZRunHarmonicOscillator(3, 8, 2));

    PetscCall(PetscFinalize());
    return 0;
}

// Keep PetscCall usable in main (PETSc macro expects PetscFunctionBegin in
// user functions; in main we rely on PETSc's main-wrapper tolerance).
