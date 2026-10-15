# Preconditioner strong-scaling benchmark (cluster)

Compares the propagator preconditioners `asm` (default), `mg` (geometric
B-spline multigrid) and `gamg` (PETSc algebraic multigrid with the same
ASM/ILU(0) smoother) on 3D problems of about 1.1 x 10^6 DOFs (degree 4,
100 elements per axis). On one 12-core node `asm` was fastest (see
PERFORMANCE.md); the multigrids need fewer iterations, and the question is
whether that pays off at large rank counts, where one-level ASM has no coarse
space and its iteration count grows.

| deck | what |
|---|---|
| `atom3d` | soft-core atom, 800 nm field along x, CAP, Crank-Nicolson, 40 steps |
| `atom3d_pade` | same with Pade(2,2), dt = 0.1, 20 steps (40 solves) |
| `varmass3d` | `atom3d` with a position-dependent mass |

All start from a Gaussian (`NBoundStates = 0`), so only the propagation is
timed. Results are identical for all preconditioners (same 1e-12 tolerance).

Run (build `tdsez.x` from current main):

    NPS="16 32 64 128 256" TDSEZ=/path/to/tdsez.x MPIRUN=mpirun ./run.sh
    python3 analyze.py --plot scaling.pdf

Each run needs about 2-3 GB per rank at 16 ranks (less with more ranks); use
`--bind-to none` or your scheduler's equivalent binding. Copy back only the
`*/run.log` files if you analyse elsewhere.

Read in the table: the solve time per step and its speed-up/efficiency per
preconditioner, the iterations (constant for a scalable preconditioner,
growing for ASM), and the one-off set-up time.
