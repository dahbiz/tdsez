#ifndef TDSEZ_MULTIGRID_HPP
#define TDSEZ_MULTIGRID_HPP

#include <petiga.h>

/**
 * @brief Geometric multigrid on the B-spline space (Preconditioner = mg).
 *
 * Coarse levels drop every other simple interior knot of each axis (C0 and
 * other repeated knots are kept), so the coarse spline spaces are nested in
 * the fine one and the prolongation is exact: knot insertion (Boehm) per axis,
 * applied as a tensor product and mapped through each level's PetIGA ordering.
 * Coarse operators are Galerkin products P^T A P, smoothing is Richardson with
 * restricted ASM/ILU(0),
 * the coarsest level is solved directly; the outer Krylov method keeps the full
 * fine operator, so the solution and its tolerance do not change.
 *
 * Configures @p pc as PCMG for the operator already set on its KSP. PETSc
 * options with the KSP prefix (-mg_levels_*, -mg_coarse_*) override the
 * defaults. @p levels returns the number of levels built.
 */
PetscErrorCode TDSEZSetupMultigrid(IGA iga, PC pc, PetscInt *levels);

#endif // TDSEZ_MULTIGRID_HPP
