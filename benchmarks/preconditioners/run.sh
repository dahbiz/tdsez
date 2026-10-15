#!/bin/bash
# Strong scaling of the propagator preconditioners: every deck x NPS x PCS.
# Each run goes to <deck>_<pc>_np<N>/run.log. Adapt mpirun to your scheduler
# (e.g. wrap the loop body in sbatch); keep one run per node allocation so the
# timings are not disturbed by other jobs.
NPS=${NPS:-"16 32 64 128 256"}
PCS=${PCS:-"asm mg gamg"}
DECKS=${DECKS:-"atom3d atom3d_pade varmass3d"}
TDSEZ=${TDSEZ:-tdsez.x}
MPIRUN=${MPIRUN:-mpirun}
# GAMG with the same ASM/ILU(0) smoother as mg (PETSc's default Chebyshev/Jacobi
# smoothing needs hundreds of iterations on this complex operator).
GAMG_OPTS="-mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -mg_levels_pc_type asm -mg_levels_sub_pc_type ilu"
for deck in $DECKS; do
  for pc in $PCS; do
    for np in $NPS; do
      d=${deck}_${pc}_np${np}
      mkdir -p $d
      sed '/^Preconditioner/d' $deck.inp > $d/in.inp
      echo "Preconditioner = $pc" >> $d/in.inp
      opts="-log_view -ksp_converged_reason -pade_stage1_ksp_converged_reason -pade_stage2_ksp_converged_reason"
      if [ $pc = gamg ]; then
        opts="$opts $GAMG_OPTS"
        case $deck in *pade*) for s in pade_stage1_ pade_stage2_; do opts="$opts $(echo $GAMG_OPTS | sed "s/-mg_/-${s}mg_/g")"; done;; esac
      fi
      (cd $d && $MPIRUN -np $np $TDSEZ -inp in.inp $opts > run.log 2>&1)
    done
  done
done
