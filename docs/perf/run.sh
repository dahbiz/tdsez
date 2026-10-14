#!/bin/bash
# Runs the three propagators on the same problem (same number of ranks).
NP=${NP:-2}; TDSEZ=${TDSEZ:-tdsez.x}
for p in cn pade22 magnus4; do
  mkdir -p run_$p && cp ho1d_$p.inp run_$p/ && (cd run_$p && mpirun -np $NP $TDSEZ -inp ho1d_$p.inp > run.log 2>&1)
done
python3 plot_propagator_dipoles.py
