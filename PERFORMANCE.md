# Performance optimisations in TDSE-Z

This page documents the optimisations of the propagation and output path, the
speed-ups measured for each, and why none of them changes the computed results
beyond rounding or the linear-solver tolerance.

All timings were measured on a workstation with 2–4 MPI ranks, using the
end-of-run **TIMING SUMMARY** printed by `tdsez.x` (wall time, maximum over
ranks). Speed-ups on a cluster are expected to be similar but have not been
re-measured at scale.

## Notation

The semi-discrete Schrödinger equation is

```math
i\,\mathbf M\,\dot{\mathbf c}(t) = \mathbf H(t)\,\mathbf c(t),
\qquad
\mathbf H(t) = \mathbf H_0 + \mathbf E(t)\cdot\mathbf D ,
```

with overlap matrix $`\mathbf M`$, field-free Hamiltonian $`\mathbf H_0`$
(kinetic $`\mathbf K`$ plus potential $`\mathbf V`$, plus the CAP when enabled) and
dipole matrices $`\mathbf D = (\mathbf D_x,\mathbf D_y,\mathbf D_z)`$.

## Summary

| Optimisation | Test case | Before | After | Gain |
|---|---|---|---|---|
| Preconditioner reuse, right preconditioning, assembly savings | 3D propagation | — | — | propagation $`2.5\times`$, run $`1.7`$–$`2.3\times`$ |
| Magnus-4 instead of Crank–Nicolson (better accuracy) | 1D driven HO | 10.8 s | 0.83 s | $`13\times`$ ($`\approx 70\times`$ at error $`10^{-6}`$) |
| Padé/Magnus stage solver | 3D HO, 200 Magnus steps | 249 s | 219 s | $`-12\%`$ |
| Monitor products + batched HDF5 writes | 1D HO, $`5\times10^4`$ steps | 15.8 s | 12.3 s | $`-22\%`$ |
| | 1D with CAP, $`2.2\times10^4`$ steps | 5.6 s | 4.1 s | $`-27\%`$ |
| | 3D variable mass, 100 steps | 3.42 s | 3.24 s | $`-5\%`$ |

## 1. Crank–Nicolson solver set-up

Crank–Nicolson solves at every step

```math
\Bigl(\mathbf M + \tfrac{i\Delta t}{2}\,\mathbf H^{n+1/2}\Bigr)\mathbf c^{n+1}
= \Bigl(\mathbf M - \tfrac{i\Delta t}{2}\,\mathbf H^{n+1/2}\Bigr)\mathbf c^{n},
\qquad
\mathbf H^{n+1/2} = \mathbf H\!\left(t_n + \tfrac{\Delta t}{2}\right),
```

with GMRES and a restricted additive-Schwarz preconditioner with ILU subdomain
solves.

* **Preconditioner reuse.** The system matrix changes between steps only
  through the scalar field $`\mathbf E(t)`$, so the ILU factorisation of the first
  step remains an effective preconditioner. The SNES layer used to reset the
  KSP reuse flag at every step and refactor; it now builds the preconditioner
  once (`SNESSetLagPreconditioner(snes, -2)` with persistence).
* **Right preconditioning.** GMRES monitors the true residual, so the relative
  tolerance $`10^{-12}`$ bounds the actual Crank–Nicolson solve error,
  independently of the preconditioner quality.
* **Assembly.** Real-valued upper-triangle assembly of the symmetric operators,
  and no finite-difference mass derivatives when the discrete Ehrenfest
  acceleration is used.

Measured on 3D test problems: propagation $`2.5\times`$ faster, complete runs
$`1.7`$–$`2.3\times`$ faster.

## 2. Fourth-order propagators (`Propagator = pade22 | magnus4`)

Both use the $`(2,2)`$ Padé approximant of the step exponential,

```math
R_{22}(z) = \frac{1 + z/2 + z^2/12}{1 - z/2 + z^2/12},
\qquad z = -i\,h\,\mathbf M^{-1}\mathbf H_{\mathrm E},
```

which is fourth order and unitary for Hermitian $`\mathbf H_{\mathrm E}`$. Its
numerator has the roots $`z_\pm = -3 \pm i\sqrt3`$, so $`R_{22}`$ factorises into
two Cayley-type stages

```math
\bigl(\mathbf M + g_k\,\mathbf H_{\mathrm E}\bigr)\,\mathbf y
= \bigl(\mathbf M + \overline{g_k}\,\mathbf H_{\mathrm E}\bigr)\,\mathbf x,
\qquad
g_{1,2} = \frac{h\,(\mp\sqrt3 + 3i)}{12}.
```

`magnus4` is the commutator-free fourth-order Magnus scheme (CF4, Blanes and
Moan 2006),

```math
U = \exp\!\bigl[h(a_1A_1 + a_2A_2)\bigr]\,\exp\!\bigl[h(a_2A_1 + a_1A_2)\bigr],
\qquad
A_j = -i\,\mathbf M^{-1}\mathbf H(t_n + c_j h),
```

with Gauss nodes $`c_{1,2} = \tfrac12 \mp \tfrac{\sqrt3}{6}`$ and
$`a_{1,2} = (3 \mp 2\sqrt3)/12`$. Because $`\mathbf H(t)`$ is linear in the field
and $`a_1 + a_2 = \tfrac12`$, each factor is an ordinary half step $`h/2`$ with the
effective field $`2(a_1\mathbf E_1 + a_2\mathbf E_2)`$ (respectively
$`2(a_2\mathbf E_1 + a_1\mathbf E_2)`$); no commutator is ever formed.

Accuracy versus cost for a strongly driven 1D harmonic oscillator,
$`E(t) = \cos(0.6\,t)`$, $`T = 50`$ ($`p = 7`$, 1200 elements on $`[-60, 60]`$, 2 ranks),
against the exact solution $`\langle x\rangle(t) = -\dfrac{E_0}{1-\omega^2}\bigl(\cos\omega t - \cos t\bigr)`$
with $`E_0 = 1`$, $`\omega = 0.6`$:

| Propagator | $`\Delta t`$ | Solves per step | $`\max_t \lvert\langle x\rangle - x_{\mathrm{ex}}\rvert`$ | Propagation |
|---|---|---|---|---|
| Crank–Nicolson | $`10^{-3}`$ | 1 | $`1.7\times10^{-4}`$ | 10.8 s |
| `pade22` | $`0.02`$ | 2 | $`2.5\times10^{-5}`$ | 1.0 s |
| `magnus4` | $`0.05`$ | 4 | $`4.3\times10^{-5}`$ | 0.83 s |

![Dipole of the driven oscillator for the three propagators](docs/perf/propagator_dipoles.png)

*(a) Dipole $`\langle x\rangle(t)`$ from the `dipoles` dataset of each run's HDF5 output, on top of the exact solution. (b) Absolute error; the legend gives the maximal error and the propagation wall time.*

Observed orders of accuracy: $`2.00`$ (Crank–Nicolson), $`3.97`$–$`4.00`$ (`magnus4`);
`pade22` is fourth order in the field-free phase but second order in the field
variation. For an error of $`10^{-6}`$, extrapolating the second-order behaviour
of Crank–Nicolson gives a cost ratio of about $`70`$ relative to `magnus4`.

The figure is reproduced by `docs/perf/run.sh` (inputs `docs/perf/ho1d_*.inp`,
plot script `docs/perf/plot_propagator_dipoles.py`).

## 3. Padé/Magnus stage solver

**Cached field-free parts.** The stage matrices are

```math
\mathbf A_k = \mathbf M + g_k\,\mathbf H_{\mathrm E}
= \underbrace{\mathbf M + g_k\,\mathbf H_0}_{\mathbf B_k}
+ g_k\,\mathbf E\cdot\mathbf D .
```

$`\mathbf B_k`$ depends only on the step length $`h`$ (and $`\mathbf H_0`$ is fixed
during propagation, including the CAP), so it is computed once per step length.
Each stage update is then one copy and one field term.

**Paired right-hand sides.** Since $`\overline{g_1} = -g_2`$,

```math
\bigl(\mathbf M + \overline{g_1}\,\mathbf H_{\mathrm E}\bigr)\mathbf x
= 2\,\mathbf M\mathbf x - \mathbf A_2\,\mathbf x,
\qquad
\bigl(\mathbf M + \overline{g_2}\,\mathbf H_{\mathrm E}\bigr)\mathbf x
= 2\,\mathbf M\mathbf x - \mathbf A_1\,\mathbf x ,
```

so both stage matrices are built once per $`R_{22}`$ application and each
right-hand side needs two sparse products ($`\mathbf M`$ and $`\mathbf A_{\text{other}}`$)
instead of three ($`\mathbf H_0`$, $`\mathbf D`$, $`\mathbf M`$).

**Extrapolated initial guess.** Each stage solve starts from the stage input
plus the increment the same stage produced on the previous step,

```math
\mathbf y_0^{(n)} = \mathbf x^{(n)} + \bigl(\mathbf y^{(n-1)} - \mathbf x^{(n-1)}\bigr),
```

which removes the leading part of the increment from the initial residual. It
can be disabled with `-pade_extrapolate_guess 0`.

3D harmonic oscillator, $`19\,683`$ DoFs, `magnus4`, $`\Delta t = 0.05`$, 200 steps,
4 ranks:

| | Before | Cached + paired | + extrapolated guess |
|---|---|---|---|
| Propagation | 249 s | 252 s | **219 s** |
| GMRES iterations per solve | 5.98 | 5.98 | **5.07** |
| PC applications per solve | 7.0 | 7.0 | **6.1** |
| Operator updates | 800 (10.2 s) | 400 (6.3 s) | 400 (5.4 s) |

## 4. Monitors and output

The observables ($`\langle H\rangle`$, energies, $`\langle x\rangle`$, acceleration,
populations, currents) are still computed and written at **every** step.

**Verified product identities.** Without a CAP, $`\mathbf H_0 = \mathbf K + \mathbf V`$,
and for a constant mass the inverse-mass matrix is
$`\mathbf M_{m^{-1}} = m^{-1}\mathbf M`$. On the first monitor call both
identities are checked on the current state,

```math
\frac{\lVert \mathbf H_0\boldsymbol\psi - \mathbf K\boldsymbol\psi - \mathbf V\boldsymbol\psi\rVert}
     {\lVert \mathbf H_0\boldsymbol\psi\rVert} \le 10^{-12},
\qquad
\frac{\lVert \mathbf M_{m^{-1}}\boldsymbol\psi - c\,\mathbf M\boldsymbol\psi\rVert}
     {\lVert \mathbf M_{m^{-1}}\boldsymbol\psi\rVert} \le 10^{-12},
\quad
c = \frac{\boldsymbol\psi^\dagger \mathbf M_{m^{-1}}\boldsymbol\psi}{\boldsymbol\psi^\dagger \mathbf M\boldsymbol\psi},
```

and only the identities that hold are used afterwards, replacing up to two
sparse products per step by vector operations. With a CAP or a variable mass
the full products are kept automatically.

**Batched HDF5 writes.** Every step is recorded in the ring buffer. The buffer
is written at multiples of `OutputStrideTS`, but at most once every 2 s of wall
time, and always every `FLUSH_INTERVAL` $`= 5000`$ steps and at the end of the
run. With `OutputStrideTS = 1` this replaces one HDF5 write per step by a few
per run; the file contents and dataset shapes are identical.

| Case | Monitor | HDF5 writes | Propagation |
|---|---|---|---|
| 1D HO, $`5\times10^4`$ steps | 6.04 → 2.31 s | 50 001 → 10 (3.28 → 0.01 s) | 15.8 → 12.3 s |
| 1D with CAP, $`2.2\times10^4`$ steps | 2.23 → 0.76 s | 22 051 → 4 (1.37 → 0.003 s) | 5.58 → 4.05 s |
| 3D variable mass, 100 steps | 1.00 → 0.86 s | 101 → 2 | 3.42 → 3.24 s |

## Effect on accuracy

* **Cached and paired stages** are exact algebraic rewritings; results differ
  only by floating-point rounding.
* **Extrapolated initial guess** changes only the GMRES starting point. The
  stopping test (true residual below $`10^{-12}`$ relative, right preconditioning)
  is unchanged. On the 3D Magnus test the new and old results differ by at most
  $`1\times10^{-10}`$ in $`\langle x\rangle`$ and $`2\times10^{-10}`$ in
  $`\langle H\rangle`$, the accumulated solver tolerance over 800 solves, five
  orders of magnitude below the time-step error of the method at the same
  $`\Delta t`$.
* **Monitor shortcuts** only read $`\boldsymbol\psi`$ and do not affect the
  propagation; they are used only when verified to $`10^{-12}`$. Across the
  regression set the written energies, dipoles and accelerations agree with the
  previous code to $`\le 3\times10^{-14}`$.
* **Batched writes** do not change any written value.

## Reproducing the measurements

Each run prints a TIMING SUMMARY (phase wall times, KSP/PC statistics, operator
updates, HDF5 writes) and a PERFORMANCE block (DoFs, ranks, time per step). To
compare propagators on the same problem, run with the same number of ranks and
change only

```text
Propagator = cn        # or pade22, magnus4
TimeStep   = 0.001     # e.g. 0.05 for magnus4 at comparable accuracy
```
