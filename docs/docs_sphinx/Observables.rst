Observables
===========

During propagation a monitor evaluates the observables at :math:`t = 0` and
after every time step and writes them, one row per step, to
``td/<name>_observables.h5`` (:doc:`HDF5-Output`). All quantities are
in atomic units. Expectation values are formed with the coefficient vector
:math:`\mathbf c` of the state and the matrices of the B-spline basis; the
inner product is the one of the mass (overlap) matrix,

.. math::
   :label: eq-observables-inner

   \langle\psi|\phi\rangle_M = \mathbf c_\psi^\dagger\,\mathbf M\,\mathbf c_\phi ,
   \qquad
   \langle A\rangle = \mathbf c^\dagger\,\mathbf A\,\mathbf c ,

where :math:`\mathbf A` is the Galerkin matrix of the operator. Expectation
values are not divided by the norm, so with a CAP they decrease with the
absorbed probability. ``PhysicsOutput`` (:doc:`Output-Control`) selects which
groups are computed; the others are written as zeros.

Here :math:`q` = ``Charge`` and :math:`s = -q` (:doc:`Physics-Conventions`);
the propagated Hamiltonian is
:math:`H_E(t) = K + sV + s\,\mathbf E(t)\cdot\mathbf r` (plus the CAP when
enabled), with :math:`V` the ``Potential`` expression.

Evaluation
----------

The matrix-vector products of a monitor call are reduced in one fused global
reduction. In 2D and 3D two further optimisations apply, each checked against
the assembled matrices on the state at the first monitor call and used only if
it agrees to a relative :math:`10^{-10}`; the matrices that are then no longer
needed are freed:

- **Factorised operators.** On the tensor B-spline basis :math:`\mathbf M`,
  the position matrices :math:`\mathbf D_a`, and for a constant mass the
  velocities and :math:`\mathbf K`, are Kronecker products of 1D banded
  matrices and are applied without the assembled matrix.
  ``-tdsez_tensor_ops 0`` disables this.
- **Quadrature expectation values.** :math:`\langle V\rangle`,
  :math:`\langle 1/m\rangle`, :math:`\langle K\rangle` (varying mass), the CAP,
  the forces and the velocities are summed directly over the quadrature points
  of the assembly. ``-tdsez_quad_obs 0`` disables this.

Neither is used with Dirichlet walls (``BoundaryType = dirichlet``) or in 1D.
At start-up TDSE-Z prints which operators and expectation values use these
paths and the largest relative difference found in the check.

Norm
----

.. math::
   :label: eq-observables-norm

   N(t) = \langle\psi(t)|\psi(t)\rangle_M = \mathbf c^\dagger\mathbf M\mathbf c .

It is the column ``Norm`` of the ``energies`` dataset.

Dipole
------

.. math::
   :label: eq-observables-dipole

   D_a(t) = q\,\langle\psi(t)|\,x_a\,|\psi(t)\rangle ,
   \qquad a \in \{x, y, z\},

computed as :math:`q\,\mathbf c^\dagger\mathbf D_a\mathbf c` with the
position matrix :math:`(\mathbf D_a)_{ij} = \int B_i\,x_a\,B_j`. The
``dipoles`` dataset stores, next to it, the physical field
:math:`E_a(t)` (the ``LaserX``/``LaserY``/``LaserZ`` value on an axis in
``Polarization``, zero otherwise) and the dipole acceleration. All spatial
axes are observed by default; with ``ObserveAllAxes = 0`` only the axes in
``Polarization`` are, and the columns of the others are zero.

Energies
--------

The ``energies`` dataset stores

.. math::
   :label: eq-observables-energies

   E_{\text{kin}} = \langle K\rangle, \qquad
   E_{\text{pot}} = \langle sV\rangle, \qquad
   E_{\text{int}} = s\,\mathbf E(t)\cdot\langle\mathbf r\rangle, \qquad
   E_{\text{tot}} = \mathrm{Re}\,\langle H\rangle + E_{\text{int}} ,

where :math:`E_{\text{pot}}` is the potential term as it enters the
Hamiltonian (including the sign :math:`s`), :math:`E_{\text{int}}` the
expectation value of the field coupling, and :math:`H` the field-free
Hamiltonian including the CAP. The CAP :math:`-iW` contributes only to the
imaginary part of :math:`\langle H\rangle`, so
:math:`E_{\text{tot}} = E_{\text{kin}} + E_{\text{pot}} + E_{\text{int}}`. The
dataset also stores

.. math::
   :label: eq-observables-invmass

   \langle m^{-1}\rangle = \langle\psi|\,m(\mathbf r)^{-1}\,|\psi\rangle ,

which enters the operator form of the acceleration and indicates which mass
regions the state occupies. The column layout is given in :doc:`HDF5-Output`.

Acceleration
------------

The dipole acceleration :math:`A_a = q\,\mathrm d\langle v_a\rangle/\mathrm dt
= \ddot D_a` (columns ``A*`` of ``dipoles``) is computed in one of two forms,
selected by ``AccelerationMethod``.

**operator** (Ehrenfest force, acceleration form [Burnett1992]_):

.. math::
   :label: eq-observables-acc-operator

   A_a(t) = q\,s\left[\Big\langle -\frac{1}{m}\,\partial_a V \Big\rangle
            + \text{(mass-gradient terms)}
            - E_a(t)\,\Big\langle \frac{1}{m}\Big\rangle\right] .

:math:`\partial_a V` is the ``PotentialDerivativeX``/``Y``/``Z`` expression
(:doc:`Potential-Mass`), which a propagation run requires on every axis. The
form is accurate for a smooth potential and mass and a wave packet that does
not reach a reflecting wall.

**discrete** (equation of motion of the Galerkin system
:eq:`eq-time-propagation-eom`):

.. math::
   :label: eq-observables-acc-discrete

   A_a(t) = -2q\,\mathrm{Im}\big(\mathbf w^\dagger\,\mathbf V_a\,\mathbf c\big),
   \qquad
   \mathbf w = \mathbf M^{-1}\big(\mathbf H_h + s\,\mathbf E(t)\cdot\mathbf D
              + s\textstyle\sum_k f_k(t)\,\mathbf G_k\big)\mathbf c ,

with the velocity matrix :math:`\mathbf V_a`, the structured field terms
:math:`\mathbf G_k` (:ref:`structured-fields`) and
:math:`\mathbf H_h = (\mathbf H + \mathbf H^\dagger)/2`, the Hermitian part of
:math:`\mathbf H`. Without a CAP this is the exact time derivative of
:math:`q\langle v_a\rangle` for the discrete dynamics, including sharp
interfaces and walls; with a CAP the absorber is excluded, which gives the
Ehrenfest acceleration of the physical Hamiltonian. It costs one solve with
:math:`\mathbf M` per step: conjugate gradients with a block-Jacobi ILU(0)
preconditioner to :math:`10^{-14}` relative (about 16 iterations; options prefix
``accel_``, e.g. ``-accel_ksp_type preonly -accel_pc_type lu`` for a direct
solve). A solve that does not converge stops the run.

``auto`` (default) uses the discrete form when the ``Mass`` expression depends
on :math:`x`, :math:`y` or :math:`z` (and ``MassIsConstant`` is not set) or a
``FieldTerm`` is given, the operator form otherwise. For smooth problems the two forms agree up to the
spatial discretisation error and converge to each other with the grid (3D,
position-dependent mass, degree 4: relative difference
:math:`2.4\times10^{-4}` at :math:`h_x = 1.1`, :math:`1.5\times10^{-5}` at
:math:`h_x = 0.75`). For an eigenstate the discrete form is exactly zero and
the operator form zero to the discretisation error. Once the wave packet
reaches a reflecting wall they differ, because only the discrete form contains
the wall force; use an absorbing potential for HHG (:doc:`HHG`).

The current :math:`J_a = \dot D_a` and the acceleration
:math:`A_a = \dot J_a` give equivalent harmonic spectra,

.. math::
   :label: eq-observables-hhg

   |\tilde A_a(\omega)|^2 = \omega^2\,|\tilde J_a(\omega)|^2
   = \omega^4\,|\tilde D_a(\omega)|^2 .

Current
-------

.. math::
   :label: eq-observables-current

   J_a(t) = q\,\langle\psi(t)|\,v_a\,|\psi(t)\rangle = \dot D_a(t),

with the velocity operator :math:`v_a` (:math:`p_a/m` for a constant mass).
The ``currents`` dataset also splits :math:`J_a` with the projections of the
state on the bound eigenstates :math:`\phi_n` kept by the eigensolver
(``NBoundStates``, extended to the end of a degenerate level),

.. math::
   :label: eq-observables-projection

   c_n(t) = \langle\phi_n|\psi(t)\rangle_M ,
   \qquad
   v^{(a)}_{mn} = \langle\phi_m|\,v_a\,|\phi_n\rangle ,

where the matrix :math:`v^{(a)}_{mn}` is computed once before the
propagation:

.. math::
   :label: eq-observables-current-split

   J^{\text{intra}}_a = q\sum_n |c_n|^2\, v^{(a)}_{nn},
   \qquad
   J^{\text{inter}}_a = q\sum_{m\ne n} \bar c_m\, c_n\, v^{(a)}_{mn},
   \qquad
   J^{\text{bc}}_a = J_a - J^{\text{intra}}_a - J^{\text{inter}}_a .

:math:`J^{\text{intra}}` is the contribution of the populations,
:math:`J^{\text{inter}}` that of the coherences between bound states, and the
remainder :math:`J^{\text{bc}}` contains all terms involving the part of the
state outside the span of the bound states (bound-continuum and
continuum-continuum).

Angular Momentum
----------------

Column ``Lz`` of ``currents`` holds
:math:`\langle L_z\rangle = \langle\psi|-i(x\partial_y - y\partial_x)|\psi\rangle`.
The :math:`L_z` matrix is assembled only in 2D with ``EnableLzDiag = 1``; in
all other cases the column is zero.

Phase Increment
---------------

Column ``dGamma`` of ``currents`` holds the phase increment of the
bound-state component between consecutive steps,

.. math::
   :label: eq-observables-dgamma

   \Delta\Gamma(t) = -\arg\Big(\sum_n \bar c_n(t - h)\,c_n(t)\Big),

with :math:`c_n` from :eq:`eq-observables-projection` (zero at the first
step). It contains the dynamical phase; the accumulated phase
:math:`\Gamma(t) = \sum_{t' \le t}\Delta\Gamma(t')` is obtained as the
cumulative sum of the column.

Autocorrelation
---------------

.. math::
   :label: eq-observables-autocorrelation

   C(t) = \langle\psi(0)|\psi(t)\rangle_M = \mathbf c(0)^\dagger\,\mathbf M\,\mathbf c(t),

with :math:`\psi(0)` the initial state. :math:`|C(t)|^2` is the survival
probability; the Fourier transform of :math:`C(t)` gives the spectrum of
the states contained in :math:`\psi(0)`. The ``autocorrelation`` dataset
stores :math:`t`, :math:`\mathrm{Re}\,C` and :math:`\mathrm{Im}\,C`.

Populations
-----------

.. math::
   :label: eq-observables-population

   P_n(t) = |c_n(t)|^2 = \big|\langle\phi_n|\psi(t)\rangle_M\big|^2 ,
   \qquad n = 0, \dots, N-1 ,

for the :math:`N` bound eigenstates kept by the eigensolver. The
``populations`` dataset stores :math:`t, P_0, \dots, P_{N-1}`. The sum
:math:`\sum_n P_n` is the population of these bound states; for normalised
states :math:`N(t) - \sum_n P_n` is the population outside them (excited,
unconverged and continuum states).
