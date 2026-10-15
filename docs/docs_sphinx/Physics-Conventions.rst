Physics Conventions
===================

Units, sign conventions, the discrete equations of motion and the operators
used throughout TDSE-Z.

Units
-----

All input and output quantities are in atomic units: energy in Hartree,
length in Bohr (:math:`a_0`), time in :math:`\hbar/E_h`, mass in electron
masses :math:`m_e`, charge in elementary charges :math:`e`, and
:math:`\hbar = 1`. The field strength is in :math:`E_h/(e a_0)`.

Hamiltonian and Carrier Sign
----------------------------

The time-dependent Schrödinger equation is solved in the length gauge:

.. math::
   :label: eq-physics-conventions-tdse

   i\,\frac{\partial\psi}{\partial t} =
   \left[-\frac{1}{2}\,\nabla\cdot\left(\frac{1}{m(\mathbf r)}\,\nabla\right)
         + s\,V(\mathbf r) + s\,\mathbf E(t)\cdot\mathbf r
         - i\,W(\mathbf r)\right]\psi,
   \qquad s = -q .

* The kinetic term has the BenDaniel-Duke form [BenDanielDuke1966]_ with the
  effective mass ``Mass`` (:doc:`Potential-Mass`).
* :math:`V` is ``Potential`` and :math:`\mathbf E(t)` the field of
  ``LaserX/Y/Z`` (or ``Laser``) on the polarised axes, zero on the others
  (:doc:`Laser-Field`).
* :math:`q` is ``Charge`` in units of :math:`e` (default :math:`-1`,
  electron). Potential and field are written for an electron: for
  :math:`q = -1` (:math:`s = 1`) they enter as written and the force of the
  field is :math:`-\mathbf E`; for :math:`q = +1` (hole) both are inverted,
  :math:`H = K - (V + \mathbf r\cdot\mathbf E)`; a general :math:`q` scales
  both, :math:`H = K - q\,(V + \mathbf r\cdot\mathbf E)`.
* :math:`W \ge 0` is the complex absorbing potential [Manolopoulos2002]_,
  present only with ``EnableCAP = 1`` and only during the propagation; the
  eigensolver uses the Hermitian :math:`H_0` (:doc:`Boundary-Conditions`).

Discrete Equations
------------------

With :math:`\psi(\mathbf r, t) = \sum_j c_j(t)\,B_j(\mathbf r)` in the
B-spline basis [deBoor2001]_ [Bachau2001]_, assembled with PetIGA
[Dalcin2016]_ [Hughes2005]_, the Galerkin equations are

.. math::
   :label: eq-physics-conventions-semidiscrete

   i\,M\,\dot{\mathbf c} = \Big(H + s \sum_{a\in P} E_a(t)\,D_a\Big)\,\mathbf c ,

with the overlap matrix :math:`M`, :math:`H = K + V - iW` (matrices of
:eq:`eq-potential-mass-matrices`, :math:`W` only with the CAP) and the dipole
matrices :math:`D_a`. The Crank-Nicolson propagator integrates the residual
form

.. math::
   :label: eq-physics-conventions-residual

   F(t, \mathbf c, \dot{\mathbf c}) =
   i\,M\,\dot{\mathbf c} - H\,\mathbf c - s \sum_{a\in P} E_a(t)\,D_a\,\mathbf c = 0 .

Operators
---------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Quantity
     - Matrix
   * - Overlap
     - :math:`M_{ij} = \int B_i B_j\,d\mathbf r`. It defines the inner
       product :math:`\langle\phi|\psi\rangle = \mathbf a^\dagger M\,\mathbf c`
       used for all norms, projections and populations.
   * - Dipole (position)
     - :math:`(D_a)_{ij} = \int B_i\,x_a\,B_j\,d\mathbf r`.
   * - Velocity
     - :math:`(V_a)_{ij} = -\dfrac{i}{2}\displaystyle\int \frac{1}{m}
       \left(B_i\,\partial_a B_j - \partial_a B_i\,B_j\right) d\mathbf r`,
       the Galerkin matrix of the symmetrised operator in
       :eq:`eq-physics-conventions-velocity`. It is :math:`-i/2` times a real
       antisymmetric matrix, hence Hermitian.
   * - Inverse mass
     - :math:`(M_d)_{ij} = \int B_i B_j / m\,d\mathbf r`.

The velocity operator of the BenDaniel-Duke Hamiltonian is

.. math::
   :label: eq-physics-conventions-velocity

   \hat v_a = i\,[H_0, x_a]
            = -\frac{i}{2}\left(\frac{1}{m}\,\partial_a
              + \partial_a\,\frac{1}{m}\right) .

The reported observables are multiplied by the charge: dipole
:math:`q\langle\mathbf r\rangle`, current :math:`q\langle\hat{\mathbf v}\rangle`
and acceleration :math:`q\,d\langle\hat{\mathbf v}\rangle/dt`
(:doc:`Observables`).

Time Stepping
-------------

``Propagator`` selects the integrator of
:eq:`eq-physics-conventions-semidiscrete` (:doc:`Time-Propagation`):

* ``cn`` (default): Crank-Nicolson [Crank1947]_, PETSc ``TSTHETA`` with
  :math:`\theta = 1/2` [PETSc]_, applied to
  :eq:`eq-physics-conventions-residual`.
* ``pade22``: exponential midpoint rule with a Padé(2,2) approximation of the
  exponential.
* ``magnus4``: commutator-free fourth-order Magnus integrator
  [BlanesMoan2006]_ [Alvermann2011]_ with Padé(2,2) exponentials.
