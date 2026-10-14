Observables
===========

Per-step diagnostics computed in the TS monitors and streamed to HDF5. All
expectation values use the PETSc mass matrix :math:`M` as the inner-product
metric: :math:`\langle \psi | \phi \rangle_M = \psi^\dagger M \phi`.

Dipole Moment
-------------

.. math::
   :nowrap:

   \begin{equation}
   d_x(t) = q \langle \psi(t) | x | \psi(t) \rangle
   \end{equation}

where :math:`q` is the particle charge and :math:`x` is the position operator.
The dipole is computed via ``MatMult(Dx(), psi, tmp)`` followed by ``VecDot(psi, tmp)`` — a
single matrix-vector product and inner product at each time step.

Units: charge * length (e * :math:`a_0`). The Fourier transform of :math:`d_x(t)`
gives the absorption spectrum.

Energy Decomposition
--------------------

The code stores five energy quantities per timestep. The total energy is the sum
of the bare Hamiltonian expectation and the interaction energy:

.. math::
   :nowrap:

   \begin{equation}
   E_{\text{total}}(t) = \langle \psi(t) | H_0 | \psi(t) \rangle + E(t)\, d_x(t)
   \end{equation}

where :math:`H_0 = K + V` is the field-free Hamiltonian, :math:`E(t)` is the
laser electric field amplitude, and :math:`d_x(t)` is the dipole moment.

Additionally, kinetic and potential energy are stored separately:

.. math::
   :nowrap:

   \begin{equation}
   E_{\text{kin}}(t) = \langle \psi(t) | K | \psi(t) \rangle, \qquad
   E_{\text{pot}}(t) = \langle \psi(t) | V | \psi(t) \rangle
   \end{equation}

The HDF5 ``energy`` table stores these as columns:

.. raw:: html

   <table class="docutils align-default">
   <thead>
   <tr class="header">
   <th><p>t</p></th>
   <th><p>E_kin</p></th>
   <th><p>E_pot</p></th>
   <th><p>E_int</p></th>
   <th><p>E_tot</p></th>
   <th><p>1/m_avg</p></th>
   <th><p>norm²</p></th>
   </tr>
   </thead>
   <tbody>
   <tr class="odd">
   <td><p>time</p></td>
   <td><p>kinetic</p></td>
   <td><p>potential</p></td>
   <td><p>interaction</p></td>
   <td><p>total</p></td>
   <td><p>inv mass</p></td>
   <td><p>norm²</p></td>
   </tr>
   </tbody>
   </table>

The ``1/m_avg`` column stores :math:`\langle \psi | m(\mathbf{r})^{-1} | \psi
\rangle`, the expectation value of the inverse mass, which is useful for
diagnosing mass-profile effects.

Acceleration
------------

Acceleration is computed from the Ehrenfest theorem:

.. math::
   :nowrap:

   \begin{equation}
   a_x(t) = \langle \psi(t) | -\nabla_x V | \psi(t) \rangle - \frac{q\,E(t)}{m_{\text{avg}}}
   \end{equation}

The first term :math:`\langle \psi | -\nabla V | \psi \rangle` is the force from
the potential gradient. The second term subtracts the external field contribution
scaled by the average inverse mass. This decomposition separates internal forces
from the driving field.

Current Decomposition
---------------------

The velocity-gauge current is measured by contracting the state with the
velocity operator :math:`\mathbf{v} = m(\mathbf{r})^{-1}\mathbf{p}`. In 2D and
3D the current is decomposed into three parts using the projection coefficients

.. math::
   c_n(t) = \langle \phi_n | M | \psi(t) \rangle

where :math:`|\phi_n\rangle` are the bound eigenstates computed by the
eigensolver. The density matrix in the bound-state basis is
:math:`\rho_{mn} = c_m^*(t) c_n(t)`.

- **Intra-band**:

  .. math::
     :nowrap:

     \begin{equation}
     J_{\text{intra}} = \sum_n |c_n|^2\, v_{nn}
     \end{equation}

  Current flowing within each individual eigenstate (diagonal density matrix
  elements).

- **Inter-band**:

  .. math::
     :nowrap:

     \begin{equation}
     J_{\text{inter}} = \sum_{m \neq n} c_m^* c_n\, v_{mn}
     \end{equation}

  Current from quantum coherences between different eigenstates (off-diagonal
  elements).

- **Bound-continuum**:

  .. math::
     :nowrap:

     \begin{equation}
     J_{\text{bc}} = J_{\text{total}} - J_{\text{intra}} - J_{\text{inter}}
     \end{equation}

  The remainder — current not accounted for by bound-state contributions,
  originating from the continuum part of the wavefunction.

The ``current`` table stores 11 columns:

.. raw:: html

   <table class="docutils align-default">
   <thead>
   <tr class="header">
   <th><p>t</p></th>
   <th><p>Lz</p></th>
   <th><p>dGamma</p></th>
   <th><p>Jx_tot</p></th>
   <th><p>Jy_tot</p></th>
   <th><p>Jx_intra</p></th>
   <th><p>Jy_intra</p></th>
   <th><p>Jx_inter</p></th>
   <th><p>Jy_inter</p></th>
   <th><p>Jx_bc</p></th>
   <th><p>Jy_bc</p></th>
   </tr>
   </thead>
   <tbody>
   <tr class="odd">
   <td><p>time</p></td>
   <td><p>Lz exp</p></td>
   <td><p>dGamma</p></td>
   <td><p>Jx total</p></td>
   <td><p>Jy total</p></td>
   <td><p>Jx intra</p></td>
   <td><p>Jy intra</p></td>
   <td><p>Jx inter</p></td>
   <td><p>Jy inter</p></td>
   <td><p>Jx bc</p></td>
   <td><p>Jy bc</p></td>
   </tr>
   </tbody>
   </table>

where :math:`L_z = \langle \psi | L_z | \psi \rangle` is the angular momentum
expectation, and :math:`d\Gamma` is the geometric phase increment per step
(described below).

This decomposition distinguishes semiconductor-like (inter-band driven) from
atomic-like (direct field ejection) ionization.

Autocorrelation
---------------

.. math::
   :nowrap:

   \begin{equation}
   A(t) = \langle \psi(0) | M | \psi(t) \rangle
   \end{equation}

:math:`|A(t)|^2` gives the survival probability (probability of finding the
system still in the initial state). The Fourier transform of :math:`A(t)` yields

The HDF5 ``autocorrelation`` table stores three columns:

.. raw:: html

   <table class="docutils align-default">
   <thead>
   <tr class="header">
   <th><p>t</p></th>
   <th><p>Re A(t)</p></th>
   <th><p>Im A(t)</p></th>
   </tr>
   </thead>
   <tbody>
   <tr class="odd">
   <td><p>time</p></td>
   <td><p>real part</p></td>
   <td><p>imag part</p></td>
   </tr>
   </tbody>
   </table>

The autocorrelation is computed as ``VecDot(tmpVec[0], psi0())`` where
``tmpVec[0] = M * psi(t)``, i.e. the mass-weighted inner product.

Berry Phase
-----------

The accumulated geometric (Berry) phase is computed from consecutive time-step
overlaps:

.. math::
   :nowrap:

   \begin{equation}
   \Delta\Gamma(t) = -\arg\Bigl( \langle \psi(t-\Delta t) | M | \psi(t) \rangle \Bigr)
   \end{equation}

.. math::
   :nowrap:

   \begin{equation}
   \Gamma(t) = \sum_{t'=0}^{t} \Delta\Gamma(t')
   \end{equation}

The geometric phase increment :math:`\Delta\Gamma` is written to the ``current``
table as ``dGamma``; the running total is stored in ``manager.BerryPhase`` and
survives across HDF5 flushes.

For circularly polarized fields, this accumulates the phase associated with the
rotating field geometry. A non-zero Berry phase indicates the state has acquired
a geometric phase beyond the dynamical phase :math:`\exp(-iEt/\hbar)`.

Populations
-----------

The population of each bound eigenstate is:

.. math::
   :nowrap:

   \begin{equation}
   P_n(t) = |c_n(t)|^2 = \bigl| \langle \phi_n | M | \psi(t) \rangle \bigr|^2
   \end{equation}

The ``population`` table stores columns ``[t, P_0, P_1, ..., P_{N-1}]`` where
N is the number of converged eigenstates. The sum :math:`\sum_n P_n(t)` gives
the bound-state population; the complement :math:`1 - \sum_n P_n` is the
continuum (ionized) fraction.