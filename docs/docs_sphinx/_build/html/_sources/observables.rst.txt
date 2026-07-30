observables
===========

Observables
===========

Dipole moment, energy decomposition, current decomposition, autocorrelation,
Berry phase.

Dipole Moment
-------------

``d(t) = q * <psi(t)|r|psi>``. Units: charge * length (e * a_0). The Fourier
transform of d(t) gives the absorption spectrum.

Energy Decomposition
--------------------

``E_total = <psi|K|psi> + <psi|V|psi> + <psi|H> + E(t) * d(t)``

- **K** = -1/2 * nabla^2/m(r) -- kinetic energy operator
- **V** -- potential energy
- **<psi\|H\|psi>** -- bare Hamiltonian expectation
- **E(t)*d(t)** -- interaction energy (length gauge)

Current Decomposition
---------------------

The velocity-gauge current ``J = <psi|v|psi>`` decomposes into three parts:

- **Intra-band**: sum_n \|c_n\|^2 * v_nn -- current within each occupied state
- **Inter-band**: sum_{m!=n} c_m* c_n * v_mn -- current from state coherences
- **Bound-continuum**: J_total - J_intra - J_inter -- continuum components

This decomposition distinguishes semiconductor-like (inter-band driven) from
atomic-like (direct field ejection) ionization.

Autocorrelation
---------------

``A(t) = <psi(0)|psi(t)>``. \|A(t)\|^2 gives the survival probability. The Fourier
transform of A(t) gives the energy spectrum.

Berry Phase
-----------

For circularly polarized fields, the accumulated geometric phase:

::

    Gamma(t) = sum_steps -arg( sum_n c_n*(t-dt) * c_n(t) )

Accumulates in ``manager.BerryPhase``.
