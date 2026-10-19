Potential & Mass
================

The static part of the Hamiltonian is defined by three inputs: the potential
:math:`V(\mathbf r)`, the position-dependent effective mass
:math:`m(\mathbf r)` and the carrier charge :math:`q`. Potential and mass are
muParser expressions of the coordinates; all quantities are in atomic units
(see :doc:`Physics-Conventions`).

Hamiltonian
-----------

The field-free Hamiltonian uses the BenDaniel-Duke form of the kinetic
operator [BenDanielDuke1966]_, which is Hermitian for a position-dependent
mass:

.. math::
   :label: eq-potential-mass-hamiltonian

   H_0 = -\frac{1}{2}\,\nabla\cdot\left(\frac{1}{m(\mathbf r)}\,\nabla\right)
         + s\,V(\mathbf r), \qquad s = -q .

In the Galerkin discretisation with basis functions :math:`B_a` (see
:doc:`Grid-Domain`) the operators of :eq:`eq-potential-mass-hamiltonian` become

.. math::
   :label: eq-potential-mass-matrices

   K_{ab} = \frac{1}{2}\int \frac{\nabla B_a\cdot\nabla B_b}{m(\mathbf r)}\,d\mathbf r,
   \qquad
   V_{ab} = s\int V(\mathbf r)\,B_a B_b\,d\mathbf r,
   \qquad
   M_{ab} = \int B_a B_b\,d\mathbf r ,

with :math:`H_0 = K + V` and :math:`M` the overlap matrix. The weak form of the
kinetic term needs only first derivatives, so the BenDaniel-Duke matching
conditions at a mass discontinuity (continuity of :math:`\psi` and of
:math:`m^{-1}\,\partial_n\psi`) are satisfied naturally; ``KnotSequence =
interface`` places C\ :sup:`0` knots at such discontinuities
(:doc:`Knot-Sequences`). :math:`m(\mathbf r)` and :math:`V(\mathbf r)` are
evaluated at every quadrature point.

Carrier Charge
--------------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Charge / ParticleCharge
     - real
     - ``-1.0``
     - Carrier charge :math:`q` in units of :math:`e`. ``-1`` is an electron,
       ``+1`` a hole or another positive carrier.

``Potential`` and the laser field are written for an electron. The carrier
sign :math:`s = -q` multiplies both the potential and the field term
(:doc:`Laser-Field`): for :math:`q = -1` they enter as written, for
:math:`q = +1` both are inverted, and a general :math:`q` scales them. A
positive ``Charge`` prints a note at start-up. The reported dipole, current
and acceleration are multiplied by :math:`q` (:doc:`Physics-Conventions`).

Potential
---------

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Potential / ExternalPotential
     - muParser
     - required
     - :math:`V(x,y,z)` in Hartree. The expression is evaluated at sample
       points inside the box before assembly; a parse error or a non-finite
       value stops the run.
   * - PotentialDerivativeX / Y / Z
     - muParser
     - none
     - Analytic :math:`\partial V/\partial x`, :math:`\partial V/\partial y`,
       :math:`\partial V/\partial z`. Required on every axis of the problem
       when ``EnablePropagation = 1`` (write ``0`` for a constant potential);
       they enter the force operator of the dipole acceleration
       (:doc:`Observables`). They are not computed from ``Potential``, and a
       wrong derivative gives a wrong acceleration.

Example (1D soft-core atom, from ``examples/hhg/hhg_1d.inp``)::

    Potential            = -1.0 / sqrt(x*x + 1.0)
    PotentialDerivativeX = x / (x*x + 1.0)^1.5

Mass
----

.. list-table::
   :widths: 30 15 15 40
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Mass
     - muParser
     - ``1.0``
     - Effective mass :math:`m(x,y,z)` in units of :math:`m_e`. It must be
       positive and finite; it is checked at sample points inside the open
       box (16 per axis), so a singularity exactly on a Dirichlet wall is
       admissible.
   * - MassIsConstant
     - bool
     - ``0`` (detected)
     - Set to 1 automatically when the ``Mass`` expression contains none of
       ``x``, ``y``, ``z``. A constant mass skips the finite-difference
       derivatives of :math:`1/m` in the force operator and selects the
       operator form of the acceleration with ``AccelerationMethod = auto``.
       Setting it to 1 for a position-dependent mass drops these gradient
       terms; the kinetic, velocity and :math:`1/m` matrices always evaluate
       :math:`m(\mathbf r)` at each quadrature point.

Example (Quesne oscillator, position-dependent mass on :math:`x > -2`)::

    Mass                 = (1.0 + x/2.0)^(-1.0)
    Potential            = x*x/(4.0*(x + 2.0))
    PotentialDerivativeX = x*(x + 4.0)/(4.0*(x + 2.0)*(x + 2.0))

Expressions and Constants
-------------------------

All expressions use muParser syntax. ``Potential``, ``PotentialDerivativeX/Y/Z``
and ``Mass`` are functions of ``x``, ``y``, ``z`` (unused coordinates are 0 in
1D and 2D); the field expressions are functions of ``t`` only
(:doc:`Laser-Field`). In the spatial expressions ``fabs(`` is accepted as a
synonym of ``abs(``. The constant ``pi`` is defined in the potential,
derivative and field expressions; in ``Mass`` use muParser's built-in ``_pi``.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - Variables
     - string
     - User-defined constants as comma-separated ``Name=value`` pairs, for
       example ``Variables = V0=1.2, w=0.5``. Values are numbers, not
       expressions. Several ``Variables`` lines accumulate (it is the only
       key that may be repeated). The constants are available in every
       expression: ``Potential``, ``PotentialDerivativeX/Y/Z``, ``Mass`` and
       the field expressions.

The following input parameters are also defined as constants in every
expression, with their parsed values:

``Q`` and ``Charge``, ``Dimension``, ``SplineDegree``, ``Nelements`` and
``NSplines``, ``LMinX``, ``LMaxX``, ``LMinY``, ``LMaxY``, ``LMinZ``,
``LMaxZ``, ``OffsetX``, ``OffsetY``, ``OffsetZ``, ``Ampx``, ``Ampy``,
``Ampz``, ``CEPx``, ``CEPy``, ``CEPz``, ``Omegax``, ``Omegay``, ``Omegaz``,
``TargetEigenvalue``, ``TimeStep``, ``FinalTime``, ``CAPKmin``,
``KnotAlpha``, ``NQuadratures``, ``NBoundStates``.

Other keys are not available as constants; define the values needed with
``Variables``.
