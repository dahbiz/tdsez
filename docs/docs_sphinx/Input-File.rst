Input File
==========

A TDSE-Z run is configured by one plain-text input file, passed with
``-inp <file>`` (default ``tdse.prm``; see :doc:`CLI-Flags`). This page
describes the syntax and lists every key that the parser accepts, with its
type, default and meaning. The physics behind each group is explained on the
linked pages.

Syntax
------

- Each setting is one line ``Key = value``. Leading and trailing white space
  of key and value is removed.
- ``#`` starts a comment that extends to the end of the line. A ``#`` therefore
  cannot occur inside a value.
- Lines without ``=`` are ignored.
- Key names are case-sensitive. Values are taken literally: quotation marks are
  not removed and become part of the value, so values are written without
  quotes.
- Each key may appear only once. A repeated key is a fatal error with
  ``StrictInput = 1`` and a warning (the last value is used) with
  ``StrictInput = 0``. ``Variables`` is the only key that may be repeated; its
  lines accumulate.
- An unknown key is a fatal error with ``StrictInput = 1`` (default) and a
  warning with ``StrictInput = 0``. ``StrictInput`` takes effect from the line
  on which it appears, so ``StrictInput = 0`` must precede the keys it should
  tolerate.
- Aliases are alternative names for the same setting. A key and its alias count
  as different keys for the repetition check; give only one of them.

Value types
~~~~~~~~~~~

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Type
     - Accepted values
   * - int, real
     - A decimal number (``1e-3`` notation allowed). Integers are read with
       ``std::stoi``, reals with ``std::stod``.
   * - bool
     - ``true``/``yes``/``on`` or ``false``/``no``/``off`` (case-insensitive),
       or an integer (non-zero is true).
   * - string
     - Free text, used as given. Enumerated values are listed per key; where
       noted they are case-insensitive.
   * - expression
     - A muParser expression (see `Expressions`_).
   * - list[3]
     - Per-axis values ``vx, vy, vz``. A single value applies to all three
       axes. With two values the third axis keeps its default (the field
       lists ``DriverAmplitude``, ``DriverFrequency``, ``DriverCEP`` and their
       aliases instead fill missing entries with the first value).

Expressions
~~~~~~~~~~~

``Potential``, ``PotentialDerivativeX/Y/Z``, ``Mass`` and the field keys
``Laser``, ``LaserX/Y/Z`` are evaluated with muParser. The spatial expressions
use the variables ``x``, ``y``, ``z`` (atomic units), the field expressions the
time ``t``. Available in every expression:

- the muParser built-in functions and operators (for example ``sin``, ``cos``,
  ``exp``, ``sqrt``, ``abs``, ``min``, ``max``, ``^`` and the ternary
  operator ``cond ? a : b``);
- the constant ``pi``;
- the input scalars ``Q`` and ``Charge``, ``Dimension``, ``SplineDegree``,
  ``Nelements`` and ``NSplines``, ``LMinX`` … ``LMaxZ``, ``OffsetX/Y/Z``,
  ``Ampx/y/z``, ``CEPx/y/z``, ``Omegax/y/z``, ``TargetEigenvalue``,
  ``TimeStep``, ``FinalTime``, ``CAPKmin``, ``KnotAlpha``, ``NQuadratures``
  and ``NBoundStates``, with the values given in the input file;
- user constants declared with ``Variables``.

In the spatial expressions ``fabs(`` is replaced by ``abs(`` before parsing.

``Variables = Name=value, Name=value, ...`` declares named numeric constants.
Several ``Variables`` lines accumulate::

    Variables = V0=1.2, w=0.5
    Variables = Rc=4.0          # adds Rc; V0 and w remain defined
    Potential = -V0*exp(-w*(x^2+y^2)/Rc^2)

Removed keys
~~~~~~~~~~~~

The following keys are recognised but not supported. They are reported
with the reason given below; with ``StrictInput = 1`` the run stops, except for
the harmless values marked in the last column, which only produce a warning.

.. list-table::
   :widths: 30 52 18
   :header-rows: 1

   * - Key
     - Replacement
     - Harmless value
   * - ``Amplitude``, ``Omega``, ``Phase``
     - ``Ampx/y/z``, ``Omegax/y/z``, ``CEPx/y/z``, or ``Variables``
     - --
   * - ``PulseDuration``, ``PulseCenter``
     - ``Variables = PulseDuration=..., PulseCenter=...``
     - --
   * - ``EnvelopeType``, ``Envelope``
     - Envelope written in the field expression (``Laser``, ``LaserX/Y/Z``)
     - ``EnvelopeType`` empty or ``custom``
   * - ``Hbar``, ``Planck``
     - None; TDSE-Z uses atomic units (:math:`\hbar = 1`)
     - ``1``
   * - ``Gamma``, ``CapStrength``
     - ``CAPKmin`` or ``CAPWidth``
     - --
   * - ``dinvMassX/Y/Z``
     - None; mass gradients are computed from ``Mass``
     - --
   * - ``LMin``, ``LMax``
     - ``Domain`` or ``LMinX`` … ``LMaxZ``
     - --
   * - ``AdaptiveKappa``, ``AdaptivePower``, ``AdaptiveWFCoarseN``,
       ``AdaptiveWFKinLambda``
     - None; the adaptive knot sequences are not available
     - --
   * - ``OutputStrideAC``
     - None; the autocorrelation is the ``autocorrelation`` dataset of
       ``td/<name>_observables.h5``, recorded every step
     - any value
   * - ``SurffCouplingSign``
     - None; the t-SURFF coupling sign follows ``Charge``
     - --

After parsing, the input is validated (``ValidateOrThrow``) before any matrix
is assembled; an inconsistent combination stops the run with a message that
names the offending key. The checks are listed with the keys below.

Key Reference
-------------

Defaults are those of ``include/tdsez_parser.hpp``. "Required" marks keys
without a usable default.

General
~~~~~~~

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``StrictInput``
     - bool
     - 1
     - Unknown, removed and repeated keys are fatal (1) or warnings (0).
   * - ``Verbose``
     - bool
     - 1
     - Verbose start-up and progress output.
   * - ``Variables``
     - string
     - --
     - User constants for all expressions, ``Name=value`` pairs (see
       `Expressions`_).
   * - ``EnablePropagation``
     - bool
     - 1
     - Run the time propagation after the eigensolve; 0 computes the bound
       states only.
   * - ``EnableGPU``
     - bool
     - 0
     - Create all PetIGA matrices and vectors on the GPU
       (``-iga_mat_type aijcusparse``, ``-iga_vec_type cuda`` unless given).
       Requires a CUDA-enabled PETSc; otherwise the run stops. Overridden by
       ``-enable_gpu``.

Grid and discretisation
~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`Grid-Domain` and :doc:`Knot-Sequences`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``Dimension``
     - int
     - 1
     - Spatial dimension, 1, 2 or 3.
   * - ``Domain``
     - string
     - --
     - Bounds of all axes: ``lo, hi`` for every axis, or
       ``[lox,hix], [loy,hiy], [loz,hiz]`` per axis (missing pairs copy the
       x pair).
   * - ``LMinX``, ``LMaxX``, ``LMinY``, ``LMaxY``, ``LMinZ``, ``LMaxZ``
     - real
     - -10, 10
     - Bounds of each axis; ``LMax > LMin`` is required.
   * - ``OffsetX``, ``OffsetY``, ``OffsetZ``
     - real
     - 0
     - Translation of the mesh along each axis (see :doc:`Grid-Domain`).
   * - ``Nelements`` (``NSplines``, ``NumberOfSplines``, ``ElementCount``,
       ``NumberOfElements``)
     - int
     - 100
     - Number of elements per axis, at least 1.
   * - ``SplineDegree``
     - int
     - 3
     - Polynomial degree :math:`p`, 1 to 14.
   * - ``NQuadratures``
     - int
     - 8
     - Quadrature points per element and axis (PetIGA provides rules up
       to 10).
   * - ``QuadratureRule`` (``QuadratureType``)
     - string
     - ``legendre``
     - ``legendre``, ``lobatto`` or ``reduced``; case-insensitive.
   * - ``Basis`` (``BasisType``)
     - string
     - ``bspline``
     - ``bspline``, ``bernstein``, ``lagrange`` or ``spectral``;
       case-insensitive.
   * - ``KnotSequence``
     - list[3]
     - ``uniform``
     - Knot sequence per axis: ``uniform``, ``symexp``, ``symtanu``,
       ``symtann``, ``symlogtan``, ``hydrogenic``, ``interface``
       (``interfaces``); case-insensitive.
   * - ``KnotAlpha`` (``KnotGrading``, ``GradingStrength``)
     - real
     - 0
     - Grading strength of the graded sequences; a value :math:`\le 0` selects
       the defaults 0.9 (``symexp``, ``symtanu``, ``symtann``) and 0.5
       (``symlogtan``).
   * - ``HydrogenicNLin`` (``HydrogenicInnerElements``, ``InnerElements``)
     - list[3]
     - 0
     - ``hydrogenic``: linear elements per half axis; 0 uses
       ``Nelements - 1``.
   * - ``HydrogenicR1`` (``HydrogenicInnerSpacing``, ``InnerSpacing``)
     - list[3]
     - 0
     - ``hydrogenic``: width of the linear elements (a.u.); 0 uses 0.1.
   * - ``HydrogenicNExp`` (``HydrogenicOuterElements``, ``OuterElements``)
     - list[3]
     - 0
     - ``hydrogenic``: exponentially graded elements per half axis; 0 uses 10.
   * - ``HydrogenicHMax`` (``HydrogenicMaxSpacing``)
     - list[3]
     - 0
     - ``hydrogenic``: largest element width (a.u.); 0 derives it per axis
       from the field when propagating (see ``TransverseElementFactor``), a
       negative value disables the cap.
   * - ``TransverseElementFactor``
     - real
     - 2
     - Element width allowed on axes not coupled to the field, as a multiple of
       the polarised-axis width :math:`\pi/(2k_{\max})`; used by the automatic
       ``HydrogenicHMax`` and the field-resolution check (:doc:`Knot-Sequences`).
       Must be at least 1.

Potential, mass and carrier
~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`Potential-Mass` and :doc:`Physics-Conventions`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``Potential`` (``ExternalPotential``)
     - expression
     - required
     - Potential :math:`V(x,y,z)` in Hartree, written for an electron. Must be
       finite at the validation sample points inside the domain.
   * - ``PotentialDerivativeX``, ``PotentialDerivativeY``,
       ``PotentialDerivativeZ``
     - expression
     - --
     - :math:`\partial V/\partial x_a`. Required on every axis of the problem
       when ``EnablePropagation = 1`` (``0`` for a constant potential).
   * - ``Mass``
     - expression
     - ``1.0``
     - Effective mass :math:`m(x,y,z)`; must be positive and finite inside the
       domain.
   * - ``MassIsConstant``
     - bool
     - 0
     - Use the constant-mass assembly. Set automatically when ``Mass`` contains
       none of ``x``, ``y``, ``z``; setting it for a position-dependent mass
       drops the mass-gradient terms.
   * - ``Charge`` (``ParticleCharge``)
     - real
     - -1
     - Carrier charge :math:`q` in units of :math:`e`: -1 electron, +1 hole.
       See :doc:`Physics-Conventions`.

Laser field
~~~~~~~~~~~

See :doc:`Laser-Field`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``Polarization`` (``LaserPolarization``, ``DriverPolarization``)
     - string
     - --
     - Field-coupled axes: ``none``, ``x``, ``y``, ``z``, ``xy``, ``xz``,
       ``yz``, ``xyz`` or ``all``; case-insensitive. Only axes of the problem
       are allowed. Required (not ``none``) when ``EnablePropagation = 1``,
       unless a ``FieldTerm`` is given.
   * - ``LaserX``, ``LaserY``, ``LaserZ`` (``DriverX``, ``DriverY``,
       ``DriverZ``)
     - expression
     - --
     - Field component :math:`E_a(t)` in a.u., envelope included. Each
       polarised axis needs one, or ``Laser``.
   * - ``Laser`` (``Driver``)
     - expression
     - --
     - Field used on every polarised axis without its own ``LaserX/Y/Z``.
   * - ``FieldTerm`` (``StructuredField``)
     - ``w(x,y,z) ; f(t)``
     - --
     - Separable structured field term: adds :math:`s\,f(t)\,w(\mathbf r)`
       to :math:`H`. Repeat the line for several terms. See
       :ref:`structured-fields`.
   * - ``Ampx``, ``Ampy``, ``Ampz``
     - real
     - 0
     - Named constants for the field expressions (peak amplitudes); also used
       by the field-scale estimate.
   * - ``Omegax``, ``Omegay``, ``Omegaz``
     - real
     - 0
     - Named constants (carrier frequencies). On polarised axes the largest
       value is the :math:`\omega` of the field-scale estimate.
   * - ``CEPx``, ``CEPy``, ``CEPz``
     - real
     - 0
     - Named constants (carrier-envelope phases, rad).
   * - ``DriverAmplitude`` (``LaserAmplitude``)
     - list[3]
     - --
     - Sets ``Ampx, Ampy, Ampz``.
   * - ``DriverFrequency`` (``LaserFrequency``)
     - list[3]
     - --
     - Sets ``Omegax, Omegay, Omegaz``.
   * - ``DriverCEP`` (``CarrierEnvelopePhase``)
     - list[3]
     - --
     - Sets ``CEPx, CEPy, CEPz``.

Eigensolver
~~~~~~~~~~~

See :doc:`Eigensolver`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``NBoundStates`` (``NumberOfBoundStates``)
     - int
     - 1
     - Number of bound states. 0 skips the eigensolve and requires an
       analytic ``InitialState``.
   * - ``TargetEigenvalue`` (``TargetEnergy``)
     - real
     - 0
     - Shift-invert target (Hartree); the states nearest to it are computed.
       A negative value is also the :math:`I_p` of the field-scale estimate
       (0.5 otherwise).
   * - ``UseDirectSolve`` (``DirectSolve``)
     - bool
     - 0
     - Try the direct factorisation first for the shift-invert solves; 0
       selects the first inner solver automatically.
   * - ``EnableLzDiag`` (``LzDiag``)
     - bool
     - 0
     - 2D only: assemble :math:`L_z` and resolve degenerate bound-state levels
       by angular momentum before propagation.
   * - ``NBoundStatesSave`` (``SaveBoundStates``)
     - bool
     - 0
     - Write the bound states to ``static/<name>_eigen.h5``.
   * - ``BoundStateFormat`` (``SaveStateFormat``, ``StateFormat``)
     - string
     - ``complex``
     - ``complex`` (eigenvectors as computed) or ``real`` (each state rotated
       to a real canonical phase).

Initial state
~~~~~~~~~~~~~

See :doc:`Initial-State`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``InitialState`` (``StartState``)
     - string
     - ``ground``
     - ``ground``, ``state:N``, ``sup: c1*N + c2*M [+ c3*P]`` (up to three
       bound states, complex coefficients in parentheses),
       ``wavepacket``, ``analytic:quesne01`` or ``analytic:massstep01``
       (1D benchmarks). Bound-state indices must be below ``NBoundStates``.
       Case-sensitive.
   * - ``NormalizeInitialState`` (``NormalizePsi0``)
     - bool
     - 1
     - Normalise the initial state to :math:`\langle\psi|M|\psi\rangle = 1`.
   * - ``WavepacketCenter``
     - list[3]
     - 0
     - Centre :math:`c_d` of the Gaussian wave packet (a.u.).
   * - ``WavepacketMomentum``
     - list[3]
     - 0
     - Mean momentum :math:`k_d` (a.u.).
   * - ``WavepacketWidth``
     - list[3]
     - 1
     - Standard deviation :math:`w_d` of :math:`|\psi_0|^2` (a.u.), > 0.

Time propagation
~~~~~~~~~~~~~~~~

See :doc:`Time-Propagation`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``TimeStep`` (``TimeStepSize``)
     - real
     - 0.01
     - Time step (a.u.), > 0 when propagating.
   * - ``FinalTime`` (``TotalTime``)
     - real
     - 100
     - End of the propagation (a.u.), > 0 when propagating.
   * - ``Propagator``
     - string
     - ``cn``
     - ``cn`` (Crank-Nicolson), ``pade22`` (exponential midpoint rule with a
       Padé(2,2) exponential) or ``magnus4`` (fourth-order commutator-free
       Magnus); case-sensitive.
   * - ``Preconditioner``
     - string
     - ``asm``
     - ``asm`` (restricted additive Schwarz, ILU(0) blocks), ``mg`` (geometric
       B-spline multigrid; requires ``Basis = bspline``) or ``gamg`` (PETSc
       algebraic multigrid); case-sensitive.
   * - ``MGLevels``
     - int
     - 0
     - Multigrid levels: 0 (automatic) or at least 2.
   * - ``MGSmooth``
     - int
     - 1
     - Multigrid smoothing sweeps before and after each coarse correction,
       at least 1.
   * - ``AccelerationMethod``
     - string
     - ``auto``
     - Dipole acceleration: ``operator`` (Ehrenfest force operator),
       ``discrete`` (time derivative of :math:`\langle v\rangle` from the
       Galerkin equation of motion) or ``auto`` (``discrete`` for a
       position-dependent mass or a ``FieldTerm``, else ``operator``). See
       :doc:`Observables`.

Boundaries and absorbing potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`Boundary-Conditions`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``BoundaryType`` (``Boundary``)
     - string
     - ``Neumann``
     - ``Neumann`` (alias value ``natural``) or ``Dirichlet`` (alias value
       ``wall``); case-insensitive.
   * - ``EnableCAP`` (``AbsorbingBoundary``)
     - bool
     - 0
     - Add the Manolopoulos complex absorbing potential at all walls.
   * - ``CAPKmin``
     - real
     - 0.1
     - Lowest momentum (a.u.) absorbed by the CAP; sets the layer width
       :math:`w = 6.555/k_{\min}`. Must be > 0.
   * - ``CAPWidth`` (``CAPLayerWidth``)
     - real
     - 0
     - Layer width per wall (a.u.). A positive value overrides ``CAPKmin``
       (:math:`k_{\min} = 6.555/w`); 0 uses ``CAPKmin``; negative values are
       rejected. The layer may cover at most 45 % of each axis.

Observables and output
~~~~~~~~~~~~~~~~~~~~~~

See :doc:`Output-Control`, :doc:`Observables` and :doc:`HDF5-Output`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``PhysicsOutput`` (``OutputQuantities``)
     - string
     - ``all``
     - Comma-separated subset of ``position``, ``acceleration``, ``dipole``
       (both), ``population``, ``energy``, ``current``, ``autocorrelation``
       (``ac``), ``wfs``, or ``all`` / ``none``; case-insensitive. Only the
       listed quantities are computed; the HDF5 layout is unchanged
       (:doc:`Output-Control`). Unknown entries stop the run.
   * - ``ObserveAllAxes``
     - bool
     - 1
     - Record dipole, current and acceleration on every axis (1) or only on
       the polarised axes (0, fewer operators).
   * - ``OutputStrideTS``
     - int
     - 0
     - > 0: every N steps, write the time-series buffer (at most every 2 s of
       wall time) and a time-stepper snapshot to ``td/<name>_ts.h5``. 0: no
       snapshots; the buffer is written every 5000 steps and at the end.
   * - ``OutputStrideWFS``
     - int
     - 100
     - Wave-function snapshot interval (steps); 0 disables the snapshots.
   * - ``HDF5Compress``
     - bool
     - 0
     - Compress the HDF5 datasets.
   * - ``HDF5CompressLevel``
     - int
     - 6
     - Compression level, 0 to 9.
   * - ``SaveDipoleMatrix``
     - bool
     - 0
     - Write the dipole operator :math:`D_x` to ``static/<name>_Dx.bin``
       (PETSc binary) when ``SaveDipoleAxes`` is empty.
   * - ``SaveDipoleAxes``
     - string
     - --
     - Dipole operators to write to ``static/<name>_D<a>.bin``: any
       combination of ``x``, ``y``, ``z``, or ``all``, or ``none``.
       Overridden by ``-save_dipole``. Only assembled operators are written.

t-SURFF photoelectron spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used only with ``TSurff = 1``; see :doc:`TSurff`.

.. list-table::
   :widths: 27 10 13 50
   :header-rows: 1

   * - Key (aliases)
     - Type
     - Default
     - Description
   * - ``TSurff``
     - bool
     - 0
     - Compute the t-SURFF momentum spectrum (requires
       ``EnablePropagation = 1``); output ``td/<name>_surff.h5``.
   * - ``SurffRadius``
     - real
     - 0
     - Half width :math:`R` of the surface :math:`|x_d| = R`; > 0, inside every
       axis and in front of the CAP layers.
   * - ``SurffKmax``
     - list[3]
     - 0
     - Upper end of the momentum grid per axis.
   * - ``SurffKmin``
     - list[3]
     - ``-SurffKmax``
     - Lower end of the momentum grid per axis.
   * - ``SurffNk``
     - list[3]
     - 0
     - Momentum grid points per axis, at least 1.
   * - ``SurffMomenta``
     - string
     - --
     - File of momenta (one ``kx [ky [kz]]`` row each) used instead of the
       grid.
   * - ``OutputStrideSurff``
     - int
     - 1
     - Flux sampling interval (steps), at least 1.
   * - ``SurffQuadrature``
     - int
     - 0
     - Quadrature points per element on the surface, 0 to 64; 0 uses
       ``NQuadratures``.
   * - ``SurffMassTolerance``
     - real
     - 1e-6
     - Allowed relative mass variation outside the surface.
   * - ``SurffPotentialTolerance``
     - real
     - 1e-6
     - Allowed potential variation outside the surface (Hartree).
   * - ``SurffInterior``
     - bool
     - 0
     - Add the projection of :math:`\psi(T)` inside the surface onto the
       Volkov states (free-carrier benchmarks; keep 0 for bound systems).
   * - ``SurffEndTerm``
     - bool
     - 0
     - Add the flux after ``FinalTime`` from the field-free resolvent (CAP as
       outgoing condition): the infinite-time spectrum for any ``FinalTime``
       after the pulse. Needs ``EnableCAP = 1``; see :doc:`TSurff`.
   * - ``SurffEndTermDk``
     - real
     - 0
     - Momentum spacing of the end-term energy nodes (0 = automatic).

Example
-------

A minimal 1D high-harmonic run (soft-core atom, :math:`\sin^2` pulse,
absorbing boundaries); a complete deck is ``examples/hhg/hhg_1d.inp``::

    # 1D model atom, HHG with absorbing boundaries
    Dimension     = 1
    Domain        = -200, 200
    Nelements     = 800
    SplineDegree  = 5
    KnotSequence  = uniform

    Variables     = a=2.0, T=1763.7
    Potential     = -1/sqrt(x^2 + a)
    PotentialDerivativeX = x/(x^2 + a)^1.5
    Mass          = 1.0

    NBoundStates     = 1
    TargetEigenvalue = -0.5

    Polarization  = x
    Ampx          = 0.0534
    Omegax        = 0.057
    LaserX        = Ampx*sin(pi*t/T)^2*sin(Omegax*t)

    TimeStep      = 0.05
    FinalTime     = 1763.7
    Propagator    = pade22
    EnableCAP     = 1
    CAPWidth      = 50
