Input File
==========

Plain-text configuration format with all parameter categories.

Format
------

The input file is plain text with ``key = value`` lines. Comments start with ``#``.
Unknown keys cause a **fatal error** by default (``StrictInput=1``).

muParser
--------

All spatial and temporal expressions are evaluated by **muParser**. The following
variables and functions are always available:

- ``x``, ``y``, ``z`` — spatial coordinates (atomic units)
- ``t`` — time (atomic units)
- ``abs(x)``, ``sin(x)``, ``cos(x)``, ``tan(x)``, ``exp(x)``, ``log(x)``, ``sqrt(x)``, ``pow(x,y)``, ``min(x,y)``, ``max(x,y)``
- ``fabs(x)`` is automatically converted to ``abs(x)``
- ``pi`` — mathematical constant :math:`\pi = 3.14159265...`
- Every scalar input parameter (e.g. ``Amplitude``, ``Omega``, ``LMin``, ``LMax``)
  is available as a muParser constant so you can reference it by name inside any
  expression. For example, ``Potential = V0*exp(-w*x^2)`` with
  ``Variables = V0=1.2, w=0.5`` works without any extra definition.
- Multiple ``Variables`` lines accumulate (not replace).

Custom constants via ``Variables = Name=number, ...`` are added to **every**
muParser instance (potential, mass, laser, envelope, ...), so they can be reused
across the entire file.

File Structure
--------------

::

    # Comments start with # and extend to end of line
    Key1 = value1       # Simple key-value
    Key2 = "quoted string with spaces"
    Key3 = expression    # muParser expression
    # Multiple Variables lines accumulate
    Variables = a=1.0, b=2.0
    Variables = c=3.0    # adds c, does NOT remove a and b

Parameter Reference
===================

All parameters are listed below grouped by their function. Aliases are shown in
parentheses where applicable — all aliases are equivalent to the canonical name.

Global Settings
---------------

.. list-table:: Global Settings
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - Dimension
     - int
     - 1
     - Spatial dimension: 1, 2, or 3.
   * - SplineDegree
     - int
     - 3
     - B-spline degree (Bezier degree). Range: 1 to 14.
   * - Nelements
     - int
     - 100
     - Number of elements (knot spans) per axis. Aliases: NSplines, NumberOfSplines, ElementCount, NumberOfElements.
   * - NQuadratures
     - int
     - 8
     - Gauss-Legendre quadrature points per element.
   * - StrictInput
     - bool
     - 1 (true)
     - 1 = abort on unknown key. 0 = warn and continue. Always keep 1.
   * - Verbose
     - bool
     - 1 (true)
     - 1 = print solver progress to stdout. 0 = minimal output.
   * - Hbar / Planck
     - real
     - 1.0
     - Reduced Planck constant. Aliases: Hbar, Planck. muParser constant.
   * - Charge / ParticleCharge
     - real
     - 1.0
     - Particle charge. Aliases: Charge, ParticleCharge. muParser constant.

Domain and Grid
---------------

.. list-table:: Domain and Grid
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - Domain
     - string
     - (none)
     - Compact domain spec. Two forms: "LMin,LMax" for uniform bounds, or "[LMin,LMax],[LMinY,LMaxY],[LMinZ,LMaxZ]" for per-axis bounds.
   * - LMin / LMax
     - real
     - -10, 10
     - Default domain bounds (all axes). See Domain for per-axis spec.
   * - LMinX, LMaxX / LMinY, LMaxY / LMinZ, LMaxZ
     - real
     - -10, 10
     - Per-axis domain bounds. Override LMin/LMax.
   * - OffsetX, OffsetY, OffsetZ
     - real
     - 0.0
     - Spatial shift of grid (basis function centers only, not physical domain).

Potential and Mass
------------------

.. list-table:: Potential and Mass
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - Potential / ExternalPotential
     - string
     - (none)
     - Potential energy V(x,y,z). muParser expression in x, y, z.
   * - PotentialDerivativeX / PotentialDerivativeY / PotentialDerivativeZ
     - string
     - (none)
     - Analytical gradient components. If omitted, auto-diff used via muParser. Provide for exact gradients or higher performance.
   * - Mass
     - string
     - "1.0"
     - Position-dependent mass m(x,y,z). Must be strictly positive everywhere in the domain. muParser expression in x, y, z.
   * - MassIsConstant
     - bool
     - false
     - 1 = mass is scalar constant 1.0 (skip spatial eval). Use when Mass = "1.0" to avoid unnecessary function calls.
   * - dinvMassX / dinvMassY / dinvMassZ
     - string
     - "0.0"
     - d(1/m)/dx, d(1/m)/dy, d(1/m)/dz. Required for BenDaniel-Duke kinetic operator when mass varies. Use "0.0" for auto-diff.

Laser Field
-----------

.. list-table:: Laser Field
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - Laser / Driver
     - string
     - (none)
     - Single-field expression E(t). muParser expression in t. All time-dependent scalars are available as constants.
   * - LaserX / DriverX / LaserY / DriverY / LaserZ / DriverZ
     - string
     - (none)
     - Per-component laser expressions Ex(t), Ey(t), Ez(t). If any set, the others default to empty string.
   * - Polarization / LaserPolarization / DriverPolarization
     - string
     - (none)
     - Field polarization type. Accepted values: "x"=linear-x, "y"=linear-y, "z"=linear-z, "xy"=two-color linear, "xz", "yz", "xyz"=three-color linear, "right"/"left"=circular polarization (right/left).
   * - Amplitude
     - real
     - 0.0
     - Scalar field amplitude (legacy single-field form). muParser constant.
   * - Omega
     - real
     - 0.056954
     - Carrier frequency (a.u.). 0.056954 a.u. = 800 nm. muParser constant.
   * - Phase
     - real
     - 0.0
     - Carrier-envelope phase (radians). muParser constant.
   * - PulseDuration
     - real
     - 0.0
     - Envelope pulse duration (atomic units).
   * - PulseCenter
     - real
     - 0.0
     - Envelope pulse center time (atomic units).
   * - EnvelopeType
     - string
     - "custom"
     - Envelope type. Accepted values: "custom", "gaussian", "sech", "top-hat".
   * - Envelope
     - string
     - (none)
     - User-defined envelope F(t) expression in t. Required when EnvelopeType = custom. E.g. sin(pi*t/PulseDuration)^2 or exp(-4*log(2)*(t-PulseCenter)^2/PulseDuration^2).
   * - Ampx, Ampy, Ampz
     - real
     - 0.0
     - Per-component field amplitudes.
   * - CEPx, CEPy, CEPz
     - real
     - 0.0
     - Per-component carrier-envelope phases (radians).
   * - Omegax, Omegay, Omegaz
     - real
     - 0.0
     - Per-component carrier frequencies (a.u.).
   * - DriverAmplitude / LaserAmplitude
     - string
     - (none)
     - Per-axis: "Ampx,Ampy,Ampz" (comma-separated). Alias: LaserAmplitude.
   * - DriverFrequency / LaserFrequency
     - string
     - (none)
     - Per-axis: "Omegax,Omegay,Omegaz". Alias: LaserFrequency.
   * - DriverCEP / CarrierEnvelopePhase
     - string
     - (none)
     - Per-axis: "CEPx,CEPy,CEPz". Alias: CarrierEnvelopePhase.

Eigensolver (TISE)
------------------

.. list-table:: Eigensolver (TISE)
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - UseDirectSolve / DirectSolve
     - bool
     - 0 (false)
     - 1 = use SLEPc direct eigensolver (ARNOLDI). 0 = implicitly restarted Lanczos (IRLM).
   * - TargetEigenvalue / TargetEnergy
     - real
     - 0.0
     - Target eigenvalue for shifted-invert mode. Alias: TargetEnergy.
   * - NBoundStates / NumberOfBoundStates
     - int
     - 1
     - Number of bound states to compute. Must be >= 1.
   * - NBoundStatesSave / SaveBoundStates
     - bool
     - 0 (false)
     - 1 = save computed eigenstates to HDF5.
   * - BoundStateFormat / SaveStateFormat / StateFormat
     - string
     - "complex"
     - Output format for eigenstates: "complex" (default) or "ascii".

Time Propagation
----------------

.. list-table:: Time Propagation
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - EnablePropagation
     - bool
     - 1 (true)
     - 1 = run time propagation after eigensolve. 0 = eigensolver only.
   * - TimeStep / TimeStepSize
     - real
     - 0.01
     - Time step in atomic units. Alias: TimeStepSize. Must be > 0.
   * - FinalTime / TotalTime
     - real
     - 100.0
     - Total propagation time in atomic units. Alias: TotalTime. Must be > 0.

Output Control
--------------

.. list-table:: Output Control
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - PhysicsOutput / OutputQuantities
     - string
     - "all"
     - Declarative enable list: comma-separated from: "dipole", "population", "energy", "current", "autocorrelation"/"ac", "wfs", "tsurff". "all" enables everything.
   * - OutputStrideTS
     - int
     - 0
     - HDF5 output stride for time-series observables. Default 0 = flush every 5000 steps or when ring buffer fills.
   * - OutputStrideAC
     - int
     - 0
     - Autocorrelation flush stride. Same behavior as OutputStrideTS.
   * - OutputStrideWFS
     - int
     - 100
     - Wavefunction snapshot save stride.
   * - HDF5Compress
     - bool
     - 0 (false)
     - 1 = compress HDF5 datasets (gzip).
   * - HDF5CompressLevel
     - int
     - 6
     - Gzip compression level: 0 (none) to 9 (max).
   * - SaveDipoleMatrix
     - bool
     - 0 (false)
     - 1 = write assembled dipole operator Dx to PETSc binary file "static/Dx_*.bin".
   * - SaveDipoleAxes
     - string
     - (none)
     - Which dipole axes to save: x, xy, xz, xyz, all, none. CLI flag -w/-all/-none overrides this.

Initial State
-------------

.. list-table:: Initial State
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - InitialState / StartState
     - string
     - "ground"
     - Initial state selector. Three formats: "ground" for 0th bound state, "state:N" for Nth state (0-based), "sup: a*N + b*M [+ c*P]" for coherent superposition of up to 3 bound states with real coefficients.
   * - NormalizeInitialState / NormalizePsi0
     - bool
     - 1 (true)
     - 1 = normalize initial state so \<psi_0|M|psi_0\> = 1.

Boundary Conditions and CAP
---------------------------

.. list-table:: Boundary Conditions and CAP
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - BoundaryType / Boundary
     - string
     - "Neumann"
     - Boundary condition type. Accepted values: "Neumann"/"Natural" for reflecting wall, "Dirichlet"/"Wall" for psi=0 at boundary.
   * - EnableCAP / AbsorbingBoundary
     - bool
     - 0 (false)
     - 1 = enable complex absorbing potential for outgoing-wave boundary conditions.
   * - CAPKmin / CapStrength
     - real
     - 0.1
     - Inner edge of CAP region (a.u.). Determines where absorption begins. Higher = more absorbing.
   * - Gamma
     - real
     - 5.0
     - CAP strength parameter gamma. Controls absorption intensity. Higher = stronger absorption but more numerical reflection at CAP edge.

Knot Sequences
--------------

.. list-table:: Knot Sequences
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - KnotSequence
     - string
     - "uniform"
     - Default knot sequence for all axes. See "Knot Sequences" page for details.
   * - KnotAlpha
     - real
     - 0.0
     - Primary tuning parameter for symexp and tangent-based sequences. Controls compression rate near origin.
   * - HydrogenicNLin
     - int[3]
     - 0,0,0
     - Number of linear segments near origin for hydrogenic knot sequence. Per-axis (comma-separated) or single value (replicated).
   * - HydrogenicNExp
     - int[3]
     - 0,0,0
     - Number of exponential segments for hydrogenic knot sequence. Per-axis.
   * - HydrogenicR1
     - real[3]
     - 0.0,0,0
     - Transition radius (a.u.) between linear and exponential regions of hydrogenic knot sequence. Per-axis.
   * - AdaptiveKappa
     - real
     - 1.0
     - Reserved for adaptive knots; currently ignored while that mode uses uniform knots.
   * - AdaptivePower
     - real
     - 1.0
     - Reserved for adaptive knots; currently ignored.
   * - AdaptiveWFCoarseN
     - int
     - 20
     - Reserved for the adaptive_wf coarse solve; currently ignored.
   * - AdaptiveWFKinLambda
     - real
     - 3.0
     - Reserved for adaptive_wf density weighting; currently ignored.

t-SURFF (Surface Flux)
----------------------

.. list-table:: t-SURFF (Surface Flux)
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - TSurff
     - bool
     - 0 (false)
     - 1 = enable t-SURFF momentum spectrum calculation. Requires PhysicsOutput include "tsurff".
   * - SurffNk
     - int
     - 40
     - Momentum grid points per axis in [-SurffKmax, +SurffKmax].
   * - SurffKmax
     - real
     - 2.0
     - Momentum cutoff in atomic units.
   * - OutputStrideSurff
     - int
     - 1
     - Save stride for t-SURFF flux accumulation.
   * - SurffCouplingSign
     - real
     - +1.0
     - Sign of field-momentum coupling: k_eff = k + s * q A(t). Flip to -1.0 if your field couples as +q E * r instead of -q E * r.
   * - SurffRadius
     - real
     - 0.0
     - If \> 0, restrict t-SURFF integration surfaces to shell r \> SurffRadius (a.u.). Suppresses near-field contributions.

Lz Diagonalization
------------------

.. list-table:: Lz Diagonalization
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - EnableLzDiag / LzDiag
     - bool
     - 0 (false)
     - 1 = compute Lz expectation values for all bound states during eigensolve.
