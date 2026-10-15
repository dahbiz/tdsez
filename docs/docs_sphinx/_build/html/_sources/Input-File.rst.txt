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
- Every scalar input parameter (e.g. ``Ampx``, ``Omegax``, ``LMinX``, ``TimeStep``)
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
   * - Charge / ParticleCharge
     - real
     - -1.0
     - Carrier charge q in units of e: -1 electron (default), +1 hole. H = K - q (V + r·E): Potential and field are written for an electron and inverted for q > 0 (a notice is printed). Outputs are q⟨r⟩, q⟨v⟩, q d⟨v⟩/dt. Aliases: Charge, ParticleCharge. muParser constant.
   * - ObserveAllAxes
     - bool
     - 1
     - 1 = dipole, current and acceleration on every spatial axis (cross-polarized response); 0 = only on the polarised axes (fewer operators, less memory).

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
   * - LMinX, LMaxX / LMinY, LMaxY / LMinZ, LMaxZ
     - real
     - -10, 10
     - Per-axis domain bounds (alternative to Domain).
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
     - Gradient of Potential along x, y, z (used for the dipole acceleration). Required on every axis of the run when EnablePropagation = 1 (0 for a constant potential).
   * - Mass
     - string
     - "1.0"
     - Position-dependent mass m(x,y,z). Must be strictly positive everywhere in the domain. muParser expression in x, y, z.
   * - MassIsConstant
     - bool
     - auto
     - 1 when Mass has no x, y, z (detected automatically): fast constant-mass assembly. Setting 1 for a varying mass drops the mass-gradient corrections.

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
     - Field E(t) for every polarised axis without its own LaserX/Y/Z. muParser expression in t, envelope included; all input scalars are available as constants.
   * - LaserX / DriverX / LaserY / DriverY / LaserZ / DriverZ
     - string
     - (none)
     - Field on each polarised axis, Ex(t), Ey(t), Ez(t), envelope included. Every axis in Polarization needs one (or Laser).
   * - Polarization / LaserPolarization / DriverPolarization
     - string
     - (none)
     - Field polarization type. Accepted values: "x"=linear-x, "y"=linear-y, "z"=linear-z, "xy", "xz", "yz", "xyz" (also "all") = several coupled axes, each with its own field (elliptical or circular fields are written through LaserX/Y).
   * - Ampx, Ampy, Ampz
     - real
     - 0.0
     - Per-component field amplitudes.
   * - CEPx, CEPy, CEPz
     - real
     - 0.0
     - Per-component carrier-envelope phases (radians).
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
     - Number of bound states to compute. 0 skips the eigensolver; allowed only with an analytic InitialState (wavepacket, analytic:quesne01, analytic:massstep01).
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
   * - Propagator
     - string
     - "cn"
     - Time stepper. "cn": Crank-Nicolson, Pade(1,1), second order, one sparse solve per step. "pade22": exponential midpoint rule with a Pade(2,2) exponential (fourth-order field-free phase, second order in the field variation, two solves per step). "magnus4": commutator-free fourth-order Magnus with Pade(2,2) exponentials (fourth order, four solves per step). The rational propagators are unitary for Hermitian H and allow much larger TimeStep at equal accuracy.
   * - Preconditioner
     - string
     - "asm"
     - Preconditioner of the propagator solves: "asm" (additive Schwarz, ILU(0) subdomains) or "mg" (geometric B-spline multigrid with exact knot-insertion prolongation). Same result, different cost; see Time Propagation.
   * - MGLevels
     - int
     - 0
     - Multigrid levels (0 = coarsen until about 2000 DOFs; otherwise >= 2).
   * - MGSmooth
     - int
     - 1
     - Richardson sweeps with ASM/ILU(0) before and after each coarse correction.

t-SURFF Photoelectron Spectra
-----------------------------

Computed only with ``TSurff = 1``; written to ``td/surff_<input>.h5``. See
:doc:`TSurff`.

.. list-table:: t-SURFF
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - TSurff
     - bool
     - 0
     - 1 = compute the t-SURFF momentum spectrum (needs EnablePropagation = 1).
   * - SurffRadius
     - real
     - (none)
     - Surface ``|x_d| = R``; inside every axis and in front of the CAP layer.
   * - SurffKmax / SurffKmin
     - real[3]
     - (none) / -SurffKmax
     - Momentum grid limits per axis.
   * - SurffNk
     - int[3]
     - (none)
     - Grid points per axis.
   * - SurffMomenta
     - string
     - (none)
     - File of momenta (rows "kx [ky kz]") instead of the grid.
   * - OutputStrideSurff
     - int
     - 1
     - Flux sampling stride in steps (the final time is always sampled).
   * - SurffQuadrature
     - int
     - 0
     - Gauss points per knot span on the surface (0 = NQuadratures).
   * - SurffInterior
     - bool
     - 0
     - 1 = add the projection of psi(FinalTime) inside the box (free-particle benchmarks; keep 0 for bound systems).
   * - SurffMassTolerance / SurffPotentialTolerance
     - real
     - 1e-6 / 1e-6
     - Allowed relative mass / absolute potential (Ha) variation outside the surface.

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
     - Declarative enable list: comma-separated from: "dipole", "population", "energy", "current", "autocorrelation"/"ac", "wfs". "all" enables everything.
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
     - Initial state selector. Three formats: "ground" for 0th bound state, "state:N" for Nth state (0-based), "sup: a*N + b*M [+ c*P]" for coherent superposition of up to 3 bound states, "wavepacket" for a Gaussian wave packet (WavepacketCenter, WavepacketMomentum, WavepacketWidth; see Initial State).
   * - NormalizeInitialState / NormalizePsi0
     - bool
     - 1 (true)
     - 1 = normalize initial state so \<psi_0|M|psi_0\> = 1.
   * - AccelerationMethod
     - string
     - "auto"
     - Dipole acceleration a(t). "operator": Ehrenfest force operator (smooth m and V only). "discrete": a = d<v>/dt from the Galerkin equation of motion (exact at sharp interfaces and hard walls; one M solve per output step). "auto": "discrete" when Mass depends on x, y or z and MassIsConstant = 0, otherwise "operator".

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
   * - CAPKmin
     - real
     - 0.1
     - Lowest momentum (a.u.) absorbed efficiently by the Manolopoulos CAP. Sets the layer width at each wall, w = 6.555/CAPKmin (65.5 a.u. for 0.1); slower electrons are partly reflected. w must be at most 45% of each axis.

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
     - Grading strength of symexp/symtanu/symtann (default 0.9 when unset) and symlogtan (default 0.5). Aliases: KnotGrading, GradingStrength.
   * - HydrogenicNLin
     - int[3]
     - 0,0,0
     - Number of linear elements per half-axis near the origin for the hydrogenic sequence. Per-axis (comma-separated) or single value (replicated). Aliases: InnerElements, HydrogenicInnerElements.
   * - HydrogenicNExp
     - int[3]
     - 0,0,0
     - Number of exponentially growing elements per half-axis from r_cross to Lmax (hydrogenic). Per-axis. Aliases: OuterElements, HydrogenicOuterElements.
   * - HydrogenicR1
     - real[3]
     - 0.0,0,0
     - Size (a.u.) of the linear near-origin elements (hydrogenic); r_cross = HydrogenicNLin*HydrogenicR1 must be < Lmax. Per-axis. Aliases: InnerSpacing, HydrogenicInnerSpacing.

