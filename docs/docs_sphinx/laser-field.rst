laser-field
===========

Laser Field
===========

Polarization, envelopes, CEP, and field expressions.

Polarization
------------

.. list-table::
   :widths: 40 15 10 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Polarization / LaserPolarization / DriverPolarization
     - string
     - --
     - Polarization axes: ``x``, ``y``, ``z``, ``xy``, ``xz``, ``yz``, ``xyz``

Field Expression
----------------

The field E(t) is specified via muParser expressions. Two modes:

- **Scalar form**: ``Laser = Amplitude*cos(Omega*t)*sin(pi*t/PulseDuration)^2``
  -- single envelope for all axes
- **Per-axis form**: ``LaserX = ...``, ``LaserY = ...``, ``LaserZ = ...``
  -- overrides scalar form when set

.. list-table::
   :widths: 35 15 15 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - Laser
     - muParser
     - 0.0
     - Unified field E(t) for scalar form
   * - LaserX / LaserY / LaserZ / DriverX / DriverY / DriverZ
     - muParser
     - 0.0
     - Per-axis field E_i(t). Overrides scalar Laser when set
   * - Amplitude
     - real
     - 0.0
     - Peak field strength (scalar)
   * - Omega / Omegax, Omegay, Omegaz
     - real
     - 0.056954
     - Carrier frequency (omega=0.056954 approx 800 nm)
   * - Phase / CEPx, CEPy, CEPz
     - real
     - 0.0
     - Carrier-envelope phase (radians)
   * - Ampx, Ampy, Ampz
     - real
     - 0.0
     - Per-axis peak amplitude
   * - PulseDuration
     - real
     - 0.0
     - Pulse FWHM duration in a.u.
   * - PulseCenter
     - real
     - 0.0
     - Pulse center time

Envelope
--------

.. list-table::
   :widths: 25 15 15 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - EnvelopeType
     - string
     - ``custom``
     - Pre-built envelope form or custom expression
   * - Envelope
     - muParser
     - --
     - Custom envelope F(t) expression
