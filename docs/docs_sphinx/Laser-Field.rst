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

The field is exactly the muParser expression you give, as a function of ``t``
(atomic units); TDSE-Z multiplies nothing into it. Write the pulse envelope
inside the expression.

- **Per-axis form**: ``LaserX = ...``, ``LaserY = ...``, ``LaserZ = ...`` — the field
  on each polarised axis.
- **Scalar form**: ``Laser = ...`` — used for every polarised axis that has no
  ``LaserX/Y/Z`` of its own (with ``Polarization = xy`` it drives x and y with the
  same field).

Every axis listed in ``Polarization`` must have a field (per-axis or ``Laser``);
the input is rejected otherwise. Only these axes are coupled to the field; the
response (dipole, current, acceleration) is recorded on every spatial axis
(``ObserveAllAxes = 0`` restricts it to the coupled axes).

Example: a 4-cycle :math:`\sin^2` pulse at 800 nm,

::

    Polarization = x
    Ampx   = 0.0534
    Omegax = 0.057
    LaserX = Ampx*sin(Omegax*t/8)^2*cos(Omegax*t + CEPx)

The parameters below are not used by the code directly; they are named
constants available inside all expressions:

.. list-table::
   :widths: 35 15 15 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Meaning (by convention)
   * - Ampx/y/z, CEPx/y/z, Omegax/y/z
     - real
     - 0.0
     - Per-axis amplitude, CEP, frequency (``Omegax`` defaults to 0: set it)
   * - LaserAmplitude, LaserFrequency, CarrierEnvelopePhase
     - list
     - --
     - Comma-separated per-axis values for Ampx/y/z, Omegax/y/z, CEPx/y/z

Further constants (pulse length, centre, ...) are defined with
``Variables = T=440, t0=220``.

Removed keys: ``Envelope`` and ``EnvelopeType`` (never applied), and the legacy
scalar ``Amplitude``, ``Omega``, ``Phase``, ``PulseDuration`` and ``PulseCenter``
(use the per-axis constants or ``Variables``). ``EnvelopeType = custom`` in old
inputs only prints a warning; the other removed keys stop the run with a message
naming the replacement.
