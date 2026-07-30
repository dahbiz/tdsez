input-file
==========

Input File
==========

Plain-text configuration format with all parameter categories.

Format
------

The input file is plain text with ``key = value`` lines. Comments start with ``#``.
Unknown keys cause a **fatal error** by default (``StrictInput=1``).

Parameter Categories
--------------------

The input file is organized into the following sections. Each is documented on a
dedicated subpage linked from the toctree.

- **Grid & Domain** (3.1) -- Spatial discretization: dimension, bounds, elements,
  splines, quadrature
- **Potential & Mass** (3.2) -- muParser potentials, position-dependent mass,
  physical constants
- **Eigensolver** (3.3) -- SLEPc EPS: bound states, direct solve, save format
- **Time Propagation** (3.4) -- Time step, final time, propagation enable
- **Laser Field** (3.5) -- Polarization, amplitude, frequency, CEP, envelopes
- **Boundary Conditions** (3.6) -- Reflecting, Dirichlet, complex absorbing
  potential (CAP)
- **Knot Sequences** (3.7) -- Uniform, symexp, symmetric, interface, hydrogenic,
  adaptive
- **Output Control** (3.8) -- Stride, observables, compression, verbosity
- **Initial State** (3.9) -- Ground, excited, superposition state selection

Minimal Example
---------------

A minimal 1D soft-Coulomb atom driven by a 3-cycle 800 nm pulse. Two lines of
potential, one line of laser -- that's all TDSE-Z needs to run.

::

    # Domain and basis
    Dimension    = 1
    LMin         = -120.0
    LMax         =  120.0
    NSplines     = 600
    SplineDegree = 5

    # 1D soft-Coulomb atom
    Mass         = 1.0
    Potential    = -1.0 / sqrt(x^2 + 1.0)

    # 3-cycle 800 nm pulse
    Amplitude   = 0.05
    Omega       = 0.056954
    Laser       = Amplitude*cos(Omega*t)*sin(pi*t/331.0)^2

    # Time propagation
    TimeStep    = 0.05
    FinalTime   = 331.0
