time-propagation
================

Time Propagation
================

Time step, final time, and propagation control.

Time Parameters
---------------

.. list-table::
   :widths: 40 15 10 35
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - EnablePropagation
     - bool
     - 1
     - Whether to run time propagation. Set to 0 for TISE-only
   * - TimeStep / TimeStepSize
     - real
     - 0.01
     - Time step dt in atomic units
   * - FinalTime / TotalTime
     - real
     - 100.0
     - Total propagation time in atomic units

Time Stepper
------------

TDSE-Z uses the PETSc TS framework with **Crank-Nicolson (theta = 0.5)** as the
implicit time-stepper. The method is unitary (norm-preserving) and second-order
accurate.

The residual at each step is evaluated as:

::

    F = i M psidot - H psi - E(t) * D psi
