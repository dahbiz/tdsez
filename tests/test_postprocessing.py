"""Post-processing tools (dipoles.x, dipole_spectrum.x, recombination.x) in 1D,
2D and 3D against exact transition dipoles (zkit.analytic, zkit validate).

* oscillators with distinct frequencies per axis: <n|x_a|n+1> = sqrt((n+1)/2w_a);
* hard-wall box: <n|x|m> = -8 L n m / (pi^2 (n^2 - m^2)^2) for n + m odd;
* the rows of dipole_spectrum.x and recombination.x (regular term) equal the
  full matrix of dipoles.x on every axis.
"""

import pytest

import zkit
from tests.test_tdsez import _step
from zkit.validation import TDM_CASES, Validator, transition_dipoles


@pytest.mark.parametrize("case", list(TDM_CASES))
def test_transition_dipoles_against_exact_results(case, tmp_path):
    v = Validator(zkit.find_binary(), nproc=2, workdir=tmp_path)
    transition_dipoles(v, case)
    for c in v.report.checks:
        _step(f"{c.case}: {c.quantity}: {c.error:.1e} <= {c.tol:g}", c.ok)
