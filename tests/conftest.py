# conftest.py — make the inherited test_tdsez.py run against THIS fused build.
#
# zkit.find_binary() resolves relative to the zkit package's parent directory
# (the repo root), so by default it would pick up ../build-split/tdsez (the
# parent modular build) instead of this self-contained snapshot. We override it
# here so the fused-version tests actually exercise fused-version/build/tdsez.
#
# zkit is an external package (pip-installable). If it is not importable we emit
# a clear, actionable error instead of a cryptic ModuleNotFoundError traceback.
import os
import glob
import sys
import pytest

_FUSED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fused_find_binary(prefer="tdsez"):
    """Locate the built tdsez executable inside THIS fused-version tree.

    Searches fused-version/build then fused-version/build-split so the tests
    never accidentally pick up the parent repo's binary.
    """
    for d in ("build", "build-split"):
        cands = sorted(glob.glob(os.path.join(_FUSED_ROOT, d, prefer)))
        if cands:
            return cands[0]
    raise FileNotFoundError(
        f"could not find '{prefer}' binary. Build it first:\n"
        f"  cd {_FUSED_ROOT} && mkdir -p build && cd build && \\\n"
        f"  cmake .. -DPETSC_DIR=$PETSC_DIR -DPETSC_ARCH=$PETSC_ARCH && make -j"
    )


def pytest_configure(config):
    try:
        import zkit  # external, pip-installable helper package
    except ImportError:
        sys.stderr.write(
            "\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  ERROR: the 'zkit' package is required to run these tests.        ║\n"
            "║                                                                      ║\n"
            "║  Install it with pip:                                              ║\n"
            "║      pip install zkit                                              ║\n"
            "║                                                                      ║\n"
            "║  (If you have it locally, point PYTHONPATH at its parent dir, e.g. ║\n"
            "║      PYTHONPATH=/home/zakaria/Documents/hermes/TDSEZ pytest ...)    ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n\n"
        )
        # Clean, non-zero exit — no traceback noise.
        pytest.exit("missing dependency: zkit (pip install zkit)", returncode=2)

    # Redirect zkit's binary discovery to this self-contained build.
    zkit.find_binary = _fused_find_binary

