"""Pytest configuration for the TDSEZ test suite.

Overrides zkit.find_binary() to locate the tdsez executable inside this
build tree (build/ or build-split/) instead of the parent repo's default
search path.

The zkit package is an external pip-installable dependency.  If it is
not importable, a clear actionable error is emitted instead of a cryptic
ModuleNotFoundError traceback.
"""

import os
import glob
import sys
import pytest

_FUSED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fused_find_binary(prefer="tdsez"):
    """Locate the built tdsez executable inside this build tree.

    Searches build/ then build-split/ so the tests always exercise
    the local build, not the parent repo's binary.

    Args:
        prefer: Executable name to search for.

    Returns:
        Path to the first matching executable found.

    Raises:
        FileNotFoundError: If no build directory contains the binary.
    """
    explicit = os.environ.get("TDSEZ_BINARY")
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        raise FileNotFoundError(f"TDSEZ_BINARY is not an executable file: {explicit}")

    for d in ("build", "build-split"):
        cands = sorted(glob.glob(os.path.join(_FUSED_ROOT, d, prefer)))
        if cands:
            return cands[0]
    raise FileNotFoundError(
        f"could not find '{prefer}' binary. Build it first:\n"
        f"  cd {_FUSED_ROOT} && mkdir -p build && cd build && \\"
        f"  cmake .. -DPETSC_DIR=$PETSC_DIR -DPETSC_ARCH=$PETSC_ARCH && make -j"
    )


def pytest_configure(config):
    """Verify zkit is importable and redirect its binary discovery.

    Args:
        config: pytest Config object (unused beyond registration).
    """
    try:
        import zkit  # external, pip-installable helper package
    except ImportError:
        sys.stderr.write(
            "\n"
            "+====================================================================+\n"
            "|  ERROR: the 'zkit' package is required to run these tests.        |\n"
            "|                                                                    |\n"
            "|  Install it with pip:                                             |\n"
            "|      pip install zkit-lib                                         |\n"
            "|                                                                    |\n"
            "+====================================================================+\n\n"
        )
        pytest.exit("missing dependency: zkit-lib (pip install zkit-lib)", returncode=2)

    zkit.find_binary = _fused_find_binary
