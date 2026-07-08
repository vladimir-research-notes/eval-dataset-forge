"""Pytest path shim.

The package uses a ``src/`` layout. This makes the suite runnable without an
editable install (``pip install -e .``) by putting ``src/`` on ``sys.path`` for
the test session. Real deployments should still install the package, but this
keeps CPU-only tests frictionless on a fresh checkout.
"""

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
