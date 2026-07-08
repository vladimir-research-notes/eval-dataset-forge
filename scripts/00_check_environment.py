#!/usr/bin/env python3
"""Environment and dependency check for eval-dataset-forge.

CPU-only, offline, no side effects. Reports the interpreter version and whether
the core and optional dependencies are importable. Exits non-zero if a *core*
dependency is missing, and prints the exact install command for this PEP-668
(externally-managed) environment.

Usage:
    python3 scripts/00_check_environment.py
"""

from __future__ import annotations

import importlib.util
import platform
import sys

CORE = [
    ("pydantic", "schema validation"),
    ("numpy", "bootstrap statistics"),
    ("yaml", "YAML dataset/config formats (pyyaml)"),
]
OPTIONAL = [
    ("pandas", "tabular report helpers"),
    ("sklearn", "TF-IDF near-duplicate detection (scikit-learn)"),
]

MIN_PYTHON = (3, 12)


def _present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def main() -> int:
    print("eval-dataset-forge — environment check")
    print("-" * 44)
    print(f"python            : {platform.python_version()}")
    print(f"platform          : {platform.system()} {platform.machine()}")

    py_ok = sys.version_info[:2] >= MIN_PYTHON
    if not py_ok:
        print(f"WARNING: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ recommended.")

    print("\ncore dependencies (required):")
    core_ok = True
    for mod, why in CORE:
        ok = _present(mod)
        core_ok = core_ok and ok
        print(f"  [{'ok' if ok else '--'}] {mod:10s} — {why}")

    print("\noptional dependencies (graceful pure-Python fallback if absent):")
    for mod, why in OPTIONAL:
        ok = _present(mod)
        print(f"  [{'ok' if ok else '--'}] {mod:10s} — {why}")

    if not core_ok:
        print("\nMissing core dependencies. Install (PEP-668 environment):")
        print("  python3 -m pip install --user --break-system-packages "
              "pydantic numpy pyyaml")
        print("Optional accelerators:")
        print("  python3 -m pip install --user --break-system-packages "
              "pandas scikit-learn")
        return 1

    print("\nEnvironment OK: core dependencies present. Runs CPU-only, offline, "
          "no paid API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
