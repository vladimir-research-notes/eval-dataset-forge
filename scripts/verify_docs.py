#!/usr/bin/env python3
"""Audit the repository's documentation against the repository (Milestone 14).

Checks that every documented CLI command names a real subcommand and real options, that every
relative link and referenced script resolves, that every cited test count is the number pytest
collects, that every cited fixture example id exists, that no live document asserts a claim a
milestone measured false, and that every number in ``docs/CLAIMS.md`` is reproduced by the command
the ledger says produces it.

Offline, CPU-only, no network. Exits 1 when any claim is unsupported.

    python3 scripts/verify_docs.py
    python3 scripts/verify_docs.py --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_dataset_forge.docs_audit import audit_docs, format_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    report = audit_docs(ROOT)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
