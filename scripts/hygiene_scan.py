#!/usr/bin/env python3
"""Scan the tracked tree for do-not-commit contract violations (Milestone 15).

Checks that no file git tracks is one ``.gitignore`` would ignore, that no tracked path is a
forbidden secret/key/weight file, and that no tracked file is dataset-sized. Content secret-scanning
is deliberately *not* done here — that lives in :mod:`eval_dataset_forge.validation.pii`, which is
tested against fixtures that contain fake secrets on purpose. See the module docstring.

Offline, no network. Exits 1 when the tracked tree violates the contract.

    python3 scripts/hygiene_scan.py
    python3 scripts/hygiene_scan.py --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_dataset_forge.hygiene import format_report, scan_repository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    report = scan_repository(ROOT)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
