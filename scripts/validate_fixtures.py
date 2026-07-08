#!/usr/bin/env python3
"""Validate the committed fixture datasets (Milestone 2).

For every fixture under ``datasets/fixtures/*/`` this checks the *structural*
health of the data — the things that must hold regardless of the defects planted
for the audit to find:

  * every record parses and satisfies the ``DatasetExample`` schema;
  * ``example_id`` values are unique within the fixture;
  * the declared ``task_family`` is consistent and known;
  * the ground-truth ``defects.yaml`` (if present) references only real
    ``example_id`` values.

It does **not** check for the planted defects themselves — those are semantic and
are the job of the audit stages (Milestones 4–6). The richer ``validate`` CLI
(missing-field diagnostics, PII scanning, redaction) lands in Milestone 3.

Runs fully offline on the committed fixtures. Exit code 0 iff every fixture is
structurally valid.

Usage:
    python3 scripts/validate_fixtures.py [--fixtures-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eval_dataset_forge.datasets import loader  # noqa: E402
from eval_dataset_forge.schemas import KNOWN_TASK_FAMILIES  # noqa: E402


def collect_example_id_refs(node: Any) -> set[str]:
    """Recursively collect every ``example_id`` referenced in a defects tree.

    References appear as ``ids: [...]`` lists, bare ``id:``/``example:`` scalars,
    so we walk the structure and pick those up wherever they occur.
    """
    refs: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ids" and isinstance(value, list):
                refs.update(str(v) for v in value)
            elif key in ("id", "example") and isinstance(value, (str, int)):
                refs.add(str(value))
            else:
                refs.update(collect_example_id_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.update(collect_example_id_refs(item))
    return refs


def validate_fixture(fixture_dir: Path) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate one fixture directory.

    Returns ``(ok, problems, stats)``. ``problems`` is a list of human-readable
    error strings; ``stats`` carries per-fixture counts for the summary line.
    """
    problems: list[str] = []
    stats: dict[str, Any] = {"examples": 0, "families": [], "defect_refs": 0}

    examples_path = fixture_dir / "examples.jsonl"
    try:
        examples = loader.load_examples(examples_path)
    except loader.FixtureError as exc:
        return False, [str(exc)], stats

    stats["examples"] = len(examples)
    if not examples:
        problems.append(f"{examples_path}: no examples")

    # ID uniqueness.
    seen: set[str] = set()
    dupes: set[str] = set()
    for ex in examples:
        if ex.example_id in seen:
            dupes.add(ex.example_id)
        seen.add(ex.example_id)
    if dupes:
        problems.append(f"duplicate example_id(s): {sorted(dupes)}")

    # task_family consistency + known-vocabulary check.
    families = sorted({ex.task_family for ex in examples})
    stats["families"] = families
    if len(families) > 1:
        problems.append(f"mixed task_family in one fixture: {families}")
    unknown = [f for f in families if f not in KNOWN_TASK_FAMILIES]
    if unknown:
        problems.append(f"unknown task_family (not in KNOWN_TASK_FAMILIES): {unknown}")

    # Ground-truth defects reference only real example ids.
    defects_path = fixture_dir / "defects.yaml"
    if defects_path.is_file():
        defects = loader.load_defects(defects_path)
        refs = collect_example_id_refs(defects.get("planted_defects", {}))
        stats["defect_refs"] = len(refs)
        missing = sorted(r for r in refs if r not in seen)
        if missing:
            problems.append(f"defects.yaml references unknown example_id(s): {missing}")
        declared = defects.get("example_count")
        if isinstance(declared, int) and declared != len(examples):
            problems.append(
                f"defects.yaml example_count ({declared}) != actual ({len(examples)})"
            )
    else:
        problems.append(f"missing {defects_path.name}")

    return not problems, problems, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate committed fixture datasets.")
    parser.add_argument(
        "--fixtures-dir",
        default=str(loader.FIXTURES_ROOT),
        help="directory containing fixture subdirectories (default: datasets/fixtures)",
    )
    args = parser.parse_args(argv)

    fixtures = loader.discover_fixtures(args.fixtures_dir)
    if not fixtures:
        print(f"no fixtures found under {args.fixtures_dir}", file=sys.stderr)
        return 1

    print(f"validating {len(fixtures)} fixture(s) under {args.fixtures_dir}\n")
    all_ok = True
    for fixture_dir in fixtures:
        ok, problems, stats = validate_fixture(fixture_dir)
        mark = "ok" if ok else "FAIL"
        fam = ",".join(stats["families"]) or "-"
        print(
            f"  [{mark:>4}] {fixture_dir.name:<18} "
            f"examples={stats['examples']:<3} family={fam:<11} "
            f"defect_refs={stats['defect_refs']}"
        )
        for problem in problems:
            print(f"         - {problem}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("All fixtures are structurally valid (schema + unique ids + ground-truth refs).")
        return 0
    print("Some fixtures failed validation.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
