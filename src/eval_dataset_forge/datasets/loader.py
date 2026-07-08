"""Minimal fixture/dataset loader (Milestone 2).

This is the smallest loader the fixture datasets and their validation script
(:mod:`scripts.validate_fixtures`) need. It reads hand-authored examples
(``.jsonl`` or ``.yaml``) into schema-validated :class:`DatasetExample` objects,
and loads the accompanying ground-truth ``defects.yaml`` files.

Loading here is deliberately *structural*: a record must satisfy the pydantic
schema (required fields, types, unknown-field rejection). The interesting content
of the fixtures — duplicates, leakage, label conflicts — is **semantic** and is
intentionally left for the audit stages to find. The full ``validate`` CLI
(ID-uniqueness reporting, missing-field diagnostics, PII scanning) lands in
Milestone 3 and will build on these helpers rather than replace them.

Importing this module pulls in pydantic (via ``schemas``) and PyYAML. The package
``__init__`` and the CLI deliberately do not import it at top level, so
``evalforge check-env`` keeps working before the scientific stack is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from eval_dataset_forge.schemas import DatasetExample

# Repo layout: this file is ``src/eval_dataset_forge/datasets/loader.py``; the
# committed fixtures live at ``<repo>/datasets/fixtures``. parents[3] is the repo
# root. Callers may override the root explicitly for tests or alternate trees.
_REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_ROOT = _REPO_ROOT / "datasets" / "fixtures"


class FixtureError(ValueError):
    """Raised when a fixture file is malformed or fails schema validation.

    The message carries the file path and 1-based record index so a broken
    fixture is easy to locate; the original ``ValidationError`` is chained.
    """


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read raw example records from a ``.jsonl`` or ``.yaml`` file.

    JSONL: one JSON object per non-empty line (``#`` full-line comments and blank
    lines are skipped). YAML: a top-level list of mappings. Every record must be a
    mapping; anything else raises :class:`FixtureError`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = _read_jsonl(path)
    elif suffix in (".yaml", ".yml"):
        records = _read_yaml_list(path)
    else:
        raise FixtureError(f"{path}: unsupported extension {suffix!r} (want .jsonl/.yaml)")
    for i, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            raise FixtureError(f"{path} record {i}: expected a mapping, got {type(rec).__name__}")
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise FixtureError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
    return records


def _read_yaml_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return []
    if not isinstance(data, list):
        raise FixtureError(f"{path}: expected a top-level YAML list of records")
    return data


def load_examples(path: str | Path) -> list[DatasetExample]:
    """Load and schema-validate every example in a fixture file.

    Raises :class:`FixtureError` on the first record that is not a well-formed
    :class:`DatasetExample`, with the file path and record index in the message.
    """
    records = read_records(path)
    examples: list[DatasetExample] = []
    for i, rec in enumerate(records, start=1):
        try:
            examples.append(DatasetExample(**rec))
        except ValidationError as exc:
            raise FixtureError(f"{path} record {i}: schema validation failed:\n{exc}") from exc
    return examples


def load_defects(path: str | Path) -> dict[str, Any]:
    """Load a ground-truth ``defects.yaml`` file into a plain dict.

    The file documents the defects deliberately planted in a fixture so the audit
    stages can be scored against known ground truth. Its shape is descriptive, not
    a pydantic model — the categories are broader than any single audit enum.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise FixtureError(f"{path}: expected a top-level mapping")
    return data


def discover_fixtures(root: str | Path | None = None) -> list[Path]:
    """Return sorted fixture directories (those containing ``examples.jsonl``)."""
    base = Path(root) if root is not None else FIXTURES_ROOT
    if not base.is_dir():
        return []
    return sorted(p.parent for p in base.glob("*/examples.jsonl"))
