"""Schema validation and ID-uniqueness checks (Milestone 3).

This module turns the :class:`~eval_dataset_forge.schemas.DatasetExample` pydantic
model into a file-level gate. Where the Milestone 2 loader raises on the *first*
malformed record (fine for hand-authored fixtures), the ``validate`` CLI needs to
report *every* problem across a file at once, so parsing here is deliberately
lenient: a bad JSONL line, a schema violation, or a non-object record becomes a
:class:`SchemaError` and the scan continues.

Every record is addressed by a 1-based **ordinal** — its position among the
non-blank, non-comment records in the file — so an error message points a human
at the right line regardless of which earlier records failed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from eval_dataset_forge.datasets import loader
from eval_dataset_forge.schemas import DatasetExample

# A record paired with its 1-based ordinal within the file.
IndexedRecord = tuple[int, dict]
IndexedExample = tuple[int, DatasetExample]


@dataclass(frozen=True)
class SchemaError:
    """One structural problem with a single record.

    ``kind`` is one of ``parse_error`` (unreadable JSON / not an object),
    ``missing_field``, ``unknown_field`` (a forbidden extra key), or
    ``type_error``. ``field`` is the dotted location within the record when known.
    """

    record_index: int
    example_id: str | None
    kind: str
    field: str | None
    detail: str


# --------------------------------------------------------------------------- #
# Lenient reading
# --------------------------------------------------------------------------- #


def load_records_lenient(path: str | Path) -> tuple[list[IndexedRecord], list[SchemaError]]:
    """Read a ``.jsonl``/``.yaml`` file into indexed records + parse errors.

    JSONL is read line by line so a single malformed line is reported without
    hiding the rest of the file. YAML is parsed whole (a YAML syntax error is not
    line-recoverable); a parse failure there yields one ``parse_error``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl_lenient(path)
    if suffix in (".yaml", ".yml"):
        try:
            records = loader.read_records(path)
        except loader.FixtureError as exc:
            return [], [SchemaError(0, None, "parse_error", None, str(exc))]
        return list(enumerate(records, start=1)), []
    return [], [
        SchemaError(0, None, "parse_error", None, f"unsupported extension {suffix!r}")
    ]


def _read_jsonl_lenient(path: Path) -> tuple[list[IndexedRecord], list[SchemaError]]:
    records: list[IndexedRecord] = []
    errors: list[SchemaError] = []
    ordinal = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ordinal += 1
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(
                    SchemaError(
                        ordinal, None, "parse_error", None, f"line {lineno}: invalid JSON: {exc}"
                    )
                )
                continue
            if not isinstance(obj, dict):
                errors.append(
                    SchemaError(
                        ordinal,
                        None,
                        "parse_error",
                        None,
                        f"line {lineno}: expected a JSON object, got {type(obj).__name__}",
                    )
                )
                continue
            records.append((ordinal, obj))
    return records, errors


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

# pydantic error type -> our coarser SchemaError.kind.
_KIND_BY_TYPE = {
    "missing": "missing_field",
    "extra_forbidden": "unknown_field",
}


def _record_id(record: dict) -> str | None:
    eid = record.get("example_id")
    return eid if isinstance(eid, str) and eid else None


def _to_schema_error(index: int, record: dict, err: dict) -> SchemaError:
    loc = err.get("loc", ())
    field = ".".join(str(part) for part in loc) or None
    kind = _KIND_BY_TYPE.get(err.get("type", ""), "type_error")
    return SchemaError(index, _record_id(record), kind, field, err.get("msg", ""))


def validate_records(
    indexed: list[IndexedRecord],
) -> tuple[list[IndexedExample], list[SchemaError]]:
    """Validate each indexed record against ``DatasetExample``.

    Returns the successfully parsed examples (with their ordinals) and a flat list
    of :class:`SchemaError`\\ s — one per pydantic sub-error, so a record with two
    problems produces two errors.
    """
    examples: list[IndexedExample] = []
    errors: list[SchemaError] = []
    for index, record in indexed:
        try:
            examples.append((index, DatasetExample(**record)))
        except ValidationError as exc:
            for err in exc.errors():
                errors.append(_to_schema_error(index, record, err))
        except TypeError as exc:  # e.g. non-string keys
            errors.append(SchemaError(index, _record_id(record), "type_error", None, str(exc)))
    return examples, errors


def find_duplicate_ids(indexed: list[IndexedRecord]) -> dict[str, list[int]]:
    """Return ``example_id`` -> sorted ordinals for ids used by >1 record.

    Operates on raw records (not parsed examples) so a duplicate id is reported
    even when one of the colliding records also fails schema validation. IDs use
    the same ordinal space as :class:`SchemaError`.
    """
    positions: dict[str, list[int]] = defaultdict(list)
    for index, record in indexed:
        eid = _record_id(record)
        if eid is not None:
            positions[eid].append(index)
    return {eid: sorted(idxs) for eid, idxs in positions.items() if len(idxs) > 1}


__all__ = [
    "SchemaError",
    "IndexedRecord",
    "IndexedExample",
    "load_records_lenient",
    "validate_records",
    "find_duplicate_ids",
]
