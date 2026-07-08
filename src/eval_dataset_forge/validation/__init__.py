"""Validation & redaction layer (Milestone 3).

This is audit dimension #1: *is the data well-formed and safe to ship?* Everything
downstream (dedup, leakage, label audit, splits, statistics) assumes it is
operating on schema-valid examples, so this layer runs first and is strict about
structure while being explicit about the limits of its heuristic safety checks.

The public entry point is :func:`validate_dataset`, which returns a
:class:`ValidationReport` aggregating:

* **schema errors** — unreadable records, missing/unknown/mistyped fields
  (:mod:`.schema`);
* **duplicate ids** — the same ``example_id`` used by more than one record;
* **quality warnings** — empty prompts/answers, off-vocabulary task families
  (:mod:`.quality`);
* **PII/secret findings** — emails, tokens, key-shaped strings (:mod:`.pii`),
  with a redacted, printable snippet each.

Severity policy (see :attr:`ValidationReport.ok`): schema errors, duplicate ids,
and unambiguous *secret* findings are always errors; personal-identifier findings
are warnings unless ``fail_on_pii`` promotes them. Nothing is auto-corrected —
:mod:`.redaction` offers opt-in sanitization, but validation only reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval_dataset_forge.validation import pii, quality, redaction, schema
from eval_dataset_forge.validation.pii import PiiFinding, scan, scan_examples, shannon_entropy
from eval_dataset_forge.validation.quality import QualityWarning, check_quality
from eval_dataset_forge.validation.redaction import redact, redact_example
from eval_dataset_forge.validation.schema import (
    SchemaError,
    find_duplicate_ids,
    load_records_lenient,
    validate_records,
)


@dataclass
class ValidationReport:
    """Aggregated outcome of validating one dataset file.

    ``fail_on_pii`` records the policy the report was computed under so
    :attr:`ok`/:attr:`errors` are self-consistent with how it was produced.
    """

    dataset_path: str
    n_records: int
    n_valid: int
    schema_errors: list[SchemaError] = field(default_factory=list)
    duplicate_ids: dict[str, list[int]] = field(default_factory=dict)
    quality_warnings: list[QualityWarning] = field(default_factory=list)
    pii_findings: list[PiiFinding] = field(default_factory=list)
    fail_on_pii: bool = False

    # -- severity partitioning -------------------------------------------- #

    @property
    def blocking_pii(self) -> list[PiiFinding]:
        """PII findings that count as errors under the current policy."""
        return [f for f in self.pii_findings if f.is_secret or self.fail_on_pii]

    @property
    def warning_pii(self) -> list[PiiFinding]:
        """PII findings that remain warnings under the current policy."""
        blocking = set(id(f) for f in self.blocking_pii)
        return [f for f in self.pii_findings if id(f) not in blocking]

    @property
    def n_errors(self) -> int:
        return len(self.schema_errors) + len(self.duplicate_ids) + len(self.blocking_pii)

    @property
    def n_warnings(self) -> int:
        return len(self.quality_warnings) + len(self.warning_pii)

    @property
    def ok(self) -> bool:
        """True iff nothing that blocks a release was found."""
        return self.n_errors == 0

    # -- serialization ---------------------------------------------------- #

    def counts(self) -> dict[str, int]:
        return {
            "records": self.n_records,
            "valid": self.n_valid,
            "schema_errors": len(self.schema_errors),
            "duplicate_ids": len(self.duplicate_ids),
            "quality_warnings": len(self.quality_warnings),
            "pii_findings": len(self.pii_findings),
            "secret_findings": sum(1 for f in self.pii_findings if f.is_secret),
            "errors": self.n_errors,
            "warnings": self.n_warnings,
        }

    def to_dict(self) -> dict:
        """A plain-dict view suitable for ``json.dumps`` (``validate --json``)."""
        return {
            "dataset_path": self.dataset_path,
            "ok": self.ok,
            "fail_on_pii": self.fail_on_pii,
            "counts": self.counts(),
            "schema_errors": [asdict(e) for e in self.schema_errors],
            "duplicate_ids": self.duplicate_ids,
            "quality_warnings": [asdict(w) for w in self.quality_warnings],
            "pii_findings": [asdict(f) for f in self.pii_findings],
        }


def validate_dataset(path: str | Path, *, fail_on_pii: bool = False) -> ValidationReport:
    """Validate a dataset file end to end and return a :class:`ValidationReport`.

    Runs, in order: lenient parse → schema validation → duplicate-id detection →
    quality checks → PII/secret scan. Fully offline and deterministic. Only
    successfully parsed examples are scanned for content quality and PII; records
    that fail parsing/schema are already reported as errors.
    """
    indexed, parse_errors = load_records_lenient(path)
    examples_idx, schema_errors = validate_records(indexed)
    all_schema_errors = parse_errors + schema_errors

    duplicate_ids = find_duplicate_ids(indexed)
    example_objs = [ex for _, ex in examples_idx]
    quality_warnings = check_quality(example_objs)
    pii_findings = scan_examples(example_objs)

    return ValidationReport(
        dataset_path=str(path),
        n_records=len(indexed) + len(parse_errors),
        n_valid=len(example_objs),
        schema_errors=all_schema_errors,
        duplicate_ids=duplicate_ids,
        quality_warnings=quality_warnings,
        pii_findings=pii_findings,
        fail_on_pii=fail_on_pii,
    )


def write_redacted(src: str | Path, dst: str | Path) -> int:
    """Write a redacted copy of the valid examples in ``src`` to ``dst``.

    Each schema-valid example is redacted with :func:`redact_example` and written
    as one JSON object per line. Records that fail schema validation are skipped
    (they are surfaced separately by :func:`validate_dataset`). Returns the number
    of examples written.
    """
    indexed, _ = load_records_lenient(src)
    examples_idx, _ = validate_records(indexed)
    lines = [
        json.dumps(redact_example(ex).model_dump(mode="json"), ensure_ascii=False)
        for _, ex in examples_idx
    ]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return len(lines)


def format_report(report: ValidationReport, max_items: int = 20) -> str:
    """Render a human-readable summary of a :class:`ValidationReport`.

    Lists up to ``max_items`` entries per category so a large broken file does not
    flood the terminal; the counts line always reflects the full totals.
    """
    counts = report.counts()
    verdict = "OK" if report.ok else "FAILED"
    lines = [
        f"validate: {report.dataset_path}",
        f"  verdict          : {verdict}",
        f"  records          : {counts['records']} ({counts['valid']} schema-valid)",
        f"  schema errors    : {counts['schema_errors']}",
        f"  duplicate ids    : {counts['duplicate_ids']}",
        f"  quality warnings : {counts['quality_warnings']}",
        f"  pii/secret flags : {counts['pii_findings']} "
        f"({counts['secret_findings']} secret-class)",
    ]

    def section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title} ({len(items)}):")
        for item in items[:max_items]:
            lines.append(f"  - {item}")
        if len(items) > max_items:
            lines.append(f"  … and {len(items) - max_items} more")

    section(
        "schema errors",
        [
            f"record {e.record_index} [{e.kind}]"
            + (f" {e.field}" if e.field else "")
            + f": {e.detail}"
            for e in report.schema_errors
        ],
    )
    section(
        "duplicate ids",
        [f"{eid!r} at records {idxs}" for eid, idxs in sorted(report.duplicate_ids.items())],
    )
    section(
        "pii / secrets",
        [
            f"{'SECRET ' if f.is_secret else ''}{f.kind} in "
            f"{f.example_id or '?'}.{f.field}: {f.snippet}"
            for f in report.pii_findings
        ],
    )
    section(
        "quality warnings",
        [f"{w.kind} in {w.example_id or '?'}: {w.detail}" for w in report.quality_warnings],
    )
    return "\n".join(lines)


__all__ = [
    # orchestration
    "ValidationReport",
    "validate_dataset",
    "format_report",
    "write_redacted",
    # submodules
    "schema",
    "quality",
    "pii",
    "redaction",
    # re-exports
    "SchemaError",
    "QualityWarning",
    "PiiFinding",
    "find_duplicate_ids",
    "load_records_lenient",
    "validate_records",
    "check_quality",
    "scan",
    "scan_examples",
    "shannon_entropy",
    "redact",
    "redact_example",
]
