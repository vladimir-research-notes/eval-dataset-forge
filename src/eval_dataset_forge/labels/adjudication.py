"""The adjudication queue: what a human still has to decide (Milestone 6).

The other two modules find defects. This one decides which of them a machine may
propose a fix for, and routes the rest to a person. It emits the fifth and last
:class:`~eval_dataset_forge.schemas.LabelIssue` kind,
:attr:`~eval_dataset_forge.schemas.LabelIssue.NEEDS_ADJUDICATION`.

**One rule.** An example is *blocking* exactly when at least one of its issues has no
resolution the audit can propose. Nothing else is special-cased, and each kind's
blocking-ness follows:

=======================  ===========================================================
issue                    proposed resolution
=======================  ===========================================================
``MISSING``              none — the tool does not know the answer
``AMBIGUOUS``            none — the tool cannot decide which answer was meant
``CONFLICTING``          the strict-majority annotator label, when one exists
``INCONSISTENT_FORMAT``  convert to the family's dominant shape or label vocabulary
=======================  ===========================================================

A conflict resolves only when annotators are its *sole* source and one label holds a
strict majority. A 2-way tie has no majority. A shared-prompt conflict pits two examples
against each other with no third vote. A relevance conflict pits an example's gold list
against its own passage flags — two authorities, no tiebreak — so it blocks even though
"copy the flags into the gold list" looks like an obvious fix: it is equally possible
that the flag is wrong, and guessing would silently rewrite the benchmark.

**Nothing is applied.** A proposal is a string in a queue, never an edit. The queue is
deterministic (blocking first, then by severity, then by ``example_id``) and one JSON
object per line, so it diffs cleanly across runs and a reviewer's resolutions can be
tracked in version control.

``adjudication_queue.jsonl`` is a report artifact, not part of a release directory
(``docs/PLAN.md`` §5D), so it is a plain dataclass rather than a pydantic on-disk model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_dataset_forge.labels.audit import FamilyExpectation, answer_shape
from eval_dataset_forge.labels.disagreement import (
    annotations,
    label_key,
    majority_value,
)
from eval_dataset_forge.schemas import DatasetExample, LabelAuditFinding, LabelIssue

# Queue ordering: the issues that make an example unusable come before the ones that
# make it merely untidy.
SEVERITY: dict[str, int] = {
    LabelIssue.MISSING: 0,
    LabelIssue.CONFLICTING: 1,
    LabelIssue.AMBIGUOUS: 2,
    LabelIssue.INCONSISTENT_FORMAT: 3,
}

# Kinds a human must resolve from scratch; see the module docstring.
_UNRESOLVABLE_KINDS = frozenset({LabelIssue.MISSING, LabelIssue.AMBIGUOUS})


@dataclass
class AdjudicationItem:
    """One example awaiting a human decision, with everything needed to make it."""

    example_id: str
    task_family: str
    split: str
    issues: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    expected_output: Any = None
    annotations: list[dict[str, Any]] = field(default_factory=list)
    proposed_resolution: str | None = None
    blocking: bool = True

    @property
    def severity(self) -> int:
        """Rank of the most severe issue on this example (lower is more severe)."""
        return min(
            (SEVERITY.get(issue, len(SEVERITY)) for issue in self.issues),
            default=len(SEVERITY),
        )

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "task_family": self.task_family,
            "split": self.split,
            "issues": self.issues,
            "details": self.details,
            "expected_output": self.expected_output,
            "annotations": self.annotations,
            "proposed_resolution": self.proposed_resolution,
            "blocking": self.blocking,
        }


# --------------------------------------------------------------------------- #
# Proposed resolutions
# --------------------------------------------------------------------------- #


def conflict_resolutions(
    examples: Iterable[DatasetExample],
    *,
    annotator: Iterable[LabelAuditFinding],
    shared_prompt: Iterable[LabelAuditFinding],
    relevance: Iterable[LabelAuditFinding],
) -> dict[str, str]:
    """Map ``example_id`` → proposal, for the conflicts the audit can resolve.

    Only annotator disagreement with a strict majority qualifies, and only when it is
    the example's sole conflict source: a majority vote cannot arbitrate a disagreement
    the annotators were never asked about.
    """
    by_id = {ex.example_id: ex for ex in examples}
    unresolvable = {finding.example_id for finding in shared_prompt}
    unresolvable |= {finding.example_id for finding in relevance}

    resolutions: dict[str, str] = {}
    for finding in annotator:
        example = by_id.get(finding.example_id)
        if example is None or finding.example_id in unresolvable:
            continue
        items = annotations(example)
        majority = majority_value(items)
        if majority is None:
            continue
        votes = sum(1 for item in items if label_key(item.value) == label_key(majority))
        resolutions[finding.example_id] = (
            f"adopt the majority annotator label {majority!r} "
            f"({votes}/{len(items)} annotators)"
        )
    return resolutions


def format_resolution(
    example: DatasetExample, expectation: FamilyExpectation | None
) -> str | None:
    """How to bring one gold answer back into its family's format, or ``None``."""
    if expectation is None:
        return None
    shape = answer_shape(example.expected_output)
    if expectation.dominant_shape and shape != expectation.dominant_shape:
        return (
            f"convert expected_output from {shape} to the family's dominant shape "
            f"{expectation.dominant_shape}"
        )
    if expectation.vocabulary:
        return (
            f"map the gold answer onto the {example.task_family!r} label vocabulary "
            f"{list(expectation.vocabulary)}"
        )
    return None


def _resolution(
    example: DatasetExample,
    finding: LabelAuditFinding,
    *,
    conflicts: dict[str, str],
    expectations: dict[str, FamilyExpectation],
) -> str | None:
    if finding.kind in _UNRESOLVABLE_KINDS:
        return None
    if finding.kind == LabelIssue.CONFLICTING:
        return conflicts.get(example.example_id)
    if finding.kind == LabelIssue.INCONSISTENT_FORMAT:
        return format_resolution(example, expectations.get(example.task_family))
    return None


# --------------------------------------------------------------------------- #
# Queue
# --------------------------------------------------------------------------- #


def build_queue(
    examples: Iterable[DatasetExample],
    findings: Iterable[LabelAuditFinding],
    *,
    conflicts: dict[str, str] | None = None,
    expectations: dict[str, FamilyExpectation] | None = None,
) -> list[AdjudicationItem]:
    """Group findings by example into a deterministic, reviewable queue.

    ``findings`` are the four primary kinds; ``NEEDS_ADJUDICATION`` is derived from the
    queue by :func:`needs_adjudication` and must not be passed back in. ``conflicts`` and
    ``expectations`` come from the orchestrator, which has already computed them.
    """
    by_id = {ex.example_id: ex for ex in examples}
    conflicts = conflicts or {}
    expectations = expectations or {}

    items: dict[str, AdjudicationItem] = {}
    resolutions: dict[str, list[str | None]] = {}
    for finding in findings:
        example = by_id.get(finding.example_id)
        if example is None:
            continue
        item = items.get(finding.example_id)
        if item is None:
            item = AdjudicationItem(
                example_id=example.example_id,
                task_family=example.task_family,
                split=str(example.split),
                expected_output=example.expected_output,
                annotations=[
                    {"annotator": ann.annotator, "value": ann.value}
                    for ann in annotations(example)
                ],
            )
            items[finding.example_id] = item
            resolutions[finding.example_id] = []
        if str(finding.kind) not in item.issues:
            item.issues.append(str(finding.kind))
        item.details.append(finding.detail)
        resolutions[finding.example_id].append(
            _resolution(example, finding, conflicts=conflicts, expectations=expectations)
        )

    for example_id, item in items.items():
        proposed = [text for text in resolutions[example_id] if text]
        item.blocking = any(text is None for text in resolutions[example_id])
        item.proposed_resolution = "; ".join(proposed) or None
        item.issues.sort(key=lambda issue: SEVERITY.get(issue, len(SEVERITY)))

    return sorted(
        items.values(),
        key=lambda item: (not item.blocking, item.severity, item.example_id),
    )


def needs_adjudication(queue: Iterable[AdjudicationItem]) -> list[LabelAuditFinding]:
    """A ``NEEDS_ADJUDICATION`` finding for every blocking queue item."""
    return [
        LabelAuditFinding(
            kind=LabelIssue.NEEDS_ADJUDICATION,
            example_id=item.example_id,
            detail=(
                f"{len(item.issues)} issue(s) ({', '.join(item.issues)}) with no "
                f"resolution the audit can propose"
            ),
        )
        for item in queue
        if item.blocking
    ]


def write_queue(queue: Iterable[AdjudicationItem], path: str | Path) -> int:
    """Write the queue to ``path`` as JSONL. Returns the number of items written.

    One compact JSON object per line, in queue order, with a trailing newline — so two
    runs over the same dataset produce byte-identical files and a reviewer's edits show
    up as line diffs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(queue)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(items)


__all__ = [
    "SEVERITY",
    "AdjudicationItem",
    "build_queue",
    "conflict_resolutions",
    "format_resolution",
    "needs_adjudication",
    "write_queue",
]
