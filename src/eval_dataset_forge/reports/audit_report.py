"""The human-readable audit report — ``audit.md`` (Milestone 12).

``docs/PLAN.md`` §5D has listed ``audit.{json,md}`` in every release bundle since Milestone 0,
and every release built before this milestone shipped only the JSON. ``REPRODUCE.md`` said so
rather than letting the gap pass unremarked. This module closes it.

**The report is a view, not a second computation.** :func:`audit_document` takes an
:class:`~eval_dataset_forge.schemas.AuditResult` and nothing else — no examples, no manifest, no
filesystem. It therefore cannot disagree with the ``audit.json`` beside it, for the same reason
the dataset card cannot disagree with the audit it reads: there is no second place for a number
to be written down. The one thing this costs is that ``audit.md`` can only say what
``AuditResult`` has a field for; evidence the schema does not carry (duplicate *source* groups,
the adjudication queue) stays in the stage reports and in the smoke audit that scores them.

**Zeros are printed, and they are annotated.** A report that listed only what it found could not
be told apart from a report whose checks never ran, so every dimension appears with its count and
with what that count is worth. ``0`` on an exact check means there is nothing of that kind.
``0`` on a heuristic check means the heuristic found nothing — an upper bound on what was looked
for, not a statement about what is there. The two are different claims and the ``certainty``
column is where they are distinguished.

**A check that could not run is not a check that passed.** When the audited rows carry no split,
the cross-split signals return zero by vacuity. :mod:`eval_dataset_forge.audit` records that in
``split_statistics`` and in a recommendation; this renderer surfaces it as a note above the table
rather than leaving a reader to infer it from a ``0``.

The certainty vocabulary and the per-kind counters live here, and
:mod:`eval_dataset_forge.reports.dataset_card` imports them. The card's findings table and this
report's make the same claims about the same dimensions; two copies of that table would be two
places for the word "heuristic" to go missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from eval_dataset_forge.reports import html as html_backend
from eval_dataset_forge.reports import markdown as md_backend
from eval_dataset_forge.reports.document import (
    Bullets,
    Document,
    Heading,
    Note,
    Paragraph,
    Table,
    blocks_of,
    rows,
)
from eval_dataset_forge.schemas import (
    AuditResult,
    DedupKind,
    LabelIssue,
    LeakageKind,
)

AUDIT_MD_FILE = "audit.md"

# How many example ids a bullet names before it stops. A report is read; an ``audit.json`` is
# queried. Six ids fit on a line and are enough to start looking.
MAX_IDS = 6

# How many finding rows a table prints per dimension before it truncates and says so. A
# truncation that did not announce itself would read as an absence.
MAX_ROWS = 25

# What each audit dimension can and cannot promise. Printed beside every count, because
# "0 near-duplicates" and "0 exact duplicates" are claims of very different strength and a bare
# table implies they are the same claim. Imported by the dataset card, which prints the same
# column about the same dimensions.
CERTAINTY: dict[str, str] = {
    "exact_duplicates": "exact",
    "normalized_duplicates": "exact after normalization",
    "near_duplicates": "heuristic",
    str(LeakageKind.TRAIN_TEST_OVERLAP): "inherits the dedup ladder (exact or heuristic)",
    str(LeakageKind.ANSWER_IN_PROMPT): "heuristic",
    str(LeakageKind.NGRAM_OVERLAP): "exact given the n-gram width",
    str(LeakageKind.RETRIEVAL_CONTAMINATION): "exact",
    str(LabelIssue.MISSING): "exact",
    str(LabelIssue.CONFLICTING): "exact",
    str(LabelIssue.AMBIGUOUS): "declaration, plus a heuristic",
    str(LabelIssue.INCONSISTENT_FORMAT): "exact",
    str(LabelIssue.NEEDS_ADJUDICATION): "derived from the rows above",
    "pii_flags": "heuristic (pattern match)",
}

# Row labels for the findings table.
LEAKAGE_LABEL = {
    str(LeakageKind.TRAIN_TEST_OVERLAP): "leakage: train/test overlap",
    str(LeakageKind.ANSWER_IN_PROMPT): "leakage: answer in prompt",
    str(LeakageKind.NGRAM_OVERLAP): "leakage: n-gram overlap",
    str(LeakageKind.RETRIEVAL_CONTAMINATION): "leakage: retrieval contamination",
}

LABEL_LABEL = {
    str(LabelIssue.MISSING): "labels: missing gold answer",
    str(LabelIssue.CONFLICTING): "labels: conflicting annotations",
    str(LabelIssue.AMBIGUOUS): "labels: ambiguous",
    str(LabelIssue.INCONSISTENT_FORMAT): "labels: inconsistent format",
    str(LabelIssue.NEEDS_ADJUDICATION): "labels: needs adjudication",
}


# --------------------------------------------------------------------------- #
# Derived views over an AuditResult
# --------------------------------------------------------------------------- #


def dedup_counts(audit: AuditResult) -> dict[str, int]:
    """Cluster counts per rung of the ladder.

    ``AuditResult`` has two buckets and the ladder has three rungs: ``normalized`` findings live
    under ``near_duplicates`` and keep their ``kind``. Split them back apart, because a normalized
    match is exact and a near match is not.
    """
    normalized = sum(1 for f in audit.near_duplicates if f.kind is DedupKind.NORMALIZED)
    return {
        "exact_duplicates": len(audit.exact_duplicates),
        "normalized_duplicates": normalized,
        "near_duplicates": len(audit.near_duplicates) - normalized,
    }


def counts_by_kind(findings: Iterable[Any], vocabulary: Iterable[Any]) -> dict[str, int]:
    """Count findings per kind, including the kinds that fired zero times.

    The zeros are the point. A report listing only what was found cannot be distinguished from a
    report whose pipeline never ran that check.
    """
    counts = dict.fromkeys((str(kind) for kind in vocabulary), 0)
    for finding in findings:
        counts[str(finding.kind)] += 1
    return counts


def ids(values: Sequence[str]) -> str:
    """Render up to :data:`MAX_IDS` example ids as code spans, then a tail count."""
    shown = ", ".join(f"`{value}`" for value in values[:MAX_IDS])
    remainder = len(values) - MAX_IDS
    return f"{shown} … and {remainder} more" if remainder > 0 else shown


def _near_method(audit: AuditResult) -> str:
    dependence = audit.split_statistics.get("dependence", {})
    return str(dependence.get("near_method") or "unspecified")


def _counts(audit: AuditResult) -> dict[str, Any]:
    return audit.split_statistics.get("counts", {})


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #


def _glance_table(audit: AuditResult) -> Table:
    stats = audit.split_statistics
    dependence = stats.get("dependence", {})
    counts = _counts(audit)
    sizes = (
        " · ".join(
            f"{name} {counts[name]}"
            for name in ("train", "dev", "test", "unassigned")
            if counts.get(name)
        )
        or "not split"
    )
    spec = stats.get("spec") or {}
    strategy = (
        f"`{spec.get('strategy')}` (seed {spec.get('seed')})"
        if spec
        else "none — the audit read the split the rows carry"
    )
    return Table(
        headers=("field", "value"),
        rows=rows(
            [
                ("dataset", f"`{audit.dataset_id}`"),
                ("version", f"`{audit.version}`"),
                ("`audit_id`", f"`{audit.audit_id}`"),
                ("examples", counts.get("examples", "n/a")),
                ("independent blocks", dependence.get("block_count", "n/a")),
                ("splits", sizes),
                ("split source", f"`{stats.get('source', 'unknown')}`"),
                ("split strategy", strategy),
                ("near-duplicate method", f"`{_near_method(audit)}`"),
            ]
        ),
    )


def _unassigned_note(audit: AuditResult) -> Note | None:
    """Say out loud that the cross-split checks were not exercised, if they were not."""
    unassigned = _counts(audit).get("unassigned", 0)
    if not unassigned:
        return None
    return Note(
        f"{unassigned} example(s) carry `split=unassigned`. Every cross-split signal below — "
        f"train/test overlap, cross-split source reuse, n-gram containment against the `train` "
        f"reference, and the `dev`/`test` duplicate count — had nothing to compare and reports "
        f"`0` by vacuity. A zero on those rows is not a clean bill of health; it is the absence "
        f"of a question."
    )


def _findings_table(audit: AuditResult) -> Table:
    duplicates = dedup_counts(audit)
    near_method = _near_method(audit)
    table: list[tuple[str, Any, str]] = [
        ("duplicates: exact", duplicates["exact_duplicates"], CERTAINTY["exact_duplicates"]),
        (
            "duplicates: normalized",
            duplicates["normalized_duplicates"],
            CERTAINTY["normalized_duplicates"],
        ),
        (
            "duplicates: near",
            duplicates["near_duplicates"],
            f"{CERTAINTY['near_duplicates']} ({near_method})",
        ),
    ]
    table += [
        (LEAKAGE_LABEL[kind], count, CERTAINTY[kind])
        for kind, count in counts_by_kind(audit.leakage_flags, LeakageKind).items()
    ]
    table += [
        (LABEL_LABEL[kind], count, CERTAINTY[kind])
        for kind, count in counts_by_kind(audit.label_warnings, LabelIssue).items()
    ]
    table.append(("PII-like patterns", len(audit.pii_flags), CERTAINTY["pii_flags"]))
    return Table(
        headers=("dimension", "count", "certainty"),
        rows=rows(table),
        caption=(
            "Every heuristic count is a lower bound. A zero on a heuristic row means the check "
            "found nothing, not that there is nothing to find."
        ),
    )


def _duplicates_table(audit: AuditResult) -> Table | None:
    findings = [*audit.exact_duplicates, *audit.near_duplicates]
    if not findings:
        return None
    return Table(
        headers=("kind", "examples", "similarity", "method"),
        rows=rows(
            [
                (
                    f"`{finding.kind!s}`",
                    " + ".join(f"`{eid}`" for eid in finding.example_ids),
                    "—" if finding.similarity is None else f"{finding.similarity:.3f}",
                    f"`{finding.method}`" if finding.method else "—",
                )
                for finding in findings[:MAX_ROWS]
            ]
        ),
        caption=(
            f"{len(findings)} duplicate cluster(s); showing {min(len(findings), MAX_ROWS)}. "
            f"An `exact` or `normalized` cluster is identical by construction and carries no "
            f"similarity."
        ),
    )


def _leakage_table(audit: AuditResult) -> Table | None:
    if not audit.leakage_flags:
        return None
    findings = audit.leakage_flags
    return Table(
        headers=("kind", "examples", "score", "evidence"),
        rows=rows(
            [
                (
                    f"`{finding.kind!s}`",
                    " + ".join(f"`{eid}`" for eid in finding.example_ids),
                    "—" if finding.score is None else f"{finding.score:.3f}",
                    finding.evidence or "—",
                )
                for finding in findings[:MAX_ROWS]
            ]
        ),
        caption=(
            f"{len(findings)} leakage finding(s); showing {min(len(findings), MAX_ROWS)}. The "
            f"four channels are not disjoint: one defect can and should light up several, and "
            f"agreement between them is corroboration rather than double counting."
        ),
    )


def _labels_table(audit: AuditResult) -> Table | None:
    if not audit.label_warnings:
        return None
    findings = audit.label_warnings
    return Table(
        headers=("kind", "example", "detail"),
        rows=rows(
            [
                (f"`{finding.kind!s}`", f"`{finding.example_id}`", finding.detail or "—")
                for finding in findings[:MAX_ROWS]
            ]
        ),
        caption=(
            f"{len(findings)} label finding(s); showing {min(len(findings), MAX_ROWS)}. Nothing "
            f"is corrected. `needs_adjudication` is derived from the rows above it and names no "
            f"example they do not."
        ),
    )


def _pii_table(audit: AuditResult) -> Table | None:
    if not audit.pii_flags:
        return None
    return Table(
        headers=("example", "field", "kind", "redacted snippet"),
        rows=rows(
            [
                (
                    f"`{flag.get('example_id') or '—'}`",
                    f"`{flag.get('field', '')}`",
                    f"`{flag.get('kind', '')}`",
                    flag.get("snippet", ""),
                )
                for flag in audit.pii_flags[:MAX_ROWS]
            ]
        ),
        caption=(
            "Snippets are redacted before they reach this table; the matched value never enters "
            "a report."
        ),
    )


def _difficulty_table(audit: AuditResult) -> Table | None:
    distribution = audit.difficulty_distribution
    if not distribution:
        return None
    total = sum(distribution.values()) or 1
    return Table(
        headers=("bucket", "examples", "share"),
        rows=rows(
            [(f"`{name}`", count, f"{count / total:.1%}") for name, count in distribution.items()]
        ),
    )


def _splits_table(audit: AuditResult) -> Table:
    counts = _counts(audit)
    dependence = audit.split_statistics.get("dependence", {})
    blocks_by_split = dependence.get("blocks_by_split", {})
    # An empty ``unassigned`` split is not a fact worth a row; a non-empty one has its own note.
    names = [
        name
        for name in ("train", "dev", "test", "unassigned")
        if name in counts and (name != "unassigned" or counts[name])
    ]
    return Table(
        headers=("split", "examples", "independent blocks touched"),
        rows=rows(
            [(f"`{name}`", counts[name], blocks_by_split.get(name, "—")) for name in names]
        ),
        caption=(
            "A block is a set of examples that must share a split: a duplicate cluster, or a "
            "source group. A block straddling a boundary is counted in every split it reaches, "
            "so these do not sum to the corpus block count."
        ),
    )


def _dependence_note(audit: AuditResult) -> Note | None:
    dependence = audit.split_statistics.get("dependence", {})
    total = _counts(audit).get("examples", 0)
    block_count = dependence.get("block_count")
    if block_count is None or block_count == total:
        return None
    straddling = dependence.get("straddling_block_count", 0)
    tail = (
        f" {straddling} block(s) straddle a split boundary."
        if straddling
        else " No block straddles a split boundary."
    )
    return Note(
        f"This dataset holds {total} examples but only {block_count} independent blocks: "
        f"{total - block_count} example(s) are duplicates or share a source record with another. "
        f"The sample size of an interval or a significance test is the block count, not the "
        f"example count. Resample blocks — `evalforge stats --resample-unit block`.{tail} No "
        f"effective sample size is quoted here: that needs an intraclass correlation, which needs "
        f"system scores, which a dataset does not have."
    )


def _limitations(audit: AuditResult) -> Bullets:
    """What this audit cannot tell a reader, stated in the audit rather than the docs."""
    items = [
        f"Near-duplicate detection is lexical (`{_near_method(audit)}`). A paraphrase below the "
        f"threshold is not detected, so every duplicate, leakage, and block count above is a "
        f"lower bound.",
        "Contamination is only measurable against text this tool was handed. A model's "
        "pretraining corpus is not ours to read, so a clean leakage report says nothing about "
        "it. Absence of a flag is not evidence of no contamination.",
        "The label audit sees structure, never truth. A gold answer that is well-formed, "
        "unanimous, unambiguous, and correctly shaped is invisible to every check here, however "
        "wrong it is. Detecting that needs a second annotator or a model.",
        "The difficulty buckets come from a transparent, model-free heuristic over surface "
        "signals, never from measured model performance. Whether that heuristic's ranking is "
        "distinguishable from chance on this corpus is a separate question — "
        "`evalforge difficulty` answers it, and the recommendations above report the verdict.",
        "PII detection is a pattern match. It finds the shapes it knows and no others.",
    ]
    if _counts(audit).get("unassigned"):
        items.append(
            "The cross-split checks were not exercised: the rows carry no split. Their zeros "
            "measure nothing."
        )
    return Bullets(tuple(items))


def audit_document(audit: AuditResult) -> Document:
    """Lay an :class:`~eval_dataset_forge.schemas.AuditResult` out as a backend-independent
    :class:`~eval_dataset_forge.reports.document.Document`.

    A pure function of ``audit``. No examples are read, so the report cannot report a defect in a
    row the audit did not describe.
    """
    return Document(
        title=f"Audit report — {audit.dataset_id} @ {audit.version}",
        blocks=blocks_of(
            Paragraph(
                "A rendering of this dataset's `audit.json`. Every number below is read from "
                "that file, never recomputed, so the report and the machine-readable evidence "
                "beside it cannot disagree. Nothing here is corrected, deleted, or scored — an "
                "audit produces evidence for a reviewer."
            ),
            _glance_table(audit),
            _unassigned_note(audit),
            Heading("Findings"),
            _findings_table(audit),
            Heading("Duplicate clusters"),
            _duplicates_table(audit)
            or Paragraph("No duplicate cluster was found by any rung of the ladder."),
            Heading("Leakage"),
            _leakage_table(audit)
            or Paragraph("No leakage finding fired on any of the four channels."),
            Heading("Label warnings"),
            _labels_table(audit) or Paragraph("The label audit raised nothing."),
            Heading("PII-like patterns"),
            _pii_table(audit) or Paragraph("The pattern scan flagged nothing."),
            Heading("Splits and dependence"),
            _splits_table(audit),
            _dependence_note(audit),
            Heading("Difficulty distribution"),
            _difficulty_table(audit) or Paragraph("No difficulty scores were recorded."),
            Heading("Recommendations"),
            Bullets(tuple(audit.recommendations))
            if audit.recommendations
            else Paragraph("The audit raised no recommendations for this dataset."),
            Heading("What this audit cannot tell you"),
            _limitations(audit),
            Note(
                "Regenerate this report from the evidence with `evalforge audit --dataset <path> "
                "--out <dir>`. It is a view of `audit.json`; to check a release's bytes against "
                "its manifest use `evalforge verify-release` instead."
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_markdown(audit: AuditResult) -> bytes:
    """The exact bytes of ``audit.md``. Deterministic; no clock, no filesystem."""
    return md_backend.to_bytes(audit_document(audit))


def render_html(audit: AuditResult) -> bytes:
    """The exact bytes of a standalone ``audit.html``. Not part of a release bundle."""
    return html_backend.to_bytes(audit_document(audit))


def write_report(
    audit: AuditResult,
    *,
    markdown_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write the requested renderings. Returns artifact name → path."""
    from eval_dataset_forge.datasets.writer import write_bytes

    written: dict[str, Path] = {}
    if markdown_path is not None:
        written["markdown"] = write_bytes(markdown_path, render_markdown(audit))
    if html_path is not None:
        written["html"] = write_bytes(html_path, render_html(audit))
    return written


__all__ = [
    "AUDIT_MD_FILE",
    "CERTAINTY",
    "LABEL_LABEL",
    "LEAKAGE_LABEL",
    "MAX_IDS",
    "MAX_ROWS",
    "audit_document",
    "counts_by_kind",
    "dedup_counts",
    "ids",
    "render_html",
    "render_markdown",
    "write_report",
]
