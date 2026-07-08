"""Cross-split overlap: the same item evaluated on and trained against (Milestone 5).

A duplicate that stays inside one split wastes sample size; a duplicate that
straddles two splits invalidates the evaluation. This module reads the
deduplication ladder's output and keeps only the clusters that cross a split
boundary, so it inherits each stage's certainty exactly: exact and normalized
findings are exact, near findings are heuristic, and the reported ``score`` is the
dedup ``similarity`` (``None`` where the pair is identical by construction).

**Any two assigned splits count.** The check is named for ``train``/``test``
because that is the canonical failure, but ``dev``/``test`` overlap is the same
defect — tuning on an item then scoring on it. ``unassigned`` is excluded: an
example not yet placed in a split cannot leak across one, and treating it as a leak
would flag every raw dataset before its splits are built.

**Source-level dependence is reported separately.** Two examples derived from one
``source_id`` are not duplicates — the fixtures' ``src-shakespeare`` yields distinct
questions about *Romeo and Juliet* and about *Hamlet* — but they are statistically
dependent, and scattering them across train and test leaks source-level
information. :func:`cross_split_sources` surfaces that as its own channel rather
than as a :class:`~eval_dataset_forge.schemas.LeakageFinding`, because "dependent"
and "duplicated" call for different remedies: the first wants a source-grouped
split (Milestone 7), the second wants a deletion.

The blind spots are inherited too. A cross-split paraphrase invisible to the near
stage is invisible here; on ``retrieval_toy`` that costs the one planted overlap
this module misses. See `docs/DEDUPLICATION.md` for why that pair is unreachable by
any lexical threshold.
"""

from __future__ import annotations

from collections.abc import Iterable

from eval_dataset_forge.dedup import DedupReport
from eval_dataset_forge.schemas import (
    DatasetExample,
    DedupKind,
    LeakageFinding,
    LeakageKind,
    Split,
)

# Splits that represent a real partition of the data; see the module docstring.
ASSIGNED_SPLITS: tuple[Split, ...] = (Split.TRAIN, Split.DEV, Split.TEST)

# How each dedup stage describes itself in a leakage finding's evidence.
_STAGE_NAME: dict[DedupKind, str] = {
    DedupKind.EXACT: "exact duplicate",
    DedupKind.NORMALIZED: "normalized duplicate",
    DedupKind.NEAR: "near-duplicate",
}


def _splits_of(examples: Iterable[DatasetExample]) -> dict[str, Split]:
    return {example.example_id: example.split for example in examples}


def _group_by_split(
    example_ids: Iterable[str], split_of: dict[str, Split]
) -> dict[Split, list[str]]:
    """Partition ids by assigned split, preserving input order within each split."""
    grouped: dict[Split, list[str]] = {}
    for example_id in example_ids:
        split = split_of.get(example_id)
        if split in ASSIGNED_SPLITS:
            grouped.setdefault(split, []).append(example_id)
    return grouped


def _describe(grouped: dict[Split, list[str]]) -> str:
    parts = [f"{split}={grouped[split]}" for split in ASSIGNED_SPLITS if split in grouped]
    return " ".join(parts)


def cross_split_clusters(
    examples: Iterable[DatasetExample], report: DedupReport
) -> list[LeakageFinding]:
    """Turn every duplicate cluster spanning ≥2 assigned splits into a finding.

    Ordering follows the dedup ladder — exact, then normalized, then near — so the
    certain findings are read first. A cluster's ``example_ids`` are carried over
    unchanged, including members sitting in ``unassigned``: they are part of the
    duplicate group even though they are not what makes it a leak.
    """
    split_of = _splits_of(examples)
    findings: list[LeakageFinding] = []
    for finding in report.findings:
        grouped = _group_by_split(finding.example_ids, split_of)
        if len(grouped) < 2:
            continue
        findings.append(
            LeakageFinding(
                kind=LeakageKind.TRAIN_TEST_OVERLAP,
                example_ids=list(finding.example_ids),
                score=finding.similarity,
                evidence=(
                    f"{_STAGE_NAME[finding.kind]} spans {len(grouped)} splits: "
                    f"{_describe(grouped)} (via {finding.method})"
                ),
            )
        )
    return findings


def cross_split_sources(
    examples: Iterable[DatasetExample], report: DedupReport
) -> dict[str, dict[str, list[str]]]:
    """Source records whose examples are spread across ≥2 assigned splits.

    Informational, not a finding: this is statistical dependence between splits, not
    a duplicated item. Consumed by the source-grouped split strategy (Milestone 7).
    """
    split_of = _splits_of(examples)
    spread: dict[str, dict[str, list[str]]] = {}
    for source_id, example_ids in sorted(report.duplicate_source_ids.items()):
        grouped = _group_by_split(example_ids, split_of)
        if len(grouped) >= 2:
            spread[source_id] = {str(split): ids for split, ids in grouped.items()}
    return spread


def split_counts(examples: Iterable[DatasetExample]) -> dict[str, int]:
    """Example count per split, including ``unassigned``. Used by report headers."""
    counts: dict[str, int] = {}
    for example in examples:
        counts[str(example.split)] = counts.get(str(example.split), 0) + 1
    return counts


__all__ = [
    "ASSIGNED_SPLITS",
    "cross_split_clusters",
    "cross_split_sources",
    "split_counts",
]
