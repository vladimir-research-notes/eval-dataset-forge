"""Deduplication layer (Milestone 4).

Audit dimension #2: *does this dataset say the same thing more than once?*
Duplicates inflate apparent sample size, reward memorization, and quietly break
the independence assumption that the bootstrap confidence intervals in Milestone
9 rest on — a benchmark with 40 copies of one item has fewer effective examples
than its row count claims.

Detection runs as a ladder of three stages, each strictly more tolerant and
strictly less certain than the one before:

======================  ==========  ====================================
stage                   certainty   catches
======================  ==========  ====================================
:mod:`.exact`           exact       byte-identical prompt + gold answer
:mod:`.fuzzy`           exact*      identical after normalizing the prompt
:mod:`.near_duplicate`  heuristic   paraphrases, edits, reorderings
======================  ==========  ====================================

\\* exact *given* the normalization, which is itself a lossy choice.

The stages are disjoint by construction: a normalized finding always merges at
least two distinct raw contents, and a near finding never covers a pair the
earlier stages already resolved. So the three lists partition the duplicate
evidence rather than triple-counting it, and :meth:`DedupReport.counts` adds up.

Nothing is deleted. :func:`find_duplicates` reports clusters and the method and
similarity behind each one; choosing a survivor per cluster is a policy decision
that belongs to the split builder (Milestone 7) and the release builder
(Milestone 10), which need to know *which* copy sits in the test split.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from eval_dataset_forge.dedup import exact, fuzzy, near_duplicate
from eval_dataset_forge.dedup.exact import (
    canonical_output,
    content_signature,
    duplicate_input_clusters,
    duplicate_sources,
    exact_duplicates,
    sha256_hex,
)
from eval_dataset_forge.dedup.fuzzy import (
    jaccard,
    normalize_text,
    normalized_duplicates,
    normalized_signature,
    tokenize,
    word_ngrams,
)
from eval_dataset_forge.dedup.near_duplicate import (
    DEFAULT_NGRAM,
    DEFAULT_THRESHOLDS,
    answers_agree,
    near_duplicates,
    resolve_near_method,
    similarity_matrix,
)
from eval_dataset_forge.schemas import DatasetExample, DeduplicationFinding


@dataclass
class DedupReport:
    """Everything the three dedup stages found on one dataset.

    ``near_method``/``near_threshold``/``near_ngram`` record the parameters the
    heuristic stage actually ran with (after ``auto`` resolution and any
    scikit-learn fallback), so a report is self-describing: a reader never has to
    guess whether a near-duplicate list came from TF-IDF or from the pure-Python
    path.
    """

    n_examples: int
    exact: list[DeduplicationFinding] = field(default_factory=list)
    normalized: list[DeduplicationFinding] = field(default_factory=list)
    near: list[DeduplicationFinding] = field(default_factory=list)
    duplicate_source_ids: dict[str, list[str]] = field(default_factory=dict)
    near_method: str = ""
    near_threshold: float = 0.0
    near_ngram: int = DEFAULT_NGRAM
    near_require_same_answer: bool = True

    # -- derived views ----------------------------------------------------- #

    @property
    def findings(self) -> list[DeduplicationFinding]:
        """All duplicate findings, exact first, then normalized, then near."""
        return [*self.exact, *self.normalized, *self.near]

    @property
    def redundant_example_ids(self) -> list[str]:
        """Ids droppable from *exact* clusters while keeping one copy of each.

        Deliberately restricted to the exact stage. Dropping a heuristic
        near-duplicate on the tool's say-so would silently delete a legitimate
        example, so near clusters are surfaced for review and never auto-resolved.
        The first member of each cluster (input order) is the survivor.
        """
        return [eid for finding in self.exact for eid in finding.example_ids[1:]]

    @property
    def n_duplicate_examples(self) -> int:
        """Distinct examples implicated in at least one duplicate cluster."""
        return len({eid for f in self.findings for eid in f.example_ids})

    def counts(self) -> dict[str, int]:
        return {
            "examples": self.n_examples,
            "exact_clusters": len(self.exact),
            "normalized_clusters": len(self.normalized),
            "near_clusters": len(self.near),
            "duplicate_sources": len(self.duplicate_source_ids),
            "redundant_examples": len(self.redundant_example_ids),
            "examples_in_clusters": self.n_duplicate_examples,
        }

    def to_dict(self) -> dict:
        """A plain-dict view suitable for ``json.dumps`` and for ``AuditResult``."""
        return {
            "counts": self.counts(),
            "near_params": {
                "method": self.near_method,
                "threshold": self.near_threshold,
                "ngram": self.near_ngram,
                "require_same_answer": self.near_require_same_answer,
            },
            "exact_duplicates": [f.model_dump(mode="json") for f in self.exact],
            "normalized_duplicates": [f.model_dump(mode="json") for f in self.normalized],
            "near_duplicates": [f.model_dump(mode="json") for f in self.near],
            "duplicate_source_ids": self.duplicate_source_ids,
        }


def find_duplicates(
    examples: Iterable[DatasetExample],
    *,
    near_method: str = "auto",
    near_threshold: float | None = None,
    near_ngram: int = DEFAULT_NGRAM,
    near_require_same_answer: bool = True,
    include_near: bool = True,
) -> DedupReport:
    """Run the full deduplication ladder over ``examples``.

    Deterministic and offline: the same input always yields the same clusters in
    the same order. ``near_method="auto"`` prefers TF-IDF cosine when scikit-learn
    is importable and falls back to pure-Python Jaccard otherwise. Pass
    ``include_near=False`` to skip the heuristic stage and keep only findings that
    cannot be false positives, or ``near_require_same_answer=False`` to let the
    heuristic stage pair up examples whose gold answers differ (needed for
    retrieval-style data whose answers are document ids).
    """
    examples = list(examples)
    resolved = resolve_near_method(near_method)
    threshold = DEFAULT_THRESHOLDS[resolved] if near_threshold is None else near_threshold

    near_findings = (
        near_duplicates(
            examples,
            method=resolved,
            threshold=threshold,
            ngram=near_ngram,
            require_same_answer=near_require_same_answer,
        )
        if include_near
        else []
    )

    return DedupReport(
        n_examples=len(examples),
        exact=exact_duplicates(examples),
        normalized=normalized_duplicates(examples),
        near=near_findings,
        duplicate_source_ids=duplicate_sources(examples),
        near_method=resolved,
        near_threshold=threshold,
        near_ngram=near_ngram,
        near_require_same_answer=near_require_same_answer,
    )


def format_report(report: DedupReport, max_items: int = 20) -> str:
    """Render a human-readable summary of a :class:`DedupReport`."""
    counts = report.counts()
    lines = [
        f"dedup: {counts['examples']} examples",
        f"  exact clusters      : {counts['exact_clusters']}",
        f"  normalized clusters : {counts['normalized_clusters']}",
        f"  near clusters       : {counts['near_clusters']} "
        f"({report.near_method}, word_{report.near_ngram}gram, "
        f"threshold {report.near_threshold:g})",
        f"  duplicate sources   : {counts['duplicate_sources']}",
        f"  redundant examples  : {counts['redundant_examples']} (exact clusters only)",
    ]

    def section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title} ({len(items)}):")
        for item in items[:max_items]:
            lines.append(f"  - {item}")
        if len(items) > max_items:
            lines.append(f"  … and {len(items) - max_items} more")

    def describe(finding: DeduplicationFinding) -> str:
        ids = ", ".join(finding.example_ids)
        score = "" if finding.similarity is None else f" (sim {finding.similarity:.3f})"
        return f"[{ids}]{score} via {finding.method}"

    section("exact duplicates", [describe(f) for f in report.exact])
    section("normalized duplicates", [describe(f) for f in report.normalized])
    section("near duplicates", [describe(f) for f in report.near])
    section(
        "duplicate source records",
        [f"{sid}: {', '.join(ids)}" for sid, ids in sorted(report.duplicate_source_ids.items())],
    )
    return "\n".join(lines)


__all__ = [
    # orchestration
    "DedupReport",
    "find_duplicates",
    "format_report",
    # submodules
    "exact",
    "fuzzy",
    "near_duplicate",
    # re-exports: exact
    "sha256_hex",
    "canonical_output",
    "content_signature",
    "exact_duplicates",
    "duplicate_input_clusters",
    "duplicate_sources",
    # re-exports: fuzzy
    "normalize_text",
    "tokenize",
    "word_ngrams",
    "jaccard",
    "normalized_signature",
    "normalized_duplicates",
    # re-exports: near
    "DEFAULT_NGRAM",
    "DEFAULT_THRESHOLDS",
    "answers_agree",
    "near_duplicates",
    "resolve_near_method",
    "similarity_matrix",
]
