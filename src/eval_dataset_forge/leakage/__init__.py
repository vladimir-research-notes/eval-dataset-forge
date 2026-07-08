"""Leakage & contamination layer (Milestone 5).

Audit dimension #4: *can a system score well here without doing the task?* Every
check below answers one version of that question, and each emits exactly one
:class:`~eval_dataset_forge.schemas.LeakageKind` — a 1:1 mapping between module and
kind that ``tests/test_leakage.py`` enforces.

==========================  ==========================================
module                      catches
==========================  ==========================================
:mod:`.train_test_overlap`  the same item in two splits
:mod:`.answer_in_prompt`    the gold answer quoted in its own prompt
:mod:`.ngram_overlap`       a prompt verbatim in a reference corpus
:mod:`.retrieval_overlap`   annotation artifacts; shared gold passages
==========================  ==========================================

Certainty differs. :mod:`.retrieval_overlap` is exact; :mod:`.ngram_overlap` is exact
*given* its n-gram width, which is a choice; :mod:`.answer_in_prompt` is heuristic;
and :mod:`.train_test_overlap` has none of its own — it filters the deduplication
ladder's clusters down to those crossing a split boundary, so an exact duplicate
yields an exact finding and a near-duplicate a heuristic one. Its blind spots are the
ladder's blind spots.

The four checks are **not** disjoint, and unlike the dedup ladder they are not
meant to be. A single defect can and should light up several channels: a test
example copied verbatim from train is a cross-split exact duplicate *and* a full
n-gram overlap against the train reference. Agreement between independent channels
is corroboration, not double counting — which is why :meth:`LeakageReport.counts`
reports per-kind counts alongside the number of distinct examples implicated.

Nothing is removed and nothing is a verdict. A prompt may legitimately restate its
answer; a benchmark may legitimately reuse a passage. :func:`find_leakage` produces
evidence for a reviewer, and the split builder (Milestone 7) decides what to do
about it.

**The honest boundary.** Contamination is only measurable against text we are
handed. A model's pretraining corpus is not ours to read, so a clean report says
nothing about it. Absence of a flag here is not evidence of no contamination.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from eval_dataset_forge.dedup import DedupReport, find_duplicates
from eval_dataset_forge.dedup.near_duplicate import DEFAULT_NGRAM as DEFAULT_NEAR_NGRAM
from eval_dataset_forge.leakage.answer_in_prompt import (
    DEFAULT_NGRAM as DEFAULT_ANSWER_NGRAM,
)
from eval_dataset_forge.leakage.answer_in_prompt import (
    DEFAULT_THRESHOLD as DEFAULT_ANSWER_THRESHOLD,
)
from eval_dataset_forge.leakage.answer_in_prompt import (
    answer_candidates,
    label_vocabulary,
    leaked_answers,
)
from eval_dataset_forge.leakage.ngram_overlap import (
    DEFAULT_MIN_CONTAINMENT,
    ReferenceIndex,
    containment,
    longest_shared_run,
    reference_overlap,
    split_reference,
    split_reference_overlap,
)
from eval_dataset_forge.leakage.ngram_overlap import (
    DEFAULT_NGRAM as DEFAULT_REFERENCE_NGRAM,
)
from eval_dataset_forge.leakage.retrieval_overlap import (
    DocRef,
    cross_split_documents,
    document_clusters,
    label_artifacts,
    retrieval_contamination,
    retrieval_docs,
)
from eval_dataset_forge.leakage.train_test_overlap import (
    ASSIGNED_SPLITS,
    cross_split_clusters,
    cross_split_sources,
    split_counts,
)
from eval_dataset_forge.schemas import DatasetExample, LeakageFinding, Split


@dataclass
class LeakageReport:
    """Everything the four leakage checks found on one dataset.

    The parameter fields record what actually ran — the resolved near-duplicate
    method after any scikit-learn fallback, the n-gram widths, the thresholds, and
    the label vocabulary the answer check chose to ignore — so a report explains
    itself without the reader having to reconstruct the call.
    """

    n_examples: int
    train_test_overlap: list[LeakageFinding] = field(default_factory=list)
    answer_in_prompt: list[LeakageFinding] = field(default_factory=list)
    ngram_overlap: list[LeakageFinding] = field(default_factory=list)
    retrieval_contamination: list[LeakageFinding] = field(default_factory=list)
    cross_split_sources: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    duplicate_documents: list[list[str]] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    # -- parameters actually used ------------------------------------------ #
    near_method: str = ""
    answer_ngram: int = DEFAULT_ANSWER_NGRAM
    answer_threshold: float = DEFAULT_ANSWER_THRESHOLD
    ignored_answers: list[str] = field(default_factory=list)
    reference_ngram: int = DEFAULT_REFERENCE_NGRAM
    reference_split: str = str(Split.TRAIN)
    target_splits: list[str] = field(
        default_factory=lambda: [str(Split.DEV), str(Split.TEST)]
    )
    reference_size: int = 0

    # -- derived views ----------------------------------------------------- #

    @property
    def findings(self) -> list[LeakageFinding]:
        """All leakage findings, ordered by check: overlap, answer, n-gram, retrieval."""
        return [
            *self.train_test_overlap,
            *self.answer_in_prompt,
            *self.ngram_overlap,
            *self.retrieval_contamination,
        ]

    @property
    def n_leaking_examples(self) -> int:
        """Distinct examples implicated in at least one finding.

        Always ``<=`` the total finding count, and usually strictly less: the
        channels overlap by design, so one contaminated example is named by several.
        """
        return len({eid for finding in self.findings for eid in finding.example_ids})

    def counts(self) -> dict[str, int]:
        return {
            "examples": self.n_examples,
            "train_test_overlap": len(self.train_test_overlap),
            "answer_in_prompt": len(self.answer_in_prompt),
            "ngram_overlap": len(self.ngram_overlap),
            "retrieval_contamination": len(self.retrieval_contamination),
            "cross_split_sources": len(self.cross_split_sources),
            "duplicate_documents": len(self.duplicate_documents),
            "leaking_examples": self.n_leaking_examples,
        }

    def to_dict(self) -> dict:
        """A plain-dict view suitable for ``json.dumps`` and for ``AuditResult``."""
        return {
            "counts": self.counts(),
            "params": {
                "near_method": self.near_method,
                "answer_ngram": self.answer_ngram,
                "answer_threshold": self.answer_threshold,
                "ignored_answers": self.ignored_answers,
                "reference_ngram": self.reference_ngram,
                "reference_split": self.reference_split,
                "reference_size": self.reference_size,
                "target_splits": self.target_splits,
            },
            "split_counts": self.split_counts,
            "train_test_overlap": [f.model_dump(mode="json") for f in self.train_test_overlap],
            "answer_in_prompt": [f.model_dump(mode="json") for f in self.answer_in_prompt],
            "ngram_overlap": [f.model_dump(mode="json") for f in self.ngram_overlap],
            "retrieval_contamination": [
                f.model_dump(mode="json") for f in self.retrieval_contamination
            ],
            "cross_split_sources": self.cross_split_sources,
            "duplicate_documents": self.duplicate_documents,
        }


def find_leakage(
    examples: Iterable[DatasetExample],
    *,
    dedup_report: DedupReport | None = None,
    near_method: str = "auto",
    near_ngram: int = DEFAULT_NEAR_NGRAM,
    near_require_same_answer: bool = True,
    answer_ngram: int = DEFAULT_ANSWER_NGRAM,
    answer_threshold: float = DEFAULT_ANSWER_THRESHOLD,
    ignore_answers: frozenset[str] | None = None,
    reference: ReferenceIndex | None = None,
    reference_split: str = Split.TRAIN,
    target_splits: Sequence[str] = (Split.DEV, Split.TEST),
    reference_ngram: int = DEFAULT_REFERENCE_NGRAM,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
) -> LeakageReport:
    """Run all four leakage checks over ``examples``.

    Deterministic and offline. ``dedup_report`` lets the audit pipeline reuse the
    ladder it already ran instead of paying for it twice; ``reference`` swaps the
    default in-dataset reference corpus (the ``train`` split) for an external one.

    ``ignore_answers=None`` means "compute the dataset's label vocabulary and skip
    it" — see :mod:`.answer_in_prompt`. The chosen vocabulary is recorded on the
    report; pass ``frozenset()`` to check every gold answer.
    """
    examples = list(examples)
    if dedup_report is None:
        dedup_report = find_duplicates(
            examples,
            near_method=near_method,
            near_ngram=near_ngram,
            near_require_same_answer=near_require_same_answer,
        )

    if ignore_answers is None:
        ignore_answers = label_vocabulary(examples)

    if reference is None:
        reference = split_reference(examples, reference_split)
    targets = [ex for ex in examples if ex.split in target_splits]

    return LeakageReport(
        n_examples=len(examples),
        train_test_overlap=cross_split_clusters(examples, dedup_report),
        answer_in_prompt=leaked_answers(
            examples,
            ngram=answer_ngram,
            threshold=answer_threshold,
            ignore_answers=ignore_answers,
        ),
        ngram_overlap=reference_overlap(
            targets,
            reference,
            ngram=reference_ngram,
            min_containment=min_containment,
        ),
        retrieval_contamination=retrieval_contamination(examples),
        cross_split_sources=cross_split_sources(examples, dedup_report),
        duplicate_documents=[
            [str(ref) for ref in refs] for refs in document_clusters(examples)
        ],
        split_counts=split_counts(examples),
        near_method=dedup_report.near_method,
        answer_ngram=answer_ngram,
        answer_threshold=answer_threshold,
        ignored_answers=sorted(ignore_answers),
        reference_ngram=reference_ngram,
        reference_split=str(reference_split),
        target_splits=[str(split) for split in target_splits],
        reference_size=len(reference),
    )


def format_report(report: LeakageReport, max_items: int = 20) -> str:
    """Render a human-readable summary of a :class:`LeakageReport`."""
    counts = report.counts()
    lines = [
        f"leakage: {counts['examples']} examples "
        f"({', '.join(f'{k}={v}' for k, v in sorted(report.split_counts.items()))})",
        f"  train/test overlap      : {counts['train_test_overlap']} "
        f"(dedup ladder, near via {report.near_method})",
        f"  answer in prompt        : {counts['answer_in_prompt']} "
        f"(word_{report.answer_ngram}gram containment >= {report.answer_threshold:g})",
        f"  n-gram overlap          : {counts['ngram_overlap']} "
        f"(word_{report.reference_ngram}gram vs {report.reference_split}, "
        f"{report.reference_size} reference texts)",
        f"  retrieval contamination : {counts['retrieval_contamination']}",
        f"  cross-split sources     : {counts['cross_split_sources']} "
        f"(dependence, not duplication)",
        f"  examples implicated     : {counts['leaking_examples']}",
    ]
    if report.ignored_answers:
        lines.append(
            f"  ignored label vocabulary: {', '.join(repr(a) for a in report.ignored_answers)}"
        )

    def section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title} ({len(items)}):")
        for item in items[:max_items]:
            lines.append(f"  - {item}")
        if len(items) > max_items:
            lines.append(f"  … and {len(items) - max_items} more")

    def describe(finding: LeakageFinding) -> str:
        ids = ", ".join(finding.example_ids)
        score = "" if finding.score is None else f" (score {finding.score:.3f})"
        return f"[{ids}]{score} {finding.evidence or ''}".rstrip()

    section("train/test overlap", [describe(f) for f in report.train_test_overlap])
    section("answer in prompt", [describe(f) for f in report.answer_in_prompt])
    section("n-gram overlap", [describe(f) for f in report.ngram_overlap])
    section("retrieval contamination", [describe(f) for f in report.retrieval_contamination])
    section(
        "cross-split source records",
        [
            f"{source_id}: " + " ".join(f"{split}={ids}" for split, ids in sorted(spread.items()))
            for source_id, spread in sorted(report.cross_split_sources.items())
        ],
    )
    section("duplicate documents", [" ↔ ".join(refs) for refs in report.duplicate_documents])
    return "\n".join(lines)


__all__ = [
    # orchestration
    "LeakageReport",
    "find_leakage",
    "format_report",
    # shared constants
    "ASSIGNED_SPLITS",
    "DEFAULT_ANSWER_NGRAM",
    "DEFAULT_ANSWER_THRESHOLD",
    "DEFAULT_MIN_CONTAINMENT",
    "DEFAULT_REFERENCE_NGRAM",
    # re-exports: train/test overlap
    "cross_split_clusters",
    "cross_split_sources",
    "split_counts",
    # re-exports: answer in prompt
    "answer_candidates",
    "label_vocabulary",
    "leaked_answers",
    # re-exports: n-gram overlap
    "ReferenceIndex",
    "containment",
    "longest_shared_run",
    "reference_overlap",
    "split_reference",
    "split_reference_overlap",
    # re-exports: retrieval
    "DocRef",
    "cross_split_documents",
    "document_clusters",
    "label_artifacts",
    "retrieval_contamination",
    "retrieval_docs",
]
