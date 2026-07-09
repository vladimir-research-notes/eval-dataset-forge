"""The audit pipeline over a dataset *as it stands* (Milestone 12).

Milestones 3–9 each built one stage and one report. This module runs them in the order
``docs/PLAN.md`` §4 draws and assembles the single
:class:`~eval_dataset_forge.schemas.AuditResult` that ``audit.json`` serializes. The release
builder (Milestone 10) already did this for the data it was about to *publish*; here the same
assembly runs over a file a curator hands us, before any release exists.

**The audit describes the split it is given, and never builds one.** ``evalforge release``
constructs an assignment and then audits it. ``evalforge audit`` reads whatever ``split`` the
rows carry. Re-splitting inside an audit would mean the report described a dataset that exists
nowhere on disk, and the two commands would disagree about the same file.

**A vacuous check reports vacuity, not health.** Four of the pipeline's signals are defined only
relative to a split: cross-split duplicate overlap, cross-split source reuse, n-gram containment
against the ``train`` reference, and the ``dev``/``test`` duplicate count. Handed a corpus whose
rows are all ``unassigned``, every one of them returns zero — not because the corpus is clean but
because it was never asked a question. A report that printed those zeros beside the substantive
ones would be actively misleading, so :func:`run_audit` counts unassigned rows, raises a warning,
and adds a recommendation naming the checks that were not exercised. This is the same distinction
the dataset card draws between "the check found nothing" and "there is nothing to find", applied
to a check that could not run at all.

**One assembly, two callers.** :func:`assemble_audit` is the only place an ``AuditResult`` is
built. :mod:`eval_dataset_forge.datasets.release` calls it with a :class:`SplitFacts` derived from
the assignment it just constructed; this module calls it with one read off the rows. The
difference between the two is recorded in ``split_statistics["source"]`` (``constructed`` vs
``observed``) rather than hidden, because a reader who cannot tell which one produced a report
cannot tell whether ``residual_overlap: 0`` was earned by a strategy or inherited from a curator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval_dataset_forge.datasets import loader
from eval_dataset_forge.datasets.versioning import content_id
from eval_dataset_forge.dedup import DedupReport, find_duplicates
from eval_dataset_forge.difficulty import DifficultyReport, score_dataset
from eval_dataset_forge.labels import LabelAuditReport, audit_labels
from eval_dataset_forge.leakage import LeakageReport, find_leakage
from eval_dataset_forge.schemas import (
    AuditResult,
    DatasetExample,
    LeakageFinding,
    Split,
    SplitSpec,
)
from eval_dataset_forge.splits import DEFAULT_EVAL_SPLITS, duplicate_blocks
from eval_dataset_forge.splits.stratified import FACETS, distribution
from eval_dataset_forge.validation.pii import scan_examples

# Difficulty calibration runs a permutation test. 2000 resamples put the Monte-Carlo standard
# error on a p near 0.05 at ~0.005 — small against the 0.05 threshold, and cheap. The
# ``difficulty`` CLI defaults to 10000 because it is the command whose whole output is that
# p-value; here it is one line of an audit.
DEFAULT_DIFFICULTY_RESAMPLES = 2_000

# What an ``AuditResult`` records as its version when it describes a raw file rather than a
# release. A release version is a claim about score comparability
# (:mod:`eval_dataset_forge.datasets.versioning`); a file on disk makes no such claim, and
# writing "0.1.0" here would manufacture one.
UNVERSIONED = "unversioned"

# Where an audit run writes its two artifacts. The names match the release bundle's, so a
# ``results/audits/<id>/`` directory and a release directory answer the same questions with the
# same filenames.
AUDIT_JSON = "audit.json"
AUDIT_MD = "audit.md"

# The signals that mean nothing without a split. Named here so the warning can list exactly what
# went unexercised rather than gesturing at "some checks".
SPLIT_DEPENDENT_CHECKS = (
    "leakage.train_test_overlap",
    "leakage.ngram_overlap",
    "leakage.cross_split_sources",
    "splits.eval_duplicates",
)


# --------------------------------------------------------------------------- #
# Split facts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SplitFacts:
    """What the audit needs to know about a split assignment, whichever produced it.

    A :class:`~eval_dataset_forge.splits.SplitReport` knows its own strategy, seed, and requested
    ratios; a dataset read off disk knows none of those and must not pretend to. Both, however,
    can answer the questions the audit asks — how large is each split, what straddles a boundary,
    which duplicated rows sit in ``dev``/``test`` — so the audit depends on this narrow view
    rather than on the builder's report.

    ``source`` distinguishes the two. ``spec`` is ``None`` for an observed assignment: there is no
    strategy that produced it, and inventing a :class:`~eval_dataset_forge.schemas.SplitSpec`
    would put a fabricated seed into an audit artifact.
    """

    source: str
    counts: dict[str, int]
    distributions: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    achieved_ratios: dict[str, float] = field(default_factory=dict)
    residual_overlap: tuple[LeakageFinding, ...] = ()
    residual_sources: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    eval_duplicates: tuple[str, ...] = ()
    quarantined: tuple[str, ...] = ()
    unassigned: tuple[str, ...] = ()
    blocks: tuple[dict, ...] = ()
    spec: SplitSpec | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """The ``split_statistics`` payload of an ``AuditResult``.

        The per-example assignment is deliberately absent: in a release the split files carry it
        verbatim, and in an observed audit ``examples.jsonl`` does.
        """
        return {
            "source": self.source,
            "counts": dict(self.counts),
            "spec": self.spec.model_dump(mode="json") if self.spec else None,
            "achieved_ratios": dict(self.achieved_ratios),
            "distributions": self.distributions,
            "blocks": [dict(block) for block in self.blocks],
            "quarantined": list(self.quarantined),
            "unassigned": list(self.unassigned),
            "residual_overlap": [f.model_dump(mode="json") for f in self.residual_overlap],
            "residual_sources": self.residual_sources,
            "eval_duplicates": list(self.eval_duplicates),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_split_report(cls, report) -> SplitFacts:  # noqa: ANN001 — avoids a circular import
        """Read the facts off a constructed :class:`~eval_dataset_forge.splits.SplitReport`."""
        return cls(
            source="constructed",
            counts=report.counts(),
            distributions=report.distributions,
            achieved_ratios=report.achieved_ratios,
            residual_overlap=tuple(report.residual_overlap),
            residual_sources=report.residual_sources,
            eval_duplicates=tuple(report.eval_duplicates),
            quarantined=tuple(report.quarantined),
            unassigned=(),
            blocks=tuple(block.to_dict() for block in report.blocks if block.size > 1),
            spec=report.spec,
            warnings=tuple(report.warnings),
        )


def observed_split_facts(
    examples: Sequence[DatasetExample],
    dedup: DedupReport,
    leakage: LeakageReport,
) -> SplitFacts:
    """Read the split facts off examples that already carry a ``split``.

    ``residual_overlap`` is taken from the leakage report rather than recomputed: the check that
    asks whether a duplicate cluster straddles a boundary is
    :func:`~eval_dataset_forge.leakage.train_test_overlap.cross_split_clusters`, and running a
    second copy of it here would give the audit two answers to one question.
    """
    assignment = {example.example_id: str(example.split) for example in examples}
    unassigned = tuple(
        example.example_id for example in examples if example.split is Split.UNASSIGNED
    )
    duplicate_ids = {
        example_id for finding in dedup.findings for example_id in finding.example_ids
    }
    blocks = duplicate_blocks(examples, dedup)
    sizes: dict[str, int] = {}
    for split in assignment.values():
        sizes[split] = sizes.get(split, 0) + 1
    eval_duplicates = tuple(
        sorted(eid for eid in duplicate_ids if assignment.get(eid) in DEFAULT_EVAL_SPLITS)
    )

    counts = {
        "examples": len(examples),
        **dict(sorted(sizes.items())),
        "blocks": len(blocks),
        "grouped_blocks": sum(1 for block in blocks if block.size > 1),
        "quarantined": 0,
        "unassigned": len(unassigned),
        "duplicate_examples": len(duplicate_ids),
        "residual_overlap": len(leakage.train_test_overlap),
        "residual_sources": len(leakage.cross_split_sources),
        "eval_duplicates": len(eval_duplicates),
    }
    return SplitFacts(
        source="observed",
        counts=counts,
        distributions={
            facet: distribution(examples, assignment, facet) for facet in FACETS
        },
        achieved_ratios=(
            {name: round(size / len(examples), 4) for name, size in sorted(sizes.items())}
            if examples
            else {}
        ),
        residual_overlap=tuple(leakage.train_test_overlap),
        residual_sources=leakage.cross_split_sources,
        eval_duplicates=eval_duplicates,
        unassigned=unassigned,
        blocks=tuple(block.to_dict() for block in blocks if block.size > 1),
        spec=None,
    )


# --------------------------------------------------------------------------- #
# Dependence
# --------------------------------------------------------------------------- #


def dependence(examples: Sequence[DatasetExample], dedup: DedupReport) -> dict[str, object]:
    """The block structure a bootstrap needs, and no effective sample size.

    ``effective_n`` is deliberately absent: it equals ``n / (1 + (m̄ - 1)·ICC)``, and the
    ICC is a property of a *system's scores* on the blocks, not of the blocks. A dataset
    that quoted one would be quoting an assumed ICC. Milestone 9 measures it per system;
    this records what that measurement needs.

    ``blocks_by_split`` counts the blocks each split *touches*, because a corpus-wide block
    count does not bound a claim about a test score — the test split's own block count does,
    and a dataset card that printed only the corpus number would let a reader spend the wrong
    sample size (Milestone 11). A block whose members straddle a boundary is counted in every
    split it reaches, so the per-split counts sum to at least ``block_count``;
    ``straddling_block_count`` says how far. Under the adversarial strategy it is zero, and
    a non-zero value is the same defect ``residual_overlap`` reports, counted differently.
    """
    blocks = duplicate_blocks(examples, dedup)
    split_of = {example.example_id: str(example.split) for example in examples}
    histogram: dict[str, int] = {}
    blocks_by_split: dict[str, int] = {}
    straddling = 0
    for block in blocks:
        key = str(block.size)
        histogram[key] = histogram.get(key, 0) + 1
        touched = {split_of[example_id] for example_id in block.example_ids}
        straddling += len(touched) > 1
        for name in touched:
            blocks_by_split[name] = blocks_by_split.get(name, 0) + 1
    return {
        "block_count": len(blocks),
        "block_size_histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
        "duplicate_block_count": sum(1 for block in blocks if block.duplicate),
        "blocks_by_split": dict(sorted(blocks_by_split.items())),
        "straddling_block_count": straddling,
        "near_method": dedup.near_method,
        "note": (
            "effective_n is a property of a scored comparison, not of a dataset: it needs an "
            "intraclass correlation, which needs system scores. Resample these blocks, not "
            "the rows — see docs/STATISTICS.md."
        ),
    }


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #


def recommendations(
    facts: SplitFacts,
    dedup: DedupReport,
    labels: LabelAuditReport,
    leakage: LeakageReport,
    difficulty: DifficultyReport,
    block_structure: Mapping[str, object],
    n_examples: int,
) -> list[str]:
    """Prioritized, actionable notes derived from what the stages actually found."""
    notes: list[str] = []

    if facts.unassigned:
        notes.append(
            f"{len(facts.unassigned)} example(s) carry `split=unassigned`, so "
            f"{', '.join(SPLIT_DEPENDENT_CHECKS)} had nothing to compare and returned zero by "
            f"vacuity rather than by cleanliness. Build splits before reading those rows: "
            f"`evalforge build-splits --strategy adversarial`."
        )
    if facts.residual_overlap:
        notes.append(
            f"{len(facts.residual_overlap)} duplicate cluster(s) straddle a split "
            f"boundary: a test score is partly a memorization score. Rebuild with "
            f"--strategy adversarial."
        )
    if leakage.answer_in_prompt:
        notes.append(
            f"{len(leakage.answer_in_prompt)} example(s) contain their gold answer in the "
            f"prompt; they measure copying, not the task."
        )
    if leakage.retrieval_contamination:
        notes.append(
            f"{len(leakage.retrieval_contamination)} retrieval example(s) carry a label "
            f"artifact in a retrieved document."
        )
    n_exact = len(dedup.exact) + len(dedup.normalized)
    if n_exact:
        notes.append(
            f"{n_exact} exact/normalized duplicate cluster(s) remain in the dataset; "
            f"`evalforge release --drop exact,normalized` removes the redundant members and "
            f"records each removal in manifest.excluded."
        )
    if facts.eval_duplicates:
        notes.append(
            f"{len(facts.eval_duplicates)} duplicate-cluster example(s) sit in "
            f"dev/test: they spend evaluation slots on repeated items and narrow every "
            f"confidence interval computed from them."
        )
    blocking = labels.blocking_example_ids
    if blocking and not facts.quarantined:
        notes.append(
            f"{len(blocking)} example(s) have a missing, conflicting, or unadjudicated gold "
            f"answer; `evalforge release --drop blocking-labels` removes them, "
            f"`--quarantine-blocking` keeps them out of dev/test without deleting them."
        )
    elif blocking:
        notes.append(
            f"{len(blocking)} example(s) still lack a settled gold answer. They are "
            f"quarantined in train, not deleted, so an adjudication pass can restore them "
            f"to the evaluation splits without rebuilding the corpus."
        )
    if not difficulty.trustworthy:
        notes.append(
            "the difficulty heuristic's ranking is not distinguishable from a random "
            "ranking of the declared tags on this corpus; do not read its buckets as "
            "difficulty labels (docs/DIFFICULTY.md)."
        )
    block_count = int(block_structure["block_count"])  # type: ignore[arg-type]
    if block_count < n_examples:
        notes.append(
            f"{n_examples} examples form {block_count} independent block(s); a bootstrap "
            f"that resamples rows will under-cover. Use `evalforge stats --resample-unit "
            f"block` (the default)."
        )
    return notes


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def assemble_audit(
    dataset_id: str,
    version: str,
    examples: Sequence[DatasetExample],
    *,
    dedup: DedupReport,
    labels: LabelAuditReport,
    leakage: LeakageReport,
    difficulty: DifficultyReport,
    facts: SplitFacts,
    block_structure: Mapping[str, object],
) -> AuditResult:
    """Assemble an ``AuditResult`` from the stage reports. The only place one is built.

    ``audit_id`` is derived from the audited content, so the same data always produces the same
    audit id and a report can be matched to the rows it describes without trusting a filename.
    ``AuditResult`` has two deduplication buckets and the ladder has three rungs: ``normalized``
    findings are filed under ``near_duplicates`` and keep their ``kind``, so nothing is lost and
    nothing is relabelled.
    """
    split_stats = facts.to_dict()
    split_stats["dependence"] = dict(block_structure)

    return AuditResult(
        audit_id="audit-" + content_id(examples).split(":", 1)[1][:16],
        dataset_id=dataset_id,
        version=version,
        exact_duplicates=list(dedup.exact),
        near_duplicates=[*dedup.normalized, *dedup.near],
        leakage_flags=leakage.findings,
        label_warnings=labels.findings,
        pii_flags=[asdict(finding) for finding in scan_examples(examples)],
        difficulty_distribution=difficulty.distribution,
        split_statistics=split_stats,
        recommendations=recommendations(
            facts, dedup, labels, leakage, difficulty, block_structure, len(examples)
        ),
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class AuditReport:
    """An audit of one dataset: the assembled result plus every stage report behind it.

    ``result`` is what ``audit.json`` serializes and what a dataset card reads. The stage reports
    are kept because they carry evidence the ``AuditResult`` schema has no field for — duplicate
    *source* groups, the label adjudication queue, the difficulty calibration's p-value — and the
    smoke audit scores against several of them.
    """

    dataset_id: str
    version: str
    result: AuditResult
    dedup: DedupReport
    labels: LabelAuditReport
    leakage: LeakageReport
    difficulty: DifficultyReport
    facts: SplitFacts
    dependence: dict[str, object] = field(default_factory=dict)
    source_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def n_examples(self) -> int:
        return self.dedup.n_examples

    def counts(self) -> dict[str, int]:
        return {"examples": self.n_examples, **self.result.summary()}

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "source_files": list(self.source_files),
            "counts": self.counts(),
            "audit": self.result.model_dump(mode="json"),
            "warnings": list(self.warnings),
        }


def run_audit(
    examples: Iterable[DatasetExample],
    *,
    dataset_id: str,
    version: str = UNVERSIONED,
    near_require_same_answer: bool = True,
    difficulty_resamples: int = DEFAULT_DIFFICULTY_RESAMPLES,
    seed: int = 0,
    source_files: Sequence[str] = (),
) -> AuditReport:
    """Run every audit stage over ``examples`` in dependency order. Nothing is written.

    Deterministic, offline, and non-mutating: the examples are audited exactly as handed over,
    including whatever ``split`` they carry. The deduplication ladder runs once and is threaded
    into the leakage checks, which would otherwise recompute it.
    """
    examples = list(examples)
    if not examples:
        raise ValueError("there is nothing to audit: the dataset holds no examples")

    dedup = find_duplicates(examples, near_require_same_answer=near_require_same_answer)
    labels = audit_labels(examples)
    leakage = find_leakage(
        examples, dedup_report=dedup, near_require_same_answer=near_require_same_answer
    )
    difficulty = score_dataset(examples, n_resamples=difficulty_resamples, seed=seed)
    facts = observed_split_facts(examples, dedup, leakage)
    block_structure = dependence(examples, dedup)

    warnings: list[str] = []
    if facts.unassigned:
        warnings.append(
            f"{len(facts.unassigned)} of {len(examples)} example(s) are unassigned; "
            f"{len(SPLIT_DEPENDENT_CHECKS)} split-dependent check(s) reported zero without "
            f"being exercised"
        )
    if not leakage.reference_size:
        warnings.append(
            "the n-gram overlap check had an empty reference corpus (no `train` split), so its "
            "zero means 'nothing to compare against', not 'no contamination'"
        )
    warnings.extend(difficulty.warnings)

    result = assemble_audit(
        dataset_id,
        version,
        examples,
        dedup=dedup,
        labels=labels,
        leakage=leakage,
        difficulty=difficulty,
        facts=facts,
        block_structure=block_structure,
    )
    return AuditReport(
        dataset_id=dataset_id,
        version=version,
        result=result,
        dedup=dedup,
        labels=labels,
        leakage=leakage,
        difficulty=difficulty,
        facts=facts,
        dependence=block_structure,
        source_files=tuple(source_files),
        warnings=tuple(warnings),
    )


def default_dataset_id(path: str | Path) -> str:
    """``datasets/fixtures/qa_toy/examples.jsonl`` → ``qa_toy``.

    A dataset's identity is the directory that holds it, not the filename every dataset shares.
    Falls back to the file stem when the file is not in a directory of its own.
    """
    path = Path(path)
    parent = path.parent.name
    return parent if parent and path.stem in ("examples", "data") else path.stem


def audit_dataset(
    path: str | Path,
    *,
    dataset_id: str = "",
    version: str = UNVERSIONED,
    **kwargs,
) -> AuditReport:
    """Load a dataset file and audit it. Raises if the file does not parse or validate."""
    examples = loader.load_examples(path)
    return run_audit(
        examples,
        dataset_id=dataset_id or default_dataset_id(path),
        version=version,
        source_files=(str(path),),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_audit(
    report: AuditReport,
    out_dir: str | Path,
    *,
    html: bool = False,
) -> dict[str, Path]:
    """Write ``audit.json`` and ``audit.md`` under ``out_dir``. Byte-identical across runs."""
    from eval_dataset_forge.datasets.writer import render_json, write_bytes
    from eval_dataset_forge.reports.audit_report import render_html, render_markdown

    out_dir = Path(out_dir)
    written = {
        AUDIT_JSON: write_bytes(
            out_dir / AUDIT_JSON, render_json(report.result.model_dump(mode="json"))
        ),
        AUDIT_MD: write_bytes(out_dir / AUDIT_MD, render_markdown(report.result)),
    }
    if html:
        written["audit.html"] = write_bytes(out_dir / "audit.html", render_html(report.result))
    return written


def format_report(report: AuditReport, max_items: int = 10) -> str:
    """Render an :class:`AuditReport` for a terminal."""
    facts = report.facts
    # ``unassigned`` is printed only when non-zero: a corpus with none does not need a line
    # saying so, and a corpus with some has a warning about it below.
    sizes = ", ".join(
        f"{name}={facts.counts[name]}"
        for name in (str(split) for split in Split)
        if facts.counts.get(name) or (name != str(Split.UNASSIGNED) and name in facts.counts)
    )
    lines = [
        f"audit: {report.dataset_id} @ {report.version} ({report.result.audit_id})",
        f"  examples     : {report.n_examples} ({sizes or 'no splits assigned'})",
        f"  splits       : {facts.source}",
        f"  blocks       : {report.dependence.get('block_count', 'n/a')} independent "
        f"({report.dependence.get('duplicate_block_count', 0)} contain a duplicate cluster)",
        f"  near method  : {report.dedup.near_method}",
        "  findings     : "
        + ", ".join(f"{key}={value}" for key, value in report.result.summary().items()),
    ]

    def section(title: str, items: Sequence[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title} ({len(items)}):")
        lines.extend(f"  - {item}" for item in items[:max_items])
        if len(items) > max_items:
            lines.append(f"  … and {len(items) - max_items} more")

    section("recommendations", report.result.recommendations)
    section("warnings", list(report.warnings))
    return "\n".join(lines)


__all__ = [
    "AUDIT_JSON",
    "AUDIT_MD",
    "DEFAULT_DIFFICULTY_RESAMPLES",
    "SPLIT_DEPENDENT_CHECKS",
    "UNVERSIONED",
    "AuditReport",
    "SplitFacts",
    "assemble_audit",
    "audit_dataset",
    "default_dataset_id",
    "dependence",
    "format_report",
    "observed_split_facts",
    "recommendations",
    "run_audit",
    "write_audit",
]
