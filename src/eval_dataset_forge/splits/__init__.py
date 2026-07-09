"""Split construction (Milestone 7).

Audit dimension #7: *does the arrangement of this dataset let a system score well
without doing the task?* The earlier stages find defects and refuse to act on them.
This one acts: it decides which examples a benchmark evaluates on, and it is the first
stage whose output changes the data rather than describing it.

Five strategies, each a choice of which dependence between examples the split refuses
to cut through:

====================  =================================  ==========================
strategy              indivisible unit                   channel it closes
====================  =================================  ==========================
:mod:`.random`        one example                        none (the i.i.d. baseline)
:mod:`.stratified`    one example                        distribution shift by facet
:mod:`.random`        a ``source_id`` group              provenance dependence
:mod:`.adversarial`   a duplicate cluster ∪ its sources  content dependence
:mod:`.temporal`      one example, in time order         (orders, does not group)
====================  =================================  ==========================

**Every build is verified, not asserted.** :func:`build_splits` applies the assignment
and then re-runs Milestone 5's cross-split checks against the splits it just produced,
so ``SplitReport.residual_overlap`` is a measurement of the output, not a restatement
of the strategy's intent. That is what makes the comparison in ``docs/SPLITS.md``
meaningful — and what makes the adversarial strategy's zero worth believing.

**The result worth reading twice.** The grouping key a dataset hands you is not the
one that closes the leak. Every planted duplicate in the fixtures carries a *fresh*
``source_id`` (``src-fr-capital`` → ``src-fr-capital-dup``, ``src-fizzbuzz`` →
``src-fizzbuzz-para``) — exactly as a real corpus records a paraphrase written as a
new record. So ``source-grouped``, whose whole purpose is to stop provenance leakage,
leaves **every one of them** free to straddle train and test, and measures no better
than ``random``. Only blocks derived from example *content* close the channel.
Provenance metadata describes where an example came from; it does not know what the
example says.

Nothing is deleted, here or anywhere upstream. Duplicate and label-blocking examples
are moved to ``train``, never dropped: removing an example is a decision for the
release builder (Milestone 10), which records it in a manifest a reader can audit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eval_dataset_forge.dedup import DedupReport, find_duplicates
from eval_dataset_forge.leakage.train_test_overlap import (
    cross_split_clusters,
    cross_split_sources,
)
from eval_dataset_forge.schemas import (
    DatasetExample,
    LeakageFinding,
    Split,
    SplitSpec,
    SplitStrategy,
)
from eval_dataset_forge.splits import adversarial, assignment, blocks, random, stratified, temporal
from eval_dataset_forge.splits.adversarial import DEFAULT_EVAL_SPLITS, adversarial_splits
from eval_dataset_forge.splits.assignment import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    SPLIT_ORDER,
    Assignment,
    apportion,
    assign_blocks,
    assign_in_order,
    normalize_ratios,
)
from eval_dataset_forge.splits.blocks import (
    Block,
    block_of_example,
    duplicate_blocks,
    singleton_blocks,
    source_blocks,
)
from eval_dataset_forge.splits.random import random_splits, source_grouped_splits
from eval_dataset_forge.splits.stratified import (
    DEFAULT_STRATIFY_BY,
    FACETS,
    UNLABELLED,
    block_stratum,
    declared_bucket,
    distribution,
    facet_value,
    stratified_splits,
    stratum_of,
)
from eval_dataset_forge.splits.temporal import parse_timestamp, temporal_order, temporal_splits

# Which group key each strategy pins together, recorded on the SplitSpec so a release
# can say what its splits protect without re-reading this module.
_GROUP_KEY: dict[SplitStrategy, str | None] = {
    SplitStrategy.RANDOM: None,
    SplitStrategy.STRATIFIED: None,
    SplitStrategy.SOURCE_GROUPED: "source_id",
    SplitStrategy.ADVERSARIAL: "duplicate_cluster",
    SplitStrategy.TEMPORAL: None,
}


@dataclass
class SplitReport:
    """A built split, its rationale, and the leakage it did or did not leave behind.

    ``residual_overlap`` and ``residual_sources`` come from re-running Milestone 5's
    cross-split checks over the produced assignment, so they describe what was built
    rather than what was intended. ``achieved`` ratios are reported instead of the
    requested ones because indivisible blocks and small strata routinely make the
    request unreachable, and forcing it would mean cutting a block.
    """

    spec: SplitSpec
    n_examples: int
    assignment: dict[str, str] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)
    duplicate_example_ids: tuple[str, ...] = ()
    residual_overlap: list[LeakageFinding] = field(default_factory=list)
    residual_sources: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    distributions: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    quarantined: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    # -- derived views ----------------------------------------------------- #

    @property
    def split_sizes(self) -> dict[str, int]:
        sizes = dict.fromkeys(self.spec.ratios, 0)
        for split in self.assignment.values():
            sizes[split] = sizes.get(split, 0) + 1
        return sizes

    @property
    def achieved_ratios(self) -> dict[str, float]:
        """Realized fraction per split. Compare against ``spec.ratios``."""
        if not self.n_examples:
            return {}
        return {
            name: round(size / self.n_examples, 4)
            for name, size in self.split_sizes.items()
        }

    @property
    def grouped_blocks(self) -> list[Block]:
        """Blocks holding more than one example — the groups a strategy kept intact."""
        return [block for block in self.blocks if block.size > 1]

    @property
    def eval_duplicates(self) -> list[str]:
        """Examples in ``dev`` or ``test`` that belong to a duplicate cluster.

        Read off the deduplication report, never off the blocks: only the adversarial
        strategy *builds* duplicate blocks, so a block-derived count would report a
        flattering zero for the four strategies that never looked. The definition is
        fixed at ``dev`` ∪ ``test`` for the same reason — so the number means the same
        thing in every row of the comparison table.

        Distinct from ``residual_overlap``, which counts clusters *straddling* a
        boundary. A cluster sitting entirely inside ``test`` leaks nothing across
        splits but still makes two correlated examples look like two independent ones,
        which quietly narrows every confidence interval Milestone 9 will compute.
        """
        return sorted(
            example_id
            for example_id in self.duplicate_example_ids
            if self.assignment.get(example_id) in DEFAULT_EVAL_SPLITS
        )

    def counts(self) -> dict[str, int]:
        return {
            "examples": self.n_examples,
            **self.split_sizes,
            "blocks": len(self.blocks),
            "grouped_blocks": len(self.grouped_blocks),
            "quarantined": len(self.quarantined),
            "duplicate_examples": len(self.duplicate_example_ids),
            "residual_overlap": len(self.residual_overlap),
            "residual_sources": len(self.residual_sources),
            "eval_duplicates": len(self.eval_duplicates),
        }

    def to_dict(self) -> dict:
        """A plain-dict view suitable for ``json.dumps`` and for ``AuditResult``."""
        return {
            "counts": self.counts(),
            "spec": self.spec.model_dump(mode="json"),
            "achieved_ratios": self.achieved_ratios,
            "distributions": self.distributions,
            "assignment": dict(sorted(self.assignment.items())),
            "blocks": [block.to_dict() for block in self.blocks if block.size > 1],
            "quarantined": list(self.quarantined),
            "duplicate_example_ids": list(self.duplicate_example_ids),
            "residual_overlap": [f.model_dump(mode="json") for f in self.residual_overlap],
            "residual_sources": self.residual_sources,
            "eval_duplicates": self.eval_duplicates,
            "warnings": list(self.warnings),
        }


def apply_splits(
    examples: Iterable[DatasetExample], assigned: dict[str, str]
) -> list[DatasetExample]:
    """Return copies of ``examples`` carrying their new ``split``.

    The inputs are never mutated: an audit that silently rewrote the dataset it was
    handed would make every later stage's report un-reproducible from the source file.
    An example missing from ``assigned`` keeps the split it arrived with.
    """
    return [
        example.model_copy(update={"split": Split(assigned[example.example_id])})
        if example.example_id in assigned
        else example.model_copy()
        for example in examples
    ]


def verify_splits(
    examples: Iterable[DatasetExample], dedup_report: DedupReport
) -> tuple[list[LeakageFinding], dict[str, dict[str, list[str]]]]:
    """Re-run the cross-split leakage checks over already-assigned examples.

    The independent check that closes the loop: it knows nothing about which strategy
    produced the assignment, and reuses Milestone 5's code unchanged. Its blind spots
    are that milestone's — a paraphrase the dedup ladder misses is invisible here too,
    so a zero means "no *detectable* cross-split duplicate", never "no duplicate".
    """
    examples = list(examples)
    return (
        cross_split_clusters(examples, dedup_report),
        cross_split_sources(examples, dedup_report),
    )


def _coverage_warnings(
    examples: list[DatasetExample],
    distributions: dict[str, dict[str, dict[str, int]]],
    ratios: dict[str, float],
) -> list[str]:
    """Facet values the corpus has and the ``test`` split does not.

    A benchmark whose test set contains no ``hard`` example cannot measure difficulty,
    and nothing else in the report says so out loud: the per-stratum quota warnings stay
    silent because each stratum's ``test`` quota rounded to zero honestly. Restricted to
    ``test`` — the split whose composition is a claim about what the benchmark measures.
    Silent when no test split was asked for, since then there is no claim to check.
    """
    if str(Split.TEST) not in ratios:
        return []
    warnings = []
    for facet, table in sorted(distributions.items()):
        present = set(table.get(str(Split.TEST), {}))
        corpus: dict[str, int] = {}
        for example in examples:
            value = facet_value(example, facet)
            corpus[value] = corpus.get(value, 0) + 1
        for value in sorted(set(corpus) - present):
            warnings.append(
                f"test contains no {facet}={value!r} example ({corpus[value]} in the corpus); "
                f"the split cannot measure it"
            )
    return warnings


def _assign(
    strategy: SplitStrategy,
    examples: list[DatasetExample],
    *,
    ratios: dict[str, float],
    seed: int,
    stratify_by: Sequence[str],
    dedup_report: DedupReport,
    eval_splits: Sequence[str],
    quarantine_ids: Sequence[str],
) -> Assignment:
    if strategy is SplitStrategy.RANDOM:
        return random_splits(examples, ratios=ratios, seed=seed)
    if strategy is SplitStrategy.STRATIFIED:
        return stratified_splits(examples, ratios=ratios, seed=seed, stratify_by=stratify_by)
    if strategy is SplitStrategy.SOURCE_GROUPED:
        return source_grouped_splits(examples, ratios=ratios, seed=seed)
    if strategy is SplitStrategy.ADVERSARIAL:
        return adversarial_splits(
            examples,
            ratios=ratios,
            seed=seed,
            stratify_by=stratify_by,
            dedup_report=dedup_report,
            eval_splits=eval_splits,
            quarantine_ids=quarantine_ids,
        )
    if strategy is SplitStrategy.TEMPORAL:
        return temporal_splits(examples, ratios=ratios)
    raise ValueError(f"unknown split strategy {strategy!r}")  # pragma: no cover


def build_splits(
    examples: Iterable[DatasetExample],
    *,
    strategy: SplitStrategy | str = SplitStrategy.ADVERSARIAL,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    stratify_by: Sequence[str] = DEFAULT_STRATIFY_BY,
    dedup_report: DedupReport | None = None,
    eval_splits: Sequence[str] = DEFAULT_EVAL_SPLITS,
    quarantine_ids: Iterable[str] = (),
) -> SplitReport:
    """Build splits with ``strategy`` and verify the result against the leakage checks.

    Deterministic and offline: the same examples, strategy, ratios, and seed always
    produce the same assignment. ``temporal`` ignores the seed — its order comes from
    the data.

    ``dedup_report`` is reused when supplied. It is needed by every strategy, not only
    the adversarial one, because the residual-leakage verification runs on all of
    them; supply the audit pipeline's report to avoid paying for the ladder twice, and
    supply one built with ``near_require_same_answer=False`` for retrieval-style data.

    ``quarantine_ids`` — typically Milestone 6's ``LabelAuditReport.blocking_example_ids``
    — keeps examples whose gold answer needs human adjudication out of the evaluation
    splits. Honoured by the adversarial strategy only: the other four make no claim
    about what belongs in an evaluation split, and silently applying the constraint
    would make ``random`` not random.
    """
    examples = list(examples)
    strategy = SplitStrategy(strategy)
    ratios = normalize_ratios(ratios)
    quarantine_ids = tuple(quarantine_ids)
    if dedup_report is None:
        dedup_report = find_duplicates(examples)

    result = _assign(
        strategy,
        examples,
        ratios=ratios,
        seed=seed,
        stratify_by=stratify_by,
        dedup_report=dedup_report,
        eval_splits=eval_splits,
        quarantine_ids=quarantine_ids,
    )

    assigned = apply_splits(examples, result.of_example)
    overlap, sources = verify_splits(assigned, dedup_report)
    distributions = {
        facet: distribution(examples, result.of_example, facet) for facet in FACETS
    }

    spec = SplitSpec(
        strategy=strategy,
        seed=seed,
        ratios=ratios,
        group_key=_GROUP_KEY[strategy],
        constraints={
            "stratify_by": list(stratify_by)
            if strategy in (SplitStrategy.STRATIFIED, SplitStrategy.ADVERSARIAL)
            else [],
            "eval_splits": list(eval_splits),
            "quarantined": list(quarantine_ids),
            "near_method": dedup_report.near_method,
        },
    )

    return SplitReport(
        spec=spec,
        n_examples=len(examples),
        assignment=dict(result.of_example),
        blocks=list(result.blocks),
        duplicate_example_ids=tuple(
            sorted({eid for finding in dedup_report.findings for eid in finding.example_ids})
        ),
        residual_overlap=overlap,
        residual_sources=sources,
        distributions=distributions,
        quarantined=quarantine_ids,
        warnings=(*result.warnings, *_coverage_warnings(examples, distributions, ratios)),
    )


def write_splits(
    examples: Iterable[DatasetExample], report: SplitReport, out_dir: str | Path
) -> dict[str, Path]:
    """Write ``<out_dir>/splits/{train,dev,test}.jsonl`` and ``split_report.json``.

    Byte-identical across runs: examples keep their input order within each split and
    are serialized field-by-field in schema order. This writes the split files of the
    release layout in ``docs/PLAN.md`` §5D; ``manifest.json``, ``audit.json``, and the
    dataset card are the release builder's (Milestone 10).
    """
    out_dir = Path(out_dir)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    assigned = apply_splits(examples, report.assignment)
    written: dict[str, Path] = {}
    for name in report.spec.ratios:
        path = split_dir / f"{name}.jsonl"
        members = [example for example in assigned if str(example.split) == name]
        path.write_text(
            "".join(
                json.dumps(example.model_dump(mode="json"), ensure_ascii=False) + "\n"
                for example in members
            ),
            encoding="utf-8",
        )
        written[name] = path

    report_path = out_dir / "split_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    written["report"] = report_path
    return written


def format_report(report: SplitReport, max_items: int = 20) -> str:
    """Render a human-readable summary of a :class:`SplitReport`."""
    counts = report.counts()
    requested = ", ".join(f"{k}={v:g}" for k, v in report.spec.ratios.items())
    achieved = ", ".join(f"{k}={v:.3f}" for k, v in report.achieved_ratios.items())
    sizes = ", ".join(f"{k}={v}" for k, v in report.split_sizes.items())
    lines = [
        f"splits: {counts['examples']} examples via {report.spec.strategy} "
        f"(seed {report.spec.seed}, group key {report.spec.group_key or 'none'})",
        f"  sizes               : {sizes}",
        f"  ratios requested    : {requested}",
        f"  ratios achieved     : {achieved}",
        f"  blocks              : {counts['blocks']} "
        f"({counts['grouped_blocks']} holding >1 example)",
        f"  quarantined         : {counts['quarantined']} (kept out of evaluation splits)",
        f"  residual overlap    : {counts['residual_overlap']} "
        f"(cross-split duplicate clusters, near via {report.spec.constraints.get('near_method')})",
        f"  residual sources    : {counts['residual_sources']} (cross-split source records)",
        f"  duplicates in eval  : {counts['eval_duplicates']} "
        f"of {counts['duplicate_examples']} duplicate-cluster example(s)",
    ]
    for facet, table in sorted(report.distributions.items()):
        rendered = "; ".join(
            f"{split}: " + " ".join(f"{value}={count}" for value, count in values.items())
            for split, values in table.items()
        )
        lines.append(f"  {facet:<20}: {rendered}")

    def section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title} ({len(items)}):")
        for item in items[:max_items]:
            lines.append(f"  - {item}")
        if len(items) > max_items:
            lines.append(f"  … and {len(items) - max_items} more")

    section(
        "grouped blocks",
        [
            f"[{', '.join(block.example_ids)}] {block.reason} "
            f"→ {report.assignment.get(block.key, '?')}"
            for block in report.grouped_blocks
        ],
    )
    section(
        "residual cross-split overlap",
        [
            f"[{', '.join(finding.example_ids)}] {finding.evidence or ''}".rstrip()
            for finding in report.residual_overlap
        ],
    )
    section(
        "residual cross-split sources",
        [
            f"{source_id}: " + " ".join(f"{split}={ids}" for split, ids in sorted(spread.items()))
            for source_id, spread in sorted(report.residual_sources.items())
        ],
    )
    section("warnings", list(report.warnings))
    return "\n".join(lines)


__all__ = [
    # orchestration
    "SplitReport",
    "build_splits",
    "apply_splits",
    "verify_splits",
    "write_splits",
    "format_report",
    # submodules
    "adversarial",
    "assignment",
    "blocks",
    "random",
    "stratified",
    "temporal",
    # shared constants
    "DEFAULT_EVAL_SPLITS",
    "DEFAULT_RATIOS",
    "DEFAULT_SEED",
    "DEFAULT_STRATIFY_BY",
    "FACETS",
    "SPLIT_ORDER",
    "UNLABELLED",
    # re-exports: blocks
    "Block",
    "block_of_example",
    "duplicate_blocks",
    "singleton_blocks",
    "source_blocks",
    # re-exports: assignment
    "Assignment",
    "apportion",
    "assign_blocks",
    "assign_in_order",
    "normalize_ratios",
    # re-exports: strategies
    "adversarial_splits",
    "random_splits",
    "source_grouped_splits",
    "stratified_splits",
    "temporal_splits",
    # re-exports: stratification helpers
    "block_stratum",
    "declared_bucket",
    "distribution",
    "facet_value",
    "stratum_of",
    # re-exports: temporal helpers
    "parse_timestamp",
    "temporal_order",
]
