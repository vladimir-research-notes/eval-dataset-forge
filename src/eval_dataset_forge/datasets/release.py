"""Release builder: dataset + evidence bundle + computed identity (Milestone 10).

Every stage before this one describes a dataset. This one *publishes* it, and publishing
forces three decisions the earlier stages were allowed to defer.

**1. The audit must describe the data that shipped.** Exclusions are applied first, and the
deduplication, leakage, label, and difficulty stages then run again over the survivors. A
release whose ``audit.json`` reported defects in rows that are not in its ``examples.jsonl``
would be an evidence bundle for a different dataset. Running the ladder twice on a 56-example
corpus costs milliseconds; getting this wrong costs the reader their trust in the bundle.

**2. Deletion is a decision, so it is recorded.** Milestone 7 promised that "removing an
example is a decision for the release builder, which records it in a manifest a reader can
audit." Nothing is dropped by default. When ``--drop`` is used, every removed example lands
in ``manifest.excluded`` with the reason and the id of the member that was kept, and
``manifest.source_content_id`` pins the pre-exclusion corpus, so the deletion is checkable
against the source file rather than taken on faith. ``--drop near`` deletes data on the say-so
of a lexical heuristic with known false positives; it is available, it is not the default, and
the manifest says which rows it took.

**3. A version number is a claim about comparability.** See
:mod:`eval_dataset_forge.datasets.versioning`. The builder computes the bump rather than
accepting one, because on this repo's own corpus a rebuild at a different split seed leaves
the dataset identical and moves the headline number by 3.5× the gap between the two systems
being compared.

**4. A card is an artifact, so it is hashed — and therefore blind to the hashes.**
``dataset_card.md`` (Milestone 11) is rendered from the manifest, and its digest belongs in
that manifest. The order is forced: build a manifest, render the card from it, rebuild the
manifest with the card's bytes included. The two manifests differ only in ``hashes``, which
:func:`~eval_dataset_forge.reports.dataset_card.build_card` does not read — nor ``created_at``,
for the reason ``REPRODUCE.md`` carries no timestamp either.

What this module still does *not* write: the human-readable ``audit.md`` (Milestone 12). The
layout in ``docs/PLAN.md`` §5D lists it; a release built today is missing it, and
``REPRODUCE.md`` says so rather than pretending the bundle is complete.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval_dataset_forge import SCHEMA_VERSION, __version__
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.datasets.manifest import (
    AUDIT_FILE,
    EXAMPLES_FILE,
    MANIFEST_FILE,
    REPRODUCE_FILE,
    SPLITS_DIR,
    build_manifest,
    load_manifest,
)
from eval_dataset_forge.datasets.versioning import (
    ReleaseDelta,
    compare,
    content_id,
    eval_id,
    next_version,
    parse_version,
)
from eval_dataset_forge.datasets.writer import render_json, render_jsonl, write_bytes
from eval_dataset_forge.dedup import DedupReport, find_duplicates
from eval_dataset_forge.difficulty import DifficultyReport, score_dataset
from eval_dataset_forge.labels import LabelAuditReport, audit_labels
from eval_dataset_forge.leakage import LeakageReport, find_leakage
from eval_dataset_forge.reports.dataset_card import CARD_FILE, build_card, render_markdown
from eval_dataset_forge.schemas import (
    AuditResult,
    CardMeta,
    DatasetCard,
    DatasetExample,
    DatasetManifest,
    DatasetRelease,
    DedupKind,
    SplitStrategy,
)
from eval_dataset_forge.splits import (
    DEFAULT_SEED,
    SplitReport,
    apply_splits,
    build_splits,
    duplicate_blocks,
)
from eval_dataset_forge.validation.pii import scan_examples

# What ``--drop`` accepts. Ordered by how much a curator should have to justify it: an exact
# duplicate is redundant by definition, a near duplicate only by a threshold.
DROP_EXACT = "exact"
DROP_NORMALIZED = "normalized"
DROP_NEAR = "near"
DROP_BLOCKING_LABELS = "blocking-labels"
DROP_CHOICES = (DROP_EXACT, DROP_NORMALIZED, DROP_NEAR, DROP_BLOCKING_LABELS)

_DROP_TO_KIND = {
    DROP_EXACT: DedupKind.EXACT,
    DROP_NORMALIZED: DedupKind.NORMALIZED,
    DROP_NEAR: DedupKind.NEAR,
}

# Difficulty calibration runs a permutation test. 2000 resamples put the Monte-Carlo standard
# error on a p near 0.05 at ~0.005 — small against the 0.05 threshold, and cheap. The
# ``difficulty`` CLI defaults to 10000 because it is the command whose whole output is that
# p-value; here it is one line of an audit.
DEFAULT_DIFFICULTY_RESAMPLES = 2_000

# Printed into REPRODUCE.md when a caller builds a release through the library rather than
# the CLI, which is the only path that knows the real command line.
_DEFAULT_COMMAND = "PYTHONPATH=src python3 -m eval_dataset_forge release …"


@dataclass(frozen=True)
class Exclusion:
    """One example removed from the release, and why. Reversible from the source file."""

    example_id: str
    reason: str
    kept_id: str | None = None
    method: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReleaseReport:
    """A built release, its evidence, and the version bump its content earns."""

    release: DatasetRelease
    split_report: SplitReport
    dedup: DedupReport
    labels: LabelAuditReport
    leakage: LeakageReport
    difficulty: DifficultyReport
    artifacts: dict[str, bytes] = field(default_factory=dict)
    exclusions: tuple[Exclusion, ...] = ()
    delta: ReleaseDelta | None = None
    warnings: tuple[str, ...] = ()

    @property
    def manifest(self) -> DatasetManifest:
        return self.release.manifest

    @property
    def audit(self) -> AuditResult:
        assert self.release.audit is not None
        return self.release.audit

    @property
    def card(self) -> DatasetCard:
        assert self.release.card is not None
        return self.release.card

    def counts(self) -> dict[str, int]:
        return {
            "examples": self.manifest.example_count,
            "excluded": len(self.exclusions),
            "artifacts": len(self.artifacts),
            **{name: int(info["count"]) for name, info in sorted(self.manifest.splits.items())},
        }

    def to_dict(self) -> dict:
        return {
            "counts": self.counts(),
            "manifest": self.manifest.model_dump(mode="json"),
            "audit_summary": self.audit.summary(),
            "recommendations": list(self.audit.recommendations),
            "exclusions": [item.to_dict() for item in self.exclusions],
            "delta": self.delta.to_dict() if self.delta else None,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Exclusions
# --------------------------------------------------------------------------- #


def plan_exclusions(
    examples: Sequence[DatasetExample],
    dedup: DedupReport,
    labels: LabelAuditReport,
    drop: Iterable[str],
) -> tuple[Exclusion, ...]:
    """Decide which examples a ``--drop`` request removes, and record why.

    Within a duplicate cluster the survivor is the member that appears **first in input
    order**. That is a convention, not a judgement about which row is better: a later
    member may carry richer metadata, and the builder cannot tell. Both the kept id and the
    dropped ids are written to the manifest so the choice can be inspected and reversed.

    Clusters overlap — a normalized cluster can contain an exact cluster — so a member
    already dropped is skipped, and a cluster left with a single survivor drops nothing.
    """
    drop = set(drop)
    unknown = drop - set(DROP_CHOICES)
    if unknown:
        raise ValueError(f"unknown drop target(s): {', '.join(sorted(unknown))}")

    position = {example.example_id: index for index, example in enumerate(examples)}
    dropped: dict[str, Exclusion] = {}

    for target in (DROP_EXACT, DROP_NORMALIZED, DROP_NEAR):
        if target not in drop:
            continue
        kind = _DROP_TO_KIND[target]
        findings = [finding for finding in dedup.findings if finding.kind is kind]
        for finding in findings:
            members = sorted(
                (eid for eid in finding.example_ids if eid not in dropped), key=position.__getitem__
            )
            if len(members) < 2:
                continue
            kept, *redundant = members
            for example_id in redundant:
                similarity = (
                    f", similarity={finding.similarity:.3f}"
                    if finding.similarity is not None
                    else ""
                )
                dropped[example_id] = Exclusion(
                    example_id=example_id,
                    reason=f"{kind} duplicate of {kept}{similarity}",
                    kept_id=kept,
                    method=finding.method or str(kind),
                )

    if DROP_BLOCKING_LABELS in drop:
        for example_id in labels.blocking_example_ids:
            if example_id not in dropped:
                dropped[example_id] = Exclusion(
                    example_id=example_id,
                    reason="label audit: gold answer is missing, conflicting, or unadjudicated",
                    method="labels.audit",
                )

    return tuple(sorted(dropped.values(), key=lambda item: position[item.example_id]))


# --------------------------------------------------------------------------- #
# Audit assembly
# --------------------------------------------------------------------------- #


def _dependence(examples: Sequence[DatasetExample], dedup: DedupReport) -> dict[str, object]:
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


def _recommendations(
    split_report: SplitReport,
    dedup: DedupReport,
    labels: LabelAuditReport,
    leakage: LeakageReport,
    difficulty: DifficultyReport,
    dependence: Mapping[str, object],
    n_examples: int,
) -> list[str]:
    """Prioritized, actionable notes derived from what the stages actually found."""
    notes: list[str] = []

    if split_report.residual_overlap:
        notes.append(
            f"{len(split_report.residual_overlap)} duplicate cluster(s) straddle a split "
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
            f"{n_exact} exact/normalized duplicate cluster(s) remain in the release; "
            f"`--drop exact,normalized` removes the redundant members and records each "
            f"removal in manifest.excluded."
        )
    if split_report.eval_duplicates:
        notes.append(
            f"{len(split_report.eval_duplicates)} duplicate-cluster example(s) sit in "
            f"dev/test: they spend evaluation slots on repeated items and narrow every "
            f"confidence interval computed from them."
        )
    blocking = labels.blocking_example_ids
    if blocking and not split_report.quarantined:
        notes.append(
            f"{len(blocking)} example(s) have a missing, conflicting, or unadjudicated gold "
            f"answer; `--drop blocking-labels` removes them, `--quarantine-blocking` keeps "
            f"them out of dev/test without deleting them."
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
    block_count = int(dependence["block_count"])  # type: ignore[arg-type]
    if block_count < n_examples:
        notes.append(
            f"{n_examples} examples form {block_count} independent block(s); a bootstrap "
            f"that resamples rows will under-cover. Use `evalforge stats --resample-unit "
            f"block` (the default)."
        )
    return notes


def build_audit(
    dataset_id: str,
    version: str,
    examples: Sequence[DatasetExample],
    *,
    dedup: DedupReport,
    labels: LabelAuditReport,
    leakage: LeakageReport,
    difficulty: DifficultyReport,
    split_report: SplitReport,
    dependence: Mapping[str, object],
) -> AuditResult:
    """Assemble the release's ``AuditResult`` from the stage reports.

    ``audit_id`` is derived from the released content, so the same data always produces the
    same audit id and a report can be matched to the rows it describes without trusting a
    filename. ``AuditResult`` has two deduplication buckets and the ladder has three rungs:
    ``normalized`` findings are filed under ``near_duplicates`` and keep their ``kind``, so
    nothing is lost and nothing is relabelled.

    ``split_statistics`` omits the per-example assignment, which the split files already
    carry verbatim.
    """
    split_stats = split_report.to_dict()
    for key in ("assignment", "duplicate_example_ids"):
        split_stats.pop(key, None)
    split_stats["dependence"] = dict(dependence)

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
        recommendations=_recommendations(
            split_report, dedup, labels, leakage, difficulty, dependence, len(examples)
        ),
    )


# --------------------------------------------------------------------------- #
# REPRODUCE.md
# --------------------------------------------------------------------------- #


def render_reproduce(
    *,
    dataset_id: str,
    version: str,
    examples: Sequence[DatasetExample],
    split_counts: Mapping[str, int],
    source_files: Sequence[str],
    command: str,
    exclusions: Sequence[Exclusion],
    identity: Mapping[str, str],
) -> bytes:
    """The release's ``REPRODUCE.md``: the command, the identities, the gaps.

    ``command`` arrives without ``--version`` and ``--out``, which are appended here: under
    ``--version auto`` neither is known until the delta has been computed, and a
    ``REPRODUCE.md`` that told a reader to rebuild into ``qa_toy_vauto`` would be worse than
    none. The recorded command writes to the canonical release path; a build that redirected
    ``--out`` elsewhere still reproduces into the canonical one.

    Deliberately free of ``created_at``. ``REPRODUCE.md`` is a hashed artifact, so anything
    it embeds becomes part of the release's file hashes; embedding the clock would make
    every rebuild differ from every other and reduce the ``none`` verdict of
    :func:`~eval_dataset_forge.datasets.versioning.compare` to unreachable decoration. The
    timestamp lives in ``manifest.json`` alone, which is the one artifact nothing hashes.
    """
    sizes = ", ".join(f"{name}={count}" for name, count in sorted(split_counts.items()))
    command = " \\\n  ".join(
        [command, f"--version {version}", f"--out datasets/releases/{dataset_id}_v{version}"]
    )
    excluded = (
        "\n".join(f"- `{item.example_id}` — {item.reason}" for item in exclusions)
        or "None. Every source example is in this release."
    )
    sources = "\n".join(f"- `{path}`" for path in source_files)
    body = f"""# Reproducing {dataset_id} v{version}

Built by eval-dataset-forge {__version__} against schema {identity["schema_version"]}.
CPU-only, offline, and deterministic given the same inputs and seed.

## Command

```bash
{command}
```

Every artifact above is byte-identical on a rebuild. `manifest.json` additionally records
`created_at`; add `--created-at <that value>` to reproduce it too. The timestamp is
provenance, not identity — it enters no hash and neither `content_id` nor `eval_id`.

## Identity

| field | value |
|---|---|
| `content_id` | `{identity["content_id"]}` |
| `eval_id` | `{identity["eval_id"]}` |
| `source_content_id` | `{identity["source_content_id"]}` |
| examples | {len(examples)} ({sizes}) |

`content_id` is an order-independent digest of the examples with `split` excluded — it
answers *"are these the same examples?"*. `eval_id` folds each example's split in — it
answers *"are two scores measured on these releases comparable?"*. Rebuilding with a
different `--seed` preserves the first and changes the second, which is the entire reason
both exist.

## Verify without rebuilding

```bash
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \\
  --release datasets/releases/{dataset_id}_v{version}
```

This recomputes every file hash and both identities from the release's own contents. It is
an **integrity** check, not an authenticity one: `manifest.json` cannot hash itself, so
anyone able to rewrite the data can rewrite the digests that describe it.

## Excluded from the source corpus

{excluded}

## Not in this bundle yet

- `audit.md`, the human-readable rendering of `audit.json` — Milestone 12.

`dataset_card.md` is present, and is a rendering of `manifest.json` + `audit.json`. Check that
this tool version still reproduces it:

```bash
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \\
  --release datasets/releases/{dataset_id}_v{version} --check
```

## Source files

{sources}
"""
    return body.encode("utf-8")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _known_limitations(difficulty: DifficultyReport, dedup: DedupReport) -> list[str]:
    """What this release cannot tell a reader, stated in the release rather than the docs."""
    limitations = [
        f"Near-duplicate detection is lexical ({dedup.near_method}). A paraphrase below the "
        f"threshold is not detected, so every duplicate, leakage, and block count here is a "
        f"lower bound.",
        "manifest.json is not self-hashed. verify-release detects accidental corruption of "
        "the data, not deliberate rewriting of the manifest that describes it.",
        "No effective sample size is quoted. It depends on an intraclass correlation, which "
        "depends on the systems being scored; manifest.dependence carries the block "
        "structure needed to compute it.",
    ]
    if not difficulty.trustworthy:
        limitations.append(
            "The difficulty buckets are an uncalibrated heuristic on this corpus: their "
            "ranking is not distinguishable from a random ranking of the declared tags."
        )
    return limitations


def _split_counts(examples: Sequence[DatasetExample], split_report: SplitReport) -> dict[str, int]:
    return {
        name: sum(1 for example in examples if str(example.split) == name)
        for name in split_report.spec.ratios
    }


def _render_artifacts(
    examples: Sequence[DatasetExample],
    split_report: SplitReport,
    audit: AuditResult,
    reproduce: bytes,
) -> dict[str, bytes]:
    artifacts = {EXAMPLES_FILE: render_jsonl(examples)}
    for name in split_report.spec.ratios:
        members = [example for example in examples if str(example.split) == name]
        artifacts[f"{SPLITS_DIR}/{name}.jsonl"] = render_jsonl(members)
    artifacts[AUDIT_FILE] = render_json(audit.model_dump(mode="json"))
    artifacts[REPRODUCE_FILE] = reproduce
    return artifacts


def build_release(
    source_files: Sequence[str | Path],
    *,
    dataset_id: str,
    version: str = "0.1.0",
    created_at: str,
    command: str = "",
    strategy: SplitStrategy | str = SplitStrategy.ADVERSARIAL,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    stratify_by: Sequence[str] = ("task_family",),
    drop: Iterable[str] = (),
    quarantine_blocking: bool = False,
    near_require_same_answer: bool = True,
    difficulty_resamples: int = DEFAULT_DIFFICULTY_RESAMPLES,
    license_notes: str = "",
    card_meta: CardMeta | None = None,
    previous: str | Path | None = None,
) -> ReleaseReport:
    """Build a versioned release from one or more source files. Nothing is written.

    Source files are pooled in the order given, and examples keep that order throughout, so
    every artifact is a deterministic function of the inputs.

    When ``previous`` names an existing release directory, the version is resolved against
    it: pass ``version="auto"`` to take the recommended bump, or an explicit version, which
    must be at least the recommended one. The delta is computed with the *previous* version
    still in place, so the hash churn that assigning a new version would cause cannot feed
    back into the bump that chose it.
    """
    source_files = [str(path) for path in source_files]
    if not source_files:
        raise ValueError("a release needs at least one source file")

    source_examples = [
        example for path in source_files for example in loader.load_examples(path)
    ]
    source_id = content_id(source_examples)

    # Pass one: find the defects that a --drop request acts on.
    source_dedup = find_duplicates(
        source_examples, near_require_same_answer=near_require_same_answer
    )
    source_labels = audit_labels(source_examples)
    exclusions = plan_exclusions(source_examples, source_dedup, source_labels, drop)
    excluded_ids = {item.example_id for item in exclusions}
    kept = [ex for ex in source_examples if ex.example_id not in excluded_ids]
    if not kept:
        raise ValueError("every example was excluded; there is no release to build")

    # Pass two: audit what actually ships. Dropping a duplicate dissolves its cluster, so a
    # report computed before the drop would describe defects the release no longer has.
    dedup = find_duplicates(kept, near_require_same_answer=near_require_same_answer)
    labels = audit_labels(kept)

    split_report = build_splits(
        kept,
        strategy=strategy,
        ratios=ratios,
        seed=seed,
        stratify_by=stratify_by,
        dedup_report=dedup,
        quarantine_ids=labels.blocking_example_ids if quarantine_blocking else (),
    )
    assigned = apply_splits(kept, split_report.assignment)

    leakage = find_leakage(
        assigned, dedup_report=dedup, near_require_same_answer=near_require_same_answer
    )
    difficulty = score_dataset(assigned, n_resamples=difficulty_resamples, seed=seed)
    dependence = _dependence(assigned, dedup)

    def assemble(
        at_version: str,
    ) -> tuple[AuditResult, DatasetCard, dict[str, bytes], DatasetManifest]:
        """Everything downstream of the audit stages, at a given version string.

        The manifest is built twice. ``dataset_card.md`` is an artifact, so its digest belongs
        in ``manifest.hashes``; the card is rendered *from* a manifest, so a manifest must
        exist before the card does. The first manifest is the one the card reads, the second
        is the one that records the card's hash, and they differ in ``hashes`` alone —
        which :func:`~eval_dataset_forge.reports.dataset_card.build_card` never reads. That
        blindness is not a convenience: a card quoting its own digest would need that digest
        to appear inside the bytes it is the digest of.
        """
        audit = build_audit(
            dataset_id,
            at_version,
            assigned,
            dedup=dedup,
            labels=labels,
            leakage=leakage,
            difficulty=difficulty,
            split_report=split_report,
            dependence=dependence,
        )
        reproduce = render_reproduce(
            dataset_id=dataset_id,
            version=at_version,
            examples=assigned,
            split_counts=_split_counts(assigned, split_report),
            source_files=source_files,
            command=command or _DEFAULT_COMMAND,
            exclusions=exclusions,
            identity={
                "schema_version": SCHEMA_VERSION,
                "content_id": content_id(assigned),
                "eval_id": eval_id(assigned),
                "source_content_id": source_id,
            },
        )
        artifacts = _render_artifacts(assigned, split_report, audit, reproduce)

        def manifest_over(rendered: Mapping[str, bytes]) -> DatasetManifest:
            return build_manifest(
                dataset_id=dataset_id,
                version=at_version,
                created_at=created_at,
                examples=assigned,
                artifacts=rendered,
                source_files=source_files,
                audit=audit,
                split_spec=split_report.spec,
                excluded=[item.to_dict() for item in exclusions],
                dependence=dependence,
                source_content_id=source_id,
                known_limitations=_known_limitations(difficulty, dedup),
                license_notes=license_notes,
                card_meta=card_meta,
            )

        card = build_card(manifest_over(artifacts), audit)
        artifacts[CARD_FILE] = render_markdown(card)
        return audit, card, artifacts, manifest_over(artifacts)

    resolved = version
    delta: ReleaseDelta | None = None
    if previous is not None:
        previous_dir = Path(previous)
        previous_manifest = load_manifest(previous_dir / MANIFEST_FILE)
        previous_examples = loader.load_examples(previous_dir / EXAMPLES_FILE)
        # The version must not influence the bump that chooses it. `audit.json` and
        # `REPRODUCE.md` embed the version string, so a candidate built at the *new* version
        # would differ from the previous release in `hashes` no matter what the data did, and
        # every rebuild would report at least a `patch`. Assembling the candidate at the
        # previous version is what makes `none` reachable and `patch` meaningful.
        *_, candidate = assemble(previous_manifest.version)
        delta = compare(previous_manifest, previous_examples, candidate, assigned)
        if version == "auto":
            resolved = delta.recommended_version
        elif parse_version(version) < parse_version(delta.recommended_version):
            raise ValueError(
                f"--version {version} is below the {delta.bump} bump this content requires "
                f"({previous_manifest.version} → {delta.recommended_version}). "
                + "; ".join(delta.reasons)
            )
    elif version == "auto":
        raise ValueError("--version auto needs --previous to compare against")

    audit, card, artifacts, manifest = assemble(resolved)
    artifacts[MANIFEST_FILE] = render_json(manifest.model_dump(mode="json"))

    warnings = list(split_report.warnings)
    if exclusions:
        warnings.append(
            f"{len(exclusions)} example(s) were removed from the source corpus; "
            f"source_content_id {source_id} pins what they were removed from"
        )
    if delta is not None:
        warnings.extend(delta.warnings)

    return ReleaseReport(
        release=DatasetRelease(
            manifest=manifest, examples=list(assigned), audit=audit, card=card
        ),
        split_report=split_report,
        dedup=dedup,
        labels=labels,
        leakage=leakage,
        difficulty=difficulty,
        artifacts=artifacts,
        exclusions=exclusions,
        delta=delta,
        warnings=tuple(warnings),
    )


def write_release(report: ReleaseReport, out_dir: str | Path) -> dict[str, Path]:
    """Write every rendered artifact under ``out_dir``. Byte-identical across runs."""
    out_dir = Path(out_dir)
    return {
        name: write_bytes(out_dir / name, data)
        for name, data in sorted(report.artifacts.items())
    }


def format_report(report: ReleaseReport, max_items: int = 10) -> str:
    """Render a :class:`ReleaseReport` for a terminal."""
    manifest = report.manifest
    sizes = ", ".join(
        "{}={}".format(name, info["count"]) for name, info in sorted(manifest.splits.items())
    )
    dependence = manifest.dependence
    lines = [
        f"release: {manifest.dataset_id} v{manifest.version} "
        f"(evalforge {manifest.tool_version}, schema {manifest.schema_version})",
        f"  examples     : {manifest.example_count} ({sizes})",
        f"  strategy     : {manifest.split_spec.strategy if manifest.split_spec else 'n/a'} "
        f"(seed {manifest.split_spec.seed if manifest.split_spec else 'n/a'})",
        f"  content_id   : {manifest.content_id}",
        f"  eval_id      : {manifest.eval_id}",
        f"  excluded     : {len(report.exclusions)} "
        f"(source_content_id {manifest.source_content_id})",
        f"  blocks       : {dependence.get('block_count', 'n/a')} independent "
        f"({dependence.get('duplicate_block_count', 0)} contain a duplicate cluster)",
        "  audit        : "
        + ", ".join(f"{k}={v}" for k, v in manifest.audit_results.items()),
    ]

    def section(title: str, items: Sequence[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title} ({len(items)}):")
        for item in items[:max_items]:
            lines.append(f"  - {item}")
        if len(items) > max_items:
            lines.append(f"  … and {len(items) - max_items} more")

    section("recommendations", report.audit.recommendations)
    section(
        "excluded",
        [f"{item.example_id}: {item.reason}" for item in report.exclusions],
    )
    section("warnings", list(report.warnings))
    if report.delta is not None:
        from eval_dataset_forge.datasets.versioning import format_delta

        lines.append("")
        lines.append(format_delta(report.delta, max_items=max_items))
    return "\n".join(lines)


__all__ = [
    "DEFAULT_DIFFICULTY_RESAMPLES",
    "DROP_BLOCKING_LABELS",
    "DROP_CHOICES",
    "DROP_EXACT",
    "DROP_NEAR",
    "DROP_NORMALIZED",
    "Exclusion",
    "ReleaseReport",
    "build_audit",
    "build_release",
    "format_report",
    "next_version",
    "plan_exclusions",
    "render_reproduce",
    "write_release",
]
