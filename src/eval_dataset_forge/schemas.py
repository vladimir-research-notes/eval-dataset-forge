"""Typed data contracts for eval-dataset-forge (pydantic v2).

This module is the authoritative implementation of the on-disk formats described
in ``docs/DATASET_SCHEMA.md``; the two must stay in sync, and ``SCHEMA_VERSION``
(in ``eval_dataset_forge.__init__``) is bumped on any breaking change.

Design notes:

- Every model derives from :class:`ForgeModel`, which forbids unknown fields.
  Malformed data therefore fails loudly at load time instead of silently
  propagating a typo (e.g. ``expcted_output``) into an audit. Free-form
  extension goes into the explicit ``metadata`` / ``labels`` dicts, not into
  stray top-level keys.
- Enumerations (:class:`Split`, :class:`DedupKind`, …) are ``str``-valued so a
  ``model_dump(mode="json")`` round-trips through plain JSONL with no custom
  encoder, and comparisons against bare strings still work.
- A handful of cheap cross-field validators (split-ratio sum, CI ordering,
  release example count) catch structurally impossible records early. They are
  documented on each model; they are not a substitute for the later audit
  stages, which look for *semantic* defects rather than *structural* ones.

Importing this module pulls in pydantic. The package ``__init__`` and the CLI
deliberately do **not** import it at top level, so ``evalforge check-env`` keeps
working before the scientific stack is installed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval_dataset_forge import SCHEMA_VERSION

# --------------------------------------------------------------------------- #
# Shared base + type aliases
# --------------------------------------------------------------------------- #


class ForgeModel(BaseModel):
    """Base model: reject unknown fields so malformed data fails at load time."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# A non-empty string. Used for identity/provenance fields where an empty value is
# never meaningful and almost always signals an upstream bug.
NonEmptyStr = Annotated[str, Field(min_length=1)]

# The expected output of an example: a gold answer (str), a scalar label
# (bool/int/float), or a structured judgement (list/dict). ``None`` is
# intentionally excluded — a missing gold answer is a *label* defect surfaced by
# the label audit, not a valid ``expected_output``.
ExpectedOutput = str | bool | int | float | list | dict

# Task families the fixtures and audits are designed around. Not enforced on the
# model (the vocabulary is intentionally open, per docs/DATASET_SCHEMA.md); the
# validation layer (Milestone 3) may warn on unknown families.
KNOWN_TASK_FAMILIES = frozenset({"qa", "agent_task", "retrieval", "bio", "safety"})


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Split(StrEnum):
    """Which split an example belongs to. ``unassigned`` until a split is built."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    UNASSIGNED = "unassigned"


class DedupKind(StrEnum):
    """How a duplicate cluster was detected. ``near`` is heuristic."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    NEAR = "near"


class LeakageKind(StrEnum):
    """Category of a leakage / contamination finding."""

    TRAIN_TEST_OVERLAP = "train_test_overlap"
    ANSWER_IN_PROMPT = "answer_in_prompt"
    NGRAM_OVERLAP = "ngram_overlap"
    RETRIEVAL_CONTAMINATION = "retrieval_contamination"


class LabelIssue(StrEnum):
    """Category of a label-quality finding."""

    MISSING = "missing"
    CONFLICTING = "conflicting"
    AMBIGUOUS = "ambiguous"
    INCONSISTENT_FORMAT = "inconsistent_format"
    NEEDS_ADJUDICATION = "needs_adjudication"


class SplitStrategy(StrEnum):
    """Split-construction strategy. Values match the ``build-splits`` CLI."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    SOURCE_GROUPED = "source-grouped"
    ADVERSARIAL = "adversarial"
    TEMPORAL = "temporal"


class DifficultyBucket(StrEnum):
    """Coarse difficulty bucket derived from the difficulty score."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# --------------------------------------------------------------------------- #
# Core example
# --------------------------------------------------------------------------- #


class DatasetExample(ForgeModel):
    """One evaluation example — the atomic unit every audit stage operates on.

    Required identity/content fields (``example_id``, ``source_id``,
    ``task_family``, ``input``, ``expected_output``) must be present and, for the
    identity fields, non-empty. Everything else has a safe default so
    hand-authored fixtures stay terse.
    """

    example_id: NonEmptyStr
    source_id: NonEmptyStr
    task_family: NonEmptyStr
    input: str
    expected_output: ExpectedOutput
    metadata: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)
    difficulty_tags: list[str] = Field(default_factory=list)
    split: Split = Split.UNASSIGNED
    provenance: str | None = None
    created_at: str | None = None
    version: str | None = None


# --------------------------------------------------------------------------- #
# Audit finding types
# --------------------------------------------------------------------------- #


class DeduplicationFinding(ForgeModel):
    """A cluster of examples judged to be duplicates of one another.

    ``similarity`` is ``None`` for exact/normalized matches (they are identical by
    construction) and a value in ``[0, 1]`` for heuristic ``near`` matches.
    """

    kind: DedupKind
    example_ids: list[str] = Field(min_length=2)
    similarity: float | None = None
    method: str = ""


class LeakageFinding(ForgeModel):
    """A leakage / contamination signal between prompt, answer, or splits.

    ``score`` is the heuristic strength (e.g. n-gram overlap fraction); ``evidence``
    is a short human-readable snippet explaining why it fired.
    """

    kind: LeakageKind
    example_ids: list[str] = Field(min_length=1)
    score: float | None = None
    evidence: str | None = None


class LabelAuditFinding(ForgeModel):
    """A single label-quality issue on one example. Nothing is auto-corrected."""

    kind: LabelIssue
    example_id: NonEmptyStr
    detail: str = ""


# --------------------------------------------------------------------------- #
# Split / difficulty / statistics contracts
# --------------------------------------------------------------------------- #


class SplitSpec(ForgeModel):
    """A reproducible description of how splits were (or will be) constructed.

    ``ratios`` maps split name → fraction and, when provided, must be
    non-negative and sum to 1.0 (within a small tolerance). It may be left empty
    for strategies that do not partition by ratio (e.g. a pure grouping).
    """

    strategy: SplitStrategy
    seed: int = 0
    ratios: dict[str, float] = Field(default_factory=dict)
    group_key: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_ratios(self) -> SplitSpec:
        if self.ratios:
            for name, ratio in self.ratios.items():
                if ratio < 0:
                    raise ValueError(f"split ratio for {name!r} is negative: {ratio}")
            total = sum(self.ratios.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self


class DifficultyScore(ForgeModel):
    """A transparent, model-free difficulty estimate for one example.

    ``components`` records the per-signal contributions so the score is fully
    explainable; ``method`` names the heuristic that produced it.
    """

    example_id: NonEmptyStr
    score: float
    bucket: DifficultyBucket
    components: dict[str, float] = Field(default_factory=dict)
    method: str = ""


class BootstrapResult(ForgeModel):
    """A bootstrap confidence interval for a single metric.

    ``ci_low`` must not exceed ``ci_high``. The point estimate is the metric on
    the full sample; the interval comes from ``n_resamples`` resamples at the
    given ``seed``.

    ``method`` and ``resample_unit`` are not decoration. A percentile interval and
    a BCa interval over the same resamples are different intervals, and an interval
    that resampled *examples* assumes an independence that one resampling *blocks*
    does not (see :mod:`eval_dataset_forge.statistics.bootstrap`). A stored result
    that omitted them could not be compared with another, so they travel with it.
    ``method`` records what actually ran: a BCa request that hit a degenerate case
    is written ``"bca->percentile"``.
    """

    metric: NonEmptyStr
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int = Field(ge=1)
    seed: int = 0
    system: str | None = None
    method: str = "percentile"
    resample_unit: str = "example"

    @model_validator(mode="after")
    def _check_interval(self) -> BootstrapResult:
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) > ci_high ({self.ci_high})")
        return self


# --------------------------------------------------------------------------- #
# Aggregate audit result
# --------------------------------------------------------------------------- #


class AuditResult(ForgeModel):
    """The machine-readable output of an audit run over one dataset version.

    Collects every finding type plus difficulty/split summaries and prioritized
    recommendations. Serialized to ``audit.json`` in a release; its counts feed
    ``DatasetManifest.audit_results`` and the dataset card's "known issues".
    """

    audit_id: NonEmptyStr
    dataset_id: NonEmptyStr
    version: str
    exact_duplicates: list[DeduplicationFinding] = Field(default_factory=list)
    near_duplicates: list[DeduplicationFinding] = Field(default_factory=list)
    leakage_flags: list[LeakageFinding] = Field(default_factory=list)
    label_warnings: list[LabelAuditFinding] = Field(default_factory=list)
    pii_flags: list[dict[str, Any]] = Field(default_factory=list)
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    split_statistics: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, int]:
        """Per-dimension finding counts, for manifests and quick reporting."""
        return {
            "exact_duplicates": len(self.exact_duplicates),
            "near_duplicates": len(self.near_duplicates),
            "leakage_flags": len(self.leakage_flags),
            "label_warnings": len(self.label_warnings),
            "pii_flags": len(self.pii_flags),
            "recommendations": len(self.recommendations),
        }


# --------------------------------------------------------------------------- #
# Release-level contracts
# --------------------------------------------------------------------------- #


class DatasetManifest(ForgeModel):
    """The provenance/inventory record written as ``manifest.json`` per release.

    ``hashes`` holds ``sha256:<hex>`` digests of the release's *files*;
    ``splits`` maps split name → ``{count, path, sha256}``; ``audit_results`` carries
    the :meth:`AuditResult.summary` counts. ``schema_version`` defaults to the current
    ``SCHEMA_VERSION`` so a release records the format it was built against.

    **Three identities, because a file hash answers the wrong question.** A hash of
    ``examples.jsonl`` changes when the JSON key order changes and stays put when the
    split assignment does. Neither is what a reader comparing two reported scores needs
    to know. So a manifest carries:

    ``hashes``
        Per-file ``sha256``. Answers *"did these bytes survive the trip?"*
    ``content_id``
        Order-independent digest over the released examples' content, ignoring ``split``.
        Answers *"are these the same examples?"*
    ``eval_id``
        The same digest with each example's split folded in. Answers *"are scores measured
        on these two releases comparable?"* — the only question a leaderboard actually
        asks, and the one that drives the version bump (see
        :mod:`eval_dataset_forge.datasets.versioning`).

    ``source_content_id`` records the content id *before* ``excluded`` examples were
    dropped, so a release that removed data says so in a form a reader can check against
    the source file.

    ``dependence`` records the duplicate-block structure (``block_count``,
    ``block_size_histogram``, ``near_method``) and deliberately does **not** record an
    ``effective_n``. Effective sample size is a property of a *scored comparison*, not of
    a dataset: it needs an intraclass correlation, which needs system scores, which a
    dataset does not have. The block structure is what a reader needs to compute it for
    their own systems; quoting a single number here would mean inventing an ICC.
    """

    dataset_id: NonEmptyStr
    version: NonEmptyStr
    created_at: str
    source_files: list[str] = Field(default_factory=list)
    example_count: int = Field(ge=0)
    schema_version: str = SCHEMA_VERSION
    hashes: dict[str, str] = Field(default_factory=dict)
    splits: dict[str, Any] = Field(default_factory=dict)
    audit_results: dict[str, Any] = Field(default_factory=dict)
    known_limitations: list[str] = Field(default_factory=list)
    license_notes: str = ""
    content_id: str = ""
    eval_id: str = ""
    source_content_id: str = ""
    tool_version: str = ""
    split_spec: SplitSpec | None = None
    excluded: list[dict[str, Any]] = Field(default_factory=list)
    dependence: dict[str, Any] = Field(default_factory=dict)


class DatasetCard(ForgeModel):
    """Structured dataset card. Rendered to ``dataset_card.md`` (Milestone 11).

    The card is generated *from* the audit and manifest so its "known issues"
    always match what the pipeline actually found, rather than being written by
    hand and drifting.
    """

    dataset_id: NonEmptyStr
    version: str
    purpose: str = ""
    construction: str = ""
    schema_summary: str = ""
    splits: dict[str, Any] = Field(default_factory=dict)
    known_issues: list[str] = Field(default_factory=list)
    audit_findings: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    recommended_use: str = ""
    non_goals: list[str] = Field(default_factory=list)


class DatasetRelease(ForgeModel):
    """In-memory bundle of everything a release directory contains.

    Materializing this to disk (``examples.jsonl`` + ``manifest.json`` +
    ``audit.{json,md}`` + ``dataset_card.md`` + ``splits/`` + ``REPRODUCE.md``)
    is the Release builder's job (Milestone 10). Here it just ties the pieces
    together and cross-checks the example count against the manifest.
    """

    manifest: DatasetManifest
    examples: list[DatasetExample] = Field(default_factory=list)
    audit: AuditResult | None = None
    card: DatasetCard | None = None

    @model_validator(mode="after")
    def _check_example_count(self) -> DatasetRelease:
        if self.examples and self.manifest.example_count != len(self.examples):
            raise ValueError(
                f"manifest.example_count ({self.manifest.example_count}) != "
                f"number of examples ({len(self.examples)})"
            )
        return self


# --------------------------------------------------------------------------- #
# Registry (introspection / tests)
# --------------------------------------------------------------------------- #

# Every public model, keyed by class name. Used by tests to assert the full
# contract set is present and JSON-round-trippable, and available for tooling.
MODELS: dict[str, type[ForgeModel]] = {
    m.__name__: m
    for m in (
        DatasetExample,
        DeduplicationFinding,
        LeakageFinding,
        LabelAuditFinding,
        SplitSpec,
        DifficultyScore,
        BootstrapResult,
        AuditResult,
        DatasetManifest,
        DatasetCard,
        DatasetRelease,
    )
}

__all__ = [
    "SCHEMA_VERSION",
    "ForgeModel",
    "NonEmptyStr",
    "ExpectedOutput",
    "KNOWN_TASK_FAMILIES",
    "Split",
    "DedupKind",
    "LeakageKind",
    "LabelIssue",
    "SplitStrategy",
    "DifficultyBucket",
    "DatasetExample",
    "DeduplicationFinding",
    "LeakageFinding",
    "LabelAuditFinding",
    "SplitSpec",
    "DifficultyScore",
    "BootstrapResult",
    "AuditResult",
    "DatasetManifest",
    "DatasetCard",
    "DatasetRelease",
    "MODELS",
]
