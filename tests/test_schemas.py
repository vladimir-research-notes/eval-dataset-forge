"""Tests for the core pydantic contracts (Milestone 1).

These lock three things:
  * the field set of the public contracts (so the on-disk format does not drift
    silently from docs/DATASET_SCHEMA.md),
  * strict loading behaviour (unknown fields and structurally impossible records
    are rejected loudly), and
  * JSON round-trip fidelity (dump -> reload -> equal) for every model, since the
    formats are persisted as JSON/JSONL.

Everything here is CPU-only and offline.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import eval_dataset_forge
from eval_dataset_forge import schemas
from eval_dataset_forge.schemas import (
    MODELS,
    AuditResult,
    BootstrapResult,
    CardMeta,
    DatasetCard,
    DatasetExample,
    DatasetManifest,
    DatasetRelease,
    DedupKind,
    DeduplicationFinding,
    DifficultyBucket,
    DifficultyScore,
    LabelAuditFinding,
    LabelIssue,
    LeakageFinding,
    LeakageKind,
    Split,
    SplitSpec,
    SplitStrategy,
)

# --------------------------------------------------------------------------- #
# Representative instances (one per model), used for round-trip + registry tests
# --------------------------------------------------------------------------- #


def _example(**kw) -> DatasetExample:
    base = dict(
        example_id="qa-001",
        source_id="src-1",
        task_family="qa",
        input="What is the capital of France?",
        expected_output="Paris",
    )
    base.update(kw)
    return DatasetExample(**base)


def _one_of_each() -> dict[str, object]:
    """A populated instance of every model in ``MODELS``."""
    example = _example(
        split="test",
        difficulty_tags=["easy"],
        metadata={"note": "seed question"},
        labels={"annotations": [{"annotator": "a1", "value": "Paris"}]},
    )
    dedup = DeduplicationFinding(
        kind="near", example_ids=["qa-001", "qa-002"], similarity=0.95, method="jaccard"
    )
    leakage = LeakageFinding(
        kind="answer_in_prompt", example_ids=["qa-003"], score=1.0, evidence="'Paris' in input"
    )
    label = LabelAuditFinding(kind="conflicting", example_id="qa-004", detail="two gold answers")
    spec = SplitSpec(
        strategy="stratified",
        seed=7,
        ratios={"train": 0.7, "dev": 0.15, "test": 0.15},
        group_key="source_id",
        constraints={"near_dups_out_of_test": True},
    )
    difficulty = DifficultyScore(
        example_id="qa-001",
        score=0.42,
        bucket="medium",
        components={"length": 0.2, "reasoning_steps": 0.22},
        method="surface_v1",
    )
    boot = BootstrapResult(
        metric="accuracy", point_estimate=0.8, ci_low=0.7, ci_high=0.9, n_resamples=1000, seed=1
    )
    audit = AuditResult(
        audit_id="audit-1",
        dataset_id="qa_toy",
        version="0.1.0",
        near_duplicates=[dedup],
        leakage_flags=[leakage],
        label_warnings=[label],
        pii_flags=[{"example_id": "qa-005", "kind": "email"}],
        difficulty_distribution={"easy": 3, "medium": 2, "hard": 1},
        recommendations=["drop near-duplicate cluster qa-001/qa-002"],
    )
    card_meta = CardMeta(
        purpose="demonstrate the audit pipeline",
        construction="hand-authored synthetic examples with planted defects",
        recommended_use="pipeline demonstration only",
        non_goals=["certifying universal benchmark quality"],
    )
    manifest = DatasetManifest(
        dataset_id="qa_toy",
        version="0.1.0",
        created_at="2026-01-01T00:00:00Z",
        source_files=["datasets/fixtures/qa_toy/examples.jsonl"],
        example_count=1,
        hashes={"examples.jsonl": "0" * 64},
        splits={"test": {"count": 1, "path": "splits/test.jsonl"}},
        audit_results=audit.summary(),
        known_limitations=["heuristic near-dup detection can miss paraphrases"],
        license_notes="synthetic; MIT",
        card_meta=card_meta,
    )
    card = DatasetCard(
        dataset_id="qa_toy",
        version="0.1.0",
        purpose="demonstrate the audit pipeline",
        construction="hand-authored synthetic examples with planted defects",
        known_issues=["contains deliberate near-duplicates"],
        limitations=["tiny; not for real ranking"],
        recommended_use="pipeline demonstration only",
        non_goals=["certifying universal benchmark quality"],
    )
    release = DatasetRelease(manifest=manifest, examples=[example], audit=audit, card=card)
    return {
        "DatasetExample": example,
        "DeduplicationFinding": dedup,
        "LeakageFinding": leakage,
        "LabelAuditFinding": label,
        "SplitSpec": spec,
        "DifficultyScore": difficulty,
        "BootstrapResult": boot,
        "AuditResult": audit,
        "CardMeta": card_meta,
        "DatasetManifest": manifest,
        "DatasetCard": card,
        "DatasetRelease": release,
    }


# --------------------------------------------------------------------------- #
# Registry / round-trip
# --------------------------------------------------------------------------- #


def test_registry_covers_every_documented_model() -> None:
    expected = {
        "DatasetExample",
        "DeduplicationFinding",
        "LeakageFinding",
        "LabelAuditFinding",
        "SplitSpec",
        "DifficultyScore",
        "BootstrapResult",
        "AuditResult",
        "CardMeta",
        "DatasetManifest",
        "DatasetCard",
        "DatasetRelease",
    }
    assert set(MODELS) == expected


def test_every_model_json_round_trips() -> None:
    instances = _one_of_each()
    assert set(instances) == set(MODELS)
    for name, instance in instances.items():
        model_cls = MODELS[name]
        # Dump to JSON-native types, serialize to a string, reload, revalidate.
        dumped = instance.model_dump(mode="json")
        text = json.dumps(dumped)  # must be plain-JSON serializable (no custom encoder)
        reloaded = model_cls.model_validate(json.loads(text))
        assert reloaded == instance, f"round-trip changed {name}"


def test_enums_serialize_to_plain_strings() -> None:
    dumped = _example(split="test").model_dump(mode="json")
    assert dumped["split"] == "test"
    finding = DeduplicationFinding(kind="exact", example_ids=["a", "b"]).model_dump(mode="json")
    assert finding["kind"] == "exact"


# --------------------------------------------------------------------------- #
# DatasetExample
# --------------------------------------------------------------------------- #


def test_example_field_set_is_locked() -> None:
    # Guards against silent drift from docs/DATASET_SCHEMA.md.
    assert set(DatasetExample.model_fields) == {
        "example_id",
        "source_id",
        "task_family",
        "input",
        "expected_output",
        "metadata",
        "labels",
        "difficulty_tags",
        "split",
        "provenance",
        "created_at",
        "version",
    }


def test_example_defaults() -> None:
    ex = _example()
    assert ex.split is Split.UNASSIGNED
    assert ex.metadata == {} and ex.labels == {} and ex.difficulty_tags == []
    assert ex.provenance is None and ex.created_at is None and ex.version is None


def test_example_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        _example(expcted_output="typo")  # misspelled -> not silently absorbed


@pytest.mark.parametrize(
    "missing",
    ["example_id", "source_id", "task_family", "input", "expected_output"],
)
def test_example_requires_core_fields(missing: str) -> None:
    kwargs = dict(
        example_id="x",
        source_id="s",
        task_family="qa",
        input="q",
        expected_output="a",
    )
    kwargs.pop(missing)
    with pytest.raises(ValidationError):
        DatasetExample(**kwargs)


@pytest.mark.parametrize("field", ["example_id", "source_id", "task_family"])
def test_example_identity_fields_reject_empty(field: str) -> None:
    with pytest.raises(ValidationError):
        _example(**{field: ""})


@pytest.mark.parametrize(
    "value",
    ["Paris", 3, 3.5, True, ["a", "b"], {"label": "positive", "score": 0.9}],
)
def test_expected_output_accepts_json_values(value: object) -> None:
    ex = _example(expected_output=value)
    # Smart-union preserves the input type (no bool<->int coercion surprises).
    assert ex.expected_output == value
    assert type(ex.expected_output) is type(value)


def test_expected_output_rejects_none() -> None:
    with pytest.raises(ValidationError):
        _example(expected_output=None)


def test_split_coerces_from_string() -> None:
    assert _example(split="train").split is Split.TRAIN
    with pytest.raises(ValidationError):
        _example(split="holdout")


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


def test_dedup_finding_needs_a_pair() -> None:
    with pytest.raises(ValidationError):
        DeduplicationFinding(kind=DedupKind.EXACT, example_ids=["only-one"])
    ok = DeduplicationFinding(kind=DedupKind.NORMALIZED, example_ids=["a", "b", "c"])
    assert ok.similarity is None


def test_leakage_finding_needs_at_least_one_id() -> None:
    with pytest.raises(ValidationError):
        LeakageFinding(kind=LeakageKind.NGRAM_OVERLAP, example_ids=[])


def test_label_finding_kind_and_id() -> None:
    finding = LabelAuditFinding(kind=LabelIssue.MISSING, example_id="q1")
    assert finding.kind is LabelIssue.MISSING
    with pytest.raises(ValidationError):
        LabelAuditFinding(kind="not-a-kind", example_id="q1")


# --------------------------------------------------------------------------- #
# SplitSpec / DifficultyScore / BootstrapResult validators
# --------------------------------------------------------------------------- #


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        SplitSpec(strategy=SplitStrategy.RANDOM, ratios={"train": 0.6, "test": 0.3})


def test_split_ratios_reject_negative() -> None:
    with pytest.raises(ValidationError):
        SplitSpec(strategy="random", ratios={"train": 1.2, "test": -0.2})


def test_split_ratios_optional() -> None:
    spec = SplitSpec(strategy=SplitStrategy.SOURCE_GROUPED, group_key="source_id")
    assert spec.ratios == {}


def test_difficulty_bucket_enum() -> None:
    score = DifficultyScore(example_id="q1", score=0.9, bucket="hard")
    assert score.bucket is DifficultyBucket.HARD
    with pytest.raises(ValidationError):
        DifficultyScore(example_id="q1", score=0.9, bucket="impossible")


def test_bootstrap_ci_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        BootstrapResult(
            metric="accuracy", point_estimate=0.5, ci_low=0.6, ci_high=0.4, n_resamples=100
        )


def test_bootstrap_requires_positive_resamples() -> None:
    with pytest.raises(ValidationError):
        BootstrapResult(
            metric="accuracy", point_estimate=0.5, ci_low=0.4, ci_high=0.6, n_resamples=0
        )


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #


def test_audit_summary_counts() -> None:
    audit = _one_of_each()["AuditResult"]
    assert audit.summary() == {
        "exact_duplicates": 0,
        "near_duplicates": 1,
        "leakage_flags": 1,
        "label_warnings": 1,
        "pii_flags": 1,
        "recommendations": 1,
    }


def test_manifest_defaults_schema_version() -> None:
    manifest = DatasetManifest(
        dataset_id="d", version="0.1.0", created_at="2026-01-01T00:00:00Z", example_count=0
    )
    assert manifest.schema_version == eval_dataset_forge.SCHEMA_VERSION
    assert manifest.license_notes == ""


def test_manifest_rejects_negative_count() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest(
            dataset_id="d", version="0.1.0", created_at="t", example_count=-1
        )


def test_release_example_count_must_match_manifest() -> None:
    manifest = DatasetManifest(
        dataset_id="d", version="0.1.0", created_at="t", example_count=5
    )
    with pytest.raises(ValidationError):
        DatasetRelease(manifest=manifest, examples=[_example()])  # 1 != 5


def test_release_allows_empty_examples() -> None:
    # A manifest can be built before examples are attached (count cross-check is
    # only enforced once examples are present).
    manifest = DatasetManifest(
        dataset_id="d", version="0.1.0", created_at="t", example_count=5
    )
    release = DatasetRelease(manifest=manifest)
    assert release.examples == [] and release.audit is None


def test_module_reexports_schema_version() -> None:
    assert schemas.SCHEMA_VERSION == eval_dataset_forge.SCHEMA_VERSION
