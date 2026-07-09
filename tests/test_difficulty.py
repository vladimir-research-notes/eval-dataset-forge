"""Tests for difficulty calibration (Milestone 8).

Covered here:

  * **Signals** — the five raw extractors on the examples that motivated them, the
    log-compression's dataset-independence and monotonicity, and the two rules that were
    chosen a priori rather than fitted: bare digits are not reasoning steps (``"What is
    2 + 2?"`` scores zero, not two), and ``tool_use`` reads truthiness so a dataset that
    writes ``{"tools": []}`` on every example does not make every example look harder.
  * **The explainability invariant** — ``sum(components.values()) == score``, bit-exact,
    for every example in every fixture. This is why nothing is rounded.
  * **No circularity** — the score never reads ``difficulty_tags``. Rewriting an example's
    declared difficulty leaves its score untouched. And the reason it matters is measured:
    blending the declared bucket back into the score at weight 0.15 lifts rho from +0.10
    to +0.49, turning a null heuristic into an apparently working one.
  * **Bucketing** — equal scores always receive equal buckets (the corpus has two tie groups
    of four); quantile bucketing therefore cannot hit equal frequencies and reports what it
    achieved; absolute thirds leave ``hard`` empty on a corpus whose maximum score is 0.658;
    and the declared marginal is apportioned to the corpus size before it is matched.
  * **Rank statistics** — ``rankdata``/``spearman`` against numpy references, the permutation
    p-value's uniformity under the null and its non-zero floor, and ``required_n`` against a
    hand-computed Fisher-z value.
  * **A bug the tests found** — ``calibrate`` scored examples with ``matrix @ weights`` while
    the report published ``sum(components)``. The two differ by 5.6e-17, which splits a
    four-way tie and moves the reported rho. Both the fix and the mechanism are pinned.
  * **The headline result** — the heuristic ranking is not distinguishable from a random
    ranking of the declared labels (rho=+0.10, p=0.50, n=51); no point on the weight
    simplex rescues it; ``task_family`` explains 43% of the score's variance and 1% of the
    declared difficulty's; the within-family correlation collapses to +0.03; and the
    marginal-matched bucket accuracy fails to beat "call everything easy".
  * **The positive control** — a synthetic corpus whose difficulty *is* driven by the
    signals is reported ``trustworthy``, and the same corpus with its labels shuffled is
    not. A harness that can only ever answer "no" would prove nothing.
  * **Honest negatives** — ``tool_use`` fires on zero examples and is reported inert rather
    than credited; two of five signals are anti-correlated with declared difficulty;
    ``input_length`` alone clears alpha but not Bonferroni; and an empty gold answer sinks
    an example to the easy end, so Milestone 6's ``MISSING`` label defect corrupts this
    milestone's score.
  * **Determinism and IO** — seeded reproducibility, input immutability, subset invariance
    of a score, byte-identical artifacts, and the CLI including its ``--fail-on-uncalibrated``
    gate.

Everything is CPU-only and offline.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from eval_dataset_forge.cli import _parse_weights, build_parser, main
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.difficulty import (
    DEFAULT_WEIGHTS,
    DIFFICULTY_ORDER,
    FIXED_CUTOFFS,
    METHOD,
    MIN_CALIBRATION_N,
    SIGNAL_NAMES,
    SIGNALS,
    CalibrationReport,
    DifficultyReport,
    assign_buckets,
    bucket_distribution,
    bucket_of,
    calibrate,
    cutpoints_for,
    declared_bucket,
    declared_distribution,
    declared_ordinal,
    distribution_by_split,
    eta_squared,
    format_report,
    marginal_cutpoints,
    normalized_signals,
    permutation_p,
    quantile_cutpoints,
    rankdata,
    raw_score,
    raw_signals,
    required_n,
    resolve_weights,
    scale_counts,
    score_dataset,
    spearman,
    weighted_components,
    write_report,
)
from eval_dataset_forge.difficulty.heuristics import (
    raw_evidence_items,
    raw_input_length,
    raw_output_complexity,
    raw_reasoning_steps,
    raw_tool_use,
    signal,
)
from eval_dataset_forge.schemas import DatasetExample, DifficultyBucket, DifficultyScore

FIXTURES = ("qa_toy", "agent_tasks_toy", "retrieval_toy", "bio_toy", "safety_toy")
FIXTURE_ROOT = "datasets/fixtures"

# Keep the permutation tests honest but cheap where the exact p-value is not the point.
FAST = {"n_resamples": 1_000, "weight_draws": 0}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _load(name: str) -> list[DatasetExample]:
    return loader.load_examples(f"{FIXTURE_ROOT}/{name}/examples.jsonl")


@pytest.fixture(scope="module")
def by_fixture() -> dict[str, list[DatasetExample]]:
    return {name: _load(name) for name in FIXTURES}


@pytest.fixture(scope="module")
def pooled(by_fixture: dict[str, list[DatasetExample]]) -> list[DatasetExample]:
    """All five fixtures concatenated: 56 examples, 51 with a declared difficulty."""
    return [example for name in sorted(FIXTURES) for example in by_fixture[name]]


@pytest.fixture(scope="module")
def pooled_report(pooled: list[DatasetExample]) -> DifficultyReport:
    """The full default report. Computed once — it runs seven permutation tests."""
    return score_dataset(pooled)


@pytest.fixture(scope="module")
def pooled_calibration(pooled_report: DifficultyReport) -> CalibrationReport:
    assert pooled_report.calibration is not None
    return pooled_report.calibration


def _example(example_id: str, **overrides) -> DatasetExample:
    payload = {
        "example_id": example_id,
        "source_id": f"src-{example_id}",
        "task_family": "qa",
        "input": "What is the capital of France?",
        "expected_output": "Paris",
    }
    payload.update(overrides)
    return DatasetExample(**payload)


def _graded_corpus(shuffle_labels: bool = False, seed: int = 0) -> list[DatasetExample]:
    """A synthetic corpus whose declared difficulty really is driven by the signals.

    Twenty examples per bucket, prompt length increasing with declared difficulty. The
    positive control for the whole calibration harness: if this does not come back
    ``trustworthy``, the harness cannot detect a signal that is unambiguously there.
    """
    tags = []
    for bucket, words in (("easy", 3), ("medium", 12), ("hard", 40)):
        tags.extend([(bucket, words)] * 20)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        labels = [tag for tag, _ in tags]
        rng.shuffle(labels)
        tags = [(label, words) for label, (_, words) in zip(labels, tags, strict=True)]
    return [
        _example(
            f"syn-{index:03d}",
            input=" ".join(f"word{index}x{position}" for position in range(words)),
            expected_output="answer",
            difficulty_tags=[bucket],
        )
        for index, (bucket, words) in enumerate(tags)
    ]


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #


def test_signal_registry_matches_names_and_default_weights():
    assert SIGNAL_NAMES == tuple(s.name for s in SIGNALS)
    assert set(DEFAULT_WEIGHTS) == set(SIGNAL_NAMES)
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_every_signal_has_a_stated_claim_and_a_cap(name: str):
    found = signal(name)
    assert found.claim and found.claim[0].islower()
    assert found.cap > 0


def test_signal_lookup_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown signal"):
        signal("vibes")


def test_input_length_counts_word_tokens():
    assert raw_input_length(_example("a", input="What is 2 + 2?")) == 4.0
    assert raw_input_length(_example("a", input="")) == 0.0


@pytest.mark.parametrize(
    ("expected_output", "is_empty"),
    [("", True), ([], True), ("Paris", False), (["d1"], False), ({"a": "b"}, False)],
)
def test_output_complexity_is_zero_exactly_for_an_empty_gold_answer(expected_output, is_empty):
    value = raw_output_complexity(_example("a", expected_output=expected_output))
    assert (value == 0.0) is is_empty


def test_output_complexity_reads_structure_not_just_text():
    structured = _example("a", expected_output={"reference_solution": "def f(): pass"})
    assert raw_output_complexity(structured) > raw_output_complexity(
        _example("b", expected_output="pass")
    )


def test_reasoning_steps_ignores_bare_digits():
    """The a-priori rule: a digit is as often an operand or a year as an enumeration.

    Counting digits gives ``"What is 2 + 2?"`` two reasoning steps and the speed-of-light
    question — the corpus's one declared-hard QA example — zero. That inversion is the
    reason the rule counts word-form cardinals only.
    """
    assert raw_reasoning_steps(_example("a", input="What is 2 + 2?")) == 0.0
    assert raw_reasoning_steps(_example("b", input="Run #4471 of the 2019 experiment?")) == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What are the four nucleotide bases found in DNA?", 1.0),
        ("How many chromosomes are in a human somatic cell?", 1.0),
        ("Compute the sum and then return the median.", 2.0),  # " and " + " then "
        ("Sort the list; return the first element.", 1.0),
        ("List the two capitals.", 2.0),  # "list " + "two"
        ("What is the capital of France?", 0.0),
    ],
)
def test_reasoning_steps_counts_declared_cues(text: str, expected: float):
    assert raw_reasoning_steps(_example("a", input=text)) == expected


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, 0.0),
        ({"tool_use": True}, 1.0),
        ({"tool_use": False}, 0.0),
        ({"tools": []}, 0.0),
        ({"tools": ["search"]}, 1.0),
        ({"requires_tool": True}, 1.0),
        ({"requires_tool": 0}, 0.0),
        ({"tools": ""}, 0.0),
        ({"tools": "search"}, 1.0),
    ],
)
def test_tool_use_reads_truthiness_not_key_presence(metadata: dict, expected: float):
    """A dataset that writes ``{"tools": []}`` on every example must not make them harder."""
    assert raw_tool_use(_example("a", metadata=metadata)) == expected


def test_evidence_items_counts_retrieval_docs_and_entities():
    example = _example(
        "a",
        metadata={
            "retrieval": {"docs": [{"doc_id": "d1"}, {"doc_id": "d2"}]},
            "entities": [{"text": "insulin"}],
        },
    )
    assert raw_evidence_items(example) == 3.0
    assert raw_evidence_items(_example("b")) == 0.0
    assert raw_evidence_items(_example("c", metadata={"retrieval": "not-a-mapping"})) == 0.0


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_normalization_is_bounded_and_zero_at_zero(name: str):
    found = signal(name)
    assert found.normalize(0.0) == 0.0
    assert found.normalize(found.cap) == pytest.approx(1.0)
    assert found.normalize(found.cap * 1000) == 1.0
    assert found.normalize(-5.0) == 0.0


@pytest.mark.parametrize("name", SIGNAL_NAMES)
def test_normalization_is_monotone(name: str):
    found = signal(name)
    values = [found.normalize(raw) for raw in (0, 1, 2, 4, 8, 16, 32)]
    assert values == sorted(values)


def test_count_normalization_is_log_compressed_not_linear():
    """One more token is worth much less at 16 tokens than at 1 — diminishing returns.

    This is what stops a single 75-token reference solution from dominating a corpus of
    one-word gold answers, which a linear scale would let it do.
    """
    length = signal("input_length")
    small = length.normalize(2) - length.normalize(1)
    large = length.normalize(17) - length.normalize(16)
    assert small > 5 * large


def test_normalization_is_dataset_independent(pooled: list[DatasetExample]):
    """A score never depends on the corpus it was computed in — only the cut points do."""
    subset = pooled[:10]
    assert [raw_score(e) for e in subset] == [raw_score(e) for e in pooled][:10]


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


def test_resolve_weights_defaults_to_the_documented_vector():
    assert resolve_weights() == DEFAULT_WEIGHTS
    assert resolve_weights(None) is not DEFAULT_WEIGHTS  # a copy, not the module constant


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ({"vibes": 1.0}, "unknown signal"),
        ({"input_length": -1.0, "reasoning_steps": 2.0}, "negative weight"),
        ({"input_length": 0.5}, "must sum to 1.0"),
        ({name: 1.0 for name in SIGNAL_NAMES}, "must sum to 1.0"),
    ],
)
def test_resolve_weights_rejects_a_malformed_vector(weights: dict, match: str):
    with pytest.raises(ValueError, match=match):
        resolve_weights(weights)


def test_a_partial_weight_vector_zero_fills_the_rest():
    resolved = resolve_weights({"input_length": 0.4, "reasoning_steps": 0.6})
    assert resolved["tool_use"] == 0.0
    assert sum(resolved.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The explainability invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURES)
def test_score_is_exactly_the_sum_of_its_components(
    name: str, by_fixture: dict[str, list[DatasetExample]]
):
    """Bit-exact, not approximate. Rounding either side breaks it — so neither is rounded."""
    for example in by_fixture[name]:
        components = weighted_components(example)
        assert sum(components.values()) == raw_score(example)


@pytest.mark.parametrize("name", FIXTURES)
def test_reported_scores_reproduce_their_components(
    name: str, by_fixture: dict[str, list[DatasetExample]]
):
    report = score_dataset(by_fixture[name], **FAST)
    for score in report.scores:
        assert sum(score.components.values()) == score.score
        assert set(score.components) == set(SIGNAL_NAMES)


@pytest.mark.parametrize("name", FIXTURES)
def test_scores_stay_inside_the_unit_interval(
    name: str, by_fixture: dict[str, list[DatasetExample]]
):
    assert all(0.0 <= raw_score(e) <= 1.0 for e in by_fixture[name])


def test_normalized_and_raw_signals_agree_on_the_registry():
    example = _example("a")
    assert set(raw_signals(example)) == set(normalized_signals(example)) == set(SIGNAL_NAMES)


# --------------------------------------------------------------------------- #
# No circularity: the score never reads difficulty_tags
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tags", [[], ["easy"], ["hard"], ["hard", "ambiguous"], ["coding"]])
def test_the_score_never_reads_difficulty_tags(tags: list[str]):
    """Rewriting the declaration must not move the score, or calibration measures itself."""
    baseline = raw_score(_example("a", difficulty_tags=[]))
    assert raw_score(_example("a", difficulty_tags=tags)) == baseline


def test_leaking_the_declared_bucket_into_the_score_inflates_concordance(
    pooled: list[DatasetExample], pooled_calibration: CalibrationReport
):
    """Why the exclusion above is load-bearing, quantified.

    ``docs/DIFFICULTY.md`` originally proposed using the hand-assigned difficulty as *both*
    a scoring signal and the sanity check. Blending it in at a weight of 0.15 lifts the
    reported rank correlation from +0.10 to +0.49 — a null heuristic reading as a working
    one — and at 0.30 to +0.77.
    """
    labelled = [e for e in pooled if declared_ordinal(e) is not None]
    target = np.array([declared_ordinal(e) for e in labelled], dtype=float)
    honest = np.array([raw_score(e) for e in labelled])

    assert spearman(honest, target) == pooled_calibration.rho
    assert pooled_calibration.rho < 0.15

    leaked_15 = 0.85 * honest + 0.15 * (target / 2.0)
    leaked_30 = 0.70 * honest + 0.30 * (target / 2.0)
    assert spearman(leaked_15, target) > 0.45
    assert spearman(leaked_30, target) > 0.75


# --------------------------------------------------------------------------- #
# Buckets
# --------------------------------------------------------------------------- #


def test_quantile_cutpoints_match_numpy():
    values = [0.1, 0.4, 0.4, 0.9, 1.5, 2.2, 2.2, 3.0]
    expected = tuple(float(v) for v in np.quantile(values, [1 / 3, 2 / 3]))
    assert quantile_cutpoints(values) == pytest.approx(expected)


def test_quantile_cutpoints_on_degenerate_inputs():
    assert quantile_cutpoints([]) == (0.0, 0.0)
    assert quantile_cutpoints([0.5]) == (0.5, 0.5)


def test_bucket_of_uses_half_open_intervals():
    cuts = (0.3, 0.6)
    assert bucket_of(0.299, cuts) is DifficultyBucket.EASY
    assert bucket_of(0.3, cuts) is DifficultyBucket.MEDIUM
    assert bucket_of(0.599, cuts) is DifficultyBucket.MEDIUM
    assert bucket_of(0.6, cuts) is DifficultyBucket.HARD


def test_equal_scores_always_receive_equal_buckets():
    """A boundary never splits a tie group — otherwise a bucket would depend on file order."""
    scores = [0.1] * 5 + [0.5] * 20 + [0.9] * 5
    for method in ("quantile", "fixed"):
        assigned = assign_buckets(scores, method)
        for value in set(scores):
            same = {b for s, b in zip(scores, assigned, strict=True) if s == value}
            assert len(same) == 1


def _tie_groups(report: DifficultyReport) -> list[list[str]]:
    by_score: dict[float, list[str]] = {}
    for score in report.scores:
        by_score.setdefault(score.score, []).append(score.example_id)
    return sorted(
        (sorted(ids) for ids in by_score.values() if len(ids) > 1),
        key=lambda ids: (-len(ids), ids),
    )


def test_the_pooled_corpus_has_two_tie_groups_of_four(pooled_report: DifficultyReport):
    assert len({score.score for score in pooled_report.scores}) == 39
    groups = _tie_groups(pooled_report)
    assert [g for g in groups if len(g) == 4] == [
        ["bio-003", "bio-006", "saf-004", "saf-005"],
        ["saf-001", "saf-002", "saf-006", "saf-008"],
    ]
    for group in groups:
        assert len({pooled_report.bucket_of_example[e] for e in group}) == 1


def test_calibration_scores_examples_the_same_way_the_report_does(
    pooled: list[DatasetExample], pooled_calibration: CalibrationReport
):
    """A regression test for a bug that hid in the 17th decimal place.

    ``calibrate`` used to build its scores with ``matrix @ weights``. numpy's pairwise
    summation and Python's sequential ``sum`` differ by ~5.6e-17 on the same five
    contributions — enough to split the four-way tie ``{bio-003, bio-006, saf-004,
    saf-005}`` into two pairs, and to move the published rho from +0.097467 to +0.097449.
    The report's rho must describe the report's scores.
    """
    labelled = [e for e in pooled if declared_ordinal(e) is not None]
    target = [declared_ordinal(e) for e in labelled]
    expected = spearman([raw_score(e) for e in labelled], target)
    assert pooled_calibration.rho == expected  # bit-exact, not approx


def test_a_matrix_product_would_have_split_a_real_tie_group(pooled: list[DatasetExample]):
    """Pins the mechanism above, so nobody "optimizes" ``calibrate`` back into the bug."""
    from eval_dataset_forge.difficulty.heuristics import score_matrix

    labelled = [e for e in pooled if declared_ordinal(e) is not None]
    _, rows = score_matrix(labelled)
    dotted = np.array(rows) @ np.array([DEFAULT_WEIGHTS[n] for n in SIGNAL_NAMES])
    summed = np.array([raw_score(e) for e in labelled])

    assert np.allclose(dotted, summed, atol=1e-15)
    assert not np.array_equal(dotted, summed)
    assert len(set(summed.tolist())) < len(set(dotted.tolist()))


def test_quantile_bucketing_cannot_hit_equal_frequencies_and_reports_what_it_achieved(
    pooled_report: DifficultyReport,
):
    assert pooled_report.distribution == {"easy": 18, "medium": 19, "hard": 19}
    assert sum(pooled_report.distribution.values()) == pooled_report.n_examples


def test_fixed_cutoffs_leave_the_hard_bucket_empty_on_the_pooled_corpus(
    pooled: list[DatasetExample],
):
    """An absolute cut point on a log-compressed score is unreachable for short answers."""
    report = score_dataset(pooled, bucketing="fixed", **FAST)
    assert report.cutpoints == FIXED_CUTOFFS
    assert report.distribution == {"easy": 45, "medium": 11, "hard": 0}
    assert max(s.score for s in report.scores) < FIXED_CUTOFFS[1]
    assert any("'hard' bucket is empty" in w for w in report.warnings)


def test_marginal_bucketing_apportions_the_declared_marginal_to_the_corpus_size(
    pooled: list[DatasetExample],
):
    """51 examples declare a difficulty; 56 get bucketed. The target must be rescaled.

    Feeding the raw 29/19/3 counts to a 56-example corpus would dump all five undeclared
    examples into ``hard``. Largest-remainder apportionment sends 29/19/3 to 32/21/3.
    """
    declared = {k: v for k, v in declared_distribution(pooled).items() if k != "unlabelled"}
    assert declared == {"easy": 29, "medium": 19, "hard": 3}
    assert scale_counts(declared, 56) == {"easy": 32, "medium": 21, "hard": 3}

    report = score_dataset(pooled, bucketing="marginal", **FAST)
    assert sum(report.distribution.values()) == 56
    assert report.distribution["hard"] < 10  # not the 11 the unscaled target produced


@pytest.mark.parametrize(
    ("counts", "total", "expected"),
    [
        ({"easy": 29, "medium": 19, "hard": 3}, 51, {"easy": 29, "medium": 19, "hard": 3}),
        ({"easy": 1, "medium": 1, "hard": 1}, 6, {"easy": 2, "medium": 2, "hard": 2}),
        ({"easy": 2, "medium": 1, "hard": 0}, 9, {"easy": 6, "medium": 3, "hard": 0}),
        ({"easy": 0, "medium": 0, "hard": 0}, 5, {"easy": 0, "medium": 0, "hard": 0}),
        ({"easy": 1, "medium": 1, "hard": 1}, 0, {"easy": 0, "medium": 0, "hard": 0}),
    ],
)
def test_scale_counts_apportions_by_largest_remainder(counts, total, expected):
    scaled = scale_counts(counts, total)
    assert scaled == expected
    assert sum(scaled.values()) == (total if sum(counts.values()) else 0)


def test_marginal_cutpoints_place_the_requested_counts_when_scores_are_distinct():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    cuts = marginal_cutpoints(scores, {"easy": 3, "medium": 2, "hard": 1})
    assigned = [bucket_of(s, cuts) for s in scores]
    assert bucket_distribution(assigned) == {"easy": 3, "medium": 2, "hard": 1}


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"easy": 0, "medium": 0, "hard": 4}, {"easy": 0, "medium": 0, "hard": 4}),
        ({"easy": 4, "medium": 0, "hard": 0}, {"easy": 4, "medium": 0, "hard": 0}),
        ({"easy": 0, "medium": 4, "hard": 0}, {"easy": 0, "medium": 4, "hard": 0}),
    ],
)
def test_marginal_cutpoints_handle_degenerate_targets(counts: dict, expected: dict):
    scores = [0.1, 0.2, 0.3, 0.4]
    cuts = marginal_cutpoints(scores, counts)
    assert bucket_distribution([bucket_of(s, cuts) for s in scores]) == expected


def test_marginal_bucketing_requires_target_counts():
    with pytest.raises(ValueError, match="requires target_counts"):
        cutpoints_for([0.1, 0.2], "marginal")


def test_unknown_bucketing_method_is_rejected():
    with pytest.raises(ValueError, match="unknown bucketing method"):
        cutpoints_for([0.1, 0.2], "vibes")


def test_bucket_distribution_always_lists_all_three_buckets():
    """An empty bucket is the finding; a dict that omitted the key would hide it."""
    assert bucket_distribution([DifficultyBucket.EASY]) == {"easy": 1, "medium": 0, "hard": 0}
    assert bucket_distribution([]) == {"easy": 0, "medium": 0, "hard": 0}


# --------------------------------------------------------------------------- #
# The declared bucket
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ([], None),
        (["easy"], DifficultyBucket.EASY),
        (["coding", "medium"], DifficultyBucket.MEDIUM),
        (["hard", "ambiguous"], DifficultyBucket.HARD),
        (["easy", "hard"], DifficultyBucket.HARD),  # the hardest declaration wins
        (["EASY"], DifficultyBucket.EASY),  # case-folded
        (["ambiguous"], None),  # not a bucket
    ],
)
def test_declared_bucket_reads_only_bucket_tags(tags: list[str], expected):
    assert declared_bucket(_example("a", difficulty_tags=tags)) is expected


def test_declared_ordinal_matches_the_bucket_order():
    assert declared_ordinal(_example("a", difficulty_tags=["easy"])) == 0
    assert declared_ordinal(_example("a", difficulty_tags=["medium"])) == 1
    assert declared_ordinal(_example("a", difficulty_tags=["hard"])) == 2
    assert declared_ordinal(_example("a")) is None


def test_splits_reexports_the_same_declared_bucket_function():
    """Milestone 8 moved it out of ``splits``; the two milestones must not diverge."""
    from eval_dataset_forge.splits import declared_bucket as from_splits
    from eval_dataset_forge.splits.stratified import DIFFICULTY_ORDER as from_splits_order

    assert from_splits is declared_bucket
    assert from_splits_order == DIFFICULTY_ORDER


def test_declared_distribution_counts_unlabelled_separately(pooled: list[DatasetExample]):
    assert declared_distribution(pooled) == {
        "easy": 29,
        "medium": 19,
        "hard": 3,
        "unlabelled": 5,
    }


def test_distribution_by_split_partitions_the_corpus(pooled: list[DatasetExample]):
    buckets = assign_buckets([raw_score(e) for e in pooled])
    table = distribution_by_split(pooled, buckets)
    assert sum(sum(row.values()) for row in table.values()) == len(pooled)
    assert set(table) == {"train", "dev", "test"}


# --------------------------------------------------------------------------- #
# Rank statistics
# --------------------------------------------------------------------------- #


def test_rankdata_averages_ties():
    assert list(rankdata([10.0, 20.0, 20.0, 30.0])) == [1.0, 2.5, 2.5, 4.0]
    assert list(rankdata([5.0, 5.0, 5.0])) == [2.0, 2.0, 2.0]
    assert list(rankdata([])) == []


def test_spearman_matches_pearson_on_ranks():
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    y = rng.normal(size=40)
    expected = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    assert spearman(x, y) == pytest.approx(expected)


def test_spearman_at_the_extremes():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    assert math.isnan(spearman([1, 2], [1, 2]))  # n < 3


def test_spearman_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        spearman([1, 2, 3], [1, 2])


def test_permutation_p_is_deterministic_given_a_seed():
    x, y = list(range(30)), [i % 3 for i in range(30)]
    assert permutation_p(x, y, n_resamples=500, seed=3) == permutation_p(
        x, y, n_resamples=500, seed=3
    )


def test_permutation_p_is_tiny_for_a_perfect_correlation():
    x = list(range(40))
    assert permutation_p(x, x, n_resamples=2_000, seed=0) < 0.002


@pytest.mark.parametrize("n_resamples", [100, 400, 1_000])
def test_permutation_p_is_never_zero(n_resamples: int):
    """The Phipson-Smyth floor. ``p = 0`` would claim certainty no resampling can support."""
    p = permutation_p(list(range(30)), list(range(30)), n_resamples=n_resamples, seed=0)
    assert p == pytest.approx(1.0 / (1.0 + n_resamples))
    assert 0.0 < p <= 1.0


def test_permutation_p_is_uniform_under_the_null():
    """A valid p-value is uniform on truly null data, so its mean must be 0.5.

    The mean converges far faster than the 5% tail does: at 200 null pairings its standard
    error is 0.020, while the tail rate's is 0.015 on a target of 0.05 — a 3-sigma band
    there spans [0.004, 0.096] and would pass almost anything. So the mean is asserted
    tightly and the tail only loosely. Seeded, hence deterministic.
    """
    rng = np.random.default_rng(11)
    values = np.array(
        [
            permutation_p(rng.normal(size=40), rng.normal(size=40), n_resamples=400, seed=trial)
            for trial in range(200)
        ]
    )
    assert values.mean() == pytest.approx(0.5, abs=0.06)  # ~3 standard errors
    assert np.median(values) == pytest.approx(0.5, abs=0.08)
    assert 0.004 <= float((values <= 0.05).mean()) <= 0.10


def test_permutation_p_detects_a_strong_effect_through_heavy_ties():
    """The declared target has three distinct values, so ties are the normal case here."""
    x = list(range(50))
    y = [0.0] * 25 + [1.0] * 25
    assert spearman(x, y) > 0.8
    assert permutation_p(x, y, n_resamples=2_000, seed=0) < 0.002


def test_required_n_matches_a_hand_computed_fisher_z():
    # n = ((1.959964 + 0.841621) / artanh(0.3))^2 + 3 = 84.07... -> 85
    assert required_n(0.3) == 85
    assert required_n(-0.3) == required_n(0.3)  # sign-free
    assert required_n(0.0) is None
    assert required_n(1.0) is None
    assert required_n(float("nan")) is None


def test_required_n_shrinks_as_the_effect_grows():
    sizes = [required_n(rho) for rho in (0.1, 0.2, 0.4, 0.8)]
    assert sizes == sorted(sizes, reverse=True)


def test_eta_squared_at_the_extremes():
    # Groups separate the values perfectly -> all variance is between groups.
    assert eta_squared([1, 1, 5, 5], ["a", "a", "b", "b"]) == pytest.approx(1.0)
    # Both groups have the same mean -> no variance is between groups.
    assert eta_squared([1, 5, 1, 5], ["a", "a", "b", "b"]) == pytest.approx(0.0)
    assert eta_squared([3, 3, 3], ["a", "b", "c"]) == 0.0
    assert eta_squared([], []) == 0.0


# --------------------------------------------------------------------------- #
# The headline result
# --------------------------------------------------------------------------- #


def test_the_heuristic_ranking_is_not_distinguishable_from_a_random_one(
    pooled_calibration: CalibrationReport,
):
    """The milestone's headline. 51 labelled examples, 10000 permutations, seed 0."""
    assert pooled_calibration.n_labelled == 51
    assert pooled_calibration.rho == pytest.approx(0.097, abs=0.005)
    assert pooled_calibration.p_value > 0.4
    assert not pooled_calibration.significant
    assert not pooled_calibration.trustworthy


def test_no_point_on_the_weight_simplex_rescues_the_heuristic(
    pooled_calibration: CalibrationReport,
):
    """So the null is a property of the signals, not of the shipped weight vector.

    The best of 2000 Dirichlet draws is chosen *in sample*, which makes it an upper bound
    on what any reweighting could honestly achieve — and it still lands below the +0.39
    that 51 examples would need to clear alpha.
    """
    sweep = pooled_calibration.weight_sensitivity
    assert sweep["draws"] == 2_000
    assert sweep["max"] < 0.35
    assert sweep["fraction_below_0.2"] > 0.9
    assert abs(sweep["median"]) < 0.1


def test_the_score_is_a_task_family_detector(pooled_calibration: CalibrationReport):
    """43% of the score's variance is task_family; 1% of the declared difficulty's is."""
    assert pooled_calibration.score_eta_squared_family > 0.40
    assert pooled_calibration.declared_eta_squared_family < 0.05
    assert any("task-family detector" in w for w in pooled_calibration.warnings)


def test_the_within_family_correlation_collapses(pooled_calibration: CalibrationReport):
    """Holding family constant leaves almost nothing — the control for the family effect."""
    assert abs(pooled_calibration.within_family_mean_rho) < 0.10
    assert set(pooled_calibration.within_family_rho) == {
        "agent_task",
        "bio",
        "qa",
        "retrieval",
        "safety",
    }
    # On retrieval the score points the wrong way outright.
    assert pooled_calibration.within_family_rho["retrieval"] < -0.3


def test_marginal_matched_accuracy_does_not_beat_calling_everything_easy(
    pooled_calibration: CalibrationReport,
):
    """Bucket accuracy with the marginal mismatch removed, so only the ranking is judged."""
    assert pooled_calibration.marginal_accuracy == pytest.approx(0.529, abs=0.005)
    assert pooled_calibration.majority_baseline == pytest.approx(0.569, abs=0.005)
    assert pooled_calibration.random_ranking_accuracy == pytest.approx(0.466, abs=0.005)
    assert not pooled_calibration.beats_majority
    assert pooled_calibration.marginal_accuracy_p > 0.05


def test_two_of_the_three_declared_hard_examples_are_bucketed_easy(
    pooled: list[DatasetExample], pooled_calibration: CalibrationReport
):
    """The confusion matrix is over the *marginal-matched* bucketing, not the default one.

    Under ``marginal`` — the bucketing that gives the ranking its fairest hearing, since
    the predicted counts match the declared ones — ``qa-014`` and ``ret-005`` land in
    ``easy``. Under the default ``quantile`` bucketing they land in ``medium``, because
    quantile cuts hand out 19 ``hard`` slots instead of 3. Neither bucketing recovers them.
    """
    hard_row = pooled_calibration.confusion["hard"]
    assert sum(hard_row.values()) == 3
    assert hard_row == {"easy": 2, "medium": 0, "hard": 1}

    quantile = score_dataset(pooled, **FAST)
    declared_hard = [e for e, b in quantile.declared_by_id.items() if b == "hard"]
    assert sorted(declared_hard) == ["agent-008", "qa-014", "ret-005"]
    assert str(quantile.bucket_of_example["agent-008"]) == "hard"
    assert str(quantile.bucket_of_example["qa-014"]) == "medium"
    assert str(quantile.bucket_of_example["ret-005"]) == "medium"


def test_the_corpus_is_far_too_small_to_detect_an_effect_this_size(
    pooled_calibration: CalibrationReport,
):
    """Turns the null into a statement about the corpus, not only about the heuristic."""
    assert pooled_calibration.required_n_for_power is not None
    assert pooled_calibration.required_n_for_power > 700
    assert pooled_calibration.n_labelled < MIN_CALIBRATION_N * 2


@pytest.mark.parametrize("name", FIXTURES)
def test_no_individual_fixture_calibrates_either(
    name: str, by_fixture: dict[str, list[DatasetExample]]
):
    report = score_dataset(by_fixture[name], **FAST)
    assert not report.trustworthy
    assert report.calibration is not None
    assert report.calibration.underpowered


# --------------------------------------------------------------------------- #
# The positive control
# --------------------------------------------------------------------------- #


def test_the_harness_certifies_a_corpus_whose_difficulty_really_is_surface_driven():
    """Without this, "not calibrated" could just mean "the harness always says no"."""
    report = score_dataset(_graded_corpus(), weight_draws=0)
    result = report.calibration
    assert result is not None
    assert result.n_labelled == 60
    assert result.rho > 0.9
    assert result.p_value < 0.01
    assert result.significant and not result.underpowered and result.beats_majority
    assert report.trustworthy is True
    assert not any("not distinguishable" in w for w in report.warnings)


def test_the_same_corpus_with_shuffled_labels_is_not_certified():
    report = score_dataset(_graded_corpus(shuffle_labels=True, seed=5), weight_draws=0)
    assert report.calibration is not None
    assert abs(report.calibration.rho) < 0.35
    assert not report.trustworthy


def test_a_corpus_with_no_declared_difficulty_cannot_be_calibrated():
    """Absence of a target is reported as absence of evidence, never as a pass."""
    examples = [
        _example(f"n-{i}", input=" ".join(["word"] * (i + 1))) for i in range(10)
    ]
    report = score_dataset(examples, **FAST)
    assert report.calibration is not None
    assert report.calibration.n_labelled == 0
    assert report.trustworthy is False
    assert report.warnings == (
        "no example declares a difficulty tag; the heuristic cannot be calibrated",
    )


def test_a_constant_score_corpus_collapses_into_one_bucket_and_says_so():
    """Every score tied: the tie rule sends them all one way, and two buckets go empty."""
    examples = [_example(f"c-{i}", difficulty_tags=["easy"]) for i in range(6)]
    report = score_dataset(examples, **FAST)
    assert len({s.score for s in report.scores}) == 1
    assert report.distribution == {"easy": 0, "medium": 0, "hard": 6}
    empty = [w for w in report.warnings if "bucket is empty" in w]
    assert len(empty) == 2


def test_an_undefined_correlation_is_not_reported_as_indistinguishable_from_chance():
    """``rho = nan`` means there is nothing to compare against a random ranking.

    Reporting "rho=+nan (p=nan) is not distinguishable from a random ranking" would be a
    category error, and it is what the first implementation printed.
    """
    examples = [_example(f"c-{i}", difficulty_tags=["easy"]) for i in range(6)]
    report = score_dataset(examples, **FAST)
    assert report.calibration is not None
    assert math.isnan(report.calibration.rho)
    assert any("rank correlation is undefined" in w for w in report.warnings)
    assert not any("nan" in w for w in report.warnings)
    assert not any("distinguishable from a random ranking" in w for w in report.warnings)
    assert not report.trustworthy


def test_an_inert_signal_says_why_it_is_inert():
    """Zero variance at zero ("fires on nothing") is not zero variance at a constant.

    ``tool_use`` on the fixtures is the first; ``input_length`` on a corpus of identical
    prompts is the second. Calling the second "fires on 0 examples" would be false.
    """
    identical = [
        _example(f"c-{i}", input="one two three", difficulty_tags=["easy" if i < 3 else "hard"])
        for i in range(6)
    ]
    report = score_dataset(identical, **FAST)
    assert report.calibration is not None
    assert "input_length" in report.calibration.inert_signals
    assert "tool_use" in report.calibration.inert_signals

    by_name = {d.name: d for d in report.calibration.per_signal}
    assert by_name["input_length"].nonzero == 6
    assert by_name["tool_use"].nonzero == 0

    warnings = "\n".join(report.warnings)
    assert "'input_length' has zero variance on this dataset (takes the same value" in warnings
    assert "'tool_use' has zero variance on this dataset (fires on 0 of 6 examples)" in warnings


def test_the_fixtures_inert_signal_is_reported_as_firing_on_nothing(
    pooled_calibration: CalibrationReport,
):
    assert any(
        "'tool_use' has zero variance on this dataset (fires on 0 of 51 examples)" in w
        for w in pooled_calibration.warnings
    )


def test_fewer_than_three_labels_short_circuits_the_calibration():
    examples = [_example("a", difficulty_tags=["easy"]), _example("b", difficulty_tags=["hard"])]
    result = calibrate(examples, **FAST)
    assert result.n_labelled == 2
    assert math.isnan(result.rho)
    assert not result.trustworthy


# --------------------------------------------------------------------------- #
# Honest negatives
# --------------------------------------------------------------------------- #


def test_tool_use_fires_on_no_example_and_is_reported_inert_not_credited(
    pooled: list[DatasetExample], pooled_calibration: CalibrationReport
):
    """A weighted signal with zero variance adds a constant. It changes no ranking."""
    assert all(raw_tool_use(e) == 0.0 for e in pooled)
    assert pooled_calibration.inert_signals == ["tool_use"]
    assert DEFAULT_WEIGHTS["tool_use"] > 0  # it is weighted, and it still does nothing
    assert any("zero variance" in w for w in pooled_calibration.warnings)


def test_an_inert_signal_changes_no_bucket(pooled: list[DatasetExample]):
    """Proof of the claim above: reweighting tool_use away leaves every bucket in place."""
    with_tool = score_dataset(pooled, **FAST)
    without = {
        "input_length": 0.20 / 0.85,
        "output_complexity": 0.20 / 0.85,
        "reasoning_steps": 0.30 / 0.85,
        "evidence_items": 0.15 / 0.85,
        "tool_use": 0.0,
    }
    rescaled = score_dataset(pooled, weights=resolve_weights(without), **FAST)
    assert with_tool.bucket_of_example == rescaled.bucket_of_example


def test_two_signals_are_anti_correlated_with_declared_difficulty(
    pooled_calibration: CalibrationReport,
):
    """They push the score the wrong way, and the report says which."""
    assert sorted(pooled_calibration.anticorrelated_signals) == [
        "evidence_items",
        "output_complexity",
    ]
    by_name = {d.name: d for d in pooled_calibration.per_signal}
    assert by_name["output_complexity"].rho < 0
    assert by_name["evidence_items"].rho < 0


def test_input_length_clears_alpha_but_not_bonferroni(pooled_calibration: CalibrationReport):
    """Five signals were tested. Quoting the best one's uncorrected p would be fishing."""
    by_name = {d.name: d for d in pooled_calibration.per_signal}
    length = by_name["input_length"]
    assert length.p_value < 0.05
    assert length.p_bonferroni > 0.05
    assert length.rho > 0.25


def test_a_missing_gold_answer_makes_an_example_look_easy(pooled_report: DifficultyReport):
    """Milestone 6's ``MISSING`` label defect corrupts Milestone 8's score.

    An absent gold answer scores zero on ``output_complexity`` — the floor — so the
    examples whose labels most need attention sink to the easy end of the ranking.
    """
    assert set(pooled_report.empty_answer_example_ids) == {
        "agent-009",
        "bio-008",
        "qa-010",
        "ret-004",
        "ret-005",
        "saf-009",
    }
    ranked = sorted(pooled_report.scores, key=lambda s: s.score)
    rank = {score.example_id: index + 1 for index, score in enumerate(ranked)}
    assert rank["agent-009"] == 1  # the single easiest-looking example in the corpus
    assert rank["qa-010"] == 2
    easy = str(DifficultyBucket.EASY)
    sunk = [
        eid
        for eid in pooled_report.empty_answer_example_ids
        if str(pooled_report.bucket_of_example[eid]) == easy
    ]
    assert sorted(sunk) == ["agent-009", "bio-008", "qa-010"]
    assert any("makes the example look easy" in w for w in pooled_report.warnings)


def test_the_highest_scoring_example_is_a_fizzbuzz_paraphrase_declared_easy(
    pooled_report: DifficultyReport,
):
    """The single clearest illustration of what the score actually measures."""
    hardest = max(pooled_report.scores, key=lambda s: s.score)
    assert hardest.example_id == "agent-002"
    assert pooled_report.declared_by_id["agent-002"] == "easy"


def test_the_declared_hard_bucket_is_too_small_for_a_per_bucket_claim(
    pooled_calibration: CalibrationReport,
):
    assert pooled_calibration.declared_counts["hard"] == 3
    assert any("per-bucket statistics" in w for w in pooled_calibration.warnings)


# --------------------------------------------------------------------------- #
# Report shape, determinism, IO
# --------------------------------------------------------------------------- #


def test_report_counts_are_internally_consistent(pooled_report: DifficultyReport):
    counts = pooled_report.counts()
    assert counts["examples"] == 56
    assert counts["labelled"] + counts["unlabelled"] == 56
    assert sum(pooled_report.distribution.values()) == 56
    assert counts["disagreements"] == len(pooled_report.disagreements)
    assert counts["disagreements"] <= counts["labelled"]


def test_every_score_validates_against_the_schema(pooled_report: DifficultyReport):
    for score in pooled_report.scores:
        assert isinstance(score, DifficultyScore)
        assert score.method == f"{METHOD}+quantile"
        round_tripped = DifficultyScore.model_validate(score.model_dump(mode="json"))
        assert round_tripped == score


def test_report_to_dict_is_json_serializable(pooled_report: DifficultyReport):
    payload = json.loads(json.dumps(pooled_report.to_dict()))
    assert payload["trustworthy"] is False
    assert payload["calibration"]["n_labelled"] == 51
    assert payload["counts"]["examples"] == 56
    assert payload["distribution"] == {"easy": 18, "medium": 19, "hard": 19}


def test_scoring_is_deterministic_and_does_not_mutate_its_input(pooled: list[DatasetExample]):
    before = [e.model_copy(deep=True) for e in pooled]
    first = score_dataset(pooled, **FAST)
    second = score_dataset(pooled, **FAST)
    assert first.to_dict() == second.to_dict()
    assert pooled == before


def test_a_different_seed_moves_only_the_permutation_p_values(pooled: list[DatasetExample]):
    a = score_dataset(pooled, seed=0, **FAST)
    b = score_dataset(pooled, seed=123, **FAST)
    assert a.score_of == b.score_of
    assert a.distribution == b.distribution
    assert a.calibration.rho == b.calibration.rho
    assert a.calibration.p_value == pytest.approx(b.calibration.p_value, abs=0.06)


def test_write_report_is_byte_identical_across_runs(pooled: list[DatasetExample], tmp_path):
    report = score_dataset(pooled, **FAST)
    first = write_report(report, tmp_path / "a")
    second = write_report(score_dataset(pooled, **FAST), tmp_path / "b")
    for key in ("report", "scores"):
        assert first[key].read_bytes() == second[key].read_bytes()

    payload = json.loads(first["report"].read_text())
    assert payload["method"] == f"{METHOD}+quantile"
    lines = first["scores"].read_text().splitlines()
    assert len(lines) == 56
    assert json.loads(lines[0])["example_id"] == report.scores[0].example_id


def test_format_report_renders_the_verdict_and_the_warnings(pooled_report: DifficultyReport):
    text = format_report(pooled_report)
    assert "NOT CALIBRATED" in text
    assert "tool_use" in text and "INERT" in text
    assert "weight simplex sweep" in text
    assert "warnings" in text


def test_format_report_renders_nan_as_not_available(by_fixture):
    """A single-family dataset has an undefined family baseline. It must not print +nan."""
    text = format_report(score_dataset(by_fixture["qa_toy"], **FAST))
    assert "family-only baseline: n/a" in text
    assert "nan" not in text.lower().replace("n/a", "")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_difficulty_is_a_real_subcommand_not_a_placeholder():
    args = build_parser().parse_args(["difficulty", "--dataset", "x.jsonl"])
    assert args.bucketing == "quantile"
    assert args.seed == 0
    assert args.fail_on_uncalibrated is False


def test_cli_difficulty_exits_zero_without_the_gate(capsys):
    code = main(
        ["difficulty", "--dataset", f"{FIXTURE_ROOT}/qa_toy/examples.jsonl", "--resamples", "200"]
    )
    assert code == 0
    assert "NOT CALIBRATED" in capsys.readouterr().out


def test_cli_difficulty_gate_exits_one_on_the_uncalibrated_fixtures(capsys):
    code = main(
        [
            "difficulty",
            "--dataset",
            f"{FIXTURE_ROOT}/qa_toy/examples.jsonl",
            "--resamples",
            "200",
            "--fail-on-uncalibrated",
        ]
    )
    assert code == 1
    assert "must not be used as a difficulty label" in capsys.readouterr().out


def test_cli_difficulty_gate_exits_zero_on_a_calibrated_dataset(tmp_path, capsys):
    path = tmp_path / "graded.jsonl"
    path.write_text(
        "".join(
            json.dumps(e.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for e in _graded_corpus()
        ),
        encoding="utf-8",
    )
    code = main(["difficulty", "--dataset", str(path), "--fail-on-uncalibrated"])
    assert code == 0
    assert "TRUSTWORTHY" in capsys.readouterr().out


def test_cli_difficulty_json_and_out(tmp_path, capsys):
    out = tmp_path / "artifacts"
    code = main(
        [
            "difficulty",
            "--dataset",
            f"{FIXTURE_ROOT}/bio_toy/examples.jsonl",
            "--json",
            "--resamples",
            "200",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured[: captured.rindex("}") + 1])
    assert payload["counts"]["examples"] == 10
    assert (out / "difficulty_report.json").is_file()
    assert (out / "difficulty_scores.jsonl").is_file()


def test_cli_difficulty_honours_the_bucketing_choice(capsys):
    code = main(
        [
            "difficulty",
            "--dataset",
            f"{FIXTURE_ROOT}/qa_toy/examples.jsonl",
            "--bucketing",
            "fixed",
            "--resamples",
            "200",
        ]
    )
    assert code == 0
    assert "surface_v1+fixed" in capsys.readouterr().out


def test_cli_difficulty_accepts_custom_weights(capsys):
    code = main(
        [
            "difficulty",
            "--dataset",
            f"{FIXTURE_ROOT}/qa_toy/examples.jsonl",
            "--weights",
            "input_length=0.5,reasoning_steps=0.5",
            "--resamples",
            "200",
        ]
    )
    assert code == 0
    assert "difficulty: 14 examples" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["input_length", "input_length=abc"])
def test_parse_weights_rejects_malformed_input(text: str):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_weights(text)


def test_parse_weights_accepts_named_pairs():
    assert _parse_weights("input_length=0.4, reasoning_steps=0.6") == {
        "input_length": 0.4,
        "reasoning_steps": 0.6,
    }
