"""Tests for statistical uncertainty & ranking stability (Milestone 9).

Covered here:

  * **Resampling units** — the ratio estimator against a naive reference, the shared draw
    matrix that makes comparisons *paired*, determinism, and the equivalence of block
    resampling with example resampling when no block has more than one member.
  * **The headline result** — a coverage simulation with known ground truth. On clustered
    data the i.i.d. percentile bootstrap covers 0.857 of the time against a nominal 0.95;
    resampling blocks recovers 0.923. Both are measured here, not asserted, and the test
    fails if the gap disappears.
  * **The positive control on the coverage harness** — on genuinely i.i.d. data the same
    percentile interval covers 0.95. Without it, "the bootstrap under-covers" would be
    indistinguishable from a broken coverage harness.
  * **BCa is not free** — it repairs the clustered interval at an extreme accuracy and
    *breaks* the i.i.d. one (0.929 -> 0.828 at p=0.95, with 6% of trials falling back). That
    is why it is offered and not defaulted, and the test pins both directions.
  * **The win fraction is not a p-value** — the "A wins 95% of resamples" rule has a
    measured type-I error of 0.105 against a nominal 0.05; the 0.975 threshold measures
    0.051. The cluster-aware sign-flip test measures 0.023: conservative, because paired
    binary differences are mostly ties.

The simulations here run at reduced settings (600–1500 trials, ``B=399``) so the suite stays
under 10 seconds, which costs a little accuracy: ``docs/STATISTICS.md`` quotes the same
experiments at 3000 trials and ``B=999``, where the clustered i.i.d. coverage is 0.882 rather
than 0.857 and the win-fraction type-I error is 0.114 rather than 0.105. The tests therefore
assert *bounds* that both settings clear, never the point values.
  * **ICC = 1 by construction** — an exact-duplicate cluster scored by a deterministic system
    has within-cluster correlation exactly 1. Not an empirical fact about these fixtures; a
    property of duplicates, and the reason the cluster bootstrap is the default.
  * **The design effect is not an artifact** — simulating the *same* block structure with
    ``icc=0`` returns a design effect of ~1.0. The widening tracks the dependence, not the
    choice of unit.
  * **Deletion beats resampling** — ``min_units_to_flip`` is exact (brute-forced against all
    subsets on small inputs), and on the demo it finds that deleting a *single duplicate
    cluster* reverses the leaderboard. The mechanism is stated, not marvelled at: a cluster
    of size m contributes m times to a margin while carrying one item's information.
  * **Honest negatives** — the top-2 gap needs ~4500 examples to be detectable; a p-value
    within Monte Carlo error of alpha is flagged rather than believed; a bootstrap CI
    excluding zero while the permutation test declines to reject is reported as a
    disagreement between two nulls, not resolved by picking the friendlier one.
  * **A verdict that can say yes** — a synthetic corpus with a genuinely dominant system
    returns ``ranking_resolved = True``. A gate that only ever refuses would prove nothing.
  * **IO, determinism, and the CLI** — score-file parsing in both shapes, strict alignment,
    byte-identical artifacts, the committed demo fixture matching its generator, and the
    ``--fail-on-unstable-ranking`` gate.

Everything is CPU-only and offline.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from eval_dataset_forge.cli import build_parser, main
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup import DedupReport, find_duplicates
from eval_dataset_forge.schemas import BootstrapResult, DatasetExample
from eval_dataset_forge.statistics import (
    BLOCK,
    DEFAULT_ALPHA,
    EXAMPLE,
    MIN_EFFECTIVE_N,
    MIN_TOP1_STABILITY,
    StatsReport,
    Units,
    align,
    analyze_dataset,
    analyze_systems,
    bca_ci,
    bootstrap_ci,
    bootstrap_distribution,
    build_units,
    compare_systems,
    effective_n,
    format_report,
    holm,
    intraclass_correlation,
    kendall_tau_b,
    kish_design_effect,
    load_scores,
    measured_design_effect,
    min_examples_to_flip,
    min_units_to_flip,
    minimum_detectable_effect,
    monte_carlo_se,
    percentile_ci,
    rank_frequencies,
    required_n_for_delta,
    resample_indices,
    sign_flip_p,
    simulate_scores,
    to_table,
    unit_totals,
    write_report,
    write_scores,
)
from eval_dataset_forge.statistics.scores import parse_scores

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "datasets" / "fixtures"
QA = FIXTURES / "qa_toy" / "examples.jsonl"
QA_SCORES = FIXTURES / "qa_toy" / "system_scores.jsonl"

# The pooled corpus's block structure, as Milestone 4's dedup ladder finds it: 34 singletons,
# 8 pairs, 2 triples over 56 examples. Hard-coded in the coverage simulations so they test
# the statistics rather than re-deriving the dedup output.
POOLED_BLOCK_SIZES = [1] * 34 + [2] * 8 + [3] * 2


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pooled() -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    for path in sorted(FIXTURES.glob("*/examples.jsonl")):
        examples.extend(loader.load_examples(path))
    return examples


@pytest.fixture(scope="module")
def pooled_dedup(pooled: list[DatasetExample]) -> DedupReport:
    return find_duplicates(pooled)


@pytest.fixture(scope="module")
def pooled_units(pooled, pooled_dedup) -> tuple[Units, Units, tuple[str, ...]]:
    return build_units(pooled, pooled_dedup)


@pytest.fixture(scope="module")
def demo_scores(pooled_units) -> dict[str, list[float]]:
    _, blocks, _ = pooled_units
    return simulate_scores(blocks, seed=0)


@pytest.fixture(scope="module")
def demo_report(pooled, pooled_dedup, demo_scores) -> StatsReport:
    return analyze_systems(pooled, demo_scores, dedup=pooled_dedup, seed=0)


def _synthetic_blocks(sizes=POOLED_BLOCK_SIZES) -> Units:
    start = 0
    members = []
    for size in sizes:
        members.append(tuple(range(start, start + size)))
        start += size
    return Units(members=tuple(members), unit=BLOCK)


def _example(example_id: str, source_id: str = "src", text: str = "q") -> DatasetExample:
    return DatasetExample(
        example_id=example_id,
        source_id=source_id,
        task_family="qa",
        input=text,
        expected_output="a",
    )


# --------------------------------------------------------------------------- #
# Units and the ratio estimator
# --------------------------------------------------------------------------- #


def test_units_reject_an_empty_member_list():
    with pytest.raises(ValueError, match="at least one unit"):
        Units(members=())


def test_units_reject_an_unknown_resample_unit():
    with pytest.raises(ValueError, match="unknown resample unit"):
        Units(members=((0,),), unit="cluster")


def test_kish_mean_size_weights_larger_clusters_more():
    units = _synthetic_blocks()
    sizes = np.array(POOLED_BLOCK_SIZES, dtype=float)
    assert units.n_examples == 56
    assert units.n_units == 44
    # arithmetic mean 1.27, Kish mean 1.50: the triples carry 3 examples each.
    assert sizes.mean() == pytest.approx(56 / 44)
    assert units.mean_size_kish == pytest.approx(1.5)


def test_pooled_corpus_has_the_block_structure_the_coverage_tests_assume(pooled_units):
    _, blocks, _ = pooled_units
    assert blocks.size_histogram() == {1: 34, 2: 8, 3: 2}
    assert blocks.n_examples == 56
    assert blocks.mean_size_kish == pytest.approx(1.5)


def test_unit_totals_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="values for"):
        unit_totals([1.0, 0.0], Units.per_example(3))


def test_bootstrap_over_examples_is_the_ordinary_resampled_mean():
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    units = Units.per_example(5)
    draw = resample_indices(5, n_resamples=50, seed=3)
    reference = np.array([values[row].mean() for row in draw])
    assert bootstrap_distribution(values, units, draw=draw) == pytest.approx(reference)


def test_bootstrap_over_blocks_is_the_ratio_estimator():
    values = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    units = Units(members=((0, 1), (2,), (3, 4)), unit=BLOCK)
    draw = resample_indices(3, n_resamples=40, seed=5)
    reference = []
    for row in draw:
        drawn = [i for unit in row for i in units.members[unit]]
        reference.append(values[drawn].mean())
    assert bootstrap_distribution(values, units, draw=draw) == pytest.approx(np.array(reference))


def test_singleton_blocks_reproduce_the_iid_bootstrap_exactly():
    values = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    per_example = Units.per_example(6)
    as_blocks = Units(members=per_example.members, unit=BLOCK)
    draw = resample_indices(6, n_resamples=100, seed=1)
    assert bootstrap_distribution(values, per_example, draw=draw) == pytest.approx(
        bootstrap_distribution(values, as_blocks, draw=draw)
    )


def test_a_shared_draw_makes_the_comparison_paired():
    """mean(a*) - mean(b*) over one draw == the resampled mean of (a - b)."""
    rng = np.random.default_rng(0)
    a, b = (rng.random(20) < 0.7).astype(float), (rng.random(20) < 0.5).astype(float)
    units = Units.per_example(20)
    draw = resample_indices(20, n_resamples=200, seed=9)
    paired = bootstrap_distribution(a - b, units, draw=draw)
    separate = bootstrap_distribution(a, units, draw=draw) - bootstrap_distribution(
        b, units, draw=draw
    )
    assert paired == pytest.approx(separate)


def test_the_resampling_distribution_is_deterministic_given_the_seed():
    values = np.array([1.0, 0.0, 1.0, 1.0])
    units = Units.per_example(4)
    first = bootstrap_distribution(values, units, n_resamples=500, seed=7)
    second = bootstrap_distribution(values, units, n_resamples=500, seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, bootstrap_distribution(values, units, n_resamples=500, seed=8))


def test_the_point_estimate_does_not_depend_on_the_resampling_unit(demo_scores, pooled_units):
    per_example, blocks, _ = pooled_units
    for values in demo_scores.values():
        iid = bootstrap_ci(values, per_example, seed=0)
        clustered = bootstrap_ci(values, blocks, seed=0)
        assert iid.point_estimate == clustered.point_estimate


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #


def test_percentile_ci_is_the_pair_of_quantiles():
    distribution = np.linspace(0.0, 1.0, 1001)
    low, high = percentile_ci(distribution, 0.05)
    assert (low, high) == pytest.approx((0.025, 0.975), abs=1e-9)


def test_bootstrap_ci_records_the_unit_and_method_it_used():
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    result = bootstrap_ci(values, Units.per_example(6), system="sys", n_resamples=200, seed=0)
    assert isinstance(result, BootstrapResult)
    assert result.system == "sys"
    assert result.method == "percentile"
    assert result.resample_unit == EXAMPLE
    assert result.ci_low <= result.point_estimate <= result.ci_high


def test_bca_is_not_computable_when_every_example_is_correct():
    """z0 is infinite when no resample falls below the point estimate. Fall back, and say so."""
    values = np.ones(20)
    units = Units.per_example(20)
    distribution = bootstrap_distribution(values, units, n_resamples=200, seed=0)
    assert bca_ci(values, units, distribution) is None

    result = bootstrap_ci(values, units, method="bca", n_resamples=200, seed=0)
    assert result.method == "bca->percentile"
    assert (result.ci_low, result.ci_high) == (1.0, 1.0)


def test_bca_returns_none_when_a_single_unit_owns_the_corpus():
    values = np.array([1.0, 0.0, 1.0])
    units = Units(members=((0, 1, 2),), unit=BLOCK)
    distribution = bootstrap_distribution(values, units, n_resamples=50, seed=0)
    assert bca_ci(values, units, distribution) is None


def test_bca_shifts_the_interval_on_a_skewed_statistic():
    rng = np.random.default_rng(2)
    values = (rng.random(40) < 0.9).astype(float)
    units = Units.per_example(40)
    distribution = bootstrap_distribution(values, units, n_resamples=4000, seed=0)
    percentile = percentile_ci(distribution)
    corrected = bca_ci(values, units, distribution)
    assert corrected is not None
    assert corrected != percentile
    assert corrected[0] <= values.mean() <= corrected[1]


# --------------------------------------------------------------------------- #
# Coverage: the headline, with known ground truth
# --------------------------------------------------------------------------- #


def _clustered_coverage(p: float, trials: int, n_resamples: int, seed: int = 0) -> dict[str, float]:
    """Draw clustered corpora with a known true accuracy; count interval hits.

    ``icc = 1``: every member of a block shares the block's outcome, which is what a
    deterministic system does on duplicates. The estimand is ``p``, the accuracy of a fresh
    corpus drawn the same way.
    """
    blocks = _synthetic_blocks()
    per_example = Units.per_example(blocks.n_examples)
    sizes = np.array(POOLED_BLOCK_SIZES)
    rng = np.random.default_rng(seed)
    hits = {"iid": 0, "cluster": 0, "cluster_bca": 0}
    for _ in range(trials):
        outcomes = (rng.random(blocks.n_units) < p).astype(float)
        values = np.repeat(outcomes, sizes)
        independent = bootstrap_distribution(
            values, per_example, n_resamples=n_resamples, seed=int(rng.integers(1 << 30))
        )
        low, high = percentile_ci(independent)
        hits["iid"] += low <= p <= high

        clustered = bootstrap_distribution(
            values, blocks, n_resamples=n_resamples, seed=int(rng.integers(1 << 30))
        )
        low, high = percentile_ci(clustered)
        hits["cluster"] += low <= p <= high

        corrected = bca_ci(values, blocks, clustered) or (low, high)
        hits["cluster_bca"] += corrected[0] <= p <= corrected[1]
    return {name: count / trials for name, count in hits.items()}


def _iid_coverage(p: float, trials: int, n_resamples: int, seed: int = 0) -> dict[str, float]:
    units = Units.per_example(56)
    rng = np.random.default_rng(seed)
    hits = {"percentile": 0, "bca": 0, "bca_fallback": 0}
    for _ in range(trials):
        values = (rng.random(56) < p).astype(float)
        distribution = bootstrap_distribution(
            values, units, n_resamples=n_resamples, seed=int(rng.integers(1 << 30))
        )
        low, high = percentile_ci(distribution)
        hits["percentile"] += low <= p <= high
        corrected = bca_ci(values, units, distribution)
        if corrected is None:
            hits["bca_fallback"] += 1
            corrected = (low, high)
        hits["bca"] += corrected[0] <= p <= corrected[1]
    return {name: count / trials for name, count in hits.items()}


@pytest.fixture(scope="module")
def clustered_coverage_70() -> dict[str, float]:
    return _clustered_coverage(0.70, trials=600, n_resamples=399, seed=0)


def test_the_iid_bootstrap_undercovers_on_clustered_data(clustered_coverage_70):
    """THE headline. Nominal 95%; measured ~0.857 when examples inside a cluster are copies.

    600 trials, so the coverage estimate carries a standard error of ~0.014. The bound is set
    well clear of it: this is a 9-point miss, not noise.
    """
    assert clustered_coverage_70["iid"] < 0.90


def test_resampling_blocks_restores_most_of_the_lost_coverage(clustered_coverage_70):
    coverage = clustered_coverage_70
    assert coverage["cluster"] > coverage["iid"] + 0.04
    assert coverage["cluster"] > 0.90


def test_the_coverage_harness_reports_nominal_coverage_on_iid_data():
    """The positive control. Without it, "the bootstrap under-covers" could be a broken test."""
    coverage = _iid_coverage(0.50, trials=600, n_resamples=399, seed=1)
    assert coverage["percentile"] == pytest.approx(0.95, abs=0.03)


def test_bca_repairs_the_clustered_interval_at_an_extreme_accuracy():
    coverage = _clustered_coverage(0.90, trials=600, n_resamples=399, seed=2)
    assert coverage["cluster"] < 0.90
    assert coverage["cluster_bca"] > coverage["cluster"] + 0.03


def test_bca_breaks_the_iid_interval_when_the_accuracy_is_near_the_boundary():
    """The honest negative that keeps BCa off the default path.

    At p=0.95 on 56 examples the bootstrap distribution is near-degenerate: a few percent of
    trials cannot compute z0 at all, and the ones that can are corrected in the wrong
    direction.
    """
    coverage = _iid_coverage(0.95, trials=800, n_resamples=399, seed=3)
    assert coverage["percentile"] > 0.90
    assert coverage["bca"] < coverage["percentile"] - 0.05
    assert coverage["bca_fallback"] > 0.02


# --------------------------------------------------------------------------- #
# ICC, design effect, effective n
# --------------------------------------------------------------------------- #


def test_an_exact_duplicate_cluster_has_icc_one_for_any_deterministic_system():
    """Not measured on the fixtures — implied by the definition of a duplicate."""
    units = Units(members=((0, 1), (2, 3), (4,), (5,)), unit=BLOCK)
    values = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])  # copies score alike, by construction
    assert intraclass_correlation(values, units) == pytest.approx(1.0)


def test_icc_is_undefined_rather_than_zero_when_every_unit_is_a_singleton():
    values = np.array([1.0, 0.0, 1.0, 0.0])
    assert math.isnan(intraclass_correlation(values, Units.per_example(4)))


def test_icc_is_undefined_when_the_score_is_constant():
    units = Units(members=((0, 1), (2, 3)), unit=BLOCK)
    assert math.isnan(intraclass_correlation(np.ones(4), units))


def test_kish_predicts_a_design_effect_of_1_5_at_icc_one():
    assert kish_design_effect(_synthetic_blocks(), 1.0) == pytest.approx(1.5)
    assert kish_design_effect(_synthetic_blocks(), 0.0) == pytest.approx(1.0)


def test_the_measured_design_effect_exceeds_kish_and_both_exceed_one(demo_report):
    """Kish assumes equal cluster sizes; these are unequal. The report quotes both."""
    for system in demo_report.systems:
        assert system.icc == pytest.approx(1.0)
        assert system.design_effect_kish == pytest.approx(1.5)
        assert 1.3 < system.design_effect < 1.7
    assert demo_report.design_effect > 1.5


def test_the_design_effect_tracks_dependence_not_the_choice_of_unit(pooled_units):
    """Same blocks, no within-block correlation: the widening vanishes. Falsifies the artifact."""
    per_example, blocks, _ = pooled_units
    independent_scores = simulate_scores(blocks, icc=0.0, seed=0)
    for values in independent_scores.values():
        values = np.array(values)
        deff = measured_design_effect(
            bootstrap_distribution(values, blocks, n_resamples=4000, seed=3),
            bootstrap_distribution(values, per_example, n_resamples=4000, seed=3),
        )
        assert deff == pytest.approx(1.0, abs=0.25)


def test_effective_n_divides_the_row_count_by_the_design_effect():
    assert effective_n(56, 1.5) == pytest.approx(37.333, abs=1e-3)
    assert math.isnan(effective_n(56, float("nan")))
    assert math.isnan(effective_n(56, 0.0))


def test_the_pooled_corpus_is_worth_fewer_examples_than_it_has(demo_report):
    assert demo_report.n_examples == 56
    assert 30 < demo_report.effective_n < 40


# --------------------------------------------------------------------------- #
# The sign-flip test
# --------------------------------------------------------------------------- #


def test_sign_flip_returns_one_when_the_systems_agree_everywhere():
    units = Units.per_example(10)
    assert sign_flip_p(np.zeros(10), units, n_resamples=200, seed=0) == pytest.approx(1.0)


def test_sign_flip_is_never_zero():
    """Phipson-Smyth floor: a finite resampling cannot support p = 0."""
    units = Units.per_example(30)
    differences = np.ones(30)  # A wins every example: as extreme as the data can be
    p = sign_flip_p(differences, units, n_resamples=500, seed=0)
    assert p == pytest.approx(1.0 / 501.0)
    assert p > 0.0


def test_sign_flip_is_symmetric_in_the_two_systems():
    rng = np.random.default_rng(4)
    differences = rng.normal(size=25)
    units = Units.per_example(25)
    forward = sign_flip_p(differences, units, n_resamples=1000, seed=2)
    backward = sign_flip_p(-differences, units, n_resamples=1000, seed=2)
    assert forward == backward


def test_sign_flip_p_is_uniform_under_the_null():
    rng = np.random.default_rng(11)
    units = Units.per_example(40)
    values = [
        sign_flip_p(rng.normal(size=40), units, n_resamples=299, seed=int(rng.integers(1 << 30)))
        for _ in range(400)
    ]
    assert float(np.mean(values)) == pytest.approx(0.5, abs=0.06)


@pytest.fixture(scope="module")
def null_decision_rates() -> dict[str, float]:
    """Type-I error of three decision rules under a true null, 1500 trials at n=56."""
    units = Units.per_example(56)
    rng = np.random.default_rng(1)
    counts = {"win95": 0, "win975": 0, "sign_flip": 0}
    for _ in range(1500):
        a = (rng.random(56) < 0.7).astype(float)
        b = (rng.random(56) < 0.7).astype(float)
        differences = a - b
        distribution = bootstrap_distribution(
            differences, units, n_resamples=399, seed=int(rng.integers(1 << 30))
        )
        won = float(np.mean(distribution > 0.0))
        counts["win95"] += (won >= 0.95) or (won <= 0.05)
        counts["win975"] += (won >= 0.975) or (won <= 0.025)
        counts["sign_flip"] += (
            sign_flip_p(differences, units, n_resamples=399, seed=int(rng.integers(1 << 30)))
            <= 0.05
        )
    return {name: count / 1500 for name, count in counts.items()}


def test_the_bootstrap_win_fraction_is_not_a_p_value(null_decision_rates):
    """"A beats B if A wins 95% of resamples" rejects a true null twice as often as it claims.

    The rule is a *one-sided* 95% statement applied in whichever direction the data happened
    to favour, so its two-sided size is ~0.10. The threshold that corresponds to alpha=0.05
    is 0.975, and it measures right.
    """
    assert null_decision_rates["win95"] > 0.085
    assert null_decision_rates["win975"] == pytest.approx(0.05, abs=0.02)


def test_the_sign_flip_test_is_conservative_on_paired_binary_scores(null_decision_rates):
    """Measured 0.023 at nominal 0.05: the many tied examples make the null distribution coarse."""
    assert null_decision_rates["sign_flip"] < 0.05
    assert null_decision_rates["sign_flip"] < null_decision_rates["win975"]


def test_monte_carlo_se_of_a_p_value_shrinks_with_resamples():
    assert monte_carlo_se(0.05, 2000) == pytest.approx(math.sqrt(0.05 * 0.95 / 2000))
    assert monte_carlo_se(0.05, 20000) < monte_carlo_se(0.05, 2000)
    assert math.isnan(monte_carlo_se(float("nan"), 100))


# --------------------------------------------------------------------------- #
# Holm and the small-N arithmetic
# --------------------------------------------------------------------------- #


def test_holm_matches_a_hand_computed_step_down():
    # ascending p is .01 .03 .04; multipliers 3, 2, 1 give .03 .06 .04, and the running max
    # carries .06 forward onto the last one. Reported in the input order.
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_monotone_never_smaller_than_the_raw_p_and_capped_at_one():
    raw = [0.001, 0.02, 0.2, 0.9]
    adjusted = holm(raw)
    assert all(a >= r for a, r in zip(adjusted, raw, strict=True))
    assert all(a <= 1.0 for a in adjusted)
    assert adjusted == sorted(adjusted)
    assert holm([]) == []


def test_holm_is_never_weaker_than_bonferroni():
    raw = [0.004, 0.011, 0.03, 0.5]
    bonferroni = [min(1.0, len(raw) * p) for p in raw]
    assert all(h <= b for h, b in zip(holm(raw), bonferroni, strict=True))


def test_minimum_detectable_effect_matches_the_closed_form():
    # (1.959964 + 0.841621) * 0.5 * sqrt(1/100)
    assert minimum_detectable_effect(0.5, 100) == pytest.approx(0.1400792, abs=1e-6)
    assert math.isnan(minimum_detectable_effect(0.0, 100))


def test_the_design_effect_inflates_the_detectable_effect_by_its_square_root():
    plain = minimum_detectable_effect(0.5, 100, design_effect=1.0)
    inflated = minimum_detectable_effect(0.5, 100, design_effect=1.5)
    assert inflated == pytest.approx(plain * math.sqrt(1.5))


def test_required_n_inverts_the_detectable_effect():
    delta = minimum_detectable_effect(0.68, 56)
    assert required_n_for_delta(delta, 0.68) == pytest.approx(56, abs=1)
    assert required_n_for_delta(0.0, 0.5) is None
    assert required_n_for_delta(0.1, 0.0) is None


def test_required_n_scales_with_the_design_effect():
    plain = required_n_for_delta(0.05, 0.6, design_effect=1.0)
    clustered = required_n_for_delta(0.05, 0.6, design_effect=1.5)
    assert clustered == pytest.approx(plain * 1.5, rel=0.01)


# --------------------------------------------------------------------------- #
# Flip sets: exact, and brute-forced against the definition
# --------------------------------------------------------------------------- #


def _brute_force_min_units(differences: np.ndarray, units: Units) -> int | None:
    """The definition, enumerated. Only usable for tiny inputs; that is the point."""
    sums, _ = unit_totals(differences, units)
    total = float(sums.sum())
    if total <= 0.0:
        return 0
    for size in range(1, units.n_units):
        for subset in combinations(range(units.n_units), size):
            if total - float(sums[list(subset)].sum()) <= 1e-12:
                return size
    return None


@pytest.mark.parametrize("seed", range(12))
def test_greedy_flip_set_equals_brute_force(seed: int):
    rng = np.random.default_rng(seed)
    differences = rng.choice([-1.0, 0.0, 1.0], size=9)
    units = Units(members=((0, 1), (2,), (3, 4, 5), (6,), (7,), (8,)), unit=BLOCK)
    expected = _brute_force_min_units(differences, units)
    found = min_units_to_flip(differences, units)
    assert (found.n_removed if found else None) == expected


def test_flip_set_is_zero_when_the_lead_is_already_absent():
    flip = min_examples_to_flip(np.array([-1.0, 0.0, 1.0]))
    assert flip is not None and flip.n_removed == 0


def test_flip_set_is_none_when_every_example_favours_the_leader():
    """A real answer: no deletion of a proper subset can erase the lead."""
    assert min_examples_to_flip(np.array([1.0, 1.0, 1.0])) is None


def test_flip_set_never_deletes_the_whole_corpus():
    # Removing the two positives leaves an empty corpus, so there is no valid deletion.
    assert min_examples_to_flip(np.array([1.0, 1.0])) is None
    # With a third, zero-valued example there is one.
    flip = min_examples_to_flip(np.array([1.0, 1.0, 0.0]))
    assert flip is not None and flip.n_removed == 2


def test_flip_set_names_the_examples_it_removes():
    flip = min_examples_to_flip(np.array([0.0, 1.0, -0.5, 0.0]), ["a", "b", "c", "d"])
    assert flip is not None
    assert flip.removed_ids == ("b",)
    assert flip.n_examples_removed == 1
    assert flip.fraction == pytest.approx(0.25)


def test_deleting_one_duplicate_cluster_reverses_the_demo_leaderboard(demo_report, pooled_units):
    """The result worth reading twice — and its mechanism, which is not a coincidence.

    A cluster of size m contributes m times to a margin while carrying one item's worth of
    information, so duplicates are the highest-leverage items in any close comparison. The
    simulator makes that exact (icc=1); a deterministic system makes it exact on *exact*
    duplicates.
    """
    _, blocks, keys = pooled_units
    top = demo_report.top_comparison
    assert top is not None
    assert (top.system_a, top.system_b) == ("sys_a", "sys_b")

    assert top.example_flip is not None
    assert top.example_flip.n_removed == 2
    assert top.example_flip.removed_ids == ("agent-005", "agent-006")

    assert top.unit_flip is not None
    assert top.unit_flip.n_removed == 1
    assert top.unit_flip.removed_ids == ("agent-005",)
    assert top.unit_flip.n_examples_removed == 2

    # The two examples are one block, not two coincidentally adjacent items.
    owner = {key: members for key, members in zip(keys, blocks.members, strict=True)}
    assert len(owner["agent-005"]) == 2


# --------------------------------------------------------------------------- #
# Ranking stability
# --------------------------------------------------------------------------- #


def test_kendall_tau_b_on_hand_computable_vectors():
    assert kendall_tau_b([3.0, 2.0, 1.0], [3.0, 2.0, 1.0]) == pytest.approx(1.0)
    assert kendall_tau_b([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]) == pytest.approx(-1.0)
    # 3 pairs: 2 concordant, 0 discordant, 1 tied in y. tau_b = 2 / sqrt((3-0)*(3-1)).
    assert kendall_tau_b([1.0, 2.0, 3.0], [1.0, 3.0, 3.0]) == pytest.approx(2.0 / math.sqrt(6))
    assert math.isnan(kendall_tau_b([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="length mismatch"):
        kendall_tau_b([1.0], [1.0, 2.0])


def test_top1_frequency_sums_to_one_and_splits_ties():
    matrix = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])  # identical systems
    stability = rank_frequencies(["a", "b"], matrix, Units.per_example(4), n_resamples=200, seed=0)
    assert sum(stability.top1_frequency.values()) == pytest.approx(1.0)
    assert stability.top1_frequency["a"] == pytest.approx(0.5)
    assert stability.tie_fraction == pytest.approx(1.0)
    assert stability.exact_order_frequency == pytest.approx(0.0)


def test_a_dominant_system_holds_first_place_in_every_resample():
    matrix = np.array([np.ones(30), np.zeros(30)])
    stability = rank_frequencies(
        ["win", "lose"], matrix, Units.per_example(30), n_resamples=300, seed=0
    )
    assert stability.point_ranking == ("win", "lose")
    assert stability.top1_frequency["win"] == pytest.approx(1.0)
    assert stability.exact_order_frequency == pytest.approx(1.0)
    assert stability.mean_kendall_tau == pytest.approx(1.0)
    assert stability.stable


def test_the_demo_leader_is_not_stable(demo_report):
    stability = demo_report.blocked
    assert stability is not None
    assert stability.point_ranking == ("sys_a", "sys_b", "sys_c")
    assert stability.leader_stability < MIN_TOP1_STABILITY
    assert not stability.stable
    assert sum(stability.top1_frequency.values()) == pytest.approx(1.0)


def test_rank_frequencies_rejects_a_matrix_that_does_not_match_the_systems():
    with pytest.raises(ValueError, match="does not match"):
        rank_frequencies(["a", "b"], np.ones((3, 4)), Units.per_example(4))


# --------------------------------------------------------------------------- #
# Score IO
# --------------------------------------------------------------------------- #


def test_parse_scores_accepts_both_shapes_and_mixes_them():
    table = parse_scores(
        [
            '{"example_id": "e1", "system": "a", "score": 1}',
            "",
            "# a comment",
            '{"example_id": "e2", "scores": {"a": 0, "b": 0.5}}',
            '{"example_id": "e1", "system": "b", "score": 1.0}',
        ]
    )
    assert table == {"a": {"e1": 1.0, "e2": 0.0}, "b": {"e2": 0.5, "e1": 1.0}}


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (['{"system": "a", "score": 1}'], "needs an 'example_id'"),
        (['{"example_id": "e1"}'], "want either"),
        (['{"example_id": "e1", "scores": 3}'], "must be an object"),
        (["{not json}"], "invalid JSON"),
        ([], "no score records"),
        (['{"example_id": "e1", "system": "a", "score": true}'], "must be a number"),
        (['{"example_id": "e1", "system": "a", "score": "1"}'], "must be a number"),
        (
            [
                '{"example_id": "e1", "system": "a", "score": 1}',
                '{"example_id": "e1", "system": "a", "score": 0}',
            ],
            "scored more than once",
        ),
    ],
)
def test_parse_scores_rejects_malformed_records(lines, message):
    with pytest.raises(ValueError, match=message):
        parse_scores(lines)


def test_align_refuses_to_impute_a_missing_score():
    examples = [_example("e1"), _example("e2")]
    with pytest.raises(ValueError, match="has no score for 1 example"):
        align({"a": {"e1": 1.0}}, examples)


def test_align_refuses_scores_for_examples_the_dataset_does_not_contain():
    examples = [_example("e1")]
    with pytest.raises(ValueError, match="absent from the dataset"):
        align({"a": {"e1": 1.0, "ghost": 0.0}}, examples)


def test_align_returns_the_matrix_in_dataset_order():
    examples = [_example("e2"), _example("e1")]
    names, matrix = align({"a": {"e1": 0.0, "e2": 1.0}}, examples)
    assert names == ["a"]
    assert matrix.tolist() == [[1.0, 0.0]]


def test_to_table_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="has 2 scores for 3 examples"):
        to_table({"a": [1.0, 0.0]}, ["e1", "e2", "e3"])


def test_scores_round_trip_through_disk(tmp_path):
    table = {"a": {"e1": 1.0, "e2": 0.0}, "b": {"e1": 0.0, "e2": 1.0}}
    path = write_scores(table, tmp_path / "scores.jsonl", example_order=["e1", "e2"])
    assert load_scores(path) == table
    first = path.read_bytes()
    write_scores(table, path, example_order=["e1", "e2"])
    assert path.read_bytes() == first


def test_the_committed_demo_scores_match_their_generator():
    """The fixture is regenerable. A simulator change must be re-committed, not drift."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_demo_scores.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_committed_demo_scores_cover_every_qa_toy_example():
    examples = loader.load_examples(QA)
    names, matrix = align(load_scores(QA_SCORES), examples)
    assert names == ["sys_a", "sys_b", "sys_c"]
    assert matrix.shape == (3, len(examples))
    assert set(np.unique(matrix)) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #


def test_simulate_is_deterministic_given_the_seed(pooled_units):
    _, blocks, _ = pooled_units
    assert simulate_scores(blocks, seed=3) == simulate_scores(blocks, seed=3)
    assert simulate_scores(blocks, seed=3) != simulate_scores(blocks, seed=4)


def test_at_icc_one_every_member_of_a_cluster_scores_alike(pooled_units):
    _, blocks, _ = pooled_units
    scores = simulate_scores(blocks, icc=1.0, seed=0)
    for values in scores.values():
        for member in blocks.members:
            assert len({values[i] for i in member}) == 1


def test_at_icc_zero_cluster_members_come_apart(pooled_units):
    _, blocks, _ = pooled_units
    scores = simulate_scores(blocks, icc=0.0, seed=0)
    clusters = [member for member in blocks.members if len(member) > 1]
    disagreements = sum(
        1
        for values in scores.values()
        for member in clusters
        if len({values[i] for i in member}) > 1
    )
    assert disagreements > 0


def test_icc_changes_the_dependence_and_holds_the_marginal(pooled_units):
    """The property that makes an icc sweep a clean experiment on the design effect."""
    _, blocks, _ = pooled_units
    means = {
        icc: float(np.mean([np.mean(simulate_scores(blocks, icc=icc, seed=s)["sys_a"])
                            for s in range(80)]))
        for icc in (0.0, 1.0)
    }
    assert means[0.0] == pytest.approx(means[1.0], abs=0.05)


def test_simulate_rejects_an_out_of_range_icc(pooled_units):
    _, blocks, _ = pooled_units
    with pytest.raises(ValueError, match=r"icc must lie in \[0, 1\]"):
        simulate_scores(blocks, icc=1.5)


def test_simulate_rejects_mismatched_names(pooled_units):
    _, blocks, _ = pooled_units
    with pytest.raises(ValueError, match="names for"):
        simulate_scores(blocks, abilities=(1.0, 0.0), names=("only_one",))


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_the_demo_leaderboard_is_not_resolved(demo_report):
    assert not demo_report.ranking_resolved
    assert demo_report.underpowered is (demo_report.effective_n < MIN_EFFECTIVE_N)
    top = demo_report.top_comparison
    assert top is not None
    assert top.delta == pytest.approx(2 / 56)
    assert top.ci_contains_zero
    assert not top.significant
    assert top.underpowered
    assert top.required_n is not None and top.required_n > 4000


def test_the_verdict_can_say_yes():
    """A gate that only ever refuses proves nothing. A dominant system on ample data passes."""
    examples = [
        _example(f"e{i:03d}", source_id=f"src-{i:03d}", text=f"question {i}") for i in range(200)
    ]
    rng = np.random.default_rng(0)
    strong = (rng.random(200) < 0.9).astype(float).tolist()
    weak = (rng.random(200) < 0.4).astype(float).tolist()
    report = analyze_systems(examples, {"strong": strong, "weak": weak}, seed=0)

    assert report.ranking_resolved
    assert report.point_ranking == ("strong", "weak")
    assert report.blocked is not None and report.blocked.stable
    top = report.top_comparison
    assert top is not None and top.significant and not top.ci_contains_zero


def test_without_a_dedup_report_the_design_effect_is_exactly_one():
    """The correct answer for a corpus whose dependence has not been measured."""
    examples = [_example(f"e{i}", source_id=f"src-{i}", text=f"q{i}") for i in range(40)]
    rng = np.random.default_rng(0)
    scores = {"a": (rng.random(40) < 0.7).astype(float).tolist()}
    report = analyze_systems(examples, scores, dedup=None, seed=0)
    assert report.n_units == report.n_examples
    assert report.design_effect == pytest.approx(1.0)
    assert report.systems[0].independent.ci_low == report.systems[0].clustered.ci_low


def test_the_report_warns_about_every_defect_it_found(demo_report):
    joined = "\n".join(demo_report.warnings)
    assert "design effect" in joined
    assert "holds first place in only" in joined
    assert "contains zero" in joined
    assert "reverses" in joined
    assert "lower bound" in joined  # the dedup recall bound the cluster bootstrap inherits


def test_a_borderline_p_value_is_flagged_rather_than_believed(demo_report):
    """sys_a vs sys_c: p = 0.051 at B=2000, whose Monte Carlo SE is 0.005."""
    pair = next(
        c for c in demo_report.comparisons if {c.system_a, c.system_b} == {"sys_a", "sys_c"}
    )
    assert pair.p_value == pytest.approx(0.05, abs=0.01)
    assert pair.p_is_borderline
    assert pair.p_mc_se == pytest.approx(monte_carlo_se(pair.p_value, 2000))


def test_a_ci_excluding_zero_can_coexist_with_a_test_that_does_not_reject(demo_report):
    """Two different nulls, not a bug. The permutation null carries the effect in its spread."""
    pair = next(
        c for c in demo_report.comparisons if {c.system_a, c.system_b} == {"sys_a", "sys_c"}
    )
    assert not pair.ci_contains_zero
    assert not pair.significant
    assert pair.ci_disagrees_with_test
    assert any("different nulls" in message for message in demo_report.warnings)


def test_holm_correction_is_applied_across_every_pair(demo_report):
    assert len(demo_report.comparisons) == 3
    for comparison in demo_report.comparisons:
        assert comparison.p_holm >= comparison.p_value
        assert comparison.delta >= 0.0  # pairs are oriented by the observed winner


def test_analyze_systems_is_deterministic(pooled, pooled_dedup, demo_scores):
    first = analyze_systems(pooled, demo_scores, dedup=pooled_dedup, seed=0).to_dict()
    second = analyze_systems(pooled, demo_scores, dedup=pooled_dedup, seed=0).to_dict()
    assert first == second


def test_the_report_serializes_to_json(demo_report):
    payload = json.dumps(demo_report.to_dict())
    restored = json.loads(payload)
    assert restored["counts"]["examples"] == 56
    assert restored["ranking_resolved"] is False
    assert restored["stability"][BLOCK]["point_ranking"] == ["sys_a", "sys_b", "sys_c"]


def test_format_report_names_the_verdict_and_the_widening(demo_report):
    text = format_report(demo_report)
    assert "ranking verdict: NOT RESOLVED" in text
    assert "win fraction is NOT a p-value" in text
    assert "design effect" in text
    assert "x wider" in text


def test_write_report_is_byte_identical_across_runs(demo_report, tmp_path):
    first = write_report(demo_report, tmp_path / "a")["report"].read_bytes()
    second = write_report(demo_report, tmp_path / "b")["report"].read_bytes()
    assert first == second


def test_analyze_systems_rejects_an_unknown_ci_method(pooled, demo_scores):
    with pytest.raises(ValueError, match="unknown ci method"):
        analyze_systems(pooled, demo_scores, ci_method="student")


def test_analyze_systems_rejects_an_unknown_resample_unit(pooled, demo_scores):
    with pytest.raises(ValueError, match="unknown resample unit"):
        analyze_systems(pooled, demo_scores, resample_unit="source")


def test_analyze_systems_rejects_a_score_vector_of_the_wrong_length(pooled):
    with pytest.raises(ValueError, match="scores for"):
        analyze_systems(pooled, {"a": [1.0, 0.0]})


def test_analyze_dataset_wants_exactly_one_score_source(pooled):
    with pytest.raises(ValueError, match="exactly one"):
        analyze_dataset(pooled)
    with pytest.raises(ValueError, match="exactly one"):
        analyze_dataset(pooled, score_table={"a": {}}, simulate_seed=0)


def test_compare_systems_orients_each_pair_by_the_observed_winner():
    units = Units.per_example(10)
    scores = {"loser": [0.0] * 10, "winner": [1.0] * 6 + [0.0] * 4}
    comparisons = compare_systems(scores, units, n_resamples=200, seed=0)
    assert len(comparisons) == 1
    assert (comparisons[0].system_a, comparisons[0].system_b) == ("winner", "loser")
    assert comparisons[0].delta > 0


def test_a_single_system_yields_an_interval_and_no_ranking_claim():
    examples = [_example(f"e{i}", source_id=f"src-{i}", text=f"q{i}") for i in range(20)]
    report = analyze_systems(examples, {"solo": [1.0] * 14 + [0.0] * 6}, seed=0)
    assert report.comparisons == []
    assert not report.ranking_resolved
    assert any("fewer than two systems" in message for message in report.warnings)


def test_a_perfect_score_is_reported_as_a_degenerate_interval():
    examples = [_example(f"e{i}", source_id=f"src-{i}", text=f"q{i}") for i in range(20)]
    report = analyze_systems(examples, {"oracle": [1.0] * 20}, seed=0)
    assert report.systems[0].clustered.ci_low == report.systems[0].clustered.ci_high == 1.0
    assert any("degenerate" in message for message in report.warnings)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_bootstrap_result_defaults_keep_old_records_loadable():
    result = BootstrapResult(
        metric="accuracy", point_estimate=0.8, ci_low=0.7, ci_high=0.9, n_resamples=1000
    )
    assert (result.system, result.method, result.resample_unit) == (None, "percentile", "example")
    assert BootstrapResult.model_validate(result.model_dump(mode="json")) == result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_stats_runs_the_simulated_demo(capsys):
    assert main(["stats", "--dataset", str(QA), "--simulate", "--resamples", "300"]) == 0
    out = capsys.readouterr().out
    assert "ranking verdict" in out
    assert "simulated(seed=0)" in out


def test_cli_stats_reads_a_committed_score_file(capsys):
    assert (
        main(["stats", "--dataset", str(QA), "--scores", str(QA_SCORES), "--resamples", "300"])
        == 0
    )
    out = capsys.readouterr().out
    assert str(QA_SCORES) in out


def test_cli_stats_pools_several_datasets(capsys):
    argv = ["stats"]
    for path in sorted(FIXTURES.glob("*/examples.jsonl")):
        argv += ["--dataset", str(path)]
    assert main([*argv, "--simulate", "--resamples", "300"]) == 0
    assert "over 56 examples in 44 blocks" in capsys.readouterr().out


def test_cli_stats_gate_fails_on_an_unstable_ranking(capsys):
    argv = ["stats", "--dataset", str(QA), "--simulate", "--resamples", "300"]
    assert main([*argv, "--fail-on-unstable-ranking"]) == 1


def test_cli_stats_emits_json(capsys):
    assert main(["stats", "--dataset", str(QA), "--simulate", "--resamples", "300", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["systems"] == 3
    assert payload["ranking_resolved"] is False


def test_cli_stats_writes_its_artifact(tmp_path, capsys):
    out = tmp_path / "stats"
    assert (
        main(["stats", "--dataset", str(QA), "--simulate", "--resamples", "300", "--out", str(out)])
        == 0
    )
    assert (out / "stats_report.json").exists()
    assert json.loads((out / "stats_report.json").read_text())["counts"]["examples"] == 14


def test_cli_stats_requires_a_score_source():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["stats", "--dataset", str(QA)])


def test_cli_stats_rejects_two_score_sources():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["stats", "--dataset", str(QA), "--simulate", "--scores", str(QA_SCORES)])


def test_cli_stats_defaults_to_the_block_resample_unit():
    args = build_parser().parse_args(["stats", "--dataset", str(QA), "--simulate"])
    assert args.resample_unit == BLOCK
    assert args.ci_method == "percentile"
    assert args.resamples == 2_000
    assert args.alpha == DEFAULT_ALPHA
