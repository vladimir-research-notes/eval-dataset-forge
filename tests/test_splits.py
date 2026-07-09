"""Tests for the split builder (Milestone 7).

Covered here:

  * **Apportionment** — the largest-remainder rule, its exact targets at the sizes the
    fixtures actually have, and the reason it runs *before* the greedy rather than
    after (a stratum of three cannot show ``dev`` a closeable deficit).
  * **Blocks** — the four constructors, union-find transitivity, and the property that
    every strategy's blocks partition the corpus. Pins that ``qa-003``/``qa-004``/
    ``qa-012`` forms one three-member block, which is the question Milestone 4's review
    note left open for this milestone.
  * **The assignment core** — singleton assignment reproduces the apportioned targets
    exactly; no block ever straddles a split; eligibility is honoured; a block with no
    eligible split is placed with a warning rather than dropped.
  * **Strategy equivalences** — ``stratified`` is *bit-for-bit* ``random`` on every
    committed fixture (each holds one task family), and ``source-grouped`` is
    bit-for-bit ``random`` on the three fixtures with no shared ``source_id``. Both are
    asserted so neither strategy is ever credited with work it did not do.
  * **The headline result** — every planted duplicate carries a fresh ``source_id``, so
    ``source-grouped`` leaves the planted leakage in place and measures no better than
    ``random``. Only content-derived blocks close the channel: ``adversarial`` reaches
    zero residual cross-split overlap on every fixture at every seed.
  * **The stratification sweep** — joint ``task_family,difficulty`` stratification is
    worse on *both* facets than either facet alone, and provably puts no ``hard``
    example in the test split. Both the sweep and the mechanism are pinned, and so are
    the two warnings that surface it.
  * **The label-audit hand-off** — five examples whose gold answer needs adjudication sit
    in ``dev``/``test`` without ``quarantine_ids``, and none do with it.
  * **Honest negatives** — the adversarial guarantee is inherited from a heuristic
    (``ret-001``/``ret-002`` straddles ``dev``/``test`` while the verifier reports zero);
    ``retrieval_toy``'s clean row is the answer gate, not an absence of leakage; and
    ``eval_splits=("test",)`` changes nothing on this corpus, exercised by a synthetic
    one where it does.
  * **Determinism and IO** — seeded reproducibility, input immutability, byte-identical
    split files, and a JSONL round-trip through the loader.

Everything is CPU-only and offline.
"""

from __future__ import annotations

import json

import pytest

from eval_dataset_forge.cli import _PENDING, _parse_ratios, build_parser, main
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup import find_duplicates
from eval_dataset_forge.labels import audit_labels
from eval_dataset_forge.schemas import DatasetExample, Split, SplitSpec, SplitStrategy
from eval_dataset_forge.splits import (
    DEFAULT_RATIOS,
    SPLIT_ORDER,
    Block,
    SplitReport,
    adversarial_splits,
    apply_splits,
    apportion,
    assign_blocks,
    assign_in_order,
    block_of_example,
    block_stratum,
    build_splits,
    declared_bucket,
    duplicate_blocks,
    facet_value,
    format_report,
    normalize_ratios,
    random_splits,
    singleton_blocks,
    source_blocks,
    source_grouped_splits,
    stratified_splits,
    stratum_of,
    temporal_splits,
    verify_splits,
    write_splits,
)
from eval_dataset_forge.splits.assignment import DEFAULT_SEED
from eval_dataset_forge.splits.blocks import (
    DUPLICATE,
    DUPLICATE_AND_SOURCE,
    SINGLETON,
    SOURCE,
    _components,
)
from eval_dataset_forge.splits.temporal import parse_timestamp, temporal_order

STRATEGIES = [s.value for s in SplitStrategy]
SEEDS = list(range(20))

FIXTURES = ("qa_toy", "agent_tasks_toy", "bio_toy", "retrieval_toy", "safety_toy")

# Fixtures carrying at least one planted duplicate the dedup ladder can see. Every
# fixture but retrieval_toy, whose paraphrases the near stage's answer gate hides.
WITH_DUPLICATES = ("qa_toy", "agent_tasks_toy", "bio_toy", "safety_toy", "pooled")

# Fixtures whose examples all carry a distinct source_id, so a source group is a
# singleton and source-grouped assignment is indistinguishable from random.
NO_SHARED_SOURCE = ("agent_tasks_toy", "bio_toy", "safety_toy")

# Fixtures with at least one source_id used by more than one example.
SHARED_SOURCE = ("qa_toy", "retrieval_toy")


# --------------------------------------------------------------------------- #
# Corpus fixtures (module-scoped: the dedup ladder is the expensive part)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def corpora() -> dict[str, list[DatasetExample]]:
    named = {
        path.name: loader.load_examples(path / "examples.jsonl")
        for path in loader.discover_fixtures()
    }
    named["pooled"] = [example for examples in named.values() for example in examples]
    return named


@pytest.fixture(scope="module")
def reports(corpora):
    return {name: find_duplicates(examples) for name, examples in corpora.items()}


@pytest.fixture(scope="module")
def qa(corpora):
    return corpora["qa_toy"]


@pytest.fixture(scope="module")
def pooled(corpora):
    return corpora["pooled"]


def _example(index: int, text: str, answer: object, **kwargs) -> DatasetExample:
    return DatasetExample(
        example_id=f"e{index:02d}",
        source_id=kwargs.pop("source_id", f"s{index:02d}"),
        task_family=kwargs.pop("task_family", "qa"),
        input=text,
        expected_output=answer,
        **kwargs,
    )


def _sizes(assignment: dict[str, str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for split in assignment.values():
        sizes[split] = sizes.get(split, 0) + 1
    return sizes


def _built(corpora, reports, name: str, strategy: str, **kwargs) -> SplitReport:
    """Build splits over a named corpus, reusing its cached dedup report."""
    return build_splits(
        corpora[name], strategy=strategy, dedup_report=reports[name], **kwargs
    )


# --------------------------------------------------------------------------- #
# Apportionment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, {"train": 0, "dev": 0, "test": 0}),
        (1, {"train": 1, "dev": 0, "test": 0}),
        (2, {"train": 1, "dev": 0, "test": 1}),
        (3, {"train": 2, "dev": 0, "test": 1}),
        (10, {"train": 7, "dev": 1, "test": 2}),
        (12, {"train": 8, "dev": 1, "test": 3}),
        (14, {"train": 10, "dev": 1, "test": 3}),
        (56, {"train": 39, "dev": 6, "test": 11}),
    ],
)
def test_apportion_matches_the_largest_remainder_rule(size, expected):
    assert apportion(size, DEFAULT_RATIOS) == expected


@pytest.mark.parametrize("size", range(0, 60))
def test_apportion_always_distributes_every_example(size):
    assert sum(apportion(size, DEFAULT_RATIOS).values()) == size


@pytest.mark.parametrize("size", [1, 3, 7, 10, 99, 1_000_001])
def test_apportion_survives_ratios_that_do_not_sum_exactly_in_binary(size):
    """At extreme sizes the floors can sum past ``size``; the targets must still sum to it."""
    thirds = {"train": 1 / 3, "dev": 1 / 3, "test": 1 - 2 / 3}
    targets = apportion(size, normalize_ratios(thirds))
    assert sum(targets.values()) == size
    assert all(count >= 0 for count in targets.values())


def test_apportion_gives_a_small_stratum_no_dev_rather_than_stealing_from_train():
    """The reason apportionment runs before the greedy, not after.

    A stratum of three shows ``dev`` a deficit of 0.3 that no assignment can close. A
    greedy chasing that float deficit lands on 1/1/1; apportioning first gives 2/0/1
    and turns the unfillable quota into a warning.
    """
    assert apportion(3, DEFAULT_RATIOS) == {"train": 2, "dev": 0, "test": 1}


def test_apportion_leftovers_break_ties_in_split_order():
    # n=2 at 0.5/0.5: floors are 1,1 -> no leftover. n=1: both frac 0.5, train wins.
    assert apportion(1, {"train": 0.5, "test": 0.5}) == {"train": 1, "test": 0}


# --------------------------------------------------------------------------- #
# Ratio validation
# --------------------------------------------------------------------------- #


def test_normalize_ratios_defaults_and_orders_by_split_order():
    assert list(normalize_ratios(None)) == list(SPLIT_ORDER)


def test_normalize_ratios_drops_zero_weight_splits():
    assert normalize_ratios({"train": 0.8, "dev": 0.0, "test": 0.2}) == {
        "train": 0.8,
        "test": 0.2,
    }


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.7, "eval": 0.3},
        {"train": 1.2, "test": -0.2},
        {"train": 0.5, "test": 0.2},
        {"train": 0.9, "test": 0.2},
    ],
)
def test_normalize_ratios_rejects_a_partial_or_impossible_partition(ratios):
    with pytest.raises(ValueError):
        normalize_ratios(ratios)


def test_split_spec_and_normalize_ratios_agree_on_the_contract(pooled, reports):
    report = build_splits(pooled, strategy="random", dedup_report=reports["pooled"])
    assert isinstance(report.spec, SplitSpec)
    assert abs(sum(report.spec.ratios.values()) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


def test_a_block_must_hold_at_least_one_example():
    with pytest.raises(ValueError, match="at least one example"):
        Block(example_ids=())


def test_block_key_and_size():
    block = Block(example_ids=("b", "a", "c"))
    assert block.key == "b"  # first in input order, not sorted
    assert block.size == 3


def test_singleton_blocks_preserve_input_order(qa):
    blocks = singleton_blocks(qa)
    assert [block.key for block in blocks] == [example.example_id for example in qa]
    assert all(block.size == 1 and not block.duplicate for block in blocks)


def test_source_blocks_group_by_source_id(qa):
    blocks = source_blocks(qa)
    shakespeare = next(block for block in blocks if block.key == "qa-003")
    assert shakespeare.example_ids == ("qa-003", "qa-012")
    assert shakespeare.reason == SOURCE
    assert not shakespeare.duplicate


def test_source_blocks_are_singletons_when_no_source_is_shared(corpora):
    for name in NO_SHARED_SOURCE:
        blocks = source_blocks(corpora[name])
        assert all(block.size == 1 for block in blocks)
        assert all(block.reason == SINGLETON for block in blocks)


def test_duplicate_blocks_merge_the_three_member_shakespeare_cluster(qa, reports):
    """Answers the open question left by Milestone 4's review note.

    Single-linkage joins ``qa-012`` to the ``qa-003``/``qa-004`` pair through a chain.
    The adversarial split wants that as *one* block, not two overlapping pairs: a pair
    decomposition would let ``qa-004`` and ``qa-012`` land in different splits.
    """
    blocks = duplicate_blocks(qa, reports["qa_toy"])
    cluster = next(block for block in blocks if "qa-003" in block.example_ids)
    assert cluster.example_ids == ("qa-003", "qa-004", "qa-012")
    assert cluster.duplicate
    # qa-003 and qa-012 also share src-shakespeare, so both edge kinds touch this block.
    assert cluster.reason == DUPLICATE_AND_SOURCE


def test_duplicate_blocks_mark_a_content_cluster_without_a_shared_source(qa, reports):
    blocks = duplicate_blocks(qa, reports["qa_toy"])
    pair = next(block for block in blocks if "qa-001" in block.example_ids)
    assert pair.example_ids == ("qa-001", "qa-002")
    assert pair.reason == DUPLICATE
    assert pair.duplicate


def test_a_pure_source_group_is_dependent_but_not_duplicated(corpora, reports):
    """``ret-006``/``ret-009`` share ``src-boiling`` and are not a dedup cluster.

    The distinction is load-bearing: ``duplicate=False`` keeps the block eligible for an
    evaluation split, because two questions written from one source are not two copies
    of one question.
    """
    examples, report = corpora["retrieval_toy"], reports["retrieval_toy"]
    assert report.findings == []  # the answer gate hides every retrieval paraphrase
    blocks = duplicate_blocks(examples, report)
    boiling = next(block for block in blocks if block.size > 1)
    assert boiling.example_ids == ("ret-006", "ret-009")
    assert boiling.reason == SOURCE
    assert not boiling.duplicate


def test_duplicate_blocks_can_ignore_source_edges(qa, reports):
    blocks = duplicate_blocks(qa, reports["qa_toy"], include_sources=False)
    cluster = next(block for block in blocks if "qa-003" in block.example_ids)
    assert cluster.reason == DUPLICATE  # the source edge no longer contributes


@pytest.mark.parametrize("name", ["qa_toy", "agent_tasks_toy", "retrieval_toy", "pooled"])
@pytest.mark.parametrize("constructor", ["singleton", "source", "duplicate"])
def test_blocks_partition_the_corpus(corpora, reports, name, constructor):
    examples = corpora[name]
    if constructor == "singleton":
        blocks = singleton_blocks(examples)
    elif constructor == "source":
        blocks = source_blocks(examples)
    else:
        blocks = duplicate_blocks(examples, reports[name])
    members = [eid for block in blocks for eid in block.example_ids]
    assert sorted(members) == sorted(example.example_id for example in examples)
    assert len(members) == len(set(members))


def test_components_are_transitive_and_ordered_by_first_appearance():
    order = ["c", "a", "b", "d"]
    assert _components(order, [("c", "a"), ("a", "b")]) == [["c", "a", "b"], ["d"]]


def test_components_ignore_edges_naming_unknown_ids():
    assert _components(["a", "b"], [("a", "zzz")]) == [["a"], ["b"]]


def test_block_of_example_is_a_reverse_index(qa, reports):
    blocks = duplicate_blocks(qa, reports["qa_toy"])
    index = block_of_example(blocks)
    assert index["qa-012"].key == "qa-003"
    assert len(index) == len(qa)


# --------------------------------------------------------------------------- #
# The assignment core
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["qa_toy", "bio_toy", "pooled"])
@pytest.mark.parametrize("seed", SEEDS)
def test_singleton_assignment_reproduces_the_apportioned_targets_exactly(corpora, name, seed):
    examples = corpora[name]
    assignment = assign_blocks(singleton_blocks(examples), seed=seed)
    assert _sizes(assignment.of_example) == apportion(len(examples), DEFAULT_RATIOS)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("name", ["qa_toy", "retrieval_toy", "pooled"])
def test_no_block_ever_straddles_a_split(corpora, reports, strategy, name):
    report = build_splits(
        corpora[name], strategy=strategy, dedup_report=reports[name], seed=7
    )
    for block in report.blocks:
        landed = {report.assignment[eid] for eid in block.example_ids}
        assert len(landed) == 1, f"{block.example_ids} straddles {landed}"


def test_eligibility_restricts_a_block_to_the_splits_it_allows():
    blocks = [Block(example_ids=(f"e{i}",)) for i in range(10)]
    pinned = {"e0", "e1"}

    def eligible_for(block):
        return ("train",) if block.key in pinned else SPLIT_ORDER

    assignment = assign_blocks(blocks, eligible_for=eligible_for, seed=1)
    assert all(assignment.of_example[eid] == "train" for eid in pinned)


def test_a_block_eligible_for_nothing_is_placed_in_train_and_warned_about():
    """Never dropped. Removing an example is the release builder's decision, not ours."""
    blocks = [Block(example_ids=("e0",)), Block(example_ids=("e1",))]
    assignment = assign_blocks(blocks, eligible_for=lambda _block: (), seed=0)
    assert set(assignment.of_example.values()) == {"train"}
    assert len(assignment.of_example) == 2
    assert any("eligible for no split" in warning for warning in assignment.warnings)


def test_assign_in_order_respects_the_ordering():
    blocks = [Block(example_ids=(f"e{i}",)) for i in range(10)]
    assignment = assign_in_order(blocks, order_key=lambda block: block.key)
    rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    positions = [rank[assignment.of_example[f"e{i}"]] for i in range(10)]
    assert positions == sorted(positions)


def test_assign_in_order_keeps_a_block_straddling_a_cut_point_intact():
    # 10 examples, cut at 7: a 4-member block spanning positions 6..9 must not be split.
    blocks = [Block(example_ids=tuple(f"e{i}" for i in range(6)))]
    blocks += [Block(example_ids=tuple(f"e{i}" for i in range(6, 10)))]
    assignment = assign_in_order(blocks, order_key=lambda block: block.key)
    assert len({assignment.of_example[f"e{i}"] for i in range(6, 10)}) == 1


def test_the_strata_warning_fires_when_the_average_stratum_cannot_fill_the_smallest_split(
    pooled,
):
    joint = build_splits(pooled, strategy="stratified", stratify_by=("task_family", "difficulty"))
    single = build_splits(pooled, strategy="stratified", stratify_by=("task_family",))
    assert any("strata for 56 examples" in warning for warning in joint.warnings)
    assert not any("strata for" in warning for warning in single.warnings)


# --------------------------------------------------------------------------- #
# Strategy contracts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("name", ["qa_toy", "agent_tasks_toy", "pooled"])
def test_every_strategy_partitions_the_corpus(corpora, reports, strategy, name):
    examples = corpora[name]
    report = build_splits(examples, strategy=strategy, dedup_report=reports[name])
    assert set(report.assignment) == {example.example_id for example in examples}
    assert sum(report.split_sizes.values()) == len(examples)
    assert set(report.assignment.values()) <= set(SPLIT_ORDER)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_is_deterministic_at_a_fixed_seed(qa, reports, strategy):
    first = build_splits(qa, strategy=strategy, seed=5, dedup_report=reports["qa_toy"])
    second = build_splits(qa, strategy=strategy, seed=5, dedup_report=reports["qa_toy"])
    assert first.assignment == second.assignment


@pytest.mark.parametrize("strategy", ["random", "stratified", "source-grouped", "adversarial"])
def test_the_seeded_strategies_actually_depend_on_the_seed(corpora, reports, strategy):
    assignments = {
        json.dumps(
            _built(corpora, reports, "pooled", strategy, seed=seed).assignment, sort_keys=True
        )
        for seed in range(5)
    }
    assert len(assignments) > 1


def test_temporal_ignores_the_seed_because_its_order_is_the_data(qa):
    assert temporal_splits(qa).of_example == temporal_splits(qa).of_example


def test_the_strategy_functions_are_exactly_what_the_facade_calls(corpora, reports):
    qa, report, seed = corpora["qa_toy"], reports["qa_toy"], 4
    facade = {
        strategy: _built(corpora, reports, "qa_toy", strategy, seed=seed).assignment
        for strategy in STRATEGIES
    }
    assert random_splits(qa, seed=seed).of_example == facade["random"]
    assert stratified_splits(qa, seed=seed).of_example == facade["stratified"]
    assert source_grouped_splits(qa, seed=seed).of_example == facade["source-grouped"]
    assert temporal_splits(qa).of_example == facade["temporal"]
    assert (
        adversarial_splits(qa, seed=seed, dedup_report=report).of_example
        == facade["adversarial"]
    )


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("seed", SEEDS)
def test_stratified_is_bit_for_bit_random_on_a_single_family_fixture(corpora, reports, name, seed):
    """Each committed fixture holds exactly one ``task_family``, so the default
    stratification has exactly one stratum and cannot constrain anything. Asserted so
    the strategy is never credited with work it did not do."""
    assert len({example.task_family for example in corpora[name]}) == 1
    assert (
        _built(corpora, reports, name, "stratified", seed=seed).assignment
        == _built(corpora, reports, name, "random", seed=seed).assignment
    )


def test_stratified_does_constrain_the_pooled_multi_family_corpus(corpora, reports):
    stratified = _built(corpora, reports, "pooled", "stratified", seed=0)
    plain = _built(corpora, reports, "pooled", "random", seed=0)
    assert stratified.assignment != plain.assignment
    assert set(stratified.distributions["task_family"]["test"]) == {
        "qa", "agent_task", "retrieval", "bio", "safety"
    }


@pytest.mark.parametrize("name", NO_SHARED_SOURCE)
@pytest.mark.parametrize("seed", SEEDS)
def test_source_grouped_is_bit_for_bit_random_without_a_shared_source(corpora, reports, name, seed):
    assert reports[name].duplicate_source_ids == {}
    assert (
        _built(corpora, reports, name, "source-grouped", seed=seed).assignment
        == _built(corpora, reports, name, "random", seed=seed).assignment
    )


@pytest.mark.parametrize("name", SHARED_SOURCE)
def test_source_grouped_diverges_from_random_where_a_source_is_shared(corpora, reports, name):
    grouped = [_built(corpora, reports, name, "source-grouped", seed=s) for s in SEEDS]
    plain = [_built(corpora, reports, name, "random", seed=s) for s in SEEDS]
    assert any(g.assignment != p.assignment for g, p in zip(grouped, plain, strict=True))
    # And it delivers on its own promise: no source record spans two splits.
    assert all(built.residual_sources == {} for built in grouped)


@pytest.mark.parametrize("name", SHARED_SOURCE)
def test_random_does_scatter_source_records_across_splits(corpora, reports, name):
    assert any(
        _built(corpora, reports, name, "random", seed=seed).residual_sources for seed in SEEDS
    )


# --------------------------------------------------------------------------- #
# The headline result: provenance does not know what an example says
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("survivor", "paraphrase"),
    [
        ("qa-001", "qa-002"),
        ("agent-001", "agent-002"),
        ("bio-001", "bio-002"),
        ("saf-004", "saf-005"),
    ],
)
def test_every_planted_duplicate_carries_a_fresh_source_id(corpora, survivor, paraphrase):
    """The mechanism behind the headline result, read straight off the fixtures.

    A paraphrase written as a new record gets a new provenance id. Grouping on
    ``source_id`` therefore cannot hold the pair together, no matter the seed.
    """
    by_id = {example.example_id: example for example in corpora["pooled"]}
    assert by_id[survivor].source_id != by_id[paraphrase].source_id


@pytest.mark.parametrize("strategy", ["source-grouped", "random"])
@pytest.mark.parametrize("name", WITH_DUPLICATES)
def test_random_and_source_grouped_both_leak_planted_duplicates(corpora, reports, strategy, name):
    leaks = sum(
        len(_built(corpora, reports, name, strategy, seed=seed).residual_overlap)
        for seed in SEEDS
    )
    assert leaks > 0, f"{strategy} is expected to leave content duplicates leaking"


@pytest.mark.parametrize("name", (*FIXTURES, "pooled"))
@pytest.mark.parametrize("seed", SEEDS)
def test_adversarial_leaves_zero_residual_cross_split_overlap(corpora, reports, name, seed):
    report = _built(corpora, reports, name, "adversarial", seed=seed)
    assert report.residual_overlap == []
    assert report.residual_sources == {}


@pytest.mark.parametrize("name", WITH_DUPLICATES)
@pytest.mark.parametrize("seed", SEEDS)
def test_adversarial_keeps_every_duplicate_out_of_dev_and_test(corpora, reports, name, seed):
    report = _built(corpora, reports, name, "adversarial", seed=seed)
    assert report.duplicate_example_ids, "the fixture must contain duplicates to mean anything"
    assert report.eval_duplicates == []


def test_eval_duplicates_is_read_from_the_dedup_report_not_from_the_blocks(corpora, reports):
    """A block-derived count would report a flattering zero for the four strategies that
    never build duplicate blocks."""
    report = _built(corpora, reports, "qa_toy", "random", seed=1)
    assert all(not block.duplicate for block in report.blocks)  # singletons
    assert report.eval_duplicates, "random put a duplicate in dev/test and must say so"


def test_verify_splits_does_not_know_which_strategy_produced_the_assignment(corpora, reports):
    report = _built(corpora, reports, "qa_toy", "random", seed=1)
    assigned = apply_splits(corpora["qa_toy"], report.assignment)
    overlap, sources = verify_splits(assigned, reports["qa_toy"])
    assert [f.example_ids for f in overlap] == [f.example_ids for f in report.residual_overlap]
    assert sources == report.residual_sources


# --------------------------------------------------------------------------- #
# Stratification: facets, buckets, and the joint-stratification sweep
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ([], None),
        (["easy"], "easy"),
        (["medium"], "medium"),
        (["hard"], "hard"),
        (["coding", "ambiguous"], None),
        (["easy", "coding"], "easy"),
        (["easy", "hard"], "hard"),
        (["hard", "easy"], "hard"),
        (["EASY"], "easy"),
    ],
)
def test_declared_bucket_takes_the_hardest_tag_and_ignores_the_rest(tags, expected):
    example = _example(0, "q", "a", difficulty_tags=tags)
    bucket = declared_bucket(example)
    assert (str(bucket) if bucket else None) == expected


def test_unlabelled_is_a_stratum_not_an_imputation():
    example = _example(0, "q", "a", difficulty_tags=["coding"])
    assert facet_value(example, "difficulty") == "unlabelled"


def test_facet_value_rejects_an_unknown_facet():
    with pytest.raises(ValueError, match="unknown stratification facet"):
        facet_value(_example(0, "q", "a"), "provenance")


def test_stratum_of_joins_facets_and_an_empty_facet_list_is_one_stratum():
    example = _example(0, "q", "a", task_family="qa", difficulty_tags=["easy"])
    assert stratum_of(example, ("task_family", "difficulty")) == "qa|easy"
    assert stratum_of(example, ()) == ""


def test_block_stratum_resolves_a_mixed_block_by_plurality():
    by_id = {
        "a": _example(1, "q", "x", difficulty_tags=["easy"]),
        "b": _example(2, "q", "x", difficulty_tags=["easy"]),
        "c": _example(3, "q", "x", difficulty_tags=["medium"]),
    }
    for key, example in list(by_id.items()):
        by_id[key] = example.model_copy(update={"example_id": key})
    block = Block(example_ids=("a", "b", "c"))
    assert block_stratum(block, by_id, ("difficulty",)) == "easy"


def test_block_stratum_breaks_a_tie_on_the_stratum_name_not_on_member_order():
    medium = _example(1, "q", "x", difficulty_tags=["medium"]).model_copy(
        update={"example_id": "a"}
    )
    easy = _example(2, "q", "x", difficulty_tags=["easy"]).model_copy(update={"example_id": "b"})
    by_id = {"a": medium, "b": easy}
    assert block_stratum(Block(example_ids=("a", "b")), by_id, ("difficulty",)) == "easy"
    assert block_stratum(Block(example_ids=("b", "a")), by_id, ("difficulty",)) == "easy"


def _tvd(report: SplitReport, examples: list[DatasetExample], facet: str) -> float:
    """Total variation distance between the test split's facet marginal and the corpus's."""
    corpus: dict[str, int] = {}
    for example in examples:
        value = facet_value(example, facet)
        corpus[value] = corpus.get(value, 0) + 1
    counts = report.distributions[facet].get("test", {})
    total = sum(counts.values())
    return 0.5 * sum(
        abs(counts.get(value, 0) / total - count / len(examples))
        for value, count in corpus.items()
    )


def _mean_tvd(corpora, reports, by, facet):
    return sum(
        _tvd(
            _built(corpora, reports, "pooled", "stratified", seed=seed, stratify_by=by),
            corpora["pooled"],
            facet,
        )
        for seed in SEEDS
    ) / len(SEEDS)


@pytest.mark.parametrize("facet", ["task_family", "difficulty"])
def test_single_facet_stratification_improves_the_facet_it_stratifies_on(corpora, reports, facet):
    assert _mean_tvd(corpora, reports, (facet,), facet) < _mean_tvd(corpora, reports, (), facet)


@pytest.mark.parametrize("facet", ["task_family", "difficulty"])
def test_joint_stratification_is_worse_on_both_facets_than_either_alone(corpora, reports, facet):
    """The result worth reading twice: more stratification, less representativeness.

    Five families times four difficulty levels gives 17 non-empty strata over 56
    examples. Every stratum is then too small to fill the small splits, so the marginal
    each facet was supposed to preserve is preserved worse than by stratifying on that
    facet alone.
    """
    joint = ("task_family", "difficulty")
    assert _mean_tvd(corpora, reports, joint, facet) > _mean_tvd(corpora, reports, (facet,), facet)


@pytest.mark.parametrize("seed", SEEDS)
def test_joint_stratification_puts_no_hard_example_in_the_test_split(corpora, reports, seed):
    """And the mechanism is exact: each of the three ``hard`` examples is alone in its
    joint stratum, and a stratum of one apportions entirely to ``train``."""
    joint = ("task_family", "difficulty")
    hard = [e for e in corpora["pooled"] if facet_value(e, "difficulty") == "hard"]
    assert len({stratum_of(e, joint) for e in hard}) == len(hard) == 3
    report = _built(corpora, reports, "pooled", "stratified", seed=seed, stratify_by=joint)
    assert report.distributions["difficulty"].get("test", {}).get("hard", 0) == 0
    assert any("test contains no difficulty='hard'" in w for w in report.warnings)


def test_the_coverage_warning_is_silent_when_no_test_split_was_asked_for(corpora, reports):
    """A split with no ``test`` makes no claim about what the benchmark measures."""
    report = _built(
        corpora, reports, "pooled", "random", ratios={"train": 0.8, "dev": 0.2}
    )
    assert not any("test contains no" in warning for warning in report.warnings)
    assert "test" not in report.split_sizes


def test_the_default_stratification_can_reach_hard_examples(corpora, reports):
    assert any(
        _built(corpora, reports, "pooled", "adversarial", seed=seed)
        .distributions["difficulty"]
        .get("test", {})
        .get("hard", 0)
        for seed in SEEDS
    )


# --------------------------------------------------------------------------- #
# The label-audit hand-off
# --------------------------------------------------------------------------- #


def test_without_quarantine_examples_needing_adjudication_are_scored_on(pooled, reports):
    blocking = audit_labels(pooled).blocking_example_ids
    assert blocking
    report = build_splits(pooled, strategy="adversarial", seed=0, dedup_report=reports["pooled"])
    scored = [eid for eid in blocking if report.assignment[eid] in ("dev", "test")]
    assert scored, "the corpus must exercise this, or the quarantine test proves nothing"


@pytest.mark.parametrize("seed", SEEDS)
def test_quarantined_examples_never_enter_an_evaluation_split(pooled, reports, seed):
    blocking = audit_labels(pooled).blocking_example_ids
    report = build_splits(
        pooled, strategy="adversarial", seed=seed, dedup_report=reports["pooled"],
        quarantine_ids=blocking,
    )
    assert all(report.assignment[eid] == "train" for eid in blocking)
    assert report.quarantined == tuple(blocking)


def test_quarantine_keeps_the_examples_in_the_dataset(pooled, reports):
    blocking = audit_labels(pooled).blocking_example_ids
    report = build_splits(
        pooled, strategy="adversarial", dedup_report=reports["pooled"], quarantine_ids=blocking
    )
    assert len(report.assignment) == len(pooled)


def test_quarantine_is_honoured_by_the_adversarial_strategy_only(pooled, reports):
    """Silently applying it to ``random`` would make random not random."""
    blocking = audit_labels(pooled).blocking_example_ids
    report = build_splits(
        pooled, strategy="random", seed=0, dedup_report=reports["pooled"], quarantine_ids=blocking
    )
    assert any(report.assignment[eid] in ("dev", "test") for eid in blocking)


def test_a_quarantined_singleton_is_pinned_to_train_even_without_duplicates():
    examples = [_example(i, f"question {i}", f"answer {i}") for i in range(10)]
    assignment = adversarial_splits(examples, quarantine_ids=["e03"], stratify_by=())
    assert assignment.of_example["e03"] == "train"


# --------------------------------------------------------------------------- #
# Temporal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-01-03", "2024-01-03"),
        ("2024-01-03T10:00:00", "2024-01-03T10:00:00"),
        ("2024-1-3", None),
        ("last tuesday", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_timestamp_is_a_validity_gate_not_a_converter(value, expected):
    assert parse_timestamp(value) == expected


def test_temporal_puts_the_oldest_in_train_and_the_newest_in_test(qa):
    report = build_splits(qa, strategy="temporal")
    by_id = {example.example_id: example for example in qa}
    stamps = {split: [] for split in SPLIT_ORDER}
    for example_id, split in report.assignment.items():
        stamps[split].append(by_id[example_id].created_at)
    assert max(stamps["train"]) <= min(stamps["dev"])
    assert max(stamps["dev"]) <= min(stamps["test"])


def test_temporal_breaks_timestamp_ties_on_insertion_order():
    examples = [_example(i, f"q{i}", "a", created_at="2024-01-01") for i in range(10)]
    keys, warnings = temporal_order(examples)
    assert warnings == []
    assert [key[1] for key in keys.values()] == list(range(10))


def test_an_undated_example_forward_fills_and_is_named_in_a_warning():
    examples = [
        _example(0, "q0", "a", created_at="2024-01-01"),
        _example(1, "q1", "a", created_at=None),
        _example(2, "q2", "a", created_at="2024-06-01"),
    ]
    keys, warnings = temporal_order(examples)
    assert keys["e01"][0] == "2024-01-01"  # inherits its predecessor, keeps its position
    assert keys["e00"] < keys["e01"] < keys["e02"]
    assert len(warnings) == 1 and "e01" in warnings[0]


def test_an_unparseable_timestamp_is_treated_as_absent_not_sorted_as_a_string():
    """``"2024-1-3"`` sorts after ``"2024-01-30"``; a silently mis-ordered temporal
    split is worse than a loud one."""
    examples = [
        _example(0, "q0", "a", created_at="2024-01-30"),
        _example(1, "q1", "a", created_at="2024-1-3"),
    ]
    keys, warnings = temporal_order(examples)
    assert keys["e00"][0] == "2024-01-30" and keys["e01"][0] == "2024-01-30"
    assert keys["e00"] < keys["e01"]
    assert len(warnings) == 1


def test_a_leading_undated_example_sorts_into_train_not_test():
    examples = [_example(0, "q0", "a", created_at=None)]
    examples += [_example(i, f"q{i}", "a", created_at=f"2024-01-{i:02d}") for i in range(1, 11)]
    report = build_splits(examples, strategy="temporal")
    assert report.assignment["e00"] == "train"


def test_temporal_makes_no_claim_about_duplicates(pooled, reports):
    """It leaks less than random on this corpus only because paraphrases were authored
    on adjacent days — a fact about the fixtures, not a property of temporal splitting.
    Six duplicate-cluster examples still land in dev/test."""
    report = build_splits(pooled, strategy="temporal", dedup_report=reports["pooled"])
    assert report.eval_duplicates
    assert report.residual_overlap


def test_temporal_on_the_pooled_corpus_destroys_the_family_balance(pooled, reports):
    """The fixtures were authored family by family, so time-ordering them is close to
    ordering them by ``task_family``. The test split then measures one family."""
    report = build_splits(pooled, strategy="temporal", dedup_report=reports["pooled"])
    assert len(report.distributions["task_family"]["test"]) < 5
    assert any("test contains no task_family=" in warning for warning in report.warnings)


# --------------------------------------------------------------------------- #
# Honest negatives
# --------------------------------------------------------------------------- #


def test_retrieval_toys_clean_row_is_the_answer_gate_not_an_absence_of_leakage(corpora, reports):
    """Its gold answers are document-id lists that differ, so the near stage's answer
    gate rejects every pair before a threshold is consulted. Zero residual overlap for
    *every* strategy on this fixture is therefore vacuous."""
    assert reports["retrieval_toy"].findings == []
    for strategy in STRATEGIES:
        report = build_splits(
            corpora["retrieval_toy"], strategy=strategy, dedup_report=reports["retrieval_toy"]
        )
        assert report.residual_overlap == []


def test_the_adversarial_guarantee_is_inherited_from_a_heuristic(corpora):
    """``ret-001``/``ret-002`` is a planted cross-split paraphrase that no lexical
    threshold can reach (``docs/DEDUPLICATION.md``). With the answer gate disabled the
    ladder finds the fixture's other two clusters and the adversarial split keeps them
    intact — while ret-001/ret-002 straddles ``dev``/``test``, and the verifier reports
    a clean zero. A guarantee inherited from a heuristic is a heuristic guarantee."""
    examples = corpora["retrieval_toy"]
    report_open = find_duplicates(examples, near_require_same_answer=False)
    assert [f.example_ids for f in report_open.findings] == [
        ["ret-003", "ret-008"],
        ["ret-006", "ret-007", "ret-009"],
    ]
    built = build_splits(examples, strategy="adversarial", seed=0, dedup_report=report_open)
    assert built.residual_overlap == []  # the verifier is blind to it too
    assert built.assignment["ret-001"] == "train"
    assert built.assignment["ret-002"] == "test"


@pytest.mark.parametrize("name", ["qa_toy", "agent_tasks_toy", "bio_toy", "safety_toy", "pooled"])
def test_protecting_only_test_changes_nothing_on_this_corpus(corpora, reports, name):
    """An honest null. Duplicate blocks are dealt first and ``train``'s target dwarfs
    ``dev``'s, so they never reach ``dev`` even when allowed to. Exercised for real by
    the synthetic corpus below."""
    for seed in SEEDS:
        report = build_splits(
            corpora[name], strategy="adversarial", seed=seed, dedup_report=reports[name],
            eval_splits=("test",),
        )
        assert not [eid for eid in report.duplicate_example_ids if report.assignment[eid] == "dev"]


def test_protecting_only_test_bites_when_duplicates_exhaust_the_train_quota():
    # Four identical pairs: every example is a duplicate, and train's target is 6 of 8.
    examples = []
    for i in range(4):
        examples.append(_example(2 * i, f"question number {i}", f"answer {i}"))
        examples.append(_example(2 * i + 1, f"question number {i}", f"answer {i}"))
    report = find_duplicates(examples)
    assert len(report.findings) == 4

    strict = build_splits(
        examples, strategy="adversarial", seed=0, dedup_report=report, stratify_by=()
    )
    assert strict.split_sizes == {"train": 8, "dev": 0, "test": 0}
    assert strict.residual_overlap == []
    assert any("quota is" in warning for warning in strict.warnings)

    relaxed = build_splits(
        examples, strategy="adversarial", seed=0, dedup_report=report, stratify_by=(),
        eval_splits=("test",),
    )
    assert relaxed.split_sizes["dev"] == 2
    assert [eid for eid in relaxed.duplicate_example_ids if relaxed.assignment[eid] == "dev"]
    assert relaxed.residual_overlap == []  # the pair moved to dev *together*


def test_an_impossible_split_warns_and_drops_nothing():
    examples = []
    for i in range(4):
        examples.append(_example(2 * i, f"question number {i}", f"answer {i}"))
        examples.append(_example(2 * i + 1, f"question number {i}", f"answer {i}"))
    report = build_splits(examples, strategy="adversarial", seed=0, stratify_by=())
    assert len(report.assignment) == len(examples)
    assert report.warnings


# --------------------------------------------------------------------------- #
# Report, immutability, and IO
# --------------------------------------------------------------------------- #


def test_apply_splits_never_mutates_its_input(qa, reports):
    before = [str(example.split) for example in qa]
    report = build_splits(qa, strategy="random", seed=2, dedup_report=reports["qa_toy"])
    assigned = apply_splits(qa, report.assignment)
    assert [str(example.split) for example in qa] == before
    assert any(str(a.split) != b for a, b in zip(assigned, before, strict=True))


def test_apply_splits_leaves_an_unassigned_example_alone():
    examples = [_example(0, "q", "a")]
    assert apply_splits(examples, {})[0].split is Split.UNASSIGNED


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_report_to_dict_is_json_serializable(qa, reports, strategy):
    report = build_splits(qa, strategy=strategy, dedup_report=reports["qa_toy"])
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["counts"]["examples"] == len(qa)
    assert payload["spec"]["strategy"] == strategy


def test_achieved_ratios_sum_to_one(pooled, reports):
    report = build_splits(pooled, strategy="adversarial", dedup_report=reports["pooled"])
    assert abs(sum(report.achieved_ratios.values()) - 1.0) < 1e-6


def test_counts_are_consistent_with_the_assignment(pooled, reports):
    report = build_splits(pooled, strategy="adversarial", dedup_report=reports["pooled"])
    counts = report.counts()
    assert counts["examples"] == sum(report.split_sizes.values())
    assert counts["grouped_blocks"] == len([b for b in report.blocks if b.size > 1])


def test_format_report_names_the_strategy_and_the_residual_leakage(qa, reports):
    text = format_report(build_splits(qa, strategy="adversarial", dedup_report=reports["qa_toy"]))
    assert "adversarial" in text
    assert "residual overlap    : 0" in text
    assert "grouped blocks" in text


def test_write_splits_is_byte_identical_across_runs(tmp_path, qa, reports):
    report = build_splits(qa, strategy="adversarial", seed=1, dedup_report=reports["qa_toy"])
    first = write_splits(qa, report, tmp_path / "a")
    second = write_splits(qa, report, tmp_path / "b")
    for name, path in first.items():
        assert path.read_bytes() == second[name].read_bytes()


def test_written_splits_round_trip_through_the_loader(tmp_path, qa, reports):
    report = build_splits(qa, strategy="adversarial", seed=1, dedup_report=reports["qa_toy"])
    written = write_splits(qa, report, tmp_path)
    recovered = {}
    for name in ("train", "dev", "test"):
        for example in loader.load_examples(written[name]):
            assert str(example.split) == name
            recovered[example.example_id] = name
    assert recovered == report.assignment


def test_write_splits_emits_a_report_json(tmp_path, qa, reports):
    report = build_splits(qa, strategy="random", dedup_report=reports["qa_toy"])
    written = write_splits(qa, report, tmp_path)
    payload = json.loads(written["report"].read_text())
    assert payload["spec"]["seed"] == DEFAULT_SEED
    assert payload["counts"]["examples"] == len(qa)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_build_splits_is_no_longer_pending():
    assert "build-splits" not in _PENDING
    action = next(
        a for a in build_parser()._subparsers._group_actions[0].choices["build-splits"]._actions
        if a.dest == "strategy"
    )
    assert set(action.choices) == set(STRATEGIES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.7,0.1,0.2", {"train": 0.7, "dev": 0.1, "test": 0.2}),
        ("train=0.8,test=0.2", {"train": 0.8, "test": 0.2}),
        (" 0.6 , 0.2 , 0.2 ", {"train": 0.6, "dev": 0.2, "test": 0.2}),
    ],
)
def test_parse_ratios(text, expected):
    assert _parse_ratios(text) == expected


def test_parse_ratios_rejects_the_wrong_number_of_positional_values():
    with pytest.raises(Exception, match="three positional values"):
        _parse_ratios("0.8,0.2")


def test_cli_build_splits_writes_a_release_skeleton(tmp_path, capsys):
    code = main([
        "build-splits",
        "--dataset", str(loader.FIXTURES_ROOT / "qa_toy" / "examples.jsonl"),
        "--strategy", "adversarial",
        "--out", str(tmp_path),
    ])
    assert code == 0
    assert (tmp_path / "splits" / "train.jsonl").exists()
    assert (tmp_path / "split_report.json").exists()
    assert "residual overlap    : 0" in capsys.readouterr().out


def test_cli_fail_on_leakage_gates_the_leaky_strategies(tmp_path):
    dataset = str(loader.FIXTURES_ROOT / "qa_toy" / "examples.jsonl")
    leaky = main(["build-splits", "--dataset", dataset, "--strategy", "random",
                  "--seed", "0", "--out", str(tmp_path / "r"), "--fail-on-leakage"])
    clean = main(["build-splits", "--dataset", dataset, "--strategy", "adversarial",
                  "--seed", "0", "--out", str(tmp_path / "a"), "--fail-on-leakage"])
    assert leaky == 1
    assert clean == 0


def test_cli_json_output_is_valid_json(tmp_path, capsys):
    main(["build-splits", "--dataset", str(loader.FIXTURES_ROOT / "bio_toy" / "examples.jsonl"),
          "--out", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out.split("\nwrote")[0])
    assert payload["spec"]["strategy"] == "stratified"


def test_cli_quarantine_blocking_keeps_adjudication_cases_out_of_evaluation(tmp_path, capsys):
    main(["build-splits", "--dataset", str(loader.FIXTURES_ROOT / "qa_toy" / "examples.jsonl"),
          "--strategy", "adversarial", "--out", str(tmp_path), "--quarantine-blocking", "--json"])
    payload = json.loads(capsys.readouterr().out.split("\nwrote")[0])
    assert payload["quarantined"]
    assert all(payload["assignment"][eid] == "train" for eid in payload["quarantined"])


def test_cli_ignore_answer_gate_changes_the_blocks(tmp_path, capsys):
    dataset = str(loader.FIXTURES_ROOT / "retrieval_toy" / "examples.jsonl")
    main(["build-splits", "--dataset", dataset, "--strategy", "adversarial",
          "--out", str(tmp_path), "--ignore-answer-gate", "--json"])
    payload = json.loads(capsys.readouterr().out.split("\nwrote")[0])
    assert payload["counts"]["grouped_blocks"] == 2
