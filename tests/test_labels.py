"""Tests for the label-quality audit (Milestone 6).

Covered here:

  * **Primitives** — the emptiness predicate and its three traps (``False``, ``0``,
    ``0.0`` are labels, not absences), coarse answer shapes, label keys, and the strict
    majority rule.
  * **Missing** — recall against every fixture's ``defects.yaml``; a nested rubric whose
    leaves are all empty; and the reason a retrieval example with an empty gold list is
    *not* reported here.
  * **Conflicts** — the three sources. Annotator disagreement is scored against ground
    truth. Shared-prompt conflict fires on nothing in the corpus (all four shared-prompt
    clusters are true duplicates), so a synthetic pair exercises it and a test pins the
    honest negative. Relevance conflict recovers ``ret-004``, which no other stage in
    the pipeline sees, while ``ret-005`` passes clean by set equality.
  * **Ambiguity** — the calibration. With the ``ambiguous`` tags withheld, the lexical
    heuristic recovers 4 of 6 declared targets with zero false positives. Both rejected
    widths are pinned by tests asserting *why* they were rejected, both guards are shown
    to be load-bearing by the false positive each one prevents, and the two examples the
    heuristic provably cannot reach are pinned as documented misses.
  * **The qa-008 / ret-004 pair** — the same sentence pattern, separated only by the
    shape of the gold answer. No rule keyed on prompt text alone can tell them apart, so
    the test flips the answer shape and shows the verdict flip with it.
  * **Format** — the shape minority and the vocabulary outlier; the key-set variation
    that is *not* a defect; the vocabulary check auto-disabling on free-text families;
    and the guards against tiny and genuinely heterogeneous families.
  * **Adjudication** — the single blocking rule (an issue with no proposed resolution),
    a 2-1 majority resolving where a 1-1 tie cannot, queue determinism, and a
    byte-identical JSONL export.
  * **The blind spot** — the three planted ``mislabeled`` defects are marked
    ``auto_detectable: false`` and are found by no check. A regression test asserts they
    stay invisible, so the limitation cannot rot into a silent claim of coverage.

Everything is CPU-only and offline.
"""

from __future__ import annotations

import json

import pytest

from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup.fuzzy import tokenize
from eval_dataset_forge.labels import (
    AdjudicationItem,
    LabelAuditReport,
    agreement,
    ambiguous_labels,
    annotations,
    annotator_conflicts,
    answer_shape,
    audit_labels,
    build_queue,
    declared_ambiguous,
    family_expectations,
    format_report,
    has_label,
    inconsistent_formats,
    label_key,
    majority_value,
    missing_labels,
    needs_adjudication,
    open_set_request,
    relevance_conflicts,
    relevance_labelled,
    shared_prompt_conflicts,
    underspecified_prompt,
    write_queue,
)
from eval_dataset_forge.labels.audit import (
    MAX_UNDERSPECIFIED_TOKENS,
    WH_WORDS,
)
from eval_dataset_forge.schemas import DatasetExample, LabelIssue

FIXTURES = ["qa_toy", "agent_tasks_toy", "retrieval_toy", "bio_toy", "safety_toy"]

# The six examples the fixtures declare ambiguous via difficulty_tags.
DECLARED_AMBIGUOUS = {"qa-008", "qa-009", "agent-009", "bio-007", "saf-007", "ret-005"}

# The four the lexical heuristic can reach without being told.
LEXICALLY_REACHABLE = {"qa-008", "qa-009", "agent-009", "bio-007"}


def _load(name: str) -> tuple[list[DatasetExample], dict]:
    fixture_dir = loader.FIXTURES_ROOT / name
    return (
        loader.load_examples(fixture_dir / "examples.jsonl"),
        loader.load_defects(fixture_dir / "defects.yaml"),
    )


def _corpus() -> list[DatasetExample]:
    """Every fixture example, concatenated. Ids are unique across fixtures."""
    return [ex for name in FIXTURES for ex in _load(name)[0]]


def _ex(example_id: str = "x-1", **overrides) -> DatasetExample:
    """A minimal valid example; override any field."""
    base = {
        "example_id": example_id,
        "source_id": f"src-{example_id}",
        "task_family": "qa",
        "input": "What is the capital of France?",
        "expected_output": "Paris",
    }
    return DatasetExample(**{**base, **overrides})


def _without_ambiguous_tags(examples: list[DatasetExample]) -> list[DatasetExample]:
    """The corpus with every ``ambiguous`` declaration withheld.

    The heuristic's real question is "would we have found these if nobody had told us?",
    which can only be asked of a corpus where nobody has.
    """
    stripped = []
    for example in examples:
        record = example.model_dump()
        record["difficulty_tags"] = [t for t in example.difficulty_tags if t != "ambiguous"]
        stripped.append(DatasetExample(**record))
    return stripped


def _planted_label_issues(defects: dict) -> list[dict]:
    return defects["planted_defects"].get("label_issues", [])


def _ids(findings) -> set[str]:
    return {finding.example_id for finding in findings}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["", "   ", [], {}, {"solution": ""}, {"a": {"b": "  "}}, [""]])
def test_has_label_is_false_for_contentless_answers(value) -> None:
    assert has_label(value) is False


@pytest.mark.parametrize("value", [False, 0, 0.0, True, "Paris", ["d1"], {"a": "b"}, 1945])
def test_has_label_is_true_for_real_labels(value) -> None:
    """``False``/``0``/``0.0`` are labels. A bare ``if not value`` would delete all three."""
    assert has_label(value) is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "boolean"),
        (False, "boolean"),
        ("Paris", "string"),
        (4, "number"),
        (2.5, "number"),
        (["d1"], "list"),
        ({"a": 1}, "mapping"),
    ],
)
def test_answer_shape(value, expected) -> None:
    assert answer_shape(value) == expected


def test_answer_shape_checks_bool_before_int() -> None:
    """``isinstance(True, int)`` is true; a boolean label is not a numeric one."""
    assert answer_shape(True) != answer_shape(1)


def test_label_key_folds_strings_but_not_types() -> None:
    assert label_key(" 1945 ") == label_key("1945")
    assert label_key("Drug") == label_key("drug")
    assert label_key("4") != label_key(4)
    assert label_key("true") != label_key(True)


def test_annotators_writing_different_types_are_not_recorded_as_agreeing() -> None:
    """A folded string must not collide with a scalar of another type.

    ``normalize_text("4")`` is ``"4"`` and so is ``str(4)``; without JSON-encoding the
    folded string, an annotator who wrote ``"4"`` and one who wrote ``4`` would agree.
    """
    example = _ex(
        expected_output="4",
        labels={
            "annotations": [{"annotator": "a1", "value": "4"}, {"annotator": "a2", "value": 4}]
        },
    )
    assert _ids(annotator_conflicts([example])) == {"x-1"}
    assert agreement([example]).n_disagreed == 1


def test_majority_value_requires_a_strict_majority() -> None:
    anns = [{"annotator": "a1", "value": "x"}, {"annotator": "a2", "value": "y"}]
    assert majority_value(annotations(_ex(labels={"annotations": anns}))) is None

    anns.append({"annotator": "a3", "value": "x"})
    assert majority_value(annotations(_ex(labels={"annotations": anns}))) == "x"


def test_majority_value_of_a_three_way_split_is_none() -> None:
    anns = [{"annotator": f"a{i}", "value": v} for i, v in enumerate("xyz")]
    assert majority_value(annotations(_ex(labels={"annotations": anns}))) is None


def test_majority_value_returns_the_unnormalized_form() -> None:
    anns = [
        {"annotator": "a1", "value": " Drug "},
        {"annotator": "a2", "value": "drug"},
        {"annotator": "a3", "value": "chemical"},
    ]
    assert majority_value(annotations(_ex(labels={"annotations": anns}))) == " Drug "


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"annotations": "not-a-list"},
        {"annotations": [1, 2]},
        {"annotations": [{"annotator": "a"}]},
    ],
)
def test_annotations_is_defensive_about_shape(labels) -> None:
    """``labels`` is free-form by contract, so a malformed block yields nothing."""
    assert annotations(_ex(labels=labels)) == []


def test_annotations_without_an_annotator_id_still_count_as_votes() -> None:
    items = annotations(_ex(labels={"annotations": [{"value": "x"}, {"value": "y"}]}))
    assert len(items) == 2
    assert {item.annotator for item in items} == {"?"}


# --------------------------------------------------------------------------- #
# Missing labels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURES)
def test_missing_labels_match_the_fixture_ground_truth(name: str) -> None:
    examples, defects = _load(name)
    planted = {p["id"] for p in _planted_label_issues(defects) if p["kind"] == "missing"}
    assert _ids(missing_labels(examples)) == planted


def test_missing_flags_a_rubric_whose_leaves_are_all_empty() -> None:
    example = _ex(expected_output={"reference_solution": "", "success_criteria": "  "})
    (finding,) = missing_labels([example])
    assert finding.kind == LabelIssue.MISSING
    assert "no non-empty leaf values" in finding.detail


def test_missing_does_not_flag_a_false_or_zero_label() -> None:
    assert missing_labels([_ex("a", expected_output=False), _ex("b", expected_output=0)]) == []


def test_missing_skips_examples_whose_relevance_is_labelled_twice() -> None:
    """An empty gold doc-id list is a *claim* the passage flags can check, not an absence."""
    example = _ex(
        expected_output=[],
        metadata={"retrieval": {"docs": [{"doc_id": "d1", "text": "…", "relevant": False}]}},
    )
    assert relevance_labelled(example) is True
    assert missing_labels([example]) == []


def test_missing_still_fires_when_a_retrieval_example_has_a_string_answer() -> None:
    """Passages plus a non-list gold answer: the relevance check cannot own this one."""
    example = _ex(
        expected_output="",
        metadata={"retrieval": {"docs": [{"doc_id": "d1", "text": "…", "relevant": True}]}},
    )
    assert relevance_labelled(example) is False
    assert _ids(missing_labels([example])) == {"x-1"}


def test_missing_reports_the_empty_shape() -> None:
    (finding,) = missing_labels([_ex(expected_output={})])
    assert "empty mapping" in finding.detail


# --------------------------------------------------------------------------- #
# Conflicts: annotators
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURES)
def test_annotator_conflicts_match_the_fixture_ground_truth(name: str) -> None:
    examples, defects = _load(name)
    planted = {p["id"] for p in _planted_label_issues(defects) if p["kind"] == "conflicting"}
    assert _ids(annotator_conflicts(examples)) == planted


def test_annotator_conflict_records_whether_a_majority_exists() -> None:
    tie = _ex("tie", labels={"annotations": [{"annotator": "a1", "value": "x"},
                                             {"annotator": "a2", "value": "y"}]})
    won = _ex("won", labels={"annotations": [{"annotator": "a1", "value": "x"},
                                             {"annotator": "a2", "value": "y"},
                                             {"annotator": "a3", "value": "x"}]})
    tie_finding, won_finding = annotator_conflicts([tie, won])
    assert "no majority among 2" in tie_finding.detail
    assert "majority 'x'" in won_finding.detail


def test_unanimous_annotators_are_not_a_conflict() -> None:
    example = _ex(labels={"annotations": [{"annotator": "a1", "value": "Paris"},
                                          {"annotator": "a2", "value": " paris "}]})
    assert annotator_conflicts([example]) == []


def test_a_single_annotator_cannot_disagree() -> None:
    lone = _ex(labels={"annotations": [{"annotator": "a1", "value": "x"}]})
    assert annotator_conflicts([lone]) == []


# --------------------------------------------------------------------------- #
# Conflicts: shared prompts
# --------------------------------------------------------------------------- #


def test_shared_prompt_conflicts_fire_on_nothing_in_the_corpus() -> None:
    """An honest negative.

    ``duplicate_input_clusters`` exposes every byte-identical prompt precisely so this
    stage can take the difference against the exact-duplicate clusters. On the committed
    fixtures the difference is empty: all four shared-prompt clusters agree on their gold
    answers, i.e. they are duplicates, not conflicts. The channel is real; this corpus
    just has no instance of it.
    """
    assert shared_prompt_conflicts(_corpus()) == []


def test_shared_prompt_conflict_fires_on_a_synthetic_pair() -> None:
    pair = [
        _ex("a", input="What is the capital of Australia?", expected_output="Sydney"),
        _ex("b", input="What is the capital of Australia?", expected_output="Canberra"),
    ]
    findings = shared_prompt_conflicts(pair)
    assert _ids(findings) == {"a", "b"}
    assert all(f.kind == LabelIssue.CONFLICTING for f in findings)
    assert '"Sydney"' in findings[0].detail and '"Canberra"' in findings[0].detail


def test_shared_prompt_with_an_identical_answer_is_a_duplicate_not_a_conflict() -> None:
    pair = [_ex("a", expected_output="Paris"), _ex("b", expected_output="Paris")]
    assert shared_prompt_conflicts(pair) == []


def test_shared_prompt_conflict_distinguishes_a_string_from_a_number() -> None:
    """``canonical_output`` keeps ``"4"`` and ``4`` apart, so they are a conflict."""
    pair = [_ex("a", input="2+2?", expected_output="4"), _ex("b", input="2+2?", expected_output=4)]
    assert _ids(shared_prompt_conflicts(pair)) == {"a", "b"}


# --------------------------------------------------------------------------- #
# Conflicts: relevance channels
# --------------------------------------------------------------------------- #


def test_relevance_conflict_recovers_the_planted_incomplete_relevance() -> None:
    examples, _ = _load("retrieval_toy")
    findings = relevance_conflicts(examples)
    assert _ids(findings) == {"ret-004"}
    assert "d9 marked relevant but absent from expected_output" in findings[0].detail


def test_the_over_specific_query_passes_clean_by_set_equality() -> None:
    """``ret-005`` has an empty gold list *and* no relevant passage. The two agree.

    A rule that read "empty ``expected_output`` means the label is missing" would flag it.
    Set equality is what makes an unanswerable query legal and ``ret-004`` illegal.
    """
    examples, _ = _load("retrieval_toy")
    ret_005 = next(ex for ex in examples if ex.example_id == "ret-005")
    assert relevance_labelled(ret_005) is True
    assert relevance_conflicts([ret_005]) == []
    assert missing_labels([ret_005]) == []


def test_relevance_conflict_names_a_spurious_gold_doc_id() -> None:
    example = _ex(
        expected_output=["d2"],
        metadata={"retrieval": {"docs": [{"doc_id": "d1", "text": "…", "relevant": True},
                                         {"doc_id": "d2", "text": "…", "relevant": False}]}},
    )
    (finding,) = relevance_conflicts([example])
    assert "d1 marked relevant but absent" in finding.detail
    assert "expected_output names d2" in finding.detail


def test_relevance_conflict_ignores_agreeing_examples() -> None:
    examples, _ = _load("retrieval_toy")
    agreeing = [ex for ex in examples if ex.example_id not in {"ret-004"}]
    assert relevance_conflicts(agreeing) == []


def test_relevance_check_ignores_examples_without_passages() -> None:
    assert relevance_conflicts([_ex(expected_output=["d1"])]) == []


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def test_corpus_agreement_rate_is_zero_by_construction() -> None:
    """Annotations were planted only where a conflict was planted.

    The rate describes how the fixtures were built, not the reliability of any real
    annotation process. Asserted so the number is never quoted as if it did.
    """
    summary = agreement(_corpus())
    assert summary.n_annotated == 4
    assert summary.n_agreed == 0
    assert summary.rate == 0.0


def test_agreement_rate_is_none_when_undefined() -> None:
    """An undefined rate and a rate of zero mean opposite things."""
    assert agreement([_ex()]).rate is None


def test_agreement_counts_only_multiply_annotated_examples() -> None:
    examples = [
        _ex("a", labels={"annotations": [{"annotator": "a1", "value": "x"},
                                         {"annotator": "a2", "value": "x"}]}),
        _ex("b", labels={"annotations": [{"annotator": "a1", "value": "x"}]}),
        _ex("c"),
    ]
    summary = agreement(examples)
    assert (summary.n_annotated, summary.n_agreed, summary.rate) == (1, 1, 1.0)


# --------------------------------------------------------------------------- #
# Ambiguity: the declaration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURES)
def test_declared_ambiguity_matches_the_fixture_tags(name: str) -> None:
    examples, _ = _load(name)
    declared = {ex.example_id for ex in examples if declared_ambiguous(ex)}
    assert declared == DECLARED_AMBIGUOUS & {ex.example_id for ex in examples}


@pytest.mark.parametrize("name", FIXTURES)
def test_ambiguous_labels_match_the_fixture_ground_truth(name: str) -> None:
    examples, defects = _load(name)
    planted = {p["id"] for p in _planted_label_issues(defects) if p["kind"] == "ambiguous"}
    found = _ids(ambiguous_labels(examples))
    assert planted <= found
    # retrieval_toy files ret-005 under `relevance_issues`, not `label_issues`; it is
    # nonetheless tagged `ambiguous` in the example itself.
    assert found <= DECLARED_AMBIGUOUS


def test_heuristics_can_be_disabled_leaving_only_declarations() -> None:
    found = _ids(ambiguous_labels(_corpus(), heuristics=False))
    assert found == DECLARED_AMBIGUOUS


# --------------------------------------------------------------------------- #
# Ambiguity: the calibration
# --------------------------------------------------------------------------- #


def test_the_heuristic_recovers_four_of_six_declarations_with_no_false_positives() -> None:
    """The headline measurement: what the heuristic finds when nobody has tagged anything."""
    found = _ids(ambiguous_labels(_without_ambiguous_tags(_corpus())))
    assert found == LEXICALLY_REACHABLE
    assert found <= DECLARED_AMBIGUOUS  # zero false positives


def test_the_two_missed_declarations_are_unreachable_by_any_lexical_rule() -> None:
    """``saf-007`` is a semantic judgement; ``ret-005`` is *over*-specified, not under-.

    Neither is short, neither is an open-set request. Declared ambiguity and detectable
    ambiguity are different sets, and this pins the difference rather than hiding it.
    """
    stripped = {ex.example_id: ex for ex in _without_ambiguous_tags(_corpus())}
    for example_id in ("saf-007", "ret-005"):
        example = stripped[example_id]
        assert len(tokenize(example.input)) > MAX_UNDERSPECIFIED_TOKENS
        assert underspecified_prompt(example) is None
        assert open_set_request(example) is None


def test_width_three_loses_targets_the_open_set_signal_cannot_recover() -> None:
    """Narrowing to 3 tokens strips the dangling-reference signal down to one example.

    The combined heuristic still recovers ``qa-008``, because ``open_set_request`` has no
    width and matches "Name a primary color." on its own — the two signals are redundant
    at the default width and complementary below it. ``agent-009`` and ``bio-007`` are
    lost outright.
    """
    stripped = _without_ambiguous_tags(_corpus())
    dangling = {ex.example_id for ex in stripped if underspecified_prompt(ex, max_tokens=3)}
    assert dangling == {"qa-009"}

    found = _ids(ambiguous_labels(stripped, max_tokens=3))
    assert found == {"qa-008", "qa-009"}
    assert LEXICALLY_REACHABLE - found == {"agent-009", "bio-007"}


def test_width_eight_admits_a_finding_the_ground_truth_files_elsewhere() -> None:
    """At ``max_tokens=8`` the heuristic flags ``bio-004``.

    ``defects.yaml`` files ``bio-004`` under ``conflicting`` — its annotators split on
    whether 'aspirin' is a drug or a chemical compound — and arguably they split *because*
    "Classify the entity 'aspirin' into a single category." never says which taxonomy.
    Scoring it as a false positive is the conservative reading, and the default width is
    not tuned on it either way.
    """
    stripped = _without_ambiguous_tags(_corpus())
    found = _ids(ambiguous_labels(stripped, max_tokens=8))
    assert found - LEXICALLY_REACHABLE == {"bio-004"}


@pytest.mark.parametrize("width", [4, 5, 6, 7])
def test_widths_four_through_seven_are_indistinguishable_on_this_corpus(width: int) -> None:
    """A sparsity plateau, not a robustness basin.

    No fixture prompt of 5-7 tokens passes the guards, so the widths agree by accident of
    the corpus. Recorded as such: the default is the *smallest* width attaining full
    reachable recall, not the middle of a basin we have evidence for.
    """
    stripped = _without_ambiguous_tags(_corpus())
    assert _ids(ambiguous_labels(stripped, max_tokens=width)) == LEXICALLY_REACHABLE


def test_dropping_the_wh_guard_admits_a_false_positive() -> None:
    """Without it, "What is 2 + 2?" — four tokens, scalar answer — is called ambiguous."""
    corpus = _without_ambiguous_tags(_corpus())
    would_fire = {
        ex.example_id
        for ex in corpus
        if len(tokenize(ex.input)) <= MAX_UNDERSPECIFIED_TOKENS
        and not isinstance(ex.expected_output, (list, dict))
    }
    assert "qa-005" in would_fire
    assert set(tokenize(next(ex for ex in corpus if ex.example_id == "qa-005").input)) & WH_WORDS
    assert "qa-005" not in _ids(ambiguous_labels(corpus))


def test_dropping_the_scalar_gold_gate_admits_a_false_positive() -> None:
    """Without it, ``ret-004`` ("Name a noble gas.") is called ambiguous."""
    corpus = _without_ambiguous_tags(_corpus())
    would_fire = {
        ex.example_id
        for ex in corpus
        if len(tokenize(ex.input)) <= MAX_UNDERSPECIFIED_TOKENS
        and not set(tokenize(ex.input)) & WH_WORDS
    }
    assert "ret-004" in would_fire
    assert "ret-004" not in _ids(ambiguous_labels(corpus))


def test_qa_008_and_ret_004_are_separated_only_by_the_shape_of_the_answer() -> None:
    """"Name a primary color." and "Name a noble gas." are the same sentence pattern.

    Any rule keyed on prompt text alone must flag both or neither. What separates them is
    that a list-valued gold answer can enumerate every acceptable response. Swapping only
    the answer shape flips both verdicts, which is the whole argument for the gate.
    """
    qa_008 = _ex("qa-008", input="Name a primary color.", expected_output="Red")
    ret_004 = _ex("ret-004", input="Name a noble gas.", expected_output=[])

    assert open_set_request(qa_008) is not None
    assert open_set_request(ret_004) is None

    # Same prompts, answer shapes exchanged: the verdicts exchange too.
    assert open_set_request(_ex(input="Name a primary color.", expected_output=[])) is None
    assert open_set_request(_ex(input="Name a noble gas.", expected_output="Helium")) is not None


def test_a_short_wh_question_is_a_documented_miss() -> None:
    """The wh-guard's cost. "What is it?" is ambiguous and is not flagged."""
    assert underspecified_prompt(_ex(input="What is it?", expected_output="yes")) is None


def test_open_set_request_is_redundant_on_the_fixtures() -> None:
    """It flags only ``qa-008``, which is short enough for the dangling-reference signal.

    The channel exists for the longer open-set prompts the fixtures do not contain.
    """
    corpus = _without_ambiguous_tags(_corpus())
    open_set = {ex.example_id for ex in corpus if open_set_request(ex)}
    dangling = {ex.example_id for ex in corpus if underspecified_prompt(ex)}
    assert open_set == {"qa-008"}
    assert open_set <= dangling


def test_open_set_request_reaches_a_long_prompt_the_width_rule_cannot() -> None:
    example = _ex(
        input="Name an element from the noble gas column of the periodic table, please.",
        expected_output="Helium",
    )
    assert underspecified_prompt(example) is None
    assert open_set_request(example) is not None
    assert _ids(ambiguous_labels([example])) == {"x-1"}


def test_ambiguous_detail_names_every_signal_that_fired() -> None:
    examples, _ = _load("qa_toy")
    (finding,) = [f for f in ambiguous_labels(examples) if f.example_id == "qa-008"]
    assert "declared ambiguous" in finding.detail
    assert "no wh-word" in finding.detail
    assert "one instance of a category" in finding.detail


def test_at_most_one_ambiguous_finding_per_example() -> None:
    findings = ambiguous_labels(_corpus())
    assert len(findings) == len(_ids(findings))


def test_ambiguity_ignores_an_empty_prompt() -> None:
    assert ambiguous_labels([_ex(input="", expected_output="x")]) == []


# --------------------------------------------------------------------------- #
# Inconsistent formats
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURES)
def test_inconsistent_formats_match_the_fixture_ground_truth(name: str) -> None:
    examples, defects = _load(name)
    planted = {
        p["id"] for p in _planted_label_issues(defects) if p["kind"] == "inconsistent_format"
    }
    assert _ids(inconsistent_formats(examples)) == planted


def test_the_shape_minority_is_flagged_and_the_reason_is_the_shape() -> None:
    examples, _ = _load("agent_tasks_toy")
    (finding,) = inconsistent_formats(examples)
    assert finding.example_id == "agent-011"
    assert "expected_output is a string" in finding.detail
    assert "10 of 11" in finding.detail and "mapping" in finding.detail


def test_the_vocabulary_outlier_is_flagged_and_the_reason_is_the_vocabulary() -> None:
    examples, _ = _load("safety_toy")
    (finding,) = inconsistent_formats(examples)
    assert finding.example_id == "saf-010"
    assert "'no'" in finding.detail
    assert "['comply', 'refuse']" in finding.detail


def test_per_subtype_key_sets_are_not_a_format_defect() -> None:
    """``agent_tasks_toy``'s coding tasks carry ``{reference_solution, success_criteria}``
    and its research tasks ``{reference_answer}``. That is a legitimate per-subtype schema;
    a key-set check would report the three research tasks as defects. Only shape is checked.
    """
    examples, _ = _load("agent_tasks_toy")
    research = {"agent-005", "agent-006", "agent-012"}
    assert research & _ids(inconsistent_formats(examples)) == set()


def test_the_vocabulary_check_auto_disables_on_a_free_text_family() -> None:
    examples, _ = _load("qa_toy")
    expectations = family_expectations(examples)
    assert expectations["qa"].vocabulary == ()
    assert inconsistent_formats(examples) == []


def test_a_repeated_out_of_vocabulary_answer_is_not_flagged() -> None:
    """A repeated answer outside the vocabulary means the vocabulary is incomplete —
    ``label_vocabulary`` only admits answers of ≤2 tokens — not that the answer is malformed.
    """
    def safety(example_id: str, prompt: str, answer: str) -> DatasetExample:
        return _ex(example_id, task_family="safety", input=prompt, expected_output=answer)

    examples = [
        *[safety(f"c{i}", f"q{i}", "comply") for i in range(4)],
        *[safety(f"r{i}", f"p{i}", "refuse") for i in range(4)],
        *[safety(f"w{i}", f"o{i}", "comply with caveats") for i in range(3)],
        safety("rare", "z", "no"),
    ]
    expectations = family_expectations(examples)
    assert expectations["safety"].vocabulary == ("comply", "refuse")
    assert _ids(inconsistent_formats(examples)) == {"rare"}


def test_a_heterogeneous_family_has_no_dominant_shape() -> None:
    """A 50/50 split across two shapes is heterogeneous by design, not defective."""
    examples = [
        *[_ex(f"s{i}", input=f"q{i}", expected_output="x") for i in range(3)],
        *[_ex(f"m{i}", input=f"p{i}", expected_output={"a": "b"}) for i in range(3)],
    ]
    assert family_expectations(examples)["qa"].dominant_shape is None
    assert inconsistent_formats(examples) == []


def test_a_tiny_family_has_no_dominant_shape() -> None:
    examples = [
        _ex("a", input="q1", expected_output="x"),
        _ex("b", input="q2", expected_output="y"),
        _ex("c", input="q3", expected_output={"a": "b"}),
    ]
    assert family_expectations(examples)["qa"].dominant_shape is None
    assert inconsistent_formats(examples) == []


def test_missing_answers_are_excluded_from_the_shape_census() -> None:
    """``agent-009``'s ``""`` must neither be flagged twice nor skew the census."""
    examples, _ = _load("agent_tasks_toy")
    expectations = family_expectations(examples)
    assert expectations["agent_task"].n_labelled == 11
    assert expectations["agent_task"].shapes == {"mapping": 10, "string": 1}
    assert "agent-009" not in _ids(inconsistent_formats(examples))


def test_at_most_one_format_finding_per_example() -> None:
    findings = inconsistent_formats(_corpus())
    assert len(findings) == len(_ids(findings))


def test_the_format_census_is_computed_per_task_family() -> None:
    """Concatenating the corpus must not make ``agent_task``'s rubrics a global minority."""
    assert _ids(inconsistent_formats(_corpus())) == {"agent-011", "saf-010"}


# --------------------------------------------------------------------------- #
# Adjudication
# --------------------------------------------------------------------------- #


def test_a_tie_blocks_and_a_majority_does_not() -> None:
    tie = _ex("tie", labels={"annotations": [{"annotator": "a1", "value": "x"},
                                             {"annotator": "a2", "value": "y"}]})
    won = _ex("won", labels={"annotations": [{"annotator": "a1", "value": "x"},
                                             {"annotator": "a2", "value": "y"},
                                             {"annotator": "a3", "value": "x"}]})
    report = audit_labels([tie, won])
    blocking = set(report.blocking_example_ids)
    assert blocking == {"tie"}
    (resolved,) = [item for item in report.queue if item.example_id == "won"]
    assert resolved.proposed_resolution == "adopt the majority annotator label 'x' (2/3 annotators)"


def test_a_relevance_conflict_blocks_even_though_a_fix_looks_obvious() -> None:
    """Two authorities, no tiebreak: the flag may be wrong rather than the gold list."""
    report = audit_labels(_load("retrieval_toy")[0])
    (item,) = [i for i in report.queue if i.example_id == "ret-004"]
    assert item.blocking is True
    assert item.proposed_resolution is None


def test_a_majority_cannot_arbitrate_a_conflict_the_annotators_were_not_asked_about() -> None:
    """An annotator majority does not resolve a *shared-prompt* conflict on the same example."""
    anns = [{"annotator": "a1", "value": "Canberra"},
            {"annotator": "a2", "value": "Sydney"},
            {"annotator": "a3", "value": "Canberra"}]
    pair = [
        _ex(
            "a",
            input="Capital of Australia?",
            expected_output="Sydney",
            labels={"annotations": anns},
        ),
        _ex("b", input="Capital of Australia?", expected_output="Canberra"),
    ]
    report = audit_labels(pair)
    assert set(report.blocking_example_ids) == {"a", "b"}


def test_format_only_examples_are_resolvable() -> None:
    report = audit_labels(_corpus())
    resolvable = {item.example_id for item in report.queue if not item.blocking}
    assert resolvable == {"agent-011", "saf-010"}


def test_missing_and_ambiguous_never_propose_a_resolution() -> None:
    report = audit_labels(_corpus())
    for item in report.queue:
        if {"missing", "ambiguous"} & set(item.issues):
            assert item.blocking is True


def test_needs_adjudication_is_exactly_the_blocking_set() -> None:
    report = audit_labels(_corpus())
    assert _ids(report.needs_adjudication) == set(report.blocking_example_ids)
    assert all(f.kind == LabelIssue.NEEDS_ADJUDICATION for f in report.needs_adjudication)


def test_an_example_carries_every_issue_it_earned() -> None:
    """``agent-009`` is planted as both ``missing`` and ``ambiguous``."""
    report = audit_labels(_load("agent_tasks_toy")[0])
    (item,) = [i for i in report.queue if i.example_id == "agent-009"]
    assert item.issues == ["missing", "ambiguous"]  # sorted by severity


def test_queue_is_ordered_blocking_first_then_by_severity_then_by_id() -> None:
    queue = audit_labels(_corpus()).queue
    keys = [(not item.blocking, item.severity, item.example_id) for item in queue]
    assert keys == sorted(keys)


def test_queue_covers_exactly_the_flagged_examples() -> None:
    report = audit_labels(_corpus())
    assert len(report.queue) == report.n_flagged_examples
    corpus_ids = {ex.example_id for ex in _corpus()}
    assert {item.example_id for item in report.queue} <= corpus_ids


def test_queue_carries_the_annotator_values_a_reviewer_needs() -> None:
    report = audit_labels(_load("qa_toy")[0])
    (item,) = [i for i in report.queue if i.example_id == "qa-011"]
    assert item.expected_output == "1945"
    assert item.annotations == [
        {"annotator": "a1", "value": "1945"},
        {"annotator": "a2", "value": "1944"},
    ]


def test_build_queue_ignores_findings_about_unknown_examples() -> None:
    assert build_queue([], missing_labels([_ex(expected_output="")])) == []


def test_queue_export_is_deterministic_and_diff_friendly(tmp_path) -> None:
    report = audit_labels(_corpus())
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert write_queue(report.queue, first) == len(report.queue)
    write_queue(audit_labels(_corpus()).queue, second)
    assert first.read_bytes() == second.read_bytes()

    lines = first.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(report.queue)
    assert [json.loads(line)["example_id"] for line in lines] == [
        item.example_id for item in report.queue
    ]


def test_queue_items_round_trip_through_json() -> None:
    for item in audit_labels(_corpus()).queue:
        restored = json.loads(json.dumps(item.to_dict()))
        assert restored["example_id"] == item.example_id
        assert restored["blocking"] == item.blocking


def test_needs_adjudication_of_an_empty_queue() -> None:
    assert needs_adjudication([]) == []


def test_adjudication_item_severity_of_no_issues() -> None:
    assert AdjudicationItem("x", "qa", "train").severity == 4


# --------------------------------------------------------------------------- #
# The blind spot
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURES)
def test_mislabeled_examples_are_invisible_to_every_check(name: str) -> None:
    """The most damaging label defect is the one this stage cannot see.

    ``qa-006`` (gold "Ag" for the symbol of gold), ``bio-009`` (23 chromosomes in a human
    somatic cell), and ``saf-006`` (a benign vaccine question labelled "refuse") are
    well-formed, unanimous, unambiguous, and correctly shaped. ``defects.yaml`` marks all
    three ``auto_detectable: false``. This pins the limitation so it cannot rot into a
    silent claim of coverage.
    """
    examples, defects = _load(name)
    mislabeled = {
        p["id"] for p in _planted_label_issues(defects) if not p.get("auto_detectable", True)
    }
    assert mislabeled & _ids(audit_labels(examples).findings) == set()


def test_the_corpus_has_three_such_defects_and_none_are_found() -> None:
    mislabeled = set()
    for name in FIXTURES:
        _, defects = _load(name)
        mislabeled |= {
            p["id"] for p in _planted_label_issues(defects) if not p.get("auto_detectable", True)
        }
    assert mislabeled == {"qa-006", "bio-009", "saf-006"}
    assert mislabeled & _ids(audit_labels(_corpus()).findings) == set()


# --------------------------------------------------------------------------- #
# Contract properties
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("check", "kind"),
    [
        (missing_labels, LabelIssue.MISSING),
        (annotator_conflicts, LabelIssue.CONFLICTING),
        (shared_prompt_conflicts, LabelIssue.CONFLICTING),
        (relevance_conflicts, LabelIssue.CONFLICTING),
        (ambiguous_labels, LabelIssue.AMBIGUOUS),
        (inconsistent_formats, LabelIssue.INCONSISTENT_FORMAT),
    ],
)
def test_each_check_emits_only_its_own_kind(check, kind) -> None:
    corpus = [
        *_corpus(),
        # a synthetic shared-prompt conflict, the one source the fixtures lack
        _ex("syn-a", input="Capital of Australia?", expected_output="Sydney"),
        _ex("syn-b", input="Capital of Australia?", expected_output="Canberra"),
    ]
    findings = check(corpus)
    assert findings, f"{check.__name__} found nothing to check the kind of"
    assert {f.kind for f in findings} == {kind}


def test_every_label_issue_kind_has_a_producer() -> None:
    """No kind may be declared in the schema without a check that can emit it."""
    corpus = [
        *_corpus(),
        _ex("syn-a", input="Capital of Australia?", expected_output="Sydney"),
        _ex("syn-b", input="Capital of Australia?", expected_output="Canberra"),
    ]
    emitted = {finding.kind for finding in audit_labels(corpus).findings}
    assert emitted == set(LabelIssue)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_report_counts_are_internally_consistent() -> None:
    report = audit_labels(_corpus())
    counts = report.counts()
    assert counts["examples"] == len(_corpus())
    assert len(report.findings) == (
        counts["missing"]
        + counts["conflicting"]
        + counts["ambiguous"]
        + counts["inconsistent_format"]
        + counts["needs_adjudication"]
    )
    assert counts["queue"] == counts["flagged_examples"]
    assert counts["needs_adjudication"] <= counts["queue"]


def test_report_names_only_examples_that_exist() -> None:
    corpus = _corpus()
    ids = {ex.example_id for ex in corpus}
    assert _ids(audit_labels(corpus).findings) <= ids


def test_report_is_deterministic() -> None:
    first, second = audit_labels(_corpus()), audit_labels(_corpus())
    assert first.to_dict() == second.to_dict()


def test_report_round_trips_through_json() -> None:
    payload = json.dumps(audit_labels(_corpus()).to_dict())
    assert json.loads(payload)["counts"]["examples"] == len(_corpus())


def test_report_records_the_parameters_it_actually_used() -> None:
    report = audit_labels(_corpus(), max_underspecified_tokens=6, heuristic_ambiguity=False)
    assert report.to_dict()["params"] == {
        "max_underspecified_tokens": 6,
        "heuristic_ambiguity": False,
    }
    assert set(report.expectations) == {"qa", "agent_task", "retrieval", "bio", "safety"}


def test_report_of_an_empty_dataset() -> None:
    report = audit_labels([])
    assert report.counts()["examples"] == 0
    assert report.findings == []
    assert report.agreement.rate is None


def test_format_report_mentions_every_section_and_the_guard() -> None:
    text = format_report(audit_labels(_corpus()))
    for expected in (
        "missing labels",
        "conflicting labels",
        "ambiguous labels",
        "inconsistent formats",
        "adjudication queue",
        "annotator agreement : 0%",
        "vocabulary ['comply', 'refuse']",
    ):
        assert expected in text


def test_format_report_of_a_clean_dataset_lists_no_sections() -> None:
    text = format_report(audit_labels([_ex(input="What is the capital of France?")]))
    assert "missing labels" not in text
    assert "n/a (no example carries" in text


def test_format_report_truncates_long_sections() -> None:
    examples = [_ex(f"e{i}", input=f"question {i}?", expected_output="") for i in range(30)]
    text = format_report(audit_labels(examples), max_items=5)
    assert "… and 25 more" in text


def test_report_type_is_exported() -> None:
    assert isinstance(audit_labels([]), LabelAuditReport)
