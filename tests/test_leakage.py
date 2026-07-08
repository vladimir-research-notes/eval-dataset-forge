"""Tests for the leakage & contamination layer (Milestone 5).

Covered here:

  * **Primitives** — asymmetric n-gram containment, the short-needle width clamp,
    the longest-shared-run evidence extractor, and the memoized reference index.
  * **Answer-in-prompt** — the heuristic one. Scored against every fixture's
    ``defects.yaml``: it must find both planted leaks and invent none. The two
    rejected n-gram widths are pinned by tests that assert *why* they were rejected
    (n=1 admits four false positives, n=3 loses a true one), and the default
    threshold is shown to sit on a plateau rather than a knife edge.
  * **Label-vocabulary guard** — demonstrated on a synthetic classification set
    where, without it, every single example would be flagged.
  * **Train/test overlap** — recall against planted ground truth for both dedup
    methods, plus the inherited blind spot and the inherited false positive.
  * **N-gram overlap** — the four verbatim cross-split duplicates, why n=8 rather
    than n=5, and the self-match skip.
  * **Retrieval contamination** — the artifact marker fires exactly once across the
    whole committed corpus; shared *distractors* are not flagged; the paraphrased
    duplicate passages are pinned as undetected.
  * **Corroboration** — ``ret-001``/``ret-002`` is unreachable by the dedup ladder
    (proven in Milestone 4) yet recovered by the retrieval channel. That is the
    argument for keeping the channels independent, so it gets a regression test.
  * **Report** — count consistency, determinism, JSON round-trip, and the guarantee
    that every reported id exists in the dataset.

Everything is CPU-only and offline. Tests that need the optional TF-IDF path skip
cleanly when scikit-learn is absent.
"""

from __future__ import annotations

import json

import pytest

from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup import find_duplicates
from eval_dataset_forge.leakage import (
    LeakageReport,
    ReferenceIndex,
    containment,
    cross_split_clusters,
    cross_split_documents,
    cross_split_sources,
    document_clusters,
    find_leakage,
    format_report,
    label_artifacts,
    label_vocabulary,
    leaked_answers,
    longest_shared_run,
    reference_overlap,
    retrieval_contamination,
    retrieval_docs,
    split_reference,
    split_reference_overlap,
)
from eval_dataset_forge.leakage.answer_in_prompt import answer_candidates
from eval_dataset_forge.leakage.retrieval_overlap import ARTIFACT_MARKER
from eval_dataset_forge.schemas import DatasetExample, LeakageKind, Split

FIXTURES = ["qa_toy", "agent_tasks_toy", "retrieval_toy", "bio_toy", "safety_toy"]


def _load(name: str) -> tuple[list[DatasetExample], dict]:
    fixture_dir = loader.FIXTURES_ROOT / name
    return (
        loader.load_examples(fixture_dir / "examples.jsonl"),
        loader.load_defects(fixture_dir / "defects.yaml"),
    )


# Loaded once at import; the data is tiny and fully offline.
_ALL: dict[str, tuple[list[DatasetExample], dict]] = {n: _load(n) for n in FIXTURES}


def _planted(name: str, kind: str) -> list[list[str]]:
    """Ground-truth id groups of one leakage kind from a fixture's ``defects.yaml``."""
    entries = _ALL[name][1]["planted_defects"].get("leakage", [])
    return [sorted(entry["ids"]) for entry in entries if entry["kind"] == kind]


def _covers(findings, group: list[str]) -> bool:
    """Whether one reported finding names every member of a planted group."""
    return any(set(group) <= set(f.example_ids) for f in findings)


def _flagged(findings) -> set[str]:
    return {eid for f in findings for eid in f.example_ids}


def _example(**kwargs) -> DatasetExample:
    """A minimal valid example; override any field via kwargs."""
    base = {
        "example_id": "e1",
        "source_id": "s1",
        "task_family": "qa",
        "input": "What is the capital of France?",
        "expected_output": "Paris",
    }
    return DatasetExample(**{**base, **kwargs})


def _doc(doc_id: str, text: str, relevant: bool) -> dict:
    return {"doc_id": doc_id, "text": text, "relevant": relevant}


def _retrieval_example(example_id: str, split: str, docs: list[dict], **kwargs) -> DatasetExample:
    return _example(
        example_id=example_id,
        source_id=f"src-{example_id}",
        task_family="retrieval",
        input=f"query for {example_id}",
        expected_output=[d["doc_id"] for d in docs if d["relevant"]],
        metadata={"retrieval": {"docs": docs}},
        split=split,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Primitives: containment
# --------------------------------------------------------------------------- #


def test_containment_is_asymmetric() -> None:
    """It measures how much of the *needle* is in the haystack, not their similarity."""
    needle, haystack = ["berlin"], ["the", "answer", "is", "berlin", "in", "germany"]
    assert containment(needle, haystack, 1) == (1.0, 1)
    # The reverse direction is almost nothing: the haystack is mostly not the needle.
    score, _width = containment(haystack, needle, 1)
    assert score == pytest.approx(1 / 6)


def test_containment_clamps_the_width_to_the_needle() -> None:
    """A 2-token answer checked at n=8 is compared as a bigram, not silently missed."""
    score, width = containment(["red", "planet"], ["mars", "the", "red", "planet"], 8)
    assert (score, width) == (1.0, 2)


def test_containment_of_an_empty_needle_is_zero_not_one() -> None:
    """An empty gold answer is 'contained' in every prompt; that must not be a leak."""
    assert containment([], ["anything"], 2) == (0.0, 0)


def test_containment_partial_overlap() -> None:
    # bigrams of the needle: (a,b) (b,c) (c,d); the haystack holds only (a,b).
    score, width = containment(list("abcd"), list("xaby"), 2)
    assert (score, width) == (pytest.approx(1 / 3), 2)


def test_containment_requires_contiguity_at_width_two() -> None:
    """Both tokens present but not adjacent: unigrams would say 1.0, bigrams say 0.0."""
    assert containment(["red", "planet"], ["red", "hot", "planet"], 2)[0] == 0.0
    assert containment(["red", "planet"], ["red", "hot", "planet"], 1)[0] == 1.0


@pytest.mark.parametrize(
    "needle, haystack, expected",
    [
        (["a", "b", "c"], ["x", "a", "b", "c", "y"], ["a", "b", "c"]),
        (["a", "b", "c"], ["b", "c", "z"], ["b", "c"]),
        (["a"], ["b"], []),
        ([], ["a"], []),
        (["a"], [], []),
    ],
)
def test_longest_shared_run(needle, haystack, expected) -> None:
    assert longest_shared_run(needle, haystack) == expected


def test_reference_index_rejects_mismatched_ids() -> None:
    with pytest.raises(ValueError, match="reference ids"):
        ReferenceIndex(["a", "b"], ids=["only-one"])


def test_reference_index_memoizes_per_width() -> None:
    index = ReferenceIndex(["the quick brown fox"])
    assert index.grams_at(2) is index.grams_at(2)
    assert index.grams_at(2) != index.grams_at(3)
    assert len(index) == 1


# --------------------------------------------------------------------------- #
# Answer candidates
# --------------------------------------------------------------------------- #


def test_answer_candidates_unpacks_a_rubric_dict() -> None:
    """The leak lives in one field of a dict; a stringified blob would never match."""
    assert answer_candidates({"success_criteria": "runs", "reference_solution": "return s"}) == [
        ("reference_solution", "return s"),
        ("success_criteria", "runs"),
    ]


def test_answer_candidates_names_the_root_field() -> None:
    assert answer_candidates("Paris") == [("expected_output", "Paris")]


def test_answer_candidates_indexes_lists_and_nests() -> None:
    assert answer_candidates(["d1", {"note": "x"}]) == [("[0]", "d1"), ("[1].note", "x")]


def test_answer_candidates_renders_scalars_and_bools() -> None:
    assert answer_candidates(4) == [("expected_output", "4")]
    assert answer_candidates(True) == [("expected_output", "true")]


def test_answer_candidates_drops_empty_leaves() -> None:
    """An absent gold answer is a label defect, and '' is contained in every prompt."""
    assert answer_candidates("") == []
    assert answer_candidates("   ") == []
    assert answer_candidates({"a": "", "b": "x"}) == [("b", "x")]


# --------------------------------------------------------------------------- #
# The label-vocabulary guard
# --------------------------------------------------------------------------- #


def _classification_dataset() -> list[DatasetExample]:
    """A sentiment set whose prompts name their own label vocabulary."""
    return [
        _example(
            example_id=f"cls-{i:02d}",
            input=f"Is review #{i} positive or negative?",
            expected_output="positive" if i % 2 else "negative",
        )
        for i in range(8)
    ]


def test_label_vocabulary_fires_on_a_classification_dataset() -> None:
    examples, _ = _ALL["safety_toy"]
    assert label_vocabulary(examples) == frozenset({"comply", "refuse"})


def test_label_vocabulary_ignores_a_merely_repeated_answer() -> None:
    """'William Shakespeare' recurs 3x but covers 21% of qa_toy: an answer, not a label."""
    examples, _ = _ALL["qa_toy"]
    assert label_vocabulary(examples) == frozenset()


def test_label_vocabulary_needs_both_repetition_and_coverage() -> None:
    examples = _classification_dataset()
    assert label_vocabulary(examples) == frozenset({"positive", "negative"})
    # Coverage is the load-bearing condition: demand more than the data can supply
    # and the guard correctly refuses to classify the dataset.
    assert label_vocabulary(examples, min_coverage=1.01) == frozenset()
    assert label_vocabulary(examples, min_repeats=5) == frozenset()


def test_label_vocabulary_ignores_structured_answers() -> None:
    """A rubric dict or a doc-id list is never a class label, however often it repeats."""
    assert label_vocabulary([_example(expected_output={"a": 1}) for _ in range(9)]) == frozenset()


def test_label_vocabulary_of_an_empty_dataset() -> None:
    assert label_vocabulary([]) == frozenset()


def test_label_guard_suppresses_a_whole_classification_dataset() -> None:
    """Without the guard every row leaks; with it, none does. This is why it exists."""
    examples = _classification_dataset()
    ungated = leaked_answers(examples, ignore_answers=frozenset())
    assert _flagged(ungated) == {ex.example_id for ex in examples}
    assert leaked_answers(examples) == []


# --------------------------------------------------------------------------- #
# Answer-in-prompt — scored against ground truth
# --------------------------------------------------------------------------- #

# Every planted answer_in_prompt defect, across all five fixtures.
PLANTED_ANSWER_LEAKS = {"qa-007", "agent-004"}


def _all_answer_leaks(**kwargs) -> set[str]:
    """Ids flagged by the answer check across every fixture, each gated on its own."""
    return {
        eid for name in FIXTURES for eid in _flagged(leaked_answers(_ALL[name][0], **kwargs))
    }


def test_answer_in_prompt_recovers_every_planted_leak_with_no_false_positives() -> None:
    assert _all_answer_leaks() == PLANTED_ANSWER_LEAKS


@pytest.mark.parametrize("name", FIXTURES)
def test_answer_in_prompt_matches_the_fixture_ground_truth(name: str) -> None:
    findings = leaked_answers(_ALL[name][0])
    planted = _planted(name, "answer_in_prompt")
    assert all(_covers(findings, group) for group in planted)
    assert len(findings) == len(planted)


def test_unigrams_admit_false_positives() -> None:
    """n=1 cannot tell a quoted answer from a prompt that shares its vocabulary.

    ``agent-011``'s answer "return sorted(arr)" scores 0.667 against a prompt that
    merely *describes* returning a sorted arr; ``qa-013``'s "Pacific Ocean" scores
    0.5 because the prompt contains the word *ocean*. This is why DEFAULT_NGRAM is 2.
    """
    flagged = _all_answer_leaks(ngram=1)
    assert flagged - PLANTED_ANSWER_LEAKS == {"qa-013", "agent-007", "agent-008", "agent-011"}


def test_trigrams_lose_a_planted_leak() -> None:
    """A paraphrased quotation keeps its bigrams long after its trigrams are gone."""
    assert _all_answer_leaks(ngram=3) == {"qa-007"}


def test_the_default_threshold_sits_on_a_plateau() -> None:
    """0.5 is not fitted to the data: the whole band [0.4, 0.6] gives the same answer.

    The weakest planted leak scores 0.600 and the strongest non-leak 0.333, so the
    default has room on both sides. It only breaks down above 0.6.
    """
    for threshold in (0.4, 0.5, 0.6):
        assert _all_answer_leaks(threshold=threshold) == PLANTED_ANSWER_LEAKS
    assert _all_answer_leaks(threshold=0.7) == {"qa-007"}


def test_answer_finding_shape_and_evidence() -> None:
    examples, _ = _ALL["agent_tasks_toy"]
    (finding,) = leaked_answers(examples)
    assert finding.kind == LeakageKind.ANSWER_IN_PROMPT
    assert finding.example_ids == ["agent-004"]
    assert finding.score == pytest.approx(0.6)
    # The evidence names the leaking field and shows what leaked, not just a number.
    assert "reference_solution" in finding.evidence
    assert "return s 1" in finding.evidence


def test_short_answers_are_checked_at_their_own_width() -> None:
    """"Berlin" is one token; a bigram check must not silently skip it."""
    examples, _ = _ALL["qa_toy"]
    (finding,) = leaked_answers(examples)
    assert finding.example_ids == ["qa-007"]
    assert finding.score == 1.0
    assert "n=1" in finding.evidence


def test_answer_check_reports_at_most_one_finding_per_example() -> None:
    """agent-004 leaks through reference_solution; success_criteria must not double-report."""
    leaky = _example(
        input="Solve it. Hint: the answer is return s[::-1] and it should reverse a string",
        expected_output={"reference_solution": "return s[::-1]", "note": "reverse a string"},
    )
    (finding,) = leaked_answers([leaky], ignore_answers=frozenset())
    assert finding.example_ids == ["e1"]


def test_answer_check_skips_empty_prompts_and_answers() -> None:
    assert leaked_answers([_example(input="", expected_output="x")]) == []
    assert leaked_answers([_example(input="anything", expected_output="")]) == []
    assert leaked_answers([]) == []


def test_answer_check_does_not_fire_on_retrieval_doc_ids() -> None:
    """Doc-id gold answers are opaque tokens; they are never quoted in a query."""
    examples, _ = _ALL["retrieval_toy"]
    assert leaked_answers(examples) == []


# --------------------------------------------------------------------------- #
# Train/test overlap
# --------------------------------------------------------------------------- #

# Planted cross-split pairs each dedup method actually recovers. Measured, not
# aspirational; ``retrieval_toy``'s pair is unreachable by any lexical threshold
# (see docs/DEDUPLICATION.md and test_corroboration below).
OVERLAP_RECALL: dict[str, dict[str, int]] = {
    "tfidf": {
        "qa_toy": 2,
        "agent_tasks_toy": 2,
        "retrieval_toy": 0,
        "bio_toy": 2,
        "safety_toy": 2,
    },
    "jaccard": {
        "qa_toy": 2,
        "agent_tasks_toy": 2,
        "retrieval_toy": 0,
        "bio_toy": 1,  # the paraphrase needs idf weighting; the exact pair is found
        "safety_toy": 2,
    },
}


def _overlap(name: str, method: str = "auto"):
    examples = _ALL[name][0]
    return cross_split_clusters(examples, find_duplicates(examples, near_method=method))


@pytest.mark.parametrize("method", ["jaccard", "tfidf"])
@pytest.mark.parametrize("name", FIXTURES)
def test_train_test_overlap_recall_is_as_documented(name: str, method: str) -> None:
    if method == "tfidf":
        pytest.importorskip("sklearn")
    findings = _overlap(name, method)
    planted = _planted(name, "train_test_overlap")
    covered = sum(_covers(findings, group) for group in planted)
    assert covered == OVERLAP_RECALL[method][name]


def test_train_test_overlap_totals_across_the_corpus() -> None:
    pytest.importorskip("sklearn")
    planted = sum(len(_planted(name, "train_test_overlap")) for name in FIXTURES)
    covered = sum(
        _covers(_overlap(name, "tfidf"), group)
        for name in FIXTURES
        for group in _planted(name, "train_test_overlap")
    )
    assert (covered, planted) == (8, 9)


def test_overlap_inherits_the_dedup_false_positive() -> None:
    """agent-003/agent-004 is not planted as overlap, yet the pair really does straddle
    train and test with an identical reference solution. The dedup layer scores it as
    a false positive against ``defects.yaml``; as *leakage* it is arguably a true one.
    Recorded here so the disagreement stays visible rather than being quietly dropped.
    """
    findings = _overlap("agent_tasks_toy")
    assert _covers(findings, ["agent-003", "agent-004"])
    planted = _planted("agent_tasks_toy", "train_test_overlap")
    assert not any(set(g) == {"agent-003", "agent-004"} for g in planted)


def test_overlap_finding_shape() -> None:
    findings = _overlap("bio_toy")
    exact = next(f for f in findings if set(f.example_ids) == {"bio-003", "bio-006"})
    assert exact.kind == LeakageKind.TRAIN_TEST_OVERLAP
    # Identical by construction, so no similarity — exactly as the dedup stage reports.
    assert exact.score is None
    assert "exact duplicate spans 2 splits" in exact.evidence
    assert "train=['bio-003']" in exact.evidence and "test=['bio-006']" in exact.evidence


def test_overlap_carries_the_near_similarity_through() -> None:
    findings = _overlap("safety_toy", "jaccard")
    near = next(f for f in findings if set(f.example_ids) == {"saf-001", "saf-003"})
    assert near.score == pytest.approx(0.3125)
    assert "near-duplicate" in near.evidence


def test_overlap_ignores_unassigned_examples() -> None:
    """A duplicate cannot leak across a split boundary that has not been drawn yet."""
    examples = [_example(example_id="a"), _example(example_id="b")]
    assert cross_split_clusters(examples, find_duplicates(examples)) == []


def test_overlap_counts_dev_test_as_a_boundary() -> None:
    """Tuning on an item then scoring on it is the same defect as train/test overlap."""
    examples = [
        _example(example_id="a", split=Split.DEV),
        _example(example_id="b", split=Split.TEST),
    ]
    (finding,) = cross_split_clusters(examples, find_duplicates(examples))
    assert finding.example_ids == ["a", "b"]
    assert "dev=['a']" in finding.evidence and "test=['b']" in finding.evidence


def test_overlap_keeps_unassigned_cluster_members_in_the_finding() -> None:
    """They are part of the duplicate group, even though they are not what leaks."""
    examples = [
        _example(example_id="a", split=Split.TRAIN),
        _example(example_id="b", split=Split.TEST),
        _example(example_id="c"),  # unassigned
    ]
    (finding,) = cross_split_clusters(examples, find_duplicates(examples))
    assert finding.example_ids == ["a", "b", "c"]
    assert "unassigned" not in finding.evidence


def test_cross_split_sources_are_dependence_not_duplication() -> None:
    """Both planted duplicate-source records straddle a split boundary."""
    examples, _ = _ALL["qa_toy"]
    spread = cross_split_sources(examples, find_duplicates(examples))
    assert spread == {"src-shakespeare": {"train": ["qa-003"], "test": ["qa-012"]}}

    examples, _ = _ALL["retrieval_toy"]
    spread = cross_split_sources(examples, find_duplicates(examples))
    assert spread == {"src-boiling": {"test": ["ret-006"], "dev": ["ret-009"]}}


def test_cross_split_sources_ignores_same_split_reuse() -> None:
    examples = [
        _example(example_id="a", source_id="s", input="one", split=Split.TRAIN),
        _example(example_id="b", source_id="s", input="two", split=Split.TRAIN),
    ]
    assert cross_split_sources(examples, find_duplicates(examples)) == {}


# --------------------------------------------------------------------------- #
# N-gram overlap against a reference corpus
# --------------------------------------------------------------------------- #


def test_split_ngram_overlap_finds_the_verbatim_cross_split_duplicates() -> None:
    """At n=8 the reference channel recovers exactly the four identical pairs."""
    flagged = {
        eid for name in FIXTURES for eid in _flagged(split_reference_overlap(_ALL[name][0]))
    }
    assert flagged == {"qa-002", "agent-006", "bio-006", "saf-005"}


def test_ngram_overlap_adds_nothing_beyond_the_exact_overlap_channel() -> None:
    """An honest null result: on these fixtures the n-gram channel is redundant.

    Every planted cross-split pair is either identical (caught by the exact dedup
    stage as well) or a paraphrase (invisible to 8-grams). The channel earns its
    place on datasets with an *external* reference corpus, not here.
    """
    for name in FIXTURES:
        examples = _ALL[name][0]
        ngram_ids = _flagged(split_reference_overlap(examples))
        exact_ids = _flagged(
            f
            for f in cross_split_clusters(examples, find_duplicates(examples))
            if f.score is None and "exact" in f.evidence
        )
        assert ngram_ids <= exact_ids


def test_width_eight_rejects_the_boilerplate_that_width_five_admits() -> None:
    """At n=5, "what is the capital of" and "chocolate chip cookies at home" flag.

    Four spurious findings appear, each a shared template rather than a shared item.
    This is the measurement behind DEFAULT_NGRAM = 8.
    """
    flagged = {
        eid
        for name in FIXTURES
        for eid in _flagged(split_reference_overlap(_ALL[name][0], ngram=5))
    }
    assert flagged - {"qa-002", "agent-006", "bio-006", "saf-005"} == {
        "qa-004",
        "qa-007",
        "agent-004",
        "saf-003",
    }


def test_ngram_finding_records_the_source_and_the_clamp() -> None:
    examples, _ = _ALL["qa_toy"]
    (finding,) = split_reference_overlap(examples)
    assert finding.kind == LeakageKind.NGRAM_OVERLAP
    # The finding is about the contaminated item; the reference is named in evidence.
    assert finding.example_ids == ["qa-002"]
    assert finding.score == 1.0
    assert "shared with qa-001" in finding.evidence
    # qa-002's prompt is 6 tokens, so the 8-gram check ran at width 6 — and says so.
    assert "clamped" in finding.evidence and "6-grams" in finding.evidence


def test_long_prompts_are_not_clamped() -> None:
    examples, _ = _ALL["bio_toy"]
    (finding,) = split_reference_overlap(examples)
    assert "8-grams" in finding.evidence and "clamped" not in finding.evidence


def test_min_containment_suppresses_partial_matches() -> None:
    examples, _ = _ALL["qa_toy"]
    assert split_reference_overlap(examples, ngram=5, min_containment=0.9) != []
    partial = split_reference_overlap(examples, ngram=5)
    assert {f.example_ids[0] for f in partial if f.score < 0.9} == {"qa-004", "qa-007"}


def test_reference_overlap_skips_an_examples_own_entry() -> None:
    """A dataset used as its own reference must report repetition, not self-identity."""
    examples, _ = _ALL["qa_toy"]
    reference = ReferenceIndex([e.input for e in examples], [e.example_id for e in examples])
    flagged = _flagged(reference_overlap(examples, reference))
    assert flagged == {"qa-001", "qa-002"}


def test_reference_overlap_accepts_an_external_corpus() -> None:
    """The reference need not be a dataset: ids degrade to positional labels."""
    example = _example(input="the mitochondria is the powerhouse of the cell")
    reference = ReferenceIndex(["everyone knows the mitochondria is the powerhouse of the cell"])
    (finding,) = reference_overlap([example], reference)
    assert finding.score == 1.0
    assert "reference[0]" in finding.evidence


def test_reference_overlap_on_an_empty_corpus_and_empty_prompt() -> None:
    assert reference_overlap([_example()], ReferenceIndex([])) == []
    assert reference_overlap([_example(input="")], ReferenceIndex(["anything"])) == []


def test_split_reference_selects_the_named_split() -> None:
    examples, _ = _ALL["qa_toy"]
    assert len(split_reference(examples, Split.TRAIN)) == 8
    assert len(split_reference(examples, Split.TEST)) == 5


# --------------------------------------------------------------------------- #
# Retrieval contamination
# --------------------------------------------------------------------------- #


def test_artifact_marker_fires_exactly_once_across_the_whole_corpus() -> None:
    """A broad pattern would fire on prose. This one matches only the planted artifact."""
    matches = [
        (ex.example_id, doc["doc_id"])
        for name in FIXTURES
        for ex in _ALL[name][0]
        for doc in retrieval_docs(ex)
        if ARTIFACT_MARKER.search(doc.get("text", ""))
    ]
    assert matches == [("ret-003", "d5")]
    # ...and never on a prompt.
    assert not [
        ex.example_id
        for name in FIXTURES
        for ex in _ALL[name][0]
        if ARTIFACT_MARKER.search(ex.input)
    ]


def test_label_artifact_finding_matches_the_planted_ground_truth() -> None:
    examples, defects = _ALL["retrieval_toy"]
    (finding,) = label_artifacts(examples)
    assert finding.kind == LeakageKind.RETRIEVAL_CONTAMINATION
    assert finding.example_ids == ["ret-003"]
    assert "ANSWER:" in finding.evidence and "d5" in finding.evidence

    planted = next(
        e
        for e in defects["planted_defects"]["leakage"]
        if e["kind"] == "retrieval_contamination"
    )
    assert planted["ids"] == ["ret-003"] and planted["doc_ids"] == ["d5"]


@pytest.mark.parametrize(
    "text",
    ["[ANSWER: Mars]", "label = positive", "Ground Truth: yes", "gold_answer: 4", "relevant=true"],
)
def test_artifact_marker_shapes(text: str) -> None:
    assert ARTIFACT_MARKER.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "Mars is often called the Red Planet.",
        "The answer depends on the context.",  # no ':' or '=' after the label word
        "Correctness is relevant to this passage.",
    ],
)
def test_artifact_marker_leaves_prose_alone(text: str) -> None:
    assert not ARTIFACT_MARKER.search(text)


def test_cross_split_gold_passage_is_contamination() -> None:
    examples, _ = _ALL["retrieval_toy"]
    (finding,) = cross_split_documents(examples)
    assert finding.example_ids == ["ret-001", "ret-002"]
    assert "ret-001/d1[train]" in finding.evidence
    assert "ret-002/d3[test]" in finding.evidence


def test_a_shared_distractor_is_not_contamination() -> None:
    """Reusing an irrelevant passage across splits is ordinary corpus construction."""
    shared = "Venus is the second planet from the Sun."
    examples = [
        _retrieval_example("r1", Split.TRAIN, [_doc("d1", "Mars is red.", True),
                                               _doc("d2", shared, False)]),
        _retrieval_example("r2", Split.TEST, [_doc("d3", "Jupiter is large.", True),
                                              _doc("d4", shared, False)]),
    ]
    assert cross_split_documents(examples) == []
    # ...but it is still reported on the informational duplicate-document channel.
    assert [[str(r) for r in c] for c in document_clusters(examples)] == [
        ["r1/d2[train]", "r2/d4[test]"]
    ]


def test_a_gold_passage_reused_inside_one_split_is_not_flagged() -> None:
    passage = "Mars is often called the Red Planet."
    examples = [
        _retrieval_example("r1", Split.TRAIN, [_doc("d1", passage, True)]),
        _retrieval_example("r2", Split.TRAIN, [_doc("d2", passage, True)]),
    ]
    assert cross_split_documents(examples) == []


def test_document_matching_normalizes_case_and_punctuation() -> None:
    examples = [
        _retrieval_example("r1", Split.TRAIN, [_doc("d1", "Mars is the Red Planet.", True)]),
        _retrieval_example("r2", Split.TEST, [_doc("d2", "mars is the red planet", True)]),
    ]
    assert len(cross_split_documents(examples)) == 1


def test_paraphrased_passages_are_not_detected() -> None:
    """A documented limitation, pinned. ret-006's d13/d14 state one fact two ways.

    Passage matching is exact-after-normalization. No threshold here separates these
    two from unrelated passages without semantics, so they are not reported — and the
    fixture's ``paraphrased_documents`` ground truth stands as a known miss.
    """
    examples, defects = _ALL["retrieval_toy"]
    clusters = [[str(ref) for ref in cluster] for cluster in document_clusters(examples)]
    assert clusters == [["ret-001/d1[train]", "ret-002/d3[test]"]]
    planted = defects["planted_defects"]["paraphrased_documents"][0]
    assert planted["doc_ids"] == ["d13", "d14"]


def test_relevance_falls_back_to_the_expected_output_doc_ids() -> None:
    """A passage without a ``relevant`` flag is gold iff its id is the gold answer."""
    examples = [
        _example(
            example_id="r1",
            task_family="retrieval",
            input="q1",
            expected_output=["d1"],
            metadata={"retrieval": {"docs": [{"doc_id": "d1", "text": "shared text here"}]}},
            split=Split.TRAIN,
        ),
        _example(
            example_id="r2",
            task_family="retrieval",
            input="q2",
            expected_output=["d2"],
            metadata={"retrieval": {"docs": [{"doc_id": "d2", "text": "shared text here"}]}},
            split=Split.TEST,
        ),
    ]
    assert len(cross_split_documents(examples)) == 1


@pytest.mark.parametrize(
    "metadata",
    [{}, {"retrieval": "nonsense"}, {"retrieval": {}}, {"retrieval": {"docs": "nope"}},
     {"retrieval": {"docs": ["not-a-dict"]}}],
)
def test_retrieval_docs_is_defensive_about_metadata_shape(metadata: dict) -> None:
    """``metadata`` is free-form by contract; a malformed block yields no passages."""
    assert retrieval_docs(_example(metadata=metadata)) == []


def test_non_retrieval_fixtures_carry_no_contamination() -> None:
    for name in ("qa_toy", "agent_tasks_toy", "bio_toy", "safety_toy"):
        assert retrieval_contamination(_ALL[name][0]) == []


# --------------------------------------------------------------------------- #
# Corroboration between independent channels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["jaccard", "tfidf"])
def test_retrieval_channel_recovers_what_the_dedup_ladder_cannot(method: str) -> None:
    """``ret-001``/``ret-002`` is unreachable by any lexical prompt threshold.

    Milestone 4 proved it: a genuine non-duplicate in the same fixture scores higher.
    The queries are a deep paraphrase, so the query-level overlap check misses the
    planted train/test overlap entirely. The *passage-level* check recovers the same
    pair, because both examples cite the identical gold passage under different ids.

    This is the argument for keeping the four channels independent and non-disjoint.
    If this ever regresses, the leakage recall table in docs/LEAKAGE.md is wrong.
    """
    if method == "tfidf":
        pytest.importorskip("sklearn")
    examples, _ = _ALL["retrieval_toy"]
    planted = _planted("retrieval_toy", "train_test_overlap")[0]

    overlap = cross_split_clusters(examples, find_duplicates(examples, near_method=method))
    assert not _covers(overlap, planted)

    assert _covers(retrieval_contamination(examples), planted)


def test_every_planted_leakage_defect_is_surfaced_by_some_channel() -> None:
    """Per-kind recall is 11/12; per-defect coverage across channels is 12/12."""
    pytest.importorskip("sklearn")
    total = covered = 0
    for name in FIXTURES:
        report = find_leakage(_ALL[name][0])
        for entry in _ALL[name][1]["planted_defects"].get("leakage", []):
            total += 1
            covered += _covers(report.findings, sorted(entry["ids"]))
    assert (covered, total) == (12, 12)


# --------------------------------------------------------------------------- #
# LeakageReport
# --------------------------------------------------------------------------- #


def test_report_counts_are_consistent() -> None:
    pytest.importorskip("sklearn")
    report = find_leakage(_ALL["qa_toy"][0])
    counts = report.counts()
    assert counts["examples"] == 14
    assert counts["train_test_overlap"] == 2
    assert counts["answer_in_prompt"] == 1
    assert counts["ngram_overlap"] == 1
    assert counts["retrieval_contamination"] == 0
    assert counts["cross_split_sources"] == 1
    # qa-001,002 + qa-003,004,012 + qa-007; qa-002 is named twice, counted once.
    assert counts["leaking_examples"] == 6
    assert len(report.findings) == 4


def test_channels_overlap_by_design() -> None:
    """Unlike the dedup ladder, the leakage checks are not a partition."""
    pytest.importorskip("sklearn")
    report = find_leakage(_ALL["bio_toy"][0])
    overlap_ids = _flagged(report.train_test_overlap)
    ngram_ids = _flagged(report.ngram_overlap)
    assert ngram_ids and ngram_ids <= overlap_ids
    assert report.n_leaking_examples < sum(len(f.example_ids) for f in report.findings)


def test_report_records_the_parameters_it_ran_with() -> None:
    report = find_leakage(_ALL["safety_toy"][0], near_method="jaccard")
    assert report.near_method == "jaccard"
    assert report.answer_ngram == 2
    assert report.answer_threshold == 0.5
    assert report.reference_ngram == 8
    assert report.reference_split == "train"
    assert report.target_splits == ["dev", "test"]
    assert report.reference_size == 7
    assert report.ignored_answers == ["comply", "refuse"]


def test_report_reuses_a_supplied_dedup_report() -> None:
    examples, _ = _ALL["agent_tasks_toy"]
    dedup = find_duplicates(examples, near_method="jaccard")
    reused = find_leakage(examples, dedup_report=dedup)
    computed = find_leakage(examples, near_method="jaccard")
    assert reused.to_dict() == computed.to_dict()


def test_find_leakage_is_deterministic() -> None:
    examples, _ = _ALL["retrieval_toy"]
    assert find_leakage(examples).to_dict() == find_leakage(examples).to_dict()


def test_report_to_dict_is_json_serializable() -> None:
    examples, _ = _ALL["retrieval_toy"]
    payload = json.loads(json.dumps(find_leakage(examples).to_dict()))
    assert payload["counts"]["retrieval_contamination"] == 2
    assert payload["cross_split_sources"] == {
        "src-boiling": {"test": ["ret-006"], "dev": ["ret-009"]}
    }
    assert payload["duplicate_documents"] == [["ret-001/d1[train]", "ret-002/d3[test]"]]
    assert payload["params"]["reference_split"] == "train"


def test_format_report_surfaces_every_channel() -> None:
    pytest.importorskip("sklearn")
    text = format_report(find_leakage(_ALL["qa_toy"][0]))
    assert "train/test overlap      : 2" in text
    assert "answer in prompt        : 1" in text
    assert "qa-007" in text and "berlin" in text
    assert "src-shakespeare" in text
    assert "word_2gram containment >= 0.5" in text


def test_format_report_names_the_ignored_label_vocabulary() -> None:
    """Nothing is suppressed silently: the guard's decision is printed."""
    text = format_report(find_leakage(_ALL["safety_toy"][0]))
    assert "ignored label vocabulary: 'comply', 'refuse'" in text
    assert "ignored label vocabulary" not in format_report(find_leakage(_ALL["qa_toy"][0]))


def test_empty_dataset_produces_an_empty_report() -> None:
    report = find_leakage([])
    assert report == LeakageReport(n_examples=0, near_method=report.near_method)
    assert report.findings == []
    assert report.n_leaking_examples == 0


@pytest.mark.parametrize("name", FIXTURES)
def test_every_reported_id_exists_in_the_dataset(name: str) -> None:
    examples, _ = _ALL[name]
    known = {e.example_id for e in examples}
    report = find_leakage(examples)
    for finding in report.findings:
        assert set(finding.example_ids) <= known
        assert finding.example_ids


@pytest.mark.parametrize("name", FIXTURES)
def test_each_channel_emits_only_its_own_kind(name: str) -> None:
    """One module, one LeakageKind — the mapping in the package docstring, enforced."""
    report = find_leakage(_ALL[name][0])
    for findings, kind in (
        (report.train_test_overlap, LeakageKind.TRAIN_TEST_OVERLAP),
        (report.answer_in_prompt, LeakageKind.ANSWER_IN_PROMPT),
        (report.ngram_overlap, LeakageKind.NGRAM_OVERLAP),
        (report.retrieval_contamination, LeakageKind.RETRIEVAL_CONTAMINATION),
    ):
        assert all(f.kind == kind for f in findings)


def test_the_four_channels_cover_every_leakage_kind() -> None:
    """No LeakageKind is declared in the schema without a check that can emit it."""
    emitted = {
        f.kind for name in FIXTURES for f in find_leakage(_ALL[name][0]).findings
    }
    assert emitted == set(LeakageKind)
