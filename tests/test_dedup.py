"""Tests for the deduplication layer (Milestone 4).

Covered here:

  * **Primitives** — normalization, tokenization, n-gram shingling, Jaccard, and
    the canonical/ hashed content signature (including separator-injection
    safety).
  * **Exact stage** — scored against every fixture's ``defects.yaml`` ground
    truth. This stage is exact, so the bar is absolute: it must find *all* planted
    exact duplicates and invent none.
  * **Normalized stage** — catches formatting-only duplicates, refuses to merge
    a shared prompt with differing answers, and never re-reports a cluster the
    exact stage already owns.
  * **Near stage** — the heuristic one. Tested for what it *does* catch (4 of the
    5 planted paraphrases with TF-IDF, 3 with Jaccard), for the answer gate that
    keeps it precise, for its scikit-learn fallback, and — importantly — for the
    two documented failures. The retrieval miss is pinned with a test that asserts
    *why* it is unreachable, so that if someone ever makes it detectable, the test
    fails and forces the documentation to be updated rather than silently drifting.
  * **Report** — count consistency, determinism, and the guarantee that only exact
    clusters ever nominate an example for removal.

Everything is CPU-only and offline. Tests that need the optional TF-IDF path skip
cleanly when scikit-learn is absent.
"""

from __future__ import annotations

import json

import pytest

from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup import (
    DedupReport,
    find_duplicates,
    format_report,
)
from eval_dataset_forge.dedup.exact import (
    canonical_output,
    content_signature,
    duplicate_input_clusters,
    duplicate_sources,
    exact_duplicates,
    sha256_hex,
)
from eval_dataset_forge.dedup.fuzzy import (
    jaccard,
    normalize_text,
    normalized_duplicates,
    normalized_signature,
    tokenize,
    word_ngrams,
)
from eval_dataset_forge.dedup.near_duplicate import (
    DEFAULT_THRESHOLDS,
    answers_agree,
    near_duplicates,
    resolve_near_method,
    similarity_matrix,
)
from eval_dataset_forge.schemas import DatasetExample, DedupKind

FIXTURES = ["qa_toy", "agent_tasks_toy", "retrieval_toy", "bio_toy", "safety_toy"]


def _load(name: str) -> tuple[list[DatasetExample], dict]:
    fixture_dir = loader.FIXTURES_ROOT / name
    return (
        loader.load_examples(fixture_dir / "examples.jsonl"),
        loader.load_defects(fixture_dir / "defects.yaml"),
    )


# Loaded once at import; the data is tiny and fully offline.
_ALL: dict[str, tuple[list[DatasetExample], dict]] = {n: _load(n) for n in FIXTURES}


def _planted(name: str, category: str) -> list[list[str]]:
    """Ground-truth id clusters of one category from a fixture's ``defects.yaml``."""
    entries = _ALL[name][1]["planted_defects"].get(category, [])
    return [sorted(entry["ids"]) for entry in entries]


def _clusters(findings) -> set[tuple[str, ...]]:
    return {tuple(sorted(f.example_ids)) for f in findings}


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


# --------------------------------------------------------------------------- #
# Primitives: normalization & tokenization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Who wrote Hamlet?", "who wrote hamlet"),
        ("  Who   wrote\tHamlet ?  ", "who wrote hamlet"),
        ("WHO WROTE HAMLET", "who wrote hamlet"),
        ("Who wrote “Hamlet”?", "who wrote hamlet"),  # NFKC smart quotes
        ("Paris.", "paris"),
        ("", ""),
        ("!!!", ""),  # punctuation-only collapses to empty
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent() -> None:
    once = normalize_text("  Who   WROTE “Hamlet”?! ")
    assert normalize_text(once) == once


def test_normalize_text_merges_only_punctuation_and_case() -> None:
    # Distinct words are never merged, however similar.
    assert normalize_text("produces") != normalize_text("producing")


def test_tokenize_splits_on_punctuation_and_lowercases() -> None:
    assert tokenize("Fizz-Buzz, for n=15!") == ["fizz", "buzz", "for", "n", "15"]


def test_tokenize_empty() -> None:
    assert tokenize("") == []
    assert tokenize("???") == []


# --------------------------------------------------------------------------- #
# Primitives: n-grams & Jaccard
# --------------------------------------------------------------------------- #


def test_word_ngrams_unigrams_are_the_bag_of_words() -> None:
    assert word_ngrams(["a", "b", "a"], 1) == {("a",), ("b",)}


def test_word_ngrams_bigrams_are_contiguous() -> None:
    assert word_ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}


def test_word_ngrams_short_sequence_returns_one_short_shingle() -> None:
    # Shorter than n: the whole sequence, so a short prompt still matches a copy.
    assert word_ngrams(["a", "b"], 3) == {("a", "b")}


def test_word_ngrams_empty_and_invalid_n() -> None:
    assert word_ngrams([], 3) == set()
    with pytest.raises(ValueError, match="n must be >= 1"):
        word_ngrams(["a"], 0)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ({1, 2}, {1, 2}, 1.0),
        ({1, 2}, {3, 4}, 0.0),
        (set(), set(), 1.0),  # both empty: no evidence they differ
        ({1}, set(), 0.0),
        ({1, 2, 3}, {2, 3, 4}, 0.5),  # |∩|=2, |∪|=4
    ],
)
def test_jaccard(a: set, b: set, expected: float) -> None:
    assert jaccard(a, b) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Primitives: canonical output & content signature
# --------------------------------------------------------------------------- #


def test_canonical_output_encodes_strings_as_json() -> None:
    assert canonical_output("Paris") == '"Paris"'


def test_canonical_output_is_key_order_invariant() -> None:
    assert canonical_output({"a": 1, "b": 2}) == canonical_output({"b": 2, "a": 1})


def test_canonical_output_distinguishes_types() -> None:
    """A quoted string and a bare scalar are different gold answers, not duplicates."""
    assert canonical_output("4") != canonical_output(4)
    assert canonical_output("true") != canonical_output(True)
    assert canonical_output(True) == "true"


def test_differently_typed_answers_are_not_exact_duplicates() -> None:
    examples = [
        _example(example_id="a", input="What is 2 + 2?", expected_output="4"),
        _example(example_id="b", input="What is 2 + 2?", expected_output=4),
    ]
    assert exact_duplicates(examples) == []
    assert duplicate_input_clusters(examples) == [["a", "b"]]


def test_sha256_hex_is_deterministic() -> None:
    assert sha256_hex("abc") == sha256_hex("abc")
    assert sha256_hex("abc") != sha256_hex("abd")


def test_content_signature_resists_field_boundary_collision() -> None:
    """("ab", "c") and ("a", "bc") must not hash alike — hence the NUL separator."""
    left = _example(input="ab", expected_output="c")
    right = _example(input="a", expected_output="bc")
    assert content_signature(left) != content_signature(right)


def test_content_signature_ignores_id_split_and_metadata() -> None:
    """Dedup identity is (input, expected_output) — nothing else."""
    left = _example(example_id="a", split="train", metadata={"x": 1})
    right = _example(example_id="b", split="test", difficulty_tags=["hard"])
    assert content_signature(left) == content_signature(right)


# --------------------------------------------------------------------------- #
# Exact stage — scored against ground truth
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURES)
def test_exact_duplicates_match_ground_truth_exactly(name: str) -> None:
    """The exact stage has no threshold: perfect recall and zero false positives."""
    examples, _ = _ALL[name]
    found = _clusters(exact_duplicates(examples))
    planted = {tuple(ids) for ids in _planted(name, "exact_duplicates")}
    assert found == planted


def test_retrieval_fixture_has_no_exact_duplicates() -> None:
    examples, _ = _ALL["retrieval_toy"]
    assert exact_duplicates(examples) == []


def test_exact_finding_shape() -> None:
    examples, _ = _ALL["qa_toy"]
    (finding,) = exact_duplicates(examples)
    assert finding.kind == DedupKind.EXACT
    # Identical by construction: a similarity would invite comparison against
    # heuristic near-duplicate scores, which are a different quantity.
    assert finding.similarity is None
    assert finding.method == "sha256(input+expected_output)"


def test_exact_requires_matching_answer_not_just_prompt() -> None:
    """A shared prompt with different answers is a label conflict, not a duplicate."""
    examples = [
        _example(example_id="a", expected_output="Paris"),
        _example(example_id="b", expected_output="Lyon"),
    ]
    assert exact_duplicates(examples) == []
    # ...but the shared prompt is still surfaced, for the label audit to consume.
    assert duplicate_input_clusters(examples) == [["a", "b"]]


def test_exact_duplicates_preserve_input_order() -> None:
    examples = [
        _example(example_id="z"),
        _example(example_id="a"),
        _example(example_id="m"),
    ]
    (finding,) = exact_duplicates(examples)
    assert finding.example_ids == ["z", "a", "m"]


@pytest.mark.parametrize("name", FIXTURES)
def test_duplicate_sources_match_ground_truth(name: str) -> None:
    examples, _ = _ALL[name]
    found = duplicate_sources(examples)
    planted = {
        entry["source_id"]: sorted(entry["ids"])
        for entry in _ALL[name][1]["planted_defects"].get("duplicate_sources", [])
    }
    assert {sid: sorted(ids) for sid, ids in found.items()} == planted


# --------------------------------------------------------------------------- #
# Normalized stage
# --------------------------------------------------------------------------- #


def test_normalized_catches_formatting_only_duplicate() -> None:
    examples = [
        _example(example_id="a", input="Who wrote Hamlet?"),
        _example(example_id="b", input="  who   wrote “Hamlet” "),
    ]
    (finding,) = normalized_duplicates(examples)
    assert finding.kind == DedupKind.NORMALIZED
    assert finding.example_ids == ["a", "b"]
    assert finding.similarity is None


def test_normalized_does_not_reannounce_exact_duplicates() -> None:
    """Byte-identical pairs belong to the exact stage; the stages must be disjoint."""
    examples = [_example(example_id="a"), _example(example_id="b")]
    assert exact_duplicates(examples) != []
    assert normalized_duplicates(examples) == []


def test_normalized_will_not_merge_differing_answers() -> None:
    """Normalization is applied to the prompt only; the answer must still match."""
    examples = [
        _example(example_id="a", input="Who wrote Hamlet?", expected_output="Shakespeare"),
        _example(example_id="b", input="who wrote hamlet", expected_output="Marlowe"),
    ]
    assert normalized_signature(examples[0]) != normalized_signature(examples[1])
    assert normalized_duplicates(examples) == []


def test_normalized_reports_mixed_cluster_containing_an_exact_pair() -> None:
    """Two byte-identical copies plus a formatting variant is one normalized cluster."""
    examples = [
        _example(example_id="a", input="Who wrote Hamlet?"),
        _example(example_id="b", input="Who wrote Hamlet?"),
        _example(example_id="c", input="who wrote hamlet"),
    ]
    (finding,) = normalized_duplicates(examples)
    assert finding.example_ids == ["a", "b", "c"]


@pytest.mark.parametrize("name", FIXTURES)
def test_fixtures_contain_no_normalized_only_duplicates(name: str) -> None:
    """No fixture plants a formatting-only duplicate, so this stage stays silent."""
    examples, _ = _ALL[name]
    assert normalized_duplicates(examples) == []


# --------------------------------------------------------------------------- #
# Near stage — the answer gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ("Paris", "Paris", True),
        ("Paris", "paris.", True),  # normalized string comparison
        ("Paris", "Lyon", False),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),  # key order irrelevant
        (["d1"], ["d3"], False),  # retrieval doc ids: the documented blind spot
        ("4", 4, False),  # a string is never equal to a non-string
        ("", "", True),
    ],
)
def test_answers_agree(left, right, expected: bool) -> None:
    assert answers_agree(left, right) is expected


def test_answer_gate_suppresses_a_high_similarity_non_duplicate() -> None:
    """qa-001 ~ qa-010 score 0.62 — above every planted pair but one — yet differ.

    Without the gate this pair outranks three of the five planted paraphrases;
    with it, it disappears. This is the empirical justification for the default.
    """
    examples, _ = _ALL["qa_toy"]
    ids = [e.example_id for e in examples]
    matrix, _method = similarity_matrix(examples, method="jaccard")
    score = matrix[ids.index("qa-001")][ids.index("qa-010")]
    assert score >= DEFAULT_THRESHOLDS["jaccard"], "premise: the pair scores above threshold"

    gated = _clusters(near_duplicates(examples, method="jaccard"))
    assert not any({"qa-001", "qa-010"} <= set(c) for c in gated)

    ungated = _clusters(near_duplicates(examples, method="jaccard", require_same_answer=False))
    assert any({"qa-001", "qa-010"} <= set(c) for c in ungated)


# --------------------------------------------------------------------------- #
# Near stage — recall against ground truth
# --------------------------------------------------------------------------- #

# Planted paraphrase pairs that each method actually recovers at its default
# threshold. Measured, not aspirational — see docs/DEDUPLICATION.md §Calibration.
NEAR_RECALL: dict[str, dict[str, bool]] = {
    "tfidf": {
        "qa_toy": True,
        "agent_tasks_toy": True,
        "retrieval_toy": False,  # unreachable: see test_retrieval_paraphrase_is_unreachable
        "bio_toy": True,
        "safety_toy": True,
    },
    "jaccard": {
        "qa_toy": True,
        "agent_tasks_toy": True,
        "retrieval_toy": False,
        "bio_toy": False,  # 0.18 unweighted overlap; TF-IDF recovers it
        "safety_toy": True,
    },
}


def _covers(clusters: set[tuple[str, ...]], pair: list[str]) -> bool:
    """Whether one reported cluster contains both members of a planted pair."""
    return any(set(pair) <= set(cluster) for cluster in clusters)


@pytest.mark.parametrize("method", ["jaccard", "tfidf"])
@pytest.mark.parametrize("name", FIXTURES)
def test_near_duplicate_recall_is_as_documented(name: str, method: str) -> None:
    if method == "tfidf":
        pytest.importorskip("sklearn")
    examples, _ = _ALL[name]
    planted = _planted(name, "near_duplicates")
    assert len(planted) == 1, "each fixture plants exactly one near-duplicate pair"

    clusters = _clusters(near_duplicates(examples, method=method))
    assert _covers(clusters, planted[0]) is NEAR_RECALL[method][name]


def test_tfidf_recovers_more_planted_paraphrases_than_jaccard() -> None:
    """TF-IDF dominates on this corpus: it is why ``auto`` prefers it."""
    pytest.importorskip("sklearn")
    def recall(method: str) -> int:
        return sum(
            _covers(
                _clusters(near_duplicates(_ALL[name][0], method=method)),
                _planted(name, "near_duplicates")[0],
            )
            for name in FIXTURES
        )

    assert {m: recall(m) for m in ("jaccard", "tfidf")} == {"jaccard": 3, "tfidf": 4}


def test_bio_paraphrase_needs_idf_weighting() -> None:
    """"produces the hormone insulin" vs "responsible for producing insulin".

    Raw overlap is 2 tokens out of 11. Only after down-weighting the shared
    function words does the pair clear the threshold.
    """
    pytest.importorskip("sklearn")
    examples, _ = _ALL["bio_toy"]
    ids = [e.example_id for e in examples]
    i, j = ids.index("bio-001"), ids.index("bio-002")
    by_jaccard, _ = similarity_matrix(examples, method="jaccard")
    by_tfidf, _ = similarity_matrix(examples, method="tfidf")
    assert by_jaccard[i][j] < DEFAULT_THRESHOLDS["jaccard"] <= by_tfidf[i][j]


# --------------------------------------------------------------------------- #
# Near stage — documented failure modes, pinned
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["jaccard", "tfidf"])
def test_retrieval_paraphrase_is_unreachable_by_any_threshold(method: str) -> None:
    """The planted paraphrase scores *below* a genuine non-duplicate in its own fixture.

    ret-001/ret-002 ("Which organelle carries out photosynthesis…" vs "Where does
    photosynthesis take place…") is a deep paraphrase. ret-006/ret-009 (boiling
    point vs freezing point of water) is not a duplicate at all. Because the
    non-duplicate scores higher, no threshold — with or without the answer gate —
    admits the first without also admitting the second. Only semantics separates
    them, and the offline build has none.

    If this assertion ever fails, the detector improved: update
    ``docs/DEDUPLICATION.md`` and ``NEAR_RECALL`` rather than deleting the test.
    """
    if method == "tfidf":
        pytest.importorskip("sklearn")
    examples, _ = _ALL["retrieval_toy"]
    ids = [e.example_id for e in examples]
    matrix, _resolved = similarity_matrix(examples, method=method)

    planted = matrix[ids.index("ret-001")][ids.index("ret-002")]
    non_duplicate = matrix[ids.index("ret-006")][ids.index("ret-009")]
    assert planted < non_duplicate


def test_template_questions_are_a_known_false_positive() -> None:
    """qa-012 ("Who wrote the play Hamlet?") joins the Romeo-and-Juliet cluster.

    Same template, same gold answer, so neither the lexical score nor the answer
    gate can exclude it. It is independently flagged as a duplicate *source*
    record, which is the signal a reviewer should act on.
    """
    examples, defects = _ALL["qa_toy"]
    (cluster,) = near_duplicates(examples)
    assert set(cluster.example_ids) == {"qa-003", "qa-004", "qa-012"}
    assert duplicate_sources(examples)["src-shakespeare"] == ["qa-003", "qa-012"]
    assert ["qa-003", "qa-012"] in [
        sorted(e["ids"]) for e in defects["planted_defects"]["duplicate_sources"]
    ]


def test_wider_ngrams_trade_recall_for_precision() -> None:
    """Contiguous shingles are destroyed by word substitution.

    Widening n=1 -> n=2 cuts Jaccard recall from 3 planted pairs to 1: only
    qa-003/qa-004, whose paraphrase inserts words rather than replacing them,
    keeps enough shared bigrams. This is why :data:`DEFAULT_NGRAM` is 1.
    """
    recovered = {
        ngram: [
            name
            for name in FIXTURES
            if _covers(
                _clusters(near_duplicates(_ALL[name][0], method="jaccard", ngram=ngram)),
                _planted(name, "near_duplicates")[0],
            )
        ]
        for ngram in (1, 2)
    }
    assert recovered[1] == ["qa_toy", "agent_tasks_toy", "safety_toy"]
    assert recovered[2] == ["qa_toy"]


# --------------------------------------------------------------------------- #
# Near stage — mechanics
# --------------------------------------------------------------------------- #


def test_near_similarity_is_the_weakest_edge_of_the_cluster() -> None:
    examples, _ = _ALL["qa_toy"]
    ids = [e.example_id for e in examples]
    matrix, _ = similarity_matrix(examples, method="jaccard")
    (cluster,) = near_duplicates(examples, method="jaccard")

    members = [ids.index(eid) for eid in cluster.example_ids]
    edges = [
        matrix[i][j]
        for i in members
        for j in members
        if i < j and matrix[i][j] >= DEFAULT_THRESHOLDS["jaccard"]
    ]
    assert cluster.similarity == pytest.approx(min(edges), abs=1e-4)
    # ...and a lower bound on every member pair, including the non-adjacent ones.
    assert cluster.similarity <= max(edges)


def test_near_skips_pairs_owned_by_the_exact_stage() -> None:
    """An exact duplicate has similarity 1.0 but must not appear as a near finding."""
    examples = [_example(example_id="a"), _example(example_id="b")]
    assert exact_duplicates(examples) != []
    assert near_duplicates(examples) == []


def test_near_threshold_is_monotonic() -> None:
    examples, _ = _ALL["qa_toy"]
    low = _clusters(near_duplicates(examples, method="jaccard", threshold=0.3))
    high = _clusters(near_duplicates(examples, method="jaccard", threshold=0.9))
    assert high == set()
    assert low != set()


def test_near_is_deterministic() -> None:
    examples, _ = _ALL["agent_tasks_toy"]
    first = [f.model_dump() for f in near_duplicates(examples, method="jaccard")]
    second = [f.model_dump() for f in near_duplicates(examples, method="jaccard")]
    assert first == second


def test_near_method_label_records_the_parameters() -> None:
    examples, _ = _ALL["qa_toy"]
    (finding,) = near_duplicates(examples, method="jaccard", threshold=0.4, ngram=1)
    assert finding.kind == DedupKind.NEAR
    assert finding.method == "jaccard(word_1gram)@0.4"


def test_near_handles_degenerate_inputs() -> None:
    assert near_duplicates([]) == []
    assert near_duplicates([_example()]) == []


def test_near_handles_empty_prompts() -> None:
    """An all-empty corpus has no TF-IDF vocabulary; it must not raise."""
    examples = [
        _example(example_id="a", input="", expected_output=""),
        _example(example_id="b", input="", expected_output=""),
    ]
    # Identical content, so the exact stage owns them and near stays silent.
    assert near_duplicates(examples) == []
    matrix, _ = similarity_matrix(examples, method="tfidf")
    assert matrix[0][1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Method resolution & the scikit-learn fallback
# --------------------------------------------------------------------------- #


def test_resolve_near_method_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown near-duplicate method"):
        resolve_near_method("embeddings")


def test_resolve_near_method_jaccard_never_upgrades() -> None:
    assert resolve_near_method("jaccard") == "jaccard"


def test_auto_prefers_tfidf_when_sklearn_is_present() -> None:
    pytest.importorskip("sklearn")
    assert resolve_near_method("auto") == "tfidf"


def test_tfidf_falls_back_to_jaccard_without_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting TF-IDF without scikit-learn degrades; it does not raise."""
    monkeypatch.setattr(
        "eval_dataset_forge.dedup.near_duplicate.sklearn_available", lambda: False
    )
    assert resolve_near_method("auto") == "jaccard"
    assert resolve_near_method("tfidf") == "jaccard"

    examples, _ = _ALL["qa_toy"]
    (finding,) = near_duplicates(examples, method="tfidf")
    # The finding records what actually ran, not what was asked for.
    assert finding.method.startswith("jaccard(")


def test_report_records_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "eval_dataset_forge.dedup.near_duplicate.sklearn_available", lambda: False
    )
    report = find_duplicates(_ALL["qa_toy"][0], near_method="tfidf")
    assert report.near_method == "jaccard"
    assert report.near_threshold == DEFAULT_THRESHOLDS["jaccard"]


# --------------------------------------------------------------------------- #
# DedupReport
# --------------------------------------------------------------------------- #


def test_find_duplicates_partitions_the_evidence() -> None:
    """The three stages are disjoint, so the counts add up rather than triple-count."""
    examples, _ = _ALL["qa_toy"]
    report = find_duplicates(examples, near_method="jaccard")
    assert len(report.findings) == len(report.exact) + len(report.normalized) + len(report.near)

    exact_pairs = _clusters(report.exact)
    near_pairs = _clusters(report.near)
    assert exact_pairs.isdisjoint(near_pairs)


def test_report_counts_are_consistent() -> None:
    examples, _ = _ALL["qa_toy"]
    report = find_duplicates(examples, near_method="jaccard")
    counts = report.counts()
    assert counts["examples"] == 14
    assert counts["exact_clusters"] == 1
    assert counts["near_clusters"] == 1
    assert counts["duplicate_sources"] == 1
    assert counts["redundant_examples"] == 1
    # qa-001,qa-002 (exact) + qa-003,qa-004,qa-012 (near)
    assert counts["examples_in_clusters"] == 5


def test_only_exact_clusters_nominate_examples_for_removal() -> None:
    """Never auto-drop on a heuristic: a near cluster contributes nothing here."""
    examples, _ = _ALL["qa_toy"]
    report = find_duplicates(examples)
    assert report.redundant_example_ids == ["qa-002"]
    near_ids = {eid for f in report.near for eid in f.example_ids}
    assert not near_ids & set(report.redundant_example_ids)


def test_redundant_ids_keep_the_first_member_of_each_cluster() -> None:
    examples = [
        _example(example_id="a"),
        _example(example_id="b"),
        _example(example_id="c"),
    ]
    report = find_duplicates(examples)
    assert report.redundant_example_ids == ["b", "c"]


def test_include_near_false_yields_only_certain_findings() -> None:
    examples, _ = _ALL["qa_toy"]
    report = find_duplicates(examples, include_near=False)
    assert report.near == []
    assert report.exact != []


def test_report_to_dict_is_json_serializable() -> None:
    examples, _ = _ALL["qa_toy"]
    report = find_duplicates(examples, near_method="jaccard")
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["near_params"] == {
        "method": "jaccard",
        "threshold": DEFAULT_THRESHOLDS["jaccard"],
        "ngram": 1,
        "require_same_answer": True,
    }
    assert payload["exact_duplicates"][0]["example_ids"] == ["qa-001", "qa-002"]
    assert payload["duplicate_source_ids"] == {"src-shakespeare": ["qa-003", "qa-012"]}


def test_format_report_surfaces_method_and_clusters() -> None:
    examples, _ = _ALL["qa_toy"]
    text = format_report(find_duplicates(examples, near_method="jaccard"))
    assert "exact clusters      : 1" in text
    assert "qa-001, qa-002" in text
    assert "jaccard(word_1gram)@0.3" in text
    assert "src-shakespeare" in text


def test_empty_dataset_produces_an_empty_report() -> None:
    report = find_duplicates([])
    assert report == DedupReport(
        n_examples=0,
        near_method=report.near_method,
        near_threshold=report.near_threshold,
    )
    assert report.findings == []


@pytest.mark.parametrize("name", FIXTURES)
def test_every_reported_id_exists_in_the_dataset(name: str) -> None:
    examples, _ = _ALL[name]
    known = {e.example_id for e in examples}
    report = find_duplicates(examples)
    for finding in report.findings:
        assert set(finding.example_ids) <= known
        assert len(finding.example_ids) >= 2
