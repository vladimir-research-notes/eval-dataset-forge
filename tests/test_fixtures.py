"""Tests for the fixture datasets (Milestone 2).

Two things are locked here:

  * **Structural health** — every fixture parses against the ``DatasetExample``
    schema, has unique ids, a single known ``task_family``, and a ``defects.yaml``
    whose ground truth references only real example ids.
  * **Ground-truth self-consistency** — the defects declared in each
    ``defects.yaml`` are actually present in the data. This keeps the planted
    defects honest: the later audit stages (Milestones 4–6) are scored against
    these files, so a stale or wrong ground-truth entry would silently corrupt the
    recall numbers. An exact-duplicate pair must really be identical; a
    train/test-overlap pair must really span both splits; an answer-in-prompt leak
    must really appear in the prompt; and so on.

Everything here is CPU-only and offline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from eval_dataset_forge.datasets import loader
from eval_dataset_forge.schemas import KNOWN_TASK_FAMILIES, DatasetExample

ROOT = Path(__file__).resolve().parents[1]

# fixture name -> expected task_family
FIXTURES: dict[str, str] = {
    "qa_toy": "qa",
    "agent_tasks_toy": "agent_task",
    "retrieval_toy": "retrieval",
    "bio_toy": "bio",
    "safety_toy": "safety",
}


def _load(name: str) -> tuple[list[DatasetExample], dict]:
    fixture_dir = loader.FIXTURES_ROOT / name
    examples = loader.load_examples(fixture_dir / "examples.jsonl")
    defects = loader.load_defects(fixture_dir / "defects.yaml")
    return examples, defects


# Load every fixture once at import; the data is tiny and fully offline.
_ALL: dict[str, tuple[list[DatasetExample], dict]] = {n: _load(n) for n in FIXTURES}


def _by_id(examples: list[DatasetExample]) -> dict[str, DatasetExample]:
    return {e.example_id: e for e in examples}


def _planted(name: str, category: str) -> list:
    return _ALL[name][1].get("planted_defects", {}).get(category, [])


# --------------------------------------------------------------------------- #
# Parametrization lists derived from the ground-truth files
# --------------------------------------------------------------------------- #

_exact_dup_cases = [(n, e["ids"]) for n in FIXTURES for e in _planted(n, "exact_duplicates")]
_near_dup_cases = [(n, e["ids"]) for n in FIXTURES for e in _planted(n, "near_duplicates")]
_leakage = {n: _planted(n, "leakage") for n in FIXTURES}
_tt_overlap_cases = [
    (n, e["ids"]) for n in FIXTURES for e in _leakage[n] if e["kind"] == "train_test_overlap"
]
_answer_in_prompt_cases = [
    (n, e["ids"]) for n in FIXTURES for e in _leakage[n] if e["kind"] == "answer_in_prompt"
]
_label_issues = {n: _planted(n, "label_issues") for n in FIXTURES}
_missing_cases = [
    (n, e["id"]) for n in FIXTURES for e in _label_issues[n] if e["kind"] == "missing"
]
_conflicting_cases = [
    (n, e["id"]) for n in FIXTURES for e in _label_issues[n] if e["kind"] == "conflicting"
]
_ambiguous_cases = [
    (n, e["id"]) for n in FIXTURES for e in _label_issues[n] if e["kind"] == "ambiguous"
]
_mislabeled_cases = [
    (n, e) for n in FIXTURES for e in _label_issues[n] if e["kind"] == "mislabeled"
]


# --------------------------------------------------------------------------- #
# Structural health
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_files_exist(name: str) -> None:
    fixture_dir = loader.FIXTURES_ROOT / name
    for fname in ("examples.jsonl", "defects.yaml", "README.md"):
        assert (fixture_dir / fname).is_file(), f"missing {name}/{fname}"


def test_discover_finds_all_fixtures() -> None:
    found = {p.name for p in loader.discover_fixtures()}
    assert set(FIXTURES).issubset(found), f"discover_fixtures missed {set(FIXTURES) - found}"


@pytest.mark.parametrize("name", FIXTURES)
def test_examples_parse(name: str) -> None:
    examples, _ = _ALL[name]
    assert examples, f"{name} has no examples"
    assert all(isinstance(e, DatasetExample) for e in examples)


@pytest.mark.parametrize("name", FIXTURES)
def test_example_ids_unique(name: str) -> None:
    examples, _ = _ALL[name]
    ids = [e.example_id for e in examples]
    assert len(ids) == len(set(ids)), f"{name} has duplicate example_id(s)"


@pytest.mark.parametrize(("name", "family"), FIXTURES.items())
def test_task_family_consistent_and_known(name: str, family: str) -> None:
    examples, _ = _ALL[name]
    families = {e.task_family for e in examples}
    assert families == {family}, f"{name} families {families} != {{{family}}}"
    assert family in KNOWN_TASK_FAMILIES


@pytest.mark.parametrize("name", FIXTURES)
def test_defects_reference_real_ids(name: str) -> None:
    examples, defects = _ALL[name]
    known = {e.example_id for e in examples}
    refs = _collect_id_refs(defects.get("planted_defects", {}))
    assert refs, f"{name}/defects.yaml declares no example references"
    missing = refs - known
    assert not missing, f"{name}/defects.yaml references unknown ids: {sorted(missing)}"


@pytest.mark.parametrize("name", FIXTURES)
def test_defect_example_count_matches(name: str) -> None:
    examples, defects = _ALL[name]
    assert defects.get("example_count") == len(examples)


def _collect_id_refs(node: object) -> set[str]:
    refs: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ids" and isinstance(value, list):
                refs.update(str(v) for v in value)
            elif key in ("id", "example") and isinstance(value, (str, int)):
                refs.add(str(value))
            else:
                refs |= _collect_id_refs(value)
    elif isinstance(node, list):
        for item in node:
            refs |= _collect_id_refs(item)
    return refs


# --------------------------------------------------------------------------- #
# Ground-truth self-consistency: the planted defects are really present
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "ids"), _exact_dup_cases)
def test_exact_duplicates_are_identical(name: str, ids: list[str]) -> None:
    by_id = _by_id(_ALL[name][0])
    a, b = (by_id[i] for i in ids)
    assert a.input == b.input, f"{name} {ids}: inputs differ"
    assert a.expected_output == b.expected_output, f"{name} {ids}: outputs differ"


@pytest.mark.parametrize(("name", "ids"), _near_dup_cases)
def test_near_duplicates_differ_textually(name: str, ids: list[str]) -> None:
    by_id = _by_id(_ALL[name][0])
    a, b = (by_id[i] for i in ids)
    # A paraphrase near-duplicate is a *different* surface string, not an exact copy.
    assert a.input != b.input, f"{name} {ids}: near-duplicate inputs are identical"


@pytest.mark.parametrize(("name", "ids"), _tt_overlap_cases)
def test_train_test_overlap_spans_splits(name: str, ids: list[str]) -> None:
    by_id = _by_id(_ALL[name][0])
    splits = {by_id[i].split for i in ids}
    assert "train" in splits and "test" in splits, f"{name} {ids}: overlap does not span train+test"


@pytest.mark.parametrize(("name", "ids"), _answer_in_prompt_cases)
def test_answer_in_prompt_leaks(name: str, ids: list[str]) -> None:
    by_id = _by_id(_ALL[name][0])
    for i in ids:
        assert _answer_leaks_into_prompt(by_id[i]), f"{name} {i}: gold answer not found in prompt"


@pytest.mark.parametrize(("name", "eid"), _missing_cases)
def test_missing_labels_are_empty(name: str, eid: str) -> None:
    ex = _by_id(_ALL[name][0])[eid]
    assert ex.expected_output in ("", [], {}), f"{name} {eid}: expected empty output for 'missing'"


@pytest.mark.parametrize(("name", "eid"), _conflicting_cases)
def test_conflicting_labels_have_disagreement(name: str, eid: str) -> None:
    ex = _by_id(_ALL[name][0])[eid]
    annotations = ex.labels.get("annotations", [])
    values = {a.get("value") for a in annotations}
    assert len(values) >= 2, f"{name} {eid}: expected >=2 distinct annotator values, got {values}"


@pytest.mark.parametrize(("name", "eid"), _ambiguous_cases)
def test_ambiguous_examples_are_tagged(name: str, eid: str) -> None:
    ex = _by_id(_ALL[name][0])[eid]
    assert "ambiguous" in ex.difficulty_tags, f"{name} {eid}: missing 'ambiguous' difficulty tag"


@pytest.mark.parametrize(("name", "entry"), _mislabeled_cases)
def test_mislabeled_marked_not_auto_detectable(name: str, entry: dict) -> None:
    # A well-formed but factually wrong gold answer is a documented limitation:
    # nothing in this project claims to detect it, so it must not be counted
    # against audit recall.
    eid = entry.get("id")
    assert entry.get("auto_detectable") is False, f"{name} {eid}: expected auto_detectable:false"


def _answer_leaks_into_prompt(example: DatasetExample) -> bool:
    prompt = example.input
    for candidate in _answer_strings(example.expected_output):
        for line in candidate.splitlines():
            fragment = line.strip()
            if len(fragment) >= 4 and fragment in prompt:
                return True
    return False


def _answer_strings(expected: object) -> list[str]:
    if isinstance(expected, str):
        return [expected]
    if isinstance(expected, list):
        return [x for x in expected if isinstance(x, str)]
    if isinstance(expected, dict):
        return [v for v in expected.values() if isinstance(v, str)]
    return []


# --------------------------------------------------------------------------- #
# Family-specific invariants
# --------------------------------------------------------------------------- #


def test_retrieval_duplicate_documents_are_identical() -> None:
    examples, _ = _ALL["retrieval_toy"]
    by_id = _by_id(examples)
    for entry in _planted("retrieval_toy", "duplicate_documents"):
        texts = {_doc_text(by_id[ref["example"]], ref["doc_id"]) for ref in entry["example_docs"]}
        assert len(texts) == 1, f"duplicate_documents {entry}: texts are not identical"


def test_retrieval_expected_output_is_doc_id_list() -> None:
    examples, _ = _ALL["retrieval_toy"]
    for ex in examples:
        assert isinstance(ex.expected_output, list), f"{ex.example_id}: output must be a list"
        docs = {d["doc_id"] for d in ex.metadata["retrieval"]["docs"]}
        assert set(ex.expected_output).issubset(docs), f"{ex.example_id}: relevance ids not in docs"


def _doc_text(example: DatasetExample, doc_id: str) -> str:
    for doc in example.metadata["retrieval"]["docs"]:
        if doc["doc_id"] == doc_id:
            return doc["text"]
    raise KeyError(f"{example.example_id} has no doc {doc_id}")


def test_safety_unsafe_items_are_placeholders() -> None:
    # Enforces the safety scope in docs/SAFETY.md: unsafe / ambiguous items carry
    # no operational content, only a SYNTHETIC-*-PLACEHOLDER category description.
    examples, _ = _ALL["safety_toy"]
    for ex in examples:
        boundary = ex.metadata.get("boundary")
        placeholder = ex.metadata.get("category") == "synthetic_placeholder"
        if placeholder or boundary in ("clear_unsafe", "ambiguous"):
            assert "PLACEHOLDER" in ex.input, f"{ex.example_id}: unsafe item is not a placeholder"


def test_safety_label_convention() -> None:
    examples, _ = _ALL["safety_toy"]
    issues = _label_issues["safety_toy"]
    inconsistent = {e["id"] for e in issues if e["kind"] == "inconsistent_format"}
    for ex in examples:
        if ex.expected_output == "" or ex.example_id in inconsistent:
            continue
        assert ex.expected_output in ("comply", "refuse"), f"{ex.example_id}: off-convention gold"
    # And the planted inconsistent-format example really does break the convention.
    for eid in inconsistent:
        assert _by_id(examples)[eid].expected_output not in ("comply", "refuse")


# --------------------------------------------------------------------------- #
# The validation script runs offline and passes on the committed fixtures
# --------------------------------------------------------------------------- #


def test_validation_script_passes() -> None:
    spec = importlib.util.spec_from_file_location(
        "validate_fixtures", ROOT / "scripts" / "validate_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main([]) == 0
