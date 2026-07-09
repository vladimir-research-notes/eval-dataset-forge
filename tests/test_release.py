"""Tests for the release builder (Milestone 10).

Covered here:

  * **Canonicalization vs presentation** — the hashed form sorts keys and the written form
    does not, and the two disagree on purpose. Reordering a record's keys, or escaping its
    non-ASCII, must not change its digest; reordering the *file* must not change
    ``content_id``.
  * **The two identities** — ``content_id`` ignores ``split`` and example order but not
    multiplicity; ``eval_id`` does not ignore ``split``. The headline property is pinned
    directly: rebuilding the same corpus at a different seed holds ``content_id`` fixed and
    moves ``eval_id``, and the reported test-set accuracy of a fixed score vector moves
    with it.
  * **Bump inference** — every level is reachable and each is forced by exactly the change
    it names. ``major`` for anything an evaluation split can see, ``minor`` for train-only
    edits, ``patch`` for packaging, ``none`` for a rebuild whose only difference is the
    clock. Two properties get their own tests because they are the ones a careless
    implementation gets wrong: a re-split with no content change is ``major``, and a new
    ``created_at`` alone is ``none``.
  * **The audit describes what shipped** — dropping an exact duplicate dissolves its
    cluster, so the released ``audit.json`` must report one fewer, not the source's count.
  * **Exclusions are recorded, not silent** — every dropped example lands in
    ``manifest.excluded`` with the id of the member that was kept, and
    ``source_content_id`` still pins the pre-exclusion corpus.
  * **Verification** — a clean release passes every check; a single flipped byte in
    ``examples.jsonl``, a split file that no longer partitions it, and a doctored count are
    each caught, with the specific check named.
  * **Determinism** — two builds at a fixed ``created_at`` are byte-identical in every
    artifact; two builds at different clocks differ in ``manifest.json`` and nothing else.
  * **Honest negatives** — ``manifest.json`` is not self-hashed, and a rewritten manifest
    that is internally consistent verifies clean. The test asserts the hole rather than
    hiding it.

Everything is CPU-only and offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_dataset_forge import SCHEMA_VERSION, __version__
from eval_dataset_forge.cli import main
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.datasets.manifest import (
    AUDIT_FILE,
    EXAMPLES_FILE,
    MANIFEST_FILE,
    REPRODUCE_FILE,
    build_manifest,
    format_verify,
    load_manifest,
    verify_release,
)
from eval_dataset_forge.datasets.release import (
    DROP_CHOICES,
    Exclusion,
    build_release,
    format_report,
    plan_exclusions,
    write_release,
)
from eval_dataset_forge.datasets.versioning import (
    BUMP_LEVELS,
    EVAL_SPLITS,
    compare,
    content_id,
    eval_id,
    example_digest,
    format_delta,
    next_version,
    parse_version,
)
from eval_dataset_forge.datasets.writer import (
    canonical_bytes,
    render_jsonl,
    sha256_bytes,
    sha256_file,
)
from eval_dataset_forge.dedup import find_duplicates
from eval_dataset_forge.labels import audit_labels
from eval_dataset_forge.schemas import DatasetExample, DatasetManifest, Split

QA_TOY = Path("datasets/fixtures/qa_toy/examples.jsonl")
FIXED_CLOCK = "2026-01-01T00:00:00+00:00"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _example(example_id: str = "e1", *, split: str = "train", **overrides) -> DatasetExample:
    payload = {
        "example_id": example_id,
        "source_id": f"src-{example_id}",
        "task_family": "qa",
        "input": "What is the capital of France?",
        "expected_output": "Paris",
        "split": split,
    }
    payload.update(overrides)
    return DatasetExample(**payload)


def _manifest(examples, version: str = "0.1.0", **overrides) -> DatasetManifest:
    payload = {
        "dataset_id": "toy",
        "version": version,
        "created_at": FIXED_CLOCK,
        "example_count": len(examples),
        "content_id": content_id(examples),
        "eval_id": eval_id(examples),
        "hashes": {EXAMPLES_FILE: sha256_bytes(render_jsonl(examples))},
    }
    payload.update(overrides)
    return DatasetManifest(**payload)


def _build(tmp_path: Path, **kwargs) -> tuple:
    """Build and write a qa_toy release under ``tmp_path``; return (report, dir)."""
    out = tmp_path / kwargs.pop("name", "rel")
    report = build_release(
        [QA_TOY],
        dataset_id="qa_toy",
        created_at=FIXED_CLOCK,
        **{"version": "0.1.0", **kwargs},
    )
    write_release(report, out)
    return report, out


@pytest.fixture(scope="module")
def qa_examples() -> list[DatasetExample]:
    return loader.load_examples(QA_TOY)


# --------------------------------------------------------------------------- #
# Canonicalization and hashing
# --------------------------------------------------------------------------- #


def test_canonical_bytes_ignores_key_order() -> None:
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})


def test_canonical_bytes_encodes_text_not_escapes() -> None:
    """``ensure_ascii=False``: the digest is of the text, not of one escaping of it."""
    assert canonical_bytes({"x": "café"}) == b'{"x":"caf\xc3\xa9"}'


def test_canonical_bytes_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_bytes({"x": float("nan")})


def test_sha256_file_matches_sha256_bytes(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello\n")
    assert sha256_file(path) == sha256_bytes(b"hello\n")


def test_rendered_jsonl_is_utf8_and_lf_terminated() -> None:
    data = render_jsonl([_example("e1", input="café"), _example("e2")])
    assert data.endswith(b"\n") and b"\r" not in data
    assert "café".encode() in data
    assert data.decode("utf-8").count("\n") == 2


# --------------------------------------------------------------------------- #
# Identity: content_id and eval_id
# --------------------------------------------------------------------------- #


def test_example_digest_ignores_split() -> None:
    assert example_digest(_example(split="train")) == example_digest(_example(split="test"))


def test_example_digest_sees_content() -> None:
    assert example_digest(_example()) != example_digest(_example(expected_output="Lyon"))


def test_content_id_ignores_example_order() -> None:
    a, b = _example("e1"), _example("e2")
    assert content_id([a, b]) == content_id([b, a])


def test_content_id_ignores_splits() -> None:
    assert content_id([_example(split="train")]) == content_id([_example(split="test")])


def test_content_id_counts_multiplicity() -> None:
    """A corpus holding an example twice is not the corpus holding it once."""
    one = _example("e1")
    assert content_id([one]) != content_id([one, one])


def test_eval_id_sees_splits() -> None:
    assert eval_id([_example(split="train")]) != eval_id([_example(split="test")])


def test_eval_id_ignores_example_order() -> None:
    a, b = _example("e1", split="train"), _example("e2", split="test")
    assert eval_id([a, b]) == eval_id([b, a])


def test_ids_are_domain_separated() -> None:
    """A content id can never be read as an eval id, even on a corpus of one train example."""
    examples = [_example(split="train")]
    assert content_id(examples) != eval_id(examples)


def test_ids_are_prefixed_and_schema_bound() -> None:
    assert content_id([_example()]).startswith("sha256:")
    assert SCHEMA_VERSION  # the aggregate binds it; a bump changes every id by construction


def test_duplicate_example_ids_make_identity_undefined() -> None:
    from eval_dataset_forge.datasets.versioning import digests_by_id

    with pytest.raises(ValueError, match="duplicate example_id"):
        digests_by_id([_example("e1"), _example("e1")])


# --------------------------------------------------------------------------- #
# The headline: same content, different eval, different leaderboard
# --------------------------------------------------------------------------- #


def test_reseeding_the_split_holds_content_and_moves_eval(qa_examples) -> None:
    from eval_dataset_forge.splits import apply_splits, build_splits

    dedup = find_duplicates(qa_examples)

    def split_at(seed: int):
        return apply_splits(
            qa_examples, build_splits(qa_examples, seed=seed, dedup_report=dedup).assignment
        )

    first, second = split_at(0), split_at(5)
    assert content_id(first) == content_id(second), "the dataset did not change"
    assert eval_id(first) != eval_id(second), "the arrangement did"


def test_reseeding_the_split_moves_the_reported_number(qa_examples) -> None:
    """The measurement that forces ``eval_id`` to exist.

    The score vector is fixed — one value per example, never re-drawn — so any movement in
    the test-set mean is caused by the split alone. If this ever stops holding, the release
    builder is free to identify a release by its content and ``eval_id`` is dead weight.
    """
    from eval_dataset_forge.splits import build_splits

    dedup = find_duplicates(qa_examples)
    # A deterministic per-example score with no relationship to the split.
    score = {ex.example_id: (index % 3 == 0) for index, ex in enumerate(qa_examples)}

    means = set()
    for seed in range(8):
        report = build_splits(qa_examples, seed=seed, dedup_report=dedup)
        test_ids = [e for e, s in report.assignment.items() if s == str(Split.TEST)]
        means.add(sum(score[e] for e in test_ids) / len(test_ids))
    assert len(means) > 1, "a fixed system's reported accuracy must depend on the split seed"


# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("previous", "bump", "expected"),
    [
        ("0.1.0", "none", "0.1.0"),
        ("0.1.0", "patch", "0.1.1"),
        ("0.1.0", "minor", "0.2.0"),
        ("0.1.0", "major", "1.0.0"),
        ("1.4.9", "minor", "1.5.0"),
        ("2.0.0", "patch", "2.0.1"),
    ],
)
def test_next_version(previous: str, bump: str, expected: str) -> None:
    assert next_version(previous, bump) == expected


def test_next_version_has_no_zero_x_special_case() -> None:
    """A major change below 1.0 still bumps the major. Documented, deliberate."""
    assert next_version("0.9.9", "major") == "1.0.0"


def test_parse_version_rejects_non_semver() -> None:
    for bad in ("1.0", "v1.0.0", "1.0.0-rc1", ""):
        with pytest.raises(ValueError):
            parse_version(bad)


def test_next_version_rejects_unknown_bump() -> None:
    with pytest.raises(ValueError, match="unknown bump"):
        next_version("0.1.0", "huge")


# --------------------------------------------------------------------------- #
# Bump inference — every level, each forced by the change it names
# --------------------------------------------------------------------------- #


def test_bump_none_when_only_the_clock_moved() -> None:
    examples = [_example("e1", split="train"), _example("e2", split="test")]
    previous = _manifest(examples)
    current = _manifest(examples, created_at="2099-12-31T23:59:59+00:00")
    delta = compare(previous, examples, current, examples)
    assert delta.bump == "none"
    assert delta.recommended_version == "0.1.0"
    assert delta.scores_comparable


def test_bump_patch_when_only_packaging_moved() -> None:
    examples = [_example("e1", split="train"), _example("e2", split="test")]
    previous = _manifest(examples)
    current = _manifest(examples, license_notes="CC-BY-4.0")
    delta = compare(previous, examples, current, examples)
    assert delta.bump == "patch"
    assert delta.recommended_version == "0.1.1"
    assert "license_notes" in delta.reasons[0]
    assert delta.scores_comparable


def test_bump_minor_when_a_train_example_is_added() -> None:
    previous_examples = [_example("e1", split="train"), _example("e2", split="test")]
    current_examples = [*previous_examples, _example("e3", split="train")]
    delta = compare(
        _manifest(previous_examples),
        previous_examples,
        _manifest(current_examples),
        current_examples,
    )
    assert delta.bump == "minor"
    assert delta.recommended_version == "0.2.0"
    assert delta.added == ("e3",)
    assert delta.eval_affected == ()
    assert delta.scores_comparable, "a test score measured before the addition still stands"


def test_bump_minor_warns_about_leakage_it_did_not_check_for() -> None:
    previous_examples = [_example("e1", split="train"), _example("e2", split="test")]
    current_examples = [*previous_examples, _example("e3", split="train")]
    delta = compare(
        _manifest(previous_examples),
        previous_examples,
        _manifest(current_examples),
        current_examples,
    )
    assert any("leakage" in warning for warning in delta.warnings)


def test_bump_major_when_a_test_example_is_edited() -> None:
    previous_examples = [_example("e1", split="train"), _example("e2", split="test")]
    current_examples = [
        _example("e1", split="train"),
        _example("e2", split="test", expected_output="Lyon"),
    ]
    delta = compare(
        _manifest(previous_examples),
        previous_examples,
        _manifest(current_examples),
        current_examples,
    )
    assert delta.bump == "major"
    assert delta.changed == ("e2",)
    assert delta.eval_affected == ("e2",)
    assert not delta.scores_comparable


def test_removing_a_train_example_is_minor() -> None:
    """Removal outside the evaluation splits is a ``minor``. The rule is about scores."""
    previous_examples = [_example("e1", split="train"), _example("e2", split="test")]
    current_examples = [_example("e2", split="test")]
    delta = compare(
        _manifest(previous_examples),
        previous_examples,
        _manifest(current_examples),
        current_examples,
    )
    assert delta.bump == "minor"
    assert delta.removed == ("e1",)


def test_bump_major_when_a_test_example_is_removed() -> None:
    previous_examples = [_example("e1", split="train"), _example("e2", split="test")]
    current_examples = [_example("e1", split="train")]
    delta = compare(
        _manifest(previous_examples),
        previous_examples,
        _manifest(current_examples),
        current_examples,
    )
    assert delta.bump == "major"
    assert delta.removed == ("e2",)


def test_bump_major_on_a_pure_resplit() -> None:
    """No content changed. ``content_id`` agrees. The scores still do not carry over."""
    previous_examples = [_example("e1", split="train"), _example("e2", split="test")]
    current_examples = [_example("e1", split="test"), _example("e2", split="train")]
    delta = compare(
        _manifest(previous_examples),
        previous_examples,
        _manifest(current_examples),
        current_examples,
    )
    assert delta.bump == "major"
    assert delta.identity["content_id_equal"] is True
    assert delta.identity["eval_id_equal"] is False
    assert set(delta.reassigned) == {"e1", "e2"}
    assert not delta.scores_comparable
    assert any("re-split" in warning for warning in delta.warnings)


def test_bump_major_on_a_schema_change() -> None:
    examples = [_example("e1", split="test")]
    previous = _manifest(examples, schema_version="0.0.9")
    current = _manifest(examples)
    delta = compare(previous, examples, current, examples)
    assert delta.bump == "major"
    assert "schema_version" in delta.reasons[0]


def test_train_to_train_reassignment_is_impossible_but_dev_to_test_is_major() -> None:
    previous_examples = [_example("e1", split="dev")]
    current_examples = [_example("e1", split="test")]
    delta = compare(
        _manifest(previous_examples),
        previous_examples,
        _manifest(current_examples),
        current_examples,
    )
    assert delta.bump == "major"
    assert str(Split.DEV) in EVAL_SPLITS and str(Split.TEST) in EVAL_SPLITS


def test_every_bump_level_is_reachable() -> None:
    """A guard against a rule that can only ever say ``major``."""
    reached = set()
    base = [_example("e1", split="train"), _example("e2", split="test")]
    reached.add(compare(_manifest(base), base, _manifest(base), base).bump)
    reached.add(
        compare(_manifest(base), base, _manifest(base, license_notes="x"), base).bump
    )
    grown = [*base, _example("e3", split="train")]
    reached.add(compare(_manifest(base), base, _manifest(grown), grown).bump)
    resplit = [_example("e1", split="test"), _example("e2", split="train")]
    reached.add(compare(_manifest(base), base, _manifest(resplit), resplit).bump)
    assert reached == set(BUMP_LEVELS)


def test_format_delta_names_the_bump() -> None:
    examples = [_example("e1", split="test")]
    delta = compare(_manifest(examples), examples, _manifest(examples), examples)
    rendered = format_delta(delta)
    assert "none" in rendered and "content_id" in rendered


# --------------------------------------------------------------------------- #
# Exclusions
# --------------------------------------------------------------------------- #


def test_plan_exclusions_drops_nothing_by_default(qa_examples) -> None:
    dedup = find_duplicates(qa_examples)
    labels = audit_labels(qa_examples)
    assert plan_exclusions(qa_examples, dedup, labels, drop=()) == ()


def test_plan_exclusions_keeps_the_first_member_of_each_cluster(qa_examples) -> None:
    dedup = find_duplicates(qa_examples)
    labels = audit_labels(qa_examples)
    exclusions = plan_exclusions(qa_examples, dedup, labels, drop=["exact"])
    assert exclusions, "qa_toy plants an exact duplicate"
    order = [example.example_id for example in qa_examples]
    for item in exclusions:
        assert item.kept_id is not None
        assert order.index(item.kept_id) < order.index(item.example_id)


def test_plan_exclusions_rejects_unknown_targets(qa_examples) -> None:
    dedup = find_duplicates(qa_examples)
    labels = audit_labels(qa_examples)
    with pytest.raises(ValueError, match="unknown drop target"):
        plan_exclusions(qa_examples, dedup, labels, drop=["everything"])


def test_drop_choices_are_all_plannable(qa_examples) -> None:
    dedup = find_duplicates(qa_examples)
    labels = audit_labels(qa_examples)
    plan_exclusions(qa_examples, dedup, labels, drop=list(DROP_CHOICES))


def test_a_release_that_drops_everything_is_refused(tmp_path: Path) -> None:
    """One example, no gold answer, ``--drop blocking-labels``. An empty release is a bug,
    not a release, and the builder says so instead of writing an empty ``examples.jsonl``."""
    source = tmp_path / "all_blocking.jsonl"
    source.write_text(
        json.dumps(
            {
                "example_id": "e1",
                "source_id": "s1",
                "task_family": "qa",
                "input": "What is the capital of France?",
                "expected_output": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no release to build"):
        build_release(
            [source], dataset_id="toy", created_at=FIXED_CLOCK, drop=["blocking-labels"]
        )


def test_dropping_a_duplicate_is_recorded_in_the_manifest(tmp_path: Path) -> None:
    report, _ = _build(tmp_path, drop=["exact"])
    manifest = report.manifest
    assert manifest.excluded, "an exact duplicate was dropped"
    for entry in manifest.excluded:
        assert entry["kept_id"] and entry["reason"].startswith("exact duplicate of")
    assert manifest.source_content_id != manifest.content_id
    assert manifest.example_count == len(report.release.examples)


def test_the_audit_describes_the_data_that_shipped(tmp_path: Path) -> None:
    """Dropping an exact duplicate dissolves its cluster; the release's audit must agree."""
    before, _ = _build(tmp_path, name="before")
    after, _ = _build(tmp_path, name="after", drop=["exact"])
    assert before.audit.summary()["exact_duplicates"] > after.audit.summary()["exact_duplicates"]
    assert after.audit.summary()["exact_duplicates"] == 0
    shipped = {example.example_id for example in after.release.examples}
    for finding in after.audit.exact_duplicates + after.audit.near_duplicates:
        assert set(finding.example_ids) <= shipped


# --------------------------------------------------------------------------- #
# Build, write, verify
# --------------------------------------------------------------------------- #


def test_release_layout(tmp_path: Path) -> None:
    _, out = _build(tmp_path)
    for name in (EXAMPLES_FILE, MANIFEST_FILE, AUDIT_FILE, REPRODUCE_FILE):
        assert (out / name).is_file()
    assert (out / "splits" / "train.jsonl").is_file()
    assert (out / "splits" / "test.jsonl").is_file()


def test_manifest_does_not_hash_itself(tmp_path: Path) -> None:
    report, _ = _build(tmp_path)
    assert MANIFEST_FILE not in report.manifest.hashes
    assert EXAMPLES_FILE in report.manifest.hashes


def test_manifest_records_block_structure_and_no_effective_n(tmp_path: Path) -> None:
    report, _ = _build(tmp_path)
    dependence = report.manifest.dependence
    assert dependence["block_count"] >= 1
    assert dependence["block_size_histogram"]
    assert "effective_n" not in dependence, "it needs an ICC, which needs system scores"


def test_manifest_records_tool_and_split_spec(tmp_path: Path) -> None:
    report, _ = _build(tmp_path, seed=3, strategy="stratified")
    assert report.manifest.tool_version == __version__
    assert report.manifest.split_spec is not None
    assert report.manifest.split_spec.seed == 3
    assert str(report.manifest.split_spec.strategy) == "stratified"


def test_verify_release_passes_on_a_clean_release(tmp_path: Path) -> None:
    _, out = _build(tmp_path)
    report = verify_release(out)
    assert report.ok, report.problems
    assert all(report.checks.values())
    assert "integrity, not authenticity" in format_verify(report)


def test_verify_release_catches_a_single_flipped_byte(tmp_path: Path) -> None:
    _, out = _build(tmp_path)
    path = out / EXAMPLES_FILE
    path.write_text(path.read_text(encoding="utf-8").replace("Paris", "Lyon", 1), encoding="utf-8")

    report = verify_release(out)
    assert not report.ok
    assert report.checks["hashes_match"] is False
    assert report.checks["content_id_matches"] is False
    assert report.checks["eval_id_matches"] is False


def test_verify_release_catches_a_split_file_that_no_longer_partitions(tmp_path: Path) -> None:
    _, out = _build(tmp_path)
    train = out / "splits" / "train.jsonl"
    lines = train.read_text(encoding="utf-8").splitlines(keepends=True)
    train.write_text("".join(lines[1:]), encoding="utf-8")

    report = verify_release(out)
    assert not report.ok
    assert report.checks["splits_partition_examples"] is False
    assert any("appear in no split file" in problem for problem in report.problems)


def test_verify_release_catches_a_doctored_example_count(tmp_path: Path) -> None:
    _, out = _build(tmp_path)
    path = out / MANIFEST_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["example_count"] += 1
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = verify_release(out)
    assert not report.ok
    assert report.checks["example_count_matches"] is False


def test_verify_release_reports_a_missing_manifest(tmp_path: Path) -> None:
    report = verify_release(tmp_path / "nowhere")
    assert not report.ok
    assert report.checks == {"manifest_parses": False}


def test_verify_release_cannot_detect_a_rewritten_manifest(tmp_path: Path) -> None:
    """The documented hole. ``manifest.json`` is not self-hashed, so a consistent forgery
    passes every check. This asserts the limitation rather than leaving it implied."""
    _, out = _build(tmp_path)
    examples = loader.load_examples(out / EXAMPLES_FILE)
    forged = load_manifest(out / MANIFEST_FILE).model_copy(
        update={
            "license_notes": "Public domain — forged",
            "created_at": "1970-01-01T00:00:00+00:00",
        }
    )
    (out / MANIFEST_FILE).write_text(
        json.dumps(forged.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    report = verify_release(out)
    assert report.ok, "integrity checks cannot see a manifest rewrite"
    assert report.manifest is not None
    assert report.manifest.license_notes.endswith("forged")
    assert content_id(examples) == report.manifest.content_id


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def _tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_two_builds_at_a_fixed_clock_are_byte_identical(tmp_path: Path) -> None:
    _, first = _build(tmp_path, name="one")
    _, second = _build(tmp_path, name="two")
    assert _tree(first) == _tree(second)


def test_rebuild_is_byte_identical_across_processes_and_hash_seeds(tmp_path: Path) -> None:
    """The in-process check above cannot see a ``set`` leaking into a hashed artifact.

    Within one interpreter a ``set[str]`` iterates in a fixed order, so a manifest field
    built from one looks deterministic and stops being so on the next machine. Two separate
    interpreters at different ``PYTHONHASHSEED`` values are the only way to catch it.
    """
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    outs = []
    for index, hashseed in enumerate(("1", "12345")):
        out = tmp_path / f"proc{index}"
        result = subprocess.run(
            [
                sys.executable, "-m", "eval_dataset_forge", "release",
                "--dataset", str(QA_TOY),
                "--dataset-id", "qa_toy",
                "--version", "0.1.0",
                "--created-at", FIXED_CLOCK,
                "--out", str(out),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(root / "src")},
        )
        assert result.returncode == 0, result.stderr
        outs.append(out)
    assert _tree(outs[0]) == _tree(outs[1])


def test_only_the_manifest_moves_when_the_clock_moves(tmp_path: Path) -> None:
    """``created_at`` is provenance. It must not reach a hashed artifact."""
    _, first = _build(tmp_path, name="one")
    later = build_release(
        [QA_TOY], dataset_id="qa_toy", version="0.1.0", created_at="2099-06-06T12:00:00+00:00"
    )
    second = tmp_path / "two"
    write_release(later, second)

    before, after = _tree(first), _tree(second)
    assert set(before) == set(after)
    differing = {name for name in before if before[name] != after[name]}
    assert differing == {MANIFEST_FILE}


def test_reproduce_md_carries_no_timestamp(tmp_path: Path) -> None:
    report, out = _build(tmp_path)
    text = (out / REPRODUCE_FILE).read_text(encoding="utf-8")
    assert FIXED_CLOCK not in text
    assert report.manifest.content_id in text
    assert report.manifest.eval_id in text
    assert "--version 0.1.0" in text and "--out datasets/releases/qa_toy_v0.1.0" in text


def test_build_release_needs_a_source_file() -> None:
    with pytest.raises(ValueError, match="at least one source file"):
        build_release([], dataset_id="x", created_at=FIXED_CLOCK)


def test_auto_version_needs_a_previous_release() -> None:
    with pytest.raises(ValueError, match="needs --previous"):
        build_release([QA_TOY], dataset_id="qa_toy", version="auto", created_at=FIXED_CLOCK)


def test_explicit_version_below_the_required_bump_is_refused(tmp_path: Path) -> None:
    _, previous = _build(tmp_path)
    with pytest.raises(ValueError, match="below the major bump"):
        build_release(
            [QA_TOY],
            dataset_id="qa_toy",
            version="0.1.1",
            seed=7,
            created_at=FIXED_CLOCK,
            previous=previous,
        )


def test_auto_version_derives_the_bump(tmp_path: Path) -> None:
    _, previous = _build(tmp_path)
    rebuilt = build_release(
        [QA_TOY], dataset_id="qa_toy", version="auto", created_at=FIXED_CLOCK, previous=previous
    )
    assert rebuilt.delta is not None and rebuilt.delta.bump == "none"
    assert rebuilt.manifest.version == "0.1.0"

    reseeded = build_release(
        [QA_TOY],
        dataset_id="qa_toy",
        version="auto",
        seed=7,
        created_at=FIXED_CLOCK,
        previous=previous,
    )
    assert reseeded.delta is not None and reseeded.delta.bump == "major"
    assert reseeded.manifest.version == "1.0.0"
    assert reseeded.delta.identity["content_id_equal"] is True


def test_a_rebuild_with_a_new_clock_is_not_a_new_release(tmp_path: Path) -> None:
    _, previous = _build(tmp_path)
    rebuilt = build_release(
        [QA_TOY],
        dataset_id="qa_toy",
        version="auto",
        created_at="2099-06-06T12:00:00+00:00",
        previous=previous,
    )
    assert rebuilt.delta is not None and rebuilt.delta.bump == "none"


# --------------------------------------------------------------------------- #
# Manifest assembly
# --------------------------------------------------------------------------- #


def test_build_manifest_hashes_the_bytes_it_is_given() -> None:
    examples = [_example("e1", split="train")]
    artifacts = {
        EXAMPLES_FILE: render_jsonl(examples),
        "splits/train.jsonl": render_jsonl(examples),
    }
    manifest = build_manifest(
        dataset_id="toy",
        version="0.1.0",
        created_at=FIXED_CLOCK,
        examples=examples,
        artifacts=artifacts,
    )
    assert manifest.hashes[EXAMPLES_FILE] == sha256_bytes(artifacts[EXAMPLES_FILE])
    assert manifest.splits["train"]["count"] == 1
    assert manifest.splits["train"]["sha256"] == manifest.hashes["splits/train.jsonl"]
    assert manifest.source_content_id == manifest.content_id


def test_old_manifests_still_load() -> None:
    """The Milestone 10 fields are additive with defaults; SCHEMA_VERSION did not move."""
    manifest = DatasetManifest(
        dataset_id="toy", version="0.1.0", created_at=FIXED_CLOCK, example_count=0
    )
    assert manifest.content_id == "" and manifest.eval_id == ""
    assert manifest.split_spec is None and manifest.excluded == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_release_and_verify_round_trip(tmp_path: Path, capsys) -> None:
    out = tmp_path / "qa_toy_v0.1.0"
    code = main(
        [
            "release",
            "--dataset", str(QA_TOY),
            "--dataset-id", "qa_toy",
            "--version", "0.1.0",
            "--created-at", FIXED_CLOCK,
            "--out", str(out),
        ]
    )
    assert code == 0
    assert "content_id" in capsys.readouterr().out

    assert main(["verify-release", "--release", str(out)]) == 0
    assert "hashes_match" in capsys.readouterr().out


def test_cli_verify_release_exits_one_on_a_bad_release(tmp_path: Path, capsys) -> None:
    _, out = _build(tmp_path)
    (out / EXAMPLES_FILE).write_text("", encoding="utf-8")
    assert main(["verify-release", "--release", str(out)]) == 1
    assert "problems" in capsys.readouterr().out


def test_cli_release_refuses_a_stale_version(tmp_path: Path, capsys) -> None:
    _, previous = _build(tmp_path)
    code = main(
        [
            "release",
            "--dataset", str(QA_TOY),
            "--dataset-id", "qa_toy",
            "--version", "0.1.0",
            "--seed", "7",
            "--created-at", FIXED_CLOCK,
            "--previous", str(previous),
            "--out", str(tmp_path / "next"),
        ]
    )
    assert code == 2
    assert "below the major bump" in capsys.readouterr().err


def test_cli_release_json_is_machine_readable(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "release",
            "--dataset", str(QA_TOY),
            "--dataset-id", "qa_toy",
            "--version", "0.1.0",
            "--created-at", FIXED_CLOCK,
            "--json",
            "--out", str(tmp_path / "rel"),
        ]
    )
    assert code == 0
    payload = json.JSONDecoder().raw_decode(capsys.readouterr().out)[0]
    assert payload["manifest"]["content_id"].startswith("sha256:")
    assert payload["counts"]["examples"] > 0


def test_format_report_names_both_identities(tmp_path: Path) -> None:
    report, _ = _build(tmp_path)
    rendered = format_report(report)
    assert "content_id" in rendered and "eval_id" in rendered
    assert "blocks" in rendered


def test_exclusion_to_dict_is_json_safe() -> None:
    item = Exclusion(example_id="e1", reason="exact duplicate of e0", kept_id="e0")
    assert json.loads(json.dumps(item.to_dict()))["kept_id"] == "e0"
