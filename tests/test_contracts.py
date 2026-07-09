"""Tests for the integration contracts (Milestone 13).

Covered here:

  * **The declared read set is real.** Every key :mod:`~eval_dataset_forge.contracts.reader`
    says it reads exists in the committed demo release and on the model that defines it.
    This is the check that would have caught Milestone 12's silent growth of
    ``split_statistics`` had the growth been a removal instead.
  * **Three verdicts, not two.** ``UNDECIDABLE`` never passes a gate unless the policy says
    so in writing, and the two requirements that reach it do so for two different reasons —
    one about the corpus (``no_retrieval_contamination`` on data with no passages), one
    about the release format (``difficulty_is_calibrated``, always).
  * **The optional accelerator changes the answer.** Rebuilding ``bio_toy`` with
    ``sklearn_available()`` forced false yields the same ``content_id`` and a *different*
    ``eval_id``, different split sizes, and a different ``examples.jsonl`` digest. The
    "optional accelerator" of ``docs/PLAN.md`` §3 is a different detector, not a faster one.
  * **Verification is environment-independent; reproduction is not.** The same monkeypatch
    that moves ``eval_id`` leaves ``verify-release`` at 10/10 on the committed release,
    because verification recomputes hashes and never runs a detector.
  * **A gate verdict is not preserved under union.** ``safety_toy`` alone passes a
    one-cluster near-duplicate budget; concatenated with four fixtures that share no
    duplicate with it, the same ten rows fail. ``GateReport.applies_to`` refuses to carry.
  * **Blocks are not additive; examples are.** 45 blocks apart, 44 pooled, under ``tfidf``.
    Under the ``jaccard`` fallback both are 46 — the property holds exactly when the
    similarity is corpus-independent, which is the mechanism, stated as a test.

Everything is CPU-only and offline. The sklearn-absent path is exercised by monkeypatching
:func:`~eval_dataset_forge.dedup.near_duplicate.sklearn_available`, never by uninstalling
anything, so the suite passes with and without the optional accelerator present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_dataset_forge.contracts import (
    AUDIT_FIELDS,
    CONTRACT_VERSION,
    DEFAULT_THRESHOLDS,
    DEPENDENCE_FIELDS,
    MANIFEST_FIELDS,
    PROFILES,
    REQUIREMENTS,
    SPLIT_COUNT_FIELDS,
    SPLIT_STAT_FIELDS,
    ContractViolation,
    GatePolicy,
    ReleaseHandle,
    Verdict,
    evaluate,
    evaluate_gate,
    format_gate,
    gate_release,
    get_profile,
    resolve_thresholds,
)
from eval_dataset_forge.contracts.report import gate_document
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.datasets.manifest import verify_release
from eval_dataset_forge.datasets.release import build_release, write_release
from eval_dataset_forge.dedup import find_duplicates
from eval_dataset_forge.schemas import AuditResult, DatasetManifest
from eval_dataset_forge.splits.blocks import duplicate_blocks

DEMO = Path("datasets/releases/qa_toy_v0.1.0")
FIXTURES = Path("datasets/fixtures")
FIXTURE_NAMES = ("agent_tasks_toy", "bio_toy", "qa_toy", "retrieval_toy", "safety_toy")
FIXED_CLOCK = "2024-01-01T00:00:00+00:00"


def _examples(name: str):
    return loader.load_examples(FIXTURES / name / "examples.jsonl")


def _no_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the TF-IDF accelerator look absent, everywhere it is consulted."""
    monkeypatch.setattr(
        "eval_dataset_forge.dedup.near_duplicate.sklearn_available", lambda: False
    )


def _build(tmp_path: Path, sources, dataset_id: str, **kwargs):
    out = tmp_path / dataset_id
    report = build_release(
        [FIXTURES / name / "examples.jsonl" for name in sources],
        dataset_id=dataset_id,
        created_at=FIXED_CLOCK,
        **{
            "version": "0.1.0",
            "strategy": "adversarial",
            "seed": 0,
            "ratios": {"train": 0.7, "dev": 0.1, "test": 0.2},
            **kwargs,
        },
    )
    write_release(report, out)
    return report, out


@pytest.fixture(scope="module")
def demo() -> ReleaseHandle:
    return ReleaseHandle.open(DEMO)


# --------------------------------------------------------------------------- #
# The contract surface
# --------------------------------------------------------------------------- #


def test_every_declared_manifest_field_exists_on_the_model() -> None:
    """A rename in ``schemas.py`` must break this, not a consumer in production."""
    for name in MANIFEST_FIELDS:
        assert name in DatasetManifest.model_fields, name


def test_every_declared_audit_field_exists_on_the_model() -> None:
    for name in AUDIT_FIELDS:
        assert name in AuditResult.model_fields, name


def test_every_declared_field_exists_in_the_committed_release() -> None:
    """The model and the artifact are two places a key can go missing. Check both."""
    manifest = json.loads((DEMO / "manifest.json").read_text())
    audit = json.loads((DEMO / "audit.json").read_text())
    for name in MANIFEST_FIELDS:
        assert name in manifest, f"manifest.json is missing {name}"
    for name in AUDIT_FIELDS:
        assert name in audit, f"audit.json is missing {name}"
    for name in DEPENDENCE_FIELDS:
        assert name in manifest["dependence"], f"manifest.dependence is missing {name}"
    for name in SPLIT_STAT_FIELDS:
        assert name in audit["split_statistics"], f"split_statistics is missing {name}"
    for name in SPLIT_COUNT_FIELDS:
        assert name in audit["split_statistics"]["counts"], f"counts is missing {name}"


def test_additive_keys_do_not_break_a_consumer() -> None:
    """The compatibility rule, stated as a test: reading is by name, never by shape."""
    manifest = json.loads((DEMO / "manifest.json").read_text())
    unread = set(manifest) - set(MANIFEST_FIELDS)
    assert unread, "the demo manifest carries keys the contract does not read; that is fine"
    handle = ReleaseHandle.open(DEMO)
    assert handle.identity["content_id"] == manifest["content_id"]


def test_opening_a_non_release_directory_raises_contract_violation(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation):
        ReleaseHandle.open(tmp_path / "nowhere")


def test_open_verifies_by_default(tmp_path: Path) -> None:
    _, out = _build(tmp_path, ["qa_toy"], "qa_toy")
    (out / "examples.jsonl").write_bytes(b'{"corrupt": true}\n')
    with pytest.raises(ContractViolation, match="failed verification"):
        ReleaseHandle.open(out)
    # …and a caller who asks not to verify gets a handle on the broken release,
    # so the `integrity` requirement can report on it instead of raising.
    assert ReleaseHandle.open(out, verify=False).manifest.dataset_id == "qa_toy"


# --------------------------------------------------------------------------- #
# Three verdicts
# --------------------------------------------------------------------------- #


def test_undecidable_is_not_a_pass(demo: ReleaseHandle) -> None:
    policy = GatePolicy(
        profile="test", requirements=["no_retrieval_contamination"], thresholds={}
    )
    report = evaluate_gate(demo, policy)
    assert report.undecided and not report.failed
    assert not report.ok
    assert report.exit_code == 1


def test_allow_undecidable_is_an_explicit_written_decision(demo: ReleaseHandle) -> None:
    policy = GatePolicy(
        profile="test",
        requirements=["no_retrieval_contamination"],
        allow_undecidable=True,
    )
    report = evaluate_gate(demo, policy)
    assert report.undecided and report.ok and report.exit_code == 0


def test_retrieval_check_is_undecidable_without_passages_and_decidable_with_them(
    tmp_path: Path,
) -> None:
    """Two corpora, one requirement, two verdicts — and neither of them is a FAIL.

    ``qa_toy`` carries no ``metadata.retrieval.docs``, so a zero finding count means the
    check never ran. ``retrieval_toy`` carries passages, so its findings are informative.
    """
    qa = ReleaseHandle.open(_build(tmp_path, ["qa_toy"], "qa")[1])
    assert qa.examples_with_retrieval_docs() == 0
    verdict = evaluate("no_retrieval_contamination", qa, DEFAULT_THRESHOLDS)
    assert verdict.verdict is Verdict.UNDECIDABLE

    ret = ReleaseHandle.open(_build(tmp_path, ["retrieval_toy"], "ret")[1])
    assert ret.examples_with_retrieval_docs() > 0
    verdict = evaluate("no_retrieval_contamination", ret, DEFAULT_THRESHOLDS)
    assert verdict.verdict is not Verdict.UNDECIDABLE


def test_difficulty_calibration_is_undecidable_from_any_release(demo: ReleaseHandle) -> None:
    """A gap in the release format, made executable.

    Milestone 8 runs the permutation test. Milestone 10 renders its result into an English
    recommendation. ``audit.json`` keeps three bucket counts and drops the verdict. When a
    machine-readable field is added, this test fails — which is the point of writing it.
    """
    result = evaluate("difficulty_is_calibrated", demo, DEFAULT_THRESHOLDS)
    assert result.verdict is Verdict.UNDECIDABLE
    assert demo.audit.difficulty_distribution  # the buckets exist…
    calibration_keys = set(demo.audit.model_dump()) & {"difficulty_calibration", "calibration"}
    assert not calibration_keys, "audit.json gained a calibration field; wire it up"
    # The fact survives only as prose, which is not a contract.
    assert any("random ranking" in line for line in demo.audit.recommendations)


def test_no_profile_includes_a_permanently_undecidable_requirement() -> None:
    """A profile that can never pass is a broken gate, not a strict one."""
    for name, profile in PROFILES.items():
        assert "difficulty_is_calibrated" not in profile.requirements, name


# --------------------------------------------------------------------------- #
# The optional accelerator is a different detector
# --------------------------------------------------------------------------- #


def test_near_method_resolves_on_sklearn_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval_dataset_forge.dedup.near_duplicate import resolve_near_method

    _no_sklearn(monkeypatch)
    assert resolve_near_method("auto") == "jaccard"
    assert resolve_near_method("tfidf") == "jaccard"


def test_an_optional_dependency_changes_eval_id_not_content_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline. Same bytes, same command, same seed; different benchmark.

    ``bio_toy``'s ``bio-001``/``bio-002`` pair clears 0.30 under idf weighting and not under
    raw set overlap. That edge is a duplicate block; the adversarial split will not cut
    through a block; ``eval_id`` folds the split in. So an "optional accelerator" decides
    which examples are held out.
    """
    with_sklearn, dir_a = _build(tmp_path, ["bio_toy"], "bio_a")
    _no_sklearn(monkeypatch)
    without_sklearn, dir_b = _build(tmp_path, ["bio_toy"], "bio_b")

    a, b = with_sklearn.release.manifest, without_sklearn.release.manifest
    assert a.dependence["near_method"] == "tfidf"
    assert b.dependence["near_method"] == "jaccard"

    assert a.content_id == b.content_id, "the examples are the same examples"
    assert a.eval_id != b.eval_id, "and they are not the same benchmark"
    assert a.dependence["block_count"] == 8
    assert b.dependence["block_count"] == 9
    assert a.hashes["examples.jsonl"] != b.hashes["examples.jsonl"]

    def held_out(release_dir: Path) -> set[str]:
        handle = ReleaseHandle.open(release_dir)
        return {example.example_id for example in handle.split("test")}

    # The held-out set itself moves. Whichever of the two a scoreboard was measured on, the
    # other one is a different benchmark wearing the same command.
    assert held_out(dir_a) != held_out(dir_b)


def test_verification_is_environment_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The counterweight to the test above, and the reason the contract is usable.

    ``verify-release`` recomputes file hashes, ``content_id``, and ``eval_id`` from bytes.
    It runs no detector, so a release built with scikit-learn verifies on a machine without
    it. Integrity travels; reproducibility does not.
    """
    before = verify_release(DEMO)
    _no_sklearn(monkeypatch)
    after = verify_release(DEMO)
    assert before.ok and after.ok
    assert before.checks == after.checks
    assert len(after.checks) == 10


def test_the_release_records_the_method_that_actually_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sklearn(monkeypatch)
    _, out = _build(tmp_path, ["bio_toy"], "bio")
    handle = ReleaseHandle.open(out)
    assert handle.near_method == "jaccard"
    assert "`jaccard` (no scikit-learn)" in (out / "REPRODUCE.md").read_text()

    result = evaluate("near_method_recorded", handle, resolve_thresholds({}))
    assert result.verdict is Verdict.PASS
    pinned = resolve_thresholds({"expected_near_method": "tfidf"})
    assert evaluate("near_method_recorded", handle, pinned).verdict is Verdict.FAIL


def test_reproduce_md_no_longer_claims_the_command_is_sufficient() -> None:
    text = (DEMO / "REPRODUCE.md").read_text()
    assert "deterministic given the same inputs and seed." not in text
    assert "The command is not sufficient." in text
    assert "`tfidf` (scikit-learn installed)" in text


def test_rewriting_reproduce_md_is_a_packaging_change_not_a_data_change() -> None:
    """Rebuilt in place at v0.1.0, as Milestone 12's `audit.md` was, and for its reason.

    ``compare()`` classifies a release whose only moved byte is one artifact hash as a
    ``patch``: ``content_id`` and ``eval_id`` hold, no example changed split, and a score
    measured on the old bundle still stands on this one.
    """
    from eval_dataset_forge.datasets.manifest import load_manifest
    from eval_dataset_forge.datasets.versioning import compare

    manifest = load_manifest(DEMO / "manifest.json")
    examples = loader.load_examples(DEMO / "examples.jsonl")
    previous = manifest.model_copy(
        update={"hashes": {**manifest.hashes, "REPRODUCE.md": "sha256:" + "0" * 64}}
    )

    delta = compare(previous, examples, manifest, examples)
    assert delta.bump == "patch"
    assert delta.recommended_version == "0.1.1"
    assert delta.identity["content_id_equal"] and delta.identity["eval_id_equal"]
    assert delta.reasons == ("packaging only: hashes",)


def test_the_committed_release_rebuilds_byte_for_byte(tmp_path: Path) -> None:
    """``REPRODUCE.md`` names a command; the command must produce the release it sits in.

    The data artifacts are compared, not ``manifest.json`` (it embeds ``created_at``),
    ``dataset_card.md`` or ``REPRODUCE.md`` (they embed the curator's prose and the invoking
    command, neither of which any audit stage reads). This is what makes the new
    "Environment" section a claim rather than a decoration: run the recorded command in the
    recorded environment, get these bytes.
    """
    _, out = _build(
        tmp_path,
        ["qa_toy"],
        "qa_toy",
        stratify_by=("task_family",),
        quarantine_blocking=True,
    )
    for name in ("examples.jsonl", "audit.json", "audit.md"):
        assert (out / name).read_bytes() == (DEMO / name).read_bytes(), name
    for split in ("train", "dev", "test"):
        assert (out / "splits" / f"{split}.jsonl").read_bytes() == (
            DEMO / "splits" / f"{split}.jsonl"
        ).read_bytes(), split


def test_the_demo_release_keeps_its_identity_when_rebuilt_without_sklearn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The qa_toy counterpart to the bio_toy divergence, and a sharper claim than REPRODUCE.md's.

    On ``bio_toy`` the two detectors disagree about a duplicate block, so the adversarial split
    moves and ``eval_id`` moves with it (see the test above). On the committed ``qa_toy`` release
    they **agree** about the split — the near cluster ``qa-003 + qa-004 + qa-012`` is one block
    under both ``tfidf`` and ``jaccard`` — so rebuilding the demo without scikit-learn preserves
    ``content_id``, ``eval_id``, and every split file byte-for-byte. The detector's disagreement
    reaches only the artifacts that *name* it (``audit.{json,md}``, ``dataset_card.md``,
    ``REPRODUCE.md``) and the manifest hashes of those.

    ``REPRODUCE.md`` warns that rebuilding without the accelerator "can produce different splits and
    a different ``eval_id``". That "can" is about the general case; this test records that on *this*
    release it does not, which is why ``verify-release`` still reports 10/10 on a no-sklearn rebuild
    even though five tracked files would show as modified.
    """
    from eval_dataset_forge.datasets.manifest import load_manifest

    committed = load_manifest(DEMO / "manifest.json")

    with_sklearn, dir_a = _build(
        tmp_path, ["qa_toy"], "qa_a", stratify_by=("task_family",), quarantine_blocking=True
    )
    _no_sklearn(monkeypatch)
    without_sklearn, dir_b = _build(
        tmp_path, ["qa_toy"], "qa_b", stratify_by=("task_family",), quarantine_blocking=True
    )

    a, b = with_sklearn.release.manifest, without_sklearn.release.manifest
    assert a.dependence["near_method"] == "tfidf"
    assert b.dependence["near_method"] == "jaccard"

    # Identity and data are environment-independent here — anchored to the committed release.
    assert a.content_id == b.content_id == committed.content_id
    assert a.eval_id == b.eval_id == committed.eval_id
    assert a.dependence["block_count"] == b.dependence["block_count"]
    assert (dir_a / "examples.jsonl").read_bytes() == (dir_b / "examples.jsonl").read_bytes()
    for split in ("train", "dev", "test"):
        assert (dir_a / "splits" / f"{split}.jsonl").read_bytes() == (
            dir_b / "splits" / f"{split}.jsonl"
        ).read_bytes(), split

    # What does move: exactly the artifacts that record which detector ran.
    assert a.hashes["audit.json"] != b.hashes["audit.json"]
    assert (dir_a / "audit.md").read_bytes() != (dir_b / "audit.md").read_bytes()


# --------------------------------------------------------------------------- #
# Composability
# --------------------------------------------------------------------------- #


def test_blocks_are_not_additive_under_tfidf() -> None:
    """56 examples either way; 45 independent blocks apart, 44 pooled.

    The lost block is ``{saf-004, saf-005, saf-007}``, entirely inside ``safety_toy``, and
    the pooled corpus contains **no** cross-fixture duplicate edge. Merging five disjoint
    datasets changed one dataset's internal verdict without finding a single duplicate
    between them.
    """
    pooled: list = []
    apart = 0
    for name in FIXTURE_NAMES:
        examples = _examples(name)
        apart += len(duplicate_blocks(examples, find_duplicates(examples)))
        pooled.extend(examples)

    together = duplicate_blocks(pooled, find_duplicates(pooled))
    assert len(pooled) == 56
    assert apart == 45
    assert len(together) == 44

    grown = {tuple(b.example_ids) for b in together if b.size == 3}
    assert ("saf-004", "saf-005", "saf-007") in grown


def test_blocks_are_additive_under_jaccard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mechanism, isolated. Jaccard has no idf, so a pair's score ignores the corpus.

    Composability is therefore a property of the *similarity function*, not of this
    repository's data — and the "optional accelerator" is what removes it.
    """
    _no_sklearn(monkeypatch)
    pooled: list = []
    apart = 0
    for name in FIXTURE_NAMES:
        examples = _examples(name)
        apart += len(duplicate_blocks(examples, find_duplicates(examples)))
        pooled.extend(examples)

    together = duplicate_blocks(pooled, find_duplicates(pooled))
    assert apart == 46
    assert len(together) == 46


def test_a_gate_verdict_is_not_preserved_under_union(tmp_path: Path) -> None:
    """``safety_toy`` passes a one-cluster budget alone and fails it inside the pooled corpus.

    Not one of the four appended fixtures shares a duplicate with ``safety_toy``. What
    changed is the idf of the boilerplate ``saf-004`` and ``saf-007`` share: common inside
    ``safety_toy``, rare against the pooled vocabulary, and rarer means heavier.
    """
    policy = GatePolicy(
        profile="test",
        requirements=["near_duplicates_within_budget"],
        thresholds={"max_near_duplicate_clusters": 1},
    )

    alone = ReleaseHandle.open(_build(tmp_path, ["safety_toy"], "alone")[1])
    assert evaluate_gate(alone, policy).ok

    merged = ReleaseHandle.open(_build(tmp_path, FIXTURE_NAMES, "merged")[1])
    merged_report = evaluate_gate(merged, policy)
    assert not merged_report.ok

    safety_clusters = [
        finding
        for finding in merged.audit.near_duplicates
        if any(i.startswith("saf-") for i in finding.example_ids)
    ]
    assert len(safety_clusters) == 2, "the same ten rows now report two clusters"


def test_a_verdict_refuses_to_answer_for_a_corpus_it_was_not_computed_on(
    tmp_path: Path,
) -> None:
    alone = ReleaseHandle.open(_build(tmp_path, ["safety_toy"], "alone")[1])
    merged = ReleaseHandle.open(_build(tmp_path, FIXTURE_NAMES, "merged")[1])
    report = gate_release(alone.path, "agent-control-bench")

    assert report.applies_to(alone)
    assert not report.applies_to(merged)
    report.assert_applies_to(alone)
    with pytest.raises(ContractViolation, match="not preserved under corpus change"):
        report.assert_applies_to(merged)


def test_a_content_scoped_verdict_survives_a_reseeded_split(tmp_path: Path) -> None:
    """A verdict about the examples is a verdict about the examples, however they are split."""
    a = ReleaseHandle.open(_build(tmp_path, ["qa_toy"], "a", seed=0)[1])
    b = ReleaseHandle.open(_build(tmp_path, ["qa_toy"], "b", seed=7)[1])
    assert a.manifest.content_id == b.manifest.content_id
    assert a.manifest.eval_id != b.manifest.eval_id

    policy = GatePolicy(
        profile="content-scoped",
        requirements=["no_exact_duplicates", "no_answer_in_prompt", "no_pii"],
    )
    report = evaluate_gate(a, policy)
    assert not report.split_sensitive and report.scope == "content_id"
    assert report.applies_to(b)
    report.assert_applies_to(b)


def test_a_split_sensitive_verdict_does_not_survive_a_reseeded_split(tmp_path: Path) -> None:
    """Four requirements read the split, so their verdicts are scoped to ``eval_id``.

    The witness is not hypothetical. Under ``--strategy random`` on ``qa_toy``, seed 9 puts no
    duplicate cluster inside an evaluation split and seed 0 puts one there. Both releases hold
    the same fourteen examples, so both carry the same ``content_id``. A gate that compared
    content alone would hand seed 9's pass to seed 0's release.
    """
    clean = ReleaseHandle.open(_build(tmp_path, ["qa_toy"], "s9", strategy="random", seed=9)[1])
    dirty = ReleaseHandle.open(_build(tmp_path, ["qa_toy"], "s0", strategy="random", seed=0)[1])
    assert clean.manifest.content_id == dirty.manifest.content_id
    assert clean.manifest.eval_id != dirty.manifest.eval_id

    policy = GatePolicy(profile="split-scoped", requirements=["eval_splits_deduplicated"])
    passing, failing = evaluate_gate(clean, policy), evaluate_gate(dirty, policy)
    assert passing.ok and not failing.ok, "the same examples, split two ways, two verdicts"

    assert passing.split_sensitive and passing.scope == "eval_id"
    assert not passing.applies_to(dirty)
    with pytest.raises(ContractViolation, match="not preserved under re-split"):
        passing.assert_applies_to(dirty)


def test_the_adversarial_strategy_makes_the_split_sensitive_checks_seed_invariant(
    tmp_path: Path,
) -> None:
    """Why the demo release's verdict is stable, and why the gate must not rely on that.

    ``applies_to`` reads which requirements ran, never which strategy was claimed.
    """
    policy = GatePolicy(profile="p", requirements=["eval_splits_deduplicated"])
    for seed in (0, 9, 13):
        handle = ReleaseHandle.open(_build(tmp_path, ["qa_toy"], f"adv{seed}", seed=seed)[1])
        result = evaluate_gate(handle, policy)
        assert result.ok, seed
        assert result.split_sensitive, "seed-invariant in fact, split-scoped by contract"


# --------------------------------------------------------------------------- #
# Profiles and the gate
# --------------------------------------------------------------------------- #


def test_the_five_profiles_of_the_plan_are_committed() -> None:
    assert set(PROFILES) == {
        "code-agent-runtime",
        "bioagentbench",
        "llm-regression",
        "scientific-claim-verifier",
        "agent-control-bench",
    }


def test_every_profile_names_known_requirements_and_thresholds() -> None:
    for name, profile in PROFILES.items():
        for requirement in profile.requirements:
            assert requirement in REQUIREMENTS, f"{name}: {requirement}"
        resolve_thresholds(profile.thresholds)  # raises on an unknown threshold
        assert profile.assumes, f"{name} states nothing it assumes"
        assert profile.does_not_check, f"{name} states nothing it omits"


def test_every_profile_declares_integrity_and_schema_first() -> None:
    """A finding read out of a release whose bytes do not match its manifest is not a finding."""
    for name, profile in PROFILES.items():
        assert profile.requirements[:2] == ("integrity", "schema_compatible"), name


def test_every_requirement_states_what_a_pass_does_not_establish() -> None:
    for name, requirement in REQUIREMENTS.items():
        assert requirement.cannot.strip(), name
        assert requirement.question.strip().endswith("?"), name


def test_agent_control_bench_does_not_require_unambiguous_labels() -> None:
    """The ambiguous boundary is what a control benchmark measures, not a defect in it."""
    profile = get_profile("agent-control-bench")
    assert "no_label_conflicts" not in profile.requirements
    assert any("Ambiguous labels" in line for line in profile.does_not_check)


def test_every_profile_rejects_the_deliberately_flawed_demo_release() -> None:
    """The fixtures were built with planted defects. A gate that passed them would be broken."""
    for name in PROFILES:
        assert not gate_release(DEMO, name).ok, name


def test_the_demo_release_fails_for_the_reasons_the_fixture_plants() -> None:
    report = gate_release(DEMO, "code-agent-runtime")
    by_name = {result.name: result for result in report.results}
    assert {result.name for result in report.failed} == {
        "no_missing_labels",
        "no_answer_in_prompt",
        "no_exact_duplicates",
    }
    assert by_name["no_answer_in_prompt"].evidence["example_ids"] == ["qa-007"]
    assert by_name["no_missing_labels"].evidence["example_ids"] == ["qa-010"]
    assert by_name["no_exact_duplicates"].evidence["clusters"] == [["qa-001", "qa-002"]]


def test_llm_regression_counts_blocks_not_rows(demo: ReleaseHandle) -> None:
    """The test split has two rows and two blocks; the gate would fail either way here.

    What the test pins is *which number is read*: a suite whose eval split is 200 rows in
    3 duplicate clusters must be measured as 3.
    """
    result = evaluate("eval_split_is_rankable", demo, resolve_thresholds({"min_eval_blocks": 30}))
    assert result.verdict is Verdict.FAIL
    assert result.evidence["blocks"] == demo.blocks_in("test") == 2
    assert result.evidence["rows"] == 2

    relaxed = resolve_thresholds({"min_eval_blocks": 2})
    assert evaluate("eval_split_is_rankable", demo, relaxed).verdict is Verdict.PASS


def test_a_clean_release_passes_a_profile(tmp_path: Path) -> None:
    """The positive control. Without it, `not report.ok` everywhere proves only that the
    gate is capable of saying no."""
    _, out = _build(
        tmp_path, ["qa_toy"], "clean", drop=["exact", "normalized", "blocking-labels"]
    )
    handle = ReleaseHandle.open(out)
    policy = GatePolicy(
        profile="clean",
        requirements=["integrity", "schema_compatible", "no_exact_duplicates", "no_pii"],
    )
    report = evaluate_gate(handle, policy)
    assert report.ok and report.exit_code == 0
    assert len(report.passed) == 4


def test_integrity_failure_is_a_report_not_a_traceback(tmp_path: Path) -> None:
    """Every other defect produces a report; a corrupt release must not produce an exception."""
    _, out = _build(tmp_path, ["qa_toy"], "broken")
    path = out / "splits" / "test.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")

    report = gate_release(out, "code-agent-runtime")
    assert not report.ok
    assert "integrity" in {result.name for result in report.failed}
    assert report.dataset_id == "broken", "the report still names the release it judged"


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def test_a_policy_rejects_an_unknown_requirement() -> None:
    with pytest.raises(ValueError, match="unknown requirement"):
        GatePolicy(profile="p", requirements=["no_such_check"])


def test_a_policy_rejects_an_unknown_threshold(demo: ReleaseHandle) -> None:
    policy = GatePolicy(
        profile="p", requirements=["no_pii"], thresholds={"max_near_duplicates": 1}
    )
    with pytest.raises(ValueError, match="unknown threshold"):
        evaluate_gate(demo, policy)


def test_a_policy_round_trips_through_yaml(tmp_path: Path, demo: ReleaseHandle) -> None:
    import yaml

    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "profile": "llm-regression",
                "requirements": ["integrity", "eval_split_is_rankable"],
                "thresholds": {"min_eval_blocks": 2, "eval_split": "test"},
            }
        )
    )
    policy = GatePolicy.from_file(path)
    assert evaluate_gate(demo, policy).ok, "the same release passes a threshold nobody in it chose"


def test_the_report_records_the_policy_that_ran_not_the_one_that_arrived(
    demo: ReleaseHandle,
) -> None:
    policy = GatePolicy.for_profile("llm-regression")
    report = evaluate_gate(demo, policy, requirements=["no_pii"])
    assert report.policy.requirements == ["no_pii"]
    assert [r.name for r in report.results] == ["no_pii"]


def test_contract_version_travels_with_the_report(demo: ReleaseHandle) -> None:
    report = gate_release(DEMO, "bioagentbench")
    assert report.contract_version == CONTRACT_VERSION
    assert report.to_dict()["contract_version"] == CONTRACT_VERSION


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_the_report_never_collapses_three_verdicts_into_two() -> None:
    report = gate_release(DEMO, "scientific-claim-verifier")
    text = format_gate(report)
    assert "undecidable" in text and "FAIL" in text and "pass" in text
    assert report.to_dict()["counts"] == {"pass": 2, "fail": 3, "undecidable": 1}


def test_the_rendered_report_carries_every_cannot_clause() -> None:
    """A gate report is read by whoever most wants it to say the dataset is fine."""
    from eval_dataset_forge.reports.document import document_strings

    report = gate_release(DEMO, "code-agent-runtime")
    strings = document_strings(gate_document(report))
    for result in report.results:
        assert result.cannot in strings, result.name


def test_the_profile_scope_reaches_the_rendered_report() -> None:
    from eval_dataset_forge.reports.document import document_strings

    profile = get_profile("llm-regression")
    strings = document_strings(gate_document(gate_release(DEMO, "llm-regression")))
    for claim in (*profile.assumes, *profile.does_not_check):
        assert claim in strings


def test_the_document_renders_in_both_backends() -> None:
    from eval_dataset_forge.reports.html import render as render_html
    from eval_dataset_forge.reports.markdown import render as render_md

    document = gate_document(gate_release(DEMO, "llm-regression"))
    assert render_md(document) and render_html(document)


def test_gate_report_json_is_stable() -> None:
    first = gate_release(DEMO, "llm-regression").to_dict()
    second = gate_release(DEMO, "llm-regression").to_dict()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_gate_exits_1_on_a_failing_release(capsys: pytest.CaptureFixture) -> None:
    from eval_dataset_forge.cli import main

    assert main(["gate", "--release", str(DEMO), "--profile", "llm-regression"]) == 1
    assert "eval_split_is_rankable" in capsys.readouterr().out


def test_cli_gate_exits_0_when_the_policy_permits_undecidable(
    capsys: pytest.CaptureFixture,
) -> None:
    from eval_dataset_forge.cli import main

    argv = ["gate", "--release", str(DEMO), "--profile", "bioagentbench"]
    assert main([*argv, "--requirement", "no_retrieval_contamination"]) == 1
    assert (
        main([*argv, "--requirement", "no_retrieval_contamination", "--allow-undecidable"]) == 0
    )
    capsys.readouterr()


def test_cli_gate_list_names_every_profile(capsys: pytest.CaptureFixture) -> None:
    from eval_dataset_forge.cli import main

    assert main(["gate", "--list"]) == 0
    out = capsys.readouterr().out
    for name in PROFILES:
        assert name in out


def test_cli_gate_rejects_a_missing_profile(capsys: pytest.CaptureFixture) -> None:
    from eval_dataset_forge.cli import main

    assert main(["gate", "--release", str(DEMO)]) == 2
    assert main(["gate", "--release", str(DEMO), "--profile", "nope"]) == 2
    capsys.readouterr()


def test_cli_gate_writes_markdown_and_html(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from eval_dataset_forge.cli import main

    md, html = tmp_path / "gate.md", tmp_path / "gate.html"
    main(
        [
            "gate",
            "--release",
            str(DEMO),
            "--profile",
            "agent-control-bench",
            "--out",
            str(md),
            "--html",
            str(html),
        ]
    )
    capsys.readouterr()
    assert md.read_text().startswith("# Gate report: qa_toy v0.1.0")
    assert "<table" in html.read_text()
