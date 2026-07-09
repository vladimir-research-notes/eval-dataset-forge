"""Milestone 12: the audit pipeline, its human-readable report, and the smoke audit's scoring.

The load-bearing tests here are the ones that check the *scoring rule*, not the detectors:

- ``test_recall_cannot_distinguish_a_detector_that_flags_everything`` — the reason every recall in
  this repository is printed beside a precision.
- ``test_pair_decomposition_charges_a_cluster_for_every_pair_it_asserts`` — the reason a duplicate
  finding is scored on the pairs it implies rather than as one prediction.
- ``test_unmapped_ground_truth_kind_raises`` — the reason a planted defect cannot leave the recall
  denominator by typo.
- ``test_unsplit_dataset_reports_vacuity_not_health`` — the reason a cross-split zero is annotated.

Everything runs offline against the committed fixtures. No network, no model, no paid API.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pytest

from eval_dataset_forge.audit import (
    AUDIT_JSON,
    AUDIT_MD,
    SPLIT_DEPENDENT_CHECKS,
    UNVERSIONED,
    audit_dataset,
    default_dataset_id,
    run_audit,
    write_audit,
)
from eval_dataset_forge.cli import main
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.reports.audit_report import (
    audit_document,
    render_markdown,
)
from eval_dataset_forge.reports.document import document_strings
from eval_dataset_forge.schemas import AuditResult, Split
from eval_dataset_forge.scoring import (
    CHANNEL_OF,
    NOT_AUTO_DETECTABLE,
    RECOVERED,
    RECOVERED_ELSEWHERE,
    UNWIRED,
    load_ground_truth,
    parse_ground_truth,
    predictions,
    score,
    score_degenerate,
    score_predictions,
)
from eval_dataset_forge.smoke import run_smoke, smoke_document, write_smoke

FIXTURES = Path("datasets/fixtures")
QA_TOY = FIXTURES / "qa_toy" / "examples.jsonl"
QA_DEFECTS = FIXTURES / "qa_toy" / "defects.yaml"
FIXTURE_NAMES = ("qa_toy", "agent_tasks_toy", "retrieval_toy", "bio_toy", "safety_toy")

# A synthetic, obviously fake AWS access-key id (the AWS docs example value), matching
# tests/test_validation.py. Never a real credential; it exists so the PII scan has something
# to catch and the audit report has something to redact.
FAKE_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


@pytest.fixture(scope="module")
def qa_audit():
    return audit_dataset(QA_TOY)


@pytest.fixture(scope="module")
def smoke():
    return run_smoke(FIXTURES)


# --------------------------------------------------------------------------- #
# The audit pipeline
# --------------------------------------------------------------------------- #


def test_audit_assembles_every_dimension(qa_audit) -> None:
    result = qa_audit.result
    assert isinstance(result, AuditResult)
    assert result.dataset_id == "qa_toy"
    assert result.version == UNVERSIONED
    assert result.audit_id.startswith("audit-")
    assert result.exact_duplicates and result.near_duplicates
    assert result.leakage_flags and result.label_warnings
    assert result.difficulty_distribution
    assert result.recommendations


def test_audit_id_is_a_function_of_the_content(qa_audit) -> None:
    # Two audits of the same rows agree; a changed row changes the id.
    again = audit_dataset(QA_TOY)
    assert again.result.audit_id == qa_audit.result.audit_id

    examples = loader.load_examples(QA_TOY)
    examples[0] = examples[0].model_copy(update={"input": examples[0].input + " ?"})
    mutated = run_audit(examples, dataset_id="qa_toy")
    assert mutated.result.audit_id != qa_audit.result.audit_id


def test_audit_reads_the_split_it_is_given_and_never_builds_one(qa_audit) -> None:
    stats = qa_audit.result.split_statistics
    assert stats["source"] == "observed"
    assert stats["spec"] is None
    on_disk = {
        example.example_id: str(example.split) for example in loader.load_examples(QA_TOY)
    }
    for name in ("train", "dev", "test"):
        assert stats["counts"][name] == sum(1 for split in on_disk.values() if split == name)


def test_audit_does_not_mutate_the_examples_it_is_handed() -> None:
    examples = loader.load_examples(QA_TOY)
    before = [example.model_dump(mode="json") for example in examples]
    run_audit(examples, dataset_id="qa_toy")
    assert [example.model_dump(mode="json") for example in examples] == before


def test_near_method_pins_the_detector_instead_of_letting_pip_choose_it() -> None:
    """Milestone 14: ``--near-method`` makes a published audit number reproducible.

    Before it existed, ``auto`` resolution meant the only way to see the Jaccard figures that
    ``docs/RESULTS.md`` and ``docs/DEDUPLICATION.md`` have published since Milestone 4 was to
    uninstall scikit-learn. The environment chose the detector, and the detectors disagree.
    """
    examples = loader.load_examples(FIXTURES / "bio_toy" / "examples.jsonl")

    tfidf = run_audit(examples, dataset_id="bio_toy", near_method="tfidf")
    jaccard = run_audit(examples, dataset_id="bio_toy", near_method="jaccard")

    assert tfidf.dedup.near_method == "tfidf"
    assert jaccard.dedup.near_method == "jaccard"

    # The two disagree on this fixture: tfidf pairs bio-001 with bio-002, jaccard does not.
    # That disagreement is the whole reason the flag had to exist.
    assert len(tfidf.dedup.near) == 1
    assert len(jaccard.dedup.near) == 0


def test_near_method_defaults_to_auto_and_changes_nothing() -> None:
    """The flag is additive: an audit run without it behaves exactly as it did before."""
    examples = loader.load_examples(QA_TOY)
    assert run_audit(examples, dataset_id="qa_toy").to_dict() == run_audit(
        examples, dataset_id="qa_toy", near_method="auto"
    ).to_dict()


def test_unsplit_dataset_reports_vacuity_not_health() -> None:
    """Four checks are undefined without a split. Their zeros must not read as a clean bill."""
    examples = [
        example.model_copy(update={"split": Split.UNASSIGNED})
        for example in loader.load_examples(QA_TOY)
    ]
    report = run_audit(examples, dataset_id="qa_toy")

    # The cross-split checks return zero...
    assert not report.leakage.train_test_overlap
    assert not report.leakage.ngram_overlap
    assert not report.facts.eval_duplicates
    # ...and the audit says why, in a warning, a recommendation, and the rendered report.
    assert len(report.facts.unassigned) == len(examples)
    assert any("unassigned" in warning for warning in report.warnings)
    assert any("vacuity" in note for note in report.result.recommendations)
    assert all(check in " ".join(report.result.recommendations) for check in SPLIT_DEPENDENT_CHECKS)
    assert b"by vacuity" in render_markdown(report.result)

    # The substantive checks still ran: an exact duplicate does not need a split.
    assert report.dedup.exact


def test_split_dataset_does_not_carry_the_vacuity_warning(qa_audit) -> None:
    assert not qa_audit.facts.unassigned
    assert not any("vacuity" in note for note in qa_audit.result.recommendations)
    assert b"by vacuity" not in render_markdown(qa_audit.result)


def test_empty_dataset_is_an_error_not_an_empty_audit() -> None:
    with pytest.raises(ValueError, match="nothing to audit"):
        run_audit([], dataset_id="empty")


def test_default_dataset_id_is_the_directory_not_the_filename() -> None:
    assert default_dataset_id("datasets/fixtures/qa_toy/examples.jsonl") == "qa_toy"
    assert default_dataset_id("/tmp/my_bench.jsonl") == "my_bench"


def test_release_and_audit_share_one_assembly() -> None:
    """A release's audit and a raw file's differ in ``source`` and in nothing structural."""
    from eval_dataset_forge.datasets.release import build_release

    release = build_release([QA_TOY], dataset_id="qa_toy", created_at="2020-01-01T00:00:00+00:00")
    constructed = release.audit.split_statistics
    observed = audit_dataset(QA_TOY).result.split_statistics

    assert constructed["source"] == "constructed"
    assert observed["source"] == "observed"
    assert constructed["spec"] is not None and observed["spec"] is None
    assert set(constructed) == set(observed)
    # The per-example assignment lives in the split files, never in the audit.
    assert "assignment" not in constructed


# --------------------------------------------------------------------------- #
# audit.md — the artifact PLAN §5D always listed
# --------------------------------------------------------------------------- #


def test_audit_md_is_a_pure_view_of_audit_json(qa_audit) -> None:
    """Rendered from the ``AuditResult`` alone: no examples, no manifest, no filesystem."""
    round_tripped = AuditResult(**json.loads(json.dumps(qa_audit.result.model_dump(mode="json"))))
    assert render_markdown(round_tripped) == render_markdown(qa_audit.result)


def test_audit_md_is_deterministic(qa_audit) -> None:
    assert render_markdown(qa_audit.result) == render_markdown(qa_audit.result)


def test_audit_document_reaches_both_backends(qa_audit) -> None:
    """No backend silently drops content. The same property ``test_reports.py`` asserts of cards.

    Compared after each backend's own escaping and with whitespace normalized: a table cell
    collapses newlines, and a literal ``[`` is escaped, in both formats by design.
    """
    from html import escape

    from eval_dataset_forge.reports import html as html_backend
    from eval_dataset_forge.reports import markdown as md_backend
    from eval_dataset_forge.reports.document import CODE, inline_spans

    def flat(text: str) -> str:
        return " ".join(text.split())

    document = audit_document(qa_audit.result)
    markdown = flat(md_backend.render(document))
    html = flat(html_backend.render(document))
    for text in document_strings(document):
        for span in inline_spans(text):
            expected = span.text if span.kind == CODE else md_backend.escape(span.text)
            assert flat(expected) in markdown, f"markdown dropped {span.text!r}"
            assert flat(escape(span.text)) in html, f"html dropped {span.text!r}"


def test_audit_md_prints_zeros_with_their_certainty(qa_audit) -> None:
    """A report listing only what it found cannot be told from one whose checks never ran."""
    text = render_markdown(qa_audit.result).decode("utf-8")
    assert "| leakage: retrieval contamination | 0 |" in text
    assert "heuristic" in text and "exact" in text


def test_audit_md_never_prints_a_raw_pii_value() -> None:
    """``PiiFinding.snippet`` is redacted upstream; the report must print nothing else."""
    examples = loader.load_examples(QA_TOY)
    examples[0] = examples[0].model_copy(
        update={"input": f"{examples[0].input} key={FAKE_AWS_KEY}"}
    )
    report = run_audit(examples, dataset_id="qa_toy")
    assert report.result.pii_flags
    assert FAKE_AWS_KEY not in render_markdown(report.result).decode("utf-8")


def test_release_bundle_now_carries_audit_md(tmp_path: Path) -> None:
    from eval_dataset_forge.datasets.manifest import verify_release
    from eval_dataset_forge.datasets.release import build_release, write_release

    report = build_release(
        [QA_TOY], dataset_id="qa_toy", created_at="2020-01-01T00:00:00+00:00"
    )
    out = tmp_path / "rel"
    write_release(report, out)
    assert (out / "audit.md").is_file()
    # It is a hashed artifact like any other, and the release still verifies.
    assert "audit.md" in report.manifest.hashes
    assert verify_release(out).ok
    # And it is the rendering of the audit.json shipped beside it.
    assert (out / "audit.md").read_bytes() == render_markdown(report.audit)


def test_adding_audit_md_is_a_packaging_change_not_a_data_change() -> None:
    """A bundle gaining a derived artifact earns a `patch`: no score measured on it moves.

    ``content_id`` and ``eval_id`` are functions of the examples, not of the artifacts rendered
    beside them, which is why the committed demo release could be rebuilt in place at v0.1.0
    rather than re-versioned. Milestone 10's machinery says so rather than the milestone note.
    """
    from eval_dataset_forge.datasets.release import build_release
    from eval_dataset_forge.datasets.versioning import compare

    report = build_release(
        [QA_TOY], dataset_id="qa_toy", created_at="2020-01-01T00:00:00+00:00"
    )
    without = report.manifest.model_copy(
        update={
            "hashes": {
                name: digest
                for name, digest in report.manifest.hashes.items()
                if name != AUDIT_MD
            }
        }
    )
    delta = compare(without, report.release.examples, report.manifest, report.release.examples)
    assert delta.bump == "patch"
    assert delta.scores_comparable
    assert not delta.eval_affected
    assert any("packaging only" in reason for reason in delta.reasons)


def test_write_audit_emits_both_artifacts(qa_audit, tmp_path: Path) -> None:
    written = write_audit(qa_audit, tmp_path)
    assert set(written) == {AUDIT_JSON, AUDIT_MD}
    reloaded = AuditResult(**json.loads((tmp_path / AUDIT_JSON).read_text(encoding="utf-8")))
    assert reloaded.audit_id == qa_audit.result.audit_id


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def test_every_committed_ground_truth_kind_is_wired() -> None:
    """A planted defect must not leave the recall denominator because nobody mapped its kind."""
    for name in FIXTURE_NAMES:
        truth = load_ground_truth(FIXTURES / name / "defects.yaml")
        assert truth.defects
        assert truth.dataset_id == name


def test_unmapped_ground_truth_kind_raises() -> None:
    with pytest.raises(ValueError, match="unmapped ground-truth defect"):
        parse_ground_truth(
            {
                "dataset_id": "toy",
                "planted_defects": {"label_issues": [{"id": "a", "kind": "sarcastic"}]},
            }
        )


def test_unmapped_section_raises() -> None:
    with pytest.raises(ValueError, match="unmapped ground-truth defect"):
        parse_ground_truth(
            {"dataset_id": "toy", "planted_defects": {"vibes": [{"ids": ["a", "b"]}]}}
        )


def test_channel_map_covers_every_section_in_every_fixture() -> None:
    sections = {
        (section, entry.get("kind"))
        for name in FIXTURE_NAMES
        for section, entries in loader.load_defects(
            FIXTURES / name / "defects.yaml"
        )["planted_defects"].items()
        for entry in entries
    }
    for section, kind in sections:
        assert (section, kind) in CHANNEL_OF or (section, None) in CHANNEL_OF


# --------------------------------------------------------------------------- #
# The scoring rule itself
# --------------------------------------------------------------------------- #


def test_recall_cannot_distinguish_a_detector_that_flags_everything(smoke) -> None:
    """The reason no recall in this repository is printed without a precision beside it."""
    assert smoke.recall(any_channel=True) == 1.0
    assert smoke.degenerate_recall() == 1.0

    # Precision is what separates them, by one to two orders of magnitude.
    real = smoke.channel_totals()
    control = smoke.channel_totals(degenerate=True)
    ratios = {
        name: real[name].precision / control[name].precision
        for name in real
        if real[name].precision and control[name].precision
    }
    assert ratios
    assert min(ratios.values()) > 9
    assert max(ratios.values()) > 100


def test_degenerate_detector_scores_full_recall_under_subset_matching_too() -> None:
    """No matching rule repairs recall: the control really does contain every planted defect."""
    truth = load_ground_truth(QA_DEFECTS)
    examples = loader.load_examples(QA_TOY)
    everything = [example.example_id for example in examples]
    for defect in truth.defects:
        if defect.scorable and defect.unit == "pair":
            assert set(defect.members) <= set(everything)


def test_flag_everything_costs_quadratically_on_pairs_and_linearly_on_examples() -> None:
    """Why clusters are decomposed: a k-cluster's assertions grow as C(k,2), not as 1.

    On the pair unit the control must assert every one of the ``C(n,2)`` pairs, so its precision
    falls as ``1/n²``. On the example unit it asserts ``n`` and its precision falls only as
    ``1/n``. Scoring a cluster as a single prediction would put duplicate detection on the second
    curve, where flagging everything is cheap.
    """
    examples = loader.load_examples(QA_TOY)
    control = score_degenerate(examples, load_ground_truth(QA_DEFECTS))
    n = len(examples)
    assert control.channels["dedup.exact"].predicted == n * (n - 1) // 2 == 91
    assert control.channels["labels.missing"].predicted == n == 14


def test_pair_decomposition_charges_a_cluster_for_every_pair_it_asserts(qa_audit) -> None:
    """A k-member cluster asserts C(k,2) pairs — including ones nothing plants."""
    near = [f for f in qa_audit.dedup.near if len(f.example_ids) > 2]
    assert near, "qa_toy's near stage is expected to merge three examples into one cluster"
    cluster = near[0].example_ids
    asserted = predictions(qa_audit)["dedup.near"]
    assert len(asserted) == len(list(combinations(sorted(cluster), 2)))

    # And the extra pair the cluster implies is surfaced, not absorbed.
    scored = score(qa_audit, load_ground_truth(QA_DEFECTS))
    assert scored.channels["dedup.near"].unexplained == ("qa-004 + qa-012",)
    assert scored.channels["dedup.near"].recall == 1.0
    assert scored.channels["dedup.near"].precision < 1.0


def test_precision_is_none_rather_than_zero_when_nothing_of_the_kind_is_planted(smoke) -> None:
    """0.0 would be the only value any non-silent detector could score. That measures nothing."""
    ngram = smoke.channel_totals()["leakage.ngram_overlap"]
    assert ngram.planted == 0
    assert ngram.predicted > 0
    assert ngram.precision is None
    # Its assertions are corroborations of defects planted under another section, not inventions.
    assert ngram.unexplained == ()
    assert ngram.corroborating


def test_corroboration_is_unit_preserving_and_not_a_blanket_pardon() -> None:
    """Both endpoints planted does not make the *pair* planted."""
    truth = load_ground_truth(QA_DEFECTS)
    explained_examples = truth.explained_atoms("example")
    explained_pairs = truth.explained_atoms("pair")
    assert {"qa-004", "qa-012"} <= explained_examples
    assert frozenset({"qa-004", "qa-012"}) not in explained_pairs


def test_a_defect_missed_by_its_channel_can_be_recovered_by_another() -> None:
    """``ret-001``/``ret-002`` is unreachable by any lexical prompt threshold. Found anyway."""
    report = audit_dataset(FIXTURES / "retrieval_toy" / "examples.jsonl")
    truth = load_ground_truth(FIXTURES / "retrieval_toy" / "defects.yaml")
    scored = score(report, truth)

    statuses = {(o.defect.section, o.defect.kind): o.status for o in scored.outcomes}
    assert statuses[("near_duplicates", "near_duplicates")] == RECOVERED_ELSEWHERE
    assert statuses[("leakage", "train_test_overlap")] == RECOVERED_ELSEWHERE
    assert not scored.missed
    # The channel that recovers it is the passage-level one, not a prompt threshold.
    assert scored.channels["leakage.duplicate_documents"].recovered == 1


def test_a_silent_detector_scores_zero_recall_and_no_precision() -> None:
    truth = load_ground_truth(QA_DEFECTS)
    scored = score_predictions({}, truth)
    assert scored.recall() == 0.0
    assert all(outcome.status != RECOVERED for outcome in scored.outcomes)
    assert len(scored.missed) == len(scored.detectable)


def test_excluded_defects_are_named_and_the_two_reasons_are_distinct(smoke) -> None:
    """`not-auto-detectable` is the fixture's claim; `no-channel` is the scorer's wiring."""
    reasons = {reason for _, _, _, reason in smoke.excluded()}
    assert reasons == {NOT_AUTO_DETECTABLE, UNWIRED}
    kinds = {kind for _, kind, _, _ in smoke.excluded()}
    assert "mislabeled" in kinds  # declared undetectable by the fixture
    assert "paraphrased_documents" in kinds  # no detector exists in this project

    # Refusing every exclusion gives a lower, and stated, recall.
    assert smoke.recall_over_all_planted() < smoke.recall(any_channel=True)


def test_mislabeled_examples_stay_invisible(smoke) -> None:
    """The label defect that does the most damage is found by nothing here. Pinned so it stays."""
    mislabeled = {
        members for _, kind, members, _ in smoke.excluded() if kind == "mislabeled"
    }
    assert mislabeled == {"qa-006", "bio-009", "saf-006"}
    for outcome in smoke.fixtures:
        flagged = {finding.example_id for finding in outcome.audit.labels.findings}
        assert not (mislabeled & flagged)


def test_negative_control_is_not_flagged_by_the_rule_it_traps(smoke) -> None:
    """``ret-005``'s empty gold answer is correct. A rule keyed on emptiness must not fire."""
    assert smoke.totals()["controls"] == 1
    assert not smoke.controls_violated()
    retrieval = next(o for o in smoke.fixtures if o.dataset_id == "retrieval_toy")
    assert "ret-005" not in {f.example_id for f in retrieval.audit.labels.missing}


def test_a_detector_that_flags_the_control_is_caught() -> None:
    truth = load_ground_truth(FIXTURES / "retrieval_toy" / "defects.yaml")
    scored = score_predictions({"labels.missing": {"ret-005"}}, truth)
    assert scored.controls_violated


def test_scoring_reproduces_the_hand_computed_stage_recalls(smoke) -> None:
    """The per-stage numbers Milestones 4–6 computed by hand, recomputed by the scorer."""
    channels = smoke.channel_totals()
    assert (channels["dedup.exact"].recovered, channels["dedup.exact"].planted) == (4, 4)
    assert (channels["dedup.near"].recovered, channels["dedup.near"].planted) == (4, 5)
    assert (
        channels["dedup.duplicate_sources"].recovered,
        channels["dedup.duplicate_sources"].planted,
    ) == (2, 2)
    overlap = channels["leakage.train_test_overlap"]
    assert (overlap.recovered, overlap.planted) == (8, 9)
    answer = channels["leakage.answer_in_prompt"]
    assert (answer.recovered, answer.planted) == (2, 2)
    contamination = channels["leakage.retrieval_contamination"]
    assert (contamination.recovered, contamination.planted) == (1, 1)


# --------------------------------------------------------------------------- #
# The smoke audit
# --------------------------------------------------------------------------- #


def test_smoke_audits_every_fixture_and_the_pooled_corpus(smoke) -> None:
    assert {outcome.dataset_id for outcome in smoke.fixtures} == set(FIXTURE_NAMES)
    assert smoke.n_examples == 56
    assert smoke.pooled is not None
    assert smoke.pooled.n_examples == 56
    # The pooled corpus holds fewer independent blocks than examples — Milestone 9's number.
    assert smoke.pooled.dependence["block_count"] == 44


def test_smoke_finds_every_planted_defect_and_invents_five_assertions(smoke) -> None:
    totals = smoke.totals()
    assert (totals["detectable"], totals["recovered_any_channel"], totals["missed"]) == (40, 40, 0)
    assert totals["recovered"] == 38  # two need a channel other than their own
    assert smoke.ok

    # Every unexplained assertion is enumerable and documented in docs/RESULTS.md.
    assert {(fixture, atom) for fixture, _, atom in smoke.unexplained()} == {
        ("qa_toy", "qa-004 + qa-012"),
        ("agent_tasks_toy", "agent-003 + agent-004"),
        ("retrieval_toy", "ret-005"),
    }


def test_a_near_duplicate_verdict_is_a_property_of_the_corpus_not_of_the_pair() -> None:
    """Pooling the fixtures changes a verdict *inside* one of them. TF-IDF's IDF is collective.

    ``saf-004``/``saf-007`` score 0.287 within ``safety_toy`` and 0.344 within the pooled corpus,
    straddling the 0.30 threshold — not because either prompt changed, but because the boilerplate
    they share grows rarer, and therefore heavier, against the wider pooled vocabulary. So a
    duplicate report is not composable: auditing shards and unioning the findings is not auditing
    the union. Surfaced by the pooled audit, which is why it is run.
    """
    from eval_dataset_forge.dedup.near_duplicate import DEFAULT_THRESHOLDS, similarity_matrix

    alone = loader.load_examples(FIXTURES / "safety_toy" / "examples.jsonl")
    pooled = [
        example
        for name in FIXTURE_NAMES
        for example in loader.load_examples(FIXTURES / name / "examples.jsonl")
    ]

    def similarity(examples, left: str, right: str) -> float:
        matrix, _ = similarity_matrix(examples, method="tfidf")
        index = {example.example_id: i for i, example in enumerate(examples)}
        return matrix[index[left]][index[right]]

    threshold = DEFAULT_THRESHOLDS["tfidf"]
    assert similarity(alone, "saf-004", "saf-007") < threshold
    assert similarity(pooled, "saf-004", "saf-007") > threshold

    # And the cluster really does appear only in the pooled audit.
    def near_clusters(examples, dataset_id):
        return {
            tuple(finding.example_ids)
            for finding in run_audit(examples, dataset_id=dataset_id).dedup.near
        }

    assert ("saf-004", "saf-005", "saf-007") not in near_clusters(alone, "safety_toy")
    assert ("saf-004", "saf-005", "saf-007") in near_clusters(pooled, "pooled")


def test_smoke_report_is_deterministic_and_carries_no_clock(smoke) -> None:
    from eval_dataset_forge.smoke import render_markdown as render_smoke

    once, twice = render_smoke(smoke), render_smoke(smoke)
    assert once == twice
    text = once.decode("utf-8")
    for stamp in ("20", "T00:", "created_at"):
        assert stamp not in text.split("## Corpus")[0].replace("Milestones 4 and 5", "")


def test_smoke_report_states_that_it_is_in_sample(smoke) -> None:
    text = smoke_document(smoke)
    joined = " ".join(document_strings(text))
    assert "in sample" in joined
    assert "resubstitution" in joined
    assert "has not seen" in joined


def test_smoke_report_prints_the_flag_everything_control(smoke) -> None:
    from eval_dataset_forge.smoke import render_markdown as render_smoke

    text = render_smoke(smoke).decode("utf-8")
    assert "flag-everything" in text
    assert "precision" in text


def test_smoke_requires_ground_truth_for_every_fixture(tmp_path: Path) -> None:
    """An unscored fixture would look exactly like a passing one."""
    (tmp_path / "orphan").mkdir()
    (tmp_path / "orphan" / "examples.jsonl").write_bytes(QA_TOY.read_bytes())
    with pytest.raises(ValueError, match="no defects.yaml"):
        run_smoke(tmp_path)


def test_smoke_writes_an_audit_per_fixture(smoke, tmp_path: Path) -> None:
    written = write_smoke(smoke, tmp_path / "smoke.md", audits_dir=tmp_path / "audits")
    assert (tmp_path / "smoke.md").is_file()
    for name in (*FIXTURE_NAMES, "all_fixtures"):
        assert (tmp_path / "audits" / name / AUDIT_JSON).is_file()
        assert (tmp_path / "audits" / name / AUDIT_MD).is_file()
    assert len(written) == 1 + 2 * (len(FIXTURE_NAMES) + 1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_audit_writes_both_artifacts(tmp_path: Path, capsys) -> None:
    code = main(["audit", "--dataset", str(QA_TOY), "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / AUDIT_JSON).is_file() and (tmp_path / AUDIT_MD).is_file()
    assert "audit: qa_toy" in capsys.readouterr().out


def test_cli_audit_json_is_valid(tmp_path: Path, capsys) -> None:
    assert main(["audit", "--dataset", str(QA_TOY), "--out", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.split("\nwrote ")[0])
    assert payload["dataset_id"] == "qa_toy"
    assert payload["audit"]["audit_id"].startswith("audit-")


def test_cli_audit_fail_on_leakage_gates_on_the_split_as_given(tmp_path: Path) -> None:
    # qa_toy ships with duplicate clusters straddling train/test: the gate fires.
    code = main(
        ["audit", "--dataset", str(QA_TOY), "--out", str(tmp_path), "--fail-on-leakage"]
    )
    assert code == 1


def test_cli_smoke_runs_and_gates(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "smoke",
            "--out",
            str(tmp_path / "smoke_audit.md"),
            "--audits-dir",
            str(tmp_path / "audits"),
            "--fail-on-missed-defect",
            "--fail-on-control-violation",
        ]
    )
    assert code == 0
    assert (tmp_path / "smoke_audit.md").is_file()
    out = capsys.readouterr().out
    assert "smoke audit: 5 fixture(s), 56 examples" in out
    assert "in-sample" in out


def test_cli_no_longer_advertises_a_pending_milestone(capsys) -> None:
    assert main(["audit", "--dataset", str(QA_TOY), "--out", "/tmp/evalforge-audit-check"]) == 0
    assert "not implemented" not in capsys.readouterr().out
