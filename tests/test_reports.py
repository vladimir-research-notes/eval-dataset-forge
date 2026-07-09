"""Tests for the report renderers and the dataset card (Milestone 11).

Covered here:

  * **One document, two backends** — every string a ``Document`` carries reaches both the
    Markdown and the HTML rendering. This is the property a pair of string templates cannot
    offer, and it is why the document model exists.
  * **Escaping** — a ``|`` in a table cell does not grow a column; ``<script>`` survives as
    text in both formats; ``*`` cannot open emphasis it did not mean to; intraword ``_`` is
    left alone, because CommonMark already forbids it from opening emphasis and a backslash
    in ``qa_toy`` would land in the most-read line of every card.
  * **The card is a fixed point** — ``dataset_card.md`` is hashed *into* ``manifest.hashes``,
    so ``build_card`` must not read that table. The test mutates every hash and asserts the
    card is unchanged. Without this the release builder's two-pass assembly would ship a card
    describing a manifest that no longer exists.
  * **The card carries no clock** — for the same reason ``REPRODUCE.md`` carries none. Two
    builds seconds apart must differ in ``manifest.json`` and nothing else, or the ``none``
    version bump becomes unreachable.
  * **The card cannot outrun its evidence** — an audit naming another release is refused; the
    counts printed are the counts in ``audit.json``; zeros appear in the findings table and
    never in the "known issues" bullets, because a bullet reading "0 duplicates" is a
    reassurance and the table row is a claim about what was checked.
  * **Sample size is two numbers** — 14 examples in 11 independent blocks, and per split.
  * **Curator prose is absent, not invented** — an unwritten purpose is named in the card and
    can gate CI; a typo in ``card.yaml`` raises rather than silently dropping a section.
  * **Prose is packaging** — editing a non-goal changes no number and forces a ``patch``.
  * **The committed demo release still regenerates its own card.**

Everything is CPU-only and offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_dataset_forge.cli import main
from eval_dataset_forge.datasets.manifest import MANIFEST_FILE, load_manifest, verify_release
from eval_dataset_forge.datasets.release import build_release, write_release
from eval_dataset_forge.datasets.versioning import _PACKAGING_FIELDS
from eval_dataset_forge.reports import document as doc
from eval_dataset_forge.reports import html as html_backend
from eval_dataset_forge.reports import markdown as md_backend
from eval_dataset_forge.reports.dataset_card import (
    CARD_FILE,
    build_card,
    card_document,
    card_from_release,
    check_card,
    format_card,
    known_issues,
    load_card_meta,
    render_html,
    render_markdown,
    write_card,
)
from eval_dataset_forge.schemas import CURATOR_SECTIONS, AuditResult, CardMeta, DatasetManifest

QA_TOY = Path("datasets/fixtures/qa_toy/examples.jsonl")
QA_TOY_CARD_META = Path("datasets/fixtures/qa_toy/card.yaml")
DEMO_RELEASE = Path("datasets/releases/qa_toy_v0.1.0")
FIXED_CLOCK = "2026-01-01T00:00:00+00:00"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _release(tmp_path: Path, name: str = "rel", **overrides):
    payload = {
        "dataset_id": "qa_toy",
        "created_at": FIXED_CLOCK,
        "version": "0.1.0",
        "quarantine_blocking": True,
    }
    payload.update(overrides)
    report = build_release([QA_TOY], **payload)
    write_release(report, tmp_path / name)
    return report, tmp_path / name


def _doc(*blocks: doc.Block) -> doc.Document:
    return doc.Document(title="T", blocks=blocks)


# --------------------------------------------------------------------------- #
# Inline spans
# --------------------------------------------------------------------------- #


def test_paired_backticks_become_code_spans() -> None:
    spans = doc.inline_spans("run `evalforge audit` now")
    assert [(span.kind, span.text) for span in spans] == [
        (doc.TEXT, "run "),
        (doc.CODE, "evalforge audit"),
        (doc.TEXT, " now"),
    ]


def test_an_unpaired_backtick_is_literal() -> None:
    """An odd backtick count means the author wrote one, not that they forgot to close it."""
    spans = doc.inline_spans("a shell prompt looks like ` here")
    assert spans == (doc.Span(doc.TEXT, "a shell prompt looks like ` here"),)


def test_inline_spans_lose_only_delimiters() -> None:
    """Backticks vanish only when they were paired; nothing else does."""
    for text in ("`a`", "a`b`c", "a`b", "``", "`", "", "a", "`a`b`c`", "a ` b ` c"):
        spans = doc.inline_spans(text)
        if len(text.split("`")) % 2 == 0:  # an unpaired backtick: none is a delimiter
            assert spans == (doc.Span(doc.TEXT, text),)
        else:
            assert "".join(span.text for span in spans) == text.replace("`", "")


# --------------------------------------------------------------------------- #
# One document, two backends
# --------------------------------------------------------------------------- #


ALL_BLOCKS = _doc(
    doc.Heading("A heading", level=2),
    doc.Paragraph("A paragraph with `code` in it."),
    doc.Bullets(("first item", "second item")),
    doc.Table(headers=("left", "right"), rows=(("a", "b"),), caption="A caption"),
    doc.Note("A note."),
    doc.Code("literal | * _ <text>", language="bash"),
)


def test_every_string_reaches_the_markdown_rendering() -> None:
    rendered = md_backend.render(ALL_BLOCKS)
    for text in doc.document_strings(ALL_BLOCKS):
        # Code spans lose their backticks to the span split; compare on the visible text.
        for span in doc.inline_spans(text):
            assert span.text in rendered, f"markdown dropped {span.text!r}"


def test_every_string_reaches_the_html_rendering() -> None:
    from html import escape

    rendered = html_backend.render(ALL_BLOCKS)
    for text in doc.document_strings(ALL_BLOCKS):
        for span in doc.inline_spans(text):
            assert escape(span.text) in rendered, f"html dropped {span.text!r}"


def test_both_backends_are_deterministic() -> None:
    assert md_backend.to_bytes(ALL_BLOCKS) == md_backend.to_bytes(ALL_BLOCKS)
    assert html_backend.to_bytes(ALL_BLOCKS) == html_backend.to_bytes(ALL_BLOCKS)


def test_markdown_ends_in_exactly_one_newline() -> None:
    text = md_backend.render(ALL_BLOCKS)
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_empty_bullets_render_as_nothing_in_both_backends() -> None:
    empty = _doc(doc.Bullets(()))
    assert md_backend.render(empty).strip() == "# T"
    assert "<ul>" not in html_backend.render(empty)


def test_a_code_block_is_never_escaped_or_span_parsed() -> None:
    block = doc.Code("a | b `c` \\d")
    assert md_backend.render_block(block) == "```\na | b `c` \\d\n```"
    assert "a | b `c` \\d" in html_backend.render_block(block)


# --------------------------------------------------------------------------- #
# Escaping
# --------------------------------------------------------------------------- #


def test_a_pipe_in_a_cell_does_not_grow_a_column() -> None:
    table = doc.Table(headers=("k", "v"), rows=(("license", "MIT | CC0"),))
    line = next(
        row for row in md_backend.render_block(table).splitlines() if "license" in row
    )
    assert line.count("|") - line.count("\\|") == 3  # two borders + one separator


def test_a_newline_in_a_cell_does_not_end_the_row() -> None:
    table = doc.Table(headers=("k",), rows=(("two\nlines",),))
    assert "| two lines |" in md_backend.render_block(table)


def test_script_tags_survive_as_text_in_both_backends() -> None:
    payload = "<script>alert(1)</script>"
    block = doc.Paragraph(payload)
    assert "\\<script>" in md_backend.render_block(block)
    rendered = html_backend.render_block(block)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered


def test_a_link_a_curator_did_not_write_is_not_created() -> None:
    assert "\\[" in md_backend.render_block(doc.Paragraph("see [docs](http://x)"))


def test_boundary_underscores_are_escaped_and_intraword_ones_are_not() -> None:
    """CommonMark forbids intraword ``_`` emphasis; escaping ``qa_toy`` would be noise."""
    assert md_backend.escape("qa_toy and effective_n") == "qa_toy and effective_n"
    assert md_backend.escape("_emphasis_") == "\\_emphasis\\_"


def test_asterisks_are_always_escaped_because_they_emphasise_intraword() -> None:
    assert md_backend.escape("a*b*c") == "a\\*b\\*c"


def test_a_leading_marker_does_not_open_a_list() -> None:
    assert md_backend.render_block(doc.Paragraph("- not a list")).startswith("\\-")
    assert md_backend.render_block(doc.Paragraph("1. not a list")).startswith("1\\.")
    assert md_backend.render_block(doc.Bullets(("- nested?",))) == "- \\- nested?"


def test_code_spans_are_not_escaped() -> None:
    assert md_backend.inline("`a|b`") == "`a|b`"
    assert html_backend.inline("`a<b`") == "<code>a&lt;b</code>"


def test_html_title_is_plain_text_not_markup() -> None:
    rendered = html_backend.render(doc.Document(title="card — `qa_toy` v1", blocks=()))
    title = next(line for line in rendered.splitlines() if line.startswith("<title>"))
    assert title == "<title>card — qa_toy v1</title>"


def test_the_html_page_is_standalone_and_carries_no_scripts() -> None:
    rendered = html_backend.render(ALL_BLOCKS)
    assert "<script" not in rendered.lower()
    assert "<link" not in rendered.lower()
    assert "<style>" in rendered


# --------------------------------------------------------------------------- #
# Block validation
# --------------------------------------------------------------------------- #


def test_a_ragged_table_row_is_refused() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        doc.Table(headers=("a", "b"), rows=(("only-one",),))


def test_a_heading_outside_level_two_to_six_is_refused() -> None:
    with pytest.raises(ValueError, match="2..6"):
        doc.Heading("title", level=1)


def test_an_unknown_block_type_is_a_type_error() -> None:
    for backend in (md_backend, html_backend):
        with pytest.raises(TypeError):
            backend.render_block("not a block")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# build_card: the two absences that are arithmetic, not taste
# --------------------------------------------------------------------------- #


def test_card_is_invariant_to_manifest_hashes(tmp_path: Path) -> None:
    """``dataset_card.md`` is hashed into ``manifest.hashes``, so it cannot read them.

    If ``build_card`` touched ``hashes``, the release builder's provisional manifest (which
    has no card hash yet) and its final manifest (which does) would render different cards,
    and the shipped card would describe a manifest that was thrown away.
    """
    report, _ = _release(tmp_path)
    manifest, audit = report.manifest, report.audit
    doctored = manifest.model_copy(
        update={"hashes": {name: "sha256:" + "f" * 64 for name in manifest.hashes}}
    )
    assert build_card(doctored, audit) == build_card(manifest, audit)
    assert render_markdown(build_card(doctored, audit)) == render_markdown(report.card)


def test_card_is_invariant_to_created_at(tmp_path: Path) -> None:
    report, _ = _release(tmp_path)
    later = report.manifest.model_copy(update={"created_at": "2099-01-01T00:00:00+00:00"})
    assert build_card(later, report.audit) == build_card(report.manifest, report.audit)


def test_the_card_prints_no_timestamp_and_no_file_hash(tmp_path: Path) -> None:
    report, out = _release(tmp_path)
    card_text = (out / CARD_FILE).read_text(encoding="utf-8")
    assert FIXED_CLOCK not in card_text
    for digest in report.manifest.hashes.values():
        assert digest not in card_text
    # The identities are functions of the examples, not of the card, so they may appear.
    assert report.manifest.content_id in card_text
    assert report.manifest.eval_id in card_text


def test_two_clocks_differ_in_the_manifest_and_nothing_else(tmp_path: Path) -> None:
    """Milestone 10's determinism property, extended to cover the new artifact."""
    first, path_a = _release(tmp_path, "a", created_at="2020-01-01T00:00:00+00:00")
    second, path_b = _release(tmp_path, "b", created_at="2031-12-31T23:59:59+00:00")
    differing = {
        name
        for name in first.artifacts
        if (path_a / name).read_bytes() != (path_b / name).read_bytes()
    }
    assert differing == {MANIFEST_FILE}
    assert first.artifacts[CARD_FILE] == second.artifacts[CARD_FILE]


def test_a_card_refuses_to_mix_two_releases(tmp_path: Path) -> None:
    report, _ = _release(tmp_path)
    stranger = report.audit.model_copy(update={"version": "9.9.9"})
    with pytest.raises(ValueError, match="would mix two releases"):
        build_card(report.manifest, stranger)


# --------------------------------------------------------------------------- #
# build_card: content
# --------------------------------------------------------------------------- #


def test_the_card_reports_examples_and_independent_blocks(tmp_path: Path) -> None:
    """The question Milestone 10 left open: 14 examples are not 14 independent things."""
    report, out = _release(tmp_path)
    card = report.card
    assert card.splits["total_examples"] == 14
    assert card.splits["blocks"]["total"] == 11
    text = (out / CARD_FILE).read_text(encoding="utf-8")
    assert "14 examples in 11 independent blocks" in text
    assert "Sample size for an interval or a significance test is the block count" in text


def test_the_card_reports_blocks_per_split(tmp_path: Path) -> None:
    """A corpus-wide block count does not bound a claim about a test score."""
    report, _ = _release(tmp_path)
    by_split = report.card.splits["by_split"]
    assert by_split["test"]["examples"] == 2 and by_split["test"]["blocks"] == 2
    assert by_split["train"]["examples"] == 11 and by_split["train"]["blocks"] == 8
    # Adversarial splits evict duplicate clusters from dev/test, so nothing straddles.
    assert report.card.splits["blocks"]["straddling"] == 0


def test_a_straddling_block_is_counted_in_both_splits(tmp_path: Path) -> None:
    report, _ = _release(tmp_path, "random", strategy="random", seed=0, quarantine_blocking=False)
    dependence = report.manifest.dependence
    blocks = report.card.splits["blocks"]
    if dependence["straddling_block_count"]:
        assert sum(dependence["blocks_by_split"].values()) > blocks["total"]
        assert report.audit.split_statistics["counts"]["residual_overlap"] > 0


def test_known_issues_name_the_defects_the_fixture_plants(tmp_path: Path) -> None:
    report, _ = _release(tmp_path)
    issues = "\n".join(report.card.known_issues)
    assert "exact duplicate cluster" in issues and "qa-001 + qa-002" in issues
    assert "answer-in-prompt" in issues and "qa-007" in issues
    assert "missing gold answer" in issues and "qa-010" in issues
    assert "conflicting annotator labels" in issues and "qa-011" in issues


def test_zero_counts_are_in_the_table_and_not_in_the_bullets(tmp_path: Path) -> None:
    """A bullet saying "0 found" is a reassurance; a table row is a statement of what ran."""
    report, out = _release(tmp_path)
    assert report.card.audit_findings["leakage"]["ngram_overlap"] == 0
    assert not any("n-gram" in issue for issue in report.card.known_issues)
    text = (out / CARD_FILE).read_text(encoding="utf-8")
    assert "| leakage: n-gram overlap | 0 |" in text


def test_every_finding_row_declares_its_certainty(tmp_path: Path) -> None:
    report, out = _release(tmp_path)
    text = (out / CARD_FILE).read_text(encoding="utf-8")
    assert "| duplicates: exact | 1 | exact |" in text
    assert "| duplicates: near | 1 | heuristic (tfidf) |" in text
    assert "Every heuristic count is a lower bound" in text


def test_an_empty_audit_says_so_without_claiming_the_data_is_clean() -> None:
    audit = AuditResult(audit_id="a", dataset_id="d", version="0.1.0")
    manifest = DatasetManifest(
        dataset_id="d", version="0.1.0", created_at=FIXED_CLOCK, example_count=0
    )
    card = build_card(manifest, audit)
    assert card.known_issues == []
    text = render_markdown(card).decode("utf-8")
    assert "That is a statement about the checks, not about the dataset" in text


def test_card_counts_match_the_audit_exactly(tmp_path: Path) -> None:
    report, _ = _release(tmp_path)
    assert report.card.audit_findings["counts"] == report.audit.summary()
    assert report.card.audit_findings["recommendations"] == report.audit.recommendations
    assert report.card.limitations == report.manifest.known_limitations


def test_known_issues_truncate_long_id_lists() -> None:
    from eval_dataset_forge.reports.dataset_card import MAX_IDS
    from eval_dataset_forge.schemas import LabelAuditFinding

    audit = AuditResult(
        audit_id="a",
        dataset_id="d",
        version="0.1.0",
        label_warnings=[
            LabelAuditFinding(kind="missing", example_id=f"e-{i:02d}") for i in range(MAX_IDS + 3)
        ],
    )
    issue = known_issues(audit, "tfidf")[0]
    assert "… and 3 more" in issue


# --------------------------------------------------------------------------- #
# Curator prose
# --------------------------------------------------------------------------- #


def test_an_unwritten_section_is_named_rather_than_omitted(tmp_path: Path) -> None:
    report, out = _release(tmp_path)  # no card_meta
    assert report.card.missing_curator_sections() == list(CURATOR_SECTIONS)
    text = (out / CARD_FILE).read_text(encoding="utf-8")
    assert "0 of 4 curator sections are supplied" in text
    assert text.count("Not supplied. The curator did not state this") == 4


def test_supplied_prose_reaches_the_card_and_the_manifest(tmp_path: Path) -> None:
    meta = load_card_meta(QA_TOY_CARD_META)
    report, out = _release(tmp_path, card_meta=meta)
    assert report.card.missing_curator_sections() == []
    assert report.manifest.card_meta == meta
    text = (out / CARD_FILE).read_text(encoding="utf-8")
    assert "Not a measure of question-answering capability" in text
    assert "4 of 4 curator sections are supplied" in text


def test_a_typo_in_card_yaml_raises_rather_than_dropping_a_section(tmp_path: Path) -> None:
    path = tmp_path / "card.yaml"
    path.write_text("purpose: fine\nnon_goal: oops\n", encoding="utf-8")
    with pytest.raises(Exception, match="non_goal"):
        load_card_meta(path)


def test_an_empty_card_yaml_is_an_empty_card_meta(tmp_path: Path) -> None:
    path = tmp_path / "card.yaml"
    path.write_text("# nothing here\n", encoding="utf-8")
    assert load_card_meta(path) == CardMeta()


def test_a_non_mapping_card_yaml_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "card.yaml"
    path.write_text("- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_card_meta(path)


def test_curator_prose_cannot_inject_markup(tmp_path: Path) -> None:
    meta = CardMeta(purpose="<script>alert(1)</script> and a | pipe")
    report, _ = _release(tmp_path, card_meta=meta)
    markdown = render_markdown(report.card).decode("utf-8")
    rendered_html = render_html(report.card).decode("utf-8")
    assert "\\<script>" in markdown
    assert "<script>alert" not in rendered_html and "&lt;script&gt;" in rendered_html


# --------------------------------------------------------------------------- #
# Release integration
# --------------------------------------------------------------------------- #


def test_the_release_bundle_carries_a_hashed_card(tmp_path: Path) -> None:
    report, out = _release(tmp_path)
    assert (out / CARD_FILE).is_file()
    assert CARD_FILE in report.manifest.hashes
    assert MANIFEST_FILE not in report.manifest.hashes  # a file cannot hash itself


def test_a_shipped_card_regenerates_byte_for_byte_from_the_release(tmp_path: Path) -> None:
    """The property the two-pass manifest exists to make true."""
    _, out = _release(tmp_path, card_meta=load_card_meta(QA_TOY_CARD_META))
    assert render_markdown(card_from_release(out)) == (out / CARD_FILE).read_bytes()
    ok, message = check_card(out)
    assert ok, message


def test_verify_release_catches_a_corrupted_card(tmp_path: Path) -> None:
    _, out = _release(tmp_path)
    card_path = out / CARD_FILE
    card_path.write_bytes(card_path.read_bytes().replace(b"14 examples", b"99 examples"))
    report = verify_release(out)
    assert not report.ok
    assert report.checks["hashes_match"] is False
    assert any(CARD_FILE in problem for problem in report.problems)


def test_check_card_reports_drift_without_calling_the_release_corrupt(tmp_path: Path) -> None:
    """Two different questions. ``verify-release`` answers one; ``--check`` answers the other."""
    _, out = _release(tmp_path)
    manifest = load_manifest(out / MANIFEST_FILE)
    # Rewrite the card *and* its recorded hash: intact bundle, stale renderer.
    from eval_dataset_forge.datasets.writer import render_json, sha256_bytes

    new_bytes = (out / CARD_FILE).read_bytes() + b"\nan editor added this line\n"
    (out / CARD_FILE).write_bytes(new_bytes)
    manifest.hashes[CARD_FILE] = sha256_bytes(new_bytes)
    (out / MANIFEST_FILE).write_bytes(render_json(manifest.model_dump(mode="json")))

    assert verify_release(out).ok  # integrity holds
    ok, message = check_card(out)  # the renderer disagrees
    assert not ok and "the renderer has changed" in message


def test_check_card_on_a_release_without_one(tmp_path: Path) -> None:
    _, out = _release(tmp_path)
    (out / CARD_FILE).unlink()
    ok, message = check_card(out)
    assert not ok and "is not present" in message


def test_card_from_a_directory_that_is_not_a_release(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a release directory"):
        card_from_release(tmp_path)


def test_dropping_a_duplicate_changes_the_card_the_release_ships(tmp_path: Path) -> None:
    """The card describes the data that shipped, because the audit it reads does."""
    _, kept = _release(tmp_path, "kept")
    _, dropped = _release(tmp_path, "dropped", drop=["exact"])
    assert "1 exact duplicate cluster" in (kept / CARD_FILE).read_text(encoding="utf-8")
    text = (dropped / CARD_FILE).read_text(encoding="utf-8")
    assert "| duplicates: exact | 0 |" in text
    assert "exact duplicate cluster" not in text
    assert "examples excluded from source | 1" in text


# --------------------------------------------------------------------------- #
# Versioning: prose is packaging
# --------------------------------------------------------------------------- #


def test_card_meta_is_a_packaging_field() -> None:
    assert "card_meta" in _PACKAGING_FIELDS


def test_editing_the_prose_forces_a_patch_and_no_more(tmp_path: Path) -> None:
    """A non-goal is not a number. Rewriting one must not claim scores stopped comparing."""
    _, previous = _release(tmp_path, "v1")
    report = build_release(
        [QA_TOY],
        dataset_id="qa_toy",
        created_at=FIXED_CLOCK,
        version="auto",
        quarantine_blocking=True,
        card_meta=CardMeta(purpose="a purpose the first release did not state"),
        previous=previous,
    )
    assert report.delta is not None
    assert report.delta.bump == "patch"
    assert report.delta.scores_comparable
    assert report.manifest.version == "0.1.1"
    assert report.manifest.content_id == load_manifest(previous / MANIFEST_FILE).content_id


# --------------------------------------------------------------------------- #
# Rendering entry points
# --------------------------------------------------------------------------- #


def test_write_card_writes_only_what_is_asked(tmp_path: Path) -> None:
    report, _ = _release(tmp_path)
    written = write_card(report.card, html_path=tmp_path / "c.html")
    assert set(written) == {"html"}
    assert (tmp_path / "c.html").read_bytes().startswith(b"<!doctype html>")


def test_the_card_document_is_a_value(tmp_path: Path) -> None:
    report, _ = _release(tmp_path)
    assert card_document(report.card) == card_document(report.card)


def test_format_card_names_the_missing_sections(tmp_path: Path) -> None:
    report, _ = _release(tmp_path)
    text = format_card(report.card)
    assert "0/4 section(s) supplied" in text
    assert "11 independent" in text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_dataset_card_writes_markdown_and_html(tmp_path: Path, capsys) -> None:
    _, out = _release(tmp_path)
    md_path, html_path = tmp_path / "card.md", tmp_path / "card.html"
    code = main(
        [
            "dataset-card",
            "--release",
            str(out),
            "--out",
            str(md_path),
            "--html",
            str(html_path),
        ]
    )
    assert code == 0
    assert md_path.read_bytes() == (out / CARD_FILE).read_bytes()
    assert b"<table>" in html_path.read_bytes()
    assert "dataset card: qa_toy v0.1.0" in capsys.readouterr().out


def test_cli_dataset_card_prints_markdown_when_no_output_is_named(tmp_path: Path, capsys) -> None:
    _, out = _release(tmp_path)
    assert main(["dataset-card", "--release", str(out)]) == 0
    assert capsys.readouterr().out.startswith("# Dataset card — qa_toy v0.1.0")


def test_cli_dataset_card_json_is_machine_readable(tmp_path: Path, capsys) -> None:
    _, out = _release(tmp_path)
    assert main(["dataset-card", "--release", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_id"] == "qa_toy"
    assert payload["splits"]["blocks"]["total"] == 11
    assert payload["identity"]["content_id"].startswith("sha256:")


def test_cli_dataset_card_check_passes_on_a_fresh_release(tmp_path: Path, capsys) -> None:
    _, out = _release(tmp_path)
    assert main(["dataset-card", "--release", str(out), "--check"]) == 0
    assert capsys.readouterr().out.startswith("ok:")


def test_cli_dataset_card_check_exits_one_on_drift(tmp_path: Path, capsys) -> None:
    _, out = _release(tmp_path)
    (out / CARD_FILE).write_text("# a card someone hand-edited\n", encoding="utf-8")
    assert main(["dataset-card", "--release", str(out), "--check"]) == 1
    assert capsys.readouterr().out.startswith("drift:")


def test_cli_dataset_card_fail_on_incomplete(tmp_path: Path, capsys) -> None:
    _, bare = _release(tmp_path, "bare")
    assert main(["dataset-card", "--release", str(bare), "--fail-on-incomplete"]) == 1

    _, full = _release(tmp_path, "full", card_meta=load_card_meta(QA_TOY_CARD_META))
    assert main(["dataset-card", "--release", str(full), "--fail-on-incomplete"]) == 0


def test_cli_dataset_card_rejects_a_non_release_directory(tmp_path: Path, capsys) -> None:
    assert main(["dataset-card", "--release", str(tmp_path)]) == 2
    assert "not a release directory" in capsys.readouterr().err


def test_cli_release_records_card_meta_in_the_reproduce_command(tmp_path: Path) -> None:
    out = tmp_path / "rel"
    code = main(
        [
            "release",
            "--dataset",
            str(QA_TOY),
            "--dataset-id",
            "qa_toy",
            "--card-meta",
            str(QA_TOY_CARD_META),
            "--created-at",
            FIXED_CLOCK,
            "--out",
            str(out),
        ]
    )
    assert code == 0
    reproduce = (out / "REPRODUCE.md").read_text(encoding="utf-8")
    assert f"--card-meta {QA_TOY_CARD_META}" in reproduce
    assert "dataset_card.md` is present" in reproduce
    assert "`dataset_card.md` — Milestone 11" not in reproduce


# --------------------------------------------------------------------------- #
# The committed demo release
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not DEMO_RELEASE.is_dir(), reason="demo release not committed")
def test_the_committed_demo_release_still_renders_its_own_card() -> None:
    ok, message = check_card(DEMO_RELEASE)
    assert ok, message


@pytest.mark.skipif(not DEMO_RELEASE.is_dir(), reason="demo release not committed")
def test_the_committed_demo_card_is_complete_and_honest() -> None:
    card = card_from_release(DEMO_RELEASE)
    assert card.missing_curator_sections() == []
    assert card.splits["blocks"]["total"] == 11
    text = (DEMO_RELEASE / CARD_FILE).read_text(encoding="utf-8")
    assert "14 examples in 11 independent blocks" in text
    assert "Not a measure of question-answering capability" in text
