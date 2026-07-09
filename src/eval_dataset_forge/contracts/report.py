"""Rendering a :class:`~.gate.GateReport` (Milestone 13).

Built as a :class:`~eval_dataset_forge.reports.document.Document` and rendered by the two
backends Milestone 11 already owns, so the Markdown a CI job archives and the HTML a human
opens cannot disagree.

Two rendering decisions carry an argument.

**The three verdicts are never collapsed into two.** ``UNDECIDABLE`` gets its own row
group and its own count, above the passes. A report that folded it into ``FAIL`` would
tell a curator to fix a dataset that has nothing wrong with it; one that folded it into
``PASS`` would be the bug this module exists to prevent.

**Every requirement prints what its pass does not establish.** The ``cannot`` column is
not a footnote — a gate report is read by whoever most wants it to say the dataset is
fine, and the sentence they most need is the one bounding what "fine" meant.
"""

from __future__ import annotations

from eval_dataset_forge.contracts.gate import GateReport
from eval_dataset_forge.contracts.profiles import get_profile
from eval_dataset_forge.contracts.requirements import REQUIREMENTS, Verdict
from eval_dataset_forge.reports.document import (
    Bullets,
    Document,
    Heading,
    Note,
    Paragraph,
    Table,
    blocks_of,
    rows,
)
from eval_dataset_forge.reports.markdown import render as render_markdown

_SYMBOL = {Verdict.PASS: "pass", Verdict.FAIL: "FAIL", Verdict.UNDECIDABLE: "undecidable"}


def _identity_table(report: GateReport) -> Table:
    if report.split_sensitive:
        sensitive = ", ".join(r.name for r in report.results if r.split_sensitive)
        caption = (
            f"This verdict is scoped to eval_id, because {sensitive} read the split "
            f"assignment. It transfers to no other corpus and to no other split of it."
        )
    else:
        caption = (
            "This verdict is scoped to content_id. It survives a re-split and transfers "
            "to no corpus that gained or lost a single example."
        )
    return Table(
        headers=("field", "value", "answers"),
        rows=rows(
            [
                ("`content_id`", f"`{report.content_id}`", "are these the same examples?"),
                ("`eval_id`", f"`{report.eval_id}`", "are two scores comparable?"),
                (
                    "`near_method`",
                    f"`{report.near_method}`",
                    "which duplicate detector produced the blocks?",
                ),
            ]
        ),
        caption=caption,
    )


def _results_table(report: GateReport) -> Table:
    return Table(
        headers=("requirement", "verdict", "detail"),
        rows=rows(
            [(f"`{r.name}`", _SYMBOL[r.verdict], r.detail) for r in report.results]
        ),
    )


def _cannot_table(report: GateReport) -> Table:
    return Table(
        headers=("requirement", "a pass does not establish"),
        rows=rows([(f"`{r.name}`", r.cannot) for r in report.results]),
    )


def _profile_blocks(report: GateReport) -> tuple:
    try:
        profile = get_profile(report.policy.profile)
    except ValueError:
        return ()
    return blocks_of(
        Heading("What this consumer must establish for itself", 3),
        Bullets(profile.assumes),
        Heading("What this profile deliberately does not check", 3),
        Bullets(profile.does_not_check),
    )


def gate_document(report: GateReport) -> Document:
    """The gate report as a backend-independent document."""
    verdict = "PASS" if report.ok else "FAIL"
    counts = (
        f"{len(report.passed)} pass, {len(report.failed)} fail, "
        f"{len(report.undecided)} undecidable"
    )
    undecided_note = (
        Note(
            "An undecidable requirement is not a passing one. This policy sets "
            "allow_undecidable, so the run proceeds anyway."
        )
        if report.undecided and report.policy.allow_undecidable
        else None
    )
    return Document(
        title=f"Gate report: {report.dataset_id} v{report.version} — {verdict}",
        blocks=blocks_of(
            Paragraph(
                f"Profile `{report.policy.profile}` against `{report.release_dir}`. "
                f"{counts}. Contract version {report.contract_version}, "
                f"eval-dataset-forge {report.tool_version}."
            ),
            Heading("Identity"),
            _identity_table(report),
            Heading("Requirements"),
            _results_table(report),
            undecided_note,
            Heading("Bounds"),
            Paragraph(
                "Each requirement below passed, failed, or could not run. None of them "
                "establishes the claim in the right-hand column."
            ),
            _cannot_table(report),
            Heading("Scope"),
            _profile_blocks(report),
        ),
    )


def format_gate(report: GateReport) -> str:
    """The gate report as Markdown."""
    return render_markdown(gate_document(report))


def format_profiles() -> str:
    """A listing of every committed profile and requirement, for ``gate --list``."""
    from eval_dataset_forge.contracts.profiles import PROFILES

    lines = ["Consumer profiles:", ""]
    for name in sorted(PROFILES):
        profile = PROFILES[name]
        lines.append(f"  {profile.name}")
        lines.append(f"    consumer     : {profile.consumer}")
        lines.append(f"    purpose      : {profile.purpose}")
        lines.append(f"    requirements : {', '.join(profile.requirements)}")
        if profile.thresholds:
            thresholds = ", ".join(f"{k}={v}" for k, v in sorted(profile.thresholds.items()))
            lines.append(f"    thresholds   : {thresholds}")
        lines.append("")
    lines.append("Requirements:")
    lines.append("")
    for name in sorted(REQUIREMENTS):
        lines.append(f"  {name}")
        lines.append(f"    {REQUIREMENTS[name].question}")
    return "\n".join(lines)


__all__ = ["format_gate", "format_profiles", "gate_document"]
