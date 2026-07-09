"""The smoke audit: the whole pipeline over every fixture, scored (Milestone 12).

``evalforge smoke`` runs validation, deduplication, leakage, label audit, and difficulty over each
committed fixture and over the pooled corpus, writes an ``audit.json`` + ``audit.md`` per fixture,
and scores every finding against the defects that fixture deliberately plants. It is the
end-to-end demonstration that the checks work, and it is fully offline: no network, no paid API,
no model.

**The demonstration is not the recall number.** Every fixture's defects were planted by the same
repository that wrote the detectors, and the near-duplicate and answer-in-prompt thresholds were
*chosen* by sweeps over these fixtures in Milestones 4 and 5. A high recall here is a
resubstitution estimate. It says the pipeline finds what it was built to find, on the data it was
built against, and it says nothing whatsoever about a corpus this repository has not seen. Any
report that printed only that number would be committing, in miniature, the exact error this
project exists to detect in other people's benchmarks.

So the report prints three things beside it:

1. **The flag-everything control.** :func:`~eval_dataset_forge.scoring.score_degenerate` puts a
   detector that flags every example and every pair through the identical scorer. It scores recall
   ``1.000`` on every fixture. Recall alone cannot distinguish the pipeline from it; precision
   can, and does, by one to two orders of magnitude.
2. **The unexplained assertions.** Findings the ground truth cannot account for anywhere. This is
   the number that would catch a detector inventing defects, and it is small and enumerable.
3. **The denominator.** Every planted defect excluded from the recall calculation is listed by id,
   with the reason — the fixture declared it undetectable, or this scorer wires no channel to it.
   A recall whose denominator is chosen by the same people who wrote the detectors must be
   checkable, not merely reported.

**The pooled corpus is audited and not scored.** Pooling the five fixtures lets the deduplication
ladder compare examples across them, and the per-fixture ground truth says nothing about a
cross-fixture pair — neither that one exists nor that one does not. Scoring it would measure the
ground truth's silence. The pooled audit is reported for its structure (examples, independent
blocks, the dependence a bootstrap needs), which is what Milestones 7 and 9 pool the corpus for.

Running it earns its keep anyway. The pooled ladder finds one duplicate cluster the per-fixture
ladders do not, and it is not a cross-fixture one: ``{saf-004, saf-005, saf-007}`` lies entirely
inside ``safety_toy``. TF-IDF's inverse-document-frequency term is a function of the *collection*,
so the boilerplate two safety prompts share is common within their fixture and rare against the
pooled vocabulary, and ``sim(saf-004, saf-007)`` moves from 0.287 to 0.344 across a 0.30 threshold
without either prompt changing. **A near-duplicate verdict is a property of a pair and its corpus,
not of the pair.** Deduplication is therefore not composable, and the pooled audit is the only
place in this repository where that is visible (``docs/RESULTS.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eval_dataset_forge import SCHEMA_VERSION, __version__
from eval_dataset_forge.audit import AuditReport, run_audit, write_audit
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.reports import html as html_backend
from eval_dataset_forge.reports import markdown as md_backend
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
from eval_dataset_forge.schemas import DatasetExample
from eval_dataset_forge.scoring import (
    DOC_CHANNELS,
    EXAMPLE_CHANNELS,
    PAIR_CHANNELS,
    ChannelScore,
    GroundTruth,
    ScoreReport,
    load_ground_truth,
    score,
    score_degenerate,
)
from eval_dataset_forge.validation import ValidationReport, validate_dataset

DEFAULT_FIXTURES_DIR = "datasets/fixtures"
DEFAULT_AUDITS_DIR = "results/audits"
DEFAULT_REPORT_PATH = "results/reports/smoke_audit.md"

# The pooled corpus is a sixth audited dataset, not a fixture. It gets its own id so its artifacts
# never overwrite a fixture's.
POOLED_ID = "all_fixtures"

# Every channel, in the order a reader meets the pipeline.
ALL_CHANNELS = (*PAIR_CHANNELS, *EXAMPLE_CHANNELS, *DOC_CHANNELS)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


@dataclass
class FixtureOutcome:
    """One fixture: what it holds, what the audit found, and how that scored."""

    dataset_id: str
    path: str
    examples: list[DatasetExample]
    validation: ValidationReport
    audit: AuditReport
    truth: GroundTruth
    scored: ScoreReport
    degenerate: ScoreReport

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "path": self.path,
            "validation": self.validation.counts(),
            "audit": self.audit.counts(),
            "score": self.scored.to_dict(),
            "degenerate_recall": self.degenerate.recall(),
        }


@dataclass
class SmokeReport:
    """The whole corpus: per-fixture outcomes, the pooled audit, and the corpus totals."""

    fixtures: tuple[FixtureOutcome, ...] = ()
    pooled: AuditReport | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # -- derived views ----------------------------------------------------- #

    @property
    def n_examples(self) -> int:
        return sum(len(outcome.examples) for outcome in self.fixtures)

    def totals(self) -> dict[str, int]:
        keys = (
            "planted",
            "detectable",
            "recovered",
            "recovered_any_channel",
            "missed",
            "excluded",
            "controls",
            "controls_violated",
        )
        totals = dict.fromkeys(keys, 0)
        for outcome in self.fixtures:
            counts = outcome.scored.counts()
            for key in keys:
                totals[key] += counts[key]
        totals["examples"] = self.n_examples
        totals["fixtures"] = len(self.fixtures)
        totals["unexplained"] = len(self.unexplained())
        return totals

    def recall(self, *, any_channel: bool = False) -> float | None:
        totals = self.totals()
        if not totals["detectable"]:
            return None
        key = "recovered_any_channel" if any_channel else "recovered"
        return totals[key] / totals["detectable"]

    def recall_over_all_planted(self) -> float | None:
        """Recall with *nothing* excluded from the denominator except the negative controls.

        The number a reader gets by refusing to accept the scorer's scope. Printed beside
        :meth:`recall` so the exclusions can be checked rather than trusted.
        """
        totals = self.totals()
        denominator = totals["planted"] - totals["controls"]
        if not denominator:
            return None
        return totals["recovered_any_channel"] / denominator

    def degenerate_recall(self) -> float | None:
        """The flag-everything detector's corpus recall.

        The number that empties recall of meaning: see the module docstring.
        """
        recovered = detectable = 0
        for outcome in self.fixtures:
            counts = outcome.degenerate.counts()
            recovered += counts["recovered"]
            detectable += counts["detectable"]
        return recovered / detectable if detectable else None

    def channel_totals(self, *, degenerate: bool = False) -> dict[str, ChannelScore]:
        """Corpus-wide per-channel scores, summed over fixtures."""
        merged: dict[str, ChannelScore] = {}
        for outcome in self.fixtures:
            source = outcome.degenerate if degenerate else outcome.scored
            for name, channel in source.channels.items():
                previous = merged.get(name)
                merged[name] = ChannelScore(
                    channel=name,
                    unit=channel.unit,
                    planted=channel.planted + (previous.planted if previous else 0),
                    recovered=channel.recovered + (previous.recovered if previous else 0),
                    predicted=channel.predicted + (previous.predicted if previous else 0),
                    unexplained=(previous.unexplained if previous else ()) + channel.unexplained,
                    corroborating=(
                        (previous.corroborating if previous else ()) + channel.corroborating
                    ),
                    missed=(previous.missed if previous else ()) + channel.missed,
                )
        return merged

    def unexplained(self) -> tuple[tuple[str, str, str], ...]:
        """``(fixture, channel, atom)`` for every assertion the ground truth cannot account for."""
        return tuple(
            (outcome.dataset_id, name, atom)
            for outcome in self.fixtures
            for name, channel in sorted(outcome.scored.channels.items())
            for atom in channel.unexplained
        )

    def excluded(self) -> tuple[tuple[str, str, str, str], ...]:
        """``(fixture, kind, members, reason)`` for every planted defect outside the denominator."""
        return tuple(
            (
                outcome.dataset_id,
                item.defect.kind,
                ", ".join(item.defect.members),
                item.status,
            )
            for outcome in self.fixtures
            for item in outcome.scored.excluded
        )

    def missed(self) -> tuple[tuple[str, str, str], ...]:
        """``(fixture, kind, members)`` for every detectable planted defect no channel recovered."""
        return tuple(
            (outcome.dataset_id, item.defect.kind, ", ".join(item.defect.members))
            for outcome in self.fixtures
            for item in outcome.scored.missed
        )

    def controls_violated(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (outcome.dataset_id, item.defect.kind, ", ".join(item.defect.members))
            for outcome in self.fixtures
            for item in outcome.scored.controls_violated
        )

    @property
    def ok(self) -> bool:
        """No planted defect went undetected by every channel, and no negative control fired."""
        return not self.missed() and not self.controls_violated()

    def to_dict(self) -> dict:
        return {
            "tool_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "totals": self.totals(),
            "recall": self.recall(),
            "recall_any_channel": self.recall(any_channel=True),
            "degenerate_recall": self.degenerate_recall(),
            "ok": self.ok,
            "fixtures": [outcome.to_dict() for outcome in self.fixtures],
            "pooled": self.pooled.counts() if self.pooled else None,
            "unexplained": [list(item) for item in self.unexplained()],
            "excluded": [list(item) for item in self.excluded()],
            "missed": [list(item) for item in self.missed()],
            "warnings": list(self.warnings),
        }


def run_smoke(
    fixtures_dir: str | Path = DEFAULT_FIXTURES_DIR,
    *,
    seed: int = 0,
    difficulty_resamples: int = 2_000,
) -> SmokeReport:
    """Audit and score every fixture, then audit the pooled corpus. Nothing is written.

    Deterministic and offline. A fixture without a ``defects.yaml`` is an error rather than an
    unscored pass: the smoke audit exists to be scored, and a silently unscored fixture would
    inflate nothing but confidence.
    """
    fixtures_dir = Path(fixtures_dir)
    outcomes: list[FixtureOutcome] = []
    pooled_examples: list[DatasetExample] = []
    warnings: list[str] = []

    for directory in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
        examples_path = directory / "examples.jsonl"
        defects_path = directory / "defects.yaml"
        if not examples_path.is_file():
            continue
        if not defects_path.is_file():
            raise ValueError(
                f"{directory} has no defects.yaml: the smoke audit scores every fixture against "
                f"planted ground truth, and an unscored fixture would look like a passing one"
            )

        examples = loader.load_examples(examples_path)
        report = run_audit(
            examples,
            dataset_id=directory.name,
            difficulty_resamples=difficulty_resamples,
            seed=seed,
            source_files=(str(examples_path),),
        )
        truth = load_ground_truth(defects_path)
        outcomes.append(
            FixtureOutcome(
                dataset_id=directory.name,
                path=str(examples_path),
                examples=examples,
                validation=validate_dataset(examples_path),
                audit=report,
                truth=truth,
                scored=score(report, truth),
                degenerate=score_degenerate(examples, truth),
            )
        )
        pooled_examples.extend(examples)
        warnings.extend(f"{directory.name}: {warning}" for warning in report.warnings)

    if not outcomes:
        raise ValueError(f"no fixtures found under {fixtures_dir}")

    pooled = run_audit(
        pooled_examples,
        dataset_id=POOLED_ID,
        difficulty_resamples=difficulty_resamples,
        seed=seed,
        source_files=tuple(outcome.path for outcome in outcomes),
    )
    return SmokeReport(fixtures=tuple(outcomes), pooled=pooled, warnings=tuple(warnings))


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #


def _corpus_table(report: SmokeReport) -> Table:
    return Table(
        headers=("fixture", "examples", "blocks", "duplicates", "leakage", "labels", "PII"),
        rows=rows(
            [
                (
                    f"`{outcome.dataset_id}`",
                    len(outcome.examples),
                    outcome.audit.dependence.get("block_count", "—"),
                    len(outcome.audit.result.exact_duplicates)
                    + len(outcome.audit.result.near_duplicates),
                    len(outcome.audit.result.leakage_flags),
                    len(outcome.audit.result.label_warnings),
                    len(outcome.audit.result.pii_flags),
                )
                for outcome in report.fixtures
            ]
            + (
                [
                    (
                        f"`{POOLED_ID}` (pooled)",
                        report.pooled.n_examples,
                        report.pooled.dependence.get("block_count", "—"),
                        len(report.pooled.result.exact_duplicates)
                        + len(report.pooled.result.near_duplicates),
                        len(report.pooled.result.leakage_flags),
                        len(report.pooled.result.label_warnings),
                        len(report.pooled.result.pii_flags),
                    )
                ]
                if report.pooled
                else []
            )
        ),
        caption=(
            "Counts are clusters and findings, not examples. The pooled row is audited and not "
            "scored: pooling lets the ladder compare examples across fixtures, and the per-fixture "
            "ground truth says nothing about a cross-fixture pair."
        ),
    )


def _recovery_table(report: SmokeReport) -> Table:
    body = [
        (
            f"`{outcome.dataset_id}`",
            outcome.scored.counts()["planted"],
            outcome.scored.counts()["detectable"],
            len(outcome.scored.recovered_own_channel),
            len(outcome.scored.recovered_any_channel),
            len(outcome.scored.missed),
            _fmt(outcome.degenerate.recall()),
        )
        for outcome in report.fixtures
    ]
    totals = report.totals()
    body.append(
        (
            "all five",
            totals["planted"],
            totals["detectable"],
            totals["recovered"],
            totals["recovered_any_channel"],
            totals["missed"],
            _fmt(report.degenerate_recall()),
        )
    )
    return Table(
        headers=(
            "fixture",
            "planted",
            "detectable",
            "recovered (own channel)",
            "recovered (any channel)",
            "missed",
            "flag-everything recall",
        ),
        rows=rows(body),
        caption=(
            "The last column is the control: a detector that flags every example and every pair, "
            "scored by the identical rule. It recovers every planted defect. Recall does not "
            "distinguish it from the pipeline; precision does."
        ),
    )


def _control_note(report: SmokeReport) -> Note:
    """State the gap the control opens, in the units the next table is printed in."""
    real = report.channel_totals()
    control = report.channel_totals(degenerate=True)
    ratios = [
        real[name].precision / control[name].precision
        for name in real
        if real[name].precision and control[name].precision
    ]
    spread = (
        f"Channel by channel, the pipeline's precision is {min(ratios):.0f}× to {max(ratios):.0f}× "
        f"the control's. "
        if ratios
        else ""
    )
    return Note(
        f"A detector that flags everything scores recall {_fmt(report.degenerate_recall())} on "
        f"this corpus. So does the pipeline. That is not a defect of the matching rule — subset "
        f"matching, exact-set matching, and the pairwise rule used here all award the control full "
        f"recall, because full recall is what it has. Recall bounds what a detector misses and "
        f"says nothing about what it invents. {spread}Read the precision column."
    )


def _channel_table(report: SmokeReport) -> Table:
    real = report.channel_totals()
    control = report.channel_totals(degenerate=True)
    body = []
    for name in ALL_CHANNELS:
        channel = real.get(name)
        if channel is None or (not channel.planted and not channel.predicted):
            continue
        degenerate = control.get(name)
        body.append(
            (
                f"`{name}`",
                channel.unit,
                channel.planted,
                channel.predicted,
                _fmt(channel.recall),
                _fmt(channel.precision),
                _fmt(degenerate.precision) if degenerate else "n/a",
                len(channel.unexplained),
            )
        )
    return Table(
        headers=(
            "channel",
            "unit",
            "planted",
            "asserted",
            "recall",
            "precision",
            "precision (flag-everything)",
            "unexplained",
        ),
        rows=rows(body),
        caption=(
            "Summed over the five fixtures. A cluster of size k asserts k(k-1)/2 pairs, so a "
            "duplicate finding is scored on the pairs it implies rather than as one prediction. "
            "`precision` is kind-specific and is `n/a` where the fixtures plant nothing of that "
            "kind — `0.0` would then be the only value any non-silent detector could score."
        ),
    )


def _unexplained_section(report: SmokeReport) -> tuple:
    unexplained = report.unexplained()
    if not unexplained:
        return (
            Paragraph(
                "Every assertion the pipeline made is accounted for by the planted ground truth."
            ),
        )
    return (
        Paragraph(
            f"{len(unexplained)} assertion(s) the ground truth cannot account for in any section. "
            f"Each is either a false positive or a defect nobody wrote down; `defects.yaml` "
            f"records what was planted, not an exhaustive labelling. These are the findings worth "
            f"a human's time — see `docs/RESULTS.md`."
        ),
        Table(
            headers=("fixture", "channel", "assertion"),
            rows=rows(unexplained),
        ),
    )


def _excluded_section(report: SmokeReport) -> tuple:
    excluded = report.excluded()
    if not excluded:
        return (Paragraph("No planted defect is excluded from the recall denominator."),)
    return (
        Paragraph(
            f"{len(excluded)} planted defect(s) are outside the recall denominator. They are "
            f"listed because a recall whose denominator is chosen by the authors of the detectors "
            f"has to be checkable. `not-auto-detectable` is the fixture's own declaration that no "
            f"heuristic here should find it; `no-channel` is a fact about this scorer — the "
            f"project wires no detector to that ground-truth kind. Collapsing the two would let a "
            f"missed defect be reclassified as an out-of-scope one."
        ),
        Table(
            headers=("fixture", "kind", "examples", "reason"),
            rows=rows(excluded),
        ),
        Note(
            f"Recall over the detectable defects is {_fmt(report.recall(any_channel=True))}. "
            f"Recall over every planted defect, refusing all the exclusions above, is "
            f"{_fmt(report.recall_over_all_planted())}."
        ),
    )


def _pooled_section(report: SmokeReport) -> tuple:
    if report.pooled is None:
        return ()
    dependence = report.pooled.dependence
    return (
        Table(
            headers=("field", "value"),
            rows=rows(
                [
                    ("examples", report.pooled.n_examples),
                    ("independent blocks", dependence.get("block_count", "—")),
                    (
                        "blocks containing a duplicate cluster",
                        dependence.get("duplicate_block_count", "—"),
                    ),
                    ("blocks straddling a split", dependence.get("straddling_block_count", "—")),
                    ("near-duplicate method", f"`{dependence.get('near_method', '')}`"),
                    ("`audit_id`", f"`{report.pooled.result.audit_id}`"),
                ]
            ),
        ),
        Note(
            "The block count, not the example count, is the sample size of an interval or a "
            "significance test computed on this corpus. No effective sample size is quoted: that "
            "needs an intraclass correlation, which needs system scores."
        ),
    )


def smoke_document(report: SmokeReport) -> Document:
    """Lay the smoke audit out as a backend-independent ``Document``. Deterministic; no clock."""
    totals = report.totals()
    controls = totals["controls"]
    return Document(
        title="Smoke audit — every fixture, every stage, scored",
        blocks=blocks_of(
            Paragraph(
                f"eval-dataset-forge {__version__}, schema {SCHEMA_VERSION}. The full audit "
                f"pipeline over {totals['fixtures']} committed fixtures ({totals['examples']} "
                f"examples), fully offline: no network, no paid API, no model. Every fixture "
                f"plants deliberate defects and records them in a machine-readable "
                f"`defects.yaml`; the audit's job is to rediscover them."
            ),
            Note(
                "Every number in this report is measured in sample. These fixtures' defects were "
                "planted by the same repository that wrote the detectors, and the near-duplicate "
                "and answer-in-prompt thresholds were selected by sweeps over these same fixtures "
                "(Milestones 4 and 5). The recall below is a resubstitution estimate: it says the "
                "pipeline finds what it was built to find, on the data it was built against. It "
                "does not estimate performance on a corpus this repository has not seen."
            ),
            Heading("Corpus"),
            _corpus_table(report),
            Heading("Recovery of planted defects"),
            _recovery_table(report),
            _control_note(report),
            Heading("Per channel"),
            _channel_table(report),
            Heading("Unexplained assertions"),
            *_unexplained_section(report),
            Heading("Excluded from the denominator"),
            *_excluded_section(report),
            Heading("Negative controls"),
            Paragraph(
                f"{controls} planted propert{'y' if controls == 1 else 'ies'} that a detector must "
                f"not flag: an unanswerable, over-specific query whose empty gold answer is "
                f"correct, planted so that a rule keyed on emptiness is caught flagging it. "
                f"{len(report.controls_violated())} violated."
            )
            if controls
            else Paragraph("No negative control is planted in this corpus."),
            Heading("Pooled corpus"),
            *_pooled_section(report),
            Heading("Verdict"),
            Bullets(
                (
                    f"{totals['recovered_any_channel']} of {totals['detectable']} detectable "
                    f"planted defects recovered by some channel; {totals['recovered']} by their "
                    f"own channel.",
                    f"{totals['missed']} planted defect(s) invisible to the entire pipeline.",
                    f"{totals['unexplained']} unexplained assertion(s) across the corpus.",
                    f"{totals['controls_violated']} negative control(s) violated.",
                    f"A detector that flags everything scores recall "
                    f"{_fmt(report.degenerate_recall())} under the same rule.",
                )
            ),
            Heading("What this report cannot tell you"),
            Bullets(
                (
                    "Whether the pipeline works on data it has not been tuned on. Nothing here "
                    "measures that; a held-out corpus would, and this repository has none.",
                    "Whether the fixtures' planted defects are representative of real benchmark "
                    "defects. They were written to exercise the checks.",
                    "Whether a defect exists that no fixture plants and no check looks for. By "
                    "construction, neither the ground truth nor the audit can see it.",
                    "Whether the gold answers are correct. Three deliberately mislabeled examples "
                    "are planted, all well-formed and unanimous, and none is found — by design, "
                    "and asserted by a regression test.",
                )
            ),
            Note(
                "Reproduce: `PYTHONPATH=src python3 -m eval_dataset_forge smoke --out "
                "results/reports/smoke_audit.md`. Deterministic and byte-identical across runs — "
                "this report carries no timestamp."
            ),
        ),
    )


def render_markdown(report: SmokeReport) -> bytes:
    """The exact bytes of ``smoke_audit.md``."""
    return md_backend.to_bytes(smoke_document(report))


def render_html(report: SmokeReport) -> bytes:
    """The exact bytes of a standalone ``smoke_audit.html``."""
    return html_backend.to_bytes(smoke_document(report))


def write_smoke(
    report: SmokeReport,
    out_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    audits_dir: str | Path | None = DEFAULT_AUDITS_DIR,
    html_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write the smoke report and, unless ``audits_dir`` is ``None``, every per-dataset audit."""
    from eval_dataset_forge.datasets.writer import write_bytes

    written = {"smoke_audit.md": write_bytes(out_path, render_markdown(report))}
    if html_path is not None:
        written["smoke_audit.html"] = write_bytes(html_path, render_html(report))
    if audits_dir is not None:
        audits_dir = Path(audits_dir)
        for outcome in report.fixtures:
            for name, path in write_audit(outcome.audit, audits_dir / outcome.dataset_id).items():
                written[f"{outcome.dataset_id}/{name}"] = path
        if report.pooled is not None:
            for name, path in write_audit(report.pooled, audits_dir / POOLED_ID).items():
                written[f"{POOLED_ID}/{name}"] = path
    return written


def format_report(report: SmokeReport, max_items: int = 10) -> str:
    """Render a :class:`SmokeReport` for a terminal."""
    totals = report.totals()
    lines = [
        f"smoke audit: {totals['fixtures']} fixture(s), {totals['examples']} examples "
        f"(evalforge {__version__}, schema {SCHEMA_VERSION})",
        f"  planted      : {totals['planted']} "
        f"({totals['detectable']} detectable, {totals['excluded']} excluded, "
        f"{totals['controls']} negative control(s))",
        f"  recovered    : {totals['recovered_any_channel']} by any channel "
        f"({totals['recovered']} by their own) — recall "
        f"{_fmt(report.recall(any_channel=True))}",
        f"  control      : a detector that flags everything scores recall "
        f"{_fmt(report.degenerate_recall())} under the same rule",
        f"  missed       : {totals['missed']}",
        f"  unexplained  : {totals['unexplained']} assertion(s) the ground truth cannot account "
        f"for",
        f"  controls     : {totals['controls_violated']} violated",
    ]
    if report.pooled is not None:
        lines.append(
            f"  pooled       : {report.pooled.n_examples} examples in "
            f"{report.pooled.dependence.get('block_count', '?')} independent blocks"
        )

    def section(title: str, items: Sequence[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title} ({len(items)}):")
        lines.extend(f"  - {item}" for item in items[:max_items])
        if len(items) > max_items:
            lines.append(f"  … and {len(items) - max_items} more")

    section("missed", [f"{f}: {k} [{m}]" for f, k, m in report.missed()])
    section("unexplained", [f"{f}: {c} asserts {a}" for f, c, a in report.unexplained()])
    section("controls violated", [f"{f}: {k} [{m}]" for f, k, m in report.controls_violated()])
    lines.append(
        "\nEvery number above is in-sample: these thresholds were tuned on these fixtures."
    )
    return "\n".join(lines)


__all__ = [
    "ALL_CHANNELS",
    "DEFAULT_AUDITS_DIR",
    "DEFAULT_FIXTURES_DIR",
    "DEFAULT_REPORT_PATH",
    "POOLED_ID",
    "FixtureOutcome",
    "SmokeReport",
    "format_report",
    "render_html",
    "render_markdown",
    "run_smoke",
    "smoke_document",
    "write_smoke",
]
