"""Label-quality audit (Milestone 6).

Audit dimension #5: *is the gold answer trustworthy?* Label noise puts a hard ceiling on
the accuracy any benchmark can measure and biases every comparison drawn from it. The
five checks below each isolate one way a label can fail to be trustworthy, and together
they emit all five :class:`~eval_dataset_forge.schemas.LabelIssue` kinds.

========================  =======================  ==========================================
check                     certainty                catches
========================  =======================  ==========================================
:func:`.audit.missing_labels`         exact        no gold answer to score against
:func:`.disagreement.label_conflicts` exact        two label sources that disagree
:func:`.audit.ambiguous_labels`       mixed        more than one defensible answer
:func:`.audit.inconsistent_formats`   heuristic    a gold answer unlike its family's
:func:`.adjudication.needs_adjudication` derived   an issue only a human can resolve
========================  =======================  ==========================================

Ambiguity is "mixed" because it fuses two different things: an ``ambiguous``
``difficulty_tag`` is an exact reading of a declaration, while the underspecified-prompt
and open-set-request signals are lexical heuristics. ``detail`` records which fired.

**Stage order is load-bearing**, unlike the leakage channels. ``missing`` runs first
because an absent answer has no format and must not enter the shape census that the
format check measures every other answer against. Everything else is independent.

Unlike leakage, the kinds are near-disjoint by construction: at most one finding per
example per kind. An example may still carry several kinds — ``agent-009`` is both
``MISSING`` and ``AMBIGUOUS``, exactly as its fixture ground truth plants it — so
:meth:`LabelAuditReport.counts` reports per-kind counts alongside the number of distinct
examples flagged.

**Nothing is corrected.** :func:`audit_labels` never edits a label; it produces findings,
an agreement summary, and a queue of decisions for a human, and the release builder
(Milestone 10) decides what ships.

**The honest boundary.** This stage sees *structure*, never *truth*. It cannot tell that
``qa-006`` answers "the chemical symbol for gold" with ``Ag``, that ``bio-009`` gives a
human somatic cell 23 chromosomes, or that ``saf-006`` refuses a benign question about
vaccines. All three are planted in the fixtures, all three are marked
``auto_detectable: false``, and all three are found by none of the checks here. A
mislabeled answer that is well-formed, unanimous, unambiguous, and correctly shaped is
invisible to every method in this file — and it is the label defect that does the most
damage to a benchmark. Catching it needs a second annotator or a model, and
``docs/LABEL_AUDIT.md`` says so rather than implying the audit is complete.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from eval_dataset_forge.labels.adjudication import (
    SEVERITY,
    AdjudicationItem,
    build_queue,
    conflict_resolutions,
    format_resolution,
    needs_adjudication,
    write_queue,
)
from eval_dataset_forge.labels.audit import (
    AMBIGUOUS_TAG,
    DOMINANT_MIN_SHARE,
    MAX_UNDERSPECIFIED_TOKENS,
    MIN_FAMILY_SIZE,
    WH_WORDS,
    FamilyExpectation,
    ambiguous_labels,
    answer_shape,
    declared_ambiguous,
    family_expectations,
    has_label,
    inconsistent_formats,
    missing_labels,
    open_set_request,
    underspecified_prompt,
)
from eval_dataset_forge.labels.disagreement import (
    AgreementSummary,
    Annotation,
    agreement,
    annotations,
    annotator_conflicts,
    label_conflicts,
    label_key,
    majority_value,
    relevance_conflicts,
    relevance_labelled,
    shared_prompt_conflicts,
)
from eval_dataset_forge.schemas import DatasetExample, LabelAuditFinding


@dataclass
class LabelAuditReport:
    """Everything the five label checks found on one dataset.

    The parameter fields record what actually ran — the ambiguity width, whether the
    lexical signals were enabled, and the per-family shape/vocabulary expectations the
    format check measured against — so a report explains itself without the reader
    having to reconstruct the call.
    """

    n_examples: int
    missing: list[LabelAuditFinding] = field(default_factory=list)
    conflicting: list[LabelAuditFinding] = field(default_factory=list)
    ambiguous: list[LabelAuditFinding] = field(default_factory=list)
    inconsistent_format: list[LabelAuditFinding] = field(default_factory=list)
    needs_adjudication: list[LabelAuditFinding] = field(default_factory=list)
    queue: list[AdjudicationItem] = field(default_factory=list)
    agreement: AgreementSummary = field(default_factory=AgreementSummary)
    # -- parameters actually used ------------------------------------------ #
    max_underspecified_tokens: int = MAX_UNDERSPECIFIED_TOKENS
    heuristic_ambiguity: bool = True
    expectations: dict[str, FamilyExpectation] = field(default_factory=dict)

    # -- derived views ----------------------------------------------------- #

    @property
    def findings(self) -> list[LabelAuditFinding]:
        """All findings: the four primary kinds, then the derived adjudication flags."""
        return [
            *self.missing,
            *self.conflicting,
            *self.ambiguous,
            *self.inconsistent_format,
            *self.needs_adjudication,
        ]

    @property
    def n_flagged_examples(self) -> int:
        """Distinct examples carrying ≥1 *primary* finding.

        ``NEEDS_ADJUDICATION`` is excluded: it is derived from the others and names no
        example they do not already name, so counting it would inflate nothing but the
        appearance of thoroughness. Equal to ``len(self.queue)``.
        """
        return len(
            {
                finding.example_id
                for finding in (
                    *self.missing,
                    *self.conflicting,
                    *self.ambiguous,
                    *self.inconsistent_format,
                )
            }
        )

    @property
    def blocking_example_ids(self) -> list[str]:
        """Examples with ≥1 issue the audit cannot propose a resolution for."""
        return [item.example_id for item in self.queue if item.blocking]

    def counts(self) -> dict[str, int]:
        return {
            "examples": self.n_examples,
            "missing": len(self.missing),
            "conflicting": len(self.conflicting),
            "ambiguous": len(self.ambiguous),
            "inconsistent_format": len(self.inconsistent_format),
            "needs_adjudication": len(self.needs_adjudication),
            "flagged_examples": self.n_flagged_examples,
            "queue": len(self.queue),
        }

    def to_dict(self) -> dict:
        """A plain-dict view suitable for ``json.dumps`` and for ``AuditResult``."""
        return {
            "counts": self.counts(),
            "params": {
                "max_underspecified_tokens": self.max_underspecified_tokens,
                "heuristic_ambiguity": self.heuristic_ambiguity,
            },
            "agreement": self.agreement.to_dict(),
            "expectations": {
                family: expectation.to_dict()
                for family, expectation in sorted(self.expectations.items())
            },
            "missing": [f.model_dump(mode="json") for f in self.missing],
            "conflicting": [f.model_dump(mode="json") for f in self.conflicting],
            "ambiguous": [f.model_dump(mode="json") for f in self.ambiguous],
            "inconsistent_format": [f.model_dump(mode="json") for f in self.inconsistent_format],
            "needs_adjudication": [f.model_dump(mode="json") for f in self.needs_adjudication],
            "queue": [item.to_dict() for item in self.queue],
        }


def audit_labels(
    examples: Iterable[DatasetExample],
    *,
    max_underspecified_tokens: int = MAX_UNDERSPECIFIED_TOKENS,
    heuristic_ambiguity: bool = True,
) -> LabelAuditReport:
    """Run all five label checks over ``examples``.

    Deterministic and offline. ``heuristic_ambiguity=False`` keeps only the examples
    their own author declared ambiguous, which is the right setting when a dataset's
    tags are trusted and lexical false positives are not wanted.
    """
    examples = list(examples)

    # `missing` first: an absent answer has no format, and letting `""` into the shape
    # census would both double-flag it and skew the census it is measured against.
    missing = missing_labels(examples)
    missing_ids = {finding.example_id for finding in missing}
    expectations = family_expectations(examples, missing_ids=missing_ids)

    annotator = annotator_conflicts(examples)
    shared_prompt = shared_prompt_conflicts(examples)
    relevance = relevance_conflicts(examples)
    conflicting = [*annotator, *shared_prompt, *relevance]

    ambiguous = ambiguous_labels(
        examples, max_tokens=max_underspecified_tokens, heuristics=heuristic_ambiguity
    )
    formats = inconsistent_formats(
        examples, missing_ids=missing_ids, expectations=expectations
    )

    queue = build_queue(
        examples,
        [*missing, *conflicting, *ambiguous, *formats],
        conflicts=conflict_resolutions(
            examples, annotator=annotator, shared_prompt=shared_prompt, relevance=relevance
        ),
        expectations=expectations,
    )

    return LabelAuditReport(
        n_examples=len(examples),
        missing=missing,
        conflicting=conflicting,
        ambiguous=ambiguous,
        inconsistent_format=formats,
        needs_adjudication=needs_adjudication(queue),
        queue=queue,
        agreement=agreement(examples),
        max_underspecified_tokens=max_underspecified_tokens,
        heuristic_ambiguity=heuristic_ambiguity,
        expectations=expectations,
    )


def format_report(report: LabelAuditReport, max_items: int = 20) -> str:
    """Render a human-readable summary of a :class:`LabelAuditReport`."""
    counts = report.counts()
    rate = report.agreement.rate
    lines = [
        f"labels: {counts['examples']} examples",
        f"  missing             : {counts['missing']}",
        f"  conflicting         : {counts['conflicting']} "
        f"(annotators, shared prompts, relevance channels)",
        f"  ambiguous           : {counts['ambiguous']} "
        f"(declared tag"
        + (
            f" + heuristic, <={report.max_underspecified_tokens} tokens)"
            if report.heuristic_ambiguity
            else " only)"
        ),
        f"  inconsistent format : {counts['inconsistent_format']}",
        f"  needs adjudication  : {counts['needs_adjudication']} (blocking)",
        f"  examples flagged    : {counts['flagged_examples']}",
        "  annotator agreement : "
        + (
            "n/a (no example carries ≥2 annotations)"
            if rate is None
            else f"{rate:.0%} over {report.agreement.n_annotated} annotated example(s)"
        ),
    ]
    for family, expectation in sorted(report.expectations.items()):
        shapes = ", ".join(f"{k}={v}" for k, v in sorted(expectation.shapes.items()))
        vocabulary = (
            f", vocabulary {list(expectation.vocabulary)}" if expectation.vocabulary else ""
        )
        lines.append(
            f"  family {family!r}: {expectation.n_labelled} labelled ({shapes}), "
            f"dominant {expectation.dominant_shape or 'none'}{vocabulary}"
        )

    def section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title} ({len(items)}):")
        for item in items[:max_items]:
            lines.append(f"  - {item}")
        if len(items) > max_items:
            lines.append(f"  … and {len(items) - max_items} more")

    def describe(finding: LabelAuditFinding) -> str:
        return f"[{finding.example_id}] {finding.detail}".rstrip()

    section("missing labels", [describe(f) for f in report.missing])
    section("conflicting labels", [describe(f) for f in report.conflicting])
    section("ambiguous labels", [describe(f) for f in report.ambiguous])
    section("inconsistent formats", [describe(f) for f in report.inconsistent_format])
    section(
        "adjudication queue",
        [
            f"[{item.example_id}] {'BLOCKING' if item.blocking else 'resolvable'} "
            f"{','.join(item.issues)}"
            + (f" → {item.proposed_resolution}" if item.proposed_resolution else "")
            for item in report.queue
        ],
    )
    return "\n".join(lines)


__all__ = [
    # orchestration
    "LabelAuditReport",
    "audit_labels",
    "format_report",
    # shared constants
    "AMBIGUOUS_TAG",
    "DOMINANT_MIN_SHARE",
    "MAX_UNDERSPECIFIED_TOKENS",
    "MIN_FAMILY_SIZE",
    "SEVERITY",
    "WH_WORDS",
    # re-exports: missing / ambiguous / format
    "FamilyExpectation",
    "ambiguous_labels",
    "answer_shape",
    "declared_ambiguous",
    "family_expectations",
    "has_label",
    "inconsistent_formats",
    "missing_labels",
    "open_set_request",
    "underspecified_prompt",
    # re-exports: conflicts & agreement
    "AgreementSummary",
    "Annotation",
    "agreement",
    "annotations",
    "annotator_conflicts",
    "label_conflicts",
    "label_key",
    "majority_value",
    "relevance_conflicts",
    "relevance_labelled",
    "shared_prompt_conflicts",
    # re-exports: adjudication
    "AdjudicationItem",
    "build_queue",
    "conflict_resolutions",
    "format_resolution",
    "needs_adjudication",
    "write_queue",
]
