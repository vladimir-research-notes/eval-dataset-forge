"""Dataset card generation (Milestone 11).

A dataset card is the document a reader consults before trusting a benchmark. The failure
mode it exists to prevent is not absence — most datasets have a README — it is **drift**: a
card written by hand in week one, describing a corpus that has been re-cut four times since.

So this card is *derived*. :func:`build_card` takes exactly two inputs, the release's
``manifest.json`` and its ``audit.json``, and every count it prints is read from them. Nothing
is recomputed, which is deliberate: re-running the deduplication ladder at render time would
let a threshold change between the audit and the card, and the card would then disagree with
the ``audit.json`` sitting beside it in the same directory. The card is a *view*, and a view
that recomputes its own source is not a view.

**The two things a card cannot compute.** Purpose, construction rationale, recommended use,
and non-goals are claims about intent. They arrive through
:class:`~eval_dataset_forge.schemas.CardMeta`, recorded in the manifest — not read from a side
file at render time, so that a card regenerated from a release directory alone reproduces the
card that shipped inside it. Any the curator left blank are named in the rendered card rather
than quietly omitted; a reader must be able to tell "no restrictions" from "nobody wrote any
down".

**The two things a card must not print.** ``dataset_card.md`` is itself hashed into
``manifest.hashes``. It therefore cannot print that table (its own digest would have to appear
inside its own bytes), and it cannot print ``created_at`` — a clock reaching a hashed artifact
makes every rebuild differ from every other, which would collapse the ``none`` version bump
that :mod:`eval_dataset_forge.datasets.versioning` depends on. ``REPRODUCE.md`` omits the
timestamp for exactly this reason. Both absences are pinned by tests
(``test_card_is_invariant_to_manifest_hashes``, ``test_card_is_invariant_to_created_at``).

**The headline number is two numbers.** A card that says *"14 examples"* about a corpus holding
11 independent blocks is not lying, but a reader will spend that 14 as a sample size, and 14 is
not the sample size of anything. Both are printed, everywhere a count appears, and per split —
because the number that bounds a claim about a test score is the block count of the *test*
split, which is neither of them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
from eval_dataset_forge.schemas import (
    CURATOR_SECTIONS,
    AuditResult,
    CardMeta,
    DatasetCard,
    DatasetExample,
    DatasetManifest,
    DedupKind,
    LabelIssue,
    LeakageKind,
)

CARD_FILE = "dataset_card.md"

# How many example ids a "known issue" bullet names before it stops. A card is read; an
# ``audit.json`` is queried. Six ids fit on a line and are enough to start looking.
MAX_IDS = 6

# What each audit dimension can and cannot promise. The card prints this column beside every
# count, because "0 near-duplicates" and "0 exact duplicates" are claims of very different
# strength and a bare table implies they are the same claim.
_CERTAINTY: dict[str, str] = {
    "exact_duplicates": "exact",
    "normalized_duplicates": "exact after normalization",
    "near_duplicates": "heuristic",
    str(LeakageKind.TRAIN_TEST_OVERLAP): "inherits the dedup ladder (exact or heuristic)",
    str(LeakageKind.ANSWER_IN_PROMPT): "heuristic",
    str(LeakageKind.NGRAM_OVERLAP): "exact given the n-gram width",
    str(LeakageKind.RETRIEVAL_CONTAMINATION): "exact",
    str(LabelIssue.MISSING): "exact",
    str(LabelIssue.CONFLICTING): "exact",
    str(LabelIssue.AMBIGUOUS): "declaration, plus a heuristic",
    str(LabelIssue.INCONSISTENT_FORMAT): "exact",
    str(LabelIssue.NEEDS_ADJUDICATION): "derived from the rows above",
    "pii_flags": "heuristic (pattern match)",
}

# Row labels for the findings table.
_LEAKAGE_LABEL = {
    str(LeakageKind.TRAIN_TEST_OVERLAP): "leakage: train/test overlap",
    str(LeakageKind.ANSWER_IN_PROMPT): "leakage: answer in prompt",
    str(LeakageKind.NGRAM_OVERLAP): "leakage: n-gram overlap",
    str(LeakageKind.RETRIEVAL_CONTAMINATION): "leakage: retrieval contamination",
}

_LABEL_LABEL = {
    str(LabelIssue.MISSING): "labels: missing gold answer",
    str(LabelIssue.CONFLICTING): "labels: conflicting annotations",
    str(LabelIssue.AMBIGUOUS): "labels: ambiguous",
    str(LabelIssue.INCONSISTENT_FORMAT): "labels: inconsistent format",
    str(LabelIssue.NEEDS_ADJUDICATION): "labels: needs adjudication",
}

# Sentence fragments for the prose bullets, which need a grammatical object rather than a
# column heading. "2 example(s) with ambiguous" is not a sentence.
_LEAKAGE_PHRASE = {
    str(LeakageKind.TRAIN_TEST_OVERLAP): "train/test overlap finding(s)",
    str(LeakageKind.ANSWER_IN_PROMPT): "answer-in-prompt finding(s)",
    str(LeakageKind.NGRAM_OVERLAP): "n-gram overlap finding(s)",
    str(LeakageKind.RETRIEVAL_CONTAMINATION): "retrieval-contamination finding(s)",
}

_LABEL_PHRASE = {
    str(LabelIssue.MISSING): "a missing gold answer",
    str(LabelIssue.CONFLICTING): "conflicting annotator labels",
    str(LabelIssue.AMBIGUOUS): "an ambiguous expected output",
    str(LabelIssue.INCONSISTENT_FORMAT): "an inconsistent label format",
    str(LabelIssue.NEEDS_ADJUDICATION): "a label that needs adjudication",
}

_ABSENT = (
    "Not supplied. The curator did not state this; it is not a claim that the question has "
    "no answer. Pass `--card-meta <file.yaml>` to `evalforge release` to record it."
)


# --------------------------------------------------------------------------- #
# Curator prose
# --------------------------------------------------------------------------- #


def load_card_meta(path: str | Path) -> CardMeta:
    """Load curator prose from a YAML (or JSON) file.

    Validated against :class:`~eval_dataset_forge.schemas.CardMeta`, which forbids unknown
    fields: a ``non_goal:`` typo raises here rather than dropping the curator's non-goals on
    the floor and shipping a card that says nobody wrote any.
    """
    import yaml  # lazy: keeps the stdlib-only CLI commands importable without pyyaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return CardMeta()
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of card fields, got {type(raw).__name__}")
    return CardMeta(**raw)


# --------------------------------------------------------------------------- #
# Derived views over the audit
# --------------------------------------------------------------------------- #


def _split_counts(audit: AuditResult) -> dict[str, int]:
    counts = audit.split_statistics.get("counts", {})
    return {key: int(value) for key, value in counts.items()} if counts else {}


def _dedup_counts(audit: AuditResult) -> dict[str, int]:
    """Cluster counts per rung of the ladder.

    ``AuditResult`` has two buckets and the ladder has three rungs: ``normalized`` findings
    live under ``near_duplicates`` and keep their ``kind``. Split them back apart, because a
    normalized match is exact and a near match is not.
    """
    normalized = sum(1 for f in audit.near_duplicates if f.kind is DedupKind.NORMALIZED)
    return {
        "exact_duplicates": len(audit.exact_duplicates),
        "normalized_duplicates": normalized,
        "near_duplicates": len(audit.near_duplicates) - normalized,
    }


def _by_kind(findings: Iterable[Any], vocabulary: Iterable[Any]) -> dict[str, int]:
    """Count findings per kind, including the kinds that fired zero times.

    The zeros are the point. A card listing only what was found cannot be distinguished from
    a card whose pipeline never ran that check.
    """
    counts = dict.fromkeys((str(kind) for kind in vocabulary), 0)
    for finding in findings:
        counts[str(finding.kind)] += 1
    return counts


def _ids(values: Sequence[str]) -> str:
    """Render up to :data:`MAX_IDS` example ids as code spans, then a tail count."""
    shown = ", ".join(f"`{value}`" for value in values[:MAX_IDS])
    remainder = len(values) - MAX_IDS
    return f"{shown} … and {remainder} more" if remainder > 0 else shown


def _cluster_ids(findings: Sequence[Any]) -> str:
    return _ids([" + ".join(finding.example_ids) for finding in findings])


def known_issues(audit: AuditResult, near_method: str) -> list[str]:
    """The defects the pipeline actually found, most structural first. Zeros are omitted.

    Zero-count dimensions are absent here and present in the findings *table*: a bullet list
    of "0 of these, 0 of those" is a list of reassurances, and the table already says which
    checks ran.
    """
    issues: list[str] = []
    dedup = _dedup_counts(audit)
    method = near_method or "unspecified"

    if dedup["exact_duplicates"]:
        issues.append(
            f"{dedup['exact_duplicates']} exact duplicate cluster(s) — identical input and "
            f"gold answer: {_cluster_ids(audit.exact_duplicates)}."
        )
    normalized = [f for f in audit.near_duplicates if f.kind is DedupKind.NORMALIZED]
    if normalized:
        issues.append(
            f"{len(normalized)} cluster(s) identical after whitespace/case normalization: "
            f"{_cluster_ids(normalized)}."
        )
    near = [f for f in audit.near_duplicates if f.kind is DedupKind.NEAR]
    if near:
        issues.append(
            f"{len(near)} near-duplicate cluster(s) above the {method} threshold: "
            f"{_cluster_ids(near)}. Heuristic — a paraphrase below the threshold is not here."
        )

    for kind, phrase in _LEAKAGE_PHRASE.items():
        findings = [f for f in audit.leakage_flags if str(f.kind) == kind]
        if findings:
            affected = sorted({eid for f in findings for eid in f.example_ids})
            issues.append(
                f"{len(findings)} {phrase} covering {len(affected)} example(s): "
                f"{_ids(affected)}."
            )

    label_counts = _by_kind(audit.label_warnings, LabelIssue)
    for kind, count in label_counts.items():
        if not count:
            continue
        affected = sorted(f.example_id for f in audit.label_warnings if str(f.kind) == kind)
        issues.append(f"{count} example(s) with {_LABEL_PHRASE[kind]}: {_ids(affected)}.")

    if audit.pii_flags:
        issues.append(
            f"{len(audit.pii_flags)} PII-like pattern(s) flagged by the validation layer."
        )

    counts = _split_counts(audit)
    if counts.get("residual_overlap"):
        issues.append(
            f"{counts['residual_overlap']} duplicate cluster(s) still straddle a split "
            f"boundary: a test score on this release is partly a memorization score."
        )
    if counts.get("eval_duplicates"):
        issues.append(
            f"{counts['eval_duplicates']} example(s) in `dev`/`test` belong to a duplicate "
            f"cluster: they spend evaluation slots on repeated items and narrow every "
            f"confidence interval computed from them."
        )
    if counts.get("quarantined"):
        issues.append(
            f"{counts['quarantined']} example(s) are quarantined in `train` because their "
            f"gold answer is missing, conflicting, or unadjudicated. They are not deleted."
        )
    return issues


def _findings_rows(card: DatasetCard) -> tuple[tuple[str, ...], ...]:
    """Every audit dimension, its count, and what that count is worth.

    Derived from the card, not from the audit. The card already carries the counts and the
    certainty map, so a rendered card is a function of the model alone — which is what makes
    ``dataset-card --json`` a complete description of what the Markdown says.
    """
    findings = card.audit_findings
    certainty: Mapping[str, str] = findings.get("certainty", _CERTAINTY)
    duplicates: Mapping[str, int] = findings.get("duplicates", {})
    near_method = str(card.provenance.get("near_method") or "unspecified")

    table: list[tuple[str, Any, str]] = [
        ("duplicates: exact", duplicates.get("exact_duplicates", 0), certainty["exact_duplicates"]),
        (
            "duplicates: normalized",
            duplicates.get("normalized_duplicates", 0),
            certainty["normalized_duplicates"],
        ),
        (
            "duplicates: near",
            duplicates.get("near_duplicates", 0),
            f"{certainty['near_duplicates']} ({near_method})",
        ),
    ]
    table += [
        (_LEAKAGE_LABEL[kind], count, certainty[kind])
        for kind, count in findings.get("leakage", {}).items()
    ]
    table += [
        (_LABEL_LABEL[kind], count, certainty[kind])
        for kind, count in findings.get("labels", {}).items()
    ]
    pii = findings.get("counts", {}).get("pii_flags", 0)
    table.append(("PII-like patterns", pii, certainty["pii_flags"]))
    return rows(table)


def _schema_summary(audit: AuditResult, schema_version: str) -> str:
    """A one-paragraph description of what a row is, and which task families are present."""
    fields = ", ".join(f"`{name}`" for name in DatasetExample.model_fields)
    families = _family_counts(audit)
    listed = (
        ", ".join(f"`{name}` ({count})" for name, count in sorted(families.items()))
        or "none recorded"
    )
    return (
        f"Each row is a `DatasetExample` under schema `{schema_version}`, with fields "
        f"{fields}. Unknown fields are rejected at load time. Task families present: {listed}. "
        f"The formal definition is `docs/DATASET_SCHEMA.md`."
    )


def _family_counts(audit: AuditResult) -> dict[str, int]:
    distributions = audit.split_statistics.get("distributions", {})
    totals: dict[str, int] = {}
    for per_split in distributions.get("task_family", {}).values():
        for family, count in per_split.items():
            totals[family] = totals.get(family, 0) + int(count)
    return totals


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_card(manifest: DatasetManifest, audit: AuditResult) -> DatasetCard:
    """Assemble a :class:`~eval_dataset_forge.schemas.DatasetCard` from a release's evidence.

    A pure function of ``manifest`` and ``audit`` — and, within the manifest, deliberately
    blind to ``hashes`` and ``created_at``. Reading either would make the card unrenderable at
    the moment it must be rendered: the release builder needs the card's bytes *before* it can
    hash them into the manifest.

    Raises :class:`ValueError` when the audit describes a different release than the manifest.
    A card that mixed the two would be the exact defect this module exists to prevent, and it
    is cheaper to refuse than to detect later.
    """
    if (audit.dataset_id, audit.version) != (manifest.dataset_id, manifest.version):
        raise ValueError(
            f"audit describes {audit.dataset_id}@{audit.version} but manifest describes "
            f"{manifest.dataset_id}@{manifest.version}: the card would mix two releases"
        )

    meta = manifest.card_meta
    dependence: Mapping[str, Any] = manifest.dependence
    near_method = str(dependence.get("near_method", ""))
    blocks_by_split: Mapping[str, int] = dependence.get("blocks_by_split", {})
    total = manifest.example_count

    by_split = {
        name: {
            "examples": int(info["count"]),
            "blocks": blocks_by_split.get(name),
            "share": round(int(info["count"]) / total, 4) if total else 0.0,
        }
        for name, info in sorted(manifest.splits.items())
    }
    spec = manifest.split_spec

    return DatasetCard(
        dataset_id=manifest.dataset_id,
        version=manifest.version,
        purpose=meta.purpose,
        construction=meta.construction,
        schema_summary=_schema_summary(audit, manifest.schema_version),
        splits={
            "total_examples": total,
            "by_split": by_split,
            "blocks": {
                "total": dependence.get("block_count"),
                "duplicate": dependence.get("duplicate_block_count"),
                "straddling": dependence.get("straddling_block_count"),
                "size_histogram": dependence.get("block_size_histogram", {}),
            },
            "strategy": spec.model_dump(mode="json") if spec else {},
        },
        known_issues=known_issues(audit, near_method),
        audit_findings={
            "counts": audit.summary(),
            "duplicates": _dedup_counts(audit),
            "leakage": _by_kind(audit.leakage_flags, LeakageKind),
            "labels": _by_kind(audit.label_warnings, LabelIssue),
            "split_statistics": _split_counts(audit),
            "difficulty_distribution": dict(audit.difficulty_distribution),
            "recommendations": list(audit.recommendations),
            "certainty": dict(_CERTAINTY),
        },
        limitations=list(manifest.known_limitations),
        recommended_use=meta.recommended_use,
        non_goals=list(meta.non_goals),
        identity={
            "content_id": manifest.content_id,
            "eval_id": manifest.eval_id,
            "source_content_id": manifest.source_content_id,
            "schema_version": manifest.schema_version,
            "audit_id": audit.audit_id,
        },
        provenance={
            "source_files": list(manifest.source_files),
            "excluded": len(manifest.excluded),
            "tool_version": manifest.tool_version,
            "license_notes": manifest.license_notes,
            "near_method": near_method,
        },
    )


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #


def _prose(text: str) -> Paragraph:
    return Paragraph(text.strip()) if text.strip() else Paragraph(_ABSENT)


def _glance_table(card: DatasetCard) -> Table:
    blocks = card.splits.get("blocks", {})
    total = card.splits.get("total_examples", 0)
    block_count = blocks.get("total")
    size = (
        f"{total} examples in {block_count} independent blocks"
        if block_count is not None
        else f"{total} examples"
    )
    sizes = " · ".join(
        f"{name} {info['examples']}" for name, info in card.splits.get("by_split", {}).items()
    )
    strategy = card.splits.get("strategy", {})
    return Table(
        headers=("field", "value"),
        rows=rows(
            [
                ("dataset", f"`{card.dataset_id}`"),
                ("version", f"`{card.version}`"),
                ("size", size),
                ("splits", sizes or "not split"),
                (
                    "split strategy",
                    f"`{strategy.get('strategy', 'n/a')}` (seed {strategy.get('seed', 'n/a')})",
                ),
                ("`content_id`", f"`{card.identity.get('content_id', '')}`"),
                ("`eval_id`", f"`{card.identity.get('eval_id', '')}`"),
                ("schema", f"`{card.identity.get('schema_version', '')}`"),
                ("built by", f"`eval-dataset-forge {card.provenance.get('tool_version', '')}`"),
                ("license", card.provenance.get("license_notes") or "not stated"),
            ]
        ),
    )


def _splits_table(card: DatasetCard) -> Table:
    return Table(
        headers=("split", "examples", "independent blocks", "share"),
        rows=rows(
            [
                (
                    f"`{name}`",
                    info["examples"],
                    "—" if info["blocks"] is None else info["blocks"],
                    f"{info['share']:.1%}",
                )
                for name, info in card.splits.get("by_split", {}).items()
            ]
        ),
        caption=(
            "A block is a set of examples that must share a split: a duplicate cluster, or a "
            "source group. It is the unit a bootstrap should resample."
        ),
    )


def _blocks_note(card: DatasetCard) -> Note | None:
    blocks = card.splits.get("blocks", {})
    total = card.splits.get("total_examples", 0)
    block_count = blocks.get("total")
    if block_count is None or block_count == total:
        return None
    straddling = blocks.get("straddling")
    tail = (
        f" {straddling} block(s) straddle a split boundary."
        if straddling
        else " No block straddles a split boundary."
    )
    return Note(
        f"This release holds {total} examples but only {block_count} independent blocks: "
        f"{total - block_count} example(s) are duplicates or share a source record with "
        f"another. Sample size for an interval or a significance test is the block count, not "
        f"the example count. Resample blocks — `evalforge stats --resample-unit block`.{tail}"
    )


def _difficulty_table(card: DatasetCard) -> Table | None:
    distribution = card.audit_findings.get("difficulty_distribution", {})
    if not distribution:
        return None
    total = sum(distribution.values()) or 1
    return Table(
        headers=("bucket", "examples", "share"),
        rows=rows(
            [(f"`{name}`", count, f"{count / total:.1%}") for name, count in distribution.items()]
        ),
    )


def _provenance_table(card: DatasetCard) -> Table:
    sources = card.provenance.get("source_files", [])
    strategy = card.splits.get("strategy", {})
    ratios = strategy.get("ratios", {})
    requested = ", ".join(f"{name}={value}" for name, value in ratios.items()) or "n/a"
    return Table(
        headers=("field", "value"),
        rows=rows(
            [
                ("source files", ", ".join(f"`{path}`" for path in sources) or "none recorded"),
                ("examples excluded from source", card.provenance.get("excluded", 0)),
                ("`source_content_id`", f"`{card.identity.get('source_content_id', '')}`"),
                ("split strategy", f"`{strategy.get('strategy', 'n/a')}`"),
                ("split seed", strategy.get("seed", "n/a")),
                ("requested ratios", requested),
                ("near-duplicate method", f"`{card.provenance.get('near_method') or 'n/a'}`"),
            ]
        ),
    )


def card_document(card: DatasetCard) -> Document:
    """Lay the card out as a backend-independent :class:`Document`."""
    missing = card.missing_curator_sections()
    findings = card.audit_findings
    recommendations = findings.get("recommendations", [])

    completeness = (
        f"{len(CURATOR_SECTIONS) - len(missing)} of {len(CURATOR_SECTIONS)} curator sections "
        f"are supplied"
        + (f"; missing: {', '.join(f'`{name}`' for name in missing)}." if missing else ".")
    )

    return Document(
        title=f"Dataset card — {card.dataset_id} v{card.version}",
        blocks=blocks_of(
            Paragraph(
                "Generated from this release's `manifest.json` and `audit.json`. Every count "
                "below is read from the audit that shipped with the data, never recomputed, so "
                "the card and the evidence beside it cannot disagree. Prose sections are "
                "supplied by the curator and are the only unverified claims here."
            ),
            _glance_table(card),
            _blocks_note(card),
            Heading("Purpose"),
            _prose(card.purpose),
            Heading("Construction"),
            _prose(card.construction),
            _provenance_table(card),
            Heading("Schema"),
            Paragraph(card.schema_summary),
            Heading("Splits"),
            _splits_table(card),
            Heading("Audit findings"),
            Table(
                headers=("dimension", "count", "certainty"),
                rows=_findings_rows(card),
                caption=(
                    "Every heuristic count is a lower bound. A zero on a heuristic row means "
                    "the check found nothing, not that there is nothing to find."
                ),
            ),
            Heading("Known issues"),
            Bullets(tuple(card.known_issues))
            if card.known_issues
            else Paragraph(
                "The audit found none of the defects it checks for. That is a statement about "
                "the checks, not about the dataset — the heuristic dimensions in the table "
                "above have known blind spots, and every zero among them is an upper bound on "
                "what was looked for, not a guarantee of what is there."
            ),
            Heading("Recommendations"),
            Bullets(tuple(recommendations))
            if recommendations
            else Paragraph("The audit raised no recommendations for this release."),
            Heading("Difficulty distribution"),
            _difficulty_table(card) or Paragraph("No difficulty scores were recorded."),
            Note(
                "Buckets come from a transparent, model-free heuristic over input length, "
                "reasoning steps, evidence count, and ambiguity tags — never from measured "
                "model performance. Whether the heuristic's ranking is distinguishable from "
                "chance on this corpus is stated under Limitations and Recommendations above; "
                "see `docs/DIFFICULTY.md`."
            ),
            Heading("Limitations"),
            Bullets(tuple(card.limitations))
            if card.limitations
            else Paragraph("No limitations were recorded, which is itself a limitation."),
            Heading("Recommended use"),
            _prose(card.recommended_use),
            Heading("Non-goals"),
            Bullets(tuple(card.non_goals)) if card.non_goals else Paragraph(_ABSENT),
            Heading("Where these numbers came from"),
            Paragraph(
                f"Generated facts are read from `manifest.json` and `audit.json`; prose is the "
                f"curator's, recorded in `manifest.card_meta`. {completeness}"
            ),
            Bullets(
                (
                    "This card prints no file hashes. It is itself hashed into "
                    "`manifest.hashes`, so a digest of its own bytes cannot appear inside "
                    "them. Run `evalforge verify-release --release <dir>` to check every "
                    "artifact against the manifest, this card included.",
                    "This card prints no build timestamp. A clock inside a hashed artifact "
                    "would make each rebuild differ from the last, and a rebuild that changes "
                    "nothing must be able to say so.",
                    "`content_id` answers whether two releases hold the same examples; "
                    "`eval_id` answers whether two scores measured on them are comparable. "
                    "Re-splitting a corpus preserves the first and changes the second.",
                )
            ),
            Note(
                "Regenerate this card, or check it against the release it shipped in, with "
                "`evalforge dataset-card --release <dir> --check`."
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_markdown(card: DatasetCard) -> bytes:
    """The exact bytes of ``dataset_card.md``. Deterministic; no clock, no filesystem."""
    return md_backend.to_bytes(card_document(card))


def render_html(card: DatasetCard) -> bytes:
    """The exact bytes of a standalone ``dataset_card.html``.

    Not part of a release bundle. ``docs/PLAN.md`` §5D lists ``dataset_card.md`` and no HTML;
    a second rendering of the same facts inside a hashed bundle doubles the surface that can
    go stale and adds no information. The HTML is a CLI output, written where the caller asks.
    """
    return html_backend.to_bytes(card_document(card))


# --------------------------------------------------------------------------- #
# Release directory entry points
# --------------------------------------------------------------------------- #


def load_release(release_dir: str | Path) -> tuple[DatasetManifest, AuditResult]:
    """Load the two artifacts a card is built from. Raises if either is missing or invalid."""
    from eval_dataset_forge.datasets.manifest import (
        AUDIT_FILE,
        MANIFEST_FILE,
        load_audit,
        load_manifest,
    )

    release_dir = Path(release_dir)
    manifest_path, audit_path = release_dir / MANIFEST_FILE, release_dir / AUDIT_FILE
    for path in (manifest_path, audit_path):
        if not path.is_file():
            raise ValueError(f"{release_dir} is not a release directory: missing {path.name}")
    return load_manifest(manifest_path), load_audit(audit_path)


def card_from_release(release_dir: str | Path) -> DatasetCard:
    """Rebuild the card of an existing release from the release directory alone."""
    manifest, audit = load_release(release_dir)
    return build_card(manifest, audit)


def check_card(release_dir: str | Path) -> tuple[bool, str]:
    """Does this tool version still render the card that shipped in ``release_dir``?

    A different question from :func:`~eval_dataset_forge.datasets.manifest.verify_release`,
    and deliberately not folded into it. ``verify-release`` asks whether the bytes on disk are
    the bytes the manifest describes — a check that depends on nothing but the release, and
    that a reader with a different version of this tool can still run. This asks whether the
    *renderer* has changed, which is a question about the repository, not about the release.
    Answering it inside ``verify-release`` would make an old, perfectly intact release fail
    verification the day the card's wording improved.
    """
    release_dir = Path(release_dir)
    shipped_path = release_dir / CARD_FILE
    if not shipped_path.is_file():
        return False, f"{CARD_FILE} is not present in {release_dir}"
    regenerated = render_markdown(card_from_release(release_dir))
    shipped = shipped_path.read_bytes()
    if regenerated == shipped:
        return True, f"{CARD_FILE} matches the card regenerated from manifest.json + audit.json"
    return False, (
        f"{CARD_FILE} differs from the card this tool version renders "
        f"({len(shipped)} bytes on disk, {len(regenerated)} bytes regenerated). The release "
        f"is not corrupt — run verify-release for that — but the renderer has changed."
    )


def format_card(card: DatasetCard) -> str:
    """Render a one-screen summary of a card for a terminal."""
    blocks = card.splits.get("blocks", {})
    missing = card.missing_curator_sections()
    counts = card.audit_findings.get("counts", {})
    sizes = ", ".join(
        f"{name}={info['examples']}" for name, info in card.splits.get("by_split", {}).items()
    )
    lines = [
        f"dataset card: {card.dataset_id} v{card.version}",
        f"  examples   : {card.splits.get('total_examples', 0)} ({sizes})",
        f"  blocks     : {blocks.get('total', 'n/a')} independent "
        f"({blocks.get('duplicate', 0)} contain a duplicate cluster)",
        f"  content_id : {card.identity.get('content_id', '')}",
        f"  eval_id    : {card.identity.get('eval_id', '')}",
        "  audit      : " + ", ".join(f"{key}={value}" for key, value in counts.items()),
        f"  curator    : {len(CURATOR_SECTIONS) - len(missing)}/{len(CURATOR_SECTIONS)} "
        f"section(s) supplied" + (f" (missing: {', '.join(missing)})" if missing else ""),
    ]
    if card.known_issues:
        lines.append(f"\nknown issues ({len(card.known_issues)}):")
        lines += [f"  - {issue}" for issue in card.known_issues]
    return "\n".join(lines)


def write_card(
    card: DatasetCard,
    *,
    markdown_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write the requested renderings. Returns artifact name → path."""
    from eval_dataset_forge.datasets.writer import write_bytes

    written: dict[str, Path] = {}
    if markdown_path is not None:
        written["markdown"] = write_bytes(markdown_path, render_markdown(card))
    if html_path is not None:
        written["html"] = write_bytes(html_path, render_html(card))
    return written


def card_json(card: DatasetCard) -> str:
    """The card as pretty JSON — a complete description of what the Markdown says."""
    return json.dumps(card.model_dump(mode="json"), indent=2, ensure_ascii=False)


__all__ = [
    "CARD_FILE",
    "MAX_IDS",
    "build_card",
    "card_document",
    "card_from_release",
    "card_json",
    "check_card",
    "format_card",
    "known_issues",
    "load_card_meta",
    "load_release",
    "render_html",
    "render_markdown",
    "write_card",
]
