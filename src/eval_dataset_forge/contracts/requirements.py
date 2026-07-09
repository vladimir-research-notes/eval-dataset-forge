"""Requirements: the named questions a consumer may ask of a release (Milestone 13).

A requirement is a predicate over a :class:`~.reader.ReleaseHandle` that returns one of
**three** verdicts, not two.

``PASS``
    The check ran on data it could act on, and found nothing.
``FAIL``
    The check ran and found something.
``UNDECIDABLE``
    The check could not run, or the release does not carry the evidence to answer it.

The third verdict is the whole design. Milestone 12 established that on an unsplit
corpus four leakage signals return zero *by vacuity*, and printed a warning beside them.
A warning is advice; a boolean gate that maps ``leakage_flags == [] → PASS`` silently
converts vacuity into permission. So ``UNDECIDABLE`` is a distinct value, it never
counts as a pass, and a policy that wants to proceed anyway must say
``allow_undecidable: true`` in writing.

Two requirements here are undecidable on real releases, for two different reasons, and
the difference is worth keeping:

:func:`_no_retrieval_contamination`
    Undecidable on ``qa_toy`` because *no example carries retrieved passages*. The check
    is fine; the corpus gave it nothing to look at. Decidable on ``retrieval_toy``.
:func:`_difficulty_is_calibrated`
    Undecidable on **every** release this tool has ever built, because ``audit.json``
    records ``difficulty_distribution`` (the bucket counts) and not the permutation test
    that says whether those buckets mean anything. Milestone 8 computes that test,
    Milestone 10 renders it into an English sentence in ``recommendations``, and no
    machine-readable field survives. A consumer can recover it only by pattern-matching
    prose, which is not a contract. Milestone 11 already asked whether ``AuditResult``
    should carry the verdict — this requirement does not discover the gap, it makes it
    *executable* and adds a second reason to close it: not only does the dataset card
    print an unconditional caveat, no consumer can gate on the answer. The requirement is
    in the registry, in no profile, and its permanent ``UNDECIDABLE`` is a fact about the
    release format rather than about any dataset.

Every requirement also carries a ``cannot``: the claim a ``PASS`` does **not** license.
It is rendered next to the verdict, because a gate whose passes are read as guarantees
is a worse instrument than no gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from eval_dataset_forge.contracts.reader import ReleaseHandle
from eval_dataset_forge.schemas import LabelIssue, LeakageKind


class Verdict(StrEnum):
    """A requirement's outcome. ``UNDECIDABLE`` is never a pass."""

    PASS = "pass"
    FAIL = "fail"
    UNDECIDABLE = "undecidable"


# What a check returns before it is dressed as a RequirementResult.
Outcome = tuple[Verdict, str, dict[str, Any]]
Check = Callable[[ReleaseHandle, Mapping[str, Any]], Outcome]


@dataclass(frozen=True)
class Requirement:
    """One named, executable question, and the claim its ``PASS`` does not make.

    ``split_sensitive`` marks a requirement whose verdict can change when the same examples
    are re-split. It is not a detail: it decides which identity scopes the verdict. A
    content-scoped verdict survives a re-split, an eval-scoped one does not, and a gate that
    conflated them would carry a `--strategy random` pass onto a failing release — measured,
    on `qa_toy`, where seeds 9 and 35 pass ``eval_splits_deduplicated`` and 38 other seeds
    fail it, all 40 sharing one ``content_id``.
    """

    name: str
    question: str
    cannot: str
    check: Check
    params: tuple[str, ...] = ()
    split_sensitive: bool = False


@dataclass(frozen=True)
class RequirementResult:
    """A requirement's verdict on one release."""

    name: str
    verdict: Verdict
    detail: str
    cannot: str
    evidence: dict[str, Any] = field(default_factory=dict)
    split_sensitive: bool = False

    @property
    def blocking(self) -> bool:
        return self.verdict is Verdict.FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": str(self.verdict),
            "detail": self.detail,
            "cannot": self.cannot,
            "split_sensitive": self.split_sensitive,
            "evidence": dict(self.evidence),
        }


#: Thresholds a policy may override. Conventions, not theorems — the same status as
#: ``MIN_TOP1_STABILITY`` in Milestone 9 and ``MIN_CALIBRATION_N`` in Milestone 8.
DEFAULT_THRESHOLDS: dict[str, Any] = {
    # A near-duplicate cluster is a review item, not a defect, so the budget is not zero.
    # It is also corpus-relative (see `near_duplicates_within_budget`), which is why no
    # profile treats a low number as evidence of a clean corpus.
    "max_near_duplicate_clusters": 0,
    # Blocks, not examples. A test split of 200 rows in 3 duplicate clusters is a test
    # split of 3. There is no defensible universal value; 30 matches MIN_EFFECTIVE_N.
    "min_eval_blocks": 30,
    "eval_split": "test",
    "supported_schema_versions": ["0.1.0"],
    # None = "any method, but it must be recorded". A regression suite comparing two
    # releases over time should pin the string, because tfidf and jaccard do not agree.
    "expected_near_method": None,
}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _integrity(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    report = handle.verify()
    passed = sum(1 for ok in report.checks.values() if ok)
    evidence = {"checks_run": len(report.checks), "checks_passed": passed}
    if report.ok:
        return Verdict.PASS, f"{passed}/{len(report.checks)} integrity checks pass", evidence
    return Verdict.FAIL, "; ".join(report.problems), {**evidence, "problems": report.problems}


def _schema_compatible(handle: ReleaseHandle, params: Mapping[str, Any]) -> Outcome:
    supported = list(params["supported_schema_versions"])
    found = handle.manifest.schema_version
    evidence = {"schema_version": found, "supported": supported}
    if found in supported:
        return Verdict.PASS, f"schema {found} is supported", evidence
    return Verdict.FAIL, f"schema {found} is not in {supported}", evidence


def _no_exact_duplicates(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    clusters = handle.audit.exact_duplicates
    evidence = {"clusters": [list(c.example_ids) for c in clusters]}
    if not clusters:
        return Verdict.PASS, "no exact or normalized duplicate clusters", evidence
    ids = ", ".join(sorted(m for c in clusters for m in c.example_ids))
    return Verdict.FAIL, f"{len(clusters)} exact/normalized cluster(s): {ids}", evidence


def _near_duplicates_within_budget(handle: ReleaseHandle, params: Mapping[str, Any]) -> Outcome:
    budget = int(params["max_near_duplicate_clusters"])
    clusters = handle.audit.near_duplicates
    evidence = {
        "clusters": len(clusters),
        "budget": budget,
        "near_method": handle.near_method,
    }
    if len(clusters) <= budget:
        return Verdict.PASS, f"{len(clusters)} near cluster(s) ≤ budget {budget}", evidence
    return Verdict.FAIL, f"{len(clusters)} near cluster(s) > budget {budget}", evidence


def _no_answer_in_prompt(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    flags = handle.leakage(LeakageKind.ANSWER_IN_PROMPT)
    ids = sorted(i for flag in flags for i in flag.example_ids)
    evidence = {"example_ids": ids}
    if not flags:
        return Verdict.PASS, "no example contains its gold answer in the prompt", evidence
    return Verdict.FAIL, f"{len(ids)} example(s) leak the answer: {', '.join(ids)}", evidence


def _no_train_test_overlap(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    flags = handle.leakage(LeakageKind.TRAIN_TEST_OVERLAP)
    evidence = {"findings": len(flags)}
    if not flags:
        return Verdict.PASS, "no duplicate cluster straddles train and an eval split", evidence
    return Verdict.FAIL, f"{len(flags)} cluster(s) straddle train and an eval split", evidence


def _no_retrieval_contamination(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    """Undecidable when nothing in the release carries retrieved passages.

    ``audit.json`` says what the check found and never what it was handed, so this is the
    one requirement that reads ``examples.jsonl``.
    """
    bearing = handle.examples_with_retrieval_docs()
    flags = handle.leakage(LeakageKind.RETRIEVAL_CONTAMINATION)
    evidence = {"examples_with_retrieval_docs": bearing, "findings": len(flags)}
    if bearing == 0:
        return (
            Verdict.UNDECIDABLE,
            "no example carries metadata.retrieval.docs; the check had nothing to inspect, "
            "and its zero findings are vacuous",
            evidence,
        )
    if not flags:
        return Verdict.PASS, f"{bearing} example(s) inspected, no contamination", evidence
    return Verdict.FAIL, f"{len(flags)} contaminated passage finding(s)", evidence


def _no_label_conflicts(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    warnings = handle.label_warnings(LabelIssue.CONFLICTING)
    ids = sorted(w.example_id for w in warnings)
    evidence = {"example_ids": ids}
    if not warnings:
        return Verdict.PASS, "no example has annotators disagreeing on its gold label", evidence
    return Verdict.FAIL, f"{len(ids)} conflicting label(s): {', '.join(ids)}", evidence


def _no_missing_labels(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    warnings = handle.label_warnings(LabelIssue.MISSING)
    ids = sorted(w.example_id for w in warnings)
    evidence = {"example_ids": ids}
    if not warnings:
        return Verdict.PASS, "every example has a gold answer", evidence
    return Verdict.FAIL, f"{len(ids)} example(s) lack a gold answer: {', '.join(ids)}", evidence


def _no_pii(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    flags = handle.audit.pii_flags
    evidence = {"findings": len(flags)}
    if not flags:
        return Verdict.PASS, "no PII-like or secret-like pattern matched", evidence
    return Verdict.FAIL, f"{len(flags)} PII/secret-like finding(s)", evidence


def _eval_splits_deduplicated(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    counts = handle.split_counts()
    missing = [key for key in ("eval_duplicates", "residual_overlap") if key not in counts]
    if missing:
        return (
            Verdict.UNDECIDABLE,
            f"audit.split_statistics.counts omits {', '.join(missing)}",
            {"counts": counts},
        )
    duplicates = int(counts["eval_duplicates"])
    residual = int(counts["residual_overlap"])
    evidence = {"eval_duplicates": duplicates, "residual_overlap": residual}
    if duplicates == 0 and residual == 0:
        return Verdict.PASS, "dev/test carry no duplicate cluster and none straddles", evidence
    return (
        Verdict.FAIL,
        f"{duplicates} duplicate(s) inside an eval split, {residual} straddling cluster(s)",
        evidence,
    )


def _eval_split_is_rankable(handle: ReleaseHandle, params: Mapping[str, Any]) -> Outcome:
    """Enough *independent* units in the eval split to separate two systems.

    Reads ``dependence.blocks_by_split``, never ``example_count``. A duplicate cluster
    contributes ``m`` rows and one unit's worth of information (Milestone 9), so a gate
    that counted rows would license a comparison the corpus cannot support.
    """
    split = str(params["eval_split"])
    minimum = int(params["min_eval_blocks"])
    blocks = handle.blocks_in(split)
    rows = handle.manifest.splits.get(split, {}).get("count")
    evidence = {"split": split, "blocks": blocks, "rows": rows, "min_eval_blocks": minimum}
    if blocks is None:
        return (
            Verdict.UNDECIDABLE,
            f"manifest.dependence has no block count for the {split!r} split",
            evidence,
        )
    if blocks >= minimum:
        return Verdict.PASS, f"{split}: {blocks} independent block(s) ≥ {minimum}", evidence
    return (
        Verdict.FAIL,
        f"{split}: {blocks} independent block(s) < {minimum} "
        f"({rows} row(s)); the ranking this suite reports will not be stable",
        evidence,
    )


def _near_method_recorded(handle: ReleaseHandle, params: Mapping[str, Any]) -> Outcome:
    """The release names the near-duplicate method that ran, and it is the expected one.

    ``expected_near_method: null`` accepts any recorded method. A consumer that compares
    two releases over time should pin it: ``tfidf`` and ``jaccard`` disagree about which
    examples are duplicates, so they disagree about the blocks, the adversarial split,
    and ``eval_id`` — and ``tfidf`` runs iff scikit-learn happened to be importable.
    """
    found = handle.near_method
    expected = params.get("expected_near_method")
    evidence = {"near_method": found, "expected": expected}
    if found == "unknown":
        return (
            Verdict.UNDECIDABLE,
            "manifest.dependence records no near_method; this release cannot say which "
            "duplicate detector produced its blocks",
            evidence,
        )
    if expected is None or found == expected:
        return Verdict.PASS, f"near-duplicate method recorded as {found!r}", evidence
    return Verdict.FAIL, f"near_method is {found!r}, this consumer pinned {expected!r}", evidence


def _difficulty_is_calibrated(handle: ReleaseHandle, _params: Mapping[str, Any]) -> Outcome:
    """Permanently ``UNDECIDABLE``: the release format drops the answer.

    Milestone 8 runs a permutation test asking whether the difficulty heuristic's ranking
    beats a random ranking of the declared tags. Milestone 10 turns the result into an
    English recommendation. ``audit.json`` keeps ``difficulty_distribution`` — three
    bucket counts — and nothing else. Buckets exist whether or not they mean anything, so
    a consumer reading this field cannot tell a calibrated heuristic from noise.

    Milestone 11's review note asked whether the field should exist. This is not a new
    finding; it is the same gap seen from the consumer's side, where it costs a verdict
    rather than a caveat. The requirement is in no profile. It is here so the gap is
    executable rather than a sentence in a document, and so it fails loudly the day the
    field is added.
    """
    distribution = dict(handle.audit.difficulty_distribution)
    return (
        Verdict.UNDECIDABLE,
        "audit.json records difficulty_distribution but no calibration verdict; the "
        "permutation test is computed at build time and survives only as prose in "
        "audit.recommendations",
        {"difficulty_distribution": distribution},
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_ALL = (
    Requirement(
        name="integrity",
        question="Do the release's bytes match the manifest that describes them?",
        cannot="establish authenticity — manifest.json cannot hash itself, so anyone able "
        "to rewrite the data can rewrite the digests describing it",
        check=_integrity,
    ),
    Requirement(
        name="schema_compatible",
        question="Was the release built against a schema version this consumer parses?",
        cannot="detect an additive key this consumer ignores but should be reading",
        check=_schema_compatible,
        params=("supported_schema_versions",),
    ),
    Requirement(
        name="no_exact_duplicates",
        question="Are all exact and normalized duplicate clusters gone?",
        cannot="say anything about paraphrases; this check compares normalized strings",
        check=_no_exact_duplicates,
    ),
    Requirement(
        name="near_duplicates_within_budget",
        question="Are there at most `max_near_duplicate_clusters` heuristic near-duplicates?",
        cannot="bound the true duplicate count. The count is a lower bound (deep "
        "paraphrases are invisible) and it is corpus-relative: under tfidf the same rows "
        "yield a different count once other data is appended",
        check=_near_duplicates_within_budget,
        params=("max_near_duplicate_clusters",),
    ),
    Requirement(
        name="no_answer_in_prompt",
        question="Is any example's gold answer quoted in its own prompt?",
        cannot="detect an answer that is entailed by the prompt rather than copied into it",
        check=_no_answer_in_prompt,
    ),
    Requirement(
        name="no_train_test_overlap",
        question="Does a duplicate cluster straddle train and an evaluation split?",
        cannot="detect contamination from any corpus outside this release — including the "
        "training data of the model being evaluated, which is the contamination that "
        "usually matters",
        check=_no_train_test_overlap,
        split_sensitive=True,
    ),
    Requirement(
        name="no_retrieval_contamination",
        question="Do retrieved passages carry annotation artifacts or cross-split gold text?",
        cannot="detect a paraphrased gold passage; passage matching is exact after "
        "normalization",
        check=_no_retrieval_contamination,
        split_sensitive=True,
    ),
    Requirement(
        name="no_label_conflicts",
        question="Do annotators disagree on any example's gold label?",
        cannot="see a label on which every annotator is confidently wrong",
        check=_no_label_conflicts,
    ),
    Requirement(
        name="no_missing_labels",
        question="Does every example have a gold answer to score against?",
        cannot="judge whether a present gold answer is the right one",
        check=_no_missing_labels,
    ),
    Requirement(
        name="no_pii",
        question="Did any PII-like or secret-like pattern match?",
        cannot="certify privacy. It is a curated regex set over the text it was given; "
        "absence of a match is not absence of personal data",
        check=_no_pii,
    ),
    Requirement(
        name="eval_splits_deduplicated",
        question="Are dev/test free of duplicate clusters, and does none straddle a boundary?",
        cannot="protect against a duplicate the lexical near stage never found",
        check=_eval_splits_deduplicated,
        split_sensitive=True,
    ),
    Requirement(
        name="eval_split_is_rankable",
        question="Does the eval split hold `min_eval_blocks` independent blocks?",
        cannot="promise a stable ranking. It bounds the sample size from below; the "
        "minimum detectable effect also depends on the systems' scores, which a dataset "
        "does not have (`evalforge stats`)",
        check=_eval_split_is_rankable,
        params=("eval_split", "min_eval_blocks"),
        split_sensitive=True,
    ),
    Requirement(
        name="near_method_recorded",
        question="Does the release name the duplicate detector that produced its blocks?",
        cannot="verify the claim. Nothing in a release directory re-derives its own "
        "near_method; verifying it means rebuilding, which needs the build environment",
        check=_near_method_recorded,
        params=("expected_near_method",),
    ),
    Requirement(
        name="difficulty_is_calibrated",
        question="Is the difficulty heuristic's ranking better than a random one?",
        cannot="be answered from a release at all. The permutation test that answers it "
        "is computed at build time and kept only as prose",
        check=_difficulty_is_calibrated,
    ),
)

REQUIREMENTS: dict[str, Requirement] = {requirement.name: requirement for requirement in _ALL}


def resolve_thresholds(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """:data:`DEFAULT_THRESHOLDS` updated by ``overrides``; unknown keys raise."""
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(
            f"unknown threshold(s): {', '.join(sorted(unknown))}. "
            f"known: {', '.join(sorted(DEFAULT_THRESHOLDS))}"
        )
    return {**DEFAULT_THRESHOLDS, **overrides}


def evaluate(name: str, handle: ReleaseHandle, thresholds: Mapping[str, Any]) -> RequirementResult:
    """Run one requirement against a release."""
    requirement = REQUIREMENTS.get(name)
    if requirement is None:
        raise ValueError(f"unknown requirement {name!r}; known: {', '.join(sorted(REQUIREMENTS))}")
    verdict, detail, evidence = requirement.check(handle, thresholds)
    return RequirementResult(
        name=name,
        verdict=verdict,
        detail=detail,
        cannot=requirement.cannot,
        evidence=evidence,
        split_sensitive=requirement.split_sensitive,
    )


def evaluate_all(
    names: Sequence[str], handle: ReleaseHandle, thresholds: Mapping[str, Any]
) -> tuple[RequirementResult, ...]:
    """Run each named requirement, in the order given."""
    return tuple(evaluate(name, handle, thresholds) for name in names)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "REQUIREMENTS",
    "Requirement",
    "RequirementResult",
    "Verdict",
    "evaluate",
    "evaluate_all",
    "resolve_thresholds",
]
