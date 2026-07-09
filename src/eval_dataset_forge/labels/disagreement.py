"""Label sources that disagree with one another (Milestone 6).

Everything in this module emits :attr:`~eval_dataset_forge.schemas.LabelIssue.CONFLICTING`.
A conflict is not a wrong label — it is *two label sources that cannot both be right*,
which is a strictly weaker and strictly checkable claim. Deciding which source wins is
a human's job; :mod:`.adjudication` queues the decision.

Three sources, in decreasing order of how often they fire in practice:

======================== ========================================================
source                   two things that disagree
======================== ========================================================
:func:`annotator_conflicts`   annotators of the same example
:func:`shared_prompt_conflicts` two examples that pose the identical question
:func:`relevance_conflicts`   an example's gold doc-id list and its own passage flags
======================== ========================================================

**Comparison is lexical, never semantic.** :func:`label_key` folds a string through
:func:`~eval_dataset_forge.dedup.fuzzy.normalize_text` and then encodes it with
:func:`~eval_dataset_forge.dedup.exact.canonical_output`, which every non-string value
goes through directly. So ``"1945"`` and ``" 1945 "`` agree while ``"1945"`` and ``1945``
do not — a string and an integer are different gold answers, and a dataset that mixes
them has a format defect rather than an agreement. The encoding step is what keeps them
apart: fold alone maps the string ``"4"`` and the integer ``4`` onto the same key.

Two annotators who write ``"a drug"`` and ``"pharmaceutical"`` are recorded as
disagreeing. Detecting that they mean the same thing needs a model, and
``docs/LABEL_AUDIT.md`` declares it out of scope.

**Why the relevance check lives here.** Retrieval examples label relevance twice: once
as ``expected_output`` (the gold doc-id list) and once as ``metadata.retrieval.docs[].relevant``.
Two encodings of one quantity, so they can contradict each other, and on
``retrieval_toy`` they do — ``ret-004`` marks ``d9`` relevant while listing no gold
documents. Milestone 5's :func:`~eval_dataset_forge.leakage.retrieval_overlap._is_relevant`
noticed the contradiction and handed it here. Crucially, the check compares *sets*: an
empty gold list agreeing with zero relevant passages (``ret-005``, an unanswerable
query) is agreement, not a missing label. That falls out of set equality rather than
from a special case.

**The agreement rate is a summary, not a coefficient.** :func:`agreement` reports the
raw fraction of multiply-annotated examples whose annotators agreed. Chance-corrected
coefficients (Cohen's/Fleiss' kappa) are a candidate addition; a raw rate is harder to
misread and needs no assumption about the label set's cardinality.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from eval_dataset_forge.dedup.exact import canonical_output, duplicate_input_clusters
from eval_dataset_forge.dedup.fuzzy import normalize_text
from eval_dataset_forge.leakage.retrieval_overlap import retrieval_docs
from eval_dataset_forge.schemas import DatasetExample, LabelAuditFinding, LabelIssue

# Where per-annotator labels live on an example. ``labels`` is free-form by contract,
# so every accessor below is defensive about shape rather than schema-enforcing.
ANNOTATIONS_KEY = "annotations"

# An annotation without an explicit annotator id still counts as a vote; it just cannot
# be attributed. Ids matter for reporting, not for detecting disagreement.
_ANONYMOUS = "?"


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Annotation:
    """One annotator's label for one example."""

    annotator: str
    value: Any

    def __str__(self) -> str:
        return f"{self.annotator}={self.value!r}"


def annotations(example: DatasetExample) -> list[Annotation]:
    """Per-annotator labels of an example, or ``[]`` when it carries none.

    Defensive about shape for the same reason
    :func:`~eval_dataset_forge.leakage.retrieval_overlap.retrieval_docs` is: ``labels``
    is a free-form dict, so a malformed ``annotations`` block yields no annotations
    instead of raising. An entry without a ``value`` key is dropped — it records that
    someone was asked, not what they said.
    """
    raw = example.labels.get(ANNOTATIONS_KEY)
    if not isinstance(raw, list):
        return []
    return [
        Annotation(annotator=str(item.get("annotator", _ANONYMOUS)), value=item["value"])
        for item in raw
        if isinstance(item, dict) and "value" in item
    ]


def label_key(value: Any) -> str:
    """A comparable key for one label value.

    Strings fold through :func:`normalize_text` (casing, punctuation, whitespace) and are
    *then* JSON-encoded, exactly as :func:`canonical_output` encodes them. The second step
    is what keeps a folded string from colliding with a scalar of another type: without it
    the string ``"4"`` and the integer ``4`` both key to ``4``, and two annotators who
    wrote different things would be recorded as agreeing. See the module docstring.
    """
    if isinstance(value, str):
        return canonical_output(normalize_text(value))
    return canonical_output(value)


def majority_value(items: Iterable[Annotation]) -> Any | None:
    """The value held by a *strict* majority of annotators, or ``None``.

    Strict means more than half, so a 2-way tie and a 1–1–1 three-way split both return
    ``None``: there is no defensible automatic resolution, and
    :mod:`.adjudication` will mark the example blocking. A 2–1 split returns the
    winner in its original (un-normalized) form.
    """
    items = list(items)
    if not items:
        return None
    counts = Counter(label_key(item.value) for item in items)
    key, count = counts.most_common(1)[0]
    if count * 2 <= len(items):
        return None
    return next(item.value for item in items if label_key(item.value) == key)


# --------------------------------------------------------------------------- #
# Agreement summary
# --------------------------------------------------------------------------- #


@dataclass
class AgreementSummary:
    """Raw inter-annotator agreement over the examples that carry ≥2 annotations.

    ``rate`` is ``None`` — not ``0.0`` — when no example carries enough annotations to
    evaluate. An undefined rate and a rate of zero mean opposite things.
    """

    n_annotated: int = 0
    n_agreed: int = 0
    n_disagreed: int = 0

    @property
    def rate(self) -> float | None:
        """Fraction of multiply-annotated examples whose annotators all agreed."""
        if not self.n_annotated:
            return None
        return self.n_agreed / self.n_annotated

    def to_dict(self) -> dict:
        return {
            "n_annotated": self.n_annotated,
            "n_agreed": self.n_agreed,
            "n_disagreed": self.n_disagreed,
            "rate": self.rate,
        }


def agreement(examples: Iterable[DatasetExample]) -> AgreementSummary:
    """Summarize annotator agreement across a dataset.

    Only examples with ≥2 annotations are evaluable; the rest are invisible here.

    On the committed fixtures this rate is **0.0**, because annotations were planted
    *only* on the examples whose conflicts are the ground truth. The number describes
    the fixtures' construction, not the reliability of any real annotation process.
    """
    summary = AgreementSummary()
    for example in examples:
        items = annotations(example)
        if len(items) < 2:
            continue
        summary.n_annotated += 1
        if len({label_key(item.value) for item in items}) == 1:
            summary.n_agreed += 1
        else:
            summary.n_disagreed += 1
    return summary


# --------------------------------------------------------------------------- #
# Source 1: annotators of one example
# --------------------------------------------------------------------------- #


def annotator_conflicts(examples: Iterable[DatasetExample]) -> list[LabelAuditFinding]:
    """Flag examples whose annotators do not all agree.

    The finding records whether a strict majority exists, because that is exactly what
    decides whether a human has to look at it.
    """
    findings: list[LabelAuditFinding] = []
    for example in examples:
        items = annotations(example)
        if len(items) < 2 or len({label_key(item.value) for item in items}) == 1:
            continue
        majority = majority_value(items)
        verdict = (
            f"no majority among {len(items)}"
            if majority is None
            else f"majority {majority!r}"
        )
        findings.append(
            LabelAuditFinding(
                kind=LabelIssue.CONFLICTING,
                example_id=example.example_id,
                detail=(
                    f"annotators disagree ({', '.join(str(item) for item in items)}); "
                    f"{verdict}"
                ),
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# Source 2: two examples, one prompt
# --------------------------------------------------------------------------- #


def shared_prompt_conflicts(examples: Iterable[DatasetExample]) -> list[LabelAuditFinding]:
    """Flag examples that pose a byte-identical prompt but expect different answers.

    Consumes :func:`~eval_dataset_forge.dedup.exact.duplicate_input_clusters`, the
    superset the deduplication ladder exposes precisely so this stage can take the
    difference: a shared prompt with an *identical* answer is a duplicate to drop, a
    shared prompt with a *differing* answer is a conflict to adjudicate.

    One finding per implicated example rather than one per cluster, because a
    :class:`~eval_dataset_forge.schemas.LabelAuditFinding` names a single example.

    On the committed fixtures this returns nothing: all four shared-prompt clusters
    agree on their answers, i.e. they are duplicates. The channel is exercised by a
    synthetic pair in ``tests/test_labels.py``.
    """
    examples = list(examples)
    by_id = {ex.example_id: ex for ex in examples}

    findings: list[LabelAuditFinding] = []
    for cluster in duplicate_input_clusters(examples):
        outputs = {eid: canonical_output(by_id[eid].expected_output) for eid in cluster}
        if len(set(outputs.values())) < 2:
            continue  # a true duplicate; the dedup ladder owns it
        for example_id in cluster:
            others = [
                f"{other}={outputs[other]}" for other in cluster if other != example_id
            ]
            findings.append(
                LabelAuditFinding(
                    kind=LabelIssue.CONFLICTING,
                    example_id=example_id,
                    detail=(
                        f"identical prompt to {len(others)} other example(s) with a "
                        f"different gold answer: this={outputs[example_id]}, "
                        f"{', '.join(others)}"
                    ),
                )
            )
    return findings


# --------------------------------------------------------------------------- #
# Source 3: an example's two relevance channels
# --------------------------------------------------------------------------- #


def relevance_labelled(example: DatasetExample) -> bool:
    """Whether an example labels relevance twice, and so can contradict itself.

    True when it carries candidate passages *and* a list-shaped ``expected_output``.
    Keyed on shape rather than on ``task_family == "retrieval"``, matching
    :mod:`~eval_dataset_forge.leakage.retrieval_overlap`.
    """
    return bool(retrieval_docs(example)) and isinstance(example.expected_output, list)


def gold_doc_ids(example: DatasetExample) -> set[str]:
    """Doc ids the gold ``expected_output`` list names."""
    expected = example.expected_output
    if not isinstance(expected, list):
        return set()
    return {str(item) for item in expected}


def relevant_doc_ids(example: DatasetExample) -> set[str]:
    """Doc ids whose passage carries ``relevant: true``."""
    return {
        str(doc.get("doc_id"))
        for doc in retrieval_docs(example)
        if doc.get("relevant") is True
    }


def relevance_conflicts(examples: Iterable[DatasetExample]) -> list[LabelAuditFinding]:
    """Flag examples whose gold doc-id list disagrees with their own relevance flags.

    Set equality, so the two ways to be wrong are named separately: a passage marked
    relevant but absent from the gold list (an *incomplete* relevance set), and a gold
    doc id whose passage is not marked relevant (a *spurious* one).

    Two empty sets agree. ``retrieval_toy``'s ``ret-005`` — an over-specific query no
    candidate passage answers — therefore passes clean, while a naive
    "empty ``expected_output`` means the label is missing" rule would flag it. That
    distinction is the whole reason this check exists rather than
    :func:`~eval_dataset_forge.labels.audit.missing_labels` swallowing both.
    """
    findings: list[LabelAuditFinding] = []
    for example in examples:
        if not relevance_labelled(example):
            continue
        gold = gold_doc_ids(example)
        relevant = relevant_doc_ids(example)
        if gold == relevant:
            continue
        reasons = []
        if missing := sorted(relevant - gold):
            reasons.append(
                f"passage(s) {', '.join(missing)} marked relevant but absent from "
                f"expected_output"
            )
        if spurious := sorted(gold - relevant):
            reasons.append(
                f"expected_output names {', '.join(spurious)} but no such passage is "
                f"marked relevant"
            )
        findings.append(
            LabelAuditFinding(
                kind=LabelIssue.CONFLICTING,
                example_id=example.example_id,
                detail=(
                    f"relevance labelled twice and inconsistently: gold "
                    f"{sorted(gold) or 'none'} vs relevant {sorted(relevant) or 'none'} "
                    f"— {'; '.join(reasons)}"
                ),
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# All three
# --------------------------------------------------------------------------- #


def label_conflicts(examples: Iterable[DatasetExample]) -> list[LabelAuditFinding]:
    """Every ``CONFLICTING`` finding: annotators, then shared prompts, then relevance."""
    examples = list(examples)
    return [
        *annotator_conflicts(examples),
        *shared_prompt_conflicts(examples),
        *relevance_conflicts(examples),
    ]


__all__ = [
    "ANNOTATIONS_KEY",
    "Annotation",
    "AgreementSummary",
    "agreement",
    "annotations",
    "annotator_conflicts",
    "gold_doc_ids",
    "label_conflicts",
    "label_key",
    "majority_value",
    "relevance_conflicts",
    "relevance_labelled",
    "relevant_doc_ids",
    "shared_prompt_conflicts",
]
