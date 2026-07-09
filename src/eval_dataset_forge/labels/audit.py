"""Missing, ambiguous, and inconsistently formatted labels (Milestone 6).

Three of the five :class:`~eval_dataset_forge.schemas.LabelIssue` kinds live here; the
:attr:`~eval_dataset_forge.schemas.LabelIssue.CONFLICTING` sources live in
:mod:`.disagreement` and :attr:`~eval_dataset_forge.schemas.LabelIssue.NEEDS_ADJUDICATION`
is derived in :mod:`.adjudication`. Each public check emits exactly one kind — a
property ``tests/test_labels.py`` enforces.

**What "missing" means, and what it deliberately does not.** A gold answer is missing
when it carries no content: ``""``, ``[]``, ``{}``, or a rubric all of whose leaves are
empty. Emptiness is decided by
:func:`~eval_dataset_forge.leakage.answer_in_prompt.answer_candidates`, which Milestone 5
wrote for exactly this hand-off, and which already gets the traps right: ``False``,
``0``, and ``0.0`` are *labels*, not absences, and a bare ``if not value`` would delete
all three. Examples that label relevance twice (:func:`.disagreement.relevance_labelled`)
are excluded here — an empty gold doc-id list is only missing if the example's own
passage flags contradict it, which is a conflict, not an absence.

**Ambiguity: a declaration and a heuristic, kept apart.** ``difficulty_tags``
containing ``ambiguous`` is a *declaration* by whoever built the dataset — exact, and
free. The heuristic asks the harder question: would we have found it if nobody had told
us? Measured on the fixtures (``docs/LABEL_AUDIT.md`` §3), with the ``ambiguous`` tags
withheld, it recovers 4 of the 6 declared-ambiguous examples with zero false positives.
Two signals fire:

- *dangling reference* — a prompt of ≤4 tokens with no wh-word ("Is it good?",
  "Is the level high?", "Make the report better."). Such a prompt cannot bind its own
  referents.
- *open-set request* — "Name a primary color.", which asks for one instance of a
  category while the gold answer records only one of the many correct instances.

Both are gated on a **scalar gold answer**. The gate is not cosmetic. ``retrieval_toy``'s
``ret-004`` ("Name a noble gas.") is the same sentence pattern as ``qa_toy``'s ``qa-008``
("Name a primary color."), and no rule keyed on prompt text alone can flag one and not
the other. What separates them is the shape of the answer: a list-valued gold answer can
enumerate every acceptable response, so an open-set prompt answered by a set is not
under-specified. Drop the gate and ``ret-004`` becomes a false positive; drop the
wh-word guard and ``qa-005`` ("What is 2 + 2?") does.

The two examples the heuristic misses are the interesting ones. ``saf-007`` is a
dual-use request whose ambiguity is a semantic judgement, and ``ret-005`` is
*over*-specified rather than under-specified — an unanswerable query. Neither is short,
neither is open-set, and no lexical rule reaches them. Declared ambiguity and detectable
ambiguity are different sets; the audit reports their union and says which signal fired.

**Format: shape, then vocabulary.** Within a ``task_family``, a gold answer whose shape
is a small minority of a dominant shape is flagged (``agent-011``, a bare string among
rubric dicts). Shape, *not* key set: ``agent_tasks_toy``'s coding tasks carry
``{reference_solution, success_criteria}`` and its research tasks ``{reference_answer}``,
a legitimate per-subtype schema that a key-set check would report as three false
positives. The second signal reuses Milestone 5's
:func:`~eval_dataset_forge.leakage.answer_in_prompt.label_vocabulary`: when a family's
gold answers form a closed class vocabulary, a *rare* answer outside it is a format
defect (``saf-010``, ``"no"`` among ``comply``/``refuse``). Both signals self-disable —
the vocabulary is empty on every free-text fixture, so no categorical check runs there.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from typing import Any

from eval_dataset_forge.dedup.fuzzy import normalize_text, tokenize
from eval_dataset_forge.labels.disagreement import relevance_labelled
from eval_dataset_forge.leakage.answer_in_prompt import (
    LABEL_MIN_REPEATS,
    answer_candidates,
    label_vocabulary,
)
from eval_dataset_forge.schemas import DatasetExample, LabelAuditFinding, LabelIssue

# --------------------------------------------------------------------------- #
# Ambiguity parameters
# --------------------------------------------------------------------------- #

# The tag a dataset author uses to declare an example ambiguous.
AMBIGUOUS_TAG = "ambiguous"

# A prompt this short cannot specify a task. Calibrated in docs/LABEL_AUDIT.md §3: it is
# the smallest width that recovers every lexically-detectable planted target. Widths 5-7
# give the identical answer on the fixtures — but only because no fixture prompt of that
# length passes the guards, so that agreement is sparsity, not a robustness basin. Width
# 8 admits `bio-004`.
MAX_UNDERSPECIFIED_TOKENS = 4

# A prompt containing any of these asks for a specific thing, so its referents are bound
# by the question itself. The guard costs recall on short wh-questions ("What is it?" is
# not flagged) and buys precision on short factual ones ("What is 2 + 2?").
WH_WORDS = frozenset({"what", "who", "whom", "whose", "which", "when", "where", "why", "how"})

# "Name a primary color." — a request for one instance of a category.
OPEN_SET_REQUEST = re.compile(
    r"^(?:name|list|give|suggest|provide|state)\s+(?:me\s+)?(?:an?|one|some|any)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Format parameters
# --------------------------------------------------------------------------- #

# Below this many labelled examples a "dominant shape" is an artifact of a tiny sample.
MIN_FAMILY_SIZE = 4

# A family split 50/50 across two shapes is heterogeneous by design, not defective;
# calling half of it a defect would be noise. Require a real majority shape first.
DOMINANT_MIN_SHARE = 0.7


# --------------------------------------------------------------------------- #
# Shared accessors
# --------------------------------------------------------------------------- #


def has_label(value: Any) -> bool:
    """Whether a gold answer carries any content at all.

    Delegates to :func:`answer_candidates`, so ``False``/``0``/``0.0`` count as labels
    while ``""``, ``[]``, ``{}``, ``"   "``, and ``{"solution": ""}`` do not.
    """
    return bool(answer_candidates(value))


def answer_shape(value: Any) -> str:
    """Coarse structural shape of a gold answer.

    ``bool`` is tested before ``int`` because ``isinstance(True, int)`` is true, and a
    boolean label is not a numeric one.
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "mapping"
    return "other"


# --------------------------------------------------------------------------- #
# Missing labels
# --------------------------------------------------------------------------- #


def missing_labels(examples: Iterable[DatasetExample]) -> list[LabelAuditFinding]:
    """Flag examples with no gold answer to score against.

    Examples whose relevance is labelled twice are skipped: for them an empty
    ``expected_output`` is a claim ("no passage is relevant") that
    :func:`.disagreement.relevance_conflicts` can check against the passage flags,
    rather than an absence this check can assert.
    """
    findings: list[LabelAuditFinding] = []
    for example in examples:
        value = example.expected_output
        if has_label(value) or relevance_labelled(example):
            continue
        shape = answer_shape(value)
        literal_empty = value == "" or (isinstance(value, (list, dict)) and not value)
        detail = (
            f"expected_output is an empty {shape}"
            if literal_empty
            else f"expected_output ({shape}) has no non-empty leaf values"
        )
        findings.append(
            LabelAuditFinding(
                kind=LabelIssue.MISSING,
                example_id=example.example_id,
                detail=f"{detail} — no gold answer to score against",
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# Ambiguous labels
# --------------------------------------------------------------------------- #


def _has_scalar_gold(example: DatasetExample) -> bool:
    """Whether the gold answer is a single value rather than a container.

    A container can enumerate every acceptable answer, which is precisely what makes an
    open-ended prompt unambiguous. See the module docstring.
    """
    return not isinstance(example.expected_output, (list, dict))


def declared_ambiguous(example: DatasetExample) -> str | None:
    """The example's own ``ambiguous`` difficulty tag, if present. Exact, not heuristic."""
    if AMBIGUOUS_TAG in example.difficulty_tags:
        return f"declared ambiguous by difficulty_tags {example.difficulty_tags}"
    return None


def underspecified_prompt(
    example: DatasetExample, *, max_tokens: int = MAX_UNDERSPECIFIED_TOKENS
) -> str | None:
    """A prompt too short to bind its own referents, or ``None``.

    Requires a scalar gold answer and the absence of any wh-word. Both guards are
    load-bearing; the module docstring names the example each one saves.
    """
    if not _has_scalar_gold(example):
        return None
    tokens = tokenize(example.input)
    if not tokens or len(tokens) > max_tokens:
        return None
    if set(tokens) & WH_WORDS:
        return None
    return (
        f"prompt is {len(tokens)} tokens with no wh-word, so its referents are unbound: "
        f"{example.input.strip()!r}"
    )


def open_set_request(example: DatasetExample) -> str | None:
    """A prompt asking for one instance of a category, or ``None``.

    Redundant on the committed fixtures — the only example it flags (``qa-008``) is
    already short enough for :func:`underspecified_prompt`. It exists for the longer
    open-set prompts the fixtures do not contain, and is exercised by a synthetic case
    in ``tests/test_labels.py``.
    """
    if not _has_scalar_gold(example):
        return None
    match = OPEN_SET_REQUEST.match(example.input.strip())
    if match is None:
        return None
    return (
        f"prompt asks for one instance of a category ({match.group(0).strip()!r}) while "
        f"the gold answer records a single response"
    )


def ambiguous_labels(
    examples: Iterable[DatasetExample],
    *,
    max_tokens: int = MAX_UNDERSPECIFIED_TOKENS,
    heuristics: bool = True,
) -> list[LabelAuditFinding]:
    """Flag examples whose expected output is not the only defensible one.

    At most one finding per example; ``detail`` names every signal that fired, so a
    declaration and a heuristic hit are distinguishable after the fact. Pass
    ``heuristics=False`` to keep only the examples their own author declared ambiguous.
    """
    findings: list[LabelAuditFinding] = []
    for example in examples:
        reasons = [declared_ambiguous(example)]
        if heuristics:
            reasons.append(underspecified_prompt(example, max_tokens=max_tokens))
            reasons.append(open_set_request(example))
        fired = [reason for reason in reasons if reason]
        if not fired:
            continue
        findings.append(
            LabelAuditFinding(
                kind=LabelIssue.AMBIGUOUS,
                example_id=example.example_id,
                detail="; ".join(fired),
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# Inconsistent label formats
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FamilyExpectation:
    """What a ``task_family``'s labelled examples agree on, and how strongly.

    ``dominant_shape`` is ``None`` when no shape clears :data:`DOMINANT_MIN_SHARE` or the
    family is smaller than :data:`MIN_FAMILY_SIZE`; ``vocabulary`` is empty when the
    family's answers are free text rather than a closed class set. Either being empty
    disables the corresponding check for that family.
    """

    task_family: str
    n_labelled: int
    shapes: dict[str, int] = field(default_factory=dict)
    dominant_shape: str | None = None
    vocabulary: tuple[str, ...] = ()
    answer_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_family": self.task_family,
            "n_labelled": self.n_labelled,
            "shapes": self.shapes,
            "dominant_shape": self.dominant_shape,
            "vocabulary": list(self.vocabulary),
        }


def family_expectations(
    examples: Iterable[DatasetExample],
    *,
    missing_ids: Collection[str] | None = None,
) -> dict[str, FamilyExpectation]:
    """Per-``task_family`` shape census and label vocabulary over the *labelled* examples.

    Examples already flagged :attr:`~eval_dataset_forge.schemas.LabelIssue.MISSING` are
    excluded: an absent answer has no format, and counting ``""`` as a string would both
    double-flag it and distort the census it is measured against. ``missing_ids=None``
    means "compute them" — the orchestrator passes the set it already has.
    """
    examples = list(examples)
    if missing_ids is None:
        missing_ids = {finding.example_id for finding in missing_labels(examples)}

    by_family: dict[str, list[DatasetExample]] = {}
    for example in examples:
        if example.example_id in missing_ids:
            continue
        by_family.setdefault(example.task_family, []).append(example)

    expectations: dict[str, FamilyExpectation] = {}
    for task_family, members in by_family.items():
        shapes = Counter(answer_shape(ex.expected_output) for ex in members)
        dominant_shape, dominant_count = shapes.most_common(1)[0]
        if len(members) < MIN_FAMILY_SIZE or dominant_count / len(members) < DOMINANT_MIN_SHARE:
            dominant_shape = None
        answer_counts = Counter(
            normalize_text(ex.expected_output)
            for ex in members
            if isinstance(ex.expected_output, str)
        )
        expectations[task_family] = FamilyExpectation(
            task_family=task_family,
            n_labelled=len(members),
            shapes=dict(shapes),
            dominant_shape=dominant_shape,
            vocabulary=tuple(sorted(label_vocabulary(members))),
            answer_counts=dict(answer_counts),
        )
    return expectations


def inconsistent_formats(
    examples: Iterable[DatasetExample],
    *,
    missing_ids: Collection[str] | None = None,
    expectations: dict[str, FamilyExpectation] | None = None,
) -> list[LabelAuditFinding]:
    """Flag gold answers that break their family's shape or label vocabulary.

    At most one finding per example, shape before vocabulary: an answer of the wrong
    *shape* has no business being compared against a vocabulary of strings.

    The vocabulary signal only flags a *rare* out-of-vocabulary answer (fewer than
    :data:`~eval_dataset_forge.leakage.answer_in_prompt.LABEL_MIN_REPEATS` occurrences).
    A repeated one means the vocabulary is incomplete — ``label_vocabulary`` only admits
    answers of ≤2 tokens — not that the answer is malformed.
    """
    examples = list(examples)
    if missing_ids is None:
        missing_ids = {finding.example_id for finding in missing_labels(examples)}
    if expectations is None:
        expectations = family_expectations(examples, missing_ids=missing_ids)

    findings: list[LabelAuditFinding] = []
    for example in examples:
        if example.example_id in missing_ids:
            continue
        expectation = expectations.get(example.task_family)
        if expectation is None:
            continue

        shape = answer_shape(example.expected_output)
        if expectation.dominant_shape and shape != expectation.dominant_shape:
            dominant_count = expectation.shapes[expectation.dominant_shape]
            findings.append(
                LabelAuditFinding(
                    kind=LabelIssue.INCONSISTENT_FORMAT,
                    example_id=example.example_id,
                    detail=(
                        f"expected_output is a {shape}, but {dominant_count} of "
                        f"{expectation.n_labelled} labelled {example.task_family!r} "
                        f"examples use {expectation.dominant_shape}"
                    ),
                )
            )
            continue

        if not expectation.vocabulary or not isinstance(example.expected_output, str):
            continue
        key = normalize_text(example.expected_output)
        if key in expectation.vocabulary:
            continue
        occurrences = expectation.answer_counts.get(key, 0)
        if occurrences >= LABEL_MIN_REPEATS:
            continue  # repeated: the vocabulary is incomplete, the answer is not malformed
        findings.append(
            LabelAuditFinding(
                kind=LabelIssue.INCONSISTENT_FORMAT,
                example_id=example.example_id,
                detail=(
                    f"gold answer {example.expected_output!r} occurs {occurrences}× and is "
                    f"outside the {example.task_family!r} label vocabulary "
                    f"{list(expectation.vocabulary)}"
                ),
            )
        )
    return findings


__all__ = [
    "AMBIGUOUS_TAG",
    "DOMINANT_MIN_SHARE",
    "MAX_UNDERSPECIFIED_TOKENS",
    "MIN_FAMILY_SIZE",
    "OPEN_SET_REQUEST",
    "WH_WORDS",
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
]
