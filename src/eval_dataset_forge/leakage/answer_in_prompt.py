"""Answer-in-prompt leakage: the gold answer contaminating its own prompt (Milestone 5).

If the expected output is quoted inside the input, a system can score by copying
rather than by solving. The check is n-gram containment of the answer within the
prompt (:func:`~eval_dataset_forge.leakage.ngram_overlap.containment`), which is a
*lexical* test: it sees a quoted answer and misses an entailed one.

**Structured answers are unpacked, not stringified.** ``expected_output`` may be a
string, a scalar, a doc-id list, or a rubric dict. Serializing a rubric to JSON and
searching for that blob would never match, so :func:`answer_candidates` walks the
value and yields each leaf string with the field path that produced it. On the
fixtures this is what finds ``agent-004``: the leak is in the
``reference_solution`` field of a two-key dict, not in the answer as a whole.

**Why n=2.** Measured on the committed fixtures (docs/LEAKAGE.md §2):

- ``n=1`` (bag of words) flags 4–7 false positives. ``agent-011``'s answer
  ``"return sorted(arr)"`` scores 0.667 against a prompt that merely says "returns
  … sorted … arr"; ``qa-013``'s ``"Pacific Ocean"`` scores 0.5 because the prompt
  contains the word *ocean*. Unigram containment cannot distinguish a quoted answer
  from a prompt that shares its vocabulary — which every well-posed question does.
- ``n=3`` misses ``agent-004`` (0.25): a paraphrased quotation preserves bigrams
  long after it has destroyed trigrams.
- ``n=2`` recovers both planted leaks with zero false positives, and the margin is
  real rather than fitted — the weakest true positive scores 0.600 and the
  strongest non-leak 0.333.

Short answers are compared at their own width (``min(n, len(answer_tokens))``), so
a one-word gold answer like ``"Berlin"`` is found by a unigram match while longer
answers keep bigram discipline.

**The label-vocabulary guard.** In a classification dataset the gold answer *is* a
class name, and a prompt that names its own label set ("should the assistant comply
or refuse?") would be flagged on every row. :func:`label_vocabulary` detects this
case — short answers, repeated, covering most of the dataset — and those answers
are then excluded. The guard is dataset-level and deliberately conservative: it
fires on ``safety_toy`` (``comply``/``refuse``, 8 of 10 examples) and *not* on
``qa_toy``, where ``"William Shakespeare"`` recurs three times but covers only 21%
of the rows. Nothing is suppressed silently — the vocabulary it computed is carried
on the report. Its cost is a true negative: a genuinely leaked class label in a
genuine classification dataset is not reported. Pass ``ignore_answers=frozenset()``
to disable it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from eval_dataset_forge.dedup.fuzzy import normalize_text, tokenize
from eval_dataset_forge.leakage.ngram_overlap import containment, longest_shared_run
from eval_dataset_forge.schemas import DatasetExample, LeakageFinding, LeakageKind

# See the module docstring: n=2 is the only width that recovers both planted leaks
# without a false positive, and 0.5 sits in the middle of the [0.4, 0.6] plateau.
DEFAULT_NGRAM = 2
DEFAULT_THRESHOLD = 0.5

# Label-vocabulary detection. A class label is short, repeated, and — the load
# bearing condition — accounts for a large share of the dataset's gold answers.
# ``min_repeats`` alone would misfire on any dataset with a few recurring factual
# answers, which is why coverage is required as well.
LABEL_MAX_TOKENS = 2
LABEL_MIN_REPEATS = 3
LABEL_MIN_COVERAGE = 0.5

_SCORE_DIGITS = 4

# The field path reported for a gold answer that is a bare string or scalar.
_ROOT_FIELD = "expected_output"


def _root_label(value: Any) -> str | None:
    """The gold answer as a class-label string, or ``None`` if it cannot be one.

    A dict rubric or a doc-id list is never a class label, however often it repeats.
    ``bool`` is checked before ``int`` because ``isinstance(True, int)`` is true.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return None


def answer_candidates(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Flatten an ``expected_output`` into ``(field_path, text)`` leaf strings.

    Dict keys are field *names*, not answers, so only values are descended into;
    they are visited in sorted order so the output is deterministic. Empty and
    whitespace-only leaves are dropped — an absent gold answer is a label defect
    (Milestone 6), and searching for the empty string would match every prompt.
    """
    if isinstance(value, bool):
        return [(path or _ROOT_FIELD, str(value).lower())]
    if isinstance(value, str):
        return [(path or _ROOT_FIELD, value)] if value.strip() else []
    if isinstance(value, (int, float)):
        return [(path or _ROOT_FIELD, str(value))]
    if isinstance(value, list):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in answer_candidates(item, f"{path}[{index}]")
        ]
    if isinstance(value, dict):
        return [
            leaf
            for key in sorted(value)
            for leaf in answer_candidates(value[key], f"{path}.{key}" if path else str(key))
        ]
    return []


def label_vocabulary(
    examples: Iterable[DatasetExample],
    *,
    max_tokens: int = LABEL_MAX_TOKENS,
    min_repeats: int = LABEL_MIN_REPEATS,
    min_coverage: float = LABEL_MIN_COVERAGE,
) -> frozenset[str]:
    """Normalized gold answers that look like class labels rather than answers.

    A candidate is short (``<= max_tokens``) and repeated (``>= min_repeats``); the
    *set* of candidates additionally has to account for ``min_coverage`` of the
    examples before any of it is trusted. Without that last condition a QA set with
    three Shakespeare questions would classify ``"William Shakespeare"`` as a label.

    Returns an empty set — meaning "this is not a classification dataset, guard
    nothing" — when the coverage test fails.
    """
    examples = list(examples)
    if not examples:
        return frozenset()

    counts: Counter[str] = Counter()
    for example in examples:
        root = _root_label(example.expected_output)
        if root is None:
            continue
        key = normalize_text(root)
        if key and len(tokenize(root)) <= max_tokens:
            counts[key] += 1

    vocabulary = {key for key, count in counts.items() if count >= min_repeats}
    covered = sum(count for key, count in counts.items() if key in vocabulary)
    if covered / len(examples) < min_coverage:
        return frozenset()
    return frozenset(vocabulary)


def leaked_answers(
    examples: Iterable[DatasetExample],
    *,
    ngram: int = DEFAULT_NGRAM,
    threshold: float = DEFAULT_THRESHOLD,
    ignore_answers: frozenset[str] | None = None,
) -> list[LeakageFinding]:
    """Flag examples whose gold answer is quoted inside their own prompt.

    At most one finding per example: the strongest candidate field wins, and its
    containment becomes the finding's ``score``. ``ignore_answers`` defaults to the
    dataset's :func:`label_vocabulary`; pass ``frozenset()`` to check every answer.

    Findings are for review, never for removal. A task can legitimately restate its
    answer ("extract the year 2020 from: … in 2020 …"), and no lexical test tells
    that apart from a leak.
    """
    examples = list(examples)
    if ignore_answers is None:
        ignore_answers = label_vocabulary(examples)

    findings: list[LeakageFinding] = []
    for example in examples:
        prompt_tokens = tokenize(example.input)
        if not prompt_tokens:
            continue

        best: tuple[float, int, str, str] | None = None
        for field, text in answer_candidates(example.expected_output):
            if normalize_text(text) in ignore_answers:
                continue
            answer_tokens = tokenize(text)
            score, width = containment(answer_tokens, prompt_tokens, ngram)
            if score < threshold:
                continue
            span = " ".join(longest_shared_run(answer_tokens, prompt_tokens))
            if best is None or score > best[0]:
                best = (score, width, field, span)

        if best is None:
            continue
        score, width, field, span = best
        findings.append(
            LeakageFinding(
                kind=LeakageKind.ANSWER_IN_PROMPT,
                example_ids=[example.example_id],
                score=round(score, _SCORE_DIGITS),
                evidence=(
                    f"{field} is {score:.0%} contained in the prompt at n={width}; "
                    f"longest shared token span: {span!r}"
                ),
            )
        )
    return findings


__all__ = [
    "DEFAULT_NGRAM",
    "DEFAULT_THRESHOLD",
    "LABEL_MAX_TOKENS",
    "LABEL_MIN_COVERAGE",
    "LABEL_MIN_REPEATS",
    "answer_candidates",
    "label_vocabulary",
    "leaked_answers",
]
