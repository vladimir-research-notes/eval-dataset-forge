"""Surface signals and the weighted difficulty score (Milestone 8).

The score is a **transparent, model-free proxy**. Five signals are read off the
example, each compressed to ``[0, 1]``, then combined by fixed documented weights:

======================  =========  ====================================================
signal                  cap        raw quantity
======================  =========  ====================================================
``input_length``        32 tokens  word tokens in ``input``
``output_complexity``   32 tokens  word tokens in the canonical ``expected_output``
``reasoning_steps``     4 cues     sequential / enumeration / aggregation cues
``evidence_items``      4 items    retrieval passages + annotated entities
``tool_use``            indicator  ``metadata`` says a tool or lookup is required
======================  =========  ====================================================

Three properties are load-bearing, and each is pinned by a test.

**The score is exactly the sum of its components.** :func:`weighted_components` returns
one weighted contribution per signal, and ``sum(components.values()) == score``. A reader
who disagrees with a number can see which signal produced it, and by how much. Nothing
is hidden in a normalization constant applied after the fact.

**Counts are log-compressed, not min-maxed.** ``log1p(x) / log1p(cap)``, clipped. Two
consequences. A single 75-token reference solution cannot dominate a corpus of one-word
gold answers the way a linear scale would. And — more importantly — the mapping is
**dataset-independent**: a score computed over one dataset means the same thing as a
score computed over another, because no corpus statistic enters it. Min-max normalization
would silently redefine "hard" every time an example was added. The cost is that the caps
are hand-chosen; they are stated above and in ``docs/DIFFICULTY.md``, and
:mod:`.calibration` measures whether the resulting ranking means anything at all.

**No signal reads ``difficulty_tags``.** ``docs/DIFFICULTY.md`` originally listed the
``ambiguous``/``hard`` tags and the hand-assigned fixture difficulty as *scoring* signals
while also proposing to sanity-check the score against those same labels. That is
circular: the check would be measuring the label against itself. Both were removed from
the scorer. The size of the effect is not hypothetical — blending the declared bucket
into the score at a weight of just 0.15 lifts the reported rank correlation from
**+0.10 to +0.49** (at 0.30, to +0.77), which would read as "the heuristic works" for a
heuristic that does not. Measured by
``test_leaking_the_declared_bucket_into_the_score_inflates_concordance``. The declared
tags are the calibration *target* (:mod:`.calibration`) and a stratification facet
(:mod:`eval_dataset_forge.splits.stratified`) — never an input here.

What the signals actually measure on the committed corpus is not what their names
suggest; :mod:`.calibration` and ``docs/DIFFICULTY.md`` report it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from eval_dataset_forge.dedup.exact import canonical_output
from eval_dataset_forge.dedup.fuzzy import tokenize
from eval_dataset_forge.schemas import DatasetExample

# Names the heuristic version by. Recorded on every ``DifficultyScore.method`` so a score
# read from an old release is never silently compared against a score from a new one.
METHOD = "surface_v1"

# --------------------------------------------------------------------------- #
# Reasoning-step cues
# --------------------------------------------------------------------------- #

# Connectives that chain one sub-task onto another ("compute X and then return Y").
SEQUENTIAL_CUES: tuple[str, ...] = (" and ", " then ", " after ", " before ")

# Requests for an aggregate or an enumeration rather than a single lookup.
AGGREGATION_CUES: tuple[str, ...] = ("how many", "list ", "for each", "each of")

# Word-form cardinals: "the *four* nucleotide bases" asks for four things.
#
# Bare digits are deliberately **not** counted. A digit in a prompt is as often an
# operand, a year, or an identifier as it is an enumeration request: counting digits
# gives "What is 2 + 2?" two reasoning steps and gives "What is the approximate speed of
# light in a vacuum?" zero. The rule was chosen on that linguistic ground *before* the
# labels were consulted, not by picking whichever variant correlated better — the two
# variants score rho +0.192 and +0.171 respectively, so the choice barely moves the
# result either way. Pinned by ``test_reasoning_steps_ignores_bare_digits``.
CARDINAL_WORDS: frozenset[str] = frozenset(
    {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
)

# ``metadata`` keys a dataset may use to declare that an example needs a tool or lookup.
# None of the committed fixtures sets any of them; see :mod:`.calibration`, which reports
# the signal as inert rather than letting it contribute a silent constant.
TOOL_USE_KEYS: tuple[str, ...] = ("tool_use", "requires_tool", "tools")


# --------------------------------------------------------------------------- #
# Raw extractors — each returns a non-negative count or indicator
# --------------------------------------------------------------------------- #


def raw_input_length(example: DatasetExample) -> float:
    """Word tokens in the prompt."""
    return float(len(tokenize(example.input)))


def raw_output_complexity(example: DatasetExample) -> float:
    """Word tokens in the canonically serialized gold answer.

    Reuses :func:`~eval_dataset_forge.dedup.exact.canonical_output`, so a structured
    answer contributes its keys and its values and a missing answer contributes zero.
    That zero is a real trap rather than an edge case: an example whose gold answer is
    absent scores at the *floor* of this signal and therefore looks easy. Four such
    examples are planted in the fixtures and flagged ``MISSING`` by Milestone 6.
    :mod:`.calibration` surfaces the interaction instead of letting it pass.
    """
    return float(len(tokenize(canonical_output(example.expected_output))))


def raw_reasoning_steps(example: DatasetExample) -> float:
    """Count of sequential, aggregation, enumeration, and clause-separator cues."""
    text = example.input.casefold()
    count = sum(text.count(cue) for cue in SEQUENTIAL_CUES)
    count += sum(text.count(cue) for cue in AGGREGATION_CUES)
    count += text.count(";")
    count += sum(1 for token in tokenize(example.input) if token in CARDINAL_WORDS)
    return float(count)


def _metadata_list(metadata: Mapping[str, Any], key: str) -> list[Any]:
    value = metadata.get(key)
    return value if isinstance(value, list) else []


def raw_evidence_items(example: DatasetExample) -> float:
    """Retrieval passages plus annotated entities the example carries."""
    metadata = example.metadata
    count = 0
    retrieval = metadata.get("retrieval")
    if isinstance(retrieval, Mapping):
        count += len(_metadata_list(retrieval, "docs"))
    count += len(_metadata_list(metadata, "entities"))
    return float(count)


def raw_tool_use(example: DatasetExample) -> float:
    """1.0 when ``metadata`` declares a tool/lookup requirement, else 0.0.

    Truthiness, not presence: ``{"tool_use": false}`` and ``{"tools": []}`` both mean
    "no tool", and a dataset that writes the key on every example must not thereby make
    every example look harder.
    """
    for key in TOOL_USE_KEYS:
        if key not in example.metadata:
            continue
        value = example.metadata[key]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (list, tuple, set, str)):
            return 1.0 if len(value) else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if value else 0.0
    return 0.0


# --------------------------------------------------------------------------- #
# Signal registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Signal:
    """One surface signal: how to read it, how to compress it, and what it claims.

    ``cap <= 1`` marks an indicator (clamped); anything larger marks a count
    (log-compressed against the cap). ``claim`` states, in one line, the difficulty
    hypothesis the signal encodes — so a reader can judge the hypothesis, not just
    the arithmetic.
    """

    name: str
    extract: Callable[[DatasetExample], float]
    cap: float
    claim: str

    @property
    def is_indicator(self) -> bool:
        return self.cap <= 1.0

    def normalize(self, raw: float) -> float:
        """Compress a raw count/indicator to ``[0, 1]``. Dataset-independent."""
        raw = max(0.0, float(raw))
        if self.is_indicator:
            return min(1.0, raw)
        return min(1.0, math.log1p(raw) / math.log1p(self.cap))

    def of(self, example: DatasetExample) -> float:
        return self.normalize(self.extract(example))


SIGNALS: tuple[Signal, ...] = (
    Signal(
        "input_length",
        raw_input_length,
        32.0,
        "a longer prompt carries more to read, track, and satisfy",
    ),
    Signal(
        "output_complexity",
        raw_output_complexity,
        32.0,
        "a longer or more structured gold answer takes more work to produce",
    ),
    Signal(
        "reasoning_steps",
        raw_reasoning_steps,
        4.0,
        "chained, enumerated, or aggregated sub-tasks need more inference steps",
    ),
    Signal(
        "evidence_items",
        raw_evidence_items,
        4.0,
        "more candidate passages or entities mean more to select among",
    ),
    Signal(
        "tool_use",
        raw_tool_use,
        1.0,
        "a task that requires an external tool cannot be answered from memory",
    ),
)

SIGNAL_NAMES: tuple[str, ...] = tuple(signal.name for signal in SIGNALS)

_BY_NAME: dict[str, Signal] = {signal.name: signal for signal in SIGNALS}

# Hand-chosen and stated, never fit. ``reasoning_steps`` leads because it is the only
# signal whose *claim* is about inference rather than about size; the two length signals
# split the remainder; ``evidence_items`` and ``tool_use`` are the two that a dataset
# must opt into by writing metadata. These weights were fixed before any label was read.
# :mod:`.calibration` sweeps the whole simplex, so the reader never has to take on faith
# that the result is a property of the signals rather than of this particular vector.
DEFAULT_WEIGHTS: dict[str, float] = {
    "input_length": 0.20,
    "output_complexity": 0.20,
    "reasoning_steps": 0.30,
    "evidence_items": 0.15,
    "tool_use": 0.15,
}


def signal(name: str) -> Signal:
    """Look up a signal by name."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown signal {name!r} (known: {list(SIGNAL_NAMES)})") from None


def resolve_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Validate a weight vector: known names, non-negative, summing to 1.0.

    Rejecting an unnormalized vector is deliberate. Weights that sum to 1 make the score
    directly comparable to its components and keep it inside ``[0, 1]``; silently
    renormalizing would let a caller believe a signal carried a weight it did not.
    """
    if weights is None:
        return dict(DEFAULT_WEIGHTS)
    unknown = set(weights) - set(SIGNAL_NAMES)
    if unknown:
        raise ValueError(f"unknown signal(s) {sorted(unknown)} (known: {list(SIGNAL_NAMES)})")
    resolved = {name: float(weights.get(name, 0.0)) for name in SIGNAL_NAMES}
    negative = [name for name, value in resolved.items() if value < 0]
    if negative:
        raise ValueError(f"negative weight(s) for {negative}")
    total = sum(resolved.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {total}")
    return resolved


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def raw_signals(example: DatasetExample) -> dict[str, float]:
    """Every signal's raw count/indicator, before compression."""
    return {signal.name: signal.extract(example) for signal in SIGNALS}


def normalized_signals(example: DatasetExample) -> dict[str, float]:
    """Every signal compressed to ``[0, 1]``, before weighting."""
    return {signal.name: signal.of(example) for signal in SIGNALS}


def weighted_components(
    example: DatasetExample, weights: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Per-signal weighted contributions. Their sum is exactly the score."""
    resolved = resolve_weights(weights)
    return {signal.name: resolved[signal.name] * signal.of(example) for signal in SIGNALS}


def raw_score(example: DatasetExample, weights: Mapping[str, float] | None = None) -> float:
    """The weighted difficulty score in ``[0, 1]``. Higher is (heuristically) harder."""
    return sum(weighted_components(example, weights).values())


def score_matrix(
    examples: Iterable[DatasetExample],
) -> tuple[list[str], list[list[float]]]:
    """``(signal_names, rows)`` of normalized signal values — the calibration input."""
    rows = [[signal.of(example) for signal in SIGNALS] for example in examples]
    return list(SIGNAL_NAMES), rows


__all__ = [
    "METHOD",
    "AGGREGATION_CUES",
    "CARDINAL_WORDS",
    "DEFAULT_WEIGHTS",
    "SEQUENTIAL_CUES",
    "SIGNALS",
    "SIGNAL_NAMES",
    "TOOL_USE_KEYS",
    "Signal",
    "normalized_signals",
    "raw_evidence_items",
    "raw_input_length",
    "raw_output_complexity",
    "raw_reasoning_steps",
    "raw_score",
    "raw_signals",
    "raw_tool_use",
    "resolve_weights",
    "score_matrix",
    "signal",
    "weighted_components",
]
