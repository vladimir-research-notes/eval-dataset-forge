"""N-gram containment: the shared primitive, and the reference-corpus check (Milestone 5).

Two of the four leakage checks reduce to the same question — *do the word n-grams
of one text appear inside another?* — so both are built on :func:`containment`
here. :mod:`.answer_in_prompt` asks it of (gold answer, prompt);
:func:`reference_overlap` below asks it of (evaluation prompt, reference corpus),
which is the standard decontamination signal.

**Containment, not similarity.** :func:`containment` is asymmetric: it divides by
the *needle's* n-gram count, so it answers "how much of the needle is inside the
haystack?" A one-sentence answer fully quoted inside a thousand-word prompt scores
1.0, where a symmetric Jaccard would score near zero. Duplicate detection wants
symmetry (:mod:`eval_dataset_forge.dedup`); leakage does not.

**The width clamp.** A needle shorter than ``n`` tokens has no ``n``-grams, so the
comparison runs at ``effective_n = min(n, len(needle_tokens))`` and the haystack is
shingled at that same width. Without this an 8-gram check would silently return
0.0 for every prompt under 8 tokens — reporting "no contamination" for exactly the
short items most likely to be duplicated verbatim. The tradeoff is that a *very*
short needle is checked at a *very* permissive width; every finding therefore
records the effective width it fired at, and :data:`DEFAULT_NGRAM` is chosen so the
clamp is the exception rather than the rule.

**What this cannot do.** It measures overlap against text it is *handed*.
Contamination of a model's private pretraining corpus is not observable from here,
and a clean report is evidence of nothing about corpora we cannot read. That is a
property of the problem, not a gap in the implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from eval_dataset_forge.dedup.fuzzy import tokenize, word_ngrams
from eval_dataset_forge.schemas import DatasetExample, LeakageFinding, LeakageKind, Split

# Word-n-gram width for the reference-corpus check. 8 is the low end of the 8–13
# band that published decontamination pipelines use, and on the committed fixtures
# it is the width at which the "any shared n-gram" rule stops firing on shared
# boilerplate: at n=5 the phrases "what is the capital of" and "chocolate chip
# cookies at home" produce four spurious flags, at n=8 none. See
# docs/LEAKAGE.md §3 for the sweep.
DEFAULT_NGRAM = 8

# Minimum containment required *in addition* to at least one shared n-gram. 0.0
# keeps the standard rule — a single shared 8-gram is the flag — while leaving the
# knob exposed for corpora where that is too eager.
DEFAULT_MIN_CONTAINMENT = 0.0

# Scores are rounded before they reach a finding so reports are stable across
# platforms and floating-point noise never shows up in a diff.
_SCORE_DIGITS = 4


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def containment(
    needle_tokens: Sequence[str], haystack_tokens: Sequence[str], n: int
) -> tuple[float, int]:
    """Fraction of the needle's n-grams that also occur in the haystack.

    Returns ``(score, effective_n)`` where ``effective_n = min(n, len(needle))``
    (see the module docstring on the width clamp). An empty needle scores 0.0 at
    width 0: there is nothing to contain, and calling that "fully contained" would
    flag every example with an empty gold answer.
    """
    if not needle_tokens:
        return 0.0, 0
    width = min(n, len(needle_tokens))
    needle_grams = word_ngrams(needle_tokens, width)
    if not needle_grams:  # pragma: no cover - unreachable while needle is non-empty
        return 0.0, width
    shared = needle_grams & word_ngrams(haystack_tokens, width)
    return len(shared) / len(needle_grams), width


def longest_shared_run(needle_tokens: Sequence[str], haystack_tokens: Sequence[str]) -> list[str]:
    """The longest contiguous token run appearing in both sequences.

    Used purely for *evidence*: a containment score says how much leaked, this says
    what leaked. Classic O(len(needle) × len(haystack)) dynamic program over a
    rolling row, which is free at the scale a prompt/answer pair lives at.
    """
    if not needle_tokens or not haystack_tokens:
        return []
    previous = [0] * (len(haystack_tokens) + 1)
    best_len = best_end = 0
    for i, needle_token in enumerate(needle_tokens, start=1):
        current = [0] * (len(haystack_tokens) + 1)
        for j, haystack_token in enumerate(haystack_tokens, start=1):
            if needle_token == haystack_token:
                current[j] = previous[j - 1] + 1
                if current[j] > best_len:
                    best_len, best_end = current[j], i
        previous = current
    return list(needle_tokens[best_end - best_len : best_end])


class ReferenceIndex:
    """A reference corpus, shingled on demand and memoized per n-gram width.

    The width is not known until a target example is seen (the clamp depends on the
    example's length), so shingling is lazy. Reference *ids* are optional: they name
    the matching source in a finding's evidence when the corpus is a set of
    examples, and degrade to positional labels when it is an opaque text file.
    """

    def __init__(self, texts: Iterable[str], ids: Iterable[str] | None = None) -> None:
        self.texts = list(texts)
        self._tokens = [tokenize(text) for text in self.texts]
        self.ids = (
            list(ids) if ids is not None else [f"reference[{i}]" for i in range(len(self.texts))]
        )
        if len(self.ids) != len(self.texts):
            raise ValueError(f"got {len(self.ids)} reference ids for {len(self.texts)} texts")
        self._grams: dict[int, list[set[tuple[str, ...]]]] = {}

    def __len__(self) -> int:
        return len(self.texts)

    def grams_at(self, width: int) -> list[set[tuple[str, ...]]]:
        """Per-reference n-gram sets at ``width``, computed once and cached."""
        if width not in self._grams:
            self._grams[width] = [word_ngrams(tokens, width) for tokens in self._tokens]
        return self._grams[width]


def split_reference(
    examples: Iterable[DatasetExample], split: str = Split.TRAIN
) -> ReferenceIndex:
    """Build a :class:`ReferenceIndex` over the prompts of one split."""
    members = [ex for ex in examples if ex.split == split]
    return ReferenceIndex([ex.input for ex in members], [ex.example_id for ex in members])


# --------------------------------------------------------------------------- #
# The reference-corpus check
# --------------------------------------------------------------------------- #


def reference_overlap(
    examples: Iterable[DatasetExample],
    reference: ReferenceIndex,
    *,
    ngram: int = DEFAULT_NGRAM,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
) -> list[LeakageFinding]:
    """Flag examples whose prompt shares an n-gram with the reference corpus.

    One finding per contaminated example, carrying only *that* example's id: the
    finding is a statement about the evaluation item, and the reference may be an
    external corpus whose ids mean nothing to this dataset. The matching reference
    is named in ``evidence`` instead.

    A reference entry whose id equals the example's own id is skipped, so passing a
    dataset as its own reference reports genuine internal repetition rather than
    every example matching itself.
    """
    findings: list[LeakageFinding] = []
    for example in examples:
        tokens = tokenize(example.input)
        if not tokens:
            continue
        width = min(ngram, len(tokens))
        example_grams = word_ngrams(tokens, width)
        if not example_grams:  # pragma: no cover - unreachable for non-empty tokens
            continue

        matched: set[tuple[str, ...]] = set()
        source: str | None = None
        for ref_id, ref_grams in zip(reference.ids, reference.grams_at(width), strict=True):
            if ref_id == example.example_id:
                continue
            shared = example_grams & ref_grams
            if shared and source is None:
                source = ref_id
            matched |= shared

        if not matched:
            continue
        score = round(len(matched) / len(example_grams), _SCORE_DIGITS)
        if score < min_containment:
            continue
        sample = " ".join(min(sorted(matched)))
        clamped = " (clamped: prompt shorter than n)" if width < ngram else ""
        findings.append(
            LeakageFinding(
                kind=LeakageKind.NGRAM_OVERLAP,
                example_ids=[example.example_id],
                score=score,
                evidence=(
                    f"{len(matched)}/{len(example_grams)} {width}-grams shared with {source}"
                    f"{clamped}; e.g. {sample!r}"
                ),
            )
        )
    return findings


def split_reference_overlap(
    examples: Iterable[DatasetExample],
    *,
    reference_split: str = Split.TRAIN,
    target_splits: Sequence[str] = (Split.DEV, Split.TEST),
    ngram: int = DEFAULT_NGRAM,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
) -> list[LeakageFinding]:
    """:func:`reference_overlap` with one split of the dataset as the reference.

    The default — held-out splits checked against ``train`` — is the in-dataset
    contamination check. Supply an external corpus through :func:`reference_overlap`
    directly when you have one; there is no third option, because the corpora that
    matter most (model pretraining data) are not ours to read.
    """
    examples = list(examples)
    targets = [ex for ex in examples if ex.split in target_splits]
    return reference_overlap(
        targets,
        split_reference(examples, reference_split),
        ngram=ngram,
        min_containment=min_containment,
    )


__all__ = [
    "DEFAULT_MIN_CONTAINMENT",
    "DEFAULT_NGRAM",
    "ReferenceIndex",
    "containment",
    "longest_shared_run",
    "reference_overlap",
    "split_reference",
    "split_reference_overlap",
]
