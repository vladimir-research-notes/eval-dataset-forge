"""Text normalization, tokenization, and set-similarity primitives (Milestone 4).

This module holds level 2 of the deduplication ladder — **normalized-text
duplicates**, i.e. examples that differ only in formatting — plus the shared
lexical primitives (:func:`tokenize`, :func:`word_ngrams`, :func:`jaccard`) that
the heuristic near-duplicate stage in :mod:`.near_duplicate` builds on.

Two normalizations live here and they are deliberately *different*:

* :func:`normalize_text` is the **duplicate key**. It folds away casing,
  Unicode presentation differences (NFKC: smart quotes, full-width forms,
  ligatures), punctuation, and whitespace runs. Two prompts with the same
  normalized form are treated as the same prompt.
* :func:`tokenize` is the **similarity input**. It extracts word tokens for
  n-gram set comparison and never collapses two distinct words together.

Normalized-text dedup keys on the *normalized input* but the *exact* expected
output (see :func:`normalized_signature`). Punctuation-folding a structured gold
answer — a code reference solution, a JSON rubric — would happily merge two
materially different answers, and a pair that shares a prompt but not an answer
is a label conflict (Milestone 6), not a duplicate. Requiring an identical answer
keeps this stage conservative: it only ever merges examples that are the same
question *and* the same answer.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from eval_dataset_forge.dedup.exact import (
    canonical_output,
    content_signature,
    sha256_hex,
)
from eval_dataset_forge.schemas import DatasetExample, DedupKind, DeduplicationFinding

# Word tokens for lexical similarity. ``\w+`` keeps digits and underscores, which
# matters for the agent/coding fixtures ("fizzbuzz", "s[::-1]" -> "s", "1").
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Field separator used when signing a multi-field key. NUL never occurs in the
# fixture text, so "a" + "bc" and "ab" + "c" cannot collide.
_SEP = "\x00"

METHOD_NORMALIZED = "sha256(normalize(input)+expected_output)"


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def normalize_text(text: str) -> str:
    """Fold a string to its normalized duplicate key.

    Applies, in order: Unicode NFKC (so ``“x”`` and ``"x"``, ``ﬁ`` and ``fi``
    agree), case folding, replacement of every Unicode punctuation/symbol
    character by a space, and whitespace-run collapsing.

    ``"  Who wrote  “Hamlet”? "`` and ``"who wrote hamlet"`` both normalize to
    ``"who wrote hamlet"``.

    This is lossy by construction. It can merge genuinely distinct short items
    whose only difference is punctuation (``"C++"`` and ``"C"``); that is why it
    is applied to prompts, never to gold answers, and why the caller additionally
    requires an identical ``expected_output``.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    folded = "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text
    )
    return " ".join(folded.split())


def tokenize(text: str) -> list[str]:
    """Split text into lowercased word tokens for lexical similarity.

    Unlike :func:`normalize_text` this never merges distinct words: punctuation
    is a separator, not a space-equivalent, and the result keeps token order so
    callers can build n-grams.
    """
    return _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())


# --------------------------------------------------------------------------- #
# Set similarity
# --------------------------------------------------------------------------- #


def word_ngrams(tokens: Sequence[str], n: int = 1) -> set[tuple[str, ...]]:
    """Return the set of contiguous word ``n``-grams (shingles) of ``tokens``.

    ``n=1`` yields the bag-of-words set. When ``len(tokens) < n`` the whole token
    sequence is returned as a single short shingle rather than an empty set, so a
    very short prompt still compares meaningfully against a copy of itself.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not tokens:
        return set()
    if len(tokens) <= n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity |a ∩ b| / |a ∪ b|, with 0.0 for the empty-vs-nonempty case.

    Two empty sets are defined as identical (1.0): both texts tokenize to nothing,
    so there is no evidence they differ.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- #
# Normalized-text duplicates
# --------------------------------------------------------------------------- #


def normalized_signature(example: DatasetExample) -> str:
    """sha256 of ``normalize_text(input)`` joined with the canonical gold answer.

    The answer is *not* normalized — see the module docstring for why.
    """
    payload = _SEP.join(
        [normalize_text(example.input), canonical_output(example.expected_output)]
    )
    return sha256_hex(payload)


def normalized_duplicates(examples: Iterable[DatasetExample]) -> list[DeduplicationFinding]:
    """Find clusters that are identical *after* normalizing the prompt.

    Clusters whose members are already byte-identical are **not** reported here —
    :func:`~eval_dataset_forge.dedup.exact.exact_duplicates` owns those. A finding
    from this function therefore always represents something the exact stage
    missed: at least two distinct raw contents share one normalized form.

    Deterministic: clusters are ordered by the position of their first member and
    each ``example_ids`` list preserves input order.
    """
    examples = list(examples)
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(normalized_signature(ex), []).append(i)

    findings: list[DeduplicationFinding] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        # Skip clusters the exact stage already reports in full.
        if len({content_signature(examples[i]) for i in indices}) < 2:
            continue
        findings.append(
            DeduplicationFinding(
                kind=DedupKind.NORMALIZED,
                example_ids=[examples[i].example_id for i in indices],
                similarity=None,
                method=METHOD_NORMALIZED,
            )
        )
    return findings


__all__ = [
    "METHOD_NORMALIZED",
    "normalize_text",
    "tokenize",
    "word_ngrams",
    "jaccard",
    "normalized_signature",
    "normalized_duplicates",
]
