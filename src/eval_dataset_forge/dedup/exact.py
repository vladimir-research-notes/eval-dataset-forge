"""Exact, hash-based duplicate detection (Milestone 4).

Level 1 of the deduplication ladder: byte-identical content. This stage is
**exact** — it has no threshold, no tunable, and no false positives. If it fires,
two examples carry the same prompt and the same gold answer, and one of them adds
nothing to the benchmark except weight.

An example's identity for dedup purposes is the pair ``(input, expected_output)``
— not ``example_id``, not ``split``, not ``metadata``. Two records with different
ids that pose the same question and expect the same answer are duplicates
regardless of which split they landed in; that is precisely the qa-001/qa-002
defect planted in the fixtures. (Repeated ``example_id``\\ s are a *structural*
error owned by the validation layer's
:func:`~eval_dataset_forge.validation.schema.find_duplicate_ids`, which reports
them against file ordinals before any example object exists.)

``expected_output`` may be a string, a scalar, or a structured rubric, so it is
run through :func:`canonical_output` before hashing: JSON with sorted keys, so
``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` hash alike, and the digest is
stable across interpreter runs (unlike ``hash()`` or ``repr()`` of a dict).
Encoding is applied to *every* value, strings included: a gold answer of ``"4"``
and one of ``4`` are different answers, and a dataset that mixes the two has an
inconsistent-format defect the label audit should see — not a duplicate this
stage should silently merge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from eval_dataset_forge.schemas import DatasetExample, DedupKind, DeduplicationFinding

# Field separator used when signing a multi-field key; see fuzzy._SEP.
_SEP = "\x00"

METHOD_EXACT = "sha256(input+expected_output)"


def sha256_hex(text: str) -> str:
    """sha256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_output(value: Any) -> str:
    """Render an ``expected_output`` to a stable, comparable string.

    Compact JSON with sorted keys, so the encoding is independent of dict
    insertion order and stable across interpreter runs.

    Strings are encoded too, rather than passed through: an unquoted passthrough
    would map the string ``"4"`` and the integer ``4`` — and ``"true"`` and
    ``True`` — onto the same digest, silently declaring two differently-typed
    gold answers to be one duplicate.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_signature(example: DatasetExample) -> str:
    """sha256 over the exact prompt and canonical gold answer of one example."""
    return sha256_hex(_SEP.join([example.input, canonical_output(example.expected_output)]))


def _cluster_by(
    examples: list[DatasetExample], key_of
) -> list[tuple[str, list[DatasetExample]]]:
    """Group examples by ``key_of(example)``, keeping only clusters of size >= 2.

    Clusters are returned in order of first appearance, and members keep input
    order, so every caller's output is deterministic.
    """
    groups: dict[str, list[DatasetExample]] = {}
    for ex in examples:
        groups.setdefault(key_of(ex), []).append(ex)
    return [(key, members) for key, members in groups.items() if len(members) >= 2]


def exact_duplicates(examples: Iterable[DatasetExample]) -> list[DeduplicationFinding]:
    """Find clusters of examples with byte-identical prompt *and* gold answer.

    ``similarity`` is ``None`` on the resulting findings: these are identical by
    construction, and reporting ``1.0`` would invite a reader to compare them
    against heuristic near-duplicate scores, which are a different quantity.
    """
    return [
        DeduplicationFinding(
            kind=DedupKind.EXACT,
            example_ids=[ex.example_id for ex in members],
            similarity=None,
            method=METHOD_EXACT,
        )
        for _, members in _cluster_by(list(examples), content_signature)
    ]


def duplicate_input_clusters(examples: Iterable[DatasetExample]) -> list[list[str]]:
    """Group example ids by byte-identical ``input``, ignoring the gold answer.

    Reported separately from :func:`exact_duplicates` because the two mean
    different things: a shared prompt with *differing* answers is not a duplicate
    to be dropped but a label conflict to be adjudicated (Milestone 6). This
    helper exposes the superset so later stages can take the difference.
    """
    return [
        [ex.example_id for ex in members]
        for _, members in _cluster_by(list(examples), lambda ex: sha256_hex(ex.input))
    ]


def duplicate_sources(examples: Iterable[DatasetExample]) -> dict[str, list[str]]:
    """Map ``source_id`` -> example ids for source records used by >1 example.

    Informational, not an error. Two examples derived from one source record are
    legitimate (the fixtures' ``src-shakespeare`` yields distinct questions about
    Romeo and Juliet and about Hamlet), but they are *statistically* dependent:
    a split builder that scatters them across train and test leaks source-level
    information, and a bootstrap that resamples them independently overstates its
    precision. Milestone 7's source-grouped split strategy consumes this.
    """
    return {
        source_id: [ex.example_id for ex in members]
        for source_id, members in _cluster_by(list(examples), lambda ex: ex.source_id)
    }


__all__ = [
    "METHOD_EXACT",
    "sha256_hex",
    "canonical_output",
    "content_signature",
    "exact_duplicates",
    "duplicate_input_clusters",
    "duplicate_sources",
]
