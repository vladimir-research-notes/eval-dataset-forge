"""Retrieval-document contamination (Milestone 5).

Retrieval examples carry candidate passages under
``metadata.retrieval.docs = [{doc_id, text, relevant}, …]``. The check is keyed on
that *shape*, not on ``task_family == "retrieval"``, so a QA item that ships
retrieved context is audited too.

Two signals, both exact:

**1. Annotation artifacts in passage text.** A passage carrying ``[ANSWER: Mars]``,
``label:``, or ``ground truth =`` has had the annotation pipeline leak into the
corpus. A retriever can then rank by the marker instead of by relevance, and the
"hard" retrieval step is free. Detected by :data:`ARTIFACT_MARKER`, a small
curated pattern rather than a general answer search.

**2. A gold passage shared across splits.** When the same passage text appears in a
training example and in a test example — under different ``doc_id``s, which is how
it evades id-level checks — the held-out evidence was never held out. Distractors
are *not* flagged: reusing an irrelevant passage across examples is normal corpus
construction, so a cluster only counts when the passage is marked relevant for at
least one of the examples it appears in.

**What is deliberately not checked.** ``docs/LEAKAGE.md`` originally proposed
flagging "the gold answer string embedded in a passage". For retrieval-family data
that check is wrong: a relevant passage containing the answer is the *definition* of
relevance, so the check would fire on every correctly-labelled example and flag
nothing. The prompt-side analogue — the answer quoted in the query — is real, and
:mod:`.answer_in_prompt` owns it.

Passage comparison is on :func:`~eval_dataset_forge.dedup.fuzzy.normalize_text`, so
casing and punctuation do not hide a duplicate. It is exact-after-normalization: a
*paraphrased* duplicate passage (``retrieval_toy``'s ``d13``/``d14``, two wordings
of the same boiling-point fact) is not detected, and no threshold here would find
it without also merging unrelated passages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from eval_dataset_forge.dedup.exact import sha256_hex
from eval_dataset_forge.dedup.fuzzy import normalize_text
from eval_dataset_forge.leakage.train_test_overlap import ASSIGNED_SPLITS
from eval_dataset_forge.schemas import DatasetExample, LeakageFinding, LeakageKind, Split

# Annotation markers that should never survive into a retrieval corpus. Curated on
# purpose: a broad pattern (a bare "correct", say) fires on ordinary prose. Each
# alternative is a label word immediately followed by ':' or '=' — the shape of a
# machine-written annotation, not of an English sentence. Over every passage and
# prompt in the committed fixtures this matches exactly once, on the planted
# `[ANSWER: Mars]` artifact in `retrieval_toy`.
ARTIFACT_MARKER = re.compile(
    r"\b(?:correct[\s_-]?answer|ground[\s_-]?truth|gold(?:[\s_-]?(?:answer|label))?"
    r"|answer|label|relevance|relevant)\b\s*[:=]",
    re.IGNORECASE,
)

# Passage snippet length in evidence strings. Long enough to see the artifact in
# context, short enough that a report stays readable.
_SNIPPET_CHARS = 90


@dataclass(frozen=True)
class DocRef:
    """One passage, located by the example that carries it."""

    example_id: str
    doc_id: str
    split: Split
    relevant: bool

    def __str__(self) -> str:
        return f"{self.example_id}/{self.doc_id}[{self.split}]"


def retrieval_docs(example: DatasetExample) -> list[dict[str, Any]]:
    """Candidate passages of an example, or ``[]`` when it carries none.

    Defensive about shape: ``metadata`` is free-form by contract, so a malformed
    ``retrieval`` block yields no passages instead of raising. Structural
    enforcement of metadata is not this module's job.
    """
    retrieval = example.metadata.get("retrieval")
    if not isinstance(retrieval, dict):
        return []
    docs = retrieval.get("docs")
    if not isinstance(docs, list):
        return []
    return [doc for doc in docs if isinstance(doc, dict)]


def _is_relevant(example: DatasetExample, doc: dict[str, Any]) -> bool:
    """Whether a passage is gold evidence for its example.

    Prefers the passage's own ``relevant`` flag; falls back to membership in a
    doc-id-list ``expected_output``. The two disagree on ``retrieval_toy``'s
    ``ret-004`` (a passage flagged relevant while ``expected_output`` is empty) —
    that is the planted incomplete-relevance defect, and the label audit owns it.
    """
    flag = doc.get("relevant")
    if isinstance(flag, bool):
        return flag
    expected = example.expected_output
    return isinstance(expected, list) and doc.get("doc_id") in expected


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_CHARS:
        return collapsed
    return collapsed[: _SNIPPET_CHARS - 1] + "…"


def label_artifacts(examples: Iterable[DatasetExample]) -> list[LeakageFinding]:
    """Passages whose text carries an annotation marker."""
    findings: list[LeakageFinding] = []
    for example in examples:
        for doc in retrieval_docs(example):
            text = doc.get("text")
            if not isinstance(text, str):
                continue
            match = ARTIFACT_MARKER.search(text)
            if match is None:
                continue
            findings.append(
                LeakageFinding(
                    kind=LeakageKind.RETRIEVAL_CONTAMINATION,
                    example_ids=[example.example_id],
                    score=None,
                    evidence=(
                        f"doc {doc.get('doc_id', '?')} contains the annotation marker "
                        f"{match.group(0)!r}: {_snippet(text)!r}"
                    ),
                )
            )
    return findings


def document_clusters(examples: Iterable[DatasetExample]) -> list[list[DocRef]]:
    """Group passages whose normalized text is identical, keeping clusters of ≥2.

    Includes clusters confined to a single example and single split. Informational:
    a repeated passage inflates a corpus, but only the cross-split ones are leakage.
    Deterministic — clusters in order of first appearance, members in input order.
    """
    groups: dict[str, list[DocRef]] = {}
    for example in examples:
        for doc in retrieval_docs(example):
            text = doc.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            key = sha256_hex(normalize_text(text))
            groups.setdefault(key, []).append(
                DocRef(
                    example_id=example.example_id,
                    doc_id=str(doc.get("doc_id", "?")),
                    split=example.split,
                    relevant=_is_relevant(example, doc),
                )
            )
    return [refs for refs in groups.values() if len(refs) >= 2]


def cross_split_documents(examples: Iterable[DatasetExample]) -> list[LeakageFinding]:
    """Gold passages shared, verbatim, by examples in ≥2 assigned splits."""
    findings: list[LeakageFinding] = []
    for refs in document_clusters(examples):
        splits = {ref.split for ref in refs if ref.split in ASSIGNED_SPLITS}
        example_ids = list(dict.fromkeys(ref.example_id for ref in refs))
        if len(splits) < 2 or len(example_ids) < 2:
            continue
        if not any(ref.relevant for ref in refs):
            continue  # a shared distractor is ordinary corpus construction
        findings.append(
            LeakageFinding(
                kind=LeakageKind.RETRIEVAL_CONTAMINATION,
                example_ids=example_ids,
                score=None,
                evidence=(
                    "identical gold passage in "
                    + " ↔ ".join(str(ref) for ref in refs)
                    + " (different doc_ids, same text)"
                ),
            )
        )
    return findings


def retrieval_contamination(examples: Iterable[DatasetExample]) -> list[LeakageFinding]:
    """Both retrieval signals: annotation artifacts, then cross-split gold passages."""
    examples = list(examples)
    return [*label_artifacts(examples), *cross_split_documents(examples)]


__all__ = [
    "ARTIFACT_MARKER",
    "DocRef",
    "cross_split_documents",
    "document_clusters",
    "label_artifacts",
    "retrieval_contamination",
    "retrieval_docs",
]
