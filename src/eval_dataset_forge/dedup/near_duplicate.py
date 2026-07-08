"""Heuristic near-duplicate detection (Milestone 4).

Level 3 of the deduplication ladder, and the only one that can be wrong in both
directions. Levels 1–2 answer "are these the same string?"; this stage answers
"are these the same *item*?", which no lexical method decides correctly. Its
output is a ranked review queue, not a verdict, and nothing here is ever
auto-dropped.

Two CPU-only methods share one feature space — the set of word n-grams of the
prompt — and differ only in how they weight and compare it:

``jaccard``
    Unweighted set overlap, |A ∩ B| / |A ∪ B|. Pure Python, no dependencies, and
    dataset-independent: a pair's score never changes when other examples are
    added. Every n-gram counts the same, so prompts sharing a template ("Who
    wrote the play *X*?") — or merely sharing stopwords — score high on
    boilerplate alone.

``tfidf``
    Cosine similarity of idf-weighted n-gram vectors (scikit-learn). Down-weights
    n-grams common across the corpus and up-weights distinctive ones, which is
    exactly the discrimination ``jaccard`` lacks. The cost is that idf is a
    property of the **corpus**: a pair's score depends on what else is in the
    dataset and is not comparable across datasets. Falls back to ``jaccard`` when
    scikit-learn is absent.

On the committed fixtures TF-IDF dominates Jaccard at every threshold — never
fewer true positives, never more false ones — which is why ``auto`` prefers it.

**The answer gate.** By default a pair must also agree on its gold answer
(:func:`answers_agree`). A paraphrased prompt with a *different* gold answer is
not a duplicate to be dropped; it is either a mislabeling or two genuinely
different questions, and both are the label audit's business (Milestone 6). The
gate is not cosmetic: on the fixtures it removes the two highest-scoring
non-duplicate pairs, each of which otherwise outranks a real planted duplicate.
It is disabled with ``require_same_answer=False``.

**What this cannot do.** Deep paraphrases with low lexical overlap are invisible
here, and not because the threshold is set wrong. In ``retrieval_toy`` the
planted paraphrase ``ret-001``/``ret-002`` scores *below* a non-duplicate pair in
the same fixture, so no threshold — with or without the gate — separates them.
Detecting it needs semantics (embeddings), which the offline build does not have.
``docs/DEDUPLICATION.md`` carries the measured calibration table and the failure
modes each default accepts.

Comparison is brute-force O(n²) over the prompt set. At fixture scale that is
microseconds; the blocking/MinHash step a large corpus would need is deliberately
deferred rather than claimed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from typing import Any

from eval_dataset_forge.dedup.exact import canonical_output
from eval_dataset_forge.dedup.fuzzy import (
    jaccard,
    normalize_text,
    normalized_signature,
    tokenize,
    word_ngrams,
)
from eval_dataset_forge.schemas import DatasetExample, DedupKind, DeduplicationFinding

# Default word-n-gram width. n=1 (bag of words) is the most permissive choice and
# the right one for paraphrase detection on short prompts: a paraphrase reorders
# and substitutes words, which destroys contiguous n-grams long before it destroys
# the vocabulary. Wider n trades recall for precision — measured in
# docs/DEDUPLICATION.md: going to n=2 drops Jaccard from 3 of the 5 planted
# paraphrase pairs to 1.
DEFAULT_NGRAM = 1

# Per-method thresholds, calibrated on the 56 committed fixture examples: 0.30 is
# the knee of the precision/recall curve for both methods (see the sweep in
# docs/DEDUPLICATION.md). Kept as a per-method mapping because the two scores are
# *not* on a common scale — an idf-weighted cosine and a raw set overlap only
# coincide here — so a different corpus will move them apart.
#
# The defaults favor recall: these findings feed a review queue and the
# near-duplicate-aware split strategy (Milestone 7), where a missed duplicate
# silently leaks across splits and a false one merely pins two examples together.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "jaccard": 0.30,
    "tfidf": 0.30,
}

# Similarity scores are rounded before they reach a finding, so a report is stable
# across platforms and floating-point noise never shows up in a diff.
_SIMILARITY_DIGITS = 4

Method = str  # "auto" | "jaccard" | "tfidf"
TextOf = Callable[[DatasetExample], str]


def default_text_of(example: DatasetExample) -> str:
    """Near-duplicate similarity is computed over the prompt.

    The gold answer is excluded from the *similarity* on purpose: answers repeat
    legitimately — every yes/no item shares one of two answers — so including them
    would make the score partly a function of label frequency. Answer agreement is
    applied separately, as a gate (:func:`answers_agree`).
    """
    return example.input


def answers_agree(left: Any, right: Any) -> bool:
    """Whether two ``expected_output`` values represent the same gold answer.

    Structured answers (rubrics, doc-id lists) compare by canonical JSON, so key
    order is irrelevant. Two *string* answers additionally compare after
    :func:`~eval_dataset_forge.dedup.fuzzy.normalize_text`, so ``"Paris"`` and
    ``"paris."`` agree. A string is never considered equal to a non-string.

    Known blind spot: for retrieval-style data whose gold answer is a list of
    document *ids*, two examples can cite identical document *text* under
    different ids and be judged to disagree. Such datasets should pass
    ``require_same_answer=False``.
    """
    left_canonical, right_canonical = canonical_output(left), canonical_output(right)
    if left_canonical == right_canonical:
        return True
    if isinstance(left, str) and isinstance(right, str):
        return normalize_text(left_canonical) == normalize_text(right_canonical)
    return False


def sklearn_available() -> bool:
    """Whether the optional TF-IDF accelerator can be imported.

    A function rather than a module-level constant so tests can monkeypatch it and
    exercise the pure-Python fallback without uninstalling scikit-learn.
    """
    try:
        return importlib.util.find_spec("sklearn") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def resolve_near_method(method: Method = "auto") -> str:
    """Resolve ``"auto"`` and degrade ``"tfidf"`` to ``"jaccard"`` when unavailable.

    Requesting ``tfidf`` without scikit-learn is not an error: the caller gets the
    pure-Python method, and the returned name — which is what lands in
    ``DeduplicationFinding.method`` — records what actually ran.
    """
    if method not in ("auto", "jaccard", "tfidf"):
        raise ValueError(f"unknown near-duplicate method {method!r}")
    if method == "jaccard":
        return "jaccard"
    return "tfidf" if sklearn_available() else "jaccard"


def method_label(method: str, ngram: int, threshold: float) -> str:
    """The self-describing ``method`` string recorded on every near finding."""
    return f"{method}(word_{ngram}gram)@{threshold:g}"


# --------------------------------------------------------------------------- #
# Pairwise similarity
# --------------------------------------------------------------------------- #


def _jaccard_matrix(texts: list[str], ngram: int) -> list[list[float]]:
    grams = [word_ngrams(tokenize(text), ngram) for text in texts]
    size = len(texts)
    sims = [[0.0] * size for _ in range(size)]
    for i in range(size):
        sims[i][i] = 1.0
        for j in range(i + 1, size):
            sims[i][j] = sims[j][i] = jaccard(grams[i], grams[j])
    return sims


def _tfidf_matrix(texts: list[str], ngram: int) -> list[list[float]]:
    """Cosine similarity of idf-weighted word-n-gram vectors.

    The analyzer reuses :func:`~eval_dataset_forge.dedup.fuzzy.tokenize` and
    :func:`~eval_dataset_forge.dedup.fuzzy.word_ngrams`, so both methods see an
    identical feature space and only the weighting differs. ``TfidfVectorizer``
    L2-normalizes its rows, so the Gram matrix ``X @ X.T`` *is* the cosine matrix.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    def analyzer(text: str) -> list[str]:
        return [" ".join(gram) for gram in sorted(word_ngrams(tokenize(text), ngram))]

    if not any(analyzer(text) for text in texts):
        # An all-empty corpus has no vocabulary; TfidfVectorizer would raise.
        return [[0.0] * len(texts) for _ in texts]

    matrix = TfidfVectorizer(analyzer=analyzer).fit_transform(texts)
    gram = (matrix @ matrix.T).toarray()
    return [[float(value) for value in row] for row in gram]


def similarity_matrix(
    examples: Iterable[DatasetExample],
    *,
    method: Method = "auto",
    ngram: int = DEFAULT_NGRAM,
    text_of: TextOf = default_text_of,
) -> tuple[list[list[float]], str]:
    """Return the full symmetric similarity matrix and the method that produced it.

    Exposed for calibration and testing: it is the raw evidence behind
    :data:`DEFAULT_THRESHOLDS`, and it deliberately applies no threshold and no
    answer gate.
    """
    resolved = resolve_near_method(method)
    texts = [text_of(example) for example in examples]
    if not texts:
        return [], resolved
    matrix = (
        _tfidf_matrix(texts, ngram) if resolved == "tfidf" else _jaccard_matrix(texts, ngram)
    )
    return matrix, resolved


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def _connected_components(edges: list[tuple[int, int, float]]) -> list[list[int]]:
    """Group indices joined by at least one edge (single-linkage).

    Single-linkage is transitive on purpose: if A~B and B~C both clear the
    threshold, all three surface as one cluster for review even when A and C do
    not directly clear it. That is the right direction for a review queue, but it
    means a chain of pairwise-similar items can merge into a cluster whose
    endpoints are dissimilar — which is why the reported ``similarity`` is the
    weakest edge in the cluster rather than the strongest.
    """
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, j, _ in edges:
        parent.setdefault(i, i)
        parent.setdefault(j, j)
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    groups: dict[int, list[int]] = {}
    for node in sorted(parent):
        groups.setdefault(find(node), []).append(node)
    return [sorted(members) for _, members in sorted(groups.items())]


def near_duplicates(
    examples: Iterable[DatasetExample],
    *,
    method: Method = "auto",
    threshold: float | None = None,
    ngram: int = DEFAULT_NGRAM,
    require_same_answer: bool = True,
    text_of: TextOf = default_text_of,
) -> list[DeduplicationFinding]:
    """Find clusters of prompts whose lexical similarity clears ``threshold``.

    ``threshold`` defaults to the per-method value in :data:`DEFAULT_THRESHOLDS`.
    Pairs that are already exact or normalized duplicates are skipped, so this
    never re-reports a cluster the cheaper stages own, and — unless
    ``require_same_answer=False`` — pairs whose gold answers disagree are skipped
    too.

    Each finding's ``similarity`` is the **weakest** edge in its cluster: for the
    common two-member cluster that is the measured pair similarity, and for a
    larger single-linkage cluster it is an honest lower bound on how alike the
    members are.
    """
    examples = list(examples)
    resolved = resolve_near_method(method)
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[resolved]
    if len(examples) < 2:
        return []

    matrix, _ = similarity_matrix(examples, method=resolved, ngram=ngram, text_of=text_of)
    signatures = [normalized_signature(example) for example in examples]

    edges: list[tuple[int, int, float]] = []
    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            if signatures[i] == signatures[j]:
                continue  # owned by the exact / normalized stages
            if require_same_answer and not answers_agree(
                examples[i].expected_output, examples[j].expected_output
            ):
                continue
            score = round(matrix[i][j], _SIMILARITY_DIGITS)
            if score >= threshold:
                edges.append((i, j, score))

    label = method_label(resolved, ngram, threshold)
    findings: list[DeduplicationFinding] = []
    for members in _connected_components(edges):
        member_set = set(members)
        weakest = min(
            score for i, j, score in edges if i in member_set and j in member_set
        )
        findings.append(
            DeduplicationFinding(
                kind=DedupKind.NEAR,
                example_ids=[examples[i].example_id for i in members],
                similarity=min(weakest, 1.0),
                method=label,
            )
        )
    return findings


__all__ = [
    "DEFAULT_NGRAM",
    "DEFAULT_THRESHOLDS",
    "Method",
    "default_text_of",
    "answers_agree",
    "sklearn_available",
    "resolve_near_method",
    "method_label",
    "similarity_matrix",
    "near_duplicates",
]
