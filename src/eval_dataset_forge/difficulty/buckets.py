"""Turning a continuous difficulty score into a coarse bucket (Milestone 8).

Three bucketing methods, differing only in where the two cut points come from:

============  ===================================  ==========================================
method        cut points                           property
============  ===================================  ==========================================
``quantile``  the score distribution's 1/3, 2/3    always fills all three buckets
``fixed``     absolute 1/3, 2/3 on ``[0, 1]``      comparable across datasets
``marginal``  wherever a target bucket count lands  reproduces a reference marginal
============  ===================================  ==========================================

**Equal scores always get equal buckets.** Every method compares ``score < cut``, so a
tie group is never split by a boundary. This matters more than it sounds: the committed
corpus has 56 examples and only 39 distinct scores, including two tie groups of four
(``bio-003``/``bio-006``/``saf-004``/``saf-005`` and
``saf-001``/``saf-002``/``saf-006``/``saf-008``). A rank-based cut that split ties would
make an example's bucket depend on its position in the input file. The price is that
``quantile`` cannot always deliver equal-frequency buckets — the pooled corpus lands at
18/19/19 rather than 19/18/19 — so :func:`bucket_distribution` reports the sizes it
achieved rather than the sizes that were requested.

Ties this dense are also why :mod:`.calibration` scores examples through
:func:`~eval_dataset_forge.difficulty.heuristics.raw_score` rather than a matrix product:
a 5.6e-17 summation difference is enough to split a tie group and move the reported rank
correlation.

**``fixed`` degenerates on short-answer corpora, and says so.** The score is a weighted
mean of log-compressed signals, so reaching 2/3 requires most signals near their cap. No
example in the committed corpus does: the maximum score is 0.658, and under absolute
thirds the buckets come out **45 / 11 / 0** — the ``hard`` bucket is empty. That is a
property of absolute cut points on a compressed scale, not of the data, and it is why
``quantile`` is the default. Pinned by
``test_fixed_cutoffs_leave_the_hard_bucket_empty_on_the_pooled_corpus``.

This module also owns :func:`declared_bucket` — the *hand-assigned* difficulty read off
``difficulty_tags``. It lives here rather than in :mod:`eval_dataset_forge.splits` because
difficulty is upstream of splits in the pipeline (``docs/PLAN.md`` §4), and because both
the calibration target and the stratification facet must read the declaration the same
way. :mod:`eval_dataset_forge.splits.stratified` re-exports it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from eval_dataset_forge.schemas import DatasetExample, DifficultyBucket

# Increasing order of difficulty; the index doubles as the ordinal encoding used by
# :mod:`.calibration` (easy=0, medium=1, hard=2).
BUCKET_ORDER: tuple[DifficultyBucket, ...] = (
    DifficultyBucket.EASY,
    DifficultyBucket.MEDIUM,
    DifficultyBucket.HARD,
)

# String view of the same order. Named ``DIFFICULTY_ORDER`` because that is the name
# :mod:`eval_dataset_forge.splits.stratified` has used since Milestone 7.
DIFFICULTY_ORDER: tuple[str, ...] = tuple(str(bucket) for bucket in BUCKET_ORDER)

# Stratum/bucket name for an example that declares no difficulty. Never imputed: an
# absent declaration is a fact about the dataset, not a value to be guessed.
UNLABELLED = "unlabelled"

# Equal-frequency cut points, as fractions of the score distribution.
DEFAULT_QUANTILES: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)

# Absolute cut points on the ``[0, 1]`` score. Naive thirds, deliberately not tuned to
# the corpus — see the module docstring for what they do to it.
FIXED_CUTOFFS: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)

BUCKETING_METHODS: tuple[str, ...] = ("quantile", "fixed", "marginal")


# --------------------------------------------------------------------------- #
# The declared (hand-assigned) bucket
# --------------------------------------------------------------------------- #


def declared_bucket(example: DatasetExample) -> DifficultyBucket | None:
    """The hardest :class:`DifficultyBucket` declared in ``difficulty_tags``, if any.

    Tags outside the bucket vocabulary (``coding``, ``ambiguous``, …) are ignored: they
    describe the example, not its difficulty. An example tagged with more than one bucket
    takes the hardest — a benchmark that mislabels a hard item as easy hurts more than
    the reverse.
    """
    tags = {tag.casefold() for tag in example.difficulty_tags}
    declared = [name for name in DIFFICULTY_ORDER if name in tags]
    return DifficultyBucket(declared[-1]) if declared else None


def declared_ordinal(example: DatasetExample) -> int | None:
    """``0``/``1``/``2`` for easy/medium/hard, or ``None`` when nothing is declared."""
    bucket = declared_bucket(example)
    return None if bucket is None else DIFFICULTY_ORDER.index(str(bucket))


# --------------------------------------------------------------------------- #
# Cut points
# --------------------------------------------------------------------------- #


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence (numpy's default)."""
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def quantile_cutpoints(
    scores: Iterable[float], quantiles: Sequence[float] = DEFAULT_QUANTILES
) -> tuple[float, float]:
    """The two score values at ``quantiles`` of the distribution. Corpus-relative."""
    ordered = sorted(float(score) for score in scores)
    if not ordered:
        return (0.0, 0.0)
    low, high = (_quantile(ordered, q) for q in quantiles)
    return (low, max(low, high))


def scale_counts(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Rescale bucket counts to sum to ``total``, preserving proportions.

    Largest-remainder apportionment, so the result sums to ``total`` exactly and no bucket
    is starved by naive rounding. Needed because the declared marginal is measured over the
    *labelled* examples while ``marginal`` bucketing is applied to *all* of them: on the
    pooled corpus that is 51 versus 56, and feeding the raw counts straight through would
    quietly dump all five undeclared examples into the ``hard`` bucket.
    """
    present = {name: max(0, int(count)) for name, count in counts.items()}
    source_total = sum(present.values())
    if source_total == 0 or total <= 0:
        return dict.fromkeys(present, 0)

    exact = {name: count * total / source_total for name, count in present.items()}
    scaled = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(scaled.values())
    order = sorted(present, key=lambda name: (-(exact[name] - scaled[name]), name))
    for name in order[:remaining]:
        scaled[name] += 1
    return scaled


def _cut_after(ordered: Sequence[float], k: int, floor: float) -> float:
    """The cut value that sends the ``k`` lowest scores below it.

    ``k == 0`` → a cut at or below the minimum, so nothing falls below it. ``k >= n`` →
    a cut above the maximum, so everything does. Otherwise the ``(k+1)``-th smallest
    score, which leaves the ``k`` lowest strictly below — except where a tie straddles
    the boundary, in which case the tied group stays together above it.
    """
    if k <= 0:
        return floor
    if k >= len(ordered):
        return ordered[-1] + 1.0
    return ordered[k]


def marginal_cutpoints(
    scores: Sequence[float], counts: Mapping[str, int]
) -> tuple[float, float]:
    """Cut points placing ``counts[easy]`` lowest scores in ``easy``, and so on.

    Used to compare a *ranking* against a reference marginal without also penalising it
    for predicting the wrong bucket frequencies. Ties are never split, so the achieved
    counts can differ from the requested ones; the caller sees the achieved ones.
    """
    ordered = sorted(float(score) for score in scores)
    if not ordered:
        return (0.0, 0.0)
    n_easy = int(counts.get(DIFFICULTY_ORDER[0], 0))
    n_medium = int(counts.get(DIFFICULTY_ORDER[1], 0))
    low = _cut_after(ordered, n_easy, floor=ordered[0])
    high = _cut_after(ordered, n_easy + n_medium, floor=low)
    return (low, max(low, high))


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def bucket_of(score: float, cutpoints: tuple[float, float]) -> DifficultyBucket:
    """Bucket one score. ``score < low`` → easy, ``< high`` → medium, else hard."""
    low, high = cutpoints
    if score < low:
        return DifficultyBucket.EASY
    if score < high:
        return DifficultyBucket.MEDIUM
    return DifficultyBucket.HARD


def cutpoints_for(
    scores: Sequence[float],
    method: str = "quantile",
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    fixed: tuple[float, float] = FIXED_CUTOFFS,
    target_counts: Mapping[str, int] | None = None,
) -> tuple[float, float]:
    """Resolve the two cut points for a bucketing ``method``."""
    if method == "quantile":
        return quantile_cutpoints(scores, quantiles)
    if method == "fixed":
        return (fixed[0], max(fixed[0], fixed[1]))
    if method == "marginal":
        if not target_counts:
            raise ValueError("bucketing method 'marginal' requires target_counts")
        return marginal_cutpoints(scores, target_counts)
    raise ValueError(f"unknown bucketing method {method!r} (known: {list(BUCKETING_METHODS)})")


def assign_buckets(
    scores: Sequence[float],
    method: str = "quantile",
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    fixed: tuple[float, float] = FIXED_CUTOFFS,
    target_counts: Mapping[str, int] | None = None,
) -> list[DifficultyBucket]:
    """Bucket every score. Equal scores always receive equal buckets."""
    cuts = cutpoints_for(
        scores, method, quantiles=quantiles, fixed=fixed, target_counts=target_counts
    )
    return [bucket_of(score, cuts) for score in scores]


# --------------------------------------------------------------------------- #
# Distributions — the milestone's reporting deliverable
# --------------------------------------------------------------------------- #


def bucket_distribution(buckets: Iterable[DifficultyBucket | str]) -> dict[str, int]:
    """``bucket name → count``, always listing all three buckets (zeros included).

    Zeros are kept because an empty ``hard`` bucket is the finding, and a dict that
    simply omitted the key would hide it. This is the shape
    :attr:`~eval_dataset_forge.schemas.AuditResult.difficulty_distribution` expects.
    """
    counts = dict.fromkeys(DIFFICULTY_ORDER, 0)
    for bucket in buckets:
        counts[str(bucket)] = counts.get(str(bucket), 0) + 1
    return counts


def declared_distribution(examples: Iterable[DatasetExample]) -> dict[str, int]:
    """``declared bucket → count``, with an ``unlabelled`` entry for undeclared examples."""
    counts = dict.fromkeys((*DIFFICULTY_ORDER, UNLABELLED), 0)
    for example in examples:
        bucket = declared_bucket(example)
        counts[UNLABELLED if bucket is None else str(bucket)] += 1
    return counts


def distribution_by_split(
    examples: Sequence[DatasetExample], buckets: Sequence[DifficultyBucket]
) -> dict[str, dict[str, int]]:
    """``split → bucket → count`` for the heuristic buckets. The imbalance report."""
    table: dict[str, list[DifficultyBucket]] = {}
    for example, bucket in zip(examples, buckets, strict=True):
        table.setdefault(str(example.split), []).append(bucket)
    return {split: bucket_distribution(members) for split, members in sorted(table.items())}


__all__ = [
    "BUCKETING_METHODS",
    "BUCKET_ORDER",
    "DEFAULT_QUANTILES",
    "DIFFICULTY_ORDER",
    "FIXED_CUTOFFS",
    "UNLABELLED",
    "assign_buckets",
    "bucket_distribution",
    "bucket_of",
    "cutpoints_for",
    "declared_bucket",
    "declared_distribution",
    "declared_ordinal",
    "distribution_by_split",
    "marginal_cutpoints",
    "quantile_cutpoints",
    "scale_counts",
]
