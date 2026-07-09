"""Resampling machinery: what a bootstrap resamples, and why it matters (Milestone 9).

The nonparametric bootstrap answers one question: *how much would this number move if we
drew a fresh dataset from the same process?* Every implementation therefore encodes an
assumption about what "a fresh dataset" is made of. The textbook version resamples
**examples**, which assumes examples are independent. The dedup ladder (Milestone 4) exists
precisely because that assumption is false on real corpora: 20 of the 56 pooled fixture
examples sit in a duplicate cluster, and a deterministic system scores every member of an
exact cluster identically. Sampling those examples independently pretends to have more
information than the corpus contains.

So the resampling **unit** is a first-class parameter here, not a hidden default:

===================  ==============================  ================================
unit                 what one draw pulls              assumption
===================  ==============================  ================================
:data:`EXAMPLE`      one example                      examples are independent
:data:`BLOCK`        one duplicate/source block       *blocks* are independent
===================  ==============================  ================================

Blocks come from :func:`eval_dataset_forge.splits.blocks.duplicate_blocks`, the same
union-find components the adversarial split strategy uses. The two stages therefore agree
about what a dependent group is, and both inherit the near-duplicate stage's recall bound:
a paraphrase the dedup ladder cannot see yields no block edge, so the cluster bootstrap
cannot widen for it either.

Both distributions are computed on every run so the report can quote the **design effect**
— the ratio of their variances. It is the number that says how much the i.i.d. bootstrap
was lying by. On the pooled fixtures it is ~1.5, meaning the honest CI is ~22% wider and
the corpus is worth ~37 independent examples, not 56.

Estimators here are the ratio estimator ``sum(scores) / count`` over the resampled units,
which reduces to the ordinary mean when the unit is the example. Two interval methods:
percentile (default) and BCa. ``docs/STATISTICS.md`` records the coverage simulation that
chose the default — BCa is *not* uniformly better, and is not the default for that reason.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from eval_dataset_forge.schemas import BootstrapResult

# Resampling units. ``BLOCK`` is the default everywhere in this package: a corpus with no
# duplicates has one example per block, so choosing it costs nothing and never
# under-corrects.
EXAMPLE = "example"
BLOCK = "block"
RESAMPLE_UNITS = (EXAMPLE, BLOCK)

# 2000, not the 1000 of the original design note. The 2.5% tail of a B=1000 bootstrap is
# estimated from 25 draws, and the resulting endpoint jitter (~0.5 accuracy points on a
# 56-example corpus) is the same order as the effects this milestone reports. 2000 halves
# it and still runs in milliseconds. See docs/milestones/09-statistics.md.
DEFAULT_RESAMPLES = 2_000
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05

# Interval methods.
PERCENTILE = "percentile"
BCA = "bca"
CI_METHODS = (PERCENTILE, BCA)

_NORMAL = NormalDist()


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Units:
    """The indivisible groups a resample draws from, as positions into a score vector.

    ``members[u]`` holds the example positions owned by unit ``u``. The union of all
    members is the whole corpus, so the point estimate is the same whichever unit is used
    — only the *variance* of the resampling distribution differs.
    """

    members: tuple[tuple[int, ...], ...]
    unit: str = EXAMPLE

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("at least one unit is required")
        if self.unit not in RESAMPLE_UNITS:
            raise ValueError(f"unknown resample unit {self.unit!r}; want one of {RESAMPLE_UNITS}")

    @classmethod
    def per_example(cls, n_examples: int) -> Units:
        """One unit per example — the i.i.d. assumption, made explicit."""
        return cls(members=tuple((i,) for i in range(n_examples)), unit=EXAMPLE)

    @classmethod
    def from_blocks(cls, blocks: Iterable, position_of: dict[str, int]) -> Units:
        """One unit per :class:`~eval_dataset_forge.splits.blocks.Block`.

        Members keep their input order, and blocks keep theirs, so a ``Units`` built twice
        from the same dedup report is identical — the bootstrap draw is then reproducible
        from the seed alone.
        """
        members = tuple(
            tuple(position_of[example_id] for example_id in block.example_ids)
            for block in blocks
        )
        return cls(members=members, unit=BLOCK)

    @property
    def n_units(self) -> int:
        return len(self.members)

    @property
    def n_examples(self) -> int:
        return sum(len(member) for member in self.members)

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(member) for member in self.members], dtype=float)

    @property
    def mean_size_kish(self) -> float:
        """``sum(m^2) / sum(m)`` — the cluster size that enters Kish's design effect.

        Not the arithmetic mean size. A corpus of 34 singletons, 8 pairs, and 2 triples
        has arithmetic mean 1.27 and Kish mean 1.50: the larger clusters carry more
        examples, so they weigh more in the variance.
        """
        sizes = self.sizes
        return float((sizes**2).sum() / sizes.sum())

    def size_histogram(self) -> dict[int, int]:
        sizes = [len(member) for member in self.members]
        return {size: sizes.count(size) for size in sorted(set(sizes))}


def unit_totals(values: Sequence[float], units: Units) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit ``(sum of scores, number of examples)``."""
    array = np.asarray(values, dtype=float)
    if array.size != units.n_examples:
        raise ValueError(f"{array.size} values for {units.n_examples} examples")
    sums = np.array([array[list(member)].sum() for member in units.members])
    return sums, units.sizes


# --------------------------------------------------------------------------- #
# The resampling distribution
# --------------------------------------------------------------------------- #


def resample_indices(n_units: int, *, n_resamples: int, seed: int) -> np.ndarray:
    """A ``(n_resamples, n_units)`` matrix of unit indices drawn with replacement.

    Shared across systems on purpose. A **paired** bootstrap requires that resample ``b``
    contain the same examples for every system, or the difference of two marginal
    distributions would carry variance that the pairing removes. Every caller in this
    package draws once and reuses.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(n_resamples, n_units))


def bootstrap_distribution(
    values: Sequence[float],
    units: Units,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    draw: np.ndarray | None = None,
) -> np.ndarray:
    """Resampling distribution of the mean score, resampling whole ``units``.

    The statistic is the ratio estimator ``sum(scores) / count`` over the drawn units. When
    units are examples this is the ordinary sample mean; when they are blocks the resampled
    corpus has a variable example count, and dividing by the *realized* count is what makes
    the estimator consistent for the population mean rather than for a size-weighted
    surrogate.
    """
    sums, counts = unit_totals(values, units)
    if draw is None:
        draw = resample_indices(units.n_units, n_resamples=n_resamples, seed=seed)
    return sums[draw].sum(axis=1) / counts[draw].sum(axis=1)


def jackknife_means(values: Sequence[float], units: Units) -> np.ndarray:
    """Leave-one-unit-out means. Feeds BCa's acceleration term."""
    sums, counts = unit_totals(values, units)
    total, n_total = sums.sum(), counts.sum()
    remaining = n_total - counts
    if np.any(remaining <= 0):  # a single unit owns the whole corpus
        return np.full(units.n_units, np.nan)
    return (total - sums) / remaining


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #


def percentile_ci(distribution: np.ndarray, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """The ``[alpha/2, 1 - alpha/2]`` quantiles of the resampling distribution."""
    low, high = np.percentile(distribution, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def bca_ci(
    values: Sequence[float],
    units: Units,
    distribution: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float] | None:
    """Bias-corrected and accelerated interval, or ``None`` when it is not computable.

    BCa shifts and stretches the percentile endpoints by a bias correction ``z0`` (how far
    the resampling distribution sits from the point estimate) and an acceleration ``a``
    (the jackknife skewness of the statistic across units).

    It returns ``None`` — rather than a number — in the two degenerate cases that a
    56-example benchmark actually hits: every resample landing on the same side of the point
    estimate (``z0`` infinite, which happens when the accuracy is at 0 or 1), and an adjusted
    quantile leaving ``(0, 100)``. Callers fall back to :func:`percentile_ci` and say so in
    the reported ``method``, because silently substituting one interval for another would
    make the label on the output false.

    **BCa is not the default.** The coverage simulation in ``docs/STATISTICS.md`` finds it
    repairs the percentile interval's under-coverage on clustered data (0.884 → 0.947 at a
    true accuracy of 0.90) and *breaks* it on near-degenerate i.i.d. data (0.926 → 0.830 at
    a true accuracy of 0.95, where 5.7% of trials fall back). An interval method that is
    better on average and worse exactly where the data is thinnest is offered, not imposed.
    """
    array = np.asarray(values, dtype=float)
    point = float(array.mean())

    below = float(np.mean(distribution < point))
    if below <= 0.0 or below >= 1.0:
        return None
    z0 = _NORMAL.inv_cdf(below)

    jackknife = jackknife_means(array, units)
    if np.any(np.isnan(jackknife)):
        return None
    deviation = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(deviation**2)) ** 1.5
    acceleration = 0.0 if denominator == 0.0 else float(np.sum(deviation**3) / denominator)

    quantiles = []
    for tail in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = _NORMAL.inv_cdf(tail)
        shifted = z0 + z
        adjustment = 1.0 - acceleration * shifted
        if adjustment == 0.0:
            return None
        quantiles.append(100.0 * _NORMAL.cdf(z0 + shifted / adjustment))
    if not 0.0 < quantiles[0] < quantiles[1] < 100.0:
        return None
    low, high = np.percentile(distribution, quantiles)
    return float(low), float(high)


def bootstrap_ci(
    values: Sequence[float],
    units: Units,
    *,
    metric: str = "accuracy",
    system: str | None = None,
    method: str = PERCENTILE,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    draw: np.ndarray | None = None,
) -> BootstrapResult:
    """A :class:`~eval_dataset_forge.schemas.BootstrapResult` for the mean of ``values``.

    ``method`` records what was actually used: a BCa request that hit a degenerate case is
    reported as ``"bca->percentile"``, never as ``"bca"``.
    """
    if method not in CI_METHODS:
        raise ValueError(f"unknown ci method {method!r}; want one of {CI_METHODS}")
    distribution = bootstrap_distribution(
        values, units, n_resamples=n_resamples, seed=seed, draw=draw
    )
    interval = None
    if method == BCA:
        interval = bca_ci(values, units, distribution, alpha)
    used = method if interval is not None else (
        f"{BCA}->{PERCENTILE}" if method == BCA else PERCENTILE
    )
    if interval is None:
        interval = percentile_ci(distribution, alpha)
    return BootstrapResult(
        metric=metric,
        system=system,
        point_estimate=float(np.asarray(values, dtype=float).mean()),
        ci_low=interval[0],
        ci_high=interval[1],
        n_resamples=n_resamples,
        seed=seed,
        method=used,
        resample_unit=units.unit,
    )


# --------------------------------------------------------------------------- #
# How much the i.i.d. assumption was worth
# --------------------------------------------------------------------------- #


def intraclass_correlation(values: Sequence[float], units: Units) -> float:
    """One-way random-effects ICC(1): the fraction of score variance that is between units.

    ``nan`` when it is undefined rather than zero — when every unit is a singleton (no
    within-unit variation exists to compare against) or when the score is constant (there is
    no variance to decompose). Both happen on real corpora and both must not be reported as
    "no dependence".

    For an **exact**-duplicate cluster the ICC is 1 by construction, for any deterministic
    scorer: the examples are byte-identical, so any function of them returns the same score.
    That is not an empirical claim about these fixtures; it is why the cluster bootstrap is
    the correct default rather than a conservative one.
    """
    array = np.asarray(values, dtype=float)
    n_units, n_examples = units.n_units, units.n_examples
    if n_units < 2 or n_examples == n_units:
        return float("nan")
    grand = array.mean()
    between = 0.0
    within = 0.0
    for member in units.members:
        group = array[list(member)]
        between += group.size * (group.mean() - grand) ** 2
        within += float(((group - group.mean()) ** 2).sum())
    mean_square_between = between / (n_units - 1)
    mean_square_within = within / (n_examples - n_units)
    # Kish's m0: the "typical" cluster size for unbalanced designs.
    m0 = (n_examples - float((units.sizes**2).sum()) / n_examples) / (n_units - 1)
    denominator = mean_square_between + (m0 - 1.0) * mean_square_within
    if denominator == 0.0:
        return float("nan")
    return float((mean_square_between - mean_square_within) / denominator)


def kish_design_effect(units: Units, icc: float) -> float:
    """``1 + (mean_size_kish - 1) * icc`` — the closed-form variance inflation.

    Derived for equal cluster sizes and applied here to unequal ones, so it is an
    approximation. On the pooled fixtures it predicts 1.50 at ``icc = 1`` where the measured
    variance ratio is 1.52–1.62; the formula understates, and the report quotes both.
    """
    if math.isnan(icc):
        return float("nan")
    return 1.0 + (units.mean_size_kish - 1.0) * icc


def measured_design_effect(clustered: np.ndarray, independent: np.ndarray) -> float:
    """Var(cluster bootstrap) / Var(i.i.d. bootstrap) — what the intervals actually did.

    Preferred over :func:`kish_design_effect` for reporting, because it is a property of the
    intervals a reader will see rather than of a formula's assumptions. It carries Monte
    Carlo noise of order ``sqrt(2 / n_resamples)``; at the default 2000 resamples that is
    ~3%.
    """
    denominator = float(independent.var())
    if denominator == 0.0:
        return float("nan")
    return float(clustered.var()) / denominator


def effective_n(n_examples: int, design_effect: float) -> float:
    """``n / deff`` — the number of independent examples this corpus is worth."""
    if math.isnan(design_effect) or design_effect <= 0.0:
        return float("nan")
    return n_examples / design_effect


__all__ = [
    "BCA",
    "BLOCK",
    "CI_METHODS",
    "DEFAULT_ALPHA",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "EXAMPLE",
    "PERCENTILE",
    "RESAMPLE_UNITS",
    "Units",
    "bca_ci",
    "bootstrap_ci",
    "bootstrap_distribution",
    "effective_n",
    "intraclass_correlation",
    "jackknife_means",
    "kish_design_effect",
    "measured_design_effect",
    "percentile_ci",
    "resample_indices",
    "unit_totals",
]
