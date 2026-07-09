"""Comparing two systems, and the three ways that goes wrong (Milestone 9).

Marginal confidence intervals are the wrong instrument for "is A better than B". Two
overlapping 95% intervals routinely hide a significant difference, because the systems are
scored on the *same* examples and that pairing removes example difficulty from the
comparison. So everything here operates on the per-example difference ``d_i = a_i - b_i``.

Three quantities, deliberately not conflated:

``delta`` + a bootstrap CI
    Effect size and its uncertainty. Resamples units, not examples (see
    :mod:`.bootstrap`), so the interval widens for duplicate clusters.

``win_fraction``
    The share of resamples in which A came out ahead. Widely reported, widely misread.
    It is **not** a p-value, and the tool never uses it as a decision rule — see below.

``p_value``
    A cluster-aware sign-flip permutation test. Under the null that A and B are
    exchangeable within each unit, flipping the sign of a whole unit's difference leaves
    the distribution unchanged; flipping *examples* independently would assume away the
    dependence the block structure exists to model.

**The win fraction is not a p-value, and the error is not small.** A common rule — "A beats
B if A wins at least 95% of the bootstrap resamples" — is a *one-sided* 95% statement,
applied post hoc in whichever direction the data happened to favour. Under a true null
(``n = 56``, two independent Bernoulli(0.7) systems, 3000 trials) it rejects **11.4%** of
the time against a nominal 5%. The threshold that corresponds to a two-sided ``alpha=0.05``
is 0.975, which measured 5.7%. The sign-flip test at ``alpha=0.05`` rejected 2.9% — it is
*conservative* on paired binary data, because the many tied examples make the null
distribution coarse. All three numbers are reported by ``docs/STATISTICS.md`` and pinned by
``tests/test_statistics.py``.

**Multiplicity.** ``K`` systems produce ``K(K-1)/2`` pairwise tests, and the chance that
some pair looks significant grows with ``K``. :func:`holm` applies the Holm–Bonferroni
step-down correction, which controls the family-wise error rate under any dependence
structure between the tests — the relevant property here, since pairwise comparisons over a
shared example set are strongly dependent. ``significant`` reads the *corrected* p.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from eval_dataset_forge.schemas import BootstrapResult
from eval_dataset_forge.statistics.bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    PERCENTILE,
    Units,
    bootstrap_ci,
    bootstrap_distribution,
    resample_indices,
    unit_totals,
)
from eval_dataset_forge.statistics.ranking_stability import (
    FlipSet,
    min_examples_to_flip,
    min_units_to_flip,
)

DEFAULT_POWER = 0.8

_NORMAL = NormalDist()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def sign_flip_p(
    differences: Sequence[float],
    units: Units,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided sign-flip permutation p-value for the mean paired difference.

    The null is that A and B are exchangeable, so ``d`` is symmetric about zero. Flipping
    signs at the **unit** level preserves whatever dependence lives inside a duplicate
    cluster; flipping per example would silently restore the independence assumption the
    cluster bootstrap was introduced to remove.

    Uses the Phipson & Smyth ``(1 + count) / (1 + B)`` estimator — the same one
    :mod:`eval_dataset_forge.difficulty.calibration` uses, for the same reason: the bare
    proportion can return exactly 0, claiming a certainty no finite resampling supports.

    When every unit's difference is exactly zero the observed statistic and the entire null
    are zero, and the estimator returns 1.0 — two systems that agree on every example are
    not distinguishable, which is the right answer.
    """
    sums, counts = unit_totals(differences, units)
    n_examples = counts.sum()
    observed = float(sums.sum() / n_examples)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, units.n_units))
    null = (signs * sums).sum(axis=1) / n_examples
    at_least_as_extreme = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    return (1.0 + at_least_as_extreme) / (1.0 + n_resamples)


def monte_carlo_se(p_value: float, n_resamples: int) -> float:
    """Standard error of a resampled p-value: ``sqrt(p(1-p)/B)``.

    A p-value estimated from ``B`` draws is itself an estimate, and at the default ``B=2000``
    its standard error near ``p=0.05`` is 0.005. A comparison whose p lands within a couple
    of standard errors of ``alpha`` has not been resolved by the resampling — it has been
    resolved by the seed. :func:`SystemComparison.p_is_borderline` says so, and the fix is
    more resamples, not a different conclusion.
    """
    if math.isnan(p_value) or n_resamples <= 0:
        return float("nan")
    return math.sqrt(max(p_value * (1.0 - p_value), 0.0) / n_resamples)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values, in the input order.

    Controls the family-wise error rate under arbitrary dependence between the tests, which
    is what pairwise comparisons over one shared example set have. Uniformly more powerful
    than plain Bonferroni and just as valid; there is no reason to prefer the latter.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------- #
# Small-N arithmetic
# --------------------------------------------------------------------------- #


def minimum_detectable_effect(
    sd: float,
    n_examples: int,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    design_effect: float = 1.0,
) -> float:
    """Smallest detectable paired difference: ``(z_alpha + z_power) * sd * sqrt(deff / n)``.

    A normal approximation to the paired test, inflated by the design effect so the answer
    describes the corpus's *effective* size rather than its row count. Approximate — the
    paired differences of binary scores are trinomial, not normal — so it is a scale, not a
    promise. It is nonetheless the number that decides whether a benchmark of this size can
    say anything at all about a gap of a given size.
    """
    if sd <= 0.0 or n_examples <= 0 or math.isnan(design_effect) or design_effect <= 0.0:
        return float("nan")
    z = _NORMAL.inv_cdf(1.0 - alpha / 2.0) + _NORMAL.inv_cdf(power)
    return z * sd * math.sqrt(design_effect / n_examples)


def required_n_for_delta(
    delta: float,
    sd: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    design_effect: float = 1.0,
) -> int | None:
    """Examples needed to detect a paired difference of ``delta``. ``None`` when undefined.

    The inverse of :func:`minimum_detectable_effect`, and the honest way to report a null:
    "this corpus cannot resolve the gap it observed, and here is how many examples could".
    """
    if delta == 0.0 or sd <= 0.0 or math.isnan(delta) or math.isnan(design_effect):
        return None
    z = _NORMAL.inv_cdf(1.0 - alpha / 2.0) + _NORMAL.inv_cdf(power)
    return int(math.ceil((z * sd / abs(delta)) ** 2 * max(design_effect, 1e-12)))


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #


@dataclass
class SystemComparison:
    """One ordered pair ``(a, b)``, with everything needed to judge ``a > b`` or refuse to.

    ``significant`` reads ``p_holm``, never ``p_value`` and never ``win_fraction``.
    """

    system_a: str
    system_b: str
    delta: float
    interval: BootstrapResult
    win_fraction: float
    p_value: float
    p_holm: float = float("nan")
    p_mc_se: float = float("nan")
    alpha: float = DEFAULT_ALPHA
    n_examples: int = 0
    n_discordant: int = 0
    sd: float = float("nan")
    design_effect: float = 1.0
    minimum_detectable_effect: float = float("nan")
    required_n: int | None = None
    example_flip: FlipSet | None = None
    unit_flip: FlipSet | None = None

    @property
    def ci_low(self) -> float:
        return self.interval.ci_low

    @property
    def ci_high(self) -> float:
        return self.interval.ci_high

    @property
    def ci_contains_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high

    @property
    def significant(self) -> bool:
        """Corrected for multiplicity. A raw p that clears alpha is not a finding."""
        return (not math.isnan(self.p_holm)) and self.p_holm <= self.alpha

    @property
    def p_is_borderline(self) -> bool:
        """Is ``p`` closer to ``alpha`` than the resampling can resolve?"""
        if math.isnan(self.p_mc_se) or self.p_mc_se == 0.0:
            return False
        return abs(self.p_value - self.alpha) <= 2.0 * self.p_mc_se

    @property
    def ci_disagrees_with_test(self) -> bool:
        """Does the interval exclude zero while the permutation test declines to reject?

        Not a bug and not a tie-break to be resolved by picking the friendlier number. The
        bootstrap interval is centred on the observed effect; the sign-flip null is centred
        on zero and keeps each unit's ``|difference|``, so the effect's own magnitude enters
        the null's spread. The permutation test is therefore the more conservative of the
        two near the boundary, and it is the one ``significant`` reads.
        """
        return (not self.ci_contains_zero) and not self.significant

    @property
    def underpowered(self) -> bool:
        """Is the observed gap smaller than the smallest gap this corpus could detect?"""
        mde = self.minimum_detectable_effect
        return (not math.isnan(mde)) and abs(self.delta) < mde

    def to_dict(self) -> dict:
        def number(value: float) -> float | None:
            return None if math.isnan(value) else round(value, 6)

        return {
            "system_a": self.system_a,
            "system_b": self.system_b,
            "delta": number(self.delta),
            "ci_low": number(self.ci_low),
            "ci_high": number(self.ci_high),
            "ci_contains_zero": self.ci_contains_zero,
            "resample_unit": self.interval.resample_unit,
            "ci_method": self.interval.method,
            "win_fraction": number(self.win_fraction),
            "p_value": number(self.p_value),
            "p_holm": number(self.p_holm),
            "p_mc_se": number(self.p_mc_se),
            "p_is_borderline": self.p_is_borderline,
            "ci_disagrees_with_test": self.ci_disagrees_with_test,
            "significant": self.significant,
            "underpowered": self.underpowered,
            "n_examples": self.n_examples,
            "n_discordant": self.n_discordant,
            "sd_of_differences": number(self.sd),
            "design_effect": number(self.design_effect),
            "minimum_detectable_effect": number(self.minimum_detectable_effect),
            "required_n": self.required_n,
            "min_examples_to_flip": self.example_flip.to_dict() if self.example_flip else None,
            "min_units_to_flip": self.unit_flip.to_dict() if self.unit_flip else None,
        }


def compare_pair(
    system_a: str,
    system_b: str,
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    units: Units,
    *,
    example_ids: Sequence[str] | None = None,
    unit_ids: Sequence[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    ci_method: str = PERCENTILE,
    design_effect: float = 1.0,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    draw: np.ndarray | None = None,
) -> SystemComparison:
    """Paired comparison of two systems on the same examples.

    ``design_effect`` only enters the *power* arithmetic. The CI and the p-value get their
    dependence correction from ``units`` directly, by resampling and sign-flipping whole
    blocks; multiplying them by a design effect as well would double-count it.
    """
    differences = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if draw is None:
        draw = resample_indices(units.n_units, n_resamples=n_resamples, seed=seed)

    interval = bootstrap_ci(
        differences,
        units,
        metric=f"delta_accuracy[{system_a}-{system_b}]",
        system=f"{system_a}-{system_b}",
        method=ci_method,
        alpha=alpha,
        n_resamples=n_resamples,
        seed=seed,
        draw=draw,
    )
    distribution = bootstrap_distribution(differences, units, draw=draw)

    n_examples = int(differences.size)
    delta = float(differences.mean())
    sd = float(differences.std(ddof=1)) if n_examples > 1 else float("nan")
    p_value = sign_flip_p(differences, units, n_resamples=n_resamples, seed=seed)
    return SystemComparison(
        system_a=system_a,
        system_b=system_b,
        delta=delta,
        interval=interval,
        win_fraction=float(np.mean(distribution > 0.0)),
        p_value=p_value,
        p_mc_se=monte_carlo_se(p_value, n_resamples),
        alpha=alpha,
        n_examples=n_examples,
        n_discordant=int(np.count_nonzero(differences)),
        sd=sd,
        design_effect=design_effect,
        minimum_detectable_effect=minimum_detectable_effect(
            sd, n_examples, alpha=alpha, power=power, design_effect=design_effect
        ),
        required_n=required_n_for_delta(
            delta, sd, alpha=alpha, power=power, design_effect=design_effect
        ),
        example_flip=min_examples_to_flip(differences, example_ids),
        unit_flip=min_units_to_flip(differences, units, unit_ids),
    )


def compare_systems(
    scores: Mapping[str, Sequence[float]],
    units: Units,
    *,
    example_ids: Sequence[str] | None = None,
    unit_ids: Sequence[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    ci_method: str = PERCENTILE,
    design_effect: float = 1.0,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[SystemComparison]:
    """Every unordered pair, oriented so ``delta >= 0``, with Holm-corrected p-values.

    Orienting each pair by the observed winner is a *post hoc* choice, which is exactly why
    the p-value is two-sided and the correction is applied across all ``K(K-1)/2`` pairs.
    """
    names = list(scores)
    draw = resample_indices(units.n_units, n_resamples=n_resamples, seed=seed)
    comparisons: list[SystemComparison] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            first, second = name_a, name_b
            if float(np.mean(scores[name_b])) > float(np.mean(scores[name_a])):
                first, second = name_b, name_a
            comparisons.append(
                compare_pair(
                    first,
                    second,
                    scores[first],
                    scores[second],
                    units,
                    example_ids=example_ids,
                    unit_ids=unit_ids,
                    alpha=alpha,
                    power=power,
                    ci_method=ci_method,
                    design_effect=design_effect,
                    n_resamples=n_resamples,
                    seed=seed,
                    draw=draw,
                )
            )
    for comparison, adjusted in zip(
        comparisons, holm([c.p_value for c in comparisons]), strict=True
    ):
        comparison.p_holm = adjusted
    return comparisons


__all__ = [
    "DEFAULT_POWER",
    "SystemComparison",
    "compare_pair",
    "compare_systems",
    "holm",
    "minimum_detectable_effect",
    "monte_carlo_se",
    "required_n_for_delta",
    "sign_flip_p",
]
