"""How much of a leaderboard survives contact with resampling (Milestone 9).

Two independent instruments, because they fail differently.

**Resampling.** Draw a fresh corpus (bootstrap over :class:`~.bootstrap.Units`), re-score,
re-rank, repeat. Report how often each system holds first place, how often the entire
ordering is reproduced, and the mean Kendall tau-b against the ordering the full corpus
gives. This measures sampling uncertainty — "would a different sample of examples have
produced this leaderboard?"

**Deletion.** Ask the exact combinatorial question instead: *what is the smallest set of
examples whose removal erases A's lead over B?* No resampling, no seed, no p-value — an
integer, computed exactly, by :func:`min_units_to_flip`.

The deletion answer is the one that tends to land. A reader can dismiss "p = 0.88" as
statistical fussiness. "Delete two of these fifty-six examples, named below, and the
winner loses" is harder to wave away, and it is not an estimate.

A note on ties. Bootstrap means over a 56-example corpus take values ``k/56``, so exact
ties between systems are common, not exotic. Breaking them by name would hand an
alphabetical system free first-place mass, so :func:`rank_frequencies` splits a tie's
credit evenly among the tied systems, and ``exact_order_frequency`` counts a resample only
when the ordering is strict. ``tie_fraction`` reports how often that mattered.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from eval_dataset_forge.statistics.bootstrap import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Units,
    resample_indices,
    unit_totals,
)

# Below this, a leaderboard's first place is a coin flip dressed as a result. A convention
# (like the 0.05 of a p-value), not a theorem; the report always prints the frequency it
# measured so a reader can apply their own bar.
MIN_TOP1_STABILITY = 0.95


# --------------------------------------------------------------------------- #
# Ranking under resampling
# --------------------------------------------------------------------------- #


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Tie-corrected Kendall tau-b between two score vectors. ``nan`` when either is flat.

    Computed on the *scores*, not on a tie-broken ranking, so two systems that tie in a
    resample contribute a tied pair rather than an arbitrary concordance.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} != {b.size}")
    if a.size < 2:
        return float("nan")
    sign_a = np.sign(a[:, None] - a[None, :])
    sign_b = np.sign(b[:, None] - b[None, :])
    upper = np.triu(np.ones_like(sign_a, dtype=bool), k=1)
    concordant = float((sign_a * sign_b)[upper].sum())
    tied_a = float((sign_a[upper] == 0).sum())
    tied_b = float((sign_b[upper] == 0).sum())
    n_pairs = float(upper.sum())
    denominator = math.sqrt((n_pairs - tied_a) * (n_pairs - tied_b))
    if denominator == 0.0:
        return float("nan")
    return concordant / denominator


@dataclass
class RankingStability:
    """What resampling does to a leaderboard.

    ``top1_frequency`` sums to 1 across systems: a resample in which two systems tie for
    first splits its credit between them.
    """

    systems: tuple[str, ...]
    resample_unit: str
    n_resamples: int
    seed: int
    point_ranking: tuple[str, ...] = ()
    top1_frequency: dict[str, float] = field(default_factory=dict)
    rank_frequency: dict[str, list[float]] = field(default_factory=dict)
    exact_order_frequency: float = float("nan")
    mean_kendall_tau: float = float("nan")
    tie_fraction: float = float("nan")

    @property
    def leader(self) -> str | None:
        return self.point_ranking[0] if self.point_ranking else None

    @property
    def leader_stability(self) -> float:
        leader = self.leader
        return self.top1_frequency.get(leader, float("nan")) if leader else float("nan")

    @property
    def stable(self) -> bool:
        """Does the corpus reproduce its own first place often enough to report one?"""
        stability = self.leader_stability
        return (not math.isnan(stability)) and stability >= MIN_TOP1_STABILITY

    def to_dict(self) -> dict:
        def number(value: float) -> float | None:
            return None if math.isnan(value) else round(value, 4)

        return {
            "systems": list(self.systems),
            "resample_unit": self.resample_unit,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "point_ranking": list(self.point_ranking),
            "leader_stability": number(self.leader_stability),
            "stable": self.stable,
            "top1_frequency": {k: round(v, 4) for k, v in self.top1_frequency.items()},
            "rank_frequency": {
                k: [round(v, 4) for v in row] for k, row in self.rank_frequency.items()
            },
            "exact_order_frequency": number(self.exact_order_frequency),
            "mean_kendall_tau": number(self.mean_kendall_tau),
            "tie_fraction": number(self.tie_fraction),
        }


def _resampled_means(matrix: np.ndarray, units: Units, draw: np.ndarray) -> np.ndarray:
    """``(n_systems, n_resamples)`` means, all systems sharing one set of draws."""
    sums = np.array([unit_totals(row, units)[0] for row in matrix])
    counts = units.sizes
    return sums[:, draw].sum(axis=2) / counts[draw].sum(axis=1)


def rank_frequencies(
    systems: Sequence[str],
    matrix: np.ndarray,
    units: Units,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    draw: np.ndarray | None = None,
) -> RankingStability:
    """Rank the systems in every resample and summarize how often the leaderboard holds.

    ``matrix`` is ``(n_systems, n_examples)`` of per-example scores. All systems are ranked
    on the *same* resampled corpus, which is the whole point: a leaderboard's uncertainty is
    about which system wins, not about how each one scores in isolation.
    """
    if matrix.ndim != 2 or matrix.shape[0] != len(systems):
        raise ValueError(f"matrix {matrix.shape} does not match {len(systems)} systems")
    if draw is None:
        draw = resample_indices(units.n_units, n_resamples=n_resamples, seed=seed)

    point_means = matrix.mean(axis=1)
    point_order = tuple(
        systems[i] for i in sorted(range(len(systems)), key=lambda i: (-point_means[i], systems[i]))
    )
    report = RankingStability(
        systems=tuple(systems),
        resample_unit=units.unit,
        n_resamples=int(draw.shape[0]),
        seed=seed,
        point_ranking=point_order,
    )
    if len(systems) < 2:
        return report

    means = _resampled_means(matrix, units, draw)  # (K, B)
    n_systems, n_draws = means.shape

    # Top-1, with ties splitting their credit rather than being broken by name.
    is_top = means == means.max(axis=0, keepdims=True)
    credit = is_top / is_top.sum(axis=0, keepdims=True)
    report.top1_frequency = dict(
        zip(systems, (credit.sum(axis=1) / n_draws).tolist(), strict=True)
    )

    # Rank distribution: position 0 = best. Ties share the positions they span.
    ranks = (means[:, None, :] < means[None, :, :]).sum(axis=1)  # strictly-better count
    frequency = np.zeros((n_systems, n_systems))
    for position in range(n_systems):
        frequency[:, position] = (ranks == position).sum(axis=1) / n_draws
    report.rank_frequency = dict(zip(systems, frequency.tolist(), strict=True))

    # Exact order: strict, so any tie disqualifies the resample.
    point_positions = np.array([point_order.index(name) for name in systems])
    strict = np.array([len(np.unique(column)) == n_systems for column in means.T])
    matches = np.all(ranks == point_positions[:, None], axis=0)
    report.exact_order_frequency = float(np.mean(matches & strict))
    report.tie_fraction = float(np.mean(~strict))

    taus = [kendall_tau_b(means[:, b], point_means) for b in range(n_draws)]
    finite = [tau for tau in taus if not math.isnan(tau)]
    report.mean_kendall_tau = float(np.mean(finite)) if finite else float("nan")
    return report


# --------------------------------------------------------------------------- #
# Ranking sensitivity to deletion: exact, not resampled
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FlipSet:
    """The smallest deletion that erases a lead, and what it costs.

    ``n_removed`` counts units (examples or blocks); ``n_examples_removed`` always counts
    examples. ``fraction`` is of the corpus, so 2/56 reads as 0.036 whether the unit was an
    example or a singleton block.
    """

    unit: str
    n_removed: int
    n_examples_removed: int
    n_examples: int
    removed_ids: tuple[str, ...] = ()

    @property
    def fraction(self) -> float:
        return self.n_examples_removed / self.n_examples if self.n_examples else float("nan")

    def to_dict(self) -> dict:
        return {
            "unit": self.unit,
            "n_removed": self.n_removed,
            "n_examples_removed": self.n_examples_removed,
            "fraction_of_corpus": round(self.fraction, 4),
            "removed_ids": list(self.removed_ids),
        }


def min_units_to_flip(
    differences: Sequence[float], units: Units, unit_ids: Sequence[str] | None = None
) -> FlipSet | None:
    """Fewest whole units whose deletion drives the mean paired difference to ``<= 0``.

    Exact, by a two-line argument. Deleting a set of units leaves mean
    ``(S - R) / (N - M)`` — total difference minus removed difference, over the surviving
    example count. Its **sign does not depend on the denominator**, so the example counts
    ``M`` of the deleted units are irrelevant: minimizing the remaining mean over all
    ``k``-unit deletions is exactly maximizing ``R``, which the ``k`` largest unit sums
    achieve. Greedy is optimal, no dynamic program is needed, and the answer is an integer
    rather than an estimate.

    The one constraint — never delete the whole corpus — costs nothing, because any
    ``k < n_units`` leaves at least one example standing.

    Blocks are the operationally meaningful unit: deleting a duplicate cluster is what a
    maintainer would actually *do* about it, so ``min_units_to_flip(d, block_units)`` reads
    as "how many clusters have to go before the leaderboard changes its mind".

    Returns ``None`` when no proper subset works. That is a real answer, not a failure: it
    means every unit's difference favours A, so A's lead is not the doing of a handful of
    items. ``n_removed == 0`` when the lead is already absent (``S <= 0``).
    """
    values = np.asarray(differences, dtype=float)
    if values.size != units.n_examples:
        raise ValueError(f"{values.size} differences for {units.n_examples} examples")
    n_units, n_examples = units.n_units, units.n_examples
    ids = tuple(unit_ids) if unit_ids is not None else tuple(str(i) for i in range(n_units))

    total = float(values.sum())
    if total <= 0.0:
        return FlipSet(unit=units.unit, n_removed=0, n_examples_removed=0, n_examples=n_examples)

    sums, sizes = unit_totals(values, units)
    order = np.argsort(-sums, kind="stable")
    removed = 0.0
    for k, index in enumerate(order[: n_units - 1], start=1):
        removed += float(sums[index])
        if total - removed <= 1e-12:
            chosen = order[:k]
            return FlipSet(
                unit=units.unit,
                n_removed=k,
                n_examples_removed=int(sizes[chosen].sum()),
                n_examples=n_examples,
                removed_ids=tuple(ids[i] for i in chosen),
            )
    return None


def min_examples_to_flip(
    differences: Sequence[float], example_ids: Sequence[str] | None = None
) -> FlipSet | None:
    """:func:`min_units_to_flip` with one example per unit — the headline deletion count."""
    return min_units_to_flip(
        differences, Units.per_example(len(differences)), unit_ids=example_ids
    )


__all__ = [
    "MIN_TOP1_STABILITY",
    "FlipSet",
    "RankingStability",
    "kendall_tau_b",
    "min_examples_to_flip",
    "min_units_to_flip",
    "rank_frequencies",
]
