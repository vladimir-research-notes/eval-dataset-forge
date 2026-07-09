"""Does the difficulty score mean anything? (Milestone 8)

Every other stage in this pipeline can be scored against ground truth planted in the
fixtures. A difficulty *proxy* cannot: there is no ground-truth difficulty, only the
hand-assigned ``difficulty_tags`` an author wrote down. So this module does the next best
thing and asks a falsifiable question — **is the heuristic's ranking distinguishable from
a random ranking of the declared labels?** — and answers it with a permutation test.

The answer on the committed corpus is *no* — rho = +0.10, p = 0.50, n = 51. That is the
milestone's headline result, and the reason this module exists at all: without it,
``eval-dataset-forge`` would ship a "difficulty score" that is, on its own fixtures, a
task-family detector.

Six measurements, each guarding against a different way of fooling oneself:

``rho`` + ``p_value``
    Tie-corrected Spearman between score and declared ordinal, with an exact permutation
    null. A bare correlation is not evidence at n=51.

``per_signal``
    Each signal's own rho, its permutation p, and its **Bonferroni-corrected** p. Five
    signals were tested; reporting the best one's uncorrected p would be a fishing
    expedition.

``inert_signals``
    Signals with zero variance on this corpus. They contribute a constant to every score,
    so they change no ranking and no quantile bucket. Reporting a weight for them would
    imply work they cannot do.

``family_eta_squared`` vs ``declared_eta_squared``
    How much of the *score's* variance ``task_family`` explains, against how much of the
    *declared difficulty's* variance it explains. When the first is large and the second
    is not, the score is measuring family rather than difficulty — and a high overall rho
    would have been a family effect in disguise.

``marginal_accuracy``
    Bucket accuracy with the cut points placed so the predicted bucket *counts* equal the
    declared ones. This separates a bad ranking from a bad marginal: plain quantile
    bucketing is punished for predicting 1/3 ``hard`` when only 6% of the corpus is
    ``hard``, which says nothing about the ranking underneath. Compared against both the
    majority-class baseline and the accuracy a random ranking would achieve.

``weight_sensitivity``
    Spearman over 2000 Dirichlet draws from the weight simplex. If the null result held
    only at the shipped weights it would be a fact about the weights; sweeping shows it is
    a fact about the signals.

``required_n`` converts the null into a statement about the corpus rather than only about
the heuristic: how many labelled examples would be needed to detect an effect this size.

Nothing here reads ``difficulty_tags`` into a *score* — see :mod:`.heuristics` for why
that would make the whole exercise circular, and by how much.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from eval_dataset_forge.difficulty.buckets import (
    DIFFICULTY_ORDER,
    assign_buckets,
    bucket_distribution,
    declared_ordinal,
)
from eval_dataset_forge.difficulty.heuristics import (
    SIGNAL_NAMES,
    raw_score,
    resolve_weights,
    score_matrix,
)
from eval_dataset_forge.schemas import DatasetExample

# Below this many declared labels, a rank correlation is noise and the report says so
# instead of quoting a number. 30 is a convention, not a theorem; ``required_n`` gives the
# corpus-specific answer.
MIN_CALIBRATION_N = 30

# A declared bucket holding fewer than this many examples cannot support a per-bucket
# claim (the committed corpus has three ``hard`` examples in total).
MIN_BUCKET_N = 5

DEFAULT_ALPHA = 0.05
DEFAULT_RESAMPLES = 10_000
DEFAULT_WEIGHT_DRAWS = 2_000
DEFAULT_SEED = 0


# --------------------------------------------------------------------------- #
# Rank statistics (pure numpy — no scipy)
# --------------------------------------------------------------------------- #


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged, matching ``scipy.stats.rankdata``'s default.

    Tie averaging is what makes the correlation below a *tie-corrected* Spearman. It is
    not cosmetic here: the corpus produces 38 distinct scores over 56 examples, and the
    declared target has only three distinct values, so almost everything is tied.
    """
    array = np.asarray(values, dtype=float)
    n = array.size
    if n == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    ordered = array[order]
    start = 0
    for index in range(1, n + 1):
        if index == n or ordered[index] != ordered[start]:
            if index - start > 1:
                ranks[order[start:index]] = (start + index + 1) / 2.0
            start = index
    return ranks


def _centered(values: Sequence[float]) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=float)
    centered = array - array.mean()
    return centered, float(np.sqrt(np.dot(centered, centered)))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Tie-corrected Spearman rank correlation. ``nan`` when either side is constant."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} != {len(y)}")
    if len(x) < 3:
        return float("nan")
    cx, nx = _centered(rankdata(x))
    cy, ny = _centered(rankdata(y))
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return float(np.dot(cx, cy) / (nx * ny))


def permutation_p(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided permutation p-value for :func:`spearman`, under the label null.

    Permuting ``y`` and re-ranking is the same as permuting ``y``'s ranks, and a
    permutation changes neither the mean nor the norm of the centred ranks. So the whole
    null distribution is one matrix product against a fixed vector — no Python loop, and
    no approximation beyond the resampling itself.

    The estimator is ``(1 + #{|null| >= |observed|}) / (1 + n_resamples)`` (Phipson & Smyth
    2010), not the bare proportion. Counting the observed statistic among the resamples
    floors the result at ``1 / (1 + n_resamples)``. The bare proportion can return exactly
    0, which claims a certainty no finite number of resamples supports; a reader would then
    see ``p = 0.000`` for an effect the test can only bound. On truly null data the two
    estimators are near-identical (0.0595 versus 0.0605 rejections at alpha=0.05 over 2000
    null pairings at ``n_resamples=400``) — the correction buys validity at the floor, not
    a materially different rejection rate. Pinned by ``test_permutation_p_is_never_zero``
    and ``test_permutation_p_is_uniform_under_the_null``.
    """
    observed = spearman(x, y)
    if math.isnan(observed):
        return float("nan")
    cx, nx = _centered(rankdata(x))
    cy, ny = _centered(rankdata(y))
    rng = np.random.default_rng(seed)
    permuted = rng.permuted(np.broadcast_to(cy, (n_resamples, cy.size)), axis=1)
    null = permuted @ cx / (nx * ny)
    at_least_as_extreme = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    return (1.0 + at_least_as_extreme) / (1.0 + n_resamples)


def required_n(
    rho: float, *, alpha: float = DEFAULT_ALPHA, power: float = 0.8
) -> int | None:
    """Sample size needed to detect a correlation of ``rho`` at ``alpha`` and ``power``.

    Fisher z-transform: ``n = ((z_(1-α/2) + z_power) / artanh|ρ|)² + 3``. ``None`` when
    ``|ρ|`` is 0 or 1 (no finite answer / no test needed). Approximate — it assumes the
    bivariate-normal Fisher approximation, which tied ordinal data violates — so it is a
    scale, not a promise.
    """
    if math.isnan(rho):
        return None
    magnitude = min(abs(float(rho)), 1.0)
    if magnitude == 0.0 or magnitude == 1.0:
        return None
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    return int(math.ceil(((z_alpha + z_power) / math.atanh(magnitude)) ** 2 + 3))


def eta_squared(values: Sequence[float], groups: Sequence[str]) -> float:
    """Fraction of ``values``' variance explained by group membership. ``0`` when flat."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    grand = array.mean()
    total = float(np.sum((array - grand) ** 2))
    if total == 0.0:
        return 0.0
    between = 0.0
    for group in set(groups):
        members = array[[index for index, name in enumerate(groups) if name == group]]
        between += members.size * (members.mean() - grand) ** 2
    return float(between / total)


# --------------------------------------------------------------------------- #
# Report types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SignalDiagnostic:
    """What one signal contributes, and whether it can contribute anything at all."""

    name: str
    weight: float
    rho: float
    p_value: float
    p_bonferroni: float
    variance: float
    nonzero: int
    n_examples: int

    @property
    def inert(self) -> bool:
        """Zero variance: the signal adds a constant, so it changes no ranking."""
        return self.variance == 0.0

    @property
    def anticorrelated(self) -> bool:
        """Points the wrong way: higher signal, *lower* declared difficulty."""
        return (not math.isnan(self.rho)) and self.rho < 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weight": round(self.weight, 6),
            "rho": None if math.isnan(self.rho) else round(self.rho, 4),
            "p_value": None if math.isnan(self.p_value) else round(self.p_value, 4),
            "p_bonferroni": (
                None if math.isnan(self.p_bonferroni) else round(self.p_bonferroni, 4)
            ),
            "variance": round(self.variance, 6),
            "nonzero": self.nonzero,
            "n_examples": self.n_examples,
            "inert": self.inert,
            "anticorrelated": self.anticorrelated,
        }


@dataclass
class CalibrationReport:
    """Whether the heuristic's ranking tracks the declared difficulty — and how sure.

    ``trustworthy`` is the single bit the rest of the pipeline should read. It is False
    unless the correlation clears ``alpha``, the sample clears :data:`MIN_CALIBRATION_N`,
    and the marginal-matched bucketing beats the majority-class baseline. On the committed
    corpus it is False, and ``docs/DIFFICULTY.md`` says what that forbids.
    """

    n_labelled: int
    n_examples: int
    alpha: float = DEFAULT_ALPHA
    seed: int = DEFAULT_SEED
    n_resamples: int = DEFAULT_RESAMPLES

    rho: float = float("nan")
    p_value: float = float("nan")
    required_n_for_power: int | None = None

    per_signal: list[SignalDiagnostic] = field(default_factory=list)
    ablation: dict[str, float] = field(default_factory=dict)
    weight_sensitivity: dict[str, float] = field(default_factory=dict)

    within_family_rho: dict[str, float] = field(default_factory=dict)
    within_family_mean_rho: float = float("nan")
    family_baseline_rho: float = float("nan")
    score_eta_squared_family: float = 0.0
    declared_eta_squared_family: float = 0.0

    marginal_accuracy: float = float("nan")
    majority_baseline: float = float("nan")
    random_ranking_accuracy: float = float("nan")
    marginal_accuracy_p: float = float("nan")
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    declared_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def significant(self) -> bool:
        return (not math.isnan(self.p_value)) and self.p_value <= self.alpha

    @property
    def underpowered(self) -> bool:
        return self.n_labelled < MIN_CALIBRATION_N

    @property
    def beats_majority(self) -> bool:
        if math.isnan(self.marginal_accuracy) or math.isnan(self.majority_baseline):
            return False
        return self.marginal_accuracy > self.majority_baseline

    @property
    def trustworthy(self) -> bool:
        """May the heuristic bucket be used as if it were a difficulty label?"""
        return self.significant and not self.underpowered and self.beats_majority

    @property
    def inert_signals(self) -> list[str]:
        return [diagnostic.name for diagnostic in self.per_signal if diagnostic.inert]

    @property
    def anticorrelated_signals(self) -> list[str]:
        return [diagnostic.name for diagnostic in self.per_signal if diagnostic.anticorrelated]

    def to_dict(self) -> dict:
        def number(value: float) -> float | None:
            return None if math.isnan(value) else round(value, 4)

        return {
            "n_examples": self.n_examples,
            "n_labelled": self.n_labelled,
            "alpha": self.alpha,
            "seed": self.seed,
            "n_resamples": self.n_resamples,
            "trustworthy": self.trustworthy,
            "significant": self.significant,
            "underpowered": self.underpowered,
            "rho": number(self.rho),
            "p_value": number(self.p_value),
            "required_n_for_power": self.required_n_for_power,
            "per_signal": [diagnostic.to_dict() for diagnostic in self.per_signal],
            "inert_signals": self.inert_signals,
            "anticorrelated_signals": self.anticorrelated_signals,
            "ablation": {name: number(value) for name, value in self.ablation.items()},
            "weight_sensitivity": {
                name: number(value) for name, value in self.weight_sensitivity.items()
            },
            "within_family_rho": {
                name: number(value) for name, value in self.within_family_rho.items()
            },
            "within_family_mean_rho": number(self.within_family_mean_rho),
            "family_baseline_rho": number(self.family_baseline_rho),
            "score_eta_squared_family": round(self.score_eta_squared_family, 4),
            "declared_eta_squared_family": round(self.declared_eta_squared_family, 4),
            "marginal_accuracy": number(self.marginal_accuracy),
            "majority_baseline": number(self.majority_baseline),
            "random_ranking_accuracy": number(self.random_ranking_accuracy),
            "marginal_accuracy_p": number(self.marginal_accuracy_p),
            "confusion": self.confusion,
            "declared_counts": self.declared_counts,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# The measurements
# --------------------------------------------------------------------------- #


def _per_signal(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: Mapping[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> list[SignalDiagnostic]:
    """One diagnostic per signal, with Bonferroni correction over the signals tested.

    A signal with zero variance is reported as inert rather than tested: its rho is
    undefined, and pretending otherwise would put a ``nan`` where a reader expects a
    number.
    """
    tested = sum(1 for column in range(matrix.shape[1]) if matrix[:, column].var() > 0)
    diagnostics = []
    for column, name in enumerate(SIGNAL_NAMES):
        values = matrix[:, column]
        variance = float(values.var())
        if variance == 0.0:
            rho = p_value = p_bonferroni = float("nan")
        else:
            rho = spearman(values, target)
            p_value = permutation_p(values, target, n_resamples=n_resamples, seed=seed)
            p_bonferroni = min(1.0, p_value * max(1, tested))
        diagnostics.append(
            SignalDiagnostic(
                name=name,
                weight=float(weights[name]),
                rho=rho,
                p_value=p_value,
                p_bonferroni=p_bonferroni,
                variance=variance,
                nonzero=int(np.count_nonzero(values)),
                n_examples=int(values.size),
            )
        )
    return diagnostics


def _ablation(
    matrix: np.ndarray, target: np.ndarray, weights: Mapping[str, float]
) -> dict[str, float]:
    """Leave-one-signal-out rho. Renormalizing the survivors is cosmetic — Spearman is
    invariant to the positive rescaling that dropping a weight induces — but it keeps the
    ablated score on the same ``[0, 1]`` scale as the full one."""
    result: dict[str, float] = {}
    for dropped in SIGNAL_NAMES:
        remaining = 1.0 - weights[dropped]
        if remaining <= 0.0:
            result[dropped] = float("nan")
            continue
        vector = np.array(
            [0.0 if name == dropped else weights[name] / remaining for name in SIGNAL_NAMES]
        )
        result[dropped] = spearman(matrix @ vector, target)
    return result


def _weight_sensitivity(
    matrix: np.ndarray, target: np.ndarray, *, draws: int, seed: int
) -> dict[str, float]:
    """Spearman over Dirichlet draws from the weight simplex.

    Answers "is the null result a property of *these* weights?". If the best draw over the
    whole simplex still fails to reach significance, the answer is no — it is a property
    of the signal set, and no reweighting rescues it.

    Uses ``matrix @ vector`` rather than :func:`raw_score`, unlike the headline score. Each
    draw is a hypothetical weight vector that no report publishes, so there is nothing for
    it to stay bit-consistent with; every draw is summed the same way, so the comparison
    between draws is sound. The same applies to :func:`_ablation`.
    """
    if draws <= 0:
        return {}
    rng = np.random.default_rng(seed)
    vectors = rng.dirichlet(np.ones(matrix.shape[1]), size=draws)
    values = np.array([spearman(matrix @ vector, target) for vector in vectors])
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return {}
    return {
        "draws": float(draws),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
        "fraction_below_0.2": float(np.mean(np.abs(finite) < 0.2)),
    }


def _within_family(
    scores: np.ndarray, target: np.ndarray, families: Sequence[str]
) -> tuple[dict[str, float], float]:
    """Rho computed inside each task family, and its size-weighted mean.

    The control for the family effect. If the score merely ranks families, then within a
    family — where family is held constant — the correlation collapses.
    """
    per_family: dict[str, float] = {}
    numerator = denominator = 0.0
    for family in sorted(set(families)):
        index = [position for position, name in enumerate(families) if name == family]
        rho = spearman(scores[index], target[index])
        per_family[family] = rho
        if not math.isnan(rho):
            numerator += rho * len(index)
            denominator += len(index)
    mean = numerator / denominator if denominator else float("nan")
    return per_family, mean


def _family_baseline(target: np.ndarray, families: Sequence[str]) -> float:
    """Rho of "predict this family's mean declared difficulty" against the declaration.

    An *oracle* baseline: the family means are computed from the very labels being
    predicted, so it is an upper bound on what ``task_family`` alone could ever achieve.
    A heuristic that only detects family cannot beat it.
    """
    means: dict[str, float] = {}
    for family in set(families):
        index = [position for position, name in enumerate(families) if name == family]
        means[family] = float(target[index].mean())
    return spearman(np.array([means[name] for name in families]), target)


def _marginal_bucketing(
    scores: np.ndarray,
    target: np.ndarray,
    declared_counts: Mapping[str, int],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float, float, dict[str, dict[str, int]]]:
    """Accuracy when the predicted bucket *counts* are forced to the declared ones.

    Strips the marginal mismatch out of the comparison so what is left is the ranking.
    Returns ``(accuracy, majority, random_ranking, p_value, confusion)``.
    """
    predicted = assign_buckets(list(scores), "marginal", target_counts=declared_counts)
    predicted_ordinals = np.array([DIFFICULTY_ORDER.index(str(b)) for b in predicted])
    accuracy = float((predicted_ordinals == target).mean())

    total = len(target)
    majority = max(declared_counts.values()) / total if total else float("nan")
    # A random ranking scores sum_k (n_k/N)^2 in expectation once the marginal is matched.
    random_ranking = sum(count**2 for count in declared_counts.values()) / total**2

    rng = np.random.default_rng(seed)
    permuted = rng.permuted(np.broadcast_to(target, (n_resamples, total)), axis=1)
    null = (permuted == predicted_ordinals).mean(axis=1)
    # Same (1 + count) / (1 + B) correction as :func:`permutation_p`, for the same reason.
    at_least_as_good = int(np.count_nonzero(null >= accuracy - 1e-12))
    p_value = (1.0 + at_least_as_good) / (1.0 + n_resamples)

    confusion = {
        name: dict.fromkeys(DIFFICULTY_ORDER, 0) for name in DIFFICULTY_ORDER
    }
    for actual, guess in zip(target.astype(int), predicted_ordinals, strict=True):
        confusion[DIFFICULTY_ORDER[actual]][DIFFICULTY_ORDER[guess]] += 1
    return accuracy, majority, random_ranking, p_value, confusion


def _warnings(report: CalibrationReport) -> tuple[str, ...]:
    """Everything a reader must know before believing the bucket. Ordered by severity."""
    messages: list[str] = []
    if report.n_labelled == 0:
        return ("no example declares a difficulty tag; the heuristic cannot be calibrated",)
    if report.underpowered:
        messages.append(
            f"only {report.n_labelled} labelled example(s) (< {MIN_CALIBRATION_N}); "
            f"a rank correlation on this many is not evidence"
        )
    if math.isnan(report.rho):
        # Constant scores, a constant declared label, or fewer than three labels. "Not
        # distinguishable from a random ranking" would be a category error: there is no
        # correlation to compare against one.
        messages.append(
            "the rank correlation is undefined (the score or the declared label takes a "
            "single value across the labelled examples); the heuristic bucket must not be "
            "used as a difficulty label"
        )
    elif not report.significant:
        messages.append(
            f"rho={report.rho:+.3f} (p={report.p_value:.3f}) is not distinguishable from a "
            f"random ranking at alpha={report.alpha}; the heuristic bucket must not be used "
            f"as a difficulty label"
        )
    if report.required_n_for_power:
        messages.append(
            f"detecting an effect of this size at 80% power would need about "
            f"{report.required_n_for_power} labelled examples; the dataset has "
            f"{report.n_labelled}"
        )
    if not report.beats_majority and not math.isnan(report.marginal_accuracy):
        messages.append(
            f"marginal-matched bucket accuracy {report.marginal_accuracy:.3f} does not beat "
            f"the majority-class baseline {report.majority_baseline:.3f}"
        )
    if (
        report.score_eta_squared_family > 0.25
        and report.declared_eta_squared_family < 0.10
    ):
        messages.append(
            f"task_family explains {report.score_eta_squared_family:.0%} of the score's "
            f"variance but only {report.declared_eta_squared_family:.0%} of the declared "
            f"difficulty's; the score is largely a task-family detector"
        )
    for diagnostic in report.per_signal:
        if not diagnostic.inert:
            continue
        # An inert signal is constant. It is constant at zero (the dataset never populates
        # it) or at some other value (every example scores the same). Both change no
        # ranking, but only the first means "fires on nothing", and saying so of the second
        # would be false.
        reason = (
            f"fires on 0 of {diagnostic.n_examples} examples"
            if diagnostic.nonzero == 0
            else f"takes the same value on all {diagnostic.n_examples} examples"
        )
        messages.append(
            f"signal {diagnostic.name!r} has zero variance on this dataset ({reason}); it "
            f"contributes a constant and changes no ranking or bucket"
        )
    for diagnostic in report.per_signal:
        if diagnostic.anticorrelated:
            messages.append(
                f"signal {diagnostic.name!r} is anti-correlated with declared difficulty "
                f"(rho={diagnostic.rho:+.3f}); it pushes the score the wrong way"
            )
    for name, count in report.declared_counts.items():
        if 0 < count < MIN_BUCKET_N:
            messages.append(
                f"declared bucket {name!r} holds {count} example(s) (< {MIN_BUCKET_N}); "
                f"per-bucket statistics for it are not meaningful"
            )
    return tuple(messages)


def calibrate(
    examples: Sequence[DatasetExample],
    *,
    weights: Mapping[str, float] | None = None,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    weight_draws: int = DEFAULT_WEIGHT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> CalibrationReport:
    """Measure the heuristic ranking against the declared ``difficulty_tags``.

    Deterministic given ``seed``. Examples that declare no bucket are excluded from every
    statistic (never imputed), so ``n_labelled`` can be smaller than ``n_examples``.
    """
    examples = list(examples)
    resolved = resolve_weights(weights)

    labelled = [
        (example, declared_ordinal(example))
        for example in examples
        if declared_ordinal(example) is not None
    ]
    report = CalibrationReport(
        n_labelled=len(labelled),
        n_examples=len(examples),
        alpha=alpha,
        seed=seed,
        n_resamples=n_resamples,
    )
    if len(labelled) < 3:
        report.warnings = _warnings(report)
        return report

    subset = [example for example, _ in labelled]
    target = np.array([ordinal for _, ordinal in labelled], dtype=float)
    families = [example.task_family for example in subset]

    _, rows = score_matrix(subset)
    matrix = np.array(rows, dtype=float)

    # The headline score comes from ``raw_score``, not from ``matrix @ vector``. The two
    # agree to 5.6e-17 and *disagree about ties*: numpy's pairwise summation and Python's
    # sequential ``sum`` place the same five contributions in a different order, which is
    # enough to split the corpus's four-way tie {bio-003, bio-006, saf-004, saf-005} into
    # two pairs and move the reported rho from +0.097447 to +0.097467. Ranks over a corpus
    # with 39 distinct scores in 56 examples are that brittle. ``score_dataset`` publishes
    # ``raw_score``, so the correlation must be computed over exactly those numbers or it
    # describes scores no reader can see. Pinned by
    # ``test_calibration_scores_examples_the_same_way_the_report_does``.
    scores = np.array([raw_score(example, resolved) for example in subset])

    report.rho = spearman(scores, target)
    report.p_value = permutation_p(scores, target, n_resamples=n_resamples, seed=seed)
    report.required_n_for_power = required_n(report.rho, alpha=alpha)
    report.per_signal = _per_signal(
        matrix, target, resolved, n_resamples=n_resamples, seed=seed
    )
    report.ablation = _ablation(matrix, target, resolved)
    report.weight_sensitivity = _weight_sensitivity(
        matrix, target, draws=weight_draws, seed=seed
    )
    report.within_family_rho, report.within_family_mean_rho = _within_family(
        scores, target, families
    )
    report.family_baseline_rho = _family_baseline(target, families)

    # eta^2 of the score uses *every* example: the score exists without a declared label.
    all_scores = np.array([raw_score(example, resolved) for example in examples])
    report.score_eta_squared_family = eta_squared(
        all_scores, [example.task_family for example in examples]
    )
    report.declared_eta_squared_family = eta_squared(target, families)

    report.declared_counts = bucket_distribution(
        DIFFICULTY_ORDER[int(ordinal)] for ordinal in target
    )
    (
        report.marginal_accuracy,
        report.majority_baseline,
        report.random_ranking_accuracy,
        report.marginal_accuracy_p,
        report.confusion,
    ) = _marginal_bucketing(
        scores, target, report.declared_counts, n_resamples=n_resamples, seed=seed
    )

    report.warnings = _warnings(report)
    return report


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "DEFAULT_WEIGHT_DRAWS",
    "MIN_BUCKET_N",
    "MIN_CALIBRATION_N",
    "CalibrationReport",
    "SignalDiagnostic",
    "calibrate",
    "eta_squared",
    "permutation_p",
    "rankdata",
    "required_n",
    "spearman",
]
