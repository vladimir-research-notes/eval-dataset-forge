"""Statistical uncertainty & ranking stability (Milestone 9).

Audit dimension #8: *can this dataset tell these systems apart?*

Every earlier stage describes the dataset. This one describes what the dataset can
**support** — and the answer, for a benchmark of the size most teams actually build, is
much less than its leaderboard implies. Three findings, each measured rather than asserted:

**1. The i.i.d. bootstrap is anti-conservative on any corpus with duplicates.** The textbook
bootstrap resamples examples, which assumes examples are independent. They are not: a
deterministic system scores every member of an exact-duplicate cluster identically, so the
within-cluster correlation is 1 *by construction*. Resampling
:class:`~.bootstrap.Units` — the duplicate blocks Milestone 7 already builds — instead of
examples widens the interval by the square root of the design effect. On the pooled
fixtures that is ~1.5, so the honest interval is ~22% wider and the corpus is worth ~37
independent examples rather than 56. A coverage simulation with known ground truth confirms
the direction: at a true accuracy of 0.70 the i.i.d. interval covers **88.2%** of the time
against a nominal 95%; the cluster interval covers 93.7%.

**2. The bootstrap "win fraction" is not a p-value.** The common rule "A beats B if A wins
95% of the resamples" is a one-sided statement applied in whichever direction the data
happened to favour. Its measured type-I error under a true null is **11.4%**, not 5%. This
package reports the win fraction because readers expect it, and decides with a cluster-aware
sign-flip permutation test.

**3. Deletion beats resampling as a way to explain fragility.** On the demo systems, the
leader's 3.6-point margin is reversed by deleting **2 of 56 examples** — which turn out to be
a single duplicate cluster. An exact combinatorial fact: no seed, no interval. The same
margin would need ~4500 examples to be detectable at 80% power. The benchmark is not
measuring what its leaderboard claims to show.

None of this depends on the demo's synthetic scores being realistic. Findings 1 and 3's
mechanisms are structural (:mod:`.simulate` produces the *scores*, not the block structure),
and the coverage simulation in ``tests/test_statistics.py`` uses known parameters throughout.
What the synthetic scores cannot support is any claim about a real system, and none is made.

**What this stage inherits, and cannot fix.** The cluster bootstrap corrects for exactly the
dependence the dedup ladder can *see*. ``ret-001``/``ret-002`` are a paraphrase pair no
lexical threshold reaches (``docs/DEDUPLICATION.md``), so they yield no block edge, sit in
different units, and are resampled as if independent. The design effect reported here is
therefore a **lower bound** on the true one, and the intervals remain slightly too narrow.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval_dataset_forge.dedup import DedupReport, find_duplicates
from eval_dataset_forge.schemas import BootstrapResult, DatasetExample
from eval_dataset_forge.splits.blocks import duplicate_blocks
from eval_dataset_forge.statistics import bootstrap, confidence, ranking_stability, scores, simulate
from eval_dataset_forge.statistics.bootstrap import (
    BCA,
    BLOCK,
    CI_METHODS,
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    EXAMPLE,
    PERCENTILE,
    RESAMPLE_UNITS,
    Units,
    bca_ci,
    bootstrap_ci,
    bootstrap_distribution,
    effective_n,
    intraclass_correlation,
    jackknife_means,
    kish_design_effect,
    measured_design_effect,
    percentile_ci,
    resample_indices,
    unit_totals,
)
from eval_dataset_forge.statistics.confidence import (
    DEFAULT_POWER,
    SystemComparison,
    compare_pair,
    compare_systems,
    holm,
    minimum_detectable_effect,
    monte_carlo_se,
    required_n_for_delta,
    sign_flip_p,
)
from eval_dataset_forge.statistics.ranking_stability import (
    MIN_TOP1_STABILITY,
    FlipSet,
    RankingStability,
    kendall_tau_b,
    min_examples_to_flip,
    min_units_to_flip,
    rank_frequencies,
)
from eval_dataset_forge.statistics.scores import (
    ScoreTable,
    align,
    load_scores,
    to_table,
    write_scores,
)
from eval_dataset_forge.statistics.simulate import (
    DEFAULT_ABILITIES,
    DEFAULT_SYSTEM_NAMES,
    simulate_scores,
)

# Fewer effective examples than this and a percentile bootstrap interval is decoration. A
# convention, like ``MIN_CALIBRATION_N``; the report always prints the number it measured.
MIN_EFFECTIVE_N = 30

# A design effect this far above 1.0 means the i.i.d. interval is materially too narrow.
# 1.05 corresponds to a ~2.5% widening — below the Monte Carlo noise of the measurement
# itself at the default resample count.
DESIGN_EFFECT_TOLERANCE = 1.05


# --------------------------------------------------------------------------- #
# Report types
# --------------------------------------------------------------------------- #


@dataclass
class SystemResult:
    """One system's score, its interval under both resampling assumptions, and the gap."""

    system: str
    accuracy: float
    n_examples: int
    independent: BootstrapResult
    clustered: BootstrapResult
    icc: float = float("nan")
    design_effect: float = float("nan")
    design_effect_kish: float = float("nan")
    effective_n: float = float("nan")

    @property
    def interval(self) -> BootstrapResult:
        """The clustered interval: the one a reader should quote."""
        return self.clustered

    @property
    def width_independent(self) -> float:
        return self.independent.ci_high - self.independent.ci_low

    @property
    def width_clustered(self) -> float:
        return self.clustered.ci_high - self.clustered.ci_low

    @property
    def widening(self) -> float:
        """How much wider the honest interval is. ``nan`` when the naive one has zero width."""
        if self.width_independent == 0.0:
            return float("nan")
        return self.width_clustered / self.width_independent

    def to_dict(self) -> dict:
        def number(value: float) -> float | None:
            return None if math.isnan(value) else round(value, 6)

        return {
            "system": self.system,
            "accuracy": round(self.accuracy, 6),
            "n_examples": self.n_examples,
            "independent": self.independent.model_dump(mode="json"),
            "clustered": self.clustered.model_dump(mode="json"),
            "icc": number(self.icc),
            "design_effect": number(self.design_effect),
            "design_effect_kish": number(self.design_effect_kish),
            "effective_n": number(self.effective_n),
            "ci_widening": number(self.widening),
        }


@dataclass
class StatsReport:
    """Everything the statistics stage measured, and whether the leaderboard survives it.

    ``ranking_resolved`` is the single bit downstream consumers should read, and it is the
    conjunction of three independent conditions: the corpus is large enough (in *effective*
    examples), the leader holds first place under resampling, and its margin over the
    runner-up survives multiplicity correction. Any one of them failing means the ordering
    is not supported, and each fails for a different, separately reported reason.
    """

    n_examples: int
    n_units: int
    resample_unit: str = BLOCK
    ci_method: str = PERCENTILE
    alpha: float = DEFAULT_ALPHA
    power: float = DEFAULT_POWER
    n_resamples: int = DEFAULT_RESAMPLES
    seed: int = DEFAULT_SEED
    scores_source: str = ""
    systems: list[SystemResult] = field(default_factory=list)
    comparisons: list[SystemComparison] = field(default_factory=list)
    stability: dict[str, RankingStability] = field(default_factory=dict)
    block_size_histogram: dict[int, int] = field(default_factory=dict)
    dedup_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    # -- derived views ----------------------------------------------------- #

    @property
    def blocked(self) -> RankingStability | None:
        return self.stability.get(BLOCK)

    @property
    def point_ranking(self) -> tuple[str, ...]:
        stability = self.blocked
        return stability.point_ranking if stability else ()

    @property
    def leader(self) -> str | None:
        ranking = self.point_ranking
        return ranking[0] if ranking else None

    @property
    def top_comparison(self) -> SystemComparison | None:
        """The leader against the runner-up — the only comparison a leaderboard implies."""
        ranking = self.point_ranking
        if len(ranking) < 2:
            return None
        top, second = ranking[0], ranking[1]
        for comparison in self.comparisons:
            if {comparison.system_a, comparison.system_b} == {top, second}:
                return comparison
        return None

    @property
    def design_effect(self) -> float:
        """The largest per-system design effect: the corpus is as dependent as its worst case."""
        values = [s.design_effect for s in self.systems if not math.isnan(s.design_effect)]
        return max(values) if values else float("nan")

    @property
    def effective_n(self) -> float:
        return effective_n(self.n_examples, self.design_effect)

    @property
    def underpowered(self) -> bool:
        n_eff = self.effective_n
        return math.isnan(n_eff) or n_eff < MIN_EFFECTIVE_N

    @property
    def ranking_resolved(self) -> bool:
        """May this leaderboard's ordering be reported as a finding?"""
        stability = self.blocked
        top = self.top_comparison
        if stability is None or top is None or self.underpowered:
            return False
        return stability.stable and top.significant

    def counts(self) -> dict[str, int]:
        return {
            "examples": self.n_examples,
            "units": self.n_units,
            "systems": len(self.systems),
            "comparisons": len(self.comparisons),
            "significant_comparisons": sum(1 for c in self.comparisons if c.significant),
            "warnings": len(self.warnings),
        }

    def to_dict(self) -> dict:
        def number(value: float) -> float | None:
            return None if math.isnan(value) else round(value, 6)

        return {
            "counts": self.counts(),
            "resample_unit": self.resample_unit,
            "ci_method": self.ci_method,
            "alpha": self.alpha,
            "power": self.power,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "scores_source": self.scores_source,
            "design_effect": number(self.design_effect),
            "effective_n": number(self.effective_n),
            "underpowered": self.underpowered,
            "ranking_resolved": self.ranking_resolved,
            "point_ranking": list(self.point_ranking),
            "block_size_histogram": {str(k): v for k, v in self.block_size_histogram.items()},
            "dedup_counts": self.dedup_counts,
            "systems": [system.to_dict() for system in self.systems],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "stability": {unit: report.to_dict() for unit, report in self.stability.items()},
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


def _warnings(report: StatsReport, dedup: DedupReport | None) -> tuple[str, ...]:
    """Everything a reader must know before quoting a number from this report."""
    messages: list[str] = []

    if len(report.systems) < 2:
        messages.append(
            "fewer than two systems were scored; confidence intervals are reported but no "
            "ranking claim can be made"
        )

    deff = report.design_effect
    if not math.isnan(deff) and deff > DESIGN_EFFECT_TOLERANCE:
        messages.append(
            f"duplicate clusters make the examples non-exchangeable (design effect "
            f"{deff:.2f}); an i.i.d. bootstrap over examples understates every interval by "
            f"~{(1 - 1 / math.sqrt(deff)) * 100:.0f}% and this corpus is worth about "
            f"{report.effective_n:.0f} independent examples, not {report.n_examples}"
        )

    if report.underpowered:
        n_eff = report.effective_n
        rendered = "undefined" if math.isnan(n_eff) else f"{n_eff:.0f}"
        messages.append(
            f"effective sample size is {rendered} (< {MIN_EFFECTIVE_N}); percentile bootstrap "
            f"intervals on this few examples have poor coverage regardless of the resampling "
            f"unit — see docs/STATISTICS.md"
        )

    for system in report.systems:
        if system.accuracy in (0.0, 1.0):
            messages.append(
                f"system {system.system!r} scored {system.accuracy:.0f} on every example; the "
                f"bootstrap distribution is degenerate and its interval is a point"
            )
        if "->" in system.clustered.method:
            messages.append(
                f"the BCa interval for {system.system!r} was not computable and fell back to "
                f"the percentile interval; the reported method says so"
            )

    stability = report.blocked
    if stability is not None and len(report.systems) >= 2 and not stability.stable:
        messages.append(
            f"the leader {stability.leader!r} holds first place in only "
            f"{stability.leader_stability:.1%} of resamples (< {MIN_TOP1_STABILITY:.0%}); the "
            f"ordering of this leaderboard is not supported by this dataset"
        )
    if stability is not None and stability.tie_fraction > 0.10:
        messages.append(
            f"{stability.tie_fraction:.0%} of resamples produce at least one exact tie between "
            f"systems; on {report.n_examples} examples a score can only take "
            f"{report.n_examples + 1} values, so ties are a property of the corpus size"
        )

    top = report.top_comparison
    if top is not None:
        if top.delta == 0.0:
            messages.append(
                f"{top.system_a!r} and {top.system_b!r} scored identically; the point ranking "
                f"puts {report.leader!r} first only because ties are broken by name"
            )
        if top.ci_contains_zero:
            messages.append(
                f"the {top.delta:+.4f} margin of {top.system_a!r} over {top.system_b!r} has a "
                f"95% CI of [{top.ci_low:+.4f}, {top.ci_high:+.4f}], which contains zero"
            )
        if top.underpowered and top.required_n:
            messages.append(
                f"the smallest gap this corpus can detect at {report.power:.0%} power is "
                f"{top.minimum_detectable_effect:.3f}; the observed gap between "
                f"{top.system_a!r} and {top.system_b!r} is {abs(top.delta):.3f}, which would "
                f"need about {top.required_n} examples"
            )
        if top.example_flip is not None and top.example_flip.n_removed > 0:
            flip = top.example_flip
            messages.append(
                f"deleting {flip.n_removed} of {report.n_examples} examples "
                f"({', '.join(flip.removed_ids[:5])}"
                f"{' …' if flip.n_removed > 5 else ''}) reverses {top.system_a!r}'s lead over "
                f"{top.system_b!r}"
            )

    for comparison in report.comparisons:
        if comparison.ci_disagrees_with_test:
            messages.append(
                f"{comparison.system_a!r} vs {comparison.system_b!r}: the bootstrap CI "
                f"[{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] excludes zero but the "
                f"sign-flip test does not reject (p_holm={comparison.p_holm:.3f}); the two use "
                f"different nulls and the test is the more conservative — it is what "
                f"'significant' reads"
            )
        if comparison.p_is_borderline:
            messages.append(
                f"{comparison.system_a!r} vs {comparison.system_b!r}: p={comparison.p_value:.4f} "
                f"lies within two Monte Carlo standard errors ({comparison.p_mc_se:.4f}) of "
                f"alpha={report.alpha} at B={report.n_resamples}; the verdict is set by the "
                f"seed, not by the data. Raise --resamples"
            )

    if len(report.comparisons) > 1:
        raw = sum(1 for c in report.comparisons if c.p_value <= report.alpha)
        corrected = sum(1 for c in report.comparisons if c.significant)
        if raw != corrected:
            messages.append(
                f"{raw} of {len(report.comparisons)} pairwise comparisons clear alpha="
                f"{report.alpha} uncorrected but only {corrected} survive Holm correction "
                f"across the {len(report.comparisons)} pairs a {len(report.systems)}-system "
                f"leaderboard implies"
            )

    if dedup is not None and dedup.near:
        count = len(dedup.near)
        messages.append(
            f"{count} duplicate cluster{'' if count == 1 else 's'} "
            f"{'was' if count == 1 else 'were'} found heuristically (near-duplicate stage, "
            f"method {dedup.near_method!r}); dependence the dedup ladder cannot see yields no "
            f"block, so the reported design effect is a lower bound and the clustered intervals "
            f"remain slightly too narrow"
        )
    return tuple(messages)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_units(
    examples: Sequence[DatasetExample], dedup: DedupReport | None
) -> tuple[Units, Units, tuple[str, ...]]:
    """``(per-example units, block units, block keys)``.

    Without a dedup report every block is a singleton and the two ``Units`` coincide — the
    design effect is then 1.0 by construction, which is the correct answer for a corpus
    whose dependence has not been measured *and* the reason the CLI always runs dedup first.
    """
    position_of = {example.example_id: i for i, example in enumerate(examples)}
    per_example = Units.per_example(len(examples))
    if dedup is None:
        return per_example, Units(members=per_example.members, unit=BLOCK), tuple(position_of)
    blocks = duplicate_blocks(examples, dedup)
    return per_example, Units.from_blocks(blocks, position_of), tuple(b.key for b in blocks)


def analyze_systems(
    examples: Sequence[DatasetExample],
    system_scores: Mapping[str, Sequence[float]] | ScoreTable,
    *,
    dedup: DedupReport | None = None,
    resample_unit: str = BLOCK,
    ci_method: str = PERCENTILE,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    scores_source: str = "",
) -> StatsReport:
    """Bootstrap intervals, paired comparisons, and ranking stability for a set of systems.

    ``system_scores`` maps a system name either to a per-example score *sequence* in the
    dataset's order, or to an ``example_id -> score`` mapping (what :func:`load_scores`
    returns), which is aligned strictly.

    Both resampling units are always computed. ``resample_unit`` selects which one the
    comparisons, the stability verdict, and ``ranking_resolved`` are based on; the other is
    still reported, because the *difference* between them is the finding.

    Deterministic given ``seed``: one resample matrix is drawn per unit type and shared by
    every system, which is what makes the comparisons paired and the ranking coherent.
    """
    if ci_method not in CI_METHODS:
        raise ValueError(f"unknown ci method {ci_method!r}; want one of {CI_METHODS}")
    if resample_unit not in RESAMPLE_UNITS:
        raise ValueError(f"unknown resample unit {resample_unit!r}; want one of {RESAMPLE_UNITS}")

    examples = list(examples)
    example_ids = [example.example_id for example in examples]
    per_example, blocked, block_keys = build_units(examples, dedup)

    if all(isinstance(value, Mapping) for value in system_scores.values()):
        names, matrix = align(system_scores, examples)
    else:
        names = list(system_scores)
        matrix = np.array([list(system_scores[name]) for name in names], dtype=float)
        if matrix.shape[1] != len(examples):
            raise ValueError(f"{matrix.shape[1]} scores for {len(examples)} examples")

    draws = {
        EXAMPLE: resample_indices(per_example.n_units, n_resamples=n_resamples, seed=seed),
        BLOCK: resample_indices(blocked.n_units, n_resamples=n_resamples, seed=seed),
    }
    units = {EXAMPLE: per_example, BLOCK: blocked}

    report = StatsReport(
        n_examples=len(examples),
        n_units=blocked.n_units,
        resample_unit=resample_unit,
        ci_method=ci_method,
        alpha=alpha,
        power=power,
        n_resamples=n_resamples,
        seed=seed,
        scores_source=scores_source,
        block_size_histogram=blocked.size_histogram(),
        dedup_counts=dedup.counts() if dedup else {},
    )

    for index, name in enumerate(names):
        values = matrix[index]
        independent = bootstrap_ci(
            values, per_example, system=name, method=ci_method, alpha=alpha,
            n_resamples=n_resamples, seed=seed, draw=draws[EXAMPLE],
        )
        clustered = bootstrap_ci(
            values, blocked, system=name, method=ci_method, alpha=alpha,
            n_resamples=n_resamples, seed=seed, draw=draws[BLOCK],
        )
        icc = intraclass_correlation(values, blocked)
        deff = measured_design_effect(
            bootstrap_distribution(values, blocked, draw=draws[BLOCK]),
            bootstrap_distribution(values, per_example, draw=draws[EXAMPLE]),
        )
        report.systems.append(
            SystemResult(
                system=name,
                accuracy=float(values.mean()),
                n_examples=len(examples),
                independent=independent,
                clustered=clustered,
                icc=icc,
                design_effect=deff,
                design_effect_kish=kish_design_effect(blocked, icc),
                effective_n=effective_n(len(examples), deff),
            )
        )

    chosen = units[resample_unit]
    unit_ids = block_keys if resample_unit == BLOCK else tuple(example_ids)

    # The design effect is a property of the *corpus*, so it enters the power arithmetic
    # (minimum detectable effect, required n) whichever unit was resampled. Zeroing it under
    # ``--resample-unit example`` would let the mode this milestone measured to be wrong also
    # report the rosier detectable effect. It does **not** enter the CI or the p-value: those
    # get their dependence correction from ``chosen`` directly, by resampling and sign-flipping
    # whole blocks, and multiplying by a design effect as well would double-count it.
    corpus_deff = report.design_effect
    if math.isnan(corpus_deff):
        corpus_deff = 1.0

    if len(names) >= 2:
        report.comparisons = compare_systems(
            {name: matrix[i] for i, name in enumerate(names)},
            chosen,
            example_ids=example_ids,
            unit_ids=unit_ids,
            alpha=alpha,
            power=power,
            ci_method=ci_method,
            design_effect=corpus_deff,
            n_resamples=n_resamples,
            seed=seed,
        )
    for unit_name, unit in units.items():
        report.stability[unit_name] = rank_frequencies(
            names, matrix, unit, n_resamples=n_resamples, seed=seed, draw=draws[unit_name]
        )

    report.warnings = _warnings(report, dedup)
    return report


def analyze_dataset(
    examples: Iterable[DatasetExample],
    *,
    score_table: ScoreTable | None = None,
    simulate_seed: int | None = None,
    scores_source: str = "",
    near_require_same_answer: bool = True,
    **kwargs,
) -> StatsReport:
    """Run dedup, build blocks, then :func:`analyze_systems`. The CLI's code path.

    Exactly one of ``score_table`` (real eval outputs) and ``simulate_seed`` (synthetic demo
    systems) must be given. The dedup ladder always runs: the block structure is what makes
    the intervals honest, and a caller who skips it gets a design effect of 1.0 whether or
    not the corpus deserves one.

    ``scores_source`` is recorded verbatim in the report. The simulated path overrides it, so
    a report built from synthetic scores always says so and can never be mistaken for one
    built from a real evaluation.
    """
    examples = list(examples)
    if (score_table is None) == (simulate_seed is None):
        raise ValueError("provide exactly one of score_table or simulate_seed")

    dedup = find_duplicates(examples, near_require_same_answer=near_require_same_answer)
    if score_table is not None:
        return analyze_systems(
            examples, score_table, dedup=dedup, scores_source=scores_source, **kwargs
        )

    _, blocked, _ = build_units(examples, dedup)
    simulated = simulate_scores(blocked, seed=simulate_seed)
    return analyze_systems(
        examples, simulated, dedup=dedup, scores_source=f"simulated(seed={simulate_seed})", **kwargs
    )


# --------------------------------------------------------------------------- #
# IO / rendering
# --------------------------------------------------------------------------- #


def write_report(report: StatsReport, out_dir: str | Path) -> dict[str, Path]:
    """Write ``<out_dir>/stats_report.json``. Byte-identical across runs at a fixed seed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stats_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return {"report": path}


def _number(value: float, spec: str = ".3f") -> str:
    return "n/a" if math.isnan(value) else format(value, spec)


def format_report(report: StatsReport, max_items: int = 20) -> str:
    """Render a human-readable summary of a :class:`StatsReport`."""
    histogram = " ".join(f"{size}x{count}" for size, count in report.block_size_histogram.items())
    lines = [
        f"statistics: {len(report.systems)} system(s) over {report.n_examples} examples in "
        f"{report.n_units} blocks ({histogram})",
        f"  resample unit       : {report.resample_unit} "
        f"(B={report.n_resamples}, seed {report.seed}, {report.ci_method} CI, "
        f"alpha={report.alpha})",
        f"  design effect       : {_number(report.design_effect, '.2f')} "
        f"-> {_number(report.effective_n, '.1f')} effective examples",
    ]
    if report.scores_source:
        lines.append(f"  scores              : {report.scores_source}")

    lines.append("")
    lines.append("per-system accuracy (i.i.d. interval vs cluster interval):")
    for system in report.systems:
        lines.append(
            f"  {system.system:<10} {system.accuracy:.4f}  "
            f"iid [{system.independent.ci_low:.4f}, {system.independent.ci_high:.4f}]  "
            f"cluster [{system.clustered.ci_low:.4f}, {system.clustered.ci_high:.4f}]  "
            f"ICC={_number(system.icc, '.2f')} deff={_number(system.design_effect, '.2f')} "
            f"({_number(system.widening, '.2f')}x wider)"
        )

    if report.comparisons:
        lines.append("")
        lines.append("paired comparisons (Holm-corrected; win fraction is NOT a p-value):")
        for comparison in report.comparisons:
            flip = comparison.example_flip
            # ``None`` (no deletion can flip it) and 0 (already tied) are different answers.
            flip_text = "flip@none" if flip is None else f"flip@{flip.n_removed}"
            lines.append(
                f"  {comparison.system_a:<8} > {comparison.system_b:<8} "
                f"delta={comparison.delta:+.4f} "
                f"CI[{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] "
                f"win={comparison.win_fraction:.3f} p={comparison.p_value:.3f} "
                f"p_holm={comparison.p_holm:.3f} "
                f"{'SIGNIFICANT' if comparison.significant else 'not significant'} {flip_text}"
            )
        top = report.top_comparison
        if top is not None:
            lines.append(
                f"  minimum detectable effect at {report.power:.0%} power: "
                f"{_number(top.minimum_detectable_effect)} "
                f"(observed top-2 gap {abs(top.delta):.4f}"
                + (f", would need ~{top.required_n} examples)" if top.required_n else ")")
            )

    for unit in (BLOCK, EXAMPLE):
        stability = report.stability.get(unit)
        if stability is None or len(stability.systems) < 2:
            continue
        frequencies = " ".join(
            f"{name}={value:.3f}" for name, value in stability.top1_frequency.items()
        )
        lines.append("")
        lines.append(f"ranking stability (resampling {unit}s):")
        lines.append(f"  point ranking       : {' > '.join(stability.point_ranking)}")
        lines.append(f"  top-1 frequency     : {frequencies}")
        lines.append(
            f"  exact order / tau   : {_number(stability.exact_order_frequency)} / "
            f"{_number(stability.mean_kendall_tau)} "
            f"(ties in {_number(stability.tie_fraction, '.1%')} of resamples)"
        )

    verdict = "RESOLVED" if report.ranking_resolved else "NOT RESOLVED"
    lines.append("")
    lines.append(f"ranking verdict: {verdict}")

    if report.warnings:
        lines.append(f"\nwarnings ({len(report.warnings)}):")
        for message in report.warnings[:max_items]:
            lines.append(f"  - {message}")
        if len(report.warnings) > max_items:
            lines.append(f"  … and {len(report.warnings) - max_items} more")
    return "\n".join(lines)


__all__ = [
    # orchestration
    "DESIGN_EFFECT_TOLERANCE",
    "MIN_EFFECTIVE_N",
    "StatsReport",
    "SystemResult",
    "analyze_dataset",
    "analyze_systems",
    "build_units",
    "format_report",
    "write_report",
    # submodules
    "bootstrap",
    "confidence",
    "ranking_stability",
    "scores",
    "simulate",
    # re-exports: bootstrap
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
    # re-exports: confidence
    "DEFAULT_POWER",
    "SystemComparison",
    "compare_pair",
    "compare_systems",
    "holm",
    "minimum_detectable_effect",
    "monte_carlo_se",
    "required_n_for_delta",
    "sign_flip_p",
    # re-exports: ranking stability
    "MIN_TOP1_STABILITY",
    "FlipSet",
    "RankingStability",
    "kendall_tau_b",
    "min_examples_to_flip",
    "min_units_to_flip",
    "rank_frequencies",
    # re-exports: scores / simulate
    "DEFAULT_ABILITIES",
    "DEFAULT_SYSTEM_NAMES",
    "ScoreTable",
    "align",
    "load_scores",
    "simulate_scores",
    "to_table",
    "write_scores",
]
