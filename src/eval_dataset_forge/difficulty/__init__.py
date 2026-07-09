"""Difficulty calibration (Milestone 8).

Audit dimension #6: *does this dataset contain a spread of difficulty, and does the
number we call "difficulty" deserve the name?*

The stage has two halves, and the second is the point. The first
(:mod:`.heuristics`, :mod:`.buckets`) computes a transparent, model-free difficulty proxy
from five surface signals and buckets it — the deliverable ``docs/PLAN.md`` §10 asks for.
The second (:mod:`.calibration`) asks whether that proxy's ranking can be distinguished
from a random one, using the hand-assigned ``difficulty_tags`` as the only available
target and a permutation test as the only honest instrument.

**On the committed corpus it cannot.** rho = +0.10, p = 0.50 over 51 labelled examples.
Sweeping the entire weight simplex does not rescue it: 94% of 2000 Dirichlet draws land
inside |rho| < 0.2, and the best draw — chosen *in sample*, so an upper bound — reaches
only +0.29. Meanwhile ``task_family`` explains 43% of the score's variance and 1% of the
declared difficulty's. The score is a task-family detector wearing a difficulty label:
the single highest-scoring example in the corpus is ``agent-002``, a paraphrase of
FizzBuzz that its author declared **easy**.

So :class:`DifficultyReport` reports its buckets **and refuses to certify them**:
``trustworthy`` is False, and the warnings say what that forbids. The one downstream
consumer that could have been fooled — difficulty stratification in
:mod:`eval_dataset_forge.splits.stratified` — reads the *declared* tags, which Milestone 7
chose on the argument that a heuristic must not define the test set it is later evaluated
on. This milestone turns that argument into a measurement.

Nothing here is deleted or corrected. A bucket is attached to every example, including
the ones whose bucket the calibration says not to believe, because the release builder
(Milestone 10) and the dataset card (Milestone 11) must be able to show a reader exactly
what the audit saw.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eval_dataset_forge.difficulty import buckets, calibration, heuristics
from eval_dataset_forge.difficulty.buckets import (
    BUCKET_ORDER,
    BUCKETING_METHODS,
    DEFAULT_QUANTILES,
    DIFFICULTY_ORDER,
    FIXED_CUTOFFS,
    UNLABELLED,
    assign_buckets,
    bucket_distribution,
    bucket_of,
    cutpoints_for,
    declared_bucket,
    declared_distribution,
    declared_ordinal,
    distribution_by_split,
    marginal_cutpoints,
    quantile_cutpoints,
    scale_counts,
)
from eval_dataset_forge.difficulty.calibration import (
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DRAWS,
    MIN_BUCKET_N,
    MIN_CALIBRATION_N,
    CalibrationReport,
    SignalDiagnostic,
    calibrate,
    eta_squared,
    permutation_p,
    rankdata,
    required_n,
    spearman,
)
from eval_dataset_forge.difficulty.heuristics import (
    DEFAULT_WEIGHTS,
    METHOD,
    SIGNAL_NAMES,
    SIGNALS,
    Signal,
    normalized_signals,
    raw_score,
    raw_signals,
    resolve_weights,
    weighted_components,
)
from eval_dataset_forge.schemas import DatasetExample, DifficultyBucket, DifficultyScore


@dataclass
class DifficultyReport:
    """Per-example scores, the distribution they induce, and whether to believe it.

    ``distribution`` is the shape
    :attr:`~eval_dataset_forge.schemas.AuditResult.difficulty_distribution` expects.
    ``trustworthy`` forwards :attr:`CalibrationReport.trustworthy`; when no example
    declares a difficulty there is nothing to calibrate against and it is False — absence
    of evidence is reported as absence of evidence, not as a pass.
    """

    n_examples: int
    method: str
    bucketing: str
    weights: dict[str, float] = field(default_factory=dict)
    cutpoints: tuple[float, float] = (0.0, 0.0)
    scores: list[DifficultyScore] = field(default_factory=list)
    declared: dict[str, int] = field(default_factory=dict)
    declared_by_id: dict[str, str] = field(default_factory=dict)
    by_split: dict[str, dict[str, int]] = field(default_factory=dict)
    empty_answer_example_ids: tuple[str, ...] = ()
    calibration: CalibrationReport | None = None
    warnings: tuple[str, ...] = ()

    # -- derived views ----------------------------------------------------- #

    @property
    def distribution(self) -> dict[str, int]:
        """``bucket → count`` over the whole dataset. All three buckets always present."""
        return bucket_distribution(score.bucket for score in self.scores)

    @property
    def trustworthy(self) -> bool:
        """May these buckets be used as difficulty labels? See :mod:`.calibration`."""
        return bool(self.calibration and self.calibration.trustworthy)

    @property
    def score_of(self) -> dict[str, float]:
        return {score.example_id: score.score for score in self.scores}

    @property
    def bucket_of_example(self) -> dict[str, DifficultyBucket]:
        return {score.example_id: score.bucket for score in self.scores}

    @property
    def disagreements(self) -> list[str]:
        """Labelled examples whose heuristic bucket differs from the declared one."""
        by_id = self.bucket_of_example
        return sorted(
            example_id
            for example_id, declared_name in self.declared_by_id.items()
            if str(by_id[example_id]) != declared_name
        )

    def counts(self) -> dict[str, int]:
        return {
            "examples": self.n_examples,
            **self.distribution,
            "labelled": sum(
                count for name, count in self.declared.items() if name != UNLABELLED
            ),
            "unlabelled": self.declared.get(UNLABELLED, 0),
            "disagreements": len(self.disagreements),
            "empty_gold_answers": len(self.empty_answer_example_ids),
        }

    def to_dict(self) -> dict:
        """A plain-dict view suitable for ``json.dumps`` and for ``AuditResult``."""
        return {
            "counts": self.counts(),
            "method": self.method,
            "bucketing": self.bucketing,
            "weights": self.weights,
            "cutpoints": list(self.cutpoints),
            "distribution": self.distribution,
            "declared_distribution": self.declared,
            "by_split": self.by_split,
            "disagreements": self.disagreements,
            "empty_gold_answers": list(self.empty_answer_example_ids),
            "trustworthy": self.trustworthy,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "scores": [score.model_dump(mode="json") for score in self.scores],
            "warnings": list(self.warnings),
        }


def _bucketing_warnings(report: DifficultyReport) -> list[str]:
    """Defects of the bucketing itself, independent of whether the score means anything."""
    messages: list[str] = []
    distribution = report.distribution
    for name, count in distribution.items():
        if count == 0:
            messages.append(
                f"the {name!r} bucket is empty under {report.bucketing!r} bucketing "
                f"(cutpoints {report.cutpoints[0]:.3f}/{report.cutpoints[1]:.3f}); "
                f"the dataset cannot measure it"
            )
    easy = str(DifficultyBucket.EASY)
    by_id = report.bucket_of_example
    sunk = [
        example_id
        for example_id in report.empty_answer_example_ids
        if str(by_id.get(example_id)) == easy
    ]
    if sunk:
        messages.append(
            f"{len(sunk)} example(s) with an empty gold answer scored into the {easy!r} "
            f"bucket ({', '.join(sunk)}); a missing label reads as low output complexity, "
            f"so the label defect makes the example look easy"
        )
    return messages


def score_dataset(
    examples: Iterable[DatasetExample],
    *,
    weights: Mapping[str, float] | None = None,
    bucketing: str = "quantile",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    fixed: tuple[float, float] = FIXED_CUTOFFS,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    weight_draws: int = DEFAULT_WEIGHT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> DifficultyReport:
    """Score, bucket, and calibrate a dataset.

    Deterministic and offline: the same examples, weights, bucketing, and seed always
    produce the same report. ``bucketing="marginal"`` reproduces the declared bucket
    *counts*, which is what :mod:`.calibration` uses internally to separate a bad ranking
    from a bad marginal; it is available here for the same reason.

    The score of one example never depends on the others — only the quantile cut points
    do. So adding an example can move a bucket boundary but never a score, and a score
    computed on a subset is the score it would have had in the whole.
    """
    examples = list(examples)
    resolved = resolve_weights(weights)
    raw_scores = [raw_score(example, resolved) for example in examples]

    # The declared marginal is measured over the labelled examples; ``marginal`` bucketing
    # is applied to all of them. Apportion so the target sums to the corpus size instead of
    # letting the undeclared surplus fall into the top bucket.
    target_counts = (
        scale_counts(
            {
                name: count
                for name, count in declared_distribution(examples).items()
                if name != UNLABELLED
            },
            len(examples),
        )
        if bucketing == "marginal"
        else None
    )
    cuts = cutpoints_for(
        raw_scores, bucketing, quantiles=quantiles, fixed=fixed, target_counts=target_counts
    )
    assigned = [bucket_of(score, cuts) for score in raw_scores]

    # Neither the score nor its components are rounded. ``raw_score`` *is*
    # ``sum(weighted_components(...).values())`` over the same dict in the same order, so
    # storing both unrounded keeps ``sum(components) == score`` true bit-for-bit. Rounding
    # either one to six places breaks the equality for examples whose five contributions
    # round in the same direction, and the explainability claim with it.
    method = f"{METHOD}+{bucketing}"
    scores = [
        DifficultyScore(
            example_id=example.example_id,
            score=score,
            bucket=bucket,
            components=weighted_components(example, resolved),
            method=method,
        )
        for example, score, bucket in zip(examples, raw_scores, assigned, strict=True)
    ]

    report = DifficultyReport(
        n_examples=len(examples),
        method=method,
        bucketing=bucketing,
        weights=dict(resolved),
        cutpoints=cuts,
        scores=scores,
        declared=declared_distribution(examples),
        by_split=distribution_by_split(examples, assigned),
        empty_answer_example_ids=tuple(
            example.example_id
            for example in examples
            if heuristics.raw_output_complexity(example) == 0.0
        ),
        calibration=calibrate(
            examples,
            weights=resolved,
            alpha=alpha,
            n_resamples=n_resamples,
            weight_draws=weight_draws,
            seed=seed,
        ),
        declared_by_id={
            example.example_id: str(declared_bucket(example))
            for example in examples
            if declared_bucket(example) is not None
        },
    )
    report.warnings = (
        *_bucketing_warnings(report),
        *(report.calibration.warnings if report.calibration else ()),
    )
    return report


def write_report(report: DifficultyReport, out_dir: str | Path) -> dict[str, Path]:
    """Write ``<out_dir>/difficulty_report.json`` and ``difficulty_scores.jsonl``.

    Byte-identical across runs: scores keep their input order and are serialized
    field-by-field in schema order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "difficulty_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    scores_path = out_dir / "difficulty_scores.jsonl"
    scores_path.write_text(
        "".join(
            json.dumps(score.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for score in report.scores
        ),
        encoding="utf-8",
    )
    return {"report": report_path, "scores": scores_path}


def _number(value: float, spec: str = "+.3f") -> str:
    """Render a float, or ``n/a`` when it is undefined.

    A single-family dataset has no between-family variation, so the family baseline is a
    correlation against a constant and is genuinely undefined. Printing ``+nan`` there
    would read as a computed value; ``n/a`` reads as the absence of one.
    """
    return "n/a" if math.isnan(value) else format(value, spec)


def format_report(report: DifficultyReport, max_items: int = 20) -> str:
    """Render a human-readable summary of a :class:`DifficultyReport`."""
    counts = report.counts()
    distribution = " ".join(f"{name}={count}" for name, count in report.distribution.items())
    declared = " ".join(f"{name}={count}" for name, count in report.declared.items())
    lines = [
        f"difficulty: {counts['examples']} examples via {report.method} "
        f"(cutpoints {report.cutpoints[0]:.3f}/{report.cutpoints[1]:.3f})",
        f"  heuristic buckets   : {distribution}",
        f"  declared buckets    : {declared}",
        f"  disagreements       : {counts['disagreements']} of {counts['labelled']} labelled",
        f"  empty gold answers  : {counts['empty_gold_answers']}",
    ]
    for split, table in report.by_split.items():
        rendered = " ".join(f"{name}={count}" for name, count in table.items())
        lines.append(f"  split {split:<14}: {rendered}")

    result = report.calibration
    if result and result.n_labelled >= 3:
        verdict = "TRUSTWORTHY" if result.trustworthy else "NOT CALIBRATED"
        n_families = len(result.within_family_rho)
        lines += [
            "",
            f"calibration vs declared difficulty_tags ({result.n_labelled} labelled): {verdict}",
            f"  spearman rho        : {_number(result.rho)} "
            f"(permutation p={_number(result.p_value, '.3f')}, {result.n_resamples} resamples, "
            f"seed {result.seed})",
            f"  within-family rho   : {_number(result.within_family_mean_rho)} "
            f"(size-weighted mean over {n_families} "
            f"famil{'y' if n_families == 1 else 'ies'})",
            f"  family-only baseline: {_number(result.family_baseline_rho)} (oracle)",
            f"  eta^2 by family     : score {result.score_eta_squared_family:.3f} vs "
            f"declared {result.declared_eta_squared_family:.3f}",
            f"  marginal-matched acc: {_number(result.marginal_accuracy, '.3f')} "
            f"(majority {_number(result.majority_baseline, '.3f')}, "
            f"random ranking {_number(result.random_ranking_accuracy, '.3f')}, "
            f"p={_number(result.marginal_accuracy_p, '.3f')})",
        ]
        if result.required_n_for_power:
            lines.append(
                f"  n for 80% power     : ~{result.required_n_for_power} labelled examples"
            )
        lines.append("  per-signal rho vs declared:")
        for diagnostic in result.per_signal:
            if diagnostic.inert:
                detail = "INERT (zero variance)"
            else:
                detail = (
                    f"rho={_number(diagnostic.rho)} "
                    f"p={_number(diagnostic.p_value, '.3f')} "
                    f"bonferroni={_number(diagnostic.p_bonferroni, '.3f')}"
                )
            lines.append(
                f"    {diagnostic.name:<20} w={diagnostic.weight:.2f} {detail}"
                + ("  [anti-correlated]" if diagnostic.anticorrelated else "")
            )
        sweep = result.weight_sensitivity
        if sweep:
            lines.append(
                f"  weight simplex sweep: {int(sweep['draws'])} draws, "
                f"rho in [{sweep['min']:+.3f}, {sweep['max']:+.3f}], "
                f"median {sweep['median']:+.3f}, "
                f"{sweep['fraction_below_0.2']:.0%} below |0.2|"
            )

    if report.warnings:
        lines.append(f"\nwarnings ({len(report.warnings)}):")
        for message in report.warnings[:max_items]:
            lines.append(f"  - {message}")
        if len(report.warnings) > max_items:
            lines.append(f"  … and {len(report.warnings) - max_items} more")
    return "\n".join(lines)


__all__ = [
    # orchestration
    "DifficultyReport",
    "score_dataset",
    "write_report",
    "format_report",
    # submodules
    "buckets",
    "calibration",
    "heuristics",
    # re-exports: heuristics
    "DEFAULT_WEIGHTS",
    "METHOD",
    "SIGNALS",
    "SIGNAL_NAMES",
    "Signal",
    "normalized_signals",
    "raw_score",
    "raw_signals",
    "resolve_weights",
    "weighted_components",
    # re-exports: buckets
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
    # re-exports: calibration
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
