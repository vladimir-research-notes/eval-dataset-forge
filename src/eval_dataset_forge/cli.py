"""Command-line entry point for eval-dataset-forge (``evalforge``).

Only ``check-env``, ``version``, and ``info`` are implemented at scaffold stage.
The audit/build subcommands are declared here with a stable interface and print a
pointer to the milestone that implements them, so the CLI contract is visible
from day one and can be filled in without reshaping the parser.

Top-level imports are stdlib-only so ``evalforge --help`` and ``check-env`` work
before the optional scientific stack is installed. Heavy imports (pydantic,
numpy, pandas, scikit-learn) happen lazily inside command handlers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from eval_dataset_forge import SCHEMA_VERSION, __version__

# Subcommands that are declared now but implemented in a later milestone. Maps
# the command name to the milestone number that lands it (see docs/PLAN.md).
_PENDING = {
    "audit": 12,
    "dataset-card": 11,
    "smoke": 12,
}

# Runtime dependencies the framework relies on, grouped by hardness.
_CORE_DEPS = ["pydantic", "numpy", "yaml"]
_OPTIONAL_DEPS = ["pandas", "sklearn"]


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def cmd_info(_args: argparse.Namespace) -> int:
    print(f"eval-dataset-forge {__version__}")
    print(f"schema version   : {SCHEMA_VERSION}")
    print(f"python           : {sys.version.split()[0]}")
    print("purpose          : audit, version, and stress-test evaluation datasets")
    print("docs             : docs/PLAN.md, docs/METHODOLOGY.md")
    return 0


def cmd_check_env(_args: argparse.Namespace) -> int:
    """Report interpreter and dependency status without failing hard.

    Missing *optional* deps are informational; missing *core* deps are flagged
    with the exact install command for this PEP-668 environment.
    """
    print(f"python           : {sys.version.split()[0]}")
    ok = True
    print("core dependencies:")
    for dep in _CORE_DEPS:
        present = _installed(dep)
        ok = ok and present
        print(f"  [{'ok' if present else '--'}] {dep}")
    print("optional dependencies (graceful fallback if absent):")
    for dep in _OPTIONAL_DEPS:
        print(f"  [{'ok' if _installed(dep) else '--'}] {dep}")
    if not ok:
        print()
        print("Missing core dependencies. On this PEP-668 environment install with:")
        print("  python3 -m pip install --user --break-system-packages "
              "pydantic numpy pyyaml")
        print("Optional accelerators:")
        print("  python3 -m pip install --user --break-system-packages "
              "pandas scikit-learn")
        return 1
    print()
    print("Environment OK: core dependencies present, CPU-only, no network required.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a dataset file: schema, ID uniqueness, quality, PII/secrets.

    Exit code 0 iff the report has no *errors* (schema errors, duplicate ids, and
    secret-class findings — plus all PII when ``--fail-on-pii`` is set). Warnings
    never change the exit code. The validation package is imported lazily so the
    stdlib-only commands keep working without the scientific stack.
    """
    from eval_dataset_forge.validation import (
        format_report,
        validate_dataset,
        write_redacted,
    )

    report = validate_dataset(args.dataset, fail_on_pii=args.fail_on_pii)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    if args.redact_out:
        n = write_redacted(args.dataset, args.redact_out)
        print(f"\nwrote {n} redacted example(s) to {args.redact_out}")
    return 0 if report.ok else 1


def _parse_ratios(text: str) -> dict[str, float]:
    """Parse ``0.7,0.1,0.2`` or ``train=0.8,test=0.2`` into a ratio mapping."""
    parts = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if any("=" in part for part in parts):
        pairs = [part.split("=", 1) for part in parts]
        return {name.strip(): float(value) for name, value in pairs}
    names = ["train", "dev", "test"]
    if len(parts) != len(names):
        raise argparse.ArgumentTypeError(
            "--ratios wants three positional values (train,dev,test) or named ones "
            "(train=0.8,test=0.2)"
        )
    return dict(zip(names, (float(part) for part in parts), strict=True))


def cmd_build_splits(args: argparse.Namespace) -> int:
    """Construct train/dev/test splits and verify them against the leakage checks.

    Writes ``<out>/splits/{train,dev,test}.jsonl`` and ``<out>/split_report.json``. Exit
    code is 0 unless ``--fail-on-leakage`` is set and the built splits still contain a
    detectable cross-split duplicate cluster — the check a CI job would gate a release
    on. The splits package is imported lazily so the stdlib-only commands keep working
    without the scientific stack.
    """
    from eval_dataset_forge.datasets import loader
    from eval_dataset_forge.dedup import find_duplicates
    from eval_dataset_forge.splits import build_splits, format_report, write_splits

    examples = loader.load_examples(args.dataset)
    dedup_report = find_duplicates(
        examples, near_require_same_answer=not args.ignore_answer_gate
    )

    quarantine_ids: list[str] = []
    if args.quarantine_blocking:
        from eval_dataset_forge.labels import audit_labels

        quarantine_ids = audit_labels(examples).blocking_example_ids

    report = build_splits(
        examples,
        strategy=args.strategy,
        ratios=_parse_ratios(args.ratios),
        seed=args.seed,
        stratify_by=[f.strip() for f in args.stratify_by.split(",") if f.strip()],
        dedup_report=dedup_report,
        quarantine_ids=quarantine_ids,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))

    written = write_splits(examples, report, args.out)
    print(f"\nwrote {len(written) - 1} split file(s) and a report under {args.out}")
    return 1 if args.fail_on_leakage and report.residual_overlap else 0


def _parse_weights(text: str) -> dict[str, float]:
    """Parse ``input_length=0.5,reasoning_steps=0.5`` into a weight mapping."""
    pairs = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not all("=" in pair for pair in pairs):
        raise argparse.ArgumentTypeError(
            "--weights wants name=value pairs, e.g. input_length=0.5,reasoning_steps=0.5"
        )
    try:
        return {name.strip(): float(value) for name, value in (p.split("=", 1) for p in pairs)}
    except ValueError as error:  # non-numeric value
        raise argparse.ArgumentTypeError(f"bad --weights value: {error}") from None


def cmd_difficulty(args: argparse.Namespace) -> int:
    """Score, bucket, and calibrate a dataset's difficulty; report the distribution.

    Exit code is 0 unless ``--fail-on-uncalibrated`` is set and the heuristic's ranking
    cannot be distinguished from a random ranking of the declared ``difficulty_tags`` — the
    check a CI job would gate on before letting anything downstream treat the heuristic
    bucket as a difficulty label. On the committed fixtures that gate **fails**, by design:
    see docs/DIFFICULTY.md. The difficulty package is imported lazily so the stdlib-only
    commands keep working without the scientific stack.
    """
    from eval_dataset_forge.datasets import loader
    from eval_dataset_forge.difficulty import format_report, score_dataset, write_report

    examples = loader.load_examples(args.dataset)
    report = score_dataset(
        examples,
        weights=_parse_weights(args.weights) if args.weights else None,
        bucketing=args.bucketing,
        alpha=args.alpha,
        n_resamples=args.resamples,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))

    if args.out:
        written = write_report(report, args.out)
        print(f"\nwrote {len(written)} difficulty artifact(s) under {args.out}")
    return 1 if args.fail_on_uncalibrated and not report.trustworthy else 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Bootstrap CIs, paired comparisons, and ranking stability for a set of systems.

    Takes one or more dataset files (pooled in order, so the fixture corpus can be analysed
    as one benchmark) plus either real per-example scores (``--scores``) or deterministic
    synthetic ones (``--simulate``, which describes no real system and says so in the
    report). Exit code is 0 unless ``--fail-on-unstable-ranking`` is set and the leaderboard's
    ordering is not supported by the data — the check a CI job would gate a leaderboard claim
    on. On the committed fixtures with simulated systems that gate **fails**, by design: see
    docs/STATISTICS.md. The statistics package is imported lazily so the stdlib-only commands
    keep working without the scientific stack.
    """
    from eval_dataset_forge.datasets import loader
    from eval_dataset_forge.statistics import (
        analyze_dataset,
        format_report,
        load_scores,
        write_report,
    )

    examples = [ex for path in args.dataset for ex in loader.load_examples(path)]
    report = analyze_dataset(
        examples,
        score_table=load_scores(args.scores) if args.scores else None,
        simulate_seed=args.seed if args.simulate else None,
        scores_source=args.scores or "",
        near_require_same_answer=not args.ignore_answer_gate,
        resample_unit=args.resample_unit,
        ci_method=args.ci_method,
        alpha=args.alpha,
        power=args.power,
        n_resamples=args.resamples,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))

    if args.out:
        written = write_report(report, args.out)
        print(f"\nwrote {len(written)} statistics artifact(s) under {args.out}")
    return 1 if args.fail_on_unstable_ranking and not report.ranking_resolved else 0


def _release_command(args: argparse.Namespace) -> str:
    """Reconstruct the canonical command for ``REPRODUCE.md``.

    Rebuilt from the parsed arguments rather than echoed from ``sys.argv``: an argv can
    carry shell quoting, an absolute path from someone's laptop, or flags in an order that
    would make two identical releases render different bytes.

    Only the flags that *determine the data* are recorded. Three are deliberately absent:

    ``--created-at``
        ``REPRODUCE.md`` is a hashed artifact and the clock is not identity.
    ``--version`` / ``--out``
        Under ``--version auto`` neither is known until the delta is computed;
        ``render_reproduce`` appends both once the version resolves.
    ``--previous``
        It decides the version *number*, which the resolved ``--version`` already states
        literally. Recording it would put a build-time comparison into a hashed artifact,
        so re-running a build with ``--previous`` pointing at its own output would report a
        spurious ``patch`` against itself — the data is identical and only the invocation
        differed.
    """
    parts = ["PYTHONPATH=src python3 -m eval_dataset_forge release"]
    parts += [f"--dataset {dataset}" for dataset in args.dataset]
    parts.append(f"--dataset-id {args.dataset_id}")
    parts.append(f"--strategy {args.strategy}")
    parts.append(f"--seed {args.seed}")
    parts.append(f"--ratios {args.ratios}")
    if args.stratify_by:
        parts.append(f"--stratify-by {args.stratify_by}")
    if args.drop:
        parts.append(f"--drop {args.drop}")
    if args.quarantine_blocking:
        parts.append("--quarantine-blocking")
    if args.ignore_answer_gate:
        parts.append("--ignore-answer-gate")
    if args.license_notes:
        parts.append(f"--license-notes {args.license_notes!r}")
    return " \\\n  ".join(parts)


def cmd_release(args: argparse.Namespace) -> int:
    """Build a versioned, hashed, audited release directory from one or more datasets.

    Exit code is 0 unless ``--fail-on-leakage`` is set and the built splits still contain a
    cross-split duplicate cluster, or the requested ``--version`` is below the bump the
    content requires. The version is *derived* from a comparison against ``--previous``:
    anything visible to ``dev``/``test`` forces a major bump, because a score measured on
    the old release does not carry over. The datasets package's release module is imported
    lazily so the stdlib-only commands keep working without the scientific stack.
    """
    from eval_dataset_forge.datasets.release import build_release, format_report, write_release

    try:
        report = build_release(
            args.dataset,
            dataset_id=args.dataset_id,
            version=args.version,
            created_at=args.created_at or _utc_now(),
            command=_release_command(args),
            strategy=args.strategy,
            ratios=_parse_ratios(args.ratios),
            seed=args.seed,
            stratify_by=[f.strip() for f in args.stratify_by.split(",") if f.strip()],
            drop=[d.strip() for d in args.drop.split(",") if d.strip()] if args.drop else (),
            quarantine_blocking=args.quarantine_blocking,
            near_require_same_answer=not args.ignore_answer_gate,
            license_notes=args.license_notes,
            previous=args.previous,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))

    # The default output directory names the *derived* version, which `--version auto` only
    # learns during the build.
    out = args.out or f"datasets/releases/{args.dataset_id}_v{report.manifest.version}"
    written = write_release(report, out)
    print(f"\nwrote {len(written)} artifact(s) to {out}")
    return 1 if args.fail_on_leakage and report.split_report.residual_overlap else 0


def cmd_verify_release(args: argparse.Namespace) -> int:
    """Check a release directory against its own manifest. Exit 1 if anything disagrees."""
    from eval_dataset_forge.datasets.manifest import format_verify, verify_release

    report = verify_release(args.release)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_verify(report))
    return 0 if report.ok else 1


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _cmd_pending(name: str):
    def handler(_args: argparse.Namespace) -> int:
        milestone = _PENDING[name]
        print(f"`evalforge {name}` is not implemented yet.")
        print(f"It lands in Milestone {milestone} — see docs/PLAN.md.")
        return 2

    return handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalforge",
        description="Audit, version, and stress-test evaluation datasets (CPU-only, offline).",
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("version", help="print the version").set_defaults(func=cmd_version)
    sub.add_parser("info", help="print project metadata").set_defaults(func=cmd_info)
    sub.add_parser("check-env", help="report interpreter and dependency status").set_defaults(
        func=cmd_check_env
    )

    p_validate = sub.add_parser(
        "validate",
        help="validate a dataset: schema, ID uniqueness, quality, PII/secrets",
    )
    p_validate.add_argument("--dataset", required=True, help="path to examples.jsonl")
    p_validate.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )
    p_validate.add_argument(
        "--fail-on-pii",
        action="store_true",
        help="treat all PII findings (not just secrets) as errors",
    )
    p_validate.add_argument(
        "--redact-out",
        metavar="PATH",
        help="also write a redacted copy of the valid examples to PATH",
    )
    p_validate.set_defaults(func=cmd_validate)

    p_audit = sub.add_parser("audit", help="run the full audit pipeline over a dataset")
    p_audit.add_argument("--dataset", required=True, help="path to examples.jsonl")
    p_audit.add_argument("--out", required=True, help="output directory for the audit report")
    p_audit.set_defaults(func=_cmd_pending("audit"))

    p_splits = sub.add_parser("build-splits", help="construct train/dev/test splits")
    p_splits.add_argument("--dataset", required=True, help="path to examples.jsonl")
    p_splits.add_argument(
        "--strategy",
        default="stratified",
        choices=["random", "stratified", "source-grouped", "adversarial", "temporal"],
        help="split construction strategy",
    )
    p_splits.add_argument("--out", required=True, help="output release directory")
    p_splits.add_argument(
        "--seed", type=int, default=0, help="seed for the shuffled strategies (default: 0)"
    )
    p_splits.add_argument(
        "--ratios",
        default="0.7,0.1,0.2",
        help="split ratios as train,dev,test or name=value pairs (must sum to 1.0)",
    )
    p_splits.add_argument(
        "--stratify-by",
        default="task_family",
        help="comma-separated facets to stratify on, from {task_family, difficulty}; "
        "stratified and adversarial strategies only. Combining both multiplies the "
        "strata and measures worse on both facets at small N — see docs/SPLITS.md",
    )
    p_splits.add_argument(
        "--quarantine-blocking",
        action="store_true",
        help="keep label-audit blocking examples out of dev/test (adversarial only)",
    )
    p_splits.add_argument(
        "--ignore-answer-gate",
        action="store_true",
        help="let the near-duplicate stage pair examples with differing gold answers "
        "(needed for retrieval-style data whose answers are document ids)",
    )
    p_splits.add_argument(
        "--fail-on-leakage",
        action="store_true",
        help="exit 1 if the built splits still contain a cross-split duplicate cluster",
    )
    p_splits.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )
    p_splits.set_defaults(func=cmd_build_splits)

    p_difficulty = sub.add_parser(
        "difficulty",
        help="score, bucket, and calibrate example difficulty; report the distribution",
    )
    p_difficulty.add_argument("--dataset", required=True, help="path to examples.jsonl")
    p_difficulty.add_argument(
        "--out", help="directory for difficulty_report.json and difficulty_scores.jsonl"
    )
    p_difficulty.add_argument(
        "--bucketing",
        default="quantile",
        choices=["quantile", "fixed", "marginal"],
        help="how the two bucket cut points are chosen (default: quantile)",
    )
    p_difficulty.add_argument(
        "--weights",
        metavar="NAME=VALUE,…",
        help="override the documented signal weights; must be non-negative and sum to 1.0",
    )
    p_difficulty.add_argument(
        "--seed", type=int, default=0, help="seed for the permutation tests (default: 0)"
    )
    p_difficulty.add_argument(
        "--resamples",
        type=int,
        default=10_000,
        help="permutation resamples for the calibration p-values (default: 10000)",
    )
    p_difficulty.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="significance level for the calibration verdict (default: 0.05)",
    )
    p_difficulty.add_argument(
        "--fail-on-uncalibrated",
        action="store_true",
        help="exit 1 when the heuristic ranking is indistinguishable from a random one",
    )
    p_difficulty.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )
    p_difficulty.set_defaults(func=cmd_difficulty)

    p_stats = sub.add_parser(
        "stats",
        help="bootstrap CIs, paired comparisons, and ranking stability for scored systems",
    )
    p_stats.add_argument(
        "--dataset",
        required=True,
        action="append",
        metavar="PATH",
        help="path to examples.jsonl; repeat to pool several datasets into one benchmark",
    )
    source = p_stats.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scores",
        metavar="PATH",
        help="JSONL of per-example system scores "
        "({example_id, system, score} or {example_id, scores:{…}})",
    )
    source.add_argument(
        "--simulate",
        action="store_true",
        help="generate deterministic synthetic systems instead (offline demo; the scores "
        "describe no real system and no conclusion about one may be drawn from them)",
    )
    p_stats.add_argument("--out", help="directory for stats_report.json")
    p_stats.add_argument(
        "--resample-unit",
        default="block",
        choices=["block", "example"],
        help="what one bootstrap draw pulls: a duplicate block (default, corrects for "
        "within-cluster dependence) or a single example (the i.i.d. assumption)",
    )
    p_stats.add_argument(
        "--ci-method",
        default="percentile",
        choices=["percentile", "bca"],
        help="interval method; bca is offered but not the default (see docs/STATISTICS.md)",
    )
    p_stats.add_argument(
        "--seed", type=int, default=0, help="seed for the resampling and for --simulate"
    )
    p_stats.add_argument(
        "--resamples", type=int, default=2_000, help="bootstrap / permutation resamples"
    )
    p_stats.add_argument(
        "--alpha", type=float, default=0.05, help="two-sided significance level (default: 0.05)"
    )
    p_stats.add_argument(
        "--power", type=float, default=0.8, help="power used by the small-N arithmetic"
    )
    p_stats.add_argument(
        "--ignore-answer-gate",
        action="store_true",
        help="let the near-duplicate stage pair examples with differing gold answers",
    )
    p_stats.add_argument(
        "--fail-on-unstable-ranking",
        action="store_true",
        help="exit 1 when the leaderboard's ordering is not supported by the data",
    )
    p_stats.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )
    p_stats.set_defaults(func=cmd_stats)

    p_release = sub.add_parser(
        "release", help="build a versioned, hashed, audited release directory"
    )
    p_release.add_argument(
        "--dataset",
        required=True,
        action="append",
        metavar="PATH",
        help="path to examples.jsonl; repeat to pool several files into one release",
    )
    p_release.add_argument("--dataset-id", required=True, help="stable id for the dataset")
    p_release.add_argument(
        "--version",
        default="0.1.0",
        help="MAJOR.MINOR.PATCH, or 'auto' to derive it from --previous. An explicit version "
        "below the bump the content requires is an error",
    )
    p_release.add_argument(
        "--previous",
        metavar="DIR",
        help="a previous release directory; the version bump is derived by comparing against it",
    )
    p_release.add_argument(
        "--out", help="output directory (default: datasets/releases/<id>_v<version>)"
    )
    p_release.add_argument(
        "--strategy",
        default="adversarial",
        choices=["random", "stratified", "source-grouped", "adversarial", "temporal"],
        help="split construction strategy (default: adversarial — it is the only one that "
        "keeps duplicate clusters out of the evaluation splits)",
    )
    p_release.add_argument("--seed", type=int, default=0, help="split seed (default: 0)")
    p_release.add_argument(
        "--ratios", default="0.7,0.1,0.2", help="split ratios as train,dev,test or name=value pairs"
    )
    p_release.add_argument(
        "--stratify-by", default="task_family", help="comma-separated facets to stratify on"
    )
    p_release.add_argument(
        "--drop",
        metavar="TARGET,…",
        help="delete redundant examples before building, from {exact, normalized, near, "
        "blocking-labels}. Nothing is dropped by default; every removal is recorded in "
        "manifest.excluded with the id of the member that was kept",
    )
    p_release.add_argument(
        "--quarantine-blocking",
        action="store_true",
        help="keep label-audit blocking examples out of dev/test without deleting them",
    )
    p_release.add_argument(
        "--ignore-answer-gate",
        action="store_true",
        help="let the near-duplicate stage pair examples with differing gold answers",
    )
    p_release.add_argument(
        "--created-at",
        metavar="ISO8601",
        help="provenance timestamp (default: now). Enters no hash and neither identity; "
        "pass it to make manifest.json byte-identical across rebuilds",
    )
    p_release.add_argument("--license-notes", default="", help="license text for the manifest")
    p_release.add_argument(
        "--fail-on-leakage",
        action="store_true",
        help="exit 1 if the built splits still contain a cross-split duplicate cluster",
    )
    p_release.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )
    p_release.set_defaults(func=cmd_release)

    p_verify = sub.add_parser(
        "verify-release", help="check a release directory against its own manifest"
    )
    p_verify.add_argument("--release", required=True, help="path to a release directory")
    p_verify.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of text"
    )
    p_verify.set_defaults(func=cmd_verify_release)

    p_card = sub.add_parser("dataset-card", help="generate a dataset card for a release")
    p_card.add_argument("--release", required=True, help="path to a release directory")
    p_card.add_argument("--out", required=True, help="output markdown path")
    p_card.set_defaults(func=_cmd_pending("dataset-card"))

    p_smoke = sub.add_parser("smoke", help="run a full smoke audit over the fixture datasets")
    p_smoke.add_argument("--out", required=True, help="output markdown report path")
    p_smoke.set_defaults(func=_cmd_pending("smoke"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
