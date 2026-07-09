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
