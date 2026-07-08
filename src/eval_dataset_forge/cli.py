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
    "build-splits": 7,
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
    p_splits.set_defaults(func=_cmd_pending("build-splits"))

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
