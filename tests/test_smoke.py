"""CPU-only smoke tests for the scaffold CLI. No network, no paid API."""

from __future__ import annotations

import eval_dataset_forge
from eval_dataset_forge.cli import build_parser, main


def test_version_string() -> None:
    assert isinstance(eval_dataset_forge.__version__, str)
    assert eval_dataset_forge.__version__


def test_schema_version_declared() -> None:
    assert eval_dataset_forge.SCHEMA_VERSION.count(".") == 2


def test_parser_builds_and_lists_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("version", "info", "check-env", "validate", "audit",
                    "build-splits", "dataset-card", "smoke"):
        assert command in help_text


def test_check_env_runs(capsys) -> None:
    # check-env must never raise and must return 0/1 depending on deps.
    rc = main(["check-env"])
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "python" in out


def test_no_args_prints_help(capsys) -> None:
    rc = main([])
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_no_subcommand_is_a_stub() -> None:
    # Milestone 12 implemented the last two (`audit`, `smoke`). Every subcommand declared in the
    # PLAN's CLI contract now has a real handler, and the parser has never changed shape.
    choices = build_parser()._subparsers._group_actions[0].choices
    for name, subparser in choices.items():
        handler = subparser.get_default("func")
        assert handler is not None, f"{name} has no handler"
        assert handler.__module__.startswith("eval_dataset_forge"), name
        assert not handler.__name__.startswith("_cmd_pending"), f"{name} is still a stub"


def test_validate_command_implemented(capsys) -> None:
    # `validate` landed in Milestone 3: a clean fixture validates and exits 0.
    rc = main(["validate", "--dataset", "datasets/fixtures/qa_toy/examples.jsonl"])
    assert rc == 0
    assert "verdict" in capsys.readouterr().out
