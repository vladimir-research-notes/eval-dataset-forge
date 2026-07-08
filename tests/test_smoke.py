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


def test_pending_commands_exit_2(capsys) -> None:
    rc = main(["validate", "--dataset", "datasets/fixtures/qa_toy/examples.jsonl"])
    assert rc == 2
    assert "Milestone" in capsys.readouterr().out
