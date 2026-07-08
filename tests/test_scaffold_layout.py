"""Structural tests: the documented repository layout and docs must exist.

These are cheap guards that keep the scaffold honest as milestones fill it in.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_files_exist() -> None:
    for rel in ("README.md", "CLAUDE.md", "pyproject.toml", ".gitignore", "LICENSE"):
        assert (ROOT / rel).is_file(), f"missing {rel}"


def test_package_modules_exist() -> None:
    pkg = ROOT / "src" / "eval_dataset_forge"
    for rel in ("__init__.py", "__main__.py", "cli.py"):
        assert (pkg / rel).is_file(), f"missing package file {rel}"


def test_required_docs_exist() -> None:
    docs = ROOT / "docs"
    required = [
        "PLAN.md",
        "METHODOLOGY.md",
        "DATASET_SCHEMA.md",
        "DEDUPLICATION.md",
        "LEAKAGE.md",
        "LABEL_AUDIT.md",
        "SPLITS.md",
        "DIFFICULTY.md",
        "STATISTICS.md",
        "RESULTS.md",
        "LIMITATIONS.md",
        "SAFETY.md",
        "REPRODUCIBILITY.md",
        "PAPER_PATH.md",
    ]
    for name in required:
        assert (docs / name).is_file(), f"missing docs/{name}"


def test_website_pages_exist() -> None:
    site = ROOT / "site"
    pages = [
        "index.html",
        "schema.html",
        "audit-pipeline.html",
        "deduplication.html",
        "leakage.html",
        "label-audit.html",
        "splits.html",
        "difficulty.html",
        "statistics.html",
        "dataset-card.html",
        "results.html",
        "limitations.html",
        "roadmap.html",
        "milestones.html",
        "style.css",
    ]
    for name in pages:
        assert (site / name).is_file(), f"missing site/{name}"


def test_runner_and_env_scripts_exist() -> None:
    scripts = ROOT / "scripts"
    for name in ("run_all_milestones_with_claude.sh", "00_check_environment.py", "serve_site.sh"):
        assert (scripts / name).is_file(), f"missing scripts/{name}"


def test_fixture_and_result_dirs_exist() -> None:
    for rel in ("datasets/fixtures", "datasets/releases", "results/audits",
                "results/reports", "results/processed"):
        assert (ROOT / rel).is_dir(), f"missing directory {rel}"
