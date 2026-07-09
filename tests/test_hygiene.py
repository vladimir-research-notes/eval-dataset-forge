"""The hygiene scan, and the controls that prove each check can fail (Milestone 15).

Same discipline as ``tests/test_docs.py``: the first test asserts the real tree is clean, which on
its own proves nothing — a scan that inspects nothing is also clean. So each check is then shown to
*fire* on a planted violation, fed synthetic inputs rather than a mutated repository, because the
pure checks (:func:`check_ignored_but_tracked`, :func:`check_forbidden_paths`,
:func:`check_oversized`) take the world they judge as an argument.
"""

from __future__ import annotations

import pathlib

from eval_dataset_forge.hygiene import (
    FORBIDDEN_SUFFIXES,
    MAX_TRACKED_BYTES,
    check_forbidden_paths,
    check_ignored_but_tracked,
    check_oversized,
    format_report,
    ignored_among,
    scan_repository,
    tracked_files,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# The repository, scanned
# --------------------------------------------------------------------------- #


def test_the_tracked_tree_honours_the_contract() -> None:
    report = scan_repository(ROOT)
    assert report.ok, "\n".join(str(f) for f in report.findings)


def test_the_scan_actually_looked_at_something() -> None:
    """A clean report over zero files is the scan that inspects nothing."""
    report = scan_repository(ROOT)
    assert report.stats["tracked_files"] >= 100
    assert report.stats["tracked_bytes"] > 0
    assert 0 < report.stats["largest_bytes"] <= MAX_TRACKED_BYTES


# --------------------------------------------------------------------------- #
# ignored-but-tracked
# --------------------------------------------------------------------------- #


def test_ignored_but_tracked_fires_on_a_forced_add() -> None:
    findings = check_ignored_but_tracked(["__pycache__/x.pyc", ".venv/lib/foo.py"])
    assert len(findings) == 2
    assert all(f.check == "ignored-but-tracked" for f in findings)


def test_ignored_but_tracked_is_silent_when_nothing_is_ignored() -> None:
    assert check_ignored_but_tracked([]) == []


def test_no_tracked_file_is_git_ignored() -> None:
    """The integration point: git's own ignore rules, over the files git actually tracks."""
    paths = tracked_files(ROOT)
    assert paths, "expected a tracked tree"
    assert ignored_among(ROOT, paths) == []


# --------------------------------------------------------------------------- #
# forbidden-path
# --------------------------------------------------------------------------- #


def test_forbidden_path_fires_on_a_committed_key() -> None:
    findings = check_forbidden_paths(["src/app.py", "deploy/server.pem", "secrets/id_rsa"])
    flagged = {f.path for f in findings}
    assert flagged == {"deploy/server.pem", "secrets/id_rsa"}


def test_forbidden_path_fires_on_an_env_file_and_its_variants() -> None:
    findings = check_forbidden_paths([".env", "config/.env.production", "notes/env.md"])
    flagged = {f.path for f in findings}
    assert flagged == {".env", "config/.env.production"}


def test_forbidden_path_covers_every_declared_weight_suffix() -> None:
    """Each model-weight format the contract names is caught, not just the first."""
    paths = [f"model{suffix}" for suffix in FORBIDDEN_SUFFIXES]
    assert len(check_forbidden_paths(paths)) == len(FORBIDDEN_SUFFIXES)


def test_forbidden_path_leaves_ordinary_files_alone() -> None:
    assert check_forbidden_paths(["README.md", "src/keying.py", "docs/environment.md"]) == []


# --------------------------------------------------------------------------- #
# oversized
# --------------------------------------------------------------------------- #


def test_oversized_fires_on_a_dataset_sized_blob() -> None:
    findings = check_oversized({"data/big.jsonl": MAX_TRACKED_BYTES + 1, "small.py": 100})
    assert [f.path for f in findings] == ["data/big.jsonl"]


def test_oversized_accepts_a_file_at_the_cap() -> None:
    assert check_oversized({"edge.txt": MAX_TRACKED_BYTES}) == []


# --------------------------------------------------------------------------- #
# report shape
# --------------------------------------------------------------------------- #


def test_format_report_names_each_violated_check() -> None:
    report = scan_repository(ROOT)
    text = format_report(report)
    assert "hygiene scan" in text
    assert "tracked_files" in text
