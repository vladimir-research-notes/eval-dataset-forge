"""Verify the repository's "do not commit" contract against the tracked tree (Milestone 15).

``CLAUDE.md`` and ``docs/SAFETY.md`` promise that certain things never enter git: secrets and keys,
``.env`` files, virtualenvs, caches, model weights, large datasets, and cached API responses. Two
mechanisms already enforce most of that by *policy* — ``.gitignore`` keeps them out of ``git add``,
and ``.claude/settings.json`` denies the tools that would fetch them. This module adds the missing
*audit*: given the set of files git actually tracks, does the contract still hold? It is the
filesystem counterpart to :mod:`~eval_dataset_forge.docs_audit`, which audits the prose.

Three checks, each objective and **allowlist-free** — the deliberate design constraint, for the
reason Milestone 14 gives about :attr:`RetiredClaim.cited_in`: an exemption list is a thing that
grows quietly until it exempts the violation you cared about.

* ``ignored-but-tracked`` — a file ``.gitignore`` would ignore, yet git tracks. This is how a
  cache, a virtualenv, or a secret path slips in: it is force-added past the ignore rule. The
  check re-asks git and trusts nothing else.
* ``forbidden-path`` — a tracked file whose name or suffix is one the contract forbids outright
  (``*.pem``, ``*.key``, ``id_rsa``, model-weight formats, ``.env``). These must never be tracked
  regardless of what ``.gitignore`` currently says, because ``.gitignore`` can be edited and this
  list is the contract.
* ``oversized`` — a tracked file above :data:`MAX_TRACKED_BYTES`. Large committed blobs are the
  signature of a dataset or a weight file that should live outside the repo.

**What this does not do, and why.** It does *not* grep file contents for secrets. That belongs to
:mod:`~eval_dataset_forge.validation.pii`, which is exercised on fixtures that *intentionally*
contain fake AWS keys and private-key headers (``tests/test_validation.py``). A repo-wide secret
grep would flag those very fixtures — the detector's own test inputs — and the only way to silence
it would be the allowlist this module refuses to keep. So content scanning stays in the validation
layer where the fake secrets are the point; hygiene checks *which files exist*, not what they say.
A real secret pasted into a source file is therefore out of this scan's reach, and that limitation
is stated rather than papered over.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Suffixes the contract forbids in tracked files: secrets, keys, and model-weight formats.
FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".pem",
    ".key",
    ".env",
    ".pt",
    ".bin",
    ".onnx",
    ".gguf",
    ".safetensors",
    ".h5",
    ".ckpt",
)

#: Exact filenames the contract forbids regardless of suffix.
FORBIDDEN_NAMES: frozenset[str] = frozenset({"id_rsa", ".env", ".pypirc", "credentials"})

#: A tracked file above this is treated as a committed dataset or weight blob. The largest file the
#: repository tracks today is a ~50 KiB test module, so this leaves generous headroom for prose and
#: code while catching anything of dataset scale.
MAX_TRACKED_BYTES: int = 512 * 1024


@dataclass(frozen=True)
class Finding:
    """One tracked path that violates the do-not-commit contract."""

    check: str
    path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.path}: [{self.check}] {self.message}"


def _is_forbidden(path: str) -> bool:
    name = pathlib.PurePosixPath(path).name
    return name in FORBIDDEN_NAMES or name.startswith(".env.") or path.endswith(FORBIDDEN_SUFFIXES)


def check_ignored_but_tracked(ignored: Sequence[str]) -> list[Finding]:
    """A tracked file that ``.gitignore`` claims to ignore was force-added past the rule."""
    return [
        Finding(
            "ignored-but-tracked",
            path,
            "tracked although .gitignore matches it; force-added past the ignore rule",
        )
        for path in ignored
    ]


def check_forbidden_paths(paths: Sequence[str]) -> list[Finding]:
    """A tracked file whose name or suffix the contract forbids outright."""
    return [
        Finding("forbidden-path", path, "a secret/key/weight path must never be tracked")
        for path in paths
        if _is_forbidden(path)
    ]


def check_oversized(sizes: Mapping[str, int], cap: int = MAX_TRACKED_BYTES) -> list[Finding]:
    """A tracked file large enough to be a committed dataset or model weight."""
    return [
        Finding("oversized", path, f"{size} bytes exceeds the {cap}-byte tracked-file cap")
        for path, size in sizes.items()
        if size > cap
    ]


@dataclass
class HygieneReport:
    """What the scan found, and how much it looked at."""

    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stats": dict(self.stats),
            "findings": [
                {"check": f.check, "path": f.path, "message": f.message} for f in self.findings
            ],
        }


def _git(root: pathlib.Path, *args: str, stdin: str | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout


def tracked_files(root: pathlib.Path) -> list[str]:
    """Every path git tracks, as ``git ls-files`` reports it."""
    return [line for line in _git(root, "ls-files").splitlines() if line]


def ignored_among(root: pathlib.Path, paths: Sequence[str]) -> list[str]:
    """The subset of ``paths`` that ``.gitignore`` matches, per ``git check-ignore``."""
    if not paths:
        return []
    out = _git(root, "check-ignore", "--stdin", stdin="\n".join(paths))
    return [line for line in out.splitlines() if line]


def scan_repository(root: pathlib.Path) -> HygieneReport:
    """Run every check against the tree git tracks at ``root``."""
    paths = tracked_files(root)
    sizes = {path: (root / path).stat().st_size for path in paths if (root / path).is_file()}
    ignored = ignored_among(root, paths)

    findings = [
        *check_ignored_but_tracked(ignored),
        *check_forbidden_paths(paths),
        *check_oversized(sizes),
    ]
    stats = {
        "tracked_files": len(paths),
        "tracked_bytes": sum(sizes.values()),
        "largest_bytes": max(sizes.values(), default=0),
    }
    return HygieneReport(findings=findings, stats=stats)


def format_report(report: HygieneReport) -> str:
    """A short, greppable summary; findings grouped by check."""
    lines = ["hygiene scan"]
    for key in sorted(report.stats):
        lines.append(f"  {key:16s}: {report.stats[key]}")
    if report.ok:
        lines.append("\nno violations: the tracked tree honours the do-not-commit contract")
        return "\n".join(lines)
    grouped: dict[str, list[Finding]] = {}
    for finding in report.findings:
        grouped.setdefault(finding.check, []).append(finding)
    lines.append(f"\n{len(report.findings)} violation(s) across {len(grouped)} check(s):")
    for check in sorted(grouped):
        lines.append(f"\n[{check}]")
        for finding in grouped[check]:
            lines.append(f"  {finding.path}: {finding.message}")
    return "\n".join(lines)


__all__ = [
    "FORBIDDEN_NAMES",
    "FORBIDDEN_SUFFIXES",
    "MAX_TRACKED_BYTES",
    "Finding",
    "HygieneReport",
    "check_forbidden_paths",
    "check_ignored_but_tracked",
    "check_oversized",
    "format_report",
    "ignored_among",
    "scan_repository",
    "tracked_files",
]
