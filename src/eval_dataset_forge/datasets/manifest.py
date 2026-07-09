"""Release manifests and third-party verification (Milestone 10).

:func:`build_manifest` assembles the inventory record; :func:`verify_release` checks a
release directory *without rebuilding it*, which is the property that makes the manifest
worth writing. A reader who receives a release and re-runs the pipeline has only proved
that the pipeline is deterministic. A reader who runs ``verify-release`` has proved that
the bytes on disk are the ones the manifest describes, using nothing but the release.

**What the check is, and is not.** Every hash here is an *integrity* check: it detects a
truncated download, a stray editor save, a split file that no longer matches
``examples.jsonl``. None of it is an *authenticity* check. ``manifest.json`` cannot hash
itself, so anyone who can rewrite the data can rewrite the digests that describe it. The
manifest detects accident; it does not detect an adversary. Signing is out of scope and out
of the dependency budget, and saying so here is cheaper than implying otherwise by silence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eval_dataset_forge import SCHEMA_VERSION, __version__
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.datasets.versioning import content_id, eval_id
from eval_dataset_forge.datasets.writer import sha256_bytes, sha256_file
from eval_dataset_forge.schemas import (
    AuditResult,
    CardMeta,
    DatasetExample,
    DatasetManifest,
    SplitSpec,
)

EXAMPLES_FILE = "examples.jsonl"
MANIFEST_FILE = "manifest.json"
AUDIT_FILE = "audit.json"
REPRODUCE_FILE = "REPRODUCE.md"
SPLITS_DIR = "splits"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_manifest(
    *,
    dataset_id: str,
    version: str,
    created_at: str,
    examples: Sequence[DatasetExample],
    artifacts: Mapping[str, bytes],
    source_files: Sequence[str] = (),
    audit: AuditResult | None = None,
    split_spec: SplitSpec | None = None,
    excluded: Sequence[dict] = (),
    dependence: Mapping[str, object] | None = None,
    source_content_id: str = "",
    known_limitations: Sequence[str] = (),
    license_notes: str = "",
    card_meta: CardMeta | None = None,
) -> DatasetManifest:
    """Assemble the manifest for a release whose artifact bytes are already rendered.

    ``artifacts`` maps release-relative path → bytes, and is hashed in memory. The manifest
    therefore describes the bytes that are about to be written rather than the bytes a
    later ``stat`` happens to find, and ``build`` needs no filesystem at all.

    ``manifest.json`` itself is necessarily absent from ``artifacts``: a file cannot
    contain its own digest. ``dataset_card.md`` is *not* absent, and that is why the release
    builder calls this function twice — once to obtain a manifest the card can be rendered
    from, and once more with the card's bytes in ``artifacts`` so its digest is recorded.
    The card never reads ``hashes``, so the two manifests agree on everything the card used.
    """
    hashes = {name: sha256_bytes(data) for name, data in sorted(artifacts.items())}

    splits: dict[str, object] = {}
    for name, data in sorted(artifacts.items()):
        if not name.startswith(f"{SPLITS_DIR}/"):
            continue
        split_name = Path(name).stem
        splits[split_name] = {
            "count": sum(1 for line in data.decode("utf-8").splitlines() if line.strip()),
            "path": name,
            "sha256": hashes[name],
        }

    return DatasetManifest(
        dataset_id=dataset_id,
        version=version,
        created_at=created_at,
        source_files=list(source_files),
        example_count=len(examples),
        schema_version=SCHEMA_VERSION,
        hashes=hashes,
        splits=splits,
        audit_results=audit.summary() if audit else {},
        known_limitations=list(known_limitations),
        license_notes=license_notes,
        content_id=content_id(examples),
        eval_id=eval_id(examples),
        source_content_id=source_content_id or content_id(examples),
        tool_version=__version__,
        split_spec=split_spec,
        excluded=[dict(item) for item in excluded],
        dependence=dict(dependence or {}),
        card_meta=card_meta or CardMeta(),
    )


def load_manifest(path: str | Path) -> DatasetManifest:
    """Load and validate a ``manifest.json``."""
    return DatasetManifest(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_audit(path: str | Path) -> AuditResult:
    """Load and validate an ``audit.json``."""
    return AuditResult(**json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #


@dataclass
class VerifyReport:
    """The outcome of checking a release directory against its own manifest.

    ``checks`` records every check that ran, including the ones that passed, because a
    verifier that only prints failures cannot be distinguished from one that did nothing.
    """

    release_dir: str
    checks: dict[str, bool] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    manifest: DatasetManifest | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def _record(self, name: str, passed: bool, problem: str = "") -> bool:
        self.checks[name] = passed
        if not passed and problem:
            self.problems.append(problem)
        return passed

    def to_dict(self) -> dict:
        return {
            "release_dir": self.release_dir,
            "ok": self.ok,
            "checks": dict(self.checks),
            "problems": list(self.problems),
            "dataset_id": self.manifest.dataset_id if self.manifest else None,
            "version": self.manifest.version if self.manifest else None,
            "content_id": self.manifest.content_id if self.manifest else None,
            "eval_id": self.manifest.eval_id if self.manifest else None,
        }


def verify_release(release_dir: str | Path) -> VerifyReport:
    """Check a release against its manifest using only what the release directory holds.

    Runs, in order: the manifest parses; every hashed artifact exists; every file hash
    matches; the example count matches; ``content_id`` and ``eval_id` recompute; the split
    files partition ``examples.jsonl`` exactly; each split file's rows carry the split they
    are filed under and agree with ``examples.jsonl``; ``audit.json`` parses and names this
    release.

    A failure never raises — it lands in ``problems`` — so a caller can report every defect
    in one pass instead of the first one.
    """
    release_dir = Path(release_dir)
    report = VerifyReport(release_dir=str(release_dir))

    manifest_path = release_dir / MANIFEST_FILE
    try:
        manifest = load_manifest(manifest_path)
    except Exception as error:  # noqa: BLE001 — any parse/validation failure is one problem
        report._record("manifest_parses", False, f"{MANIFEST_FILE}: {error}")
        return report
    report.manifest = manifest
    report._record("manifest_parses", True)

    missing = [name for name in manifest.hashes if not (release_dir / name).is_file()]
    report._record(
        "artifacts_present", not missing, f"missing artifact(s): {', '.join(sorted(missing))}"
    )

    mismatched = [
        name
        for name, digest in sorted(manifest.hashes.items())
        if (release_dir / name).is_file() and sha256_file(release_dir / name) != digest
    ]
    report._record(
        "hashes_match",
        not mismatched,
        f"file hash mismatch: {', '.join(mismatched)} (content differs from the manifest)",
    )

    examples_path = release_dir / EXAMPLES_FILE
    if not examples_path.is_file():
        report._record("examples_load", False, f"missing {EXAMPLES_FILE}")
        return report
    try:
        examples = loader.load_examples(examples_path)
    except Exception as error:  # noqa: BLE001
        report._record("examples_load", False, f"{EXAMPLES_FILE}: {error}")
        return report
    report._record("examples_load", True)

    report._record(
        "example_count_matches",
        manifest.example_count == len(examples),
        f"manifest.example_count {manifest.example_count} != {len(examples)} rows in "
        f"{EXAMPLES_FILE}",
    )

    recomputed_content = content_id(examples)
    report._record(
        "content_id_matches",
        manifest.content_id == recomputed_content,
        f"content_id mismatch: manifest says {manifest.content_id}, examples hash to "
        f"{recomputed_content}",
    )
    recomputed_eval = eval_id(examples)
    report._record(
        "eval_id_matches",
        manifest.eval_id == recomputed_eval,
        f"eval_id mismatch: manifest says {manifest.eval_id}, examples hash to "
        f"{recomputed_eval}",
    )

    _verify_splits(report, release_dir, manifest, examples)
    _verify_audit(report, release_dir, manifest)
    return report


def _verify_splits(
    report: VerifyReport,
    release_dir: Path,
    manifest: DatasetManifest,
    examples: Sequence[DatasetExample],
) -> None:
    assigned = {example.example_id: str(example.split) for example in examples}
    seen: dict[str, str] = {}
    problems_before = len(report.problems)

    for name, info in sorted(manifest.splits.items()):
        path = release_dir / str(info["path"])
        if not path.is_file():
            report.problems.append(f"missing split file {info['path']}")
            continue
        members = loader.load_examples(path)
        if len(members) != int(info["count"]):
            report.problems.append(
                f"splits.{name}.count is {info['count']} but {info['path']} holds "
                f"{len(members)} rows"
            )
        for member in members:
            if str(member.split) != name:
                report.problems.append(
                    f"{info['path']} holds {member.example_id} labelled "
                    f"split={member.split!s}"
                )
            if member.example_id in seen:
                report.problems.append(
                    f"{member.example_id} appears in both {seen[member.example_id]} and {name}"
                )
            seen[member.example_id] = name
            if assigned.get(member.example_id) != name:
                report.problems.append(
                    f"{member.example_id} is {assigned.get(member.example_id)!r} in "
                    f"{EXAMPLES_FILE} and {name!r} in the split files"
                )

    unfiled = sorted(set(assigned) - set(seen))
    if unfiled:
        report.problems.append(
            f"{len(unfiled)} example(s) in {EXAMPLES_FILE} appear in no split file: "
            f"{', '.join(unfiled[:5])}"
        )
    report.checks["splits_partition_examples"] = len(report.problems) == problems_before


def _verify_audit(report: VerifyReport, release_dir: Path, manifest: DatasetManifest) -> None:
    audit_path = release_dir / AUDIT_FILE
    if AUDIT_FILE not in manifest.hashes and not audit_path.is_file():
        report.checks["audit_parses"] = True  # a release may ship without one
        return
    try:
        audit = load_audit(audit_path)
    except Exception as error:  # noqa: BLE001
        report._record("audit_parses", False, f"{AUDIT_FILE}: {error}")
        return
    report._record("audit_parses", True)
    report._record(
        "audit_names_this_release",
        audit.dataset_id == manifest.dataset_id and audit.version == manifest.version,
        f"{AUDIT_FILE} describes {audit.dataset_id}@{audit.version}, manifest describes "
        f"{manifest.dataset_id}@{manifest.version}",
    )


def format_verify(report: VerifyReport) -> str:
    """Render a :class:`VerifyReport` for a terminal."""
    manifest = report.manifest
    header = (
        f"verify: {report.release_dir}"
        if manifest is None
        else f"verify: {manifest.dataset_id}@{manifest.version} ({report.release_dir})"
    )
    lines = [header]
    if manifest is not None:
        sizes = ", ".join(
            "{}={}".format(name, info["count"]) for name, info in sorted(manifest.splits.items())
        )
        lines += [
            f"  content_id : {manifest.content_id}",
            f"  eval_id    : {manifest.eval_id}",
            f"  examples   : {manifest.example_count} ({sizes})",
        ]
    for name, passed in report.checks.items():
        lines.append(f"  [{'ok' if passed else 'FAIL'}] {name}")
    if report.problems:
        lines.append(f"\nproblems ({len(report.problems)}):")
        lines += [f"  - {problem}" for problem in report.problems]
    else:
        lines.append(
            "\nrelease is internally consistent (integrity, not authenticity: manifest.json "
            "is not self-hashed)"
        )
    return "\n".join(lines)


__all__ = [
    "AUDIT_FILE",
    "EXAMPLES_FILE",
    "MANIFEST_FILE",
    "REPRODUCE_FILE",
    "SPLITS_DIR",
    "VerifyReport",
    "build_manifest",
    "format_verify",
    "load_audit",
    "load_manifest",
    "verify_release",
]
