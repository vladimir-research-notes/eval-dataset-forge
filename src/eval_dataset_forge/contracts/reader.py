"""The read-only consumption surface: :class:`ReleaseHandle` (Milestone 13).

A downstream project that opens ``manifest.json`` with :func:`json.load` has taken a
dependency on every key in it, including the ones it does not read, because it cannot
tell a reader which keys those are. Milestone 12 renamed nothing and still moved this
surface: ``split_statistics`` gained ``source`` and ``unassigned``, and two
recommendation strings were reworded. Under a strict parser that is a break; under a
permissive one it is invisible until the day a key it *did* read disappears.

So the reads are declared. :data:`MANIFEST_FIELDS`, :data:`AUDIT_FIELDS`,
:data:`DEPENDENCE_FIELDS`, and :data:`SPLIT_STAT_FIELDS` are the entire set of names
this package takes from a release, and ``test_contracts.py`` asserts each one exists in
the committed demo release. The compatibility rule that follows is checkable rather
than aspirational:

    Adding a key is a ``patch``. Removing or retyping one of the declared keys is a
    ``major``, and :data:`CONTRACT_VERSION` moves with it.

**What a handle is not.** It is not a cache. Every property reads from the release
directory that was opened, and no method recomputes an audit stage — with one exception,
:meth:`ReleaseHandle.examples_with_retrieval_docs`, which exists precisely because
``audit.json`` cannot answer the question it asks. See :mod:`.requirements`.
"""

from __future__ import annotations

import functools
from pathlib import Path

from eval_dataset_forge.datasets import loader
from eval_dataset_forge.datasets.manifest import (
    AUDIT_FILE,
    EXAMPLES_FILE,
    MANIFEST_FILE,
    SPLITS_DIR,
    VerifyReport,
    load_audit,
    load_manifest,
    verify_release,
)
from eval_dataset_forge.leakage.retrieval_overlap import retrieval_docs
from eval_dataset_forge.schemas import (
    AuditResult,
    DatasetExample,
    DatasetManifest,
    LabelAuditFinding,
    LabelIssue,
    LeakageFinding,
    LeakageKind,
)

# --------------------------------------------------------------------------- #
# The declared read set — the contract surface itself
# --------------------------------------------------------------------------- #

#: Top-level ``manifest.json`` keys this package reads.
MANIFEST_FIELDS = (
    "schema_version",
    "dataset_id",
    "version",
    "content_id",
    "eval_id",
    "example_count",
    "splits",
    "dependence",
    "tool_version",
)

#: Top-level ``audit.json`` keys this package reads.
AUDIT_FIELDS = (
    "exact_duplicates",
    "near_duplicates",
    "leakage_flags",
    "label_warnings",
    "pii_flags",
    "split_statistics",
)

#: Keys read from ``manifest.dependence``.
DEPENDENCE_FIELDS = ("block_count", "blocks_by_split", "near_method")

#: Keys read from ``audit.split_statistics``.
SPLIT_STAT_FIELDS = ("source", "counts")

#: Keys read from ``audit.split_statistics["counts"]``.
SPLIT_COUNT_FIELDS = ("eval_duplicates", "residual_overlap")


class ContractViolation(RuntimeError):
    """A release does not satisfy what a consumer is entitled to assume.

    Raised on open when the release fails its own integrity check, and by
    :meth:`ReleaseHandle.require_same_corpus` when a verdict computed on one corpus is
    about to be applied to another. It is deliberately not a ``ValueError``: a consumer
    catching ``ValueError`` around a parse should not also swallow this.
    """


class ReleaseHandle:
    """A verified, read-only view of one release directory.

    Construct with :meth:`open`. Every accessor is a projection of ``manifest.json``,
    ``audit.json``, ``examples.jsonl``, or ``splits/*.jsonl``, and the class holds no
    state a rebuild could invalidate.
    """

    def __init__(self, path: Path, manifest: DatasetManifest, audit: AuditResult) -> None:
        self.path = path
        self.manifest = manifest
        self.audit = audit

    # ----------------------------------------------------------------- open #

    @classmethod
    def open(cls, release_dir: str | Path, *, verify: bool = True) -> ReleaseHandle:
        """Open a release directory.

        ``verify=True`` runs :func:`~eval_dataset_forge.datasets.manifest.verify_release`
        first and raises :class:`ContractViolation` on any problem. It is the default
        because a consumer that gated on an audit whose bytes do not match its manifest
        gated on nothing. Pass ``verify=False`` only to inspect a release you already
        know is broken.

        Integrity is checkable offline and needs no optional dependency — see
        :meth:`verify`. That is not true of *rebuilding* the release.
        """
        release_dir = Path(release_dir)
        if not release_dir.is_dir():
            raise ContractViolation(f"{release_dir} is not a release directory")
        if verify:
            report = verify_release(release_dir)
            if not report.ok:
                problems = "; ".join(report.problems)
                raise ContractViolation(f"{release_dir} failed verification: {problems}")
        try:
            manifest = load_manifest(release_dir / MANIFEST_FILE)
            audit = load_audit(release_dir / AUDIT_FILE)
        except Exception as error:  # noqa: BLE001 — any parse failure is one violation
            raise ContractViolation(f"{release_dir}: {error}") from error
        return cls(release_dir, manifest, audit)

    def verify(self) -> VerifyReport:
        """Re-run the integrity check.

        Recomputes file hashes, ``content_id``, and ``eval_id`` from the release's own
        bytes. It runs no detector, so its answer does not depend on which optional
        packages are importable — unlike the build that produced the release. Pinned by
        ``test_verification_is_environment_independent``.
        """
        return verify_release(self.path)

    # ------------------------------------------------------------- identity #

    @property
    def identity(self) -> dict[str, str]:
        """The three answers a consumer needs before it trusts a number.

        ``content_id`` — *are these the same examples?* A gate verdict is scoped to it.
        ``eval_id`` — *are two scores comparable?* A leaderboard row is scoped to it.
        ``near_method`` — *would this build reproduce?* See :mod:`.gate`.
        """
        return {
            "schema_version": self.manifest.schema_version,
            "content_id": self.manifest.content_id,
            "eval_id": self.manifest.eval_id,
            "near_method": self.near_method,
            "tool_version": self.manifest.tool_version,
        }

    @property
    def near_method(self) -> str:
        """The near-duplicate method that actually ran (``tfidf`` or ``jaccard``).

        Not decoration. ``tfidf`` runs iff scikit-learn was importable on the building
        machine, and the two methods disagree about which examples are duplicates —
        which changes the blocks, the adversarial split, and therefore ``eval_id``.
        ``docs/INTEGRATION.md`` has the measurement.
        """
        return str(self.manifest.dependence.get("near_method", "unknown"))

    @property
    def dependence(self) -> dict[str, object]:
        """The block structure. Contains no ``effective_n``; see the manifest's docstring."""
        return dict(self.manifest.dependence)

    def blocks_in(self, split: str) -> int | None:
        """Independent blocks touching ``split``, or ``None`` if the release omits them.

        This — not ``len(handle.split("test"))`` — is the sample size a bootstrap over
        the test split gets to spend.
        """
        by_split = self.manifest.dependence.get("blocks_by_split")
        if not isinstance(by_split, dict) or split not in by_split:
            return None
        return int(by_split[split])

    def require_same_corpus(self, content_id: str) -> None:
        """Raise unless ``content_id`` names the examples in this release.

        A gate verdict is a statement about a corpus, and this repository has measured
        that the statement does not survive the corpus changing under it — not even when
        the change adds only unrelated rows. See :meth:`~.gate.GateReport.applies_to`.
        """
        if content_id != self.manifest.content_id:
            raise ContractViolation(
                f"verdict was computed on {content_id}, this release is "
                f"{self.manifest.content_id}; re-run the gate"
            )

    # ----------------------------------------------------------------- data #

    @functools.cached_property
    def examples(self) -> list[DatasetExample]:
        """Every released example, in file order."""
        return loader.load_examples(self.path / EXAMPLES_FILE)

    def split(self, name: str) -> list[DatasetExample]:
        """The rows of ``splits/<name>.jsonl``. Raises if the release has no such split."""
        path = self.path / SPLITS_DIR / f"{name}.jsonl"
        if not path.is_file():
            available = ", ".join(sorted(self.manifest.splits)) or "none"
            raise ContractViolation(f"no {name!r} split in {self.path} (have: {available})")
        return loader.load_examples(path)

    # ------------------------------------------------------------- findings #

    def leakage(self, kind: LeakageKind) -> list[LeakageFinding]:
        return [flag for flag in self.audit.leakage_flags if flag.kind is kind]

    def label_warnings(self, kind: LabelIssue) -> list[LabelAuditFinding]:
        return [warning for warning in self.audit.label_warnings if warning.kind is kind]

    def split_counts(self) -> dict[str, object]:
        counts = self.audit.split_statistics.get("counts")
        return dict(counts) if isinstance(counts, dict) else {}

    def examples_with_retrieval_docs(self) -> int:
        """How many released examples carry ``metadata.retrieval.docs``.

        The one place this package looks past the evidence bundle at the data, and it
        does so to answer a question the bundle cannot: ``audit.json`` records what the
        retrieval-contamination check *found*, never what it was *given*. Zero findings
        over zero passages and zero findings over a thousand clean passages serialize
        identically. Milestone 12 made that distinction for a check that could not run;
        a consumer gating on ``leakage_flags == []`` re-erases it.
        """
        return sum(1 for example in self.examples if retrieval_docs(example))


__all__ = [
    "AUDIT_FIELDS",
    "DEPENDENCE_FIELDS",
    "MANIFEST_FIELDS",
    "SPLIT_COUNT_FIELDS",
    "SPLIT_STAT_FIELDS",
    "ContractViolation",
    "ReleaseHandle",
]
