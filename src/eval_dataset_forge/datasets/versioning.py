"""Dataset identity and version-bump inference (Milestone 10).

A benchmark release is a claim, and the claim is not *"here are some rows"*. It is
*"a score measured on this is comparable to a score measured on that"*. A filename and a
hand-typed version string do not support the claim; a hand-typed version string is, in
practice, a number someone forgot to increment.

So identity is **computed**, at two levels:

``content_id``
    An order-independent digest over the released examples' content, with ``split``
    excluded. Permuting ``examples.jsonl``, reformatting its JSON, or rebuilding the
    splits leaves it alone. It answers *"are these the same examples?"*

``eval_id``
    The same digest with each example's split folded in. It answers *"are two reported
    scores comparable?"* — and its answer differs from ``content_id``'s far more often
    than curators expect.

**Why the second one has to exist.** On this repository's own pooled fixture corpus
(56 examples, 3 simulated systems), re-running the split builder with a different seed
leaves ``content_id`` untouched — the dataset is identical, byte-for-byte, as a set of
examples — and moves the reported test-set accuracy of ``sys_a`` across ``[0.364, 1.000]``
with a standard deviation of 0.127. The gap between the top two systems is **0.036**. The
split seed is therefore worth ~3.5× the effect the benchmark exists to measure, and under
the ``random`` strategy the test split reproduces the corpus's own ordering of the top two
systems **48.7%** of the time — a coin flip. A release identified only by its content hash
would call those two releases the same. They are not. See ``docs/RELEASES.md`` and
``scripts/split_identity_experiment.py``.

**The bump rule follows from that, not from taste.**

===========  ====================================================================
bump         what changed
===========  ====================================================================
``major``    anything visible to an evaluation split (``dev`` ∪ ``test``): an
             example added, removed, edited, or reassigned across the boundary —
             or a ``schema_version`` change. Old scores are not comparable.
``minor``    content changed, but only within ``train``. A reported ``test``
             number is unaffected.
``patch``    no content and no split changed; artifacts, provenance, or audit
             counts did. Same numbers, different packaging.
``none``     nothing but the clock.
===========  ====================================================================

``minor`` rests on an assumption that is true for score *comparability* and false for
benchmark *validity*: adding a train example cannot change what a fixed system scores on
``test``, but it can introduce train→test leakage that changes what that score *means* for
a system which trains on the release. The delta reports the level; it does not re-audit.
That is what the leakage stage is for, and why :func:`compare` also warns when a ``minor``
bump touched examples that duplicate an evaluation example.

Examples are joined across releases by ``example_id``. Re-issuing an id with different
content reads as ``changed``; renumbering ids without touching content reads as
``removed`` + ``added``. Both are conservative in the direction that matters: a leaderboard
keyed by ``example_id`` really does break under either.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from eval_dataset_forge import SCHEMA_VERSION
from eval_dataset_forge.datasets.writer import HASH_PREFIX, canonical_bytes
from eval_dataset_forge.schemas import DatasetExample, DatasetManifest, Split

# Fields excluded from an example's *content* digest. ``split`` is the only one: it is not
# a property of the example, it is a property of the release's arrangement of examples, and
# folding it into ``content_id`` would make a re-split look like new data.
CONTENT_EXCLUDED_FIELDS = frozenset({"split"})

# Splits a reported score is measured on. A change confined outside these cannot move a
# number computed on them.
EVAL_SPLITS: frozenset[str] = frozenset({str(Split.DEV), str(Split.TEST)})

# Domain separation. Each tag pins a *purpose* and a schema version into the digest, so an
# id computed under one schema can never collide with an id computed under another, and a
# content id can never be mistaken for an eval id.
_EXAMPLE_TAG = b"evalforge-example-v1"
_CONTENT_TAG = "evalforge-content-v1"
_EVAL_TAG = "evalforge-eval-v1"

BUMP_LEVELS = ("none", "patch", "minor", "major")

# Manifest fields that describe the *packaging* rather than the data. A difference in one
# of these, with identity unchanged, is a ``patch``. ``created_at`` is absent on purpose:
# a rebuild at a later clock reading is not a new release.
_PACKAGING_FIELDS = (
    "hashes",
    "source_files",
    "license_notes",
    "known_limitations",
    "tool_version",
    "audit_results",
    "excluded",
    "dependence",
    "splits",
)

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def content_fields(example: DatasetExample) -> dict[str, Any]:
    """The example's JSON form with :data:`CONTENT_EXCLUDED_FIELDS` removed."""
    dumped = example.model_dump(mode="json")
    return {key: value for key, value in dumped.items() if key not in CONTENT_EXCLUDED_FIELDS}


def example_digest(example: DatasetExample) -> str:
    """A 64-char hex digest of one example's content. Independent of key order and split."""
    payload = _EXAMPLE_TAG + b"\n" + canonical_bytes(content_fields(example))
    return hashlib.sha256(payload).hexdigest()


def _aggregate(tag: str, lines: Iterable[str]) -> str:
    payload = "\n".join([tag, SCHEMA_VERSION, *sorted(lines)])
    return HASH_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_id(examples: Iterable[DatasetExample]) -> str:
    """Order-independent ``sha256:`` id of the *set* of examples, ignoring splits.

    Sorted digests rather than a set of them, so an accidental duplicate row still changes
    the id. A dataset that contains the same example twice is a different dataset from one
    that contains it once — the deduplication stage exists because that difference matters.
    """
    return _aggregate(_CONTENT_TAG, (example_digest(example) for example in examples))


def eval_id(examples: Iterable[DatasetExample]) -> str:
    """Order-independent ``sha256:`` id of the examples *and* their split assignment.

    Two builds that reach the same assignment by different seeds share an ``eval_id``, and
    should: their numbers are comparable. The id is a function of the arrangement, never of
    the strategy that produced it — which is why the manifest stores ``split_spec`` for
    provenance and hashes the *result* for identity.
    """
    return _aggregate(
        _EVAL_TAG, (f"{example_digest(example)} {example.split}" for example in examples)
    )


def digests_by_id(examples: Iterable[DatasetExample]) -> dict[str, str]:
    """``example_id`` → content digest. Raises on a duplicate id (identity is undefined)."""
    table: dict[str, str] = {}
    for example in examples:
        if example.example_id in table:
            raise ValueError(f"duplicate example_id {example.example_id!r}: identity is undefined")
        table[example.example_id] = example_digest(example)
    return table


def splits_by_id(examples: Iterable[DatasetExample]) -> dict[str, str]:
    """``example_id`` → split name."""
    return {example.example_id: str(example.split) for example in examples}


# --------------------------------------------------------------------------- #
# Semantic versions
# --------------------------------------------------------------------------- #


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse ``MAJOR.MINOR.PATCH``. Raises :class:`ValueError` on anything else."""
    match = _SEMVER.match(version.strip())
    if not match:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def next_version(previous: str, bump: str) -> str:
    """Apply ``bump`` to ``previous`` under strict semantic versioning.

    No ``0.x`` special case. The common convention — *"anything goes below 1.0, so bump the
    minor"* — would collapse ``major`` and ``minor`` into one number exactly on the datasets
    most likely to be re-cut weekly, and the one bit this module exists to publish is
    whether yesterday's scores still stand. ``0.1.0`` + ``major`` is ``1.0.0``.
    """
    if bump not in BUMP_LEVELS:
        raise ValueError(f"unknown bump {bump!r}; want one of {BUMP_LEVELS}")
    major, minor, patch = parse_version(previous)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return f"{major}.{minor}.{patch}"


# --------------------------------------------------------------------------- #
# Deltas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReleaseDelta:
    """What changed between two releases, and the smallest honest version bump.

    ``reasons`` is ordered most-severe first and names the examples responsible, so the
    bump can be argued with rather than merely obeyed.
    """

    bump: str
    previous_version: str
    reasons: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    reassigned: tuple[str, ...] = ()
    eval_affected: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    identity: dict[str, bool] = field(default_factory=dict)

    @property
    def recommended_version(self) -> str:
        return next_version(self.previous_version, self.bump)

    @property
    def scores_comparable(self) -> bool:
        """True iff a score measured on the previous release still stands on this one."""
        return self.bump in ("none", "patch", "minor")

    def counts(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "changed": len(self.changed),
            "reassigned": len(self.reassigned),
            "eval_affected": len(self.eval_affected),
        }

    def to_dict(self) -> dict:
        return {
            "bump": self.bump,
            "previous_version": self.previous_version,
            "recommended_version": self.recommended_version,
            "scores_comparable": self.scores_comparable,
            "counts": self.counts(),
            "identity": dict(self.identity),
            "reasons": list(self.reasons),
            "added": list(self.added),
            "removed": list(self.removed),
            "changed": list(self.changed),
            "reassigned": list(self.reassigned),
            "eval_affected": list(self.eval_affected),
            "warnings": list(self.warnings),
        }


def _eval_fingerprint(examples: Sequence[DatasetExample]) -> list[str]:
    """Sorted ``digest split`` pairs, restricted to the evaluation splits."""
    return sorted(
        f"{example_digest(example)} {example.split}"
        for example in examples
        if str(example.split) in EVAL_SPLITS
    )


def _packaging_changed(previous: DatasetManifest, current: DatasetManifest) -> list[str]:
    dumped_previous = previous.model_dump(mode="json")
    dumped_current = current.model_dump(mode="json")
    return [
        name
        for name in _PACKAGING_FIELDS
        if dumped_previous.get(name) != dumped_current.get(name)
    ]


def compare(
    previous_manifest: DatasetManifest,
    previous_examples: Sequence[DatasetExample],
    current_manifest: DatasetManifest,
    current_examples: Sequence[DatasetExample],
) -> ReleaseDelta:
    """Classify the change from one release to the next and recommend a version.

    The classification is computed from the two example sets, never from the manifests'
    stored ids: a manifest is a summary, and a summary that disagrees with its data should
    be caught by :func:`~eval_dataset_forge.datasets.manifest.verify_release` rather than
    trusted here.
    """
    previous_digest = digests_by_id(previous_examples)
    current_digest = digests_by_id(current_examples)
    previous_split = splits_by_id(previous_examples)
    current_split = splits_by_id(current_examples)

    added = tuple(sorted(set(current_digest) - set(previous_digest)))
    removed = tuple(sorted(set(previous_digest) - set(current_digest)))
    retained = sorted(set(previous_digest) & set(current_digest))
    changed = tuple(eid for eid in retained if previous_digest[eid] != current_digest[eid])
    reassigned = tuple(
        eid
        for eid in retained
        if previous_digest[eid] == current_digest[eid]
        and previous_split[eid] != current_split[eid]
    )

    touched = {*added, *removed, *changed, *reassigned}
    eval_affected = tuple(
        sorted(
            eid
            for eid in touched
            if previous_split.get(eid) in EVAL_SPLITS or current_split.get(eid) in EVAL_SPLITS
        )
    )

    schema_changed = previous_manifest.schema_version != current_manifest.schema_version
    eval_changed = _eval_fingerprint(previous_examples) != _eval_fingerprint(current_examples)
    content_changed = bool(added or removed or changed)
    packaging = _packaging_changed(previous_manifest, current_manifest)

    reasons: list[str] = []
    if schema_changed:
        reasons.append(
            f"schema_version {previous_manifest.schema_version} → "
            f"{current_manifest.schema_version}: every stored record is reinterpreted"
        )
    if eval_changed:
        reasons.append(
            f"{len(eval_affected)} example(s) added to, removed from, edited in, or "
            f"reassigned across dev/test: scores on the previous release do not carry over"
        )
    if content_changed and not eval_changed:
        reasons.append(
            f"{len(added) + len(removed) + len(changed)} example(s) changed, all confined "
            f"to train: a score measured on dev/test is unaffected"
        )
    if packaging and not (schema_changed or eval_changed or content_changed):
        reasons.append("packaging only: " + ", ".join(packaging))

    if schema_changed or eval_changed:
        bump = "major"
    elif content_changed:
        bump = "minor"
    elif packaging:
        bump = "patch"
    else:
        bump = "none"

    warnings: list[str] = []
    if bump == "minor":
        warnings.append(
            "a train-only change cannot move a fixed system's test score, but it can "
            "introduce train→test leakage that changes what that score measures for a "
            "system trained on this release; re-read the leakage section of audit.json"
        )
    if reassigned and not (added or removed or changed):
        warnings.append(
            f"{len(reassigned)} example(s) were re-split with no content change: "
            f"content_id is unchanged and eval_id is not"
        )

    return ReleaseDelta(
        bump=bump,
        previous_version=previous_manifest.version,
        reasons=tuple(reasons),
        added=added,
        removed=removed,
        changed=changed,
        reassigned=reassigned,
        eval_affected=eval_affected,
        warnings=tuple(warnings),
        identity={
            "content_id_equal": content_id(previous_examples) == content_id(current_examples),
            "eval_id_equal": eval_id(previous_examples) == eval_id(current_examples),
            "schema_version_equal": not schema_changed,
        },
    )


def format_delta(delta: ReleaseDelta, max_items: int = 10) -> str:
    """Render a :class:`ReleaseDelta` for a terminal."""
    counts = delta.counts()
    lines = [
        f"delta: {delta.previous_version} → {delta.recommended_version} "
        f"({delta.bump} bump; scores "
        f"{'still comparable' if delta.scores_comparable else 'NOT comparable'})",
        f"  content_id : {'same' if delta.identity.get('content_id_equal') else 'changed'}",
        f"  eval_id    : {'same' if delta.identity.get('eval_id_equal') else 'changed'}",
        "  examples   : "
        + ", ".join(f"{name}={value}" for name, value in counts.items()),
    ]
    for reason in delta.reasons:
        lines.append(f"  - {reason}")
    for label, ids in (
        ("added", delta.added),
        ("removed", delta.removed),
        ("changed", delta.changed),
        ("reassigned", delta.reassigned),
    ):
        if ids:
            shown = ", ".join(ids[:max_items])
            more = f" … and {len(ids) - max_items} more" if len(ids) > max_items else ""
            lines.append(f"  {label:<11}: {shown}{more}")
    for warning in delta.warnings:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)


__all__ = [
    "BUMP_LEVELS",
    "CONTENT_EXCLUDED_FIELDS",
    "EVAL_SPLITS",
    "ReleaseDelta",
    "compare",
    "content_fields",
    "content_id",
    "digests_by_id",
    "eval_id",
    "example_digest",
    "format_delta",
    "next_version",
    "parse_version",
    "splits_by_id",
]
