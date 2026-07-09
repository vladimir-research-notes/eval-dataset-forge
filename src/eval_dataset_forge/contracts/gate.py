"""The gate: a policy, a verdict, and the corpus that verdict is about (Milestone 13).

An audit has no opinion. It reports that ``qa-007`` quotes its gold answer in its prompt
and leaves the reader to decide whether that is disqualifying. A *gate* supplies the
opinion, and the opinion is not a property of the dataset — it belongs to the consumer.
So a policy is a first-class, serializable object that travels with the consumer rather
than with the release, and :class:`GateReport` records which policy produced it.

**A gate verdict is scoped to a corpus, and this repository has measured that it does not
survive that corpus changing — even when the change adds only unrelated rows.**
``safety_toy`` alone reports one near-duplicate cluster and passes a
``max_near_duplicate_clusters = 1`` policy. Concatenated with the four other fixtures —
which share not one duplicate with it — the same ten rows report two clusters and fail.
Nothing about ``safety_toy`` changed: TF-IDF's idf term is computed over the collection,
so the boilerplate ``saf-004`` and ``saf-007`` share is common inside ``safety_toy`` and
rare against the pooled vocabulary, and rarer means heavier. ``sim`` moves 0.2866 → 0.3438
across a 0.30 threshold.

The consequence is not a caveat, it is an API constraint. :class:`GateReport` carries the
``content_id`` it was computed on, :meth:`GateReport.applies_to` refuses to answer for any
other, and a consumer that merges two gated releases has an ungated corpus. Union is not a
gate-preserving operation. (Under the ``jaccard`` fallback it is; see
:mod:`.requirements` and ``docs/INTEGRATION.md``.)

**Nor is re-splitting, for four of the fourteen requirements.** ``no_train_test_overlap``,
``no_retrieval_contamination``, ``eval_splits_deduplicated``, and ``eval_split_is_rankable``
read the split assignment, so their answers move when the seed does while ``content_id``
stands still: under ``--strategy random`` on ``qa_toy``, seeds 9 and 35 pass
``eval_splits_deduplicated`` and thirty-eight other seeds fail it, across forty releases
sharing one ``content_id``. A report is therefore scoped to ``eval_id`` whenever one of those
ran, and to ``content_id`` otherwise — :attr:`GateReport.scope` says which, and the report
derives it from the requirements that actually executed rather than from the split strategy
anyone claims to have used.

Exit-code semantics, which are the whole point of a gate:

===============  =====================================================
``report.ok``    no ``FAIL``, and no ``UNDECIDABLE`` unless the policy
                 set ``allow_undecidable``
``exit 0``       ``report.ok``
``exit 1``       anything else
===============  =====================================================
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from eval_dataset_forge import __version__
from eval_dataset_forge.contracts.profiles import Profile, get_profile
from eval_dataset_forge.contracts.reader import ContractViolation, ReleaseHandle
from eval_dataset_forge.contracts.requirements import (
    REQUIREMENTS,
    RequirementResult,
    Verdict,
    evaluate_all,
    resolve_thresholds,
)
from eval_dataset_forge.schemas import ForgeModel, NonEmptyStr

#: Bumped when a declared key in :mod:`.reader` is removed or retyped, when a requirement
#: changes its verdict for the same evidence, or when the exit-code rule changes. Adding a
#: requirement, a profile, or a threshold is additive and does not move it.
CONTRACT_VERSION = "1.0"


class GatePolicy(ForgeModel):
    """What a consumer requires of a release. Serializable, so it can be reviewed.

    ``allow_undecidable`` defaults to ``False``: a requirement that could not run is not a
    requirement that passed. Setting it true is a decision a reader of the policy file can
    see, which is the only reason it is a field rather than a flag buried in a call.
    """

    profile: NonEmptyStr
    contract_version: str = CONTRACT_VERSION
    requirements: list[str] = Field(min_length=1)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    allow_undecidable: bool = False

    @field_validator("requirements")
    @classmethod
    def _known_requirements(cls, names: list[str]) -> list[str]:
        unknown = [name for name in names if name not in REQUIREMENTS]
        if unknown:
            raise ValueError(
                f"unknown requirement(s): {', '.join(unknown)}. "
                f"known: {', '.join(sorted(REQUIREMENTS))}"
            )
        return names

    @classmethod
    def for_profile(cls, name: str, **overrides: Any) -> GatePolicy:
        """The committed policy for one of the five consumer profiles."""
        profile: Profile = get_profile(name)
        return cls(
            profile=profile.name,
            requirements=list(profile.requirements),
            thresholds=dict(profile.thresholds),
            **overrides,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> GatePolicy:
        """Load a policy from YAML or JSON. Unknown keys and requirements raise."""
        import yaml

        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a policy must be a mapping, got {type(data).__name__}")
        return cls(**data)

    def resolved_thresholds(self) -> dict[str, Any]:
        return resolve_thresholds(self.thresholds)


@dataclass(frozen=True)
class GateReport:
    """The outcome of one policy applied to one release.

    ``content_id`` is not provenance decoration: it is the scope of every verdict in
    ``results``. See :meth:`applies_to`.
    """

    release_dir: str
    dataset_id: str
    version: str
    content_id: str
    eval_id: str
    near_method: str
    policy: GatePolicy
    results: tuple[RequirementResult, ...]
    tool_version: str = __version__
    contract_version: str = CONTRACT_VERSION

    def _with(self, verdict: Verdict) -> tuple[RequirementResult, ...]:
        return tuple(result for result in self.results if result.verdict is verdict)

    @property
    def failed(self) -> tuple[RequirementResult, ...]:
        return self._with(Verdict.FAIL)

    @property
    def undecided(self) -> tuple[RequirementResult, ...]:
        return self._with(Verdict.UNDECIDABLE)

    @property
    def passed(self) -> tuple[RequirementResult, ...]:
        return self._with(Verdict.PASS)

    @property
    def ok(self) -> bool:
        """Pass iff nothing failed and nothing was undecidable the policy did not permit."""
        if self.failed:
            return False
        return self.policy.allow_undecidable or not self.undecided

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    @property
    def split_sensitive(self) -> bool:
        """Whether any requirement that ran can change verdict under a re-split."""
        return any(result.split_sensitive for result in self.results)

    @property
    def scope(self) -> str:
        """The identity this verdict is scoped to: ``eval_id`` or ``content_id``."""
        return "eval_id" if self.split_sensitive else "content_id"

    def applies_to(self, handle: ReleaseHandle) -> bool:
        """Whether this verdict is a statement about ``handle``.

        ``content_id`` always. ``eval_id`` **as well**, whenever a split-sensitive
        requirement ran — because four of them read the split, and a re-split moves their
        answer while leaving ``content_id`` fixed. Under ``--strategy random`` on ``qa_toy``,
        seeds 9 and 35 pass ``eval_splits_deduplicated`` and thirty-eight other seeds fail
        it, and all forty releases share one ``content_id``. A gate that compared content
        alone would carry a pass onto a failing release.

        The adversarial strategy holds those four at zero for every seed, so an adversarial
        verdict is content-scoped in practice. The gate does not rely on that: it reads which
        requirements actually ran.

        Not covered: ``integrity`` is a statement about the *bytes of the directory this
        report names*. Two releases can share both identities and differ in ``created_at`` or
        in a card's wording, so a report is never evidence about a directory other than
        :attr:`release_dir`, whatever this method says.
        """
        if self.content_id != handle.manifest.content_id:
            return False
        return not self.split_sensitive or self.eval_id == handle.manifest.eval_id

    def assert_applies_to(self, handle: ReleaseHandle) -> None:
        """Raise :class:`~.reader.ContractViolation` unless :meth:`applies_to`."""
        if self.content_id != handle.manifest.content_id:
            raise ContractViolation(
                f"gate verdict for {self.dataset_id} was computed on corpus "
                f"{self.content_id}, but this release is {handle.manifest.content_id}; "
                f"a gate verdict is not preserved under corpus change — re-run the gate"
            )
        if self.split_sensitive and self.eval_id != handle.manifest.eval_id:
            sensitive = ", ".join(r.name for r in self.results if r.split_sensitive)
            raise ContractViolation(
                f"gate verdict for {self.dataset_id} is scoped to eval_id {self.eval_id} "
                f"because {sensitive} read the split, but this release is "
                f"{handle.manifest.eval_id}; a split-sensitive verdict is not preserved "
                f"under re-split — re-run the gate"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_dir": self.release_dir,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "content_id": self.content_id,
            "eval_id": self.eval_id,
            "near_method": self.near_method,
            "scope": self.scope,
            "tool_version": self.tool_version,
            "contract_version": self.contract_version,
            "policy": self.policy.model_dump(mode="json"),
            "ok": self.ok,
            "counts": {
                "pass": len(self.passed),
                "fail": len(self.failed),
                "undecidable": len(self.undecided),
            },
            "results": [result.to_dict() for result in self.results],
        }


def evaluate_gate(
    handle: ReleaseHandle,
    policy: GatePolicy,
    *,
    requirements: Sequence[str] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> GateReport:
    """Apply ``policy`` to ``handle``.

    ``requirements`` and ``thresholds`` override the policy's own, for the ad-hoc case
    (``evalforge gate --requirement …``). The report records the policy as it was *used*,
    not as it arrived, so a stored report cannot describe a run that did not happen.
    """
    names = list(requirements) if requirements is not None else list(policy.requirements)
    merged = {**policy.thresholds, **dict(thresholds or {})}
    used = policy.model_copy(update={"requirements": names, "thresholds": merged})
    results = evaluate_all(names, handle, resolve_thresholds(merged))
    return GateReport(
        release_dir=str(handle.path),
        dataset_id=handle.manifest.dataset_id,
        version=handle.manifest.version,
        content_id=handle.manifest.content_id,
        eval_id=handle.manifest.eval_id,
        near_method=handle.near_method,
        policy=used,
        results=results,
    )


def gate_release(release_dir: str | Path, profile: str, **overrides: Any) -> GateReport:
    """Open a release and run one of the five committed consumer profiles against it.

    Opens with ``verify=False`` on purpose. Verification is itself the ``integrity``
    requirement, and a gate that raised on a corrupt release would hand its caller an
    exception where every other failure produces a report. A release whose bytes do not
    match its manifest gets ``integrity: FAIL`` and exit 1, like any other defect. Only a
    release that cannot be *parsed* raises :class:`~.reader.ContractViolation`, because
    there is then no ``dataset_id`` to write a report about.
    """
    handle = ReleaseHandle.open(release_dir, verify=False)
    return evaluate_gate(handle, GatePolicy.for_profile(profile, **overrides))


__all__ = [
    "CONTRACT_VERSION",
    "GatePolicy",
    "GateReport",
    "evaluate_gate",
    "gate_release",
]
