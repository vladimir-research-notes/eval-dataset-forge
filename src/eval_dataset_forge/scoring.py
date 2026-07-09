"""Scoring an audit against a fixture's planted ground truth (Milestone 12).

Each fixture ships a ``defects.yaml`` naming the defects deliberately planted in it. The smoke
audit rediscovers them, and this module turns "rediscovered" into a number. Three decisions do
the work, and each exists because the obvious alternative is a way of grading one's own homework.

**1. Recall is not a scoring rule.** A detector that flags every example finds every planted
defect. On ``qa_toy`` this is not hypothetical arithmetic: the flag-everything detector scores
recall ``1.000`` on the duplicate channels, and so does the real one. No matching rule repairs
this — subset matching, exact-set matching, and the pairwise rule below all award the degenerate
detector full recall, because full recall is what it has. Recall bounds what a detector *misses*
and says nothing about what it *invents*. So every recall printed here is printed beside a
precision, and :func:`degenerate_predictions` puts the flag-everything detector through the same
scorer so a reader can see what an uninformative answer scores.

**2. Precision needs a unit, and a cluster is the wrong one.** Duplicate findings are clusters,
not pairs. Scoring a cluster as one prediction forces a bad choice: count it correct if it
*contains* a planted pair, and one 14-member cluster is "correct"; count it correct only if it
*equals* a planted pair, and a true 3-member cluster covering a planted pair is "wrong". Both
rules are wrong because a cluster is not an assertion — it is a *conjunction* of assertions. A
cluster of size ``k`` asserts all ``k(k-1)/2`` of its pairs. Decomposed that way, precision and
recall live on the same unit, a cluster's cost grows quadratically with its size, and the
degenerate detector's precision collapses to ``3/91 = 0.033`` while the real detector holds
``0.750``. The decomposition also earns its keep on real output: the near stage merges
``qa-003``, ``qa-004`` and ``qa-012`` into one cluster, and thereby asserts ``qa-004 ≈ qa-012``,
which nothing plants. A cluster-level view files that finding as correct and never shows the
claim.

**3. The denominator is chosen by the people who wrote the detectors.** Two classes of planted
defect are excluded from recall: those the fixture marks ``auto_detectable: false`` (the curator's
declaration that no heuristic here should find it) and those this scorer wires to no channel at
all (a fact about *this module*, not about the fixture). Collapsing the two would let a missed
defect be reclassified as an out-of-scope one. They are counted separately, listed by id, and
:class:`ScoreReport` reports the recall over *all* planted defects alongside the recall over the
detectable ones, so a reader can check the denominator rather than accept it.

A ground-truth kind this module does not know raises :class:`ValueError` rather than being
skipped. A silent skip would remove a defect from the denominator by typo, which is precisely the
failure the paragraph above is about.

**What "precision" means here, exactly.** ``defects.yaml`` records what was *planted*, not an
exhaustive labelling of every pair in the fixture, so an assertion missing from the ground truth
is not automatically wrong. Each channel therefore reports two numbers. ``precision`` is strict
and kind-specific — of the atoms this channel asserted, how many are planted *as this kind*? It is
the number that collapses for the flag-everything detector, and it is undefined (``None``) when
the fixture plants nothing of that kind, because ``0.0`` would then be the only possible value for
any non-silent detector. ``unexplained`` counts the atoms the ground truth cannot account for in
*any* section: the n-gram check's ``qa-002`` is planted as half of an ``exact_duplicates`` pair
and is a corroboration, while the near stage's implied ``qa-004 ≈ qa-012`` — whose endpoints are
each planted but whose *pairing* is not — is an invention, and only the second is a reason to
worry. The projection is unit-preserving so that this distinction cannot degrade into a blanket
pardon (see :meth:`GroundTruth.explained_atoms`).

**Every number here is in-sample.** The near-duplicate and answer-in-prompt thresholds were
selected by sweeps over these same fixtures in Milestones 4 and 5. A recall measured on the data a
threshold was tuned on is a resubstitution estimate; it does not estimate performance on a corpus
this repository has not seen. That is the same defect this project exists to detect in other
people's benchmarks, and it is stated here rather than left for a reader to notice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from eval_dataset_forge.audit import AuditReport
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.schemas import DatasetExample

# The three units a planted defect can be stated in. A unit is what recall and precision are both
# counted over, which is the whole reason to name them.
PAIR = "pair"
EXAMPLE = "example"
DOC = "doc"

# A planted defect that no audit dimension in this project claims to detect. Excluded from the
# recall denominator, counted and named in the report. Distinct from ``auto_detectable: false``:
# that is the *fixture's* claim about detectability, this is *this scorer's* wiring.
NO_CHANNEL = "none"

# A planted property that is not a defect and must NOT be flagged. Scored inverted.
NEGATIVE_CONTROL = "negative-control"

# Ground-truth ``(section, kind)`` → ``(channel, unit)``. ``kind`` is ``None`` for sections whose
# entries carry no ``kind:`` field. An unmapped key raises rather than silently leaving the
# denominator.
CHANNEL_OF: dict[tuple[str, str | None], tuple[str, str]] = {
    ("exact_duplicates", None): ("dedup.exact", PAIR),
    ("near_duplicates", None): ("dedup.near", PAIR),
    ("duplicate_sources", None): ("dedup.duplicate_sources", PAIR),
    ("duplicate_documents", None): ("leakage.duplicate_documents", DOC),
    # A lexical paraphrase of a *passage*. The dedup ladder compares prompts, not retrieved
    # documents, so nothing in this project looks for it.
    ("paraphrased_documents", None): (NO_CHANNEL, DOC),
    ("leakage", "train_test_overlap"): ("leakage.train_test_overlap", PAIR),
    ("leakage", "answer_in_prompt"): ("leakage.answer_in_prompt", EXAMPLE),
    ("leakage", "ngram_overlap"): ("leakage.ngram_overlap", EXAMPLE),
    ("leakage", "retrieval_contamination"): ("leakage.retrieval_contamination", EXAMPLE),
    ("label_issues", "missing"): ("labels.missing", EXAMPLE),
    ("label_issues", "conflicting"): ("labels.conflicting", EXAMPLE),
    ("label_issues", "ambiguous"): ("labels.ambiguous", EXAMPLE),
    ("label_issues", "inconsistent_format"): ("labels.inconsistent_format", EXAMPLE),
    # A gold answer that is well-formed, unanimous, unambiguous, correctly shaped — and wrong.
    # No structural check can see it; the fixtures also mark it ``auto_detectable: false``.
    ("label_issues", "mislabeled"): (NO_CHANNEL, EXAMPLE),
    # A relevance judgement that contradicts the gold doc-id list. Detected by the label audit's
    # conflict channel comparing the two relevance channels as sets — a different section of the
    # ground truth than the channel that finds it, which is why the mapping is explicit.
    ("relevance_issues", "incomplete_relevance"): ("labels.conflicting", EXAMPLE),
    # Not a defect: an unanswerable, over-specific query whose empty gold list is *correct*. It is
    # planted so that a rule keyed on emptiness is caught flagging it.
    ("relevance_issues", "over_specific_query"): (NEGATIVE_CONTROL, EXAMPLE),
}

# For a negative control, the channel(s) that must stay silent. Naming them is the point: "must not
# be flagged by anything" is untestable, "must not be flagged by the missing-label check" is a
# claim about a specific rule that a specific fixture example was written to trap.
CONTROL_CHANNELS: dict[str, tuple[str, ...]] = {
    "over_specific_query": ("labels.missing",),
}

# Channels grouped by unit, for the corroboration pass ("recovered by *any* channel") and for the
# planted-universe a precision is measured against.
PAIR_CHANNELS = (
    "dedup.exact",
    "dedup.normalized",
    "dedup.near",
    "dedup.duplicate_sources",
    "leakage.train_test_overlap",
)
EXAMPLE_CHANNELS = (
    "leakage.answer_in_prompt",
    "leakage.ngram_overlap",
    "leakage.retrieval_contamination",
    "labels.missing",
    "labels.conflicting",
    "labels.ambiguous",
    "labels.inconsistent_format",
)
DOC_CHANNELS = ("leakage.duplicate_documents",)

# Outcome vocabulary for one planted defect.
RECOVERED = "recovered"
RECOVERED_ELSEWHERE = "recovered-by-another-channel"
MISSED = "missed"
NOT_AUTO_DETECTABLE = "not-auto-detectable"
UNWIRED = "no-channel"
CONTROL_HELD = "control-held"
CONTROL_VIOLATED = "control-violated"


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Planted:
    """One deliberately planted defect, as ``defects.yaml`` states it."""

    section: str
    kind: str
    channel: str
    unit: str
    members: tuple[str, ...]
    auto_detectable: bool = True
    note: str = ""

    @property
    def scorable(self) -> bool:
        """Does a wired channel claim this defect, and does the fixture expect it found?"""
        return (
            self.auto_detectable
            and self.channel != NO_CHANNEL
            and self.channel != NEGATIVE_CONTROL
        )

    def units(self) -> set[frozenset[str] | str]:
        """The atoms this defect asserts: every pair for a cluster, the id itself otherwise."""
        if self.unit == EXAMPLE:
            return set(self.members)
        return {frozenset(pair) for pair in combinations(sorted(self.members), 2)}


@dataclass(frozen=True)
class GroundTruth:
    """Every planted defect of one fixture."""

    dataset_id: str
    defects: tuple[Planted, ...] = ()

    def scorable(self) -> tuple[Planted, ...]:
        return tuple(defect for defect in self.defects if defect.scorable)

    def _planted(self) -> tuple[Planted, ...]:
        """Planted defects that assert something. Negative controls assert the opposite."""
        return tuple(d for d in self.defects if d.channel != NEGATIVE_CONTROL)

    def explained_atoms(self, unit: str) -> set[frozenset[str] | str]:
        """Every atom of ``unit`` the ground truth can account for, *whichever* section records it.

        Not the same as a channel's own planted set, and the difference is the point. A prediction
        absent from its channel's ground truth may still be a defect the fixture records elsewhere:
        the n-gram check flags ``qa-002``, planted as half of an ``exact_duplicates`` pair; the
        retrieval check flags ``ret-001``/``ret-002``, planted as a ``duplicate_documents`` pair.
        Those are corroborations across channels, not inventions, and a scorer that called them
        false positives would be scoring the ground truth's filing system.

        The projections are unit-preserving, which is what stops this from becoming a blanket
        pardon. An example atom is explained if that example is named anywhere. A **pair** atom is
        explained only if that exact pair is planted somewhere — so the near stage's implied
        ``qa-004 ≈ qa-012``, whose endpoints are each planted but whose pairing is not, stays
        unexplained.
        """
        if unit == EXAMPLE:
            return {
                member.split("/", 1)[0] if defect.unit == DOC else member
                for defect in self._planted()
                for member in defect.members
            }
        if unit == DOC:
            return {atom for d in self._planted() if d.unit == DOC for atom in d.units()}

        pairs: set[frozenset[str] | str] = {
            atom for d in self._planted() if d.unit == PAIR for atom in d.units()
        }
        for defect in self._planted():
            if defect.unit != DOC:
                continue
            owners = {member.split("/", 1)[0] for member in defect.members}
            pairs |= {frozenset(pair) for pair in combinations(sorted(owners), 2)}
        return pairs


def _doc_refs(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """``example``/``doc_id`` pairs of a document-level defect, as ``example/doc_id`` strings."""
    if "example_docs" in entry:
        return tuple(f"{d['example']}/{d['doc_id']}" for d in entry["example_docs"])
    if "doc_ids" in entry:
        return tuple(f"{entry['example']}/{doc_id}" for doc_id in entry["doc_ids"])
    raise ValueError(f"document defect names no documents: {entry!r}")


def _members(section: str, unit: str, entry: Mapping[str, Any]) -> tuple[str, ...]:
    if unit == DOC:
        return _doc_refs(entry)
    if "ids" in entry:
        return tuple(str(value) for value in entry["ids"])
    if "id" in entry:
        return (str(entry["id"]),)
    raise ValueError(f"{section} defect names no examples: {entry!r}")


def parse_ground_truth(defects: Mapping[str, Any]) -> GroundTruth:
    """Parse a loaded ``defects.yaml`` mapping. Raises on any kind this module cannot place."""
    planted: list[Planted] = []
    for section, entries in defects.get("planted_defects", {}).items():
        for entry in entries:
            kind = entry.get("kind")
            key = (section, kind) if (section, kind) in CHANNEL_OF else (section, None)
            if key not in CHANNEL_OF:
                raise ValueError(
                    f"unmapped ground-truth defect {section}/{kind!r} in "
                    f"{defects.get('dataset_id', '?')}: add it to scoring.CHANNEL_OF. Skipping "
                    f"it would drop a planted defect out of the recall denominator."
                )
            channel, unit = CHANNEL_OF[key]
            planted.append(
                Planted(
                    section=section,
                    kind=str(kind or section),
                    channel=channel,
                    unit=unit,
                    members=_members(section, unit, entry),
                    auto_detectable=bool(entry.get("auto_detectable", True)),
                    note=str(entry.get("note", "")),
                )
            )
    return GroundTruth(dataset_id=str(defects.get("dataset_id", "")), defects=tuple(planted))


def load_ground_truth(path: str | Path) -> GroundTruth:
    """Load and parse a fixture's ``defects.yaml``."""
    return parse_ground_truth(loader.load_defects(path))


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #


def _pairs(clusters: Iterable[Sequence[str]]) -> set[frozenset[str]]:
    """Decompose clusters into the assertions they make: ``k`` members, ``C(k,2)`` pairs."""
    return {
        frozenset(pair)
        for cluster in clusters
        for pair in combinations(sorted(cluster), 2)
    }


def _doc_pairs(clusters: Iterable[Sequence[str]]) -> set[frozenset[str]]:
    """Same, over ``example/doc_id`` refs. ``LeakageReport`` renders them ``ex/doc[split]``."""
    return _pairs([[ref.split("[", 1)[0] for ref in cluster] for cluster in clusters])


def predictions(report: AuditReport) -> dict[str, set]:
    """Every channel's assertions, as atoms of that channel's unit.

    Read off the *stage* reports rather than the ``AuditResult``, because two channels the ground
    truth names — duplicate source groups and duplicate documents — have no field in the audit
    schema. An audit is what a release publishes; this is what a scorer needs.
    """
    dedup, leakage, labels = report.dedup, report.leakage, report.labels
    return {
        "dedup.exact": _pairs(f.example_ids for f in dedup.exact),
        "dedup.normalized": _pairs(f.example_ids for f in dedup.normalized),
        "dedup.near": _pairs(f.example_ids for f in dedup.near),
        "dedup.duplicate_sources": _pairs(dedup.duplicate_source_ids.values()),
        "leakage.train_test_overlap": _pairs(f.example_ids for f in leakage.train_test_overlap),
        "leakage.answer_in_prompt": {
            eid for f in leakage.answer_in_prompt for eid in f.example_ids
        },
        "leakage.ngram_overlap": {eid for f in leakage.ngram_overlap for eid in f.example_ids},
        "leakage.retrieval_contamination": {
            eid for f in leakage.retrieval_contamination for eid in f.example_ids
        },
        "leakage.duplicate_documents": _doc_pairs(leakage.duplicate_documents),
        "labels.missing": {f.example_id for f in labels.missing},
        "labels.conflicting": {f.example_id for f in labels.conflicting},
        "labels.ambiguous": {f.example_id for f in labels.ambiguous},
        "labels.inconsistent_format": {f.example_id for f in labels.inconsistent_format},
    }


def degenerate_predictions(examples: Sequence[DatasetExample]) -> dict[str, set]:
    """What a detector that flags *everything* predicts. The control this scorer is calibrated on.

    Not a straw man: it is the detector that maximizes recall, and a scoring rule that cannot
    separate it from a real one is not measuring detection. Every channel asserts every atom of
    its unit — all ``C(n,2)`` example pairs, all ``n`` examples, all ``C(d,2)`` document pairs.
    """
    from eval_dataset_forge.leakage.retrieval_overlap import retrieval_docs

    ids = [example.example_id for example in examples]
    every_doc = [
        f"{example.example_id}/{doc.get('doc_id', '?')}"
        for example in examples
        for doc in retrieval_docs(example)
    ]
    return {
        **{channel: _pairs([ids]) for channel in PAIR_CHANNELS},
        **{channel: set(ids) for channel in EXAMPLE_CHANNELS},
        **{channel: _pairs([every_doc]) for channel in DOC_CHANNELS},
    }


def _corroboration(predicted: Mapping[str, set]) -> dict[str, set]:
    """Every atom asserted by *any* channel of each unit type.

    A planted defect missed by its own channel may be recovered by an independent one — the
    ``ret-001``/``ret-002`` paraphrase is invisible to every lexical prompt threshold and is found
    anyway, because both examples cite the same gold passage. Agreement between independent
    channels is corroboration, so the report states recovery twice: by the defect's own channel,
    and by any channel at all.
    """
    pair_atoms: set = set()
    for channel in PAIR_CHANNELS:
        pair_atoms |= predicted.get(channel, set())
    # A shared passage across two examples is an example-level pair claim, stated in doc units.
    for atom in predicted.get("leakage.duplicate_documents", set()):
        owners = {ref.split("/", 1)[0] for ref in atom}
        if len(owners) == 2:
            pair_atoms.add(frozenset(owners))

    example_atoms: set = set()
    for channel in EXAMPLE_CHANNELS:
        example_atoms |= predicted.get(channel, set())

    doc_atoms: set = set()
    for channel in DOC_CHANNELS:
        doc_atoms |= predicted.get(channel, set())
    return {PAIR: pair_atoms, EXAMPLE: example_atoms, DOC: doc_atoms}


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChannelScore:
    """One channel's recall and precision, on the unit that channel asserts.

    Two different questions get two different numbers, because one metric cannot answer both:

    ``precision``
        Strict and kind-specific — of the atoms this channel asserted, how many are planted
        *as this kind of defect*? This is the number that collapses for a detector that flags
        everything, and it is ``None`` when the fixture plants nothing of this kind, because
        precision against an empty planted set is ``0.0`` for any non-silent detector and
        therefore measures nothing.
    ``unexplained``
        The atoms the ground truth cannot account for *anywhere*. An assertion here is either a
        genuine false positive or a defect nobody wrote down. It is the number to look at when
        ``precision`` is ``None`` or when a channel fires on a fixture that plants none of its
        kind — the honest-null-result case.
    """

    channel: str
    unit: str
    planted: int
    recovered: int
    predicted: int
    unexplained: tuple[str, ...] = ()
    corroborating: tuple[str, ...] = ()
    missed: tuple[str, ...] = ()

    @property
    def recall(self) -> float | None:
        """``None`` when the fixture plants nothing for this channel — not ``0.0``, not ``1.0``."""
        return self.recovered / self.planted if self.planted else None

    @property
    def precision(self) -> float | None:
        """Kind-specific precision. ``None`` when nothing of this kind is planted."""
        if not self.planted or not self.predicted:
            return None
        return self.recovered / self.predicted

    @property
    def explained(self) -> float | None:
        """Fraction of asserted atoms the ground truth accounts for, in any section.

        A **lower bound** on true precision: ``defects.yaml`` records what was planted, not an
        exhaustive labelling of every pair in the fixture.
        """
        if not self.predicted:
            return None
        return (self.predicted - len(self.unexplained)) / self.predicted

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "unit": self.unit,
            "planted": self.planted,
            "recovered": self.recovered,
            "predicted": self.predicted,
            "recall": self.recall,
            "precision": self.precision,
            "explained": self.explained,
            "unexplained": list(self.unexplained),
            "corroborating": list(self.corroborating),
            "missed": list(self.missed),
        }


@dataclass(frozen=True)
class DefectOutcome:
    """What became of one planted defect."""

    defect: Planted
    status: str

    def to_dict(self) -> dict:
        return {
            "section": self.defect.section,
            "kind": self.defect.kind,
            "channel": self.defect.channel,
            "members": list(self.defect.members),
            "status": self.status,
            "note": self.defect.note,
        }


@dataclass
class ScoreReport:
    """An audit scored against one fixture's planted ground truth."""

    dataset_id: str
    channels: dict[str, ChannelScore] = field(default_factory=dict)
    outcomes: tuple[DefectOutcome, ...] = ()

    # -- derived views ----------------------------------------------------- #

    def _with(self, *statuses: str) -> tuple[DefectOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status in statuses)

    @property
    def planted_total(self) -> int:
        """Every planted defect, including the ones excluded from the recall denominator."""
        return len(self.outcomes)

    @property
    def detectable(self) -> tuple[DefectOutcome, ...]:
        return self._with(RECOVERED, RECOVERED_ELSEWHERE, MISSED)

    @property
    def recovered_own_channel(self) -> tuple[DefectOutcome, ...]:
        return self._with(RECOVERED)

    @property
    def recovered_any_channel(self) -> tuple[DefectOutcome, ...]:
        return self._with(RECOVERED, RECOVERED_ELSEWHERE)

    @property
    def missed(self) -> tuple[DefectOutcome, ...]:
        return self._with(MISSED)

    @property
    def excluded(self) -> tuple[DefectOutcome, ...]:
        """Planted defects kept out of the recall denominator, and why. Never silently dropped."""
        return self._with(NOT_AUTO_DETECTABLE, UNWIRED)

    @property
    def controls_violated(self) -> tuple[DefectOutcome, ...]:
        return self._with(CONTROL_VIOLATED)

    def recall(self, *, any_channel: bool = False) -> float | None:
        """Recall over the *detectable* planted defects. ``None`` if the fixture plants none."""
        denominator = len(self.detectable)
        if not denominator:
            return None
        numerator = len(self.recovered_any_channel if any_channel else self.recovered_own_channel)
        return numerator / denominator

    def recall_over_all_planted(self) -> float | None:
        """Recall with *nothing* excluded from the denominator.

        The number a reader gets by refusing to accept the scorer's scope. Printed beside
        :meth:`recall` so that the excluded defects can be checked rather than trusted.
        """
        if not self.planted_total:
            return None
        controls = len(self._with(CONTROL_HELD, CONTROL_VIOLATED))
        denominator = self.planted_total - controls
        if not denominator:
            return None
        return len(self.recovered_any_channel) / denominator

    def counts(self) -> dict[str, int]:
        return {
            "planted": self.planted_total,
            "detectable": len(self.detectable),
            "recovered": len(self.recovered_own_channel),
            "recovered_any_channel": len(self.recovered_any_channel),
            "missed": len(self.missed),
            "excluded": len(self.excluded),
            "controls": len(self._with(CONTROL_HELD, CONTROL_VIOLATED)),
            "controls_violated": len(self.controls_violated),
        }

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "counts": self.counts(),
            "recall": self.recall(),
            "recall_any_channel": self.recall(any_channel=True),
            "recall_over_all_planted": self.recall_over_all_planted(),
            "channels": {name: score.to_dict() for name, score in sorted(self.channels.items())},
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


def _atom_label(atom: frozenset[str] | str) -> str:
    return " + ".join(sorted(atom)) if isinstance(atom, frozenset) else str(atom)


def _unit_of(channel: str) -> str:
    """The unit a channel's assertions live in. Unknown channels are pairs by convention."""
    for channels, unit in ((PAIR_CHANNELS, PAIR), (EXAMPLE_CHANNELS, EXAMPLE), (DOC_CHANNELS, DOC)):
        if channel in channels:
            return unit
    return PAIR


def score_predictions(predicted: Mapping[str, set], truth: GroundTruth) -> ScoreReport:
    """Score a channel → atoms mapping against planted ground truth.

    Split out from :func:`score` so the same rule scores the real detector and the flag-everything
    control. A scoring rule that is only ever run on the detector it was written for has not been
    tested.
    """
    corroborated = _corroboration(predicted)

    outcomes: list[DefectOutcome] = []
    for defect in truth.defects:
        if defect.channel == NEGATIVE_CONTROL:
            channels = CONTROL_CHANNELS.get(defect.kind, ())
            flagged = any(
                atom in predicted.get(channel, set())
                for channel in channels
                for atom in defect.units()
            )
            outcomes.append(
                DefectOutcome(defect, CONTROL_VIOLATED if flagged else CONTROL_HELD)
            )
        elif not defect.auto_detectable:
            outcomes.append(DefectOutcome(defect, NOT_AUTO_DETECTABLE))
        elif defect.channel == NO_CHANNEL:
            outcomes.append(DefectOutcome(defect, UNWIRED))
        else:
            atoms = defect.units()
            own = predicted.get(defect.channel, set())
            if atoms <= own:
                status = RECOVERED
            elif atoms <= corroborated[defect.unit]:
                status = RECOVERED_ELSEWHERE
            else:
                status = MISSED
            outcomes.append(DefectOutcome(defect, status))

    channels: dict[str, ChannelScore] = {}
    for channel, atoms in sorted(predicted.items()):
        unit = _unit_of(channel)
        planted_here = {
            atom
            for defect in truth.defects
            if defect.channel == channel and defect.scorable
            for atom in defect.units()
        }
        explained = truth.explained_atoms(unit)
        channels[channel] = ChannelScore(
            channel=channel,
            unit=unit,
            planted=len(planted_here),
            recovered=len(planted_here & atoms),
            predicted=len(atoms),
            unexplained=tuple(sorted(_atom_label(atom) for atom in atoms - explained)),
            corroborating=tuple(
                sorted(_atom_label(atom) for atom in (atoms & explained) - planted_here)
            ),
            missed=tuple(sorted(_atom_label(atom) for atom in planted_here - atoms)),
        )

    return ScoreReport(dataset_id=truth.dataset_id, channels=channels, outcomes=tuple(outcomes))


def score(report: AuditReport, truth: GroundTruth) -> ScoreReport:
    """Score an :class:`~eval_dataset_forge.audit.AuditReport` against planted ground truth."""
    return score_predictions(predictions(report), truth)


def score_degenerate(examples: Sequence[DatasetExample], truth: GroundTruth) -> ScoreReport:
    """Score the flag-everything detector under the identical rule. The metric's own control."""
    return score_predictions(degenerate_predictions(examples), truth)


__all__ = [
    "CHANNEL_OF",
    "CONTROL_CHANNELS",
    "CONTROL_HELD",
    "CONTROL_VIOLATED",
    "DOC",
    "DOC_CHANNELS",
    "EXAMPLE",
    "EXAMPLE_CHANNELS",
    "MISSED",
    "NEGATIVE_CONTROL",
    "NOT_AUTO_DETECTABLE",
    "NO_CHANNEL",
    "PAIR",
    "PAIR_CHANNELS",
    "RECOVERED",
    "RECOVERED_ELSEWHERE",
    "UNWIRED",
    "ChannelScore",
    "DefectOutcome",
    "GroundTruth",
    "Planted",
    "ScoreReport",
    "degenerate_predictions",
    "load_ground_truth",
    "parse_ground_truth",
    "predictions",
    "score",
    "score_degenerate",
    "score_predictions",
]
