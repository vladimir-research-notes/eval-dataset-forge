"""Integration contracts: what a downstream project may assume about a release.

Milestone 13. Five adjacent projects — ``code-agent-runtime``, ``BioAgentBench``, an LLM
regression/observability suite, ``scientific-claim-verifier``, and ``agent-control-bench``
— want to consume audited releases. None of them is a dependency of this package, and none
of them exists in this repository. What is committed here is the *shape* of the question
each one asks, made executable so it can be tested against the fixtures.

Three questions, three identities
---------------------------------

Milestone 10 gave a release three identities because a file hash answers the wrong
question. Each one turns out to be the scope of exactly one consumer-side claim:

======================  ===============================================================
``content_id``          the scope of a **gate verdict**. Duplicate, leakage, label, and
                        PII findings are computed over a corpus, so a verdict about them
                        is a statement about that corpus and no other.
``eval_id``             the scope of a **score**. Two numbers are comparable iff the
                        examples *and their split assignment* agree.
``dependence.``         the scope of a **rebuild**. The near-duplicate detector that ran
``near_method``         determines the blocks, hence the adversarial split, hence
                        ``eval_id``.
======================  ===============================================================

The third row is the one this milestone had to discover. ``near_method`` resolves to
``tfidf`` if and only if scikit-learn is importable, and the two methods disagree:
rebuilding ``bio_toy`` from identical bytes, an identical command, and an identical seed
produces a **different** ``eval_id`` on a machine without the "optional accelerator".
``REPRODUCE.md`` used to say "deterministic given the same inputs and seed". The
environment is an input.

Two properties that do not hold
-------------------------------

**A gate verdict is not preserved under union.** ``safety_toy`` alone reports one
near-duplicate cluster; concatenated with four fixtures that share no duplicate with it,
the same ten rows report two. Merging two gated releases yields an ungated corpus, and
:meth:`~.gate.GateReport.applies_to` refuses to pretend otherwise.

**Independent-block count is not additive, although example count is.** The five fixtures
hold 45 blocks apart and 44 pooled. A regression suite that doubles its example count has
not necessarily bought ``√2`` of anything.

Both are consequences of idf being a property of the collection. Under the pure-Python
``jaccard`` fallback both properties hold — which means the optional dependency changes
the *semantics* of a duplicate report, not its speed. ``docs/INTEGRATION.md`` has the
measurements.

What no contract here can supply
--------------------------------

Whether the model under test was trained on the benchmark. Contamination is measurable
only against text the auditor holds.
"""

from __future__ import annotations

from eval_dataset_forge.contracts.gate import (
    CONTRACT_VERSION,
    GatePolicy,
    GateReport,
    evaluate_gate,
    gate_release,
)
from eval_dataset_forge.contracts.profiles import PROFILES, Profile, get_profile
from eval_dataset_forge.contracts.reader import (
    AUDIT_FIELDS,
    DEPENDENCE_FIELDS,
    MANIFEST_FIELDS,
    SPLIT_COUNT_FIELDS,
    SPLIT_STAT_FIELDS,
    ContractViolation,
    ReleaseHandle,
)
from eval_dataset_forge.contracts.report import format_gate, format_profiles, gate_document
from eval_dataset_forge.contracts.requirements import (
    DEFAULT_THRESHOLDS,
    REQUIREMENTS,
    Requirement,
    RequirementResult,
    Verdict,
    evaluate,
    resolve_thresholds,
)

__all__ = [
    "AUDIT_FIELDS",
    "CONTRACT_VERSION",
    "DEFAULT_THRESHOLDS",
    "DEPENDENCE_FIELDS",
    "MANIFEST_FIELDS",
    "PROFILES",
    "REQUIREMENTS",
    "SPLIT_COUNT_FIELDS",
    "SPLIT_STAT_FIELDS",
    "ContractViolation",
    "GatePolicy",
    "GateReport",
    "Profile",
    "ReleaseHandle",
    "Requirement",
    "RequirementResult",
    "Verdict",
    "evaluate",
    "evaluate_gate",
    "format_gate",
    "format_profiles",
    "gate_document",
    "gate_release",
    "get_profile",
    "resolve_thresholds",
]
