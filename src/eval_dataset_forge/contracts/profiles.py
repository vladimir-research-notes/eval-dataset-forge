"""Consumer profiles: the five integration contracts of ``docs/PLAN.md`` §10 row 13.

Each profile names a downstream project, the requirements it turns into a build gate, and
— at equal weight — the two lists nothing else in this repository writes down:

``assumes``
    What the consumer must establish for itself. A gate that stayed silent here would be
    read as a warrant of fitness, which it is not.
``does_not_check``
    Requirements deliberately left out, with the reason. An omission that goes unstated
    is indistinguishable from an oversight.

**No profile imports the project it is named for, and none of them exists in this
repository.** A contract that depended on its consumer would be a coupling, not a
contract. What is committed here is the *shape* of the question each consumer asks —
executable, testable against the fixtures, and stable under :data:`CONTRACT_VERSION`.

The thresholds are conventions. ``min_eval_blocks = 30`` is ``MIN_EFFECTIVE_N`` from
Milestone 9 and is not a theorem; a consumer that disagrees writes a policy file rather
than editing this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Profile:
    """One consumer's contract with a release."""

    name: str
    consumer: str
    purpose: str
    requirements: tuple[str, ...]
    thresholds: dict[str, Any] = field(default_factory=dict)
    assumes: tuple[str, ...] = ()
    does_not_check: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "consumer": self.consumer,
            "purpose": self.purpose,
            "requirements": list(self.requirements),
            "thresholds": dict(self.thresholds),
            "assumes": list(self.assumes),
            "does_not_check": list(self.does_not_check),
        }


_PROFILES = (
    Profile(
        name="code-agent-runtime",
        consumer="code-agent-runtime",
        purpose="Audit an agent-task benchmark before it gates a run.",
        requirements=(
            "integrity",
            "schema_compatible",
            "no_missing_labels",
            "no_answer_in_prompt",
            "no_exact_duplicates",
            "eval_splits_deduplicated",
        ),
        assumes=(
            "The tasks are executable in the runtime's sandbox. Nothing in a release is "
            "run, so no artifact here distinguishes a task from a task that cannot start.",
            "The scoring rubric matches the gold answers. `no_missing_labels` asks whether "
            "a gold answer exists, never whether the harness can compare against it.",
        ),
        does_not_check=(
            "near_duplicates_within_budget — a paraphrased task is a legitimate second "
            "instance of a skill, and a runtime that refused to start on one would refuse "
            "on most real suites.",
            "eval_split_is_rankable — a run gate asks whether the tasks are sound, not "
            "whether the suite can separate two agents. That is `llm-regression`'s question.",
        ),
    ),
    Profile(
        name="bioagentbench",
        consumer="BioAgentBench",
        purpose="Audit synthetic biomedical evidence tasks before release.",
        requirements=(
            "integrity",
            "schema_compatible",
            "no_pii",
            "no_exact_duplicates",
            "no_label_conflicts",
            "no_missing_labels",
        ),
        assumes=(
            "The corpus is synthetic or public-safe. `no_pii` matches a curated pattern "
            "set; a passing release with real patient data in it passes.",
            "A domain expert has read the gold labels. Entity-level correctness is outside "
            "every check in this repository.",
        ),
        does_not_check=(
            "Clinical validity, in any form. This tool is not a medical authority "
            "(docs/SAFETY.md), and no arrangement of these requirements would make it one.",
            "no_retrieval_contamination — included only when the corpus carries passages; "
            "on a lookup-style bio set it would return UNDECIDABLE and block the gate for "
            "a reason unrelated to the data's quality.",
        ),
    ),
    Profile(
        name="llm-regression",
        consumer="LLM regression / observability",
        purpose="Refuse to rank models on a suite that is too small or too leaky to rank them.",
        requirements=(
            "integrity",
            "schema_compatible",
            "near_method_recorded",
            "no_train_test_overlap",
            "eval_splits_deduplicated",
            "eval_split_is_rankable",
        ),
        thresholds={"min_eval_blocks": 30, "eval_split": "test"},
        assumes=(
            "The models under test were not trained on this benchmark. No release can "
            "answer this: contamination is measurable only against text the auditor holds, "
            "and nobody holds a frontier model's training set.",
            "Two releases being compared share an `eval_id`. A regression suite that "
            "silently re-splits between runs is comparing two benchmarks.",
        ),
        does_not_check=(
            "The minimum detectable effect, which needs system scores and an intraclass "
            "correlation. `eval_split_is_rankable` bounds the sample size from below; "
            "`evalforge stats` computes the rest once scores exist.",
            "That `min_eval_blocks = 30` is enough. It is a convention. On a corpus with a "
            "3.6-point margin, Milestone 9 measured that ~4500 examples were needed.",
        ),
    ),
    Profile(
        name="scientific-claim-verifier",
        consumer="scientific-claim-verifier",
        purpose="Audit claim/evidence datasets for label conflicts and evidence leakage.",
        requirements=(
            "integrity",
            "schema_compatible",
            "no_label_conflicts",
            "no_answer_in_prompt",
            "no_retrieval_contamination",
            "no_exact_duplicates",
        ),
        assumes=(
            "The evidence passages support the claims they are attached to. Relevance is a "
            "label, and this gate checks labels for conflict, not for truth.",
        ),
        does_not_check=(
            "Whether a claim is scientifically correct. That is the system under test's "
            "job, and a dataset that encoded the answer would be the leakage this gate "
            "exists to find.",
            "Paraphrased evidence passages. `retrieval_toy`'s planted `d13`/`d14` pair is "
            "invisible to exact-after-normalization passage matching, and no lexical "
            "threshold separates it (docs/DEDUPLICATION.md).",
        ),
    ),
    Profile(
        name="agent-control-bench",
        consumer="agent-control-bench",
        purpose="Audit control / safety task sets before they measure refusal behavior.",
        requirements=(
            "integrity",
            "schema_compatible",
            "no_pii",
            "no_exact_duplicates",
            "no_answer_in_prompt",
            "eval_splits_deduplicated",
        ),
        assumes=(
            "The boundary cases were written by someone who understands the boundary. No "
            "check here reads a safety policy.",
        ),
        does_not_check=(
            "Ambiguous labels — deliberately. An over-refusal / under-refusal set exists to "
            "probe the ambiguous boundary, so `LabelIssue.AMBIGUOUS` is a feature of the "
            "corpus, not a defect. A profile requiring zero ambiguous labels would reject "
            "every well-constructed control benchmark, `safety_toy` included.",
            "Whether the task set covers the behaviors it claims to. Coverage is a claim "
            "about a taxonomy, and no artifact in a release carries one.",
        ),
    ),
)

PROFILES: dict[str, Profile] = {profile.name: profile for profile in _PROFILES}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown profile {name!r}; known: {known}")
    return PROFILES[name]


__all__ = ["PROFILES", "Profile", "get_profile"]
