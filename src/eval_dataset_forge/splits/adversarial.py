"""The adversarial strategy: a test set that cannot be answered from memory (Milestone 7).

Composes every constraint the earlier stages earned the right to impose:

1. **Duplicate clusters are blocks.** Edges from the full deduplication ladder —
   exact, normalized, and heuristic near — plus, by default, the ``duplicate_sources``
   groups. Nothing that Milestone 4 called the same item is cut in half.
2. **A duplicate block may not enter an evaluation split.** ``dev`` and ``test`` are
   both evaluation splits: tuning on an item and then scoring on its twin is the same
   defect as training on it. Duplicate blocks land in ``train``.
3. **Label-blocking examples may not enter an evaluation split.** Milestone 6's
   ``blocking_example_ids`` names examples whose gold answer no automated check can
   resolve — missing, conflicting, or ambiguous. Scoring a system against a label that
   needs adjudication measures the label, not the system. Passed in as
   ``quarantine_ids``; they stay in the dataset and stay in ``train``.
4. **Strata are preserved** where the first three constraints leave room.

Constraint 2 is stronger than "no cluster straddles train and test". Both readings
kill cross-split leakage, but only this one keeps the *test set itself* free of
correlated items: two paraphrases sitting together in test are one effective example
wearing two hats, and every bootstrap interval computed over them is too narrow.

**What this guarantees, exactly.** After an adversarial build,
:func:`~eval_dataset_forge.leakage.cross_split_clusters` returns nothing, and no
evaluation example belongs to a duplicate cluster. Both hold *by construction* — the
blocks are the clusters — so the guarantee is only as good as the clusters, and the
clusters are only as good as the near stage's recall. ``ret-001``/``ret-002`` is a
planted cross-split paraphrase that the ladder provably cannot see at any lexical
threshold (``docs/DEDUPLICATION.md``); this strategy therefore cannot see it either,
and will straddle it exactly as random would. A guarantee inherited from a heuristic
is a heuristic guarantee. ``tests/test_splits.py`` pins the miss so it cannot be
quietly upgraded to a promise.

**What it costs.** Duplicate blocks and quarantined examples all land in ``train``,
so the evaluation splits are drawn from a smaller pool than the ratios ask for and the
achieved ratios drift. On the pooled 56-example corpus that is 24 of 56 examples
pinned to ``train`` before a single quota is consulted. The report shows the drift
instead of forcing the ratio.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from eval_dataset_forge.dedup import DedupReport, find_duplicates
from eval_dataset_forge.schemas import DatasetExample, Split
from eval_dataset_forge.splits.assignment import (
    DEFAULT_SEED,
    SPLIT_ORDER,
    Assignment,
    EligibleFor,
    assign_blocks,
)
from eval_dataset_forge.splits.blocks import Block, duplicate_blocks
from eval_dataset_forge.splits.stratified import DEFAULT_STRATIFY_BY, block_stratum

# Splits used to select or score a system. A duplicate must reach neither.
DEFAULT_EVAL_SPLITS: tuple[str, ...] = (str(Split.DEV), str(Split.TEST))


def _eligibility(eval_splits: Sequence[str], quarantine: frozenset[str]) -> EligibleFor:
    blocked = tuple(name for name in SPLIT_ORDER if name not in eval_splits)

    def eligible_for(block: Block) -> tuple[str, ...]:
        quarantined = any(member in quarantine for member in block.example_ids)
        return blocked if (block.duplicate or quarantined) else SPLIT_ORDER

    return eligible_for


def adversarial_splits(
    examples: Iterable[DatasetExample],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    stratify_by: Sequence[str] = DEFAULT_STRATIFY_BY,
    dedup_report: DedupReport | None = None,
    eval_splits: Sequence[str] = DEFAULT_EVAL_SPLITS,
    quarantine_ids: Iterable[str] = (),
    include_sources: bool = True,
) -> Assignment:
    """Build splits that keep duplicate clusters intact and out of evaluation.

    ``dedup_report`` lets the audit pipeline reuse the ladder it already ran; omit it
    and one is computed with the package defaults. Retrieval-style datasets, whose
    gold answers are document ids, should pass a report built with
    ``near_require_same_answer=False`` — otherwise the near stage's answer gate hides
    their paraphrases from the block builder, exactly as it hides them from the audit.

    ``eval_splits=("test",)`` relaxes constraint 2 to protect only the test split,
    which recovers ratio fidelity at the cost of duplicates in ``dev``.
    """
    examples = list(examples)
    if dedup_report is None:
        dedup_report = find_duplicates(examples)

    by_id = {example.example_id: example for example in examples}
    blocks = duplicate_blocks(examples, dedup_report, include_sources=include_sources)

    return assign_blocks(
        blocks,
        ratios=ratios,
        seed=seed,
        stratum_of=lambda block: block_stratum(block, by_id, stratify_by),
        eligible_for=_eligibility(tuple(eval_splits), frozenset(quarantine_ids)),
    )


__all__ = ["DEFAULT_EVAL_SPLITS", "adversarial_splits"]
