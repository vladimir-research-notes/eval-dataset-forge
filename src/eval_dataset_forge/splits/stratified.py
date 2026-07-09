"""Stratification: keep each split representative (Milestone 7).

A test set that happens to be easier than the training set makes every system look
better than it is; a test set missing a task family measures nothing about it.
Stratification constrains the assigner to fill each split's quota *within* every
stratum, so the marginal distribution of the stratifying facet is preserved
(up to the rounding that indivisible blocks and small strata force).

Two facets ship, both read straight off the example:

``task_family``
    Exact. The field is required and open-vocabulary.

``difficulty``
    The example's **declared** ``difficulty_tags`` filtered to
    :class:`~eval_dataset_forge.schemas.DifficultyBucket`. Examples carrying no
    bucket tag form their own ``unlabelled`` stratum rather than being imputed. This
    is deliberately the *declared* difficulty, not a modelled one: the calibrated
    difficulty score lands in Milestone 8, and stratifying on a score this module
    computed itself would let a difficulty heuristic quietly define the test set it
    is later evaluated on.

**Stratify on one facet, not two.** The default is ``("task_family",)``, and the
reason is measured rather than assumed. Strata multiply when facets are combined while
the corpus does not: five families × four difficulty levels gives 17 non-empty strata
over 56 pooled examples, a mean of 3.3 each. A stratum of three cannot fill a ``dev``
split at ratio ``0.1`` — its whole quota rounds to zero — so the small splits are
starved by *every* stratum at once, and the marginal that stratification was meant to
preserve is preserved worse than if one facet had been used alone:

======================  ========  ==============  ==================  =========
``stratify_by``         strata    TVD family ↓    TVD difficulty ↓    dev size
======================  ========  ==============  ==================  =========
(none)                  1         0.216           0.156               6.0
**task_family**         5         **0.036**       0.152               5.0
difficulty              4         0.184           **0.030**           5.0
task_family,difficulty  17        0.083           0.077               3.0
======================  ========  ==============  ==================  =========

(Total variation distance between the test split's distribution and the corpus's,
mean over 20 seeds, pooled corpus, ratios ``0.7/0.1/0.2``.) Joint stratification is
**worse on both facets** than the single-facet stratification of either. It also
guarantees an empty cell: the corpus's three ``hard`` examples sit in three different
family strata, each of size one, and a stratum of one apportions entirely to ``train``
— so no ``hard`` example reaches the test set at all. :func:`~.assignment.assign_blocks`
warns when the average stratum cannot fill the smallest split, and ``build_splits``
warns for each facet value missing from ``test``.

Stratify on both only when N per joint cell is large enough to fill every split.

**A block can span strata.** ``qa-003`` (``easy``) and ``qa-012`` (``medium``) sit in
one adversarial block, so the block belongs to no single stratum. :func:`block_stratum`
resolves it by plurality — ties by the stratum's own sort order — which means a mixed
block distorts the facet it was counted under by exactly its minority members. The
alternative, cutting the block, would reintroduce the leakage the block exists to
prevent. The report's achieved distribution shows the residue rather than hiding it.

**Stratifying by ``task_family`` is a no-op on every committed fixture**, because each
fixture holds exactly one family. It only bites on the pooled 56-example corpus.
``docs/SPLITS.md`` says so, and a test asserts it, so the strategy is never credited
with work it did not do.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from eval_dataset_forge.schemas import DatasetExample, DifficultyBucket
from eval_dataset_forge.splits.assignment import DEFAULT_SEED, Assignment, assign_blocks
from eval_dataset_forge.splits.blocks import Block, singleton_blocks

# Increasing order of difficulty. An example tagged with more than one bucket is
# stratified as the hardest of them: a benchmark that mislabels a hard item as easy
# hurts more than the reverse.
DIFFICULTY_ORDER: tuple[str, ...] = (
    str(DifficultyBucket.EASY),
    str(DifficultyBucket.MEDIUM),
    str(DifficultyBucket.HARD),
)

# Stratum name for an example carrying no bucket tag. Not imputed — see the docstring.
UNLABELLED = "unlabelled"

# Facets `stratum_of` understands, and always reports the achieved distribution of.
FACETS: tuple[str, ...] = ("task_family", "difficulty")

# One facet by default. Combining the two multiplies the strata without multiplying the
# corpus, and measures worse on both facets — see the module docstring's table.
DEFAULT_STRATIFY_BY: tuple[str, ...] = ("task_family",)

_FACET_SEPARATOR = "|"


def declared_bucket(example: DatasetExample) -> DifficultyBucket | None:
    """The hardest :class:`DifficultyBucket` declared in ``difficulty_tags``, if any.

    Tags outside the bucket vocabulary (``coding``, ``ambiguous``, …) are ignored:
    they describe the example, not its difficulty.
    """
    tags = {tag.lower() for tag in example.difficulty_tags}
    declared = [name for name in DIFFICULTY_ORDER if name in tags]
    return DifficultyBucket(declared[-1]) if declared else None


def facet_value(example: DatasetExample, facet: str) -> str:
    """One facet of one example, as a stratum component."""
    if facet == "task_family":
        return example.task_family
    if facet == "difficulty":
        bucket = declared_bucket(example)
        return str(bucket) if bucket else UNLABELLED
    raise ValueError(f"unknown stratification facet {facet!r} (known: {list(FACETS)})")


def stratum_of(example: DatasetExample, by: Sequence[str] = DEFAULT_STRATIFY_BY) -> str:
    """The stratum name of one example, e.g. ``"qa|easy"``. Empty ``by`` → one stratum."""
    return _FACET_SEPARATOR.join(facet_value(example, facet) for facet in by)


def block_stratum(
    block: Block,
    examples_by_id: dict[str, DatasetExample],
    by: Sequence[str] = DEFAULT_STRATIFY_BY,
) -> str:
    """The stratum of an indivisible block: the plurality stratum of its members.

    Ties break on the stratum name so the choice is deterministic and independent of
    member order. A singleton block trivially takes its own example's stratum.
    """
    counts: dict[str, int] = {}
    for example_id in block.example_ids:
        stratum = stratum_of(examples_by_id[example_id], by)
        counts[stratum] = counts.get(stratum, 0) + 1
    return min(counts, key=lambda stratum: (-counts[stratum], stratum))


def stratified_splits(
    examples: Iterable[DatasetExample],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    stratify_by: Sequence[str] = DEFAULT_STRATIFY_BY,
) -> Assignment:
    """Assign examples independently, filling each split's quota within every stratum.

    Blocks are singletons — stratification constrains *where* an example may go, not
    *with whom* it travels — so this strategy preserves the facet distribution and
    nothing else. A near-duplicate pair in the same stratum still straddles.
    Composing both guarantees is :func:`~.adversarial.adversarial_splits`.
    """
    examples = list(examples)
    by_id = {example.example_id: example for example in examples}
    return assign_blocks(
        singleton_blocks(examples),
        ratios=ratios,
        seed=seed,
        stratum_of=lambda block: block_stratum(block, by_id, stratify_by),
    )


def distribution(
    examples: Iterable[DatasetExample], assignment: dict[str, str], facet: str
) -> dict[str, dict[str, int]]:
    """``split → facet value → count`` for the built splits. The achieved distribution."""
    table: dict[str, dict[str, int]] = {}
    for example in examples:
        split = assignment.get(example.example_id)
        if split is None:
            continue
        counts = table.setdefault(split, {})
        value = facet_value(example, facet)
        counts[value] = counts.get(value, 0) + 1
    return {split: dict(sorted(counts.items())) for split, counts in sorted(table.items())}


__all__ = [
    "DEFAULT_STRATIFY_BY",
    "DIFFICULTY_ORDER",
    "FACETS",
    "UNLABELLED",
    "block_stratum",
    "declared_bucket",
    "distribution",
    "facet_value",
    "stratified_splits",
    "stratum_of",
]
