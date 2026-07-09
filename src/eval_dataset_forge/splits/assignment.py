"""The assignment core shared by every split strategy (Milestone 7).

Two assigners, both deterministic, both operating on :class:`~.blocks.Block` units
rather than on examples, so an indivisible group can never be cut in half:

:func:`assign_blocks`
    Proportional allocation in a seeded random order. Backs random, stratified,
    source-grouped, and adversarial.

:func:`assign_in_order`
    Sequential allocation along a caller-supplied ordering. Backs temporal, where
    shuffling would destroy the only thing the strategy is about.

**Quotas are integers before any block moves.** :func:`apportion` turns a ratio vector
and a stratum size into whole-example targets by the largest-remainder rule: floor each
quota, then hand the leftovers to the largest fractional parts. Only then does the
greedy deal blocks to whichever split is furthest below its target. Rounding
*afterwards* — greedily chasing a float deficit — starves the small splits, because a
stratum of three examples always shows ``dev`` a deficit of ``0.3`` that no assignment
can ever close, and the greedy keeps trying: at ``n=3`` and ratios ``0.7/0.1/0.2`` it
lands on ``1/1/1``. Apportioning first gives ``2/0/1``, and the dropped ``dev`` example
becomes a warning rather than a silent theft from ``train``.

With singleton blocks the deal reproduces the apportioned targets exactly. With larger
blocks it cannot — three examples do not divide ``0.7/0.1/0.2`` — so blocks are dealt
largest-first, leaving the singletons to absorb the rounding error.

**Best-effort is a real behaviour, not a disclaimer.** The assigner never forces a
ratio and never drops an example. It records what it achieved and warns about the
specific quota it could not fill, and the caller reports both.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

from eval_dataset_forge.schemas import Split
from eval_dataset_forge.splits.blocks import Block

# Splits an assigner may fill, in the order ties are broken.
SPLIT_ORDER: tuple[str, ...] = (str(Split.TRAIN), str(Split.DEV), str(Split.TEST))

# Train-heavy default: enough dev to select on, enough test to measure on. Any
# ratios summing to 1.0 over a subset of SPLIT_ORDER are accepted.
DEFAULT_RATIOS: dict[str, float] = {"train": 0.7, "dev": 0.1, "test": 0.2}

DEFAULT_SEED = 0

# A quota below this rounds to zero examples, so leaving it unfilled is arithmetic,
# not a defect. At or above it, an empty split is worth a warning.
_QUOTA_WARN = 0.5

StratumOf = Callable[[Block], str]
EligibleFor = Callable[[Block], tuple[str, ...]]


@dataclass(frozen=True)
class Assignment:
    """The result of dealing blocks into splits.

    ``of_example`` is the flattened view the rest of the pipeline consumes;
    ``blocks`` and ``of_block`` keep the block-level decision, so a report can explain
    *why* an example landed where it did rather than only *that* it did.
    """

    blocks: tuple[Block, ...] = ()
    of_block: dict[str, str] = field(default_factory=dict)
    of_example: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def normalize_ratios(ratios: dict[str, float] | None) -> dict[str, float]:
    """Validate split ratios and drop the zero-weight splits.

    Rejects unknown split names, negative weights, and any set that does not sum to
    1.0 — the same contract as
    :class:`~eval_dataset_forge.schemas.SplitSpec`, enforced here too so a caller
    using the assigner directly cannot silently build a partial partition.
    """
    if ratios is None:
        ratios = DEFAULT_RATIOS
    unknown = set(ratios) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"unknown split name(s): {sorted(unknown)}")
    for name, ratio in ratios.items():
        if ratio < 0:
            raise ValueError(f"split ratio for {name!r} is negative: {ratio}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    return {name: ratios[name] for name in SPLIT_ORDER if ratios.get(name, 0.0) > 0.0}


def apportion(size: int, ratios: dict[str, float]) -> dict[str, int]:
    """Whole-example targets for ``size`` examples, by the largest-remainder rule.

    Floors every quota, then gives the leftover examples to the largest fractional
    parts, ties broken by :data:`SPLIT_ORDER`. The targets always sum to ``size``, and
    a split whose quota is below 1 gets 0 — which is the honest answer for a stratum of
    three examples and a ``dev`` ratio of ``0.1``, not a bug to round away.
    """
    targets = dict.fromkeys(ratios, 0)
    if size <= 0:
        return targets
    quotas = {name: ratio * size for name, ratio in ratios.items()}
    for name, quota in quotas.items():
        targets[name] = int(quota)
    # Ratios are validated to sum to 1.0 only within a tolerance, so at extreme sizes
    # the floors could in principle sum past `size`. Clamping keeps the invariant
    # "targets sum to size" unconditional; a negative slice would silently *add*.
    leftover = max(0, min(size - sum(targets.values()), len(ratios)))
    ranked = sorted(
        ratios,
        key=lambda name: (-(quotas[name] - targets[name]), SPLIT_ORDER.index(name)),
    )
    for name in ranked[:leftover]:
        targets[name] += 1
    return targets


def _all_splits(_block: Block) -> tuple[str, ...]:
    return SPLIT_ORDER


def _flatten(of_block: dict[str, str], blocks: Iterable[Block]) -> dict[str, str]:
    return {
        example_id: of_block[block.key]
        for block in blocks
        for example_id in block.example_ids
    }


def _quota_warnings(
    stratum: str, quotas: dict[str, float], filled: dict[str, int]
) -> list[str]:
    """Warn only where a split *should* have received an example and received none."""
    label = f"stratum {stratum!r}: " if stratum else ""
    return [
        f"{label}{name} quota is {quotas[name]:.1f} example(s) but the split is empty; "
        f"blocks are indivisible and could not fill it"
        for name in quotas
        if filled.get(name, 0) == 0 and quotas[name] >= _QUOTA_WARN
    ]


def assign_blocks(
    blocks: Sequence[Block],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    stratum_of: StratumOf | None = None,
    eligible_for: EligibleFor | None = None,
) -> Assignment:
    """Deal ``blocks`` into splits proportionally, in a seeded random order.

    The block list is shuffled once with ``Random(seed)`` and then partitioned into
    strata, so a stratum's internal order depends on the seed but not on which other
    strata exist. Within a stratum, blocks are dealt largest-first (a stable sort, so
    equal-size blocks keep their shuffled order) into whichever split is furthest below
    the integer target :func:`apportion` set for it.

    ``eligible_for`` restricts a block to a subset of splits — the mechanism behind
    the adversarial strategy's "no duplicate cluster in an evaluation split". A block
    with no eligible split falls back to ``train`` with a warning rather than being
    dropped: this builder never silently removes an example from the dataset.
    """
    ratios = normalize_ratios(ratios)
    stratum_of = stratum_of or (lambda _block: "")
    eligible_for = eligible_for or _all_splits

    shuffled = list(blocks)
    Random(seed).shuffle(shuffled)

    strata: dict[str, list[Block]] = {}
    for block in shuffled:
        strata.setdefault(stratum_of(block), []).append(block)

    of_block: dict[str, str] = {}
    warnings: list[str] = []
    total = sum(block.size for block in shuffled)
    if len(strata) > 1 and (total / len(strata)) * min(ratios.values()) < 1.0:
        smallest = min(ratios, key=lambda name: ratios[name])
        warnings.append(
            f"stratification produced {len(strata)} strata for {total} examples "
            f"(mean {total / len(strata):.1f} each); the average stratum cannot fill "
            f"{smallest!r} at ratio {ratios[smallest]:g}, so most strata will contribute "
            f"nothing to it"
        )

    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda block: -block.size)
        size = sum(block.size for block in members)
        targets = apportion(size, ratios)
        filled = dict.fromkeys(ratios, 0)

        for block in members:
            candidates = [name for name in eligible_for(block) if name in ratios]
            if not candidates:
                warnings.append(
                    f"block {block.key!r} is eligible for no split with a positive "
                    f"ratio; assigned to {SPLIT_ORDER[0]}"
                )
                candidates = [SPLIT_ORDER[0]]
            chosen = min(
                candidates,
                key=lambda name: (
                    filled.get(name, 0) - targets.get(name, 0),
                    SPLIT_ORDER.index(name),
                ),
            )
            of_block[block.key] = chosen
            filled[chosen] = filled.get(chosen, 0) + block.size

        quotas = {name: ratio * size for name, ratio in ratios.items()}
        warnings.extend(_quota_warnings(stratum, quotas, filled))

    return Assignment(
        blocks=tuple(blocks),
        of_block=of_block,
        of_example=_flatten(of_block, blocks),
        warnings=tuple(warnings),
    )


def assign_in_order(
    blocks: Sequence[Block],
    *,
    ratios: dict[str, float] | None = None,
    order_key: Callable[[Block], Any],
) -> Assignment:
    """Deal ``blocks`` into splits sequentially along ``order_key``.

    Splits are filled in ``SPLIT_ORDER`` — earliest items to ``train``, latest to
    ``test`` — because the point of an ordered split is that every evaluation item
    comes *after* every training item. Cut points are the cumulative
    :func:`apportion` targets, and a block straddling one goes to the split its
    midpoint falls in, so a large block cannot be silently truncated and singleton
    blocks land exactly on the apportioned counts.
    """
    ratios = normalize_ratios(ratios)
    ordered = sorted(blocks, key=order_key)
    size = sum(block.size for block in ordered)
    targets = apportion(size, ratios)

    boundaries: list[tuple[str, int]] = []
    cumulative = 0
    for name in ratios:
        cumulative += targets[name]
        boundaries.append((name, cumulative))

    of_block: dict[str, str] = {}
    filled = dict.fromkeys(ratios, 0)
    position = 0
    for block in ordered:
        midpoint = position + block.size / 2
        chosen = next(
            (name for name, bound in boundaries if midpoint < bound), boundaries[-1][0]
        )
        of_block[block.key] = chosen
        filled[chosen] += block.size
        position += block.size

    quotas = {name: ratio * size for name, ratio in ratios.items()}
    return Assignment(
        blocks=tuple(blocks),
        of_block=of_block,
        of_example=_flatten(of_block, blocks),
        warnings=tuple(_quota_warnings("", quotas, filled)),
    )


__all__ = [
    "DEFAULT_RATIOS",
    "DEFAULT_SEED",
    "SPLIT_ORDER",
    "Assignment",
    "EligibleFor",
    "StratumOf",
    "apportion",
    "assign_blocks",
    "assign_in_order",
    "normalize_ratios",
]
