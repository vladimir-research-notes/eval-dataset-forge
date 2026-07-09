"""Unstratified strategies: random and source-grouped (Milestone 7).

The two live together because they are **the same call**. Both deal blocks into
splits with :func:`~.assignment.assign_blocks`, unstratified, at a fixed seed. They
differ in one argument: what a block is.

.. code-block:: text

    random          assign_blocks(singleton_blocks(examples), …)
    source-grouped  assign_blocks(source_blocks(examples),    …)

That is the whole difference, and it is worth seeing in one file, because it makes
the load-bearing claim of this package concrete: **a split strategy is a choice of
which dependence between examples it refuses to cut through.** Random refuses none —
it assumes the examples are i.i.d., which is precisely the assumption the dedup and
leakage stages exist to test. Source-grouped refuses to cut a provenance group.

Neither refuses to cut a *content* group, so neither closes the channel the fixtures
plant: ``qa-001`` and ``qa-002`` are the same question with the same answer under two
source ids (``src-fr-capital``, ``src-fr-capital-dup``), and both strategies will put
one in train and the other in test whenever the seed says so. See
:mod:`.adversarial`, and the measured leakage-versus-strategy table in
``docs/SPLITS.md``.

(This module is named ``random`` after the strategy, per ``docs/PLAN.md`` §4. Python 3
resolves ``import random`` absolutely, so the stdlib module of that name is still
reachable from every other module in this package — :mod:`.assignment` imports it.)
"""

from __future__ import annotations

from collections.abc import Iterable

from eval_dataset_forge.schemas import DatasetExample
from eval_dataset_forge.splits.assignment import (
    DEFAULT_SEED,
    Assignment,
    assign_blocks,
)
from eval_dataset_forge.splits.blocks import singleton_blocks, source_blocks


def random_splits(
    examples: Iterable[DatasetExample],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> Assignment:
    """Assign each example independently — the i.i.d. baseline.

    Reproducible at a fixed ``seed`` and nothing more. Every dependence between
    examples (duplicates, paraphrases, shared sources, shared time) is free to
    straddle a split boundary, so this strategy's output should be read together with
    the residual-leakage count the report carries.
    """
    return assign_blocks(singleton_blocks(examples), ratios=ratios, seed=seed)


def source_grouped_splits(
    examples: Iterable[DatasetExample],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> Assignment:
    """Keep every example sharing a ``source_id`` in one split.

    Closes the source-level dependence channel that
    :func:`~eval_dataset_forge.leakage.cross_split_sources` reports: two questions
    written from one Wikipedia paragraph are not duplicates, but scoring on one after
    training on the other still leaks.

    It closes only that channel. A duplicate recorded under a fresh ``source_id`` is
    invisible to it — which, on the committed fixtures, is *every* planted duplicate.
    Ratio control is also coarser than random's, because a source group is
    indivisible; the report's achieved ratios show the cost.
    """
    return assign_blocks(source_blocks(examples), ratios=ratios, seed=seed)


__all__ = ["random_splits", "source_grouped_splits"]
