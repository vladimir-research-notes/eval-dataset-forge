"""Blocks: the atomic units of split assignment (Milestone 7).

Every strategy in this package is the same algorithm — deal units into splits until
each split's quota is full — and the strategies differ *only* in what a unit is.
That unit is a :class:`Block`.

=====================  ====================================  =========================
strategy               block                                  what it protects
=====================  ====================================  =========================
random / stratified    one example                            nothing
source-grouped         all examples sharing a ``source_id``   provenance dependence
adversarial            a duplicate cluster ∪ its source group content dependence
=====================  ====================================  =========================

A block is indivisible: its members always land in the same split. So the choice of
block *is* the choice of which dependence between examples the split refuses to cut
through — and, as the fixtures show, choosing the wrong one closes no channel at
all. ``src-fr-capital`` and ``src-fr-capital-dup`` are the same question under two
provenance ids, so a source-grouped split will happily place one in train and the
other in test. Only :func:`duplicate_blocks`, whose edges come from the content of
the examples rather than from their metadata, keeps them together.

Blocks are built by union-find over an edge set, so grouping is transitive: the
three-member ``qa-003``/``qa-004``/``qa-012`` cluster that Milestone 4's
single-linkage clustering reports becomes one block, not two overlapping pairs.
Transitivity is inherited, blind spots included — a paraphrase the dedup ladder
cannot see (``ret-001``/``ret-002``) yields no edge here either, and no strategy in
this package will keep that pair together.

Ordering is by first appearance in the input, never by id, so a block list is
deterministic without depending on how example ids happen to sort.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from eval_dataset_forge.dedup import DedupReport
from eval_dataset_forge.schemas import DatasetExample

# Why a block's members are pinned together. Recorded on every block so a split
# report can explain any grouping without re-deriving it.
SINGLETON = "singleton"
SOURCE = "source"
DUPLICATE = "duplicate"
DUPLICATE_AND_SOURCE = "duplicate+source"


@dataclass(frozen=True)
class Block:
    """An indivisible group of examples that must share a split.

    ``duplicate`` marks a block containing at least one member of a deduplication
    cluster. It is distinct from ``size > 1``: two examples derived from one source
    record are *dependent* but not *duplicated* (``src-shakespeare`` asks about
    Romeo and Juliet and about Hamlet), and only the duplicate flag makes a block
    ineligible for an evaluation split under the adversarial strategy.
    """

    example_ids: tuple[str, ...]
    reason: str = SINGLETON
    duplicate: bool = False

    def __post_init__(self) -> None:
        if not self.example_ids:
            raise ValueError("a block must contain at least one example")

    @property
    def key(self) -> str:
        """Stable block id: the first member in input order."""
        return self.example_ids[0]

    @property
    def size(self) -> int:
        return len(self.example_ids)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "example_ids": list(self.example_ids),
            "reason": self.reason,
            "duplicate": self.duplicate,
        }


def _components(order: list[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Union-find components over ``order``, both blocks and members in input order.

    Every id in ``order`` yields a component, so singletons survive. Edges naming an
    id outside ``order`` are ignored rather than raising: a caller may legitimately
    hand in a dedup report computed over a superset of the examples being split.
    """
    index = {example_id: i for i, example_id in enumerate(order)}
    parent = dict(zip(order, order, strict=True))

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        if left not in parent or right not in parent:
            continue
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # Keep the earlier-appearing id as the root so component keys are stable.
            if index[root_left] <= index[root_right]:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

    groups: dict[str, list[str]] = {}
    for example_id in order:
        groups.setdefault(find(example_id), []).append(example_id)
    return [groups[root] for root in sorted(groups, key=lambda root: index[root])]


def _star_edges(members: list[str]) -> list[tuple[str, str]]:
    """Join every member of a cluster to its first member."""
    return [(members[0], other) for other in members[1:]]


def singleton_blocks(examples: Iterable[DatasetExample]) -> list[Block]:
    """One block per example — the i.i.d. assumption, made explicit.

    Used by the random, stratified, and temporal strategies. Correct exactly when no
    two examples are dependent, which the dedup and leakage stages exist to check.
    """
    return [Block(example_ids=(example.example_id,)) for example in examples]


def source_blocks(examples: Iterable[DatasetExample]) -> list[Block]:
    """One block per ``source_id`` — examples from one source record stay together.

    Prevents *provenance* leakage. It does not prevent *content* leakage, and on the
    committed fixtures it prevents almost none of it: the planted duplicates were
    each given a fresh source id (``…-dup``, ``…-para``), exactly as a real dataset
    would record a paraphrase written as a new record. See :func:`duplicate_blocks`.
    """
    groups: dict[str, list[str]] = {}
    for example in examples:
        groups.setdefault(example.source_id, []).append(example.example_id)
    return [
        Block(
            example_ids=tuple(members),
            reason=SINGLETON if len(members) == 1 else SOURCE,
        )
        for members in groups.values()
    ]


def duplicate_blocks(
    examples: Iterable[DatasetExample],
    report: DedupReport,
    *,
    include_sources: bool = True,
) -> list[Block]:
    """Blocks joined by duplicate *content*, and (by default) by shared provenance.

    Edges come from every stage of the deduplication ladder — exact, normalized, and
    the heuristic near stage — so a false near-duplicate pins two unrelated examples
    into one split (harmless) while a missed one leaves them free to straddle
    (the failure that matters). The near stage's recall bound is therefore this
    strategy's guarantee bound, and both are documented in ``docs/DEDUPLICATION.md``.

    ``include_sources=True`` additionally merges each ``duplicate_sources`` group, so
    an adversarial block subsumes a source-grouped one. The resulting block is
    ``duplicate`` only if a *content* edge touches it; a pure source group is
    dependent, not duplicated, and stays eligible for an evaluation split.
    """
    examples = list(examples)
    order = [example.example_id for example in examples]
    source_of = {example.example_id: example.source_id for example in examples}

    edges: list[tuple[str, str]] = []
    duplicate_members: set[str] = set()
    for finding in report.findings:
        edges.extend(_star_edges(list(finding.example_ids)))
        duplicate_members.update(finding.example_ids)

    if include_sources:
        for _, members in sorted(report.duplicate_source_ids.items()):
            edges.extend(_star_edges(list(members)))

    blocks: list[Block] = []
    for members in _components(order, edges):
        has_duplicate = any(member in duplicate_members for member in members)
        source_ids = [source_of[member] for member in members]
        # `reason` names the edge kinds that actually pinned these members together. A
        # shared source_id inside a content cluster is only one of them when source
        # edges were built at all.
        shares_source = include_sources and len(set(source_ids)) < len(source_ids)
        blocks.append(
            Block(
                example_ids=tuple(members),
                reason=_reason(len(members), has_duplicate, shares_source),
                duplicate=has_duplicate,
            )
        )
    return blocks


def _reason(size: int, has_duplicate: bool, shares_source: bool) -> str:
    if size == 1:
        return SINGLETON
    if has_duplicate and shares_source:
        return DUPLICATE_AND_SOURCE
    if has_duplicate:
        return DUPLICATE
    return SOURCE


def block_of_example(blocks: Iterable[Block]) -> dict[str, Block]:
    """Reverse index: example id → the block that owns it."""
    return {example_id: block for block in blocks for example_id in block.example_ids}


__all__ = [
    "SINGLETON",
    "SOURCE",
    "DUPLICATE",
    "DUPLICATE_AND_SOURCE",
    "Block",
    "block_of_example",
    "duplicate_blocks",
    "singleton_blocks",
    "source_blocks",
]
