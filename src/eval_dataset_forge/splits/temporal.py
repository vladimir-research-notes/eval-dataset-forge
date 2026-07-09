"""Temporal splits: evaluate on the future (Milestone 7).

Order the examples by ``created_at`` and hold out the most recent ones. The split
then simulates the only condition a deployed system actually faces — data written
after the data it learned from — and measures the distribution shift a random split
averages away.

Splits fill in ``train`` → ``dev`` → ``test`` order, so every evaluation item is newer
than every training item, and the items used for model selection sit between the two.

**Timestamps this module trusts.** ISO-8601 dates and datetimes, parsed with
:meth:`datetime.datetime.fromisoformat`. A value it cannot parse is treated as absent
rather than sorted as a string, because ``"2024-1-3"`` sorts after ``"2024-01-30"``
and a silently mis-ordered temporal split is worse than a loud one.

**Absent timestamps forward-fill.** An example with no usable ``created_at`` inherits
the last timestamp seen before it in input order, so it keeps its position among its
neighbours instead of being swept to one end of the corpus — the "fall back to
insertion order" rule of ``docs/SPLITS.md``, applied locally rather than discarding
every good timestamp in the file. Every such example is named in a warning. Leading
examples with no predecessor sort first, i.e. into ``train``, which is the safe
direction: an undated example cannot then silently become part of the test set.

**The honest caveat.** On the committed fixtures a temporal split leaks *less* than a
random one, and not because it is designed to. Paraphrases were authored on adjacent
days, so time-ordering keeps most of them on the same side of the cut. That is a fact
about how the fixtures were written, not a property of temporal splitting: a
duplicate re-recorded a year later straddles just as readily. ``docs/SPLITS.md``
carries the measurement and the caveat together, and this module offers no guarantee
against duplicates at all. Compose with :mod:`.adversarial` if one is wanted.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from eval_dataset_forge.schemas import DatasetExample
from eval_dataset_forge.splits.assignment import Assignment, assign_in_order
from eval_dataset_forge.splits.blocks import Block, singleton_blocks

# Sorts before every real ISO-8601 timestamp, so undated leading examples land in the
# oldest bucket (``train``) rather than the newest (``test``).
_EPOCH = ""


def parse_timestamp(value: str | None) -> str | None:
    """Normalize an ISO-8601 date/datetime to a sortable string, or ``None``.

    Returns the *original* string when it parses, because ISO-8601 is lexicographically
    ordered once it is known to be well-formed; parsing is used as a validity gate, not
    as a conversion.
    """
    if not value:
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


def temporal_order(
    examples: Iterable[DatasetExample],
) -> tuple[dict[str, tuple[str, int]], list[str]]:
    """Sort keys ``example_id → (timestamp, insertion index)`` plus warnings.

    Ties on the timestamp break on insertion order, so two examples created the same
    day keep the order the file gave them.
    """
    keys: dict[str, tuple[str, int]] = {}
    undated: list[str] = []
    last_seen = _EPOCH
    for index, example in enumerate(examples):
        timestamp = parse_timestamp(example.created_at)
        if timestamp is None:
            undated.append(example.example_id)
        else:
            last_seen = timestamp
        keys[example.example_id] = (last_seen if timestamp is None else timestamp, index)

    warnings = []
    if undated:
        shown = ", ".join(undated[:5]) + (" …" if len(undated) > 5 else "")
        warnings.append(
            f"{len(undated)} example(s) have no usable created_at and fall back to "
            f"insertion order: {shown}"
        )
    return keys, warnings


def temporal_splits(
    examples: Iterable[DatasetExample],
    *,
    ratios: dict[str, float] | None = None,
) -> Assignment:
    """Assign the oldest examples to ``train`` and the newest to ``test``.

    Takes no seed: the ordering is the data's, not a generator's, so the strategy is
    reproducible without one. Blocks are singletons — a temporal split makes no claim
    about duplicates, and pretending otherwise by grouping them would silently reorder
    the timeline.
    """
    examples = list(examples)
    keys, warnings = temporal_order(examples)

    def order_key(block: Block) -> tuple[str, int]:
        return keys[block.key]

    assignment = assign_in_order(
        singleton_blocks(examples), ratios=ratios, order_key=order_key
    )
    return Assignment(
        blocks=assignment.blocks,
        of_block=assignment.of_block,
        of_example=assignment.of_example,
        warnings=(*warnings, *assignment.warnings),
    )


__all__ = ["parse_timestamp", "temporal_order", "temporal_splits"]
