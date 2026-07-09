"""Reading and writing per-example system scores (Milestone 9).

The statistics stage is the only one in this pipeline whose input is not the dataset. It
needs an *evaluation*: what each system scored on each example. That file comes from
whatever harness ran the systems, so the reader accepts the two shapes such harnesses
actually emit, one JSON object per line:

.. code-block:: text

    {"example_id": "qa-001", "system": "sys_a", "score": 1.0}      # long
    {"example_id": "qa-001", "scores": {"sys_a": 1.0, "sys_b": 0}} # wide

Shapes may be mixed within a file; the same ``(example_id, system)`` pair may not be scored
twice. Scores are floats and are not required to be binary — accuracy, F1, and a graded
rubric all work, and only the interpretation of the mean changes.

**Alignment is strict.** A score for an example the dataset does not contain, or an example
no system scored, raises rather than warns. The alternative is to impute, and every
imputation of a missing eval result is a silent decision about a system's score on an
example it may have crashed on. Milestone 6 refuses to impute a missing label; this refuses
for the same reason.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from eval_dataset_forge.schemas import DatasetExample

# system -> example_id -> score
ScoreTable = dict[str, dict[str, float]]


def _coerce(value: object, example_id: str, system: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"score for ({example_id!r}, {system!r}) must be a number, got {value!r}"
        )
    return float(value)


def parse_scores(lines: Iterable[str]) -> ScoreTable:
    """Parse JSONL score records in either the long or the wide shape."""
    table: ScoreTable = {}
    for number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {number}: invalid JSON ({error.msg})") from None
        if not isinstance(record, dict) or "example_id" not in record:
            raise ValueError(f"line {number}: every record needs an 'example_id'")
        example_id = str(record["example_id"])

        if "scores" in record:
            pairs = record["scores"]
            if not isinstance(pairs, dict):
                raise ValueError(f"line {number}: 'scores' must be an object")
            items = list(pairs.items())
        elif "system" in record and "score" in record:
            items = [(str(record["system"]), record["score"])]
        else:
            raise ValueError(
                f"line {number}: want either 'system' + 'score' or a 'scores' object"
            )

        for system, value in items:
            system = str(system)
            bucket = table.setdefault(system, {})
            if example_id in bucket:
                raise ValueError(
                    f"line {number}: ({example_id!r}, {system!r}) is scored more than once"
                )
            bucket[example_id] = _coerce(value, example_id, system)
    if not table:
        raise ValueError("no score records found")
    return table


def load_scores(path: str | Path) -> ScoreTable:
    """Read a JSONL score file. See :func:`parse_scores` for the accepted shapes."""
    return parse_scores(Path(path).read_text(encoding="utf-8").splitlines())


def to_table(scores: Mapping[str, Sequence[float]], example_ids: Sequence[str]) -> ScoreTable:
    """Key positional score vectors by example id, restoring what the position stood for.

    :func:`~eval_dataset_forge.statistics.simulate.simulate_scores` works in positions, since
    it never sees an example. Anything written to disk must be keyed by id, or a reordering
    of the dataset would silently reassign every score.
    """
    table: ScoreTable = {}
    for system, values in scores.items():
        if len(values) != len(example_ids):
            raise ValueError(
                f"system {system!r} has {len(values)} scores for {len(example_ids)} examples"
            )
        table[system] = dict(zip(example_ids, (float(v) for v in values), strict=True))
    return table


def write_scores(
    scores: Mapping[str, Mapping[str, float]],
    path: str | Path,
    *,
    example_order: Sequence[str] | None = None,
) -> Path:
    """Write the long shape, one record per ``(example, system)``. Byte-identical per call.

    Examples follow ``example_order`` when given (the dataset's order), systems follow the
    mapping's order. Nothing is sorted implicitly, so a regenerated file diffs cleanly
    against a committed one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    systems = list(scores)
    ordered = list(example_order) if example_order is not None else list(scores[systems[0]])
    lines = [
        json.dumps(
            {"example_id": example_id, "system": system, "score": scores[system][example_id]},
            ensure_ascii=False,
        )
        for example_id in ordered
        for system in systems
    ]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def align(
    scores: Mapping[str, Mapping[str, float]], examples: Sequence[DatasetExample]
) -> tuple[list[str], np.ndarray]:
    """``(system names, (n_systems, n_examples) matrix)`` in the dataset's example order.

    Raises on any mismatch in either direction. See the module docstring for why.
    """
    if not scores:
        raise ValueError("no systems to compare")
    wanted = [example.example_id for example in examples]
    known = set(wanted)
    if len(known) != len(wanted):
        raise ValueError("dataset contains duplicate example_ids; scores cannot be aligned")

    names = list(scores)
    for system in names:
        scored = set(scores[system])
        if missing := sorted(known - scored):
            raise ValueError(
                f"system {system!r} has no score for {len(missing)} example(s): "
                f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}"
            )
        if extra := sorted(scored - known):
            raise ValueError(
                f"system {system!r} scores {len(extra)} example(s) absent from the dataset: "
                f"{', '.join(extra[:5])}{' …' if len(extra) > 5 else ''}"
            )
    matrix = np.array(
        [[scores[system][example_id] for example_id in wanted] for system in names], dtype=float
    )
    return names, matrix


__all__ = ["ScoreTable", "align", "load_scores", "parse_scores", "to_table", "write_scores"]
