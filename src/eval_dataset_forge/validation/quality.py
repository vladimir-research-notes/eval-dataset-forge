"""Lightweight content-quality checks (Milestone 3).

These are *warnings*, not schema errors: a record can be perfectly well-formed
and still be a poor evaluation example. The checks here are intentionally shallow
and cheap — an empty prompt, an empty gold answer, an off-vocabulary task family.
The deeper, semantic label analysis (conflicting / ambiguous / inconsistent
labels, adjudication) is the Milestone 6 label audit; this layer only surfaces
the surface-level signals that are obvious at validation time.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval_dataset_forge.schemas import KNOWN_TASK_FAMILIES


@dataclass(frozen=True)
class QualityWarning:
    """A non-fatal content-quality signal on one example.

    ``kind`` is ``empty_input``, ``empty_expected_output``, or
    ``unknown_task_family``.
    """

    example_id: str | None
    kind: str
    detail: str


def check_quality(examples) -> list[QualityWarning]:
    """Return quality warnings for a sequence of :class:`DatasetExample`.

    - ``empty_input`` — the prompt is empty or whitespace-only.
    - ``empty_expected_output`` — a textual gold answer is empty/whitespace
      (a likely missing label; the label audit examines this more fully).
    - ``unknown_task_family`` — ``task_family`` is outside
      :data:`KNOWN_TASK_FAMILIES`; the vocabulary is open by design, so this is a
      warning, never an error.
    """
    warnings: list[QualityWarning] = []
    for example in examples:
        if not example.input.strip():
            warnings.append(
                QualityWarning(
                    example.example_id, "empty_input", "input is empty or whitespace-only"
                )
            )
        expected = example.expected_output
        if isinstance(expected, str) and not expected.strip():
            warnings.append(
                QualityWarning(
                    example.example_id,
                    "empty_expected_output",
                    "expected_output is empty (possible missing label)",
                )
            )
        if example.task_family not in KNOWN_TASK_FAMILIES:
            warnings.append(
                QualityWarning(
                    example.example_id,
                    "unknown_task_family",
                    f"task_family {example.task_family!r} is not in KNOWN_TASK_FAMILIES",
                )
            )
    return warnings


__all__ = ["QualityWarning", "check_quality"]
