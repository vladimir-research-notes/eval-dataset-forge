# Milestone 1 — Core schemas

## Summary

Implement the typed data contracts every later stage depends on as pydantic v2
models in `src/eval_dataset_forge/schemas.py`, with a CPU-only test suite that
locks the field set and the JSON round-trip. No audit logic yet — this milestone
defines *what the data looks like*, so subsequent stages can be reasoned about
and tested against a fixed, validated shape.

## Deliverables

- `src/eval_dataset_forge/schemas.py` — eleven models plus their enums:
  - **Example:** `DatasetExample`.
  - **Findings:** `DeduplicationFinding`, `LeakageFinding`, `LabelAuditFinding`.
  - **Split / difficulty / statistics:** `SplitSpec`, `DifficultyScore`,
    `BootstrapResult`.
  - **Aggregates / release:** `AuditResult`, `DatasetManifest`, `DatasetCard`,
    `DatasetRelease`.
  - **Enums (`StrEnum`):** `Split`, `DedupKind`, `LeakageKind`, `LabelIssue`,
    `SplitStrategy`, `DifficultyBucket`; plus the `KNOWN_TASK_FAMILIES` constant
    and a `MODELS` registry for introspection/tests.
- `tests/test_schemas.py` — 37 tests: registry coverage, per-model JSON
  round-trip, `DatasetExample` field-set lock, strict rejection of unknown /
  missing / empty / `None` fields, `expected_output` JSON-type preservation,
  enum coercion, and every cross-field validator.

## Design decisions

- **Strictness by default.** A shared `ForgeModel` base sets `extra="forbid"`, so
  typos and stray keys fail loudly at load time. This matches the project thesis
  that malformed data should never silently reach an audit. Genuine free-form
  data lives in the explicit `metadata` / `labels` dicts.
- **String enums.** Enums subclass `StrEnum` so `model_dump(mode="json")` yields
  plain strings and records round-trip through JSONL with no custom encoder.
- **`expected_output` is `str | bool | int | float | list | dict`.** It covers
  gold strings, scalar labels, and structured judgements. `None` is excluded on
  purpose — a missing answer is a *label* defect (Milestone 6), not a valid
  value. pydantic's smart union preserves the input type (no `bool`↔`int`
  surprises), which is asserted in the tests.
- **`task_family` is an open string, not an enum.** The vocabulary is meant to
  grow; unknown families are a Milestone 3 validation *warning*, not a schema
  error. `KNOWN_TASK_FAMILIES` documents the designed-for set.
- **Cheap structural validators only.** Split-ratio sum, CI ordering, and
  release/manifest count are `O(1)` cross-field checks that reject impossible
  records early. Semantic checks stay in their own milestones.
- **No package-level import of pydantic.** `schemas.py` imports pydantic, but the
  package `__init__` and CLI do not import `schemas`, so `evalforge check-env`
  keeps working before the scientific stack is installed. Consumers import
  `from eval_dataset_forge.schemas import ...`.

## Validation

```bash
python3 -m pytest -q          # 49 passed (12 scaffold + 37 schema)
ruff check .                  # all checks passed
```

## Status

Complete. Next: Milestone 2 — Fixture datasets (the five toy sets with planted
defects, plus a validation script), which these schemas will type and load.
