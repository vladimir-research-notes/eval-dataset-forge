# safety_toy — safety-behavior fixture (synthetic placeholders)

A small, synthetic safety dataset (10 examples, `task_family: safety`) of
over-refusal / under-refusal / ambiguous-boundary cases with defects planted on
purpose. `expected_output` is the correct behavior label — `"comply"` or
`"refuse"` — and `metadata.boundary` records the intended difficulty
(`clear_safe` / `clear_unsafe` / `ambiguous`).

**Safety scope.** There are **no real harmful instructions**. Every unsafe or
ambiguous item is a `SYNTHETIC-*-PLACEHOLDER` string that names the request
*category* and contains no operational content; `tests/test_fixtures.py` enforces
this. See [`docs/SAFETY.md`](../../../docs/SAFETY.md). Ground truth is in
[`defects.yaml`](defects.yaml).

## Planted defects

| defect | example(s) | notes |
|--------|------------|-------|
| exact duplicate | `saf-004` / `saf-005` | identical placeholder |
| near duplicate (paraphrase) | `saf-001` / `saf-003` | paraphrased cookie recipe |
| over-refusal mislabel | `saf-006` | benign request labeled `"refuse"` — **not auto-detectable** |
| ambiguous boundary | `saf-007` | dual-use; either label defensible |
| conflicting labels | `saf-008` | annotators disagree (comply vs refuse) |
| missing label | `saf-009` | empty `expected_output` |
| inconsistent format | `saf-010` | gold `"no"` breaks the comply/refuse convention |
| train/test overlap | `saf-001`↔`saf-003`, `saf-004`↔`saf-005` | duplicates across splits |

Whether a request is *truly* safe is a judgment the tool does not make — the
over-refusal mislabel is `auto_detectable: false`. What the audit *can* surface is
structural label quality: missing, conflicting, ambiguous, and inconsistently
formatted labels.
