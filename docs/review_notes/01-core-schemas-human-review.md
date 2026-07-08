# Human review: Milestone 1 — Core schemas

TL;DR
The typed data contracts every later stage depends on are implemented as
pydantic v2 models in `src/eval_dataset_forge/schemas.py`, with strict
unknown-field rejection, string-valued enums for clean JSONL serialization, a few
structural cross-field validators, and a 37-test suite that locks the field set
and a JSON dump→reload round-trip for all eleven models.

What changed
- Added `src/eval_dataset_forge/schemas.py` (11 models + 6 enums + `MODELS`
  registry + `KNOWN_TASK_FAMILIES`).
- Added `tests/test_schemas.py` (37 tests; total suite now 49).
- Updated the website (`schema.html`, `milestones.html`, `index.html`) and
  `docs/DATASET_SCHEMA.md` to mark the schema as implemented and document what it
  enforces.

What should be checked later
- Field/type choices vs. what the audit stages actually need once dedup, leakage,
  label audit, splits, difficulty, and statistics land — especially the
  free-dict fields (`metadata`, `labels`, `split_statistics`, `audit_results`).
- Whether `expected_output` should stay a broad union or gain a stricter,
  per-task-family shape (e.g. retrieval relevance judgements).
- Whether `extra="forbid"` is the right default for every model or too strict for
  forward-compatible manifests/cards; a `SCHEMA_VERSION` bump is the intended
  escape hatch for additive changes.
- The `labels.annotations: [{annotator, value}]` shape referenced in
  `docs/DATASET_SCHEMA.md` is currently just a free dict; Milestone 6 will decide
  whether to model it explicitly.

Technical risks
- Low. This milestone is pure data definitions + tests; no I/O, no network, no
  paid API, and nothing that mutates the repo at runtime.
- `StrEnum` and PEP 604 unions require Python ≥ 3.11; the project targets 3.12, so
  this is fine, but worth noting for any backport.

Commands to verify
```
python3 -m pytest -q
ruff check .
PYTHONPATH=src python3 -c "from eval_dataset_forge.schemas import MODELS; print(sorted(MODELS))"
```

Human feedback
<!-- Human will fill this later. -->
