# Milestone 2 — Fixture datasets

## Summary

Author the five small, synthetic toy datasets the rest of the pipeline is tested
against, with defects planted on purpose so the audits (Milestones 4–6) have real
problems to rediscover. Each fixture ships a machine-readable ground-truth file
(`defects.yaml`) so those later stages can be scored for **recall** against known
defects, not merely "it found something". A minimal loader and a validation script
check structural health; a test suite additionally verifies the planted defects
are actually present.

## Deliverables

- `datasets/fixtures/<name>/` for `qa_toy`, `agent_tasks_toy`, `retrieval_toy`,
  `bio_toy`, `safety_toy` (56 examples total). Each directory contains:
  - `examples.jsonl` — one `DatasetExample` per line, structurally valid but
    semantically flawed.
  - `defects.yaml` — ground truth of the planted defects, keyed by category
    (`exact_duplicates`, `near_duplicates`, `leakage`, `label_issues`,
    `duplicate_sources`, plus family-specific categories such as
    `duplicate_documents`, `paraphrased_documents`, `relevance_issues`).
  - `README.md` — a human-readable table of the planted defects.
- `src/eval_dataset_forge/datasets/loader.py` — minimal loader: `load_examples`
  (JSONL/YAML → schema-validated `DatasetExample`s), `load_defects`,
  `read_records`, `discover_fixtures`, and a `FixtureError` with file/record
  context.
- `scripts/validate_fixtures.py` — offline validator: schema parse, `example_id`
  uniqueness, single/known `task_family`, ground-truth references resolve, and
  `defects.yaml` `example_count` matches. Exit 0 iff all fixtures are valid.
- `tests/test_fixtures.py` — 72 tests: structural health plus ground-truth
  self-consistency (exact duplicates are identical, near-duplicates differ,
  train/test overlaps span both splits, answer-in-prompt leaks really appear,
  missing labels are empty, conflicting labels disagree, ambiguous labels are
  tagged, mislabeled entries are `auto_detectable: false`, retrieval duplicate
  docs are byte-identical, safety unsafe items are placeholders, the safety
  comply/refuse convention holds except where an inconsistent-format defect is
  planted) and a check that the validation script passes.

## Planted-defect coverage

Across the five fixtures: exact duplicates, paraphrase (near) duplicates,
train/test overlap, answer-in-prompt leakage, retrieval-document contamination,
duplicated / paraphrased documents, incomplete relevance labels, over-specific
queries, duplicate source records, and label issues (missing, conflicting,
ambiguous, inconsistent-format). Factually-wrong-but-well-formed gold answers are
included as documented, non-auto-detectable defects.

## Design decisions

- **Ground truth as data.** Planting defects without recording them makes "the
  audit found N issues" unfalsifiable. `defects.yaml` turns each fixture into a
  labeled test set, so Milestones 4–6 can report recall. It is descriptive ground
  truth (broader than any single audit enum), not a pydantic model.
- **Honest about the undetectable.** A well-formed but wrong gold answer
  (`qa-006` "Ag", `bio-009` "23", the `saf-006` over-refusal mislabel) is marked
  `auto_detectable: false` and excluded from recall — a limitation stated in data,
  not hidden.
- **Structural vs. semantic.** The fixtures parse cleanly against the schema; the
  defects are semantic. The loader and validation script assert only structural
  health, leaving semantic detection to the audit stages.
- **Safety scope enforced in tests.** Every `safety_toy` unsafe/ambiguous item is
  a `SYNTHETIC-*-PLACEHOLDER` string with no operational content, and a test fails
  if that ever stops being true.
- **Minimal loader now.** Just enough to load and validate fixtures; the full
  `validate` CLI (missing-field diagnostics, PII scanning, redaction) lands in
  Milestone 3 on top of this loader, and the writer/manifest modules in
  Milestone 10.

## Validation

```bash
python3 -m pytest -q                     # 121 passed (49 prior + 72 fixture)
python3 scripts/validate_fixtures.py     # all 5 fixtures structurally valid, exit 0
ruff check .                             # all checks passed
```

## Status

Complete. Next: Milestone 3 — Validation & redaction (schema validation, ID
uniqueness, missing-field checks, fake-secret detection, email/token redaction,
safe-data checks), which builds the `validate` CLI on top of this loader.
