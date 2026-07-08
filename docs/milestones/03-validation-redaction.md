# Milestone 3 — Validation & redaction

## Summary

Land the first stage of the audit pipeline: a validation & redaction layer that
turns the Milestone 1 schema into an enforced, file-level gate and adds
heuristic PII/secret detection with deterministic redaction. This is audit
dimension #1 — every later stage assumes it runs on schema-valid examples — so it
is strict about structure while being explicit about the limits of its heuristic
safety checks. Wires the `evalforge validate` CLI (previously a stub).

## Deliverables

- `src/eval_dataset_forge/validation/` package:
  - `schema.py` — lenient parse (`load_records_lenient`: JSONL read line by line
    so one bad line doesn't hide the rest), full-file schema validation
    (`validate_records` → one `SchemaError` per pydantic sub-error, kinds:
    `parse_error` / `missing_field` / `unknown_field` / `type_error`), and
    `find_duplicate_ids` over raw records. Records addressed by 1-based ordinal.
  - `pii.py` — `scan(text)` returning non-overlapping `PiiSpan`s from named
    patterns (email, IPv4 with octet-range validation, `AKIA…` access keys, PEM
    private-key headers, `keyword: value` assignments) plus a Shannon-entropy
    heuristic for bare high-entropy tokens; phone matches range-checked to 10–15
    digits. `scan_example`/`scan_examples` walk `input`, `expected_output`,
    `metadata`, `labels` and report a dotted field path with a **redacted**
    snippet. Two severity classes: `SECRET_KINDS` vs `PII_KINDS`.
  - `redaction.py` — deterministic typed-placeholder redaction (`<EMAIL>`,
    `<TOKEN>`, `<SECRET>`, …) applied right-to-left; `redacted_context` (no-leak
    report snippets); `redact_example` (scrubs nested string leaves, returns a
    copy).
  - `quality.py` — surface content warnings: `empty_input`,
    `empty_expected_output`, `unknown_task_family`.
  - `__init__.py` — `validate_dataset(path, *, fail_on_pii=False) →
    ValidationReport` orchestrator, severity policy (`ok`/`n_errors`),
    `format_report`, `to_dict` (for `--json`), and `write_redacted`.
- `src/eval_dataset_forge/cli.py` — real `cmd_validate` (lazy import) with
  `--json`, `--fail-on-pii`, `--redact-out`; `validate` removed from `_PENDING`.
- `tests/test_validation.py` — 40+ tests: per-detector positives and negatives,
  deterministic + index-safe redaction, no-leak snippet guarantee, every
  `SchemaError` kind, duplicate-id detection, quality warnings, severity policy,
  CLI exit codes, redacted round-trip, and a safe-data check that every committed
  fixture scans clean. `tests/test_smoke.py` updated (validate now implemented).
- Docs: `docs/VALIDATION.md`; website `site/validation.html` (new page + nav link
  across the site); status updates to `audit-pipeline.html`, `schema.html`,
  `milestones.html`, `index.html`, `limitations.html`.

## Design decisions

- **Lenient, one-pass reporting.** The loader raises on the first bad record
  (right for hand-authored fixtures); the CLI reports *all* problems at once, each
  keyed to a record ordinal, so fixing a dataset is not a game of whack-a-mole.
- **Severity policy: secrets vs PII.** An unambiguous credential shape (AWS key,
  PEM header, `password = …`) is always an error — a benchmark must never ship
  one. Ambiguous personal identifiers (email/phone/IP/bare token) are warnings by
  default, promotable with `--fail-on-pii`. This keeps `validate` usable on real
  datasets that legitimately mention an address without hard-failing, while the
  release builder can enforce the stricter policy.
- **Never leak while reporting.** A privacy tool that prints the secret it found
  is a bug. Report snippets are redacted before printing; a test asserts the raw
  value is absent from the snippet.
- **Warnings ≠ errors.** Empty gold answers (planted missing-label defects like
  `qa-010`) are warnings, not failures — the semantic label audit is Milestone 6.
  Validation only surfaces surface-level signals.
- **Honest heuristics.** The entropy/phone detectors' false-positive and
  false-negative modes are documented in `docs/VALIDATION.md` and on the site;
  that is exactly why bare-token/PII findings are warnings.
- **Report types are dataclasses, not release schema.** The `ValidationReport` is
  ephemeral CLI output, not a versioned on-disk artifact, so it stays decoupled
  from `SCHEMA_VERSION`; PII findings serialize to plain dicts, matching
  `AuditResult.pii_flags` for later reuse.

## Validation

```bash
python3 -m pytest -q                     # 168 passed (128 prior + 40 validation)
python3 scripts/validate_fixtures.py     # all 5 fixtures structurally valid, exit 0
PYTHONPATH=src python3 -m eval_dataset_forge validate \
  --dataset datasets/fixtures/qa_toy/examples.jsonl   # verdict OK, exit 0
ruff check .                             # all checks passed
```

## Status

Complete. Next: Milestone 4 — Deduplication (exact, normalized-text, and
heuristic near-duplicate detection with token n-gram Jaccard and an optional
TF-IDF cosine path), the second pipeline stage, tested against the fixtures'
known duplicate clusters.
