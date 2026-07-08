# Human review: Milestone 3 — Validation & redaction

TL;DR
The first audit stage landed: a `validation/` package (schema + ID uniqueness +
PII/secret detection + redaction + content quality) exposed as the
`evalforge validate` CLI. Parsing is lenient so every problem in a file is
reported in one pass; a severity policy separates always-fatal *secrets* from
*PII* warnings (promotable with `--fail-on-pii`); redaction uses typed
placeholders and report snippets never leak the raw value. Full suite: 168
passed; fixtures validate; ruff clean.

What changed
- Added `src/eval_dataset_forge/validation/{__init__,schema,pii,redaction,quality}.py`.
- Wired `cmd_validate` in `cli.py` (`--json`, `--fail-on-pii`, `--redact-out`);
  removed `validate` from the pending-command map.
- Added `tests/test_validation.py` (~40 tests); updated `tests/test_smoke.py`
  (validate is implemented; `audit` is now the pending-command exemplar).
- Added `docs/VALIDATION.md`; new site page `site/validation.html` + a Validation
  nav link across all site pages; status updates on `audit-pipeline.html`,
  `schema.html`, `milestones.html`, `index.html`, `limitations.html`.

What should be checked later
- **Detector tuning.** The high-entropy token heuristic (len ≥ 20, three char
  classes, entropy ≥ 3.5) and the phone digit-count window (10–15) are judgement
  calls. Worth a look on more realistic text: do they over/under-fire? They are
  warnings by default precisely because they are noisy.
- **`secret_assignment` false positives.** The `keyword: value` matcher errors by
  default. Confirm the stopword/plausibility filter (`_plausible_secret_value`)
  is not too aggressive or too lax for real configs embedded in prompts.
- **Severity choice.** Is "secret ⇒ error, PII ⇒ warning" the right default for
  the release builder (Milestone 10)? It may want `--fail-on-pii` on by default.
- **Field coverage.** PII scanning covers `input`, `expected_output`, `metadata`,
  `labels`; identity/timestamp fields are intentionally excluded. Confirm that is
  the desired scope.
- **PII-flag shape.** Findings serialize to plain dicts to match
  `AuditResult.pii_flags`; verify that shape is what Milestone 12 expects when it
  folds validation output into the audit.

Technical risks
- Low. Pure-Python, stdlib + pydantic only; no network, no paid API, no runtime
  mutation of the repo. `--redact-out` writes only to a caller-supplied path.
- The safety-relevant risk is a *false sense of safety*: this is a heuristic
  sanitizer, not a DLP guarantee. This is stated in `docs/VALIDATION.md`,
  `docs/SAFETY.md`, and the Limitations page. A test asserts report snippets do
  not contain the raw detected value.

Commands to verify
```
python3 -m pytest -q
python3 scripts/validate_fixtures.py
PYTHONPATH=src python3 -m eval_dataset_forge validate --dataset datasets/fixtures/qa_toy/examples.jsonl
ruff check .
```

Human feedback
<!-- Human will fill this later. -->
