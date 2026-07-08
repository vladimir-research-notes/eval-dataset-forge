# Validation & redaction

> Status: implemented in Milestone 3
> (`src/eval_dataset_forge/validation/`, `tests/test_validation.py`).

Validation is audit dimension #1. Every later stage (dedup, leakage, label audit,
splits, statistics) assumes it is operating on schema-valid examples, so this
layer runs first: strict about structure, explicit about the limits of its
heuristic safety checks. It **reports**; it never silently mutates data.

The public entry point is `validation.validate_dataset(path, *, fail_on_pii=False)
→ ValidationReport`, exposed as the `evalforge validate` CLI.

## What it checks

| check | module | severity | exact / heuristic |
|-------|--------|----------|-------------------|
| schema — missing / unknown / mistyped fields | `schema.py` | error | exact |
| ID uniqueness — one `example_id` per record | `schema.py` | error | exact |
| secret-shaped strings | `pii.py` | error | heuristic |
| PII-shaped strings | `pii.py` | warning\* | heuristic |
| content quality — empty prompt/answer, unknown family | `quality.py` | warning | exact |

\* Personal-identifier findings are warnings by default and promoted to errors
under `--fail-on-pii`. Unambiguous *secrets* are always errors.

## 1. Schema & ID uniqueness

Where the Milestone 2 loader raises on the first malformed record, the `validate`
command needs to report *every* problem across a file in one pass. Parsing is
therefore **lenient**: a bad JSONL line, a schema violation, or a non-object
record becomes a `SchemaError` and the scan continues. Each record is addressed
by a 1-based **ordinal** (its position among the non-blank, non-comment records),
so a message points at the right record regardless of which earlier ones failed.

`SchemaError.kind` is one of `parse_error`, `missing_field`, `unknown_field`
(a forbidden extra key), or `type_error`. Duplicate ids are detected on the raw
records — so a collision is reported even when one of the colliding records also
fails schema validation — and reported as `example_id → [ordinals]`.

## 2. PII / secret detection

Two severity classes are distinguished so the caller can decide what blocks a
release:

- **secret** (`aws_access_key`, `private_key`, `secret_assignment`) — an
  unambiguous credential shape. A benchmark should never ship one, so these are
  errors by default.
- **pii** (`email`, `phone`, `ipv4`, `token`) — a personal-identifier shape or a
  bare high-entropy token. Ambiguous enough to be warnings by default.

Detection combines high-precision named patterns with a low-precision entropy
heuristic:

- **Named patterns** — email; IPv4 with per-octet range validation; `AKIA…/ASIA…`
  access-key ids; PEM `-----BEGIN … PRIVATE KEY-----` headers; and
  `keyword: value` / `keyword = value` assignments (the value is post-filtered so
  prose like `password: required` is not flagged).
- **Phone** — a country-code-optional digit run, kept only if it carries 10–15
  digits (E.164-ish), which rejects dates and short numbers.
- **High-entropy token** — a standalone 20+ char string with three of
  {lower, upper, digit} at Shannon entropy ≥ 3.5 bits/char (or a very long, very
  high-entropy string). This catches API-key-like values that no fixed pattern
  matches, while ignoring hex hashes and English prose.

Overlapping matches are de-duplicated (earlier start wins; ties go to the longer
span), and because named-pattern spans are collected before token candidates, a
precise hit (an AWS key) suppresses the generic token candidate over the same
characters.

## 3. Redaction

`redaction.redact(text)` replaces every detected span with a typed placeholder
(`<EMAIL>`, `<PHONE>`, `<IP>`, `<AWS_KEY>`, `<PRIVATE_KEY>`, `<SECRET>`,
`<TOKEN>`), applied right-to-left so span indices stay valid. `redact_example`
scrubs `input`, a textual `expected_output`, and every string leaf in `metadata`
and `labels`, returning a new object (the original is untouched).

Report snippets are produced by `redacted_context`, which redacts a small window
around the finding **before** it is printed — so the raw value a finding flags
never leaks into the terminal, a log, or `--json` output.

## Severity policy & exit code

`ValidationReport.ok` is true iff there are **no errors**. Errors are: schema
errors, duplicate ids, and secret-class PII findings — plus *all* PII findings
when `fail_on_pii` is set. Warnings (quality, non-secret PII) never change the
verdict or the CLI exit code. `evalforge validate` exits `0` when `ok`, else `1`.

## Usage

```bash
evalforge validate --dataset datasets/fixtures/qa_toy/examples.jsonl
evalforge validate --dataset data.jsonl --json                  # machine-readable
evalforge validate --dataset data.jsonl --fail-on-pii           # PII blocks too
evalforge validate --dataset data.jsonl --redact-out clean.jsonl
```

On the committed fixtures the command exits `0`: they are schema-valid, uniquely
keyed, and PII-free (the safe-data guarantee, asserted by a test). Planted
missing-label defects (an empty gold answer, e.g. `qa-010`) surface as
`empty_expected_output` **warnings** — the deeper label analysis is the
Milestone 6 label audit ([`LABEL_AUDIT.md`](LABEL_AUDIT.md)).

## Known limitations

- The token and phone detectors are the noisiest: a long high-entropy hash or a
  digit-heavy string can be a false positive, and unusual formats can be false
  negatives. This is why bare-token/PII findings are warnings by default.
- Detection is regex + entropy, not a trained NER model or a DLP system. It
  removes only what it recognizes and is a project-local sanitizer, not a
  guarantee of privacy safety. See [`SAFETY.md`](SAFETY.md).
- Content-quality checks here are surface-level (empty fields, unknown family).
  Conflicting / ambiguous / inconsistent labels and adjudication are the
  Milestone 6 label audit.
