# Safety, privacy & data scope

## Data scope

All data committed to this repository is **synthetic or public-safe**. There is
no real personal data, no clinical or patient data, and no confidential material.
The fixture datasets are hand-authored to exhibit specific defects (duplicates,
leakage, label conflicts) for testing the audit — they are illustrative, not
real-world records.

## Privacy controls

The validation layer (Milestone 3) includes:

- **PII-like string detection** — heuristic matchers for email addresses,
  phone-like numbers, and token/secret-like strings, so that a dataset can be
  scanned before release. These are pattern heuristics, not a guarantee.
- **Redaction utilities** — deterministic replacement of detected PII-like spans
  with typed placeholders (e.g. `<EMAIL>`, `<TOKEN>`), used to sanitize text.
- **Fake-secret detection** — flags high-entropy / key-shaped strings so a
  committed fixture cannot accidentally embed a credential.

These utilities exist to keep *this* project safe and to be reusable by
downstream eval projects. They are not a substitute for a dedicated DLP system.

## Biomedical content

The `bio_toy` fixtures are non-clinical and non-patient-specific: synthetic
protein/database-lookup and paper-evidence style tasks with deliberate label
conflicts. They contain no medical advice, no dosing, and no patient information.
This tool is a benchmark-quality instrument, **not** a medical, legal, or
security authority, and must not be used as one.

## Safety fixtures

The `safety_toy` fixtures are synthetic over-refusal / under-refusal /
ambiguous-boundary examples used to test how a benchmark represents policy
boundaries. They contain **no real harmful instructions** — the "unsafe" items
are placeholders describing the category, not operational content.

## What we never commit

Secrets, API keys, `.env`, virtualenvs, caches, model weights, large datasets,
and cached API responses. `.gitignore` keeps them out of `git add` and
`.claude/settings.json` denies the tools that fetch them; the standing *audit* is
`scripts/hygiene_scan.py` (`eval_dataset_forge.hygiene`), which fails if any file
git tracks is ignored-but-forced-in, a forbidden secret/key/weight path, or
dataset-sized. It checks which files exist, not their contents — content
secret-detection lives in the validation layer, which is tested against fixtures
that hold fake secrets on purpose.

```bash
python3 scripts/hygiene_scan.py            # exit 1 on any contract violation
```

## Optional paid-API adapters

No paid API is required anywhere in this version. If a paid-API adapter is added
later it must ship disabled by default and with: a dry-run mode, a response
cache, a token/cost estimator, and a hard max-cost guard. See
[`../CLAUDE.md`](../CLAUDE.md).
