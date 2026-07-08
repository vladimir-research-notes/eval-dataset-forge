# Reproducibility

Everything here runs CPU-only and offline. No GPU, no network at runtime, no paid
API.

## Environment

- Python 3.12 (the target instance provides `python3`; there is no bare
  `python`).
- This is a PEP-668 "externally managed" environment, so `pip` needs
  `--break-system-packages` (or a virtualenv, if `python3-venv` is available).

Install dependencies:

```bash
# Core (required): schema validation, statistics, YAML.
python3 -m pip install --user --break-system-packages pydantic numpy pyyaml
# Optional accelerators: pandas, scikit-learn (TF-IDF near-duplicate detection).
python3 -m pip install --user --break-system-packages pandas scikit-learn
# Test/lint tooling:
python3 -m pip install --user --break-system-packages pytest ruff
```

Verify:

```bash
python3 scripts/00_check_environment.py
PYTHONPATH=src python3 -m eval_dataset_forge check-env
```

## Run the tests

```bash
python3 -m pytest -q
```

The `src/` layout plus `conftest.py` puts the package on `sys.path`, so tests run
without an editable install.

## Reproduce the audit pipeline (once milestones land)

```bash
# Validate a fixture dataset against the schema.
PYTHONPATH=src python3 -m eval_dataset_forge validate \
  --dataset datasets/fixtures/qa_toy/examples.jsonl

# Run the full audit and write a report.
PYTHONPATH=src python3 -m eval_dataset_forge audit \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --out results/audits/qa_toy/

# Build a versioned release with stratified splits.
PYTHONPATH=src python3 -m eval_dataset_forge build-splits \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --strategy stratified \
  --out datasets/releases/qa_toy_v0.1/

# Generate a dataset card for the release.
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1/ \
  --out results/reports/qa_toy_dataset_card.md

# Run the full smoke audit over all fixtures.
PYTHONPATH=src python3 -m eval_dataset_forge smoke \
  --out results/reports/smoke_audit.md
```

## Determinism

Randomized steps (split construction, bootstrap resampling) take an explicit
`--seed`. Given the same fixtures and seed, audits, splits, and statistics are
byte-reproducible. Content hashes in each release `manifest.json` let you verify
that a rebuilt release matches a prior one.

## Rebuilding the milestone history

The repository is built one milestone per Claude Code run:

```bash
./scripts/run_all_milestones_with_claude.sh 1 15
```

The runner enforces, between milestones, that a new commit was created, the tree
is clean, and the tests pass. See [`PLAN.md`](PLAN.md) §13.

Permissions are preconfigured in the committed `.claude/settings.json`
(`acceptEdits` + a safe allow/deny list that denies `git push`, `sudo`, network
tools, `rm`, and secret reads), so the runner does not prompt. Tune per run with
`CLAUDE_EXTRA_ARGS` (default `--permission-mode acceptEdits`); a weekly cost guard
is available via `WEEKLY_BUDGET_USD` and a per-milestone turn cap via `MAX_TURNS`.

## Website

```bash
./scripts/serve_site.sh          # serves site/ at http://localhost:9004/
```
