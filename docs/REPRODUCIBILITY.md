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

# Score, bucket, and calibrate example difficulty (exits 1 with --fail-on-uncalibrated).
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl

# Bootstrap CIs, paired comparison, and ranking stability. Pass real per-example scores
# with --scores, or --simulate for deterministic synthetic systems (offline demo only —
# they describe no real system). --dataset repeats to pool several files into one
# benchmark. Exits 1 with --fail-on-unstable-ranking when the ordering is unsupported.
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --scores datasets/fixtures/qa_toy/system_scores.jsonl

# Regenerate the committed synthetic score fixture (byte-identical; --check verifies).
python3 scripts/make_demo_scores.py --check

# Generate a dataset card for the release.
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1/ \
  --out results/reports/qa_toy_dataset_card.md

# Run the full smoke audit over all fixtures.
PYTHONPATH=src python3 -m eval_dataset_forge smoke \
  --out results/reports/smoke_audit.md
```

## Determinism

Randomized steps (split construction, bootstrap resampling, permutation tests)
take an explicit `--seed`. Given the same fixtures and seed, audits, splits, and
statistics are byte-reproducible. Content hashes in each release `manifest.json`
let you verify that a rebuilt release matches a prior one.

Two determinism details are load-bearing rather than incidental. The statistics
stage draws **one** resample-index matrix per resampling unit and shares it across
every system, which is what makes the comparisons *paired* — without it the
difference of two marginal distributions would carry variance the pairing removes.
And a resampled p-value is itself an estimate: `stats_report.json` records
`p_mc_se = sqrt(p(1-p)/B)` and flags any comparison whose verdict is decided within
two standard errors of `alpha`, because such a verdict is set by the seed and not by
the data. Raise `--resamples` rather than reporting it.

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
