# Milestone 0 — Scaffold & project contract

## Summary

Establish the project skeleton and the full build contract so that milestones
1–15 can be run one-per-Claude-Code-invocation via
`scripts/run_all_milestones_with_claude.sh`. No audit subsystems yet — those land
in their own milestones.

## Deliverables

- `pyproject.toml` — packaging, `src/` layout, `evalforge` console script,
  pytest + ruff config; small hard dep set (`pydantic`, `numpy`, `pyyaml`) with
  optional `full`/`dev` extras.
- `src/eval_dataset_forge/` — `__init__.py` (version + `SCHEMA_VERSION`),
  `__main__.py`, `cli.py` (argparse: `check-env`/`version`/`info` live; audit/
  build subcommands declared and pointing at their milestone).
- `conftest.py` — `src/` path shim so tests run without an install.
- `tests/test_smoke.py`, `tests/test_scaffold_layout.py` — CPU-only tests that
  also assert the documented layout, docs, and website pages exist.
- `docs/` — `PLAN.md` (full contract) plus `METHODOLOGY`, `DATASET_SCHEMA`,
  `DEDUPLICATION`, `LEAKAGE`, `LABEL_AUDIT`, `SPLITS`, `DIFFICULTY`, `STATISTICS`,
  `RESULTS`, `LIMITATIONS`, `SAFETY`, `REPRODUCIBILITY`, `PAPER_PATH`, and the
  milestone-runner prompt templates under `docs/prompts/`.
- `site/` — multi-page static website (Overview, Schema, Audit pipeline,
  Deduplication, Leakage, Label audit, Splits, Difficulty, Statistics, Dataset
  card, Results, Limitations, Roadmap, Milestones) + shared `style.css`.
- `scripts/` — `run_all_milestones_with_claude.sh` (one milestone per run, with
  commit/clean-tree/pytest gates and a cost guard), `00_check_environment.py`,
  `serve_site.sh`.
- `README.md`, `LICENSE` (MIT), `.gitignore`.
- Directory tree for `datasets/fixtures`, `datasets/releases`, and `results/`.

## Validation

- `python3 -m pytest -q` — all scaffold tests pass.
- `PYTHONPATH=src python3 -m eval_dataset_forge check-env` — reports deps.
- `python3 scripts/00_check_environment.py` — environment check passes.

## Notes / decisions

- The environment is Python 3.12 and PEP-668 externally-managed; dependency
  install uses `--break-system-packages` (documented in `REPRODUCIBILITY.md`).
- CLI is `argparse`-based (stdlib) to keep the footprint small; heavy imports are
  lazy so `check-env` works before the scientific stack is installed.
- Generated `results/` artifacts are gitignored; a single small demo release may
  be committed later to show the release format.

## Status

Complete (prepared as launch scaffold). Next: Milestone 1 — Core schemas.
