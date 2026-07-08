# Claude Code instructions for this repository

This repository is a research-grade engineering project: a CPU-first framework
for constructing, auditing, versioning, and stress-testing **evaluation
datasets** for LLM apps, agents, scientific workflows, and retrieval systems.

Read this file and [`docs/PLAN.md`](docs/PLAN.md) before doing any work.

## Public framing

eval-dataset-forge turns a raw JSONL/YAML dataset into an **audited, versioned
benchmark release**: schema validation, deduplication, leakage/contamination
checks, label-quality audit, split construction, difficulty calibration,
bootstrap uncertainty estimates, a dataset card, and a reproducible release
manifest.

This is not a toy dataset script and not a learning demo. It is a small but real
research/engineering platform. It does **not** claim to certify universal
benchmark quality — it surfaces specific, common, checkable defects, and is
honest about what each heuristic can miss.

## Operating principles

- Build milestone by milestone (see `docs/PLAN.md` §10).
- Commit locally after each coherent milestone. **Do not push.**
- Do not wait for human review. Create a review placeholder per milestone, then
  continue automatically.
- Keep default tests CPU-only and free. Never require paid LLM API calls.
- Everything must run fully **offline** using the small committed fixtures. Tests
  must not require internet.
- Keep the dependency set small and honest: hard deps are `pydantic`, `numpy`,
  `pyyaml`; `pandas` and `scikit-learn` are optional accelerators with
  pure-Python fallbacks. Do not add heavy or networked dependencies.
- Optional paid-API adapters stay disabled by default and must ship with a
  dry-run mode, a response cache, a token/cost estimator, and a hard max-cost
  guard before they can make any real call.
- Prefer deterministic mock adapters (`MockLLMLabeler`) for anything that would
  otherwise call a model.
- Be honest about limitations. Label heuristic/approximate methods as such and
  document their failure modes. Avoid marketing language; documentation should be
  precise, technical, and readable — not a learning diary.

## Safety & privacy

- Use only synthetic or public-safe fixtures. No real personal, clinical, or
  patient data ever enters the repo.
- Biomedical examples stay non-clinical and non-patient-specific.
- Provide and use redaction + PII-like detection utilities in the validation
  layer.
- This is a benchmark-quality tool, not a medical/legal/security authority.

## Do not commit

Secrets, API keys, `.env`, `.claude/settings.local.json`, virtualenvs, caches
(`__pycache__`, `.pytest_cache`, `.ruff_cache`), `node_modules`, model weights,
large datasets, cached API responses, or generated junk. Generated
audit/report/processed artifacts under `results/` are gitignored (reproducible
from fixtures); one small demo release under `datasets/releases/` may be
committed to show the release format.

## Quality bar

Every major claim must be backed by at least one of: a test, a reproducible
example command, a result table, a documented limitation, or an explicit
non-goal. The repo should be understandable to a strong senior engineer in 5
minutes and survive deeper technical questioning after 30.

## Required behavior after each milestone

1. Run the relevant tests (`python3 -m pytest -q`) and any milestone validation
   commands.
2. Update the static website under `site/` (including `milestones.html`).
3. Update `docs/RESULTS.md` or the relevant docs if results changed.
4. Create/update the milestone note under `docs/milestones/NN-<slug>.md`.
5. Create/update the human-review placeholder under
   `docs/review_notes/NN-<slug>-human-review.md`.
6. Commit locally (do not push).
7. Print: milestone title, commit hash, tests run, website pages updated, review
   placeholder path, and the next milestone.

Do not stop for review unless continuing would risk secrets, paid API spend,
data loss, or repository corruption.

## Human-review placeholder format

`docs/review_notes/NN-<slug>-human-review.md`:

```
# Human review: <milestone title>

TL;DR
What changed
What should be checked later
Technical risks
Commands to verify
Human feedback
<!-- Human will fill this later. -->
```

Do not claim human review happened.

## Website requirements

Maintain the multi-page static site under `site/`. Required pages:

- `index.html` (Overview)
- `schema.html` (Dataset schema)
- `audit-pipeline.html` (Audit pipeline)
- `deduplication.html`
- `leakage.html`
- `label-audit.html`
- `splits.html`
- `difficulty.html`
- `statistics.html` (Statistical uncertainty)
- `dataset-card.html` (Example dataset card)
- `results.html`
- `limitations.html`
- `roadmap.html`
- `milestones.html` (per-milestone explanations — update the milestone's status
  each time)

Each content page should carry: a brief explanation, a deeper technical
explanation, current implementation status, relevant files, current test/result
status, and open questions or limitations. Update the site at every milestone.
Serve locally with `./scripts/serve_site.sh` (port 9004).

## Permissions (autonomous runs)

`.claude/settings.json` is committed and defines the permission contract for
milestone runs: `defaultMode: acceptEdits`, `disableBypassPermissionsMode`, an
`allow` list of safe tools (Read/Edit/Write/Grep/Glob and common `python3`,
`pytest`, `pip`, `git`, and file commands), and a `deny` list that blocks
`git push`, `sudo`, `curl`/`wget`/`nc`/`ssh`/`scp`, `rm`, and reads of secret
paths (`.env`, `~/.aws`, `~/.ssh`, `*.pem`, …). This lets
`scripts/run_all_milestones_with_claude.sh` run without prompts while keeping the
"do not push / no network / no secrets" guarantees enforced by policy, not just
convention. `.claude/settings.local.json` (personal overrides) stays gitignored.
Never use `--dangerously-skip-permissions`.

## Commands

```bash
python3 -m pytest -q                              # CPU-only tests
PYTHONPATH=src python3 -m eval_dataset_forge info # CLI
python3 scripts/00_check_environment.py           # env + dependency check
ruff check .                                       # lint (if installed)
```
