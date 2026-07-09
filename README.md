# eval-dataset-forge

A CPU-first framework for **constructing, auditing, versioning, and
stress-testing evaluation datasets** for LLM apps, agents, scientific workflows,
and retrieval systems.

Raw examples go in; **audited, versioned benchmark releases** come out — with
schema validation, deduplication, leakage/contamination checks, label-quality
audits, difficulty calibration, split construction, bootstrap uncertainty
estimates, dataset cards, and reproducible release manifests.

> Status: **Milestone 13 — Integration contracts.** The audit subsystems (schemas →
> dedup → leakage → labels → difficulty → splits → statistics → release → card →
> smoke audit) are implemented and scored against planted ground truth, and five
> downstream consumers now have executable contracts (`evalforge gate`). Remaining:
> the portfolio report (M14) and final cleanup (M15). See [`docs/PLAN.md`](docs/PLAN.md).

## The problem

A benchmark that reports a single accuracy number hides most of what determines
whether that number means anything. Datasets routinely contain **duplicates**
(inflating sample size), **leakage** (the answer is in the prompt, or the test
item is in the train set), **label noise** (missing/conflicting/ambiguous gold
labels), **split artifacts** (difficulty or source distribution differs across
splits), **difficulty imbalance**, and **unstable rankings** (the winner flips
under resampling). eval-dataset-forge turns each of these into a measured,
reported property of a dataset.

## The idea

A benchmark dataset should not be a bare JSONL file — it should ship as a
**dataset + evidence bundle**:

```
examples.jsonl          the data (schema-validated)
manifest.json           version, hashes, splits, audit summary, license notes
audit.json / audit.md   exact + near duplicates, leakage, label warnings, PII flags
dataset_card.md         generated from manifest + audit, so its "known issues" cannot drift
splits/{train,dev,test} constructed with a documented, reproducible rationale
REPRODUCE.md            exact commands to rebuild the release
```

## Architecture

```
raw examples → validate → dedup → leakage → label-audit → difficulty
             → splits → statistics → report → versioned release
```

Each stage is a separable, independently tested module. The pipeline is
deterministic given a fixed input and seed, runs fully offline, and needs no GPU
and no paid API. See [`docs/PLAN.md`](docs/PLAN.md) §4 for the full diagram and
package layout.

## Quick start

This environment uses `python3` (Python 3.12). Install the small dependency set
(PEP-668 environment → `--break-system-packages`):

```bash
python3 -m pip install --user --break-system-packages pydantic numpy pyyaml
# optional accelerators (TF-IDF near-dup, tabular reports):
python3 -m pip install --user --break-system-packages pandas scikit-learn

python3 scripts/00_check_environment.py            # environment + deps check
python3 -m pytest -q                               # CPU-only test suite
PYTHONPATH=src python3 -m eval_dataset_forge info  # CLI
```

## CLI (target)

```bash
evalforge check-env
evalforge validate     --dataset datasets/fixtures/qa_toy/examples.jsonl
evalforge audit        --dataset datasets/fixtures/qa_toy/examples.jsonl --out results/audits/qa_toy/
evalforge build-splits --dataset datasets/fixtures/qa_toy/examples.jsonl --strategy stratified --out datasets/releases/qa_toy_v0.1/
evalforge dataset-card --release datasets/releases/qa_toy_v0.1.0 --out results/reports/qa_toy_dataset_card.md
evalforge smoke        --out results/reports/smoke_audit.md
evalforge gate         --release datasets/releases/qa_toy_v0.1.0 --profile llm-regression
```

All 13 subcommands are implemented: `check-env`, `version`, `info`, `validate`,
`audit`, `build-splits`, `difficulty`, `stats`, `release`, `verify-release`,
`dataset-card`, `smoke`, `gate`.

## Fixture datasets

Small, synthetic, **deliberately flawed** datasets so the audits have real
defects to find (no real personal or clinical data):

`qa_toy` · `agent_tasks_toy` · `retrieval_toy` · `bio_toy` · `safety_toy`

## Design constraints

- CPU-first; runs on a general instance (e.g. `m5.2xlarge`). No GPU.
- Fully **offline** by default; **no paid API** anywhere in this version.
- Small, honest dependency set with pure-Python fallbacks.
- Privacy-safe fixtures; redaction + PII-like detection built in.
- No secrets, virtualenvs, caches, large datasets, or cached API responses
  committed.
- Honest about heuristics: every approximate method documents its failure modes.

## Documentation & website

Full docs under [`docs/`](docs/): methodology, dataset schema, deduplication,
leakage, label audit, splits, difficulty, statistics, releases, dataset card,
[integration contracts](docs/INTEGRATION.md), results, limitations, safety,
reproducibility, and the paper path. A multi-page static site lives under
[`site/`](site/):

```bash
./scripts/serve_site.sh   # http://localhost:9004/
```

## How it is built

The repository is constructed one milestone per Claude Code run:

```bash
./scripts/run_all_milestones_with_claude.sh 1 15
```

The runner enforces, between milestones, that a new commit was created, the tree
is clean, and the tests pass. See [`docs/PLAN.md`](docs/PLAN.md) §10–13.

## Intended uses

- An Anthropic-relevant SWE/research portfolio artifact.
- A reusable **dataset-quality layer** for agent/eval projects.
- A benchmark-building tool for adjacent projects (code-agent-runtime,
  BioAgentBench, LLM regression/observability, scientific-claim-verifier,
  agent-control-bench). Each is an executable profile of `evalforge gate`; see
  [`docs/INTEGRATION.md`](docs/INTEGRATION.md). None of them is a dependency.

## Limitations

The fixtures are synthetic; most checks are heuristic; contamination can only be
measured against text we are given; statistics quantify sampling noise, not
representativeness. This is **not** a certifier of universal benchmark quality and
**not** a medical/legal/security authority. See
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and
[`docs/SAFETY.md`](docs/SAFETY.md).

## License

MIT — see [`LICENSE`](LICENSE).
