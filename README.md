# eval-dataset-forge

A CPU-first framework for **constructing, auditing, versioning, and stress-testing
evaluation datasets** for LLM apps, agents, scientific workflows, and retrieval systems.

Raw examples go in; an **audited, versioned benchmark release** comes out — schema
validation, deduplication, leakage and contamination checks, a label-quality audit,
difficulty calibration, split construction, bootstrap uncertainty estimates, a dataset
card, and a reproducible release manifest.

Everything runs offline on a CPU. No GPU, no network, no paid API — not as a fallback
mode, but as the only mode.

> **Status: complete — all milestones (0–15, through final cleanup) have landed.** The
> pipeline is implemented end to end and scored against planted ground truth; five
> downstream consumers have executable contracts; and the documentation is audited by the
> repository itself. The full gate runs clean offline (1852 tests). See
> [`docs/PLAN.md`](docs/PLAN.md) and [`docs/RESULTS.md`](docs/RESULTS.md).

---

## The problem

A benchmark that reports a single accuracy number hides most of what determines whether
that number means anything.

Datasets routinely contain **duplicates** that inflate the apparent sample size,
**leakage** where the answer sits in the prompt or the test item sits in the train set,
**label noise** that caps measurable accuracy, **split artifacts** that turn distribution
shift into an apparent capability gain, **difficulty imbalance** that saturates the
benchmark, and **unstable rankings** where the winner flips under resampling.

None of these are visible in the JSONL file. All of them change the conclusion.

This project makes each one a measured, reported property of a dataset. It does **not**
certify benchmark quality in any general sense; it surfaces specific, common, checkable
defects, and documents what each check misses.

## The idea

A benchmark dataset should not be a bare file. A trustworthy release is a **dataset plus
an evidence bundle**:

```text
examples.jsonl          the data, schema-validated
manifest.json           version, hashes, splits, audit summary, license notes
audit.json / audit.md   duplicates, leakage, label warnings, PII flags
dataset_card.md         generated from manifest + audit, so "known issues" cannot drift
splits/{train,dev,test} constructed by a documented, reproducible strategy
REPRODUCE.md            the exact command, and the environment it needs
```

## Architecture

```text
raw examples → validate → dedup → leakage → label-audit → difficulty
             → splits → statistics → report → versioned release
```

Each stage is a separable, independently tested module. Randomized steps take an explicit
seed. Reproduction additionally requires a fixed **environment**, for a reason measured in
Milestone 13 and described under [Limitations](#limitations). See
[`docs/PLAN.md`](docs/PLAN.md) §4 for the full diagram and package layout.

## Quick start

Python 3.12. Install the small dependency set (this is a PEP-668 environment, hence
`--break-system-packages`):

```bash
python3 -m pip install --user --break-system-packages pydantic numpy pyyaml
# optional: pandas (tabular reports) and scikit-learn (TF-IDF near-duplicate detection).
# scikit-learn is not an accelerator — see Limitations.
python3 -m pip install --user --break-system-packages pandas scikit-learn

python3 scripts/00_check_environment.py            # environment + dependency check
python3 -m pytest -q                               # CPU-only test suite
PYTHONPATH=src python3 -m eval_dataset_forge info  # CLI
```

The documentation spells the CLI `evalforge`, which `pyproject.toml` installs as a console
script. Without `pip install -e .` the same program is `PYTHONPATH=src python3 -m
eval_dataset_forge`. Both spellings are checked against the real argument parser by
[`scripts/verify_docs.py`](scripts/verify_docs.py).

## A worked example

`qa_toy` is one of 5 committed fixture datasets — small, synthetic, and **deliberately
flawed**, so the audits have real defects to find. Auditing it:

```bash
PYTHONPATH=src python3 -m eval_dataset_forge audit \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --out results/audits/qa_toy/
```

```text
audit: qa_toy @ unversioned (audit-a271768e2254cd69)
  examples     : 14 (train=8, dev=1, test=5)
  splits       : observed
  blocks       : 11 independent (2 contain a duplicate cluster)
  near method  : tfidf
  findings     : exact_duplicates=1, near_duplicates=1, leakage_flags=4,
                 label_warnings=8, pii_flags=0, recommendations=7

recommendations (7):
  - 2 duplicate cluster(s) straddle a split boundary: a test score is partly a
    memorization score. Rebuild with --strategy adversarial.
  - 1 example(s) contain their gold answer in the prompt; they measure copying,
    not the task.
  - 4 example(s) have a missing, conflicting, or unadjudicated gold answer.
  - the difficulty heuristic's ranking is not distinguishable from a random
    ranking of the declared tags on this corpus; do not read its buckets as
    difficulty labels (docs/DIFFICULTY.md).
  - 14 examples form 11 independent block(s); a bootstrap that resamples rows
    will under-cover. Use `evalforge stats --resample-unit block` (the default).
```

Fourteen rows, eleven independent blocks, and a recommendation that the last number is the
one a confidence interval may use. That is the whole thesis in one report.

## What it finds

Scored against the defects each fixture plants in its own `defects.yaml` — ground truth,
not a bare count. Across a pooled corpus of 56 examples the pipeline recovers **40 of 40
detectable planted defects, 0 missed, 5 unexplained assertions**.

That number, alone, is worthless. A detector that flags *everything* also scores recall
1.000, and [`evalforge smoke`](docs/RESULTS.md) puts exactly that degenerate detector
through the identical scorer, every run:

| channel | pipeline precision | flag-everything control |
|---|---|---|
| `dedup.exact` | 1.000 | 0.014 |
| `dedup.near` | 0.571 | 0.017 |
| `leakage.train_test_overlap` | 0.727 | 0.031 |
| `labels.missing` | 1.000 | 0.071 |
| `labels.ambiguous` | 0.833 | 0.089 |

Recall bounds what a detector *misses* and says nothing about what it *invents*. Every
recall in this repository is printed beside a precision, and the five unexplained
assertions are listed by id.

Selected results, each reproducible offline and each with a documented failure mode:

- **A duplicate cluster is not one example, and the textbook bootstrap counts it as one.**
  56 examples are worth ≈36 independent ones. The i.i.d. percentile interval covers 0.847
  where it promises 0.95; the cluster bootstrap recovers 0.884. ([`STATISTICS.md`](docs/STATISTICS.md))
- **One duplicate cluster decides the demo leaderboard.** `sys_a` leads `sys_b` by 3.6
  accuracy points; deleting 2 of 56 examples reverses it, and those two are a single
  near-duplicate cluster. Detecting a gap that size would need ≈4500 examples.
- **The difficulty heuristic is a task-family detector.** Spearman ρ = +0.097 (p = 0.497)
  against declared difficulty — not distinguishable from a random ranking. 43% of its
  variance is task family. Reported as **not calibrated**, which is the honest answer.
  ([`DIFFICULTY.md`](docs/DIFFICULTY.md))
- **The split seed moves the headline by 3.6× the effect it must resolve.** Holding the
  corpus and every per-example score fixed, `sys_a`'s test accuracy ranges over
  [0.364, 1.000] across seeds. So a release carries two identities: `content_id` ("the same
  examples?") and `eval_id` ("comparable scores?"). ([`RELEASES.md`](docs/RELEASES.md))
- **Deduplication is not composable.** Merging five disjoint fixtures that share *no*
  duplicate creates a duplicate cluster inside one of them, because TF-IDF's idf term is
  computed over the collection. There are 44 independent blocks in the pooled corpus, but
  45 when the five are audited separately and summed.
- **The "optional accelerator" is a different detector.** ([Limitations](#limitations))

Full tables: [`docs/RESULTS.md`](docs/RESULTS.md).

## The release, and its card

```bash
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset-id qa_toy --strategy adversarial \
  --out datasets/releases/
```

The committed demo release [`datasets/releases/qa_toy_v0.1.0`](datasets/releases/qa_toy_v0.1.0)
passes all 10 `verify-release` checks, and the command printed in its own `REPRODUCE.md`
regenerates it byte for byte.

Its [dataset card](datasets/releases/qa_toy_v0.1.0/dataset_card.md) is *generated* from the
manifest and the audit, never hand-written, so its "Known issues" section cannot drift from
the evidence beside it:

| field | value |
|---|---|
| examples | 14 |
| independent blocks | **11** |
| blocks in `test` | **2** |
| blocks straddling a split boundary | 0 (adversarial strategy) |

> Known issues — 1 exact duplicate cluster: `qa-001 + qa-002`. 1 near-duplicate cluster
> above the tfidf threshold: `qa-003 + qa-004 + qa-012` (heuristic — a paraphrase below the
> threshold is not here). 1 answer-in-prompt finding: `qa-007`. 1 missing gold answer:
> `qa-010`. …

A reader spends the first number as a sample size. The number that governs a confidence
interval on a test score is the third. The card prints both, in that order, on purpose.

## CLI

```bash
evalforge check-env
evalforge validate     --dataset datasets/fixtures/qa_toy/examples.jsonl
evalforge audit        --dataset datasets/fixtures/qa_toy/examples.jsonl --out results/audits/qa_toy/
evalforge build-splits --dataset datasets/fixtures/qa_toy/examples.jsonl --strategy stratified --out datasets/releases/qa_toy_v0.1/
evalforge dataset-card --release datasets/releases/qa_toy_v0.1.0 --out results/reports/qa_toy_dataset_card.md
evalforge smoke        --out results/reports/smoke_audit.md
evalforge gate         --release datasets/releases/qa_toy_v0.1.0 --profile llm-regression
```

All 13 subcommands are implemented: `check-env`, `version`, `info`, `validate`, `audit`,
`build-splits`, `difficulty`, `stats`, `release`, `verify-release`, `dataset-card`,
`smoke`, `gate`.

## Reproduce everything

```bash
python3 -m pytest -q                                          # the full CPU-only suite
python3 scripts/validate_fixtures.py                          # fixtures parse and validate
python3 scripts/verify_docs.py                                # the docs match the code
PYTHONPATH=src python3 -m eval_dataset_forge smoke \
  --out results/reports/smoke_audit.md \
  --fail-on-missed-defect --fail-on-control-violation         # the whole pipeline, scored
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0                   # integrity of the demo release
```

Every number in the documentation is either recomputed by `scripts/verify_docs.py` from the
code and the committed artifacts, or is a dated record in a milestone note. See
[`docs/CLAIMS.md`](docs/CLAIMS.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Integration

Five downstream consumers have executable contracts — `code-agent-runtime`, `BioAgentBench`,
an LLM regression/observability suite, `scientific-claim-verifier`, and `agent-control-bench`.
Each is a profile of `evalforge gate`, which turns a consumer's requirements into an exit
code. None of the five is a dependency.

All five **reject** the demo release, which is the only correct answer: `qa_toy` is a
deliberately flawed fixture, and a gate that passed it would be broken. The positive control
is a repaired release that passes. See [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

## Limitations

The honest summary is that most of this is heuristic, and the heuristics are labelled.

- **Contamination is measurable only against text the auditor holds.** Every leakage check
  here is intra-corpus. Whether a model was trained on the benchmark is not knowable from
  the benchmark, and nothing in this repository claims otherwise.
- **The fixtures are synthetic, and the thresholds were tuned on them.** Recall figures are
  resubstitution estimates: the pipeline finds what it was built to find, on the data it was
  built against. There is no held-out corpus.
- **Label *truth* is invisible to the label audit.** Three deliberately mislabeled examples
  are found by no check, and a regression test asserts they stay invisible so the limitation
  cannot rot into a silent claim of coverage.
- **`scikit-learn` is not an accelerator.** Its presence resolves near-duplicate detection to
  TF-IDF instead of Jaccard; the two disagree about which examples are duplicates; duplicates
  become blocks; the adversarial split will not cut a block; and `eval_id` folds the split in.
  Rebuilding from identical bytes, an identical command, and an identical seed yields a
  different `eval_id` and a different held-out set. **The environment is an input.**
  Verification, which recomputes hashes and runs no detector, is environment-independent.
- **Statistics quantify sampling noise, not representativeness.** A tight interval on an
  unrepresentative corpus is a precise measurement of the wrong thing.
- **This is a benchmark-quality tool, not a medical, legal, or security authority.**

Full list: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/SAFETY.md`](docs/SAFETY.md).

## Research path

Open directions, stated as questions rather than results: a standardized **reliability
profile** for benchmarks and whether its metrics predict unstable rankings; **leakage →
ranking-change** experiments quantifying how much reported progress is a dataset artifact;
difficulty calibration against **item-response** data rather than surface features;
required-N as a standard release artifact; and embedding-based contamination detection as an
opt-in mode with a measured cost/coverage tradeoff. See [`docs/PAPER_PATH.md`](docs/PAPER_PATH.md).

Non-goals: building a leaderboard, certifying benchmarks as clean, and any method that needs
private data to reproduce.

## Documentation & website

Full docs under [`docs/`](docs/): [methodology](docs/METHODOLOGY.md),
[schema](docs/DATASET_SCHEMA.md), [deduplication](docs/DEDUPLICATION.md),
[leakage](docs/LEAKAGE.md), [label audit](docs/LABEL_AUDIT.md), [splits](docs/SPLITS.md),
[difficulty](docs/DIFFICULTY.md), [statistics](docs/STATISTICS.md),
[releases](docs/RELEASES.md), [dataset card](docs/DATASET_CARD.md),
[integration contracts](docs/INTEGRATION.md), [claims](docs/CLAIMS.md),
[results](docs/RESULTS.md), [limitations](docs/LIMITATIONS.md), [safety](docs/SAFETY.md),
[reproducibility](docs/REPRODUCIBILITY.md), and the [paper path](docs/PAPER_PATH.md).

A multi-page static site lives under [`site/`](site/):

```bash
./scripts/serve_site.sh   # http://localhost:9004/
```

## How it is built

One milestone per Claude Code run:

```bash
./scripts/run_all_milestones_with_claude.sh 1 15
```

The runner enforces, between milestones, that a new commit exists, the tree is clean, and
the tests pass. See [`docs/PLAN.md`](docs/PLAN.md) §10–13.

## License

MIT — see [`LICENSE`](LICENSE).
