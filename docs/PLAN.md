# eval-dataset-forge — Build Plan

This document is the project contract. It defines the research question, the
architecture, the on-disk formats, the audit dimensions, the release concept,
the milestone plan, and the quality gates. It is the source of truth that the
per-milestone Claude Code runs (`scripts/run_all_milestones_with_claude.sh`)
read before doing any work.

---

## 1. Research question

> How can we make evaluation datasets more trustworthy by systematically
> detecting duplication, leakage, label noise, split artifacts, difficulty
> imbalance, and unstable model rankings?

A benchmark that reports a single accuracy number hides most of what determines
whether that number means anything. Two systems can be separated by a gap that
vanishes under resampling; a "hard" test set can be trivially solved because the
answer leaks into the prompt; a 5-point improvement can be an artifact of 40
near-duplicate examples clustered in the test split. eval-dataset-forge treats
these failure modes as first-class, measurable properties of a dataset.

## 2. Thesis

A benchmark dataset should not be a bare JSONL file. A trustworthy release is a
**dataset + evidence bundle**:

1. schema validation
2. deduplication report (exact + near-duplicate)
3. leakage / contamination checks
4. label-quality audit
5. split construction rationale
6. difficulty calibration
7. bootstrap confidence intervals
8. adversarial / stress splits
9. dataset card
10. reproducible release manifest

Everything the tool asserts is backed by a test, a reproducible command, a
result table, or an explicitly documented limitation. Where a method is a
heuristic (most of them are), it is labelled as such and its failure modes are
written down. This project does **not** claim to certify universal benchmark
quality; it claims to surface specific, common, checkable defects.

## 3. Design constraints

- **CPU-first.** Target is a general AWS `m5.2xlarge`. No GPU anywhere.
- **Offline by default.** Every default command and every test runs with no
  network, using small committed fixtures.
- **No paid API required.** Deterministic mock adapters are first-class. Any
  future paid-API mode must ship with a dry-run mode, a response cache, a
  token/cost estimator, and a hard max-cost guard, and stays disabled by default.
- **Small, honest dependency set.** Hard deps: `pydantic` (schema validation),
  `numpy` (bootstrap statistics), `pyyaml` (YAML formats). Optional accelerators:
  `pandas`, `scikit-learn` (TF-IDF near-duplicate detection) — the core degrades
  to pure-Python fallbacks when they are absent.
- **Privacy-safe.** Only synthetic or public-safe fixtures. No real personal,
  clinical, or patient data. Redaction and PII-like detection utilities are part
  of the validation layer. Biomedical examples stay non-clinical and generic.
- **Clean repo.** No secrets, virtualenvs, caches, large datasets, cached API
  responses, or generated junk committed. Generated audit/report/release
  artifacts are reproducible and are gitignored except for one small demo release.

## 4. Architecture

```text
raw examples (JSONL/YAML fixtures)
        │
        ▼
  [datasets.loader]  ── parse ──►  DatasetExample[]  (pydantic-validated)
        │
        ▼
  [validation]   schema · ID uniqueness · PII/redaction · quality
        │
        ▼
  [dedup]        exact · normalized-text · near-duplicate (token/TF-IDF)
        │
        ▼
  [leakage]      train/test overlap · n-gram overlap · answer-in-prompt · retrieval contamination
        │
        ▼
  [labels]       missing · conflicting · ambiguous · format · adjudication queue
        │
        ▼
  [difficulty]   heuristic scoring · calibration · buckets
        │
        ▼
  [splits]       random · stratified · source-grouped · adversarial · temporal
        │
        ▼
  [statistics]   bootstrap CIs · paired comparison · ranking stability
        │
        ├──► AuditResult (pydantic model) ──► results/audits/<dataset>/
        │
        ▼
  [datasets.writer + manifest + versioning]
        │
        ▼
  [reports]  dataset_card (md/html) · markdown/html audit report
        │
        ▼
  Benchmark release:  datasets/releases/<id>_v<x.y>/
     ├─ examples.jsonl          (the data)
     ├─ manifest.json           (DatasetManifest: hashes, splits, counts)
     ├─ audit.json / audit.md   (AuditResult + human-readable report)
     ├─ dataset_card.md         (purpose, construction, known issues, non-goals)
     ├─ splits/{train,dev,test}.jsonl
     └─ REPRODUCE.md            (exact commands)
```

The pipeline is deterministic given a fixed input and seed. Every stage is a
separable module with its own tests, so a stage can be reasoned about (and
verified) in isolation.

### Package layout

```text
src/eval_dataset_forge/
  __init__.py          version + SCHEMA_VERSION
  schemas.py           pydantic models (Milestone 1)
  cli.py               argparse CLI: check-env, validate, audit, build-splits,
                       dataset-card, smoke

  datasets/            loader.py · writer.py · manifest.py · versioning.py
  validation/          schema.py · redaction.py · pii.py · quality.py
  dedup/               exact.py · fuzzy.py · near_duplicate.py
  leakage/             train_test_overlap.py · ngram_overlap.py ·
                       retrieval_overlap.py · contamination.py
  labels/              audit.py · disagreement.py · adjudication.py
  splits/              random.py · stratified.py · adversarial.py · temporal.py
  difficulty/          heuristics.py · calibration.py · buckets.py
  statistics/          bootstrap.py · confidence.py · ranking_stability.py
  reports/             dataset_card.py · markdown.py · html.py
  adapters/            mock_llm.py · api_stub.py
```

The CLI is `argparse`-based (stdlib) to keep the dependency footprint small;
`typer` is a permitted alternative but not required. Heavy imports are lazy so
`evalforge --help` and `evalforge check-env` work before the scientific stack is
installed.

## 5. Core data contracts

Formal definitions live in [`DATASET_SCHEMA.md`](DATASET_SCHEMA.md). Summary:

### A. DatasetExample

```text
example_id        stable unique id
source_id         provenance id of the originating record/file
task_family       qa | agent_task | retrieval | bio | safety | ...
input             the prompt / query / task statement
expected_output   gold answer / label / relevance judgement
metadata          free-form dict (source, notes, retrieval docs, ...)
labels            structured labels (may include multiple annotator labels)
difficulty_tags   manually assigned tags (easy/medium/hard, ambiguous, ...)
split             train | dev | test | unassigned
provenance        how the example was created (synthetic, adapted, ...)
created_at        ISO-8601 timestamp
version           schema/content version
```

### B. DatasetManifest (per release)

```text
dataset_id · version · created_at · source_files · example_count ·
schema_version · hashes · splits · audit_results · known_limitations · license_notes
```

### C. AuditResult

```text
audit_id · dataset_id · version · exact_duplicates · near_duplicates ·
leakage_flags · label_warnings · pii_flags · difficulty_distribution ·
split_statistics · recommendations
```

### D. Benchmark release (directory)

```text
examples.jsonl · manifest.json · audit.{json,md} · dataset_card.md ·
splits/*.jsonl · REPRODUCE.md
```

## 6. Audit dimensions

1. **Schema validity** — required fields, type checks, ID uniqueness, metadata
   consistency.
2. **Exact deduplication** — identical inputs, identical expected outputs,
   duplicate IDs, duplicate source records.
3. **Fuzzy / near deduplication** — normalized-text equality, token n-gram
   (Jaccard) similarity, optional TF-IDF cosine similarity.
4. **Leakage** — train/test exact overlap, high n-gram overlap across splits,
   expected answer appearing verbatim in the prompt, retrieved document
   containing a label artifact, cross-split contamination.
5. **Label audit** — missing labels, conflicting labels, ambiguous expected
   outputs, inconsistent label formats, examples needing adjudication.
6. **Difficulty calibration** — transparent heuristic score from input length,
   reasoning-step count, evidence-item count, tool-use requirement, ambiguity
   tags, plus any manually assigned fixture difficulty.
7. **Split quality** — split sizes, task-family / difficulty / source
   distributions, leakage across splits, adversarial split options.
8. **Ranking stability** — bootstrap confidence intervals for system scores,
   paired comparison, ranking sensitivity to example removal, and a warning when
   the dataset is too small to rank systems stably.

Every dimension documents whether it is exact or heuristic and what it can miss.

## 7. Fixture dataset families

Small, synthetic, deliberately flawed datasets so the audits have real defects to
find. All are safe to commit (no real personal/clinical data).

1. **qa_toy** — duplicate questions, paraphrased duplicates, mislabeled answers,
   ambiguous questions, train/test overlap.
2. **agent_tasks_toy** — easy/hard coding/research tasks, near-duplicates, tasks
   with leaked solutions, tasks with ambiguous scoring.
3. **retrieval_toy** — queries/documents/relevance labels with duplicated docs,
   query–document leakage, incomplete relevance labels, over-specific queries,
   paraphrased documents.
4. **bio_toy** — safe synthetic biomedical-style lookup/evidence tasks, entity
   labels, deliberate label conflicts. Non-clinical, non-patient-specific.
5. **safety_toy** — synthetic over-refusal / under-refusal / ambiguous-boundary
   examples. No real harmful instructions.

## 8. CLI contract (target)

```bash
evalforge check-env
evalforge validate     --dataset datasets/fixtures/qa_toy/examples.jsonl
evalforge audit        --dataset datasets/fixtures/qa_toy/examples.jsonl --out results/audits/qa_toy/
evalforge build-splits --dataset datasets/fixtures/qa_toy/examples.jsonl --strategy stratified --out datasets/releases/qa_toy_v0.1/
evalforge dataset-card --release datasets/releases/qa_toy_v0.1/ --out results/reports/qa_toy_dataset_card.md
evalforge smoke        --out results/reports/smoke_audit.md
```

## 9. Adapters

- **MockLLMLabeler** — deterministic, offline, used by tests.
- **ApiLabelerStub** — interface only, disabled by default; documents the
  dry-run / cache / cost-estimate / max-cost-guard requirements any real
  implementation must satisfy.

No paid API is required in this version.

## 10. Milestone plan

Each milestone is one coherent, committed unit. The runner does exactly one
milestone per `claude -p` invocation, verifies a new commit + clean tree +
passing tests, then proceeds.

| # | Milestone | Key deliverables |
|---|-----------|------------------|
| 0 | Scaffold & project contract | repo tree, package skeleton, CLI stub, `docs/PLAN.md`, `CLAUDE.md`, README, `.gitignore`, website, env check, first tests |
| 1 | Core schemas | pydantic models: DatasetExample, DatasetManifest, DatasetRelease, AuditResult, Deduplication/Leakage/LabelAudit findings, SplitSpec, DifficultyScore, BootstrapResult, DatasetCard + tests |
| 2 | Fixture datasets | qa/agent/retrieval/bio/safety toy sets with deliberate defects; validation script |
| 3 | Validation & redaction | schema validation, ID uniqueness, missing-field checks, fake-secret detection, email/token redaction, safe-data checks |
| 4 | Deduplication | exact, normalized-text, near-duplicate (token overlap / optional TF-IDF) + tests with known duplicates |
| 5 | Leakage checks | train/test overlap, prompt–answer leakage, n-gram overlap warnings, retrieval-document contamination |
| 6 | Label audit | missing / conflicting / ambiguous / format checks; adjudication queue export |
| 7 | Split builder | random, stratified (family/difficulty), source-grouped, adversarial (near-dups out of test) |
| 8 | Difficulty calibration | transparent heuristics + difficulty distribution reports |
| 9 | Statistics & ranking stability | bootstrap CIs, paired comparison, ranking stability under resampling, small-N warnings |
| 10 | Release builder | dataset JSONL, manifest, split files, audit summary, hashes, versioned release dir |
| 11 | Dataset card generator | Markdown + optional HTML cards |
| 12 | Smoke audit & results | full offline audit over all fixtures → `results/` + `docs/RESULTS.md` |
| 13 | Integration contracts | documented contracts for code-agent-runtime, BioAgentBench, LLM regression/observability, scientific-claim-verifier, agent-control-bench |
| 14 | Final portfolio report | README + docs to portfolio quality |
| 15 | Final cleanup | tests, lint, fixture validation, smoke audit, release build, report gen, hygiene scan, clean tree |

Detailed per-milestone notes live in [`milestones/`](milestones/). Each milestone
also writes a human-review placeholder under [`review_notes/`](review_notes/).

## 11. Quality gates

The repository is **not complete** unless all of the following hold:

- [ ] tests pass (`python3 -m pytest -q`)
- [ ] fixture datasets validate
- [ ] the smoke audit runs fully offline
- [ ] duplicate examples are detected
- [ ] leakage examples are detected
- [ ] label conflicts are detected
- [ ] at least one dataset release is built
- [ ] at least one dataset card is generated
- [ ] the bootstrap / ranking-stability demo runs
- [ ] docs explain methodology, limitations, safety, and reproduction
- [ ] no paid API is needed anywhere
- [ ] the working tree is clean
- [ ] nothing is pushed

## 12. Commit discipline

- Commit after each coherent milestone; do not push.
- Keep commits meaningful; avoid giant formatting-only commits.
- Public docs read as professional engineering/research documentation, not a
  learning diary.
- Any heuristic or approximate method is documented as such, with its failure
  modes.

## 13. How milestones are executed

The user launches milestones themselves via:

```bash
# One milestone per Claude Code run; verifies commit + clean tree + tests between each.
./scripts/run_all_milestones_with_claude.sh <START> <END>
# e.g. resume from milestone 1 through 15:
./scripts/run_all_milestones_with_claude.sh 1 15
```

Each run reads [`prompts/run-one-milestone-template.md`](prompts/run-one-milestone-template.md),
implements exactly one milestone, updates docs + website, writes the milestone /
review notes, commits locally, and stops. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
for the full reproduction path.

Permissions for autonomous runs are preconfigured in the committed
`.claude/settings.json`: `defaultMode: acceptEdits` plus an `allow` list of safe
tools and a `deny` list that blocks `git push`, `sudo`, network tools
(`curl`/`wget`/`nc`/`ssh`/`scp`), `rm`, and reads of secret paths. This lets the
runner proceed without prompts while the "do not push / no network / no secrets"
guarantees are enforced by policy. Override or add flags per run with
`CLAUDE_EXTRA_ARGS` (defaults to `--permission-mode acceptEdits`); never use
`--dangerously-skip-permissions`.
