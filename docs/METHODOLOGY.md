# Methodology

## Why a raw JSONL file is not a benchmark

A JSONL file is a container. A *benchmark* is a claim: "a system that scores X on
this data is better than a system that scores Y." That claim is only as good as
the data's structural properties, and those properties are usually invisible in
the file itself. Common, well-documented failure modes:

- **Duplication.** Exact or near-duplicate examples inflate apparent sample size
  and let a model's memorization of one instance count many times.
- **Leakage / contamination.** The answer appears in the prompt; the same
  example appears in both train and test; a retrieved passage contains a label
  artifact; the test set overlaps with public pretraining corpora.
- **Label noise.** Missing, conflicting, ambiguous, or inconsistently formatted
  gold labels put a ceiling on measurable accuracy and bias comparisons.
- **Split artifacts.** Difficulty, source, or task-family distributions differ
  across splits, so a "test" gain reflects distribution shift, not capability.
- **Difficulty imbalance.** A benchmark dominated by trivial items saturates and
  stops discriminating between systems.
- **Unstable rankings.** With few examples or a small effect size, the ranking of
  two systems can flip under resampling — the reported ordering is noise.

eval-dataset-forge makes each of these a measured, reported property of a dataset
rather than an assumption.

## Operating stance

- **Transparent over clever.** Every check is a documented, inspectable
  heuristic or exact computation. We prefer a simple method whose failure modes
  are understood to a sophisticated one whose behavior is opaque.
- **Exact vs. heuristic is always labelled.** Exact-duplicate and ID-uniqueness
  checks are exact. Near-duplicate, difficulty, and answer-leakage checks are
  heuristic; each ships with a documented notion of what it flags and what it
  misses (false positives and false negatives).
- **Findings, not verdicts.** The tool emits findings and recommendations. It
  does not silently mutate a dataset or certify it as "clean." A human decides
  what to fix.
- **Determinism.** Given a fixed input and seed, the pipeline is reproducible.
  Randomized steps (split construction, bootstrap) take an explicit seed.
- **Offline and CPU-only.** No step requires a network or a GPU. Deterministic
  mock adapters stand in for any model call in tests.

## Pipeline

The audit is a linear pipeline of separable stages, each independently tested:

```
load → validate → dedup → leakage → label-audit → difficulty → splits → statistics → report
```

A stage consumes validated `DatasetExample` objects and emits typed findings
(see [`DATASET_SCHEMA.md`](DATASET_SCHEMA.md)). The `AuditResult` aggregates all
findings plus distribution summaries and human-readable recommendations. The
release builder packages the data, manifest, splits, audit, and dataset card
into a versioned directory.

## What this methodology deliberately does not claim

- It does not certify that a dataset is "high quality" in any universal sense.
- It does not detect contamination against private pretraining corpora it cannot
  see; n-gram overlap is computed only against provided reference text.
- It does not replace human adjudication of ambiguous labels — it queues them.
- Its difficulty score is a proxy, not ground truth (see
  [`DIFFICULTY.md`](DIFFICULTY.md)).

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full list.
