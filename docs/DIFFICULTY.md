# Difficulty calibration

> Status: design (implemented in Milestone 8).

A benchmark that is all-easy or all-hard stops discriminating between systems.
We estimate a **transparent, model-free difficulty proxy** for each example so a
dataset's difficulty distribution can be inspected and balanced. This is a
proxy, not ground truth.

## Signals

The heuristic difficulty score is a documented, weighted combination of:

- **Input length** — longer prompts tend to be harder (tokens/characters).
- **Expected-output complexity** — length / structure of the gold answer.
- **Reasoning-step count** — heuristic count of required steps (enumerations,
  multi-part questions, conjunctions).
- **Evidence-item count** — number of evidence/retrieval items an example
  references.
- **Tool-use requirement** — whether the task metadata indicates a tool/lookup is
  needed.
- **Ambiguity tags** — presence of `ambiguous`/`hard` tags.
- **Manually assigned fixture difficulty** — when a fixture carries a
  hand-labelled difficulty, it is recorded alongside the heuristic and used to
  sanity-check it.

Each `DifficultyScore` records the overall `score`, the assigned `bucket`
(`easy`/`medium`/`hard` by quantile or fixed cutoffs), the per-signal
`components`, and the `method`, so the number is fully explainable.

## Calibration

"Calibration" here means: check that the heuristic buckets line up with any
available signal of true difficulty (manual fixture labels, or — in a future,
optional mode — observed pass rates from mock/real systems). We report the
agreement and, where fixtures provide labels, a simple concordance summary. We do
**not** claim the heuristic recovers true difficulty.

## Reports

The audit emits the difficulty distribution (bucket → count) per dataset and per
split, so imbalance is visible and the stratified-by-difficulty split can act on
it.

## Known limitations

- The score is a proxy built from surface features; it can be fooled (a short
  prompt can be very hard; a long one can be trivial).
- Weights are hand-chosen and documented; they are a starting point, not a
  learned model.
- Item-response-theory-style difficulty estimation from real response data is a
  future direction ([`PAPER_PATH.md`](PAPER_PATH.md)), not part of the offline
  default.
