# Split construction

> Status: design (implemented in Milestone 7).

A split is a modelling decision, not a coin flip. Each strategy encodes an
assumption about how the benchmark will be used, and each has tradeoffs. The
split builder records a `SplitSpec` (strategy, seed, ratios, group key,
constraints) so the rationale is reproducible and auditable.

## Strategies

1. **Random.** i.i.d. shuffle with a fixed seed. Baseline; assumes examples are
   independent. Fails when near-duplicates or grouped items exist (they leak
   across splits).
2. **Stratified by task family.** Preserve the proportion of each `task_family`
   in every split, so the test set is representative. Prevents a split from
   accidentally over-weighting one family.
3. **Stratified by difficulty.** Preserve the difficulty-bucket distribution
   (see [`DIFFICULTY.md`](DIFFICULTY.md)) across splits, so a test gain reflects
   capability rather than an easier test set.
4. **Source-grouped.** Keep all examples sharing a `source_id` (or other group
   key) together in a single split. Prevents train/test leakage through shared
   provenance.
5. **Adversarial (near-duplicate aware).** Ensure that near-duplicate clusters
   are not split across train and test — the harder, honest arrangement. Uses the
   deduplication clusters to constrain assignment. This is the recommended
   default for a leakage-resistant test set.
6. **Temporal.** Order by `created_at` and hold out the most recent items,
   simulating deployment on future data / distribution shift.

## Guarantees and checks

After construction, the builder re-runs leakage checks across the produced splits
and reports:

- split sizes and ratios;
- task-family / difficulty / source distributions per split;
- any residual cross-split leakage (should be zero for the grouped/adversarial
  strategies; reported if not).

## Known limitations

- Stratification is best-effort under small-N: exact proportions may be
  unreachable with few examples, and the builder reports the achieved
  distribution rather than silently forcing it.
- Temporal splits require trustworthy `created_at` values; missing timestamps
  fall back to insertion order with a warning.
