# Statistical uncertainty & ranking stability

> Status: design (implemented in Milestone 9).

A single accuracy number without an uncertainty estimate invites over-reading.
This module quantifies how much a benchmark's verdicts can be trusted, using
transparent resampling methods. Inputs are per-example scores for one or more
systems (synthetic fixture scores in the offline demo; real eval outputs in use).

## Methods

### Bootstrap confidence intervals

For a metric (e.g. accuracy), resample examples with replacement `B` times
(default `B=1000`, fixed seed), recompute the metric on each resample, and report
the percentile confidence interval (`ci_low`, `ci_high`) around the point
estimate. Emitted as a `BootstrapResult`. This is the standard nonparametric
bootstrap; it assumes examples are exchangeable (which the dedup/leakage checks
help ensure).

### Paired comparison

To compare two systems, resample the *same* examples for both and look at the
distribution of the score **difference**. Report the fraction of resamples in
which system A beats system B and a CI on the difference. Pairing controls for
example-level difficulty and is more sensitive than comparing marginal CIs.

### Ranking stability

Given ≥2 systems, resample examples repeatedly and record the induced ranking
each time. Report how often the top-ranked system stays on top and how often the
full ordering is preserved. A ranking that flips frequently is a warning that the
benchmark cannot separate these systems at this sample size.

We also report ranking sensitivity to **removing** examples (leave-a-fraction-out)
to expose orderings that hinge on a handful of items.

### Small-N / instability warnings

The audit emits an explicit warning when the dataset is too small, or the effect
size too small relative to the CI width, for the reported ranking to be stable.
The threshold and rationale are documented in the finding rather than hidden.

## Deliberate non-overclaiming

- We report percentile bootstrap intervals and resampling frequencies — simple,
  inspectable quantities. We do not claim exact coverage or apply corrections we
  cannot justify on this data.
- Multiple-comparison issues (ranking many systems) are noted, not silently
  ignored.
- These methods quantify sampling uncertainty over the *given* examples; they say
  nothing about whether those examples are representative of the real task — that
  is what the rest of the audit is for.
