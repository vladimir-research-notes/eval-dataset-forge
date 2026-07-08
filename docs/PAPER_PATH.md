# Paper path — honest future research directions

This file records where the project could go as a research contribution. It is a
roadmap of open questions, not a claim that any of them are solved.

## Motivating question

Can we give evaluation datasets a **reliability profile** — a compact, computable
summary of the ways a benchmark can mislead — and show empirically that acting on
that profile changes which system "wins"?

## Directions

1. **A reliability profile for benchmarks.** Define a small, standardized set of
   computable dataset-quality metrics (duplication rate, cross-split leakage,
   label-conflict rate, difficulty entropy, ranking-stability index) and study
   their distribution across a set of public benchmarks. Contribution: a
   reproducible profile + evidence about which metrics actually predict unstable
   rankings.

2. **Leakage → ranking-change experiments.** Take benchmarks with detectable
   leakage/duplication, remove the flagged items, and measure how often the
   system ranking changes. Contribution: quantify how much reported progress is
   attributable to dataset artifacts rather than capability.

3. **Difficulty calibration against response data.** Move from the surface-feature
   difficulty proxy to item-response-theory-style difficulty estimated from
   observed pass/fail data across many systems, and test whether the proxy is a
   usable pre-collection substitute. Contribution: a validated cheap proxy, with
   its error bars.

4. **Ranking-stability as a first-class benchmark statistic.** Formalize when a
   benchmark is "large enough" to separate two systems, and report required-N as
   a standard artifact of a release. Contribution: a practical stopping rule for
   benchmark size.

5. **Semantic duplication / contamination (opt-in).** Add embedding-based near-
   duplicate and contamination detection as an optional, clearly-bounded mode,
   and compare its yield against the cheap lexical methods. Contribution: a
   cost/coverage tradeoff study for offline vs. model-based detection.

## What would make this credible

- Reproducible pipelines and released audit artifacts for every claim.
- Ablations showing each metric adds signal.
- Honest reporting of false-positive/negative rates for every heuristic.
- No overclaiming: the tool measures specific defects, not "quality" in general.

## Non-goals for the paper

- Building a leaderboard.
- Certifying benchmarks as "clean."
- Any method that requires private/unshareable data to reproduce.
