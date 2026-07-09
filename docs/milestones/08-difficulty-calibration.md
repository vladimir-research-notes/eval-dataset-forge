# Milestone 8 — Difficulty calibration

## Summary

Land audit dimension #6: a transparent, model-free difficulty proxy from five documented
surface signals, three bucketing methods, per-dataset and per-split distribution reports,
and the `evalforge difficulty` CLI.

The distinguishing work was not the heuristic. It was (a) noticing that the design doc's
own signal list made the milestone **circular** — it proposed using the hand-assigned
difficulty as both a scoring input and the sanity check on it — and quantifying the
inflation that would have caused; (b) building a calibration harness rigorous enough to
**falsify the heuristic**, and then reporting that it did: the score's ranking is not
distinguishable from a random ranking of the declared labels; (c) showing by measurement
what the score *does* track — `task_family`, at 43% of its variance against 1% of the
declared label's; and (d) wiring the verdict into a `trustworthy` bit and a CI gate, so a
number the project cannot defend is never silently consumed downstream.

## Deliverables

- `src/eval_dataset_forge/difficulty/` package:
  - `heuristics.py` — `Signal` (name, extractor, cap, stated claim), the five extractors,
    `DEFAULT_WEIGHTS`, `resolve_weights()`, `weighted_components()`, `raw_score()`,
    `score_matrix()`.
  - `buckets.py` — `declared_bucket()`/`declared_ordinal()` (moved here from `splits`),
    `quantile_cutpoints()`, `marginal_cutpoints()`, `scale_counts()`, `assign_buckets()`,
    `bucket_distribution()`, `declared_distribution()`, `distribution_by_split()`.
  - `calibration.py` — `rankdata()`, `spearman()`, `permutation_p()` (Phipson–Smyth),
    `required_n()` (Fisher z), `eta_squared()`, `SignalDiagnostic`, `CalibrationReport`,
    `calibrate()`.
  - `__init__.py` — `DifficultyReport`, `score_dataset()`, `write_report()`,
    `format_report()`.
- `cli.py` — `evalforge difficulty` with `--bucketing`, `--weights`, `--seed`,
  `--resamples`, `--alpha`, `--out`, `--json`, `--fail-on-uncalibrated`.
- `splits/stratified.py` — now imports `declared_bucket`/`DIFFICULTY_ORDER`/`UNLABELLED`
  from `difficulty.buckets` and re-exports them. No behaviour change; 1245 prior tests
  pass unmodified.
- `tests/test_difficulty.py` — 163 tests: the signals, the a-priori rules, the
  explainability invariant, the non-circularity, bucketing and ties, the rank statistics
  against numpy references, the headline null with all six of its controls, a **positive
  control**, the honest negatives, determinism, byte-identical IO, and the CLI.
- Docs: `docs/DIFFICULTY.md` rewritten around the measurements; `docs/RESULTS.md` gains its
  difficulty section; `docs/LIMITATIONS.md` gains "the difficulty score does not measure
  difficulty". Website: `difficulty.html` rewritten; status updates to `milestones.html`,
  `audit-pipeline.html`, `index.html`, `results.html`, `limitations.html`.

## Design decisions

- **The calibration target is never a scoring signal.** `docs/DIFFICULTY.md` originally
  listed the `ambiguous`/`hard` tags and the manual fixture difficulty as scoring signals,
  and *also* proposed checking the score against those labels. Both were removed from the
  scorer. This is the milestone's single most consequential decision; see "The result worth
  reading twice".
- **Log-compression, not min–max.** `log1p(x)/log1p(cap)` keeps the score
  **dataset-independent**: no corpus statistic enters it, so a score means the same thing
  in every dataset, and adding an example can move a bucket boundary but never a score.
  Min–max would silently redefine "hard" on every insert.
- **The score is exactly the sum of its components — bit-for-bit.** Nothing is rounded.
  Rounding both to six places breaks the equality for examples whose five contributions
  round the same way, and with it the explainability claim.
- **Two extraction rules chosen a priori, on linguistic grounds, before any label was
  read.** Bare digits are not reasoning steps (else `"What is 2 + 2?"` scores two and the
  speed-of-light question scores zero). `tool_use` reads truthiness, not key presence
  (else `{"tools": []}` on every row makes every row harder). Picking whichever variant
  correlated better would have been fitting a 5-parameter model to 51 points.
- **Equal scores get equal buckets.** Every cut is `score < cut`, so no boundary splits a
  tie group — the corpus has two tie groups of four. The cost is that `quantile` cannot hit
  equal frequencies (it lands at 18/19/19) and the report states what it *achieved*.
- **`quantile` is the default because `fixed` degenerates.** The maximum score on this
  corpus is 0.658, so absolute thirds leave `hard` empty (45/11/0). That is a property of
  absolute cut points on a compressed scale; a warning fires rather than a silent zero.
- **`marginal` apportions before it matches.** The declared marginal is measured over 51
  labelled examples but applied to 56, so largest-remainder scales 29/19/3 → 32/21/3.
  Without it, all five undeclared examples fall into `hard`.
- **`declared_bucket` moved from `splits` to `difficulty`.** Difficulty is upstream of
  splits (`PLAN.md` §4), and the calibration target and the stratification facet must read
  the declaration identically or the two milestones would disagree about what `hard` means.
  `splits.stratified` re-exports it; a test asserts the two names are the same object.
- **Nothing is deleted or imputed.** Every example gets a bucket, including the ones the
  calibration says not to believe, because the release builder and dataset card must show a
  reader exactly what the audit saw. Examples with no declared difficulty are excluded from
  every calibration statistic, never imputed.
- **The verdict is a single bit with teeth.** `trustworthy` requires significance, ≥ 30
  labelled examples, *and* beating the majority baseline. `--fail-on-uncalibrated` exits 1,
  mirroring Milestone 7's `--fail-on-leakage`.

## Deviations from the design doc

- **Two scoring signals removed.** `PLAN.md` §6.6 and the old `DIFFICULTY.md` list
  "ambiguity tags" and "manually assigned fixture difficulty" as inputs to the score. Both
  are `difficulty_tags` entries and both are the calibration target. Keeping them would
  have made the milestone's central claim unfalsifiable.
- **A CLI subcommand was added.** `PLAN.md` §8's CLI contract does not list `difficulty`;
  the stage was to surface through `audit` (Milestone 12). But "difficulty distribution
  reports" is the milestone's literal deliverable, and a reproducible validation command is
  required by the process. `evalforge difficulty` is additive and does not reshape the
  parser; `audit` will still fold it in.
- **`buckets.py` owns `declared_bucket`.** `PLAN.md` §4 sketches `difficulty/{heuristics,
  calibration,buckets}.py`, which is exactly what shipped; the move of `declared_bucket`
  *into* it from `splits/stratified.py` is a layering fix, not a new module.

## The result worth reading twice

**The difficulty heuristic is a task-family detector wearing a difficulty label.**

Over the 51 labelled examples of the pooled corpus:

| quantity | value | reading |
|---|---|---|
| Spearman ρ (score vs declared) | **+0.097** (p = **0.497**) | not distinguishable from chance |
| within-family ρ (size-weighted) | +0.030 | collapses when family is held constant |
| η² of the **score** by `task_family` | **0.434** | the score is 43% task family |
| η² of the **declared label** by `task_family` | **0.014** | declared difficulty is 1% task family |
| marginal-matched bucket accuracy | 0.529 | below "call everything easy" (0.569) |
| weight-simplex sweep (2000 draws) | median +0.024, max +0.286 | 94% inside \|ρ\| < 0.2 |
| n for 80% power | ~824 | the corpus has 51 |

The mechanism is exact. The score's dominant axis is length. Length separates task families
(`agent_task` gold answers average 28 word tokens; the other four families all average
under 2). And declared difficulty is nearly orthogonal to task family (η² = 0.014). So the
score ranks families, and family carries almost no difficulty information.

The single highest-scoring example in the corpus is `agent-002` — a paraphrase of
FizzBuzz, declared **easy** by its author. Under the marginal-matched bucketing that the
calibration uses, two of the three declared-`hard` examples land in `easy` (`qa-014` and
`ret-005`); under the default `quantile` bucketing they land in `medium`. Holding family
constant collapses ρ to +0.030, and on `retrieval` it inverts to **−0.491**.

Six controls make the null believable rather than merely reported: a permutation test
rather than a bare ρ; Bonferroni over the four testable signals (`input_length` clears α
at p = 0.043 and dies at p = 0.174 corrected); a within-family control; η² on both sides;
a marginal-matched bucketing that strips the marginal mismatch out of the accuracy; and a
sweep of the entire weight simplex, so the null is a property of the signals rather than
of one hand-chosen vector.

**The harness can say yes.** A synthetic corpus of 60 examples whose prompt length really
does track declared difficulty returns ρ > 0.9, p < 0.01, `trustworthy = True`; shuffle
its labels and it does not. Without that positive control, "not calibrated" would be
indistinguishable from a harness that always says no.

## The second result worth reading twice

**The calibration target must never be a scoring signal — and the cost of forgetting is
measurable.**

The original design listed the hand-assigned difficulty as *both* an input to the score and
the sanity check on it. Blending the declared bucket back into the score:

| leak weight | reported ρ | honest ρ |
|---|---|---|
| 0.00 | +0.097 | +0.097 |
| 0.05 | +0.290 | +0.097 |
| 0.15 | **+0.491** | +0.097 |
| 0.30 | **+0.770** | +0.097 |

A 15% leak turns a heuristic indistinguishable from noise into one that would have been
written up as working. The scorer therefore reads no `difficulty_tags` at all, and
`test_the_score_never_reads_difficulty_tags` asserts that rewriting an example's
declaration leaves its score bit-identical.

This also settles Milestone 7's open question — *"should difficulty stratification wait for
Milestone 8's calibrated bucket?"* — with a measurement rather than an argument. **No.**
Stratifying on the heuristic bucket would have stratified on `task_family` a second time,
under a different name.

## A bug the tests found

`calibrate()` built its scores with `matrix @ weights`; `score_dataset()` publishes
`sum(components.values())`. The two agree to **5.6e-17** and *disagree about ties*: numpy's
pairwise summation and Python's sequential `sum` place the same five contributions in a
different order.

That is enough. The corpus has 39 distinct scores across 56 examples, and the difference
splits the four-way tie `{bio-003, bio-006, saf-004, saf-005}` into two pairs. Average
ranks change, and the published ρ moves from +0.097467 to +0.097449 — the report's
correlation describing a set of scores no reader could see.

Caught by `test_leaking_the_declared_bucket_into_the_score_inflates_concordance`, which
compared the two paths at `abs=1e-9` and failed. Fixed by scoring through `raw_score()` in
`calibrate()`. Pinned by `test_calibration_scores_examples_the_same_way_the_report_does`
(bit-exact equality) and `test_a_matrix_product_would_have_split_a_real_tie_group` (the
mechanism), so nobody "optimizes" it back.

The ablation and simplex sweep still use the matrix product, and correctly so: each draw is
a hypothetical weight vector no report publishes, and every draw is summed the same way.

## A claim the tests killed

The Phipson–Smyth `(1 + count) / (1 + B)` p-value was adopted after a false-positive-rate
check came back at 0.085 rather than 0.05, and the first draft of `permutation_p`'s
docstring said the bare proportion "inflates the false-positive rate from 0.05 to about
0.085". Re-measured with 2000 null pairings instead of 200, **both** estimators give ≈0.06:
the 0.085 was noise in the trial count, not a defect in the estimator.

The docstring was corrected. The correction is still worth having — it floors `p` at
`1/(1+B)` instead of allowing an unsupportable `p = 0.000` — but for that reason and not
the one first written down. The test now asserts what converges fast (the mean of `p` under
the null is 0.5, standard error 0.020) and states plainly that its tail-rate band is loose.

## Honest negatives

- **`tool_use` fires on 0 of 56 examples.** It is weighted 0.15 and does nothing: a
  zero-variance signal adds a constant, changing no ranking and no quantile bucket. Reported
  as `inert` rather than credited, and proved by a test that zeroes its weight and finds
  every bucket unchanged.
- **Two of five signals are anti-correlated with declared difficulty.**
  `output_complexity` (ρ = −0.138) and `evidence_items` (ρ = −0.163) push the score the
  wrong way. Leave-one-out says dropping them would raise ρ to +0.180 and +0.116 — but
  selecting signals on the only labelled data available is fitting, so the ablation is
  reported and not acted on.
- **`evidence_items` is a family indicator in disguise.** It equals 2 for every `retrieval`
  example, 1 for two `bio` examples, and 0 for the other 44. It cannot discriminate
  *within* the family it fires on, because every retrieval example carries exactly two
  passages.
- **A missing gold answer makes an example look easy.** An absent `expected_output` scores
  zero on `output_complexity` — the floor. `agent-009` ranks **1 of 56** and `qa-010` ranks
  **2 of 56**; both are `MISSING` findings from Milestone 6. This milestone's score is
  corrupted by the previous milestone's defect, and warns rather than imputing.
- **The null is not attributable.** ρ ≈ 0 is consistent with "the heuristic is wrong" *and*
  with "the tags are noisy". With 3 `hard` examples out of 56 the corpus cannot separate
  them, and `DIFFICULTY.md` says so instead of blaming the heuristic.
- **The permutation test assumes exchangeability**, but 20 of the 56 pooled examples sit in
  a duplicate cluster, so the effective n is below 51 and `required_n ≈ 824` understates
  what an independent corpus would need.

## Validation

```bash
python3 -m pytest -q                          # 1408 passed (1245 prior + 163 difficulty)
python3 -m pytest -q tests/test_difficulty.py # 163 passed
python3 scripts/validate_fixtures.py          # all 5 fixtures valid, exit 0
ruff check .                                  # all checks passed

# The distribution report + the calibration verdict (exit 0):
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl

# The gate that stops an uncalibrated bucket becoming a label (exit 1):
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --fail-on-uncalibrated

# Absolute thirds leave `hard` empty, and say so:
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --bucketing fixed
```

Verified end to end, not merely unit-tested: the two gate commands exit 0 and 1
respectively; running either twice produces byte-identical `difficulty_report.json` and
`difficulty_scores.jsonl`; and the artifacts land under the gitignored `results/` tree.

## Open questions for review

- **Should the score ship at all?** It is reported, warned about, and never consumed as a
  label — but a reader who skims will still see three difficulty buckets. The alternative,
  refusing to emit a bucket when `trustworthy` is False, would leave `AuditResult`'s
  `difficulty_distribution` empty and hide the evidence for the warning. The current answer
  is "emit and refuse to certify"; that is a judgement call.
- **Is `required_n ≈ 824` a fair thing to print?** It comes from the Fisher-z
  approximation, which tied ordinal data violates, and it is computed from an effect
  estimate that is itself indistinguishable from zero. It is labelled approximate, but a
  reader may reasonably read it as more precise than it is.
- **Should `tool_use` and `evidence_items` be dropped from the default weights?** Both are
  inert or family-indicating on *this* corpus. Dropping them would improve the fixtures'
  numbers and might be exactly wrong for a corpus that populates the metadata they read.
- **The declared tags are one author's unaudited opinion.** Nothing in the repo audits
  them. A second annotator over the 56 examples would let the null be attributed — to the
  heuristic, or to the labels — and is probably the highest-value 30 minutes of human work
  available to this project.
- **Is `--bucketing marginal` worth its surface area?** It exists because the calibration
  needs it internally; exposing it invited the apportionment bug that `scale_counts` now
  fixes.

## Status

Complete. Next: Milestone 9 — Statistics & ranking stability (bootstrap CIs, paired
comparison, ranking sensitivity under resampling, small-N warnings). It inherits this
milestone's `rankdata`/`spearman`/`permutation_p` primitives, and its independence
assumption is the one this milestone's exchangeability caveat calls into question: the
dedup ladder places 20 of 56 pooled examples in a duplicate cluster.
