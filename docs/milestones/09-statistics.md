# Milestone 9 — Statistics & ranking stability

## Summary

Land audit dimension #8: bootstrap confidence intervals, paired system comparison, ranking
stability under resampling and deletion, small-N warnings, and the `evalforge stats` CLI.

The distinguishing work was not the bootstrap. It was (a) noticing that the design doc's own
method — *"resample examples with replacement"* — is **wrong on this repository's own
corpus**, because Milestone 4 puts 20 of 56 examples in duplicate clusters and a
deterministic system scores every member of an exact cluster identically; (b) measuring the
cost with a coverage simulation against known ground truth (0.882 against a nominal 0.95) and
fixing it by resampling the duplicate blocks Milestone 7 already builds; (c) measuring that
the "win fraction" everyone reports has a type-I error of 11.4% at a nominal 5%, and
declining to use it as a decision rule; and (d) replacing "the ranking is unstable" with an
exact integer — deleting **2 of 56 examples**, which turn out to be one duplicate cluster,
reverses the demo leaderboard.

## Deliverables

- `src/eval_dataset_forge/statistics/` package:
  - `bootstrap.py` — `Units` (the resampling unit as a first-class parameter),
    `resample_indices()` (one shared draw matrix, which is what makes comparisons *paired*),
    `bootstrap_distribution()` (ratio estimator), `percentile_ci()`, `bca_ci()`,
    `jackknife_means()`, `bootstrap_ci()`, `intraclass_correlation()`,
    `kish_design_effect()`, `measured_design_effect()`, `effective_n()`.
  - `confidence.py` — `sign_flip_p()` (cluster-aware, Phipson–Smyth), `holm()`,
    `monte_carlo_se()`, `minimum_detectable_effect()`, `required_n_for_delta()`,
    `SystemComparison`, `compare_pair()`, `compare_systems()`.
  - `ranking_stability.py` — `kendall_tau_b()`, `rank_frequencies()`, `RankingStability`,
    `FlipSet`, `min_units_to_flip()`, `min_examples_to_flip()`.
  - `scores.py` — `parse_scores()` / `load_scores()` / `write_scores()` / `to_table()` /
    `align()` (strict in both directions; nothing imputed).
  - `simulate.py` — a Rasch (1PL) generator with block-level item difficulty and a tunable
    `icc`. Not part of the audit: it produces the offline demo and the ground truth the
    coverage tests check against.
  - `__init__.py` — `StatsReport`, `SystemResult`, `build_units()`, `analyze_systems()`,
    `analyze_dataset()`, `write_report()`, `format_report()`.
- `schemas.py` — `BootstrapResult` gains `system`, `method`, `resample_unit` (additive, with
  defaults; old records still load).
- `cli.py` — `evalforge stats` with repeatable `--dataset`, `--scores` | `--simulate`,
  `--resample-unit`, `--ci-method`, `--seed`, `--resamples`, `--alpha`, `--power`, `--out`,
  `--json`, `--fail-on-unstable-ranking`.
- `scripts/make_demo_scores.py` + `datasets/fixtures/qa_toy/system_scores.jsonl` — a
  committed, regenerable, explicitly synthetic example of the score-file format.
- `tests/test_statistics.py` — 115 tests: the ratio estimator against a naive reference, the
  coverage simulations with their positive control, BCa in both directions, the win-fraction
  type-I error, ICC = 1 by construction, the design effect falsified at `icc = 0`, the flip
  set brute-forced against every subset, ties, score IO, determinism, byte-identical
  artifacts, and the CLI.
- Docs: `docs/STATISTICS.md` rewritten around the measurements; `docs/RESULTS.md` gains its
  statistics section; `docs/LIMITATIONS.md` gains two new sections; `docs/DATASET_SCHEMA.md`
  documents the new `BootstrapResult` fields and the score-file format. Website:
  `statistics.html` rewritten; status updates to `milestones.html` (including Milestone 8's
  stale table row), `audit-pipeline.html`, `index.html`, `results.html`, `limitations.html`.

## Design decisions

- **The resampling unit is named, not assumed.** `Units` is a parameter of every function
  here. `--resample-unit block` is the default because a corpus with no duplicates has one
  example per block, so choosing it costs nothing and never under-corrects. Both
  distributions are always computed, because the *difference* between them is the finding.
- **Blocks are Milestone 7's blocks.** `splits.blocks.duplicate_blocks` — the same union-find
  components the adversarial split uses. Two stages must not disagree about what a dependent
  group is, and both inherit the near stage's recall bound.
- **The design effect is reported twice.** Kish's closed form (1.50) and the measured
  variance ratio (1.39–1.56). The formula assumes equal cluster sizes and these are unequal;
  quoting only the flattering one would be a choice.
- **One shared draw matrix.** All systems are resampled on the *same* corpora, or the
  difference of two marginal distributions would carry variance the pairing removes. Pinned
  by `test_a_shared_draw_makes_the_comparison_paired`.
- **The win fraction is reported and never used.** `significant` reads `p_holm`. Removing the
  win fraction entirely was considered and rejected: readers will compute it anyway, and it
  is better to print it next to the number that should have been used.
- **Sign-flipping happens at the block level.** Flipping per example would silently restore
  the independence assumption the cluster bootstrap was introduced to remove.
- **Holm, not Bonferroni.** Uniformly more powerful, equally valid under arbitrary dependence
  between the tests — which is what pairwise comparisons over a shared example set have.
- **A p-value is an estimate.** `p_mc_se = sqrt(p(1-p)/B)`, and `p_is_borderline` fires when
  `|p - alpha| <= 2·SE`. A verdict decided by the third decimal of a resampled p is decided
  by the seed.
- **`None` and `0` are different flip answers.** `None` means no proper subset can erase the
  lead; `0` means the lead is already absent. Conflating them into "flip@never" was a bug the
  `qa_toy` demo surfaced within minutes of the first run.
- **Ties split their credit.** Breaking a first-place tie by name would hand an alphabetical
  system free top-1 mass. `exact_order_frequency` counts only strict orderings, and
  `tie_fraction` reports how often it mattered (35% on `qa_toy`).
- **`B = 2000`, not the design note's 1000.** The 2.5% tail of a `B = 1000` bootstrap is
  estimated from 25 draws, and the endpoint jitter is the same order as the effects this
  milestone reports.
- **Nothing is imputed.** A score for an unknown example, or an example no system scored,
  raises. Every imputation of a missing eval result is a silent decision about a system's
  score on an example it may have crashed on. Milestone 6 refuses to impute a label; this
  refuses for the same reason.

## Deviations from the design doc

- **The i.i.d. bootstrap was demoted from the method to a comparison baseline.**
  `docs/STATISTICS.md` originally specified "resample examples with replacement `B` times…
  it assumes examples are exchangeable (which the dedup/leakage checks help ensure)". The
  dedup checks *detect* non-exchangeability; they do not remove it. Resampling blocks is the
  correction, and the original method survives as `--resample-unit example` so the report can
  quote its cost.
- **Two modules were added.** `PLAN.md` §4 lists `bootstrap.py · confidence.py ·
  ranking_stability.py`, all three of which shipped. `scores.py` (the stage's input format —
  it is the only stage whose input is not the dataset) and `simulate.py` (the demo generator
  and the coverage tests' ground truth) are additive.
- **A CLI subcommand was added.** `PLAN.md` §8 does not list `stats`; the stage was to
  surface through `audit` (Milestone 12). But a reproducible validation command is required
  by the process, and `audit` has no way to accept system scores. `evalforge stats` is
  additive and does not reshape the parser.
- **`--dataset` is repeatable.** Every other subcommand takes one dataset. The headline result
  is about the pooled 56-example corpus, and a benchmark assembled from several files is the
  normal case for this stage.
- **`BootstrapResult` gained three fields.** Additive, with defaults, so `SCHEMA_VERSION` is
  unchanged and pre-existing records load. Recorded in `docs/DATASET_SCHEMA.md`.

## The result worth reading twice

**The textbook bootstrap is anti-conservative on any corpus with duplicates, and this
repository's own dedup report says which ones.**

For an *exact*-duplicate cluster the within-cluster correlation of a deterministic system's
scores is 1 — by the definition of "duplicate", not by measurement. Resampling those examples
independently pretends to have more information than the corpus contains.

Measured against known ground truth (3000 trials, `B = 999`, 44 blocks with the pooled
corpus's exact size profile, nominal 0.95):

| true accuracy | i.i.d. bootstrap | **cluster bootstrap** | cluster + BCa |
|---|---|---|---|
| 0.50 | 0.904 | **0.944** | 0.954 |
| 0.70 | 0.882 | **0.937** | 0.956 |
| 0.90 | 0.847 | **0.884** | 0.947 |

On genuinely i.i.d. data the same percentile interval covers **0.954**. That positive control
is what makes the table a finding rather than a broken harness. And simulating the same
blocks with `icc = 0` returns a design effect of 1.0, so the widening tracks the dependence
rather than the choice of unit.

What it costs the pooled corpus: design effect 1.39–1.56 measured (Kish predicts 1.50), CIs
1.16×–1.29× wider, **56 examples worth ≈ 36 independent ones**.

## The second result worth reading twice

**The bootstrap "win fraction" is not a p-value, and reporting it as one doubles the false
positives.**

The rule *"A beats B if A wins ≥ 95% of the bootstrap resamples"* is a one-sided 95%
statement applied post hoc in whichever direction the data happened to favour. Under a true
null (`n = 56`, two independent Bernoulli(0.7) systems, 3000 trials):

| decision rule | type-I error (nominal 0.05) |
|---|---|
| win fraction ≥ 0.95, used two-sidedly | **0.114** |
| win fraction ≥ 0.975 | 0.057 |
| cluster-aware sign-flip test, `p ≤ 0.05` | **0.029** |

The threshold that corresponds to a two-sided α = 0.05 is 0.975. The tool prints the win
fraction and decides with the sign-flip test — which is itself **conservative** on paired
binary scores, because most examples are ties and the null distribution is coarse. Both
numbers are printed; neither is flattering.

## The third result worth reading twice

**Deletion explains fragility better than resampling, and the answer is an exact integer.**

Deleting a set of units leaves mean `(S - R) / (N - M)`. The *sign of a mean does not depend
on its denominator*, so the example counts of the deleted units are irrelevant: minimizing the
remaining mean over all `k`-unit deletions is exactly maximizing `R`, which the `k` largest
unit sums achieve. Greedy is optimal — no dynamic program, no search. (The first draft *was* a
DP over `(blocks used, examples removed)`; it was correct in value, carried a real
reconstruction hazard, and was 40 lines longer than the observation that replaced it.)

On the demo systems, `sys_a` leads `sys_b` by 3.6 accuracy points:

| question | answer |
|---|---|
| examples whose deletion reverses the lead | **2 of 56** (`agent-005`, `agent-006`) |
| duplicate clusters whose deletion reverses it | **1** |
| 95% CI on the margin | `[-0.185, +0.264]` — contains zero |
| sign-flip p (Holm-corrected) | 0.883 |
| times `sys_a` holds first place under resampling | 63.2% |
| smallest gap this corpus can detect at 80% power | 0.321 |
| examples needed to detect the observed 0.036 gap | ≈ 4500 |

The two examples are one block. **This is a mechanism, not a coincidence:** a cluster of size
*m* contributes *m* times to a margin while carrying one item's worth of information, so
duplicates are the highest-leverage items in any close comparison. The simulator makes it
exact by construction (`icc = 1`); a deterministic system makes it exact on *exact*
duplicates.

## A claim the measurements killed

BCa was added expecting it to be a strictly better interval, and the first draft of its
docstring said so. The coverage sweep disagreed:

| corpus | true accuracy | percentile | BCa |
|---|---|---|---|
| clustered | 0.90 | 0.884 | **0.947** |
| i.i.d. | 0.90 | 0.910 | 0.921 |
| i.i.d. | **0.95** | 0.926 | **0.830** |

At `p = 0.95` on 56 examples, 5.7% of trials cannot compute the bias correction `z0` at all
(every resample lands on one side of the point estimate), and the ones that can are corrected
in the wrong direction. So BCa is shipped, measured, documented, and **not** the default: an
interval method that is better on average and worse exactly where the data is thinnest is
offered, not imposed. When it cannot be computed the report says `method: "bca->percentile"`
rather than relabelling a percentile interval as a BCa one.

## Honest negatives

- **The demo's system scores are synthetic.** No model runs offline. The block structure is
  real (it comes from the fixtures via the dedup ladder) and the coverage tests use known
  parameters, so findings 1 and 3 stand — but nothing here is a claim about any real system,
  and `scores_source` records `simulated(seed=N)` so a report can never be mistaken for one.
- **The cluster bootstrap inherits Milestone 4's recall bound.** `ret-001`/`ret-002` is a
  paraphrase pair no lexical threshold reaches, so it yields no block edge and is resampled as
  if independent. **The reported design effect is a lower bound** and the intervals remain
  slightly too narrow. The audit's blind spot in one stage is the correction's blind spot in
  the next.
- **The percentile bootstrap under-covers at extreme accuracies regardless of the unit.**
  0.910 at `p = 0.90` on 56 i.i.d. examples. Small `n` plus a proportion near the boundary is
  hard for every nonparametric interval, and BCa does not reliably fix it here.
- **The sign-flip test is conservative** (0.029 at nominal 0.05) on paired binary scores. It
  will miss real differences — though on a corpus this size it would miss them anyway: the
  minimum detectable effect is 0.321.
- **A bootstrap CI can exclude zero while the test declines to reject.** `sys_a` vs `sys_c`:
  CI `[+0.033, +0.411]`, `p = 0.051`. Not a bug — the bootstrap centres on the observed
  effect while the sign-flip null centres on zero and preserves each block's `|d_b|`, so the
  effect's magnitude enters the null's spread (null sd 0.101 vs bootstrap sd 0.096). The
  report emits a warning naming the disagreement instead of quoting whichever number is
  friendlier.
- **That same p sits within Monte Carlo error of α.** `p = 0.051` at `B = 2000` has
  `SE = 0.005`. `p_is_borderline` fires. The fix is more resamples, not a conclusion.
- **ICC(1) can come out negative** on data with no clustering (−0.34 on one `icc = 0`
  simulation). It is reported as `nan` when undefined — all-singleton units, or a constant
  score — rather than as zero, because absence of evidence is not evidence of independence.
- **Kish's design-effect formula understates by up to 8%** on these unequal cluster sizes. It
  is reported alongside the measured ratio, not instead of it.
- **`minimum_detectable_effect` and `required_n` are normal approximations** to a trinomial
  quantity. A scale, not a promise — the same caveat Milestone 8's `required_n` carries.

## Validation

```bash
python3 -m pytest -q                          # 1523 passed (1408 prior + 115 statistics)
python3 -m pytest -q tests/test_statistics.py # 115 passed
python3 scripts/make_demo_scores.py --check   # fixture matches its generator, exit 0
python3 scripts/validate_fixtures.py          # all 5 fixtures valid, exit 0
ruff check .                                  # all checks passed

# The demo on the pooled corpus (exit 0):
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/agent_tasks_toy/examples.jsonl \
  --dataset datasets/fixtures/bio_toy/examples.jsonl \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset datasets/fixtures/retrieval_toy/examples.jsonl \
  --dataset datasets/fixtures/safety_toy/examples.jsonl \
  --simulate

# The gate that stops an unsupported leaderboard being reported (exit 1):
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --simulate --fail-on-unstable-ranking

# A real score file, in the committed long shape:
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --scores datasets/fixtures/qa_toy/system_scores.jsonl

# The cost of the i.i.d. assumption, asked for explicitly:
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --simulate --resample-unit example
```

Verified end to end, not merely unit-tested: the two gate commands exit 0 and 1 respectively;
running either twice produces a byte-identical `stats_report.json`; the committed score
fixture round-trips through its generator; and the artifacts land under the gitignored
`results/` tree.

## Open questions for review

- **Should `--resample-unit example` exist at all?** It reproduces a method this milestone
  measured to be wrong. It is retained so the report can quote the cost, and because a corpus
  with no duplicates is entitled to it — but a reader who reaches for it because it produces
  narrower intervals will be worse off, and nothing stops them.
- **Should the near-duplicate stage's *false positives* widen the interval?** They do today. A
  false near-duplicate pins two independent examples into one block and inflates the design
  effect. Conservative, but not free, and on `agent-003`/`agent-004` the "false positive" is
  itself disputed (Milestone 5).
- **Is the sign-flip test the right test for binary scores?** An exact McNemar test on the
  discordant pairs would be tighter, at the cost of a special case the graded-score path does
  not need. The current test's 0.029 measured size is real power left on the table.
- **`MIN_TOP1_STABILITY = 0.95` and `MIN_EFFECTIVE_N = 30` are conventions**, not theorems,
  exactly as `MIN_CALIBRATION_N = 30` is in Milestone 8. The report always prints the measured
  frequency so a reader can apply their own bar — but the `--fail-on-unstable-ranking` gate
  does not let them.
- **Should the dataset card print `effective_n` next to the row count?** A card that says "56
  examples" when the corpus is worth 36 is not lying, but it is not helping. Milestone 11's
  call.
- **The Rasch simulator is a modelling choice with consequences.** It makes item difficulty
  shared across systems, which is why paired comparison beats marginal intervals here. A
  simulator with independent systems would have made the pairing look pointless. That the
  choice is *right* (real systems do find the same items hard) is an argument, not a
  measurement, and this repo has no data to settle it.

## Status

Complete. Next: Milestone 10 — Release builder (dataset JSONL, manifest, split files, audit
summary, hashes, versioned release dir). It will need to decide whether a manifest records
`effective_n` alongside `example_count`, and whether the release's `audit.json` carries the
`BootstrapResult` records this milestone now emits.
