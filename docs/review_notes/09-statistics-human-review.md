# Human review: Milestone 9 — Statistics & ranking stability

## TL;DR

The bootstrap this repository's own design doc specified — *"resample examples with
replacement"* — is **wrong on this repository's own fixtures**, and the milestone measures by
how much. Milestone 4 places 20 of 56 pooled examples in duplicate clusters; a deterministic
system scores every member of an exact cluster identically, so the within-cluster correlation
is 1 by construction. A coverage simulation with known ground truth shows the i.i.d.
percentile interval covering **0.882** against a nominal 0.95. Resampling the duplicate
blocks Milestone 7 already builds recovers 0.937, and the pooled corpus's 56 examples are
reported as worth **≈ 36** independent ones.

Two further measurements. The bootstrap **win fraction is not a p-value**: the ubiquitous
"A wins 95% of resamples" rule has a type-I error of 11.4% at a nominal 5%, so the tool prints
it and decides with a cluster-aware sign-flip test. And **deletion explains fragility better
than resampling**: the demo leader's 3.6-point margin is reversed by deleting 2 of 56
examples, which turn out to be a single duplicate cluster.

`StatsReport.ranking_resolved` is `False` on the committed fixtures and
`evalforge stats --fail-on-unstable-ranking` exits 1.

The reviewer's job here is mostly to decide whether the *simulator* is doing too much work,
and whether the conservatism (block resampling + a conservative test + Holm) has been stacked
so deep that the tool can no longer detect a real difference.

## What changed

- New `src/eval_dataset_forge/statistics/` package: `bootstrap.py` (`Units` as a first-class
  resampling unit, the shared draw matrix, ratio estimator, percentile + BCa intervals,
  ICC(1), Kish and measured design effects, effective n), `confidence.py` (cluster-aware
  sign-flip permutation test, Holm, Monte Carlo SE of a p-value, minimum detectable effect,
  `SystemComparison`), `ranking_stability.py` (`rank_frequencies`, tie-aware top-1 credit,
  Kendall tau-b, exact `min_units_to_flip`), `scores.py` (strict score-file IO),
  `simulate.py` (Rasch generator; demo scores + coverage ground truth), `__init__.py`
  (`StatsReport`, `analyze_systems`, `analyze_dataset`).
- New CLI subcommand `evalforge stats`, with a `--fail-on-unstable-ranking` gate mirroring
  Milestone 8's `--fail-on-uncalibrated`. `--dataset` is repeatable so the pooled corpus can
  be analysed as one benchmark.
- `schemas.py`: `BootstrapResult` gains `system`, `method`, `resample_unit` — additive, with
  defaults, so `SCHEMA_VERSION` is unchanged and older records still load. All 1408 prior
  tests pass unmodified.
- `scripts/make_demo_scores.py` and `datasets/fixtures/qa_toy/system_scores.jsonl`: a
  committed, regenerable, explicitly **synthetic** example of the score-file format.
- `tests/test_statistics.py`, 115 tests.
- `docs/STATISTICS.md` rewritten; `docs/RESULTS.md` and `docs/LIMITATIONS.md` gain their
  statistics sections; `docs/DATASET_SCHEMA.md` documents the new fields and the score
  format; `site/statistics.html` rewritten; status updated on `milestones.html` (which also
  still carried a stale "planned" row for Milestone 8), `audit-pipeline.html`, `index.html`,
  `results.html`, `limitations.html`.

## What should be checked later

1. **Is the Rasch simulator carrying an argument it should not be?** It makes item difficulty
   shared across systems, which is *why* paired comparison beats marginal intervals here; a
   simulator with independent systems would have made the pairing look pointless. The claim
   that real systems do find the same items hard is an argument, not a measurement, and this
   repo has no data to settle it. **This is the milestone's biggest unexamined assumption.**
2. **Has conservatism been stacked too deep?** Block resampling widens the interval; the
   sign-flip test measures 0.029 against a nominal 0.05; Holm corrects across all pairs. Each
   is individually defensible. Together the tool needs a 0.321 accuracy gap on this corpus
   before it will say anything. A reviewer may reasonably think the stack is now
   under-powered rather than honest, and the two are hard to distinguish from inside.
3. **Should `--resample-unit example` exist at all?** It reproduces a method this milestone
   measured to be wrong. It is retained so the report can quote the cost — but a reader who
   reaches for it because it produces narrower intervals will be worse off, and nothing stops
   them.
4. **Should the near-duplicate stage's *false positives* widen the interval?** They do. A
   false near-duplicate pins two independent examples into one block and inflates the design
   effect. On `agent-003`/`agent-004` the "false positive" is itself disputed (Milestone 5),
   so the answer may be "yes, and that is fine" — but it has not been argued.
5. **An exact McNemar test would be tighter for binary scores.** The sign-flip test's measured
   0.029 size is real power left on the table. The cost is a special case the graded-score
   path does not need.
6. **`MIN_TOP1_STABILITY = 0.95` and `MIN_EFFECTIVE_N = 30` are conventions**, not theorems,
   exactly as `MIN_CALIBRATION_N = 30` is in Milestone 8. The report prints the measured
   frequency so a reader can apply their own bar — the `--fail-on-unstable-ranking` gate does
   not let them.
7. **The design effect is a lower bound and the docs say so — is that enough?** The cluster
   bootstrap corrects only for dependence the dedup ladder can *see*. `ret-001`/`ret-002`
   yields no block edge and is resampled as if independent. A reader who takes
   `effective_n = 36` as an upper bound on optimism is being misled by a number that is itself
   optimistic.
8. **`BootstrapResult` gained fields without a `SCHEMA_VERSION` bump.** The additions are
   additive with defaults, so old records load; but `ForgeModel` forbids extra fields, so a
   *new* record cannot be read by an old reader. Whether that direction counts as breaking is
   a policy question the schema doc now answers by assertion.

## Technical risks

- **The whole correction depends on `duplicate_blocks` being right.** The statistics stage now
  consumes the split builder's block structure, so a regression in the dedup ladder silently
  changes every confidence interval in the pipeline — and it changes them in the direction of
  *narrower*, which is the direction nobody notices. `test_pooled_corpus_has_the_block_structure
  _the_coverage_tests_assume` pins the histogram (34 singletons, 8 pairs, 2 triples) so the
  regression would at least fail loudly here, but the coupling is now the pipeline's
  load-bearing one and it crosses three milestones.
- **A `nan` that must not read as a zero.** `ICC(1)` is `nan` when undefined and can be
  *negative* on genuinely independent data (measured −0.34); `kish_design_effect` propagates
  both. `StatsReport.design_effect` takes `max` over systems and skips `nan`, falling back to
  `nan` when every system is constant — exercised by
  `test_a_perfect_score_is_reported_as_a_degenerate_interval` for one system, but not for a
  mixture of constant and non-constant systems, where the `max` quietly ignores the constant
  one.
- **The `p = 0.051` on `sys_a` vs `sys_c` straddles α.** It is flagged (`p_is_borderline`) and
  the CI-vs-test disagreement is warned about, but a test asserts
  `p_value == approx(0.05, abs=0.01)` — if a future change to the resampling nudges it, that
  test fails for a reason unrelated to the change. It is pinned deliberately; it is still a
  brittle pin.
- **Monte Carlo cost.** `analyze_systems` at defaults draws two `2000 × n_units` index
  matrices and runs one sign-flip test per pair. ~0.4 s on the pooled corpus with 3 systems;
  quadratic in systems. Milestone 12's smoke audit should pass a smaller `--resamples`.
- **The demo score fixture is pinned to the simulator.** `make_demo_scores.py --check` fails
  if `simulate.py` changes, which is intended (a drift must be re-committed on purpose) but
  will read as an unrelated test failure to whoever touches the simulator next.

## Commands to verify

```bash
python3 -m pytest -q                          # 1523 passed
python3 -m pytest -q tests/test_statistics.py # 115 passed
python3 scripts/make_demo_scores.py --check   # fixture matches its generator, exit 0
python3 scripts/validate_fixtures.py          # all 5 fixtures valid, exit 0
ruff check .                                  # all checks passed

# The report + the verdict (exit 0)
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --simulate

# The CI gate (exit 1 — this is the intended result on these fixtures)
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --simulate --fail-on-unstable-ranking

# The cost of the i.i.d. assumption, asked for explicitly
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --simulate --resample-unit example
```

The three claims most worth reproducing by hand:

```bash
# 1. The i.i.d. bootstrap under-covers on clustered data, and the harness is not
#    simply broken — it reports nominal coverage on genuinely i.i.d. data.
python3 -m pytest -q tests/test_statistics.py -k "undercovers or restores or nominal_coverage"

# 2. The design effect tracks the dependence, not the choice of resampling unit:
#    the same blocks with icc=0 return deff = 1.0.
python3 -m pytest -q tests/test_statistics.py -k "tracks_dependence or icc_one_for_any"

# 3. The win fraction is not a p-value, and the exact flip set is exact.
python3 -m pytest -q tests/test_statistics.py -k "win_fraction_is_not or brute_force"
```

## Human feedback

<!-- Human will fill this later. -->
