# Human review: Milestone 8 — Difficulty calibration

## TL;DR

The difficulty heuristic works, in the sense that it computes a transparent, explainable,
dataset-independent score. It does **not** work in the sense that matters: on the committed
fixtures its ranking is statistically indistinguishable from a random ranking of the
hand-assigned `difficulty_tags` (Spearman ρ = +0.097, permutation p = 0.497, n = 51). What
it actually measures is `task_family` — 43% of the score's variance, against 1% of the
declared label's.

The milestone therefore ships the score, the distribution reports, **and a calibration
harness that falsifies the score**. `DifficultyReport.trustworthy` is `False`,
`evalforge difficulty --fail-on-uncalibrated` exits 1, and the heuristic bucket is never
consumed as a difficulty label anywhere in the pipeline.

The reviewer's job here is mostly to decide whether that is the right call, and whether the
null should be attributed to the heuristic or to the labels.

## What changed

- New `src/eval_dataset_forge/difficulty/` package: `heuristics.py` (five surface signals,
  documented weights, log-compression), `buckets.py` (quantile / fixed / marginal cut
  points, tie-safe assignment, the declared bucket), `calibration.py` (tie-corrected
  Spearman, Phipson–Smyth permutation test, Bonferroni, within-family control, η², a
  marginal-matched bucket accuracy, a 2000-draw weight-simplex sweep, Fisher-z power),
  `__init__.py` (`DifficultyReport`, `score_dataset`, `write_report`, `format_report`).
- New CLI subcommand `evalforge difficulty`, with a `--fail-on-uncalibrated` gate mirroring
  Milestone 7's `--fail-on-leakage`.
- `splits/stratified.py` now imports `declared_bucket` / `DIFFICULTY_ORDER` / `UNLABELLED`
  from `difficulty.buckets` and re-exports them (layering fix; no behaviour change, all
  1245 prior tests pass unmodified).
- `tests/test_difficulty.py`, 163 tests.
- `docs/DIFFICULTY.md` rewritten; `docs/RESULTS.md` and `docs/LIMITATIONS.md` gain their
  difficulty sections; `site/difficulty.html` rewritten; status updated on
  `milestones.html`, `audit-pipeline.html`, `index.html`, `results.html`,
  `limitations.html`.

## What should be checked later

1. **Is "emit the bucket but refuse to certify it" the right policy?** The alternative —
   emitting no bucket when `trustworthy` is `False` — would leave `AuditResult`'s
   `difficulty_distribution` empty and hide the evidence that justifies the warning. A
   skimming reader still sees three difficulty buckets. This is the milestone's biggest
   judgement call.
2. **The null is not attributable, and that is the interesting question.** ρ ≈ 0 is equally
   consistent with "the heuristic is wrong" and "the tags are noisy". The corpus has three
   `hard` examples and five with no declaration; it cannot separate the two hypotheses.
   **A second annotator over the 56 examples would settle it, and is probably the highest-
   value 30 minutes of human work available to this repository.**
3. **Should `tool_use` and `evidence_items` stay in the default weights?** `tool_use` fires
   on 0/56 examples (weight 0.15, provably inert). `evidence_items` is 2 for every
   `retrieval` example and 0 almost everywhere else — a family indicator that cannot
   discriminate within the family it fires on. Dropping them would flatter the fixtures and
   might be exactly wrong for a corpus that populates those metadata keys.
4. **Two of five signals are anti-correlated with declared difficulty**
   (`output_complexity` −0.138, `evidence_items` −0.163). Leave-one-out says removing them
   would raise ρ to +0.180 / +0.116. That was deliberately *not* acted on — selecting
   signals on the only labelled data available is fitting a 5-parameter model to 51 points
   — but a reviewer may disagree about where the line between "design" and "fitting" sits.
5. **Is printing `required_n ≈ 824` responsible?** It uses the Fisher-z approximation
   (which tied ordinal data violates) on an effect estimate indistinguishable from zero. It
   is labelled approximate. It may still read as more precise than it is.
6. **The permutation test assumes exchangeability**, but 20 of the 56 pooled examples sit
   in a duplicate cluster. Effective n is below 51. This makes the evidence weaker, not
   stronger, so the null stands — but `required_n` understates what an independent corpus
   would need.
7. **`--bucketing marginal` on the public CLI.** It exists because the calibration needs it
   internally. Exposing it invited the apportionment bug that `scale_counts()` now fixes
   (raw declared counts sum to 51 but bucket 56 examples, dumping the surplus into `hard`).
   Worth deciding whether it earns its surface area.
8. **The hand-chosen caps** (32 tokens, 4 cues, 4 items) determine where the log-compression
   saturates. The simplex sweep varies the *weights*, not the caps. A cap sweep was not run.

## Technical risks

- **The heuristic is on the public API of a tool whose own documentation says not to trust
  it.** If a downstream consumer reads `DifficultyScore.bucket` without reading
  `DifficultyReport.trustworthy`, the whole exercise is defeated. Nothing in the type system
  prevents it. Milestone 10's release builder and Milestone 11's dataset card must both
  surface the verdict, not just the distribution.
- **Float summation and ties.** `calibrate()` originally scored with `matrix @ weights`
  while the report published `sum(components)`. The two differ by 5.6e-17, which splits the
  four-way tie `{bio-003, bio-006, saf-004, saf-005}` and moved the published ρ. Fixed and
  pinned by two tests, but it is a reminder that rank statistics over this corpus are
  brittle at the last bit, and that Milestone 9's bootstrap will face the same hazard.
- **`declared_bucket` moved packages.** `splits.stratified` re-exports it and a test asserts
  the two names are the same object, but any external importer of
  `eval_dataset_forge.splits.stratified.declared_bucket` is now getting a re-export.
- **Permutation cost.** `score_dataset()` at defaults runs seven permutation tests
  (10 000 resamples each) plus a 2000-draw simplex sweep: ~0.65 s on the pooled corpus.
  Fine now; Milestone 12's smoke audit over all fixtures should pass smaller `n_resamples`
  or reuse one report.
- **`fixed` bucketing is effectively unusable** on short-answer corpora (max score 0.658, so
  the `hard` bucket is empty). It warns, but it is offered as a first-class choice.

## Commands to verify

```bash
python3 -m pytest -q                          # 1408 passed
python3 -m pytest -q tests/test_difficulty.py # 163 passed
python3 scripts/validate_fixtures.py          # all 5 fixtures valid, exit 0
ruff check .                                  # all checks passed

# The report + the verdict (exit 0)
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl

# The CI gate (exit 1 — this is the intended result on these fixtures)
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --fail-on-uncalibrated

# The `hard` bucket is empty under absolute thirds, and the tool says so
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --bucketing fixed
```

The two claims most worth reproducing by hand:

```bash
# 1. The score never reads difficulty_tags, so calibration is not circular.
python3 -m pytest -q tests/test_difficulty.py -k "never_reads_difficulty_tags or leaking"

# 2. The harness can certify a corpus whose difficulty really is surface-driven,
#    so "not calibrated" is a finding rather than a foregone conclusion.
python3 -m pytest -q tests/test_difficulty.py -k "certifies or shuffled_labels"
```

## Human feedback

<!-- Human will fill this later. -->
