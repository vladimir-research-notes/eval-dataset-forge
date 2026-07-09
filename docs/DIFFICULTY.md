# Difficulty calibration

> Status: implemented (Milestone 8). `src/eval_dataset_forge/difficulty/`,
> `tests/test_difficulty.py` (163 tests), `evalforge difficulty`.

A benchmark that is all-easy or all-hard stops discriminating between systems. So we
estimate a **transparent, model-free difficulty proxy** for each example and report the
distribution it induces.

Then we check whether the proxy means anything. On the committed fixtures it does not,
and this document explains how we know, in enough detail that a reader can disagree with
the conclusion rather than with the reporting.

---

## 1. The score

Five surface signals, each read off the example, compressed to `[0, 1]`, and combined by
fixed documented weights.

| signal | weight | cap | raw quantity | difficulty hypothesis |
|---|---|---|---|---|
| `input_length` | 0.20 | 32 tokens | word tokens in `input` | a longer prompt carries more to read, track, and satisfy |
| `output_complexity` | 0.20 | 32 tokens | word tokens in the canonical `expected_output` | a longer or more structured gold answer takes more work to produce |
| `reasoning_steps` | 0.30 | 4 cues | sequential / enumeration / aggregation cues | chained or enumerated sub-tasks need more inference steps |
| `evidence_items` | 0.15 | 4 items | retrieval passages + annotated entities | more candidates mean more to select among |
| `tool_use` | 0.15 | indicator | `metadata` declares a tool/lookup requirement | a task needing a tool cannot be answered from memory |

Three properties are load-bearing.

**The score is exactly the sum of its components.** `DifficultyScore.components` holds one
weighted contribution per signal and `sum(components.values()) == score`, bit-for-bit.
Neither is rounded, because rounding to six places breaks the equality for examples whose
five contributions round the same way — and with it the claim that the number is
explainable.

**Counts are log-compressed, not min-maxed:** `log1p(x) / log1p(cap)`, clipped. A single
75-token reference solution therefore cannot dominate a corpus of one-word gold answers.
More importantly the mapping is **dataset-independent** — no corpus statistic enters it —
so a score computed over one dataset means the same thing as a score computed over
another. Min-max normalization would silently redefine "hard" every time an example was
added. The price is hand-chosen caps, stated above.

**No signal reads `difficulty_tags`.** An earlier draft of this document listed the
`ambiguous`/`hard` tags and the hand-assigned fixture difficulty as *scoring* signals
while also proposing to sanity-check the score against those same labels. That is
circular. Both were removed from the scorer. The effect is not hypothetical: blending the
declared bucket back into the score at a weight of 0.15 lifts the reported rank
correlation from **+0.10 to +0.49**, and at 0.30 to **+0.77**. A null heuristic would have
read as a working one.

### Two rules chosen a priori

Both were fixed on linguistic grounds *before* any label was consulted, and both are
pinned by tests. Choosing them by whichever variant correlated better would be fitting a
five-parameter model to 51 points.

- **Bare digits are not reasoning steps.** A digit in a prompt is as often an operand, a
  year, or an identifier as an enumeration request. Counting digits gives `"What is 2 +
  2?"` two reasoning steps and gives the corpus's one declared-`hard` QA example — `"What
  is the approximate speed of light in a vacuum?"` — zero. Only word-form cardinals
  (`two` … `ten`) count. (For the record: the two variants score ρ = +0.192 and +0.171, so
  the choice barely moves the result.)
- **`tool_use` reads truthiness, not key presence.** A dataset that writes `{"tools": []}`
  on every example must not thereby make every example look harder.

---

## 2. Bucketing

Three methods, differing only in where the two cut points come from.

| method | cut points | property |
|---|---|---|
| `quantile` (default) | the score distribution's 1/3, 2/3 | always fills all three buckets |
| `fixed` | absolute 1/3, 2/3 on `[0, 1]` | comparable across datasets |
| `marginal` | wherever a target bucket count lands | reproduces a reference marginal |

**Equal scores always get equal buckets.** Every method compares `score < cut`, so a tie
group is never split by a boundary. The pooled corpus has 56 examples and only 39 distinct
scores, including two tie groups of four. The price is that `quantile` cannot always
deliver equal frequencies — it lands at 18/19/19 — so the report states the sizes it
*achieved*, not the sizes requested.

Bucket distributions on the pooled corpus:

| bucketing | easy | medium | hard | cut points |
|---|---|---|---|---|
| `quantile` (default) | 18 | 19 | 19 | 0.182 / 0.301 |
| `fixed` | 45 | 11 | **0** | 0.333 / 0.667 |
| `marginal` | 32 | 21 | 3 | 0.274 / 0.524 |
| *declared* | 29 | 19 | 3 | (5 unlabelled) |

**`fixed` degenerates on short-answer corpora, and says so.** The score is a weighted mean
of log-compressed signals, so reaching 2/3 requires most signals near their cap. No example
here does — the maximum score is 0.658 — and the `hard` bucket comes out empty. That is a
property of absolute cut points on a compressed scale, not of the data, and it is why
`quantile` is the default. A warning fires.

`marginal` apportions the declared counts to the corpus size by largest remainder
(29/19/3 over 51 labelled → 32/21/3 over 56), because otherwise the five undeclared
examples would all fall into `hard`.

---

## 3. Calibration: does the score mean anything?

Every other stage in this pipeline can be scored against defects planted in the fixtures.
A difficulty *proxy* cannot: there is no ground-truth difficulty, only the `difficulty_tags`
an author wrote down. So `difficulty/calibration.py` asks a falsifiable question —

> Is the heuristic's ranking distinguishable from a **random ranking** of the declared labels?

— and answers it with a permutation test. The permutation p-value uses the Phipson–Smyth
`(1 + count) / (1 + B)` estimator, so it is floored at `1/(1+B)` rather than able to
return an unsupportable `0.000`.

### The answer, on 51 labelled examples

| quantity | value | reading |
|---|---|---|
| Spearman ρ (score vs declared) | **+0.097** (p = **0.497**) | not distinguishable from chance |
| within-family ρ (size-weighted) | +0.030 | collapses when family is held constant |
| family-only *oracle* baseline | +0.162 | family alone barely predicts difficulty either |
| η² of the **score** by `task_family` | **0.434** | the score is 43% task family |
| η² of the **declared label** by `task_family` | **0.014** | declared difficulty is 1% task family |
| marginal-matched bucket accuracy | 0.529 (p = 0.179) | |
| majority-class baseline ("everything is easy") | **0.569** | the heuristic does not beat it |
| accuracy of a random ranking | 0.466 | |
| n needed to detect this effect at 80% power | ~**824** | the corpus has 51 |
| weight-simplex sweep, 2000 Dirichlet draws | median +0.024, max +0.286 | 94% of draws inside \|ρ\| < 0.2 |

**The heuristic is a task-family detector wearing a difficulty label.** Its dominant axis
is length; length separates task families; and declared difficulty is almost orthogonal to
task family. The single highest-scoring example in the whole corpus is `agent-002` — a
paraphrase of FizzBuzz, which its author declared `easy`.

Six guards, each against a different way of fooling oneself:

1. **A permutation test, not a bare correlation.** ρ = +0.097 at n = 51 looks like
   something. It is not.
2. **Bonferroni over the signals tested.** `input_length` alone reaches ρ = +0.283,
   p = 0.043 — apparently significant. Four signals were testable; corrected, p = 0.174.
   Quoting the uncorrected number would have been a fishing expedition.
3. **A within-family control.** Holding `task_family` constant drops the size-weighted ρ
   to +0.030. On `retrieval` it is **−0.491**: the score points the wrong way outright.
4. **η² on both sides.** A high overall ρ could have been a family effect in disguise.
   Comparing how much family explains of the *score* (43%) against how much it explains of
   the *declared label* (1%) shows what the score is actually tracking.
5. **A marginal-matched bucketing.** Plain quantile bucketing is punished for predicting
   one-third `hard` when only 6% of the corpus is `hard`, which says nothing about the
   ranking underneath. With the predicted bucket *counts* forced to the declared ones, the
   accuracy is 0.529 — below the 0.569 you get by calling everything `easy`.
6. **A sweep of the whole weight simplex.** If the null held only at the shipped weights it
   would be a fact about the weights. 94% of 2000 draws land inside \|ρ\| < 0.2, and the
   best draw — chosen *in sample*, hence an upper bound — reaches only +0.286.

### Per-signal

| signal | weight | ρ | permutation p | Bonferroni (4 tested) | verdict |
|---|---|---|---|---|---|
| `input_length` | 0.20 | +0.283 | 0.043 | 0.174 | clears α, not Bonferroni |
| `output_complexity` | 0.20 | −0.138 | 0.324 | 1.000 | **anti-correlated** |
| `reasoning_steps` | 0.30 | +0.192 | 0.186 | 0.745 | no effect |
| `evidence_items` | 0.15 | −0.163 | 0.260 | 1.000 | **anti-correlated** |
| `tool_use` | 0.15 | — | — | — | **inert**: fires on 0 / 56 examples |

`tool_use` has zero variance on this corpus. A weighted signal with zero variance adds a
constant to every score, so it changes no ranking and no quantile bucket; the report says
so rather than letting a weight imply work the signal cannot do. A test proves it: zeroing
`tool_use` and renormalizing leaves every bucket in place.

Two of five signals push the score the *wrong way*. Leave-one-out ablation says dropping
`output_complexity` would raise ρ from +0.097 to +0.180, and dropping `evidence_items` to
+0.116 — but selecting signals on the only labelled data available is fitting, and at n =
51 with 5 parameters it means nothing. The ablation is reported, not acted on.

### No individual fixture calibrates either

| fixture | n | labelled | ρ | p | verdict |
|---|---|---|---|---|---|
| `qa_toy` | 14 | 12 | −0.080 | 0.789 | not calibrated |
| `agent_tasks_toy` | 12 | 11 | +0.230 | 0.516 | not calibrated |
| `retrieval_toy` | 10 | 10 | −0.491 | 0.164 | not calibrated |
| `bio_toy` | 10 | 9 | +0.174 | 0.667 | not calibrated |
| `safety_toy` | 10 | 9 | +0.365 | 0.357 | not calibrated |
| **all five pooled** | 56 | 51 | **+0.097** | **0.497** | **not calibrated** |

Each is underpowered on its own (9–12 labels). The pooled corpus is the only one with any
power at all, and even there `required_n ≈ 824` says the effect is too small to detect
with 51 labels. **This is as much a statement about the corpus as about the heuristic**,
and the report makes that statement out loud rather than reporting a bare null.

### The harness can say yes

A calibration harness that only ever answers "no" proves nothing. A synthetic corpus of
60 examples whose prompt length really does increase with declared difficulty comes back
ρ > 0.9, p < 0.01, `trustworthy = True`; the same corpus with its labels shuffled does
not. Both are tests
(`test_the_harness_certifies_a_corpus_whose_difficulty_really_is_surface_driven`,
`test_the_same_corpus_with_shuffled_labels_is_not_certified`).

---

## 4. What the verdict changes

`DifficultyReport.trustworthy` is a single bit: true only when the correlation clears α,
the sample clears 30 labelled examples, and the marginal-matched bucketing beats the
majority baseline. On the committed corpus it is **False**, and that forbids one thing:

> the heuristic bucket must not be used as a difficulty label.

The one downstream consumer that could have been fooled is difficulty stratification in
[`SPLITS.md`](SPLITS.md). Milestone 7 chose to stratify on the *declared* tags, arguing
that a heuristic must not define the test set it is later evaluated on. This milestone
turns that argument into a measurement, and finds a second, stronger reason: stratifying
on the heuristic bucket would have stratified on `task_family` a second time, under a
different name.

Nothing is deleted or corrected. Every example still gets a bucket — the release builder
(Milestone 10) and the dataset card (Milestone 11) must be able to show a reader exactly
what the audit saw.

`evalforge difficulty --fail-on-uncalibrated` exits 1 on this corpus. That is the intended
result, and it is the CI gate a real dataset would have to pass before anything downstream
treated the bucket as a label.

---

## 5. A defect this stage inherits from Milestone 6

**A missing gold answer makes an example look easy.** An absent `expected_output` scores
zero on `output_complexity` — the floor of the signal — so the examples whose labels most
need attention sink to the easy end of the ranking:

| example | declared | heuristic bucket | rank (1 = easiest of 56) |
|---|---|---|---|
| `agent-009` | *(none)* | easy | **1** |
| `qa-010` | medium | easy | **2** |
| `bio-008` | medium | easy | 7 |
| `ret-004` | medium | medium | 24 |
| `ret-005` | **hard** | medium | 32 |
| `saf-009` | medium | medium | 37 |

All six are examples the [label audit](LABEL_AUDIT.md) flags. `agent-009` — an empty gold
answer on the prompt `"Make the report better."` — is the single easiest-looking example
in the corpus. The report warns; it does not silently impute a value.

---

## 6. Known limitations

- **The score is a surface proxy and can be fooled.** A short prompt can be very hard
  (`qa-014`, the speed of light, ranks 25/56); a long one can be trivial (`agent-002`,
  FizzBuzz, ranks 56/56). On this corpus it is fooled systematically, not occasionally.
- **The weights are hand-chosen, not learned.** They are stated, and the simplex sweep
  shows the conclusion does not depend on them. Fitting them to the 51 available labels
  would be fitting to the test set.
- **The calibration target is one author's opinion.** `difficulty_tags` are hand-assigned,
  unaudited, and thin: 3 `hard` examples out of 56, and 5 examples with no declaration at
  all. A null result is consistent with "the heuristic is wrong" *and* with "the tags are
  noisy". We cannot separate the two with this corpus, and we do not pretend to.
- **`required_n` is approximate.** It assumes the Fisher-z bivariate-normal
  approximation, which tied ordinal data violates. Read it as a scale, not a promise.
- **The permutation test assumes exchangeability under the null.** Near-duplicate examples
  are not independent, so the effective sample size is below 51 and the p-values are
  approximate. The dedup ladder ([`DEDUPLICATION.md`](DEDUPLICATION.md)) places 20 of the
  56 pooled examples in a duplicate cluster. This does not rescue the null — a *smaller*
  effective n makes the evidence weaker, not stronger — but it means `required_n ≈ 824`
  understates what an independent corpus would need.
- **Item-response-theory difficulty from observed pass rates is out of scope.** It needs
  response data from real systems; the offline default has none. See
  [`PAPER_PATH.md`](PAPER_PATH.md).

---

## 7. Reproduce

```bash
python3 -m pytest -q tests/test_difficulty.py      # 163 tests

# The distribution report + the calibration verdict (exit 0):
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl

# The CI gate a dataset must pass before its buckets are treated as labels (exit 1):
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --fail-on-uncalibrated

# Absolute thirds leave the `hard` bucket empty, and say so:
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --bucketing fixed

# Artifacts (gitignored):
PYTHONPATH=src python3 -m eval_dataset_forge difficulty \
  --dataset datasets/fixtures/bio_toy/examples.jsonl \
  --out results/processed/bio_toy_difficulty
```
