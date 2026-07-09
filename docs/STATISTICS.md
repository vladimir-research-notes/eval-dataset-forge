# Statistical uncertainty & ranking stability

> Status: implemented (Milestone 9). `src/eval_dataset_forge/statistics/`,
> `tests/test_statistics.py` (115 tests), `evalforge stats`.

Audit dimension #8: **can this dataset tell these systems apart?**

Every earlier stage describes the dataset. This one describes what the dataset can
*support*. The inputs are per-example scores for one or more systems — real eval
outputs in use, deterministic synthetic ones in the offline demo.

Three findings, each measured rather than asserted, and each with a reproducible
command below.

---

## 1. The i.i.d. bootstrap is anti-conservative on any corpus with duplicates

The textbook bootstrap resamples **examples**, which assumes examples are
independent. On a corpus with duplicates they are not. A deterministic system
scores every member of an *exact*-duplicate cluster identically — this is a
property of the definition of "duplicate", not an empirical claim — so the
within-cluster correlation of its scores is exactly 1.

Sampling those examples independently pretends to have more information than the
corpus contains, and the interval comes out too narrow.

The fix is to resample the **duplicate blocks** that Milestone 7 already builds
(`splits.blocks.duplicate_blocks`, union-find over every stage of the dedup
ladder). Both are always computed, and their variance ratio is the **design
effect**.

### Coverage, with known ground truth

A confidence interval's only testable property is that it contains the true
parameter 95% of the time. That sentence needs a true parameter, which a real
benchmark never has — so we simulate one. 44 blocks with the pooled corpus's exact
size profile (34 singletons, 8 pairs, 2 triples = 56 examples); each block's
outcome is Bernoulli(*p*) and is copied to its members. The estimand is *p*, the
accuracy of a fresh corpus drawn the same way.

3000 trials, `B = 999`. Nominal coverage 0.95; Monte Carlo error ±0.004.

| true accuracy | i.i.d. bootstrap | **cluster bootstrap** | cluster + BCa |
|---|---|---|---|
| 0.50 | 0.904 | **0.944** | 0.954 |
| 0.70 | 0.882 | **0.937** | 0.956 |
| 0.90 | 0.847 | **0.884** | 0.947 |

The i.i.d. interval misses by 5–10 points of coverage. Resampling blocks recovers
most of it.

### Positive control

A coverage harness that always reports under-coverage would prove nothing. On
genuinely i.i.d. data (56 independent examples) the same percentile interval
covers **0.954** at *p* = 0.50 — nominal, as it should be. The under-coverage
above is caused by the clustering, not by the harness.

### What it costs the pooled corpus

| quantity | value |
|---|---|
| examples | 56 |
| duplicate blocks | 44 (34 singletons, 8 pairs, 2 triples) |
| Kish mean cluster size, `Σm²/Σm` | 1.50 |
| design effect, Kish closed form at ICC = 1 | 1.50 |
| design effect, **measured** variance ratio | 1.39 – 1.56 |
| CI widening | 1.16× – 1.29× |
| effective sample size | **≈ 36**, not 56 |

Kish's `1 + (m̄ - 1)·ICC` is derived for equal cluster sizes and applied here to
unequal ones. It is an approximation, it disagrees with the measured ratio by up
to 8%, and the report quotes **both** rather than picking the flattering one. The
measured ratio is what the intervals actually did.

### Falsification

The widening must track the *dependence*, not the choice of unit. Simulating the
same 44 blocks with `icc = 0` — blocks exist, but their members are conditionally
independent — returns a measured design effect of 1.0 ± 0.25 and CIs that
coincide. Pinned by `test_the_design_effect_tracks_dependence_not_the_choice_of_unit`.

---

## 2. The bootstrap "win fraction" is not a p-value

The share of resamples in which A outscores B is widely reported and widely
misread. The common rule — *"A beats B if A wins ≥ 95% of the bootstrap
resamples"* — is a **one-sided** 95% statement, applied post hoc in whichever
direction the data happened to favour. Its two-sided size is therefore ≈ 0.10, not
0.05.

Measured under a true null (`n = 56`, two independent Bernoulli(0.7) systems, 3000
trials, `B = 999`, nominal α = 0.05):

| decision rule | type-I error |
|---|---|
| win fraction ≥ 0.95 (used two-sidedly) | **0.114** |
| win fraction ≥ 0.975 | 0.057 |
| cluster-aware sign-flip test, `p ≤ 0.05` | **0.029** |

So the 0.95 rule rejects a true null more than twice as often as it claims. The
threshold that corresponds to a two-sided α = 0.05 is 0.975.

`eval-dataset-forge` reports the win fraction — readers expect it — and **decides**
with the sign-flip permutation test. `SystemComparison.significant` reads the
Holm-corrected p-value and never the win fraction.

### The sign-flip test, and its honest cost

Under the null that A and B are exchangeable, the paired difference `d_i` is
symmetric about zero, so flipping signs leaves the distribution unchanged. Signs
are flipped at the **block** level: flipping per example would silently restore
the independence assumption the cluster bootstrap exists to remove. The p-value
uses the Phipson–Smyth `(1 + count) / (1 + B)` estimator, so it is floored at
`1/(1+B)` rather than claiming an unsupportable `p = 0`.

The cost is visible in the table: at nominal 0.05 the test rejects only 0.029. It
is **conservative** on paired binary scores, because most examples are ties
(`d_i = 0`) and the null distribution is coarse. This is reported, not hidden. A
test that under-rejects will miss real differences; on a corpus this size it will
miss them anyway (see §3).

### Two nulls can disagree, and that is not a bug

On the demo, `sys_a` vs `sys_c` has a 95% bootstrap CI of `[+0.033, +0.411]` —
excluding zero — and a sign-flip `p = 0.051`. Both are correct. The bootstrap
interval is centred on the *observed* effect; the sign-flip null is centred on
zero and preserves each block's `|d_b|`, so the effect's own magnitude enters the
null's spread (null sd 0.101 vs bootstrap sd 0.096). The permutation test is the
more conservative of the two near the boundary, and it is what `significant`
reads. The report emits a warning naming the disagreement rather than resolving it
by picking the friendlier number.

### A p-value is itself an estimate

`p = 0.051` at `B = 2000` carries a Monte Carlo standard error of
`sqrt(p(1-p)/B) = 0.005`. A verdict that turns on the third decimal of such a p is
set by the seed, not by the data. Every comparison reports `p_mc_se`, and
`p_is_borderline` fires when `|p - α| ≤ 2·SE`. The fix is more resamples, not a
different conclusion.

### Multiplicity

`K` systems imply `K(K-1)/2` pairwise tests over one shared example set. Holm–
Bonferroni step-down correction is applied across all of them; it controls the
family-wise error rate under arbitrary dependence between the tests, which is what
comparisons sharing an example set have. It is uniformly more powerful than plain
Bonferroni and just as valid, so there is no reason to prefer the latter.

---

## 3. Deletion explains fragility better than resampling

Resampling answers "would a different sample have produced this leaderboard?".
Deletion answers a sharper question: **what is the smallest set of examples whose
removal erases A's lead over B?** No seed, no interval — an integer, computed
exactly.

The argument is two lines. Deleting a set of units leaves mean `(S - R) / (N - M)`.
The **sign of a mean does not depend on its denominator**, so the example counts of
the deleted units are irrelevant: minimising the remaining mean over all `k`-unit
deletions is exactly maximising `R`, which the `k` largest unit sums achieve.
Greedy is optimal; no dynamic program is needed. (Brute-forced against every
subset on small inputs in `test_greedy_flip_set_equals_brute_force`.)

`None` is a real answer, not a failure: it means every unit favours A, so A's lead
is not the doing of a handful of items.

### On the demo systems

`sys_a` leads `sys_b` by 3.6 accuracy points (0.6964 vs 0.6607).

| question | answer |
|---|---|
| examples whose deletion reverses the lead | **2 of 56** (`agent-005`, `agent-006`) |
| duplicate clusters whose deletion reverses it | **1** |
| 95% CI on the margin (cluster bootstrap) | `[-0.185, +0.264]` — contains zero |
| sign-flip p (Holm-corrected) | 0.883 |
| times `sys_a` holds first place under resampling | **63.2%** |
| smallest gap this corpus can detect at 80% power | **0.321** |
| examples needed to detect the observed 0.036 gap | **≈ 4500** |

The two examples are one block. **This is a mechanism, not a coincidence:** a
cluster of size *m* contributes *m* times to a margin while carrying one item's
worth of information, so duplicates are the highest-leverage items in any close
comparison. The simulator makes it exact by construction (`icc = 1`); a
deterministic system makes it exact on *exact* duplicates.

A benchmark whose winner is decided by one duplicate cluster is not measuring what
its leaderboard claims to show.

---

## What is reported

| quantity | meaning |
|---|---|
| `BootstrapResult` | point estimate + CI, tagged with `method` and `resample_unit` |
| `icc` | one-way random-effects ICC(1) of the score within blocks |
| `design_effect` | measured `Var(cluster) / Var(i.i.d.)`; `design_effect_kish` is the closed form |
| `effective_n` | `n / deff` — the independent examples the corpus is worth |
| `win_fraction` | reported, never used as a decision rule |
| `p_value`, `p_holm`, `p_mc_se` | sign-flip test, multiplicity correction, and the p's own error |
| `minimum_detectable_effect`, `required_n` | what a corpus this size can and cannot resolve |
| `min_examples_to_flip`, `min_units_to_flip` | exact deletion counts |
| `top1_frequency`, `exact_order_frequency`, `mean_kendall_tau` | ranking under resampling |
| `ranking_resolved` | the single verdict bit |

`ranking_resolved` is true only when three independent conditions hold: the
effective sample size clears 30, the leader holds first place in ≥ 95% of
resamples, and its margin over the runner-up survives Holm correction. On the
committed fixtures it is **False**, and `evalforge stats --fail-on-unstable-ranking`
exits 1.

Ties are common, not exotic: a mean over 56 examples takes one of 57 values.
`top1_frequency` splits a tie's credit evenly rather than breaking it by name
(which would hand an alphabetical system free first-place mass);
`exact_order_frequency` counts only strict orderings; `tie_fraction` reports how
often it mattered (8.8% of block resamples on the demo, 35% on `qa_toy` alone).

### Interval methods

The default is the **percentile** interval. BCa is available via
`--ci-method bca` and is *not* the default, because it is not uniformly better:

| corpus | true accuracy | percentile | BCa |
|---|---|---|---|
| clustered | 0.90 | 0.884 | **0.947** |
| i.i.d. | 0.90 | 0.910 | 0.921 |
| i.i.d. | 0.95 | 0.926 | **0.830** |

At `p = 0.95` on 56 examples the bootstrap distribution is near-degenerate: 5.7%
of trials cannot compute the bias correction `z0` at all (the interval falls back
to percentile, and `method` records `bca->percentile` rather than lying), and the
ones that can are corrected in the wrong direction. An interval method that is
better on average and worse exactly where the data is thinnest is offered, not
imposed.

---

## Reproduce

```bash
# The demo, on the pooled fixture corpus (exit 0):
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

# Real score files, in either the long or the wide JSONL shape:
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --scores datasets/fixtures/qa_toy/system_scores.jsonl

# Show the i.i.d. assumption's cost by asking for it explicitly:
PYTHONPATH=src python3 -m eval_dataset_forge stats \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --simulate --resample-unit example

# The coverage simulations and the type-I error measurements:
python3 -m pytest -q tests/test_statistics.py
```

### The score file format

One JSON object per line, in either shape; they may be mixed in one file.

```json
{"example_id": "qa-001", "system": "sys_a", "score": 1.0}
{"example_id": "qa-001", "scores": {"sys_a": 1.0, "sys_b": 0}}
```

Alignment is **strict**. A score for an example the dataset does not contain, or
an example no system scored, raises. The alternative is to impute, and every
imputation of a missing eval result is a silent decision about a system's score on
an example it may have crashed on.

`datasets/fixtures/qa_toy/system_scores.jsonl` is a committed example of the
format. Its scores are **synthetic** (regenerate with
`python3 scripts/make_demo_scores.py`); they describe no real system, and no
conclusion about one may be drawn from them.

---

## Limitations

- **The demo's system scores are synthetic.** They exist so the machinery can be
  exercised offline and so the coverage tests have ground truth. Findings 1 and 3
  are structural (the block structure comes from the real fixtures, not the
  simulator); nothing here is a claim about any real model.
- **The cluster bootstrap corrects only for dependence the dedup ladder can see.**
  `ret-001`/`ret-002` are a paraphrase pair no lexical threshold reaches
  ([`DEDUPLICATION.md`](DEDUPLICATION.md)), so they yield no block edge and are
  resampled as if independent. The reported design effect is a **lower bound** and
  the clustered intervals remain slightly too narrow. Milestone 4's recall bound is
  this milestone's correction bound.
- **The percentile bootstrap under-covers at extreme accuracies regardless of the
  resampling unit.** 0.910 at *p* = 0.90 on 56 i.i.d. examples. Small `n` and a
  proportion near the boundary is a hard case for every nonparametric interval, and
  BCa does not reliably fix it here.
- **`minimum_detectable_effect` and `required_n` are normal approximations.** Paired
  differences of binary scores are trinomial, not normal. They are a scale, not a
  promise.
- **The sign-flip test assumes exchangeability of A and B within a block.** That is
  the right null for "same system, different name", and a weaker one than
  independence — but it is still an assumption, and a system whose errors correlate
  with block *membership* violates it.
- **The ICC is estimated, not known.** ICC(1) can come out negative on data with no
  real clustering (measured −0.34 on one `icc = 0` simulation). It is reported as
  `nan` when it is undefined — all-singleton units, or a constant score — rather
  than as zero, because absence of evidence is not evidence of independence.
- **This quantifies sampling uncertainty over the *given* examples.** It says
  nothing about whether those examples are representative of the real task. That is
  what the rest of the audit is for, and it is not something any resampling can
  recover.
