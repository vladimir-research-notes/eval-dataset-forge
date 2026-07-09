# Results

The smoke audit (`evalforge smoke --out results/reports/smoke_audit.md`) runs the
entire pipeline over every fixture dataset, fully offline, and this document
summarizes the outcome. Because the fixtures are hand-authored with deliberate
defects, the audit's job is to **rediscover the planted defects**.

The five fixtures landed in Milestone 2 (`datasets/fixtures/`, 56 examples). Each
carries a machine-readable `defects.yaml` recording its planted defects as ground
truth, so every stage below is scored against them rather than reported as a bare
count. The fixtures also validate structurally
(`python3 scripts/validate_fixtures.py`).

> **Every number on this page is measured in sample.** These fixtures' defects were
> planted by the same repository that wrote the detectors, and the near-duplicate
> and answer-in-prompt thresholds were *selected* by sweeps over these same
> fixtures (Milestones 4 and 5). Recall here is a resubstitution estimate: it says
> the pipeline finds what it was built to find, on the data it was built against.
> It does not estimate performance on a corpus this repository has not seen. There
> is no held-out corpus, and saying so is cheaper than implying otherwise by
> silence.

## The headline, and why it is not the recall

**40 of 40 detectable planted defects are recovered, 0 missed, 5 unexplained
assertions across 56 examples.** That number, alone, is worthless — and the smoke
audit proves it by scoring a detector that flags *everything* under the identical
rule:

| detector | recall | precision (`dedup.exact`) | precision (`labels.missing`) |
|---|---|---|---|
| the pipeline | **1.000** | 1.000 | 1.000 |
| flag every example and every pair | **1.000** | 0.014 | 0.071 |

Recall is maximized by asserting everything, so it bounds what a detector *misses*
and says nothing about what it *invents*. No matching rule repairs this: subset
matching, exact-set matching, and the pairwise rule all award the degenerate
detector full recall, because full recall is what it has. Milestone 12 therefore
prints a precision beside every recall, and ships the degenerate control inside the
report so the reader can see what an uninformative answer scores. Details:
[Milestone 12](#smoke-audit--scoring-the-scorer-milestone-12), below.

## Measured so far

Stages land one milestone at a time; each contributes its numbers here as it is
implemented. These are reproducible today, offline, from the committed fixtures.

### Deduplication (Milestone 4)

Scored against the `near_duplicates` / `exact_duplicates` / `duplicate_sources`
ground truth in each fixture's `defects.yaml`. Full method, calibration sweep, and
failure-mode analysis: [`DEDUPLICATION.md`](DEDUPLICATION.md).

| stage | certainty | recall | false positives |
|-------|-----------|--------|-----------------|
| exact | exact | **4 / 4** planted pairs | 0 |
| normalized | exact given normalization | n/a (none planted) | 0 |
| near (TF-IDF, default) | heuristic | **4 / 5** planted paraphrases | 2 pairs, both explained below |
| near (Jaccard, no-sklearn fallback) | heuristic | 3 / 5 | 2 pairs |
| duplicate sources | exact | **2 / 2** planted | 0 |

The two near-duplicate false positives are documented and pinned by tests, not
papered over:

- `qa-003`/`qa-012` — same template ("Who wrote the play *X*?"), same gold answer.
  Independently flagged as a duplicate *source* record, which is the actionable
  signal.
- `agent-003`/`agent-004` — the same `reverse(s)` task, one with the reference
  solution quoted in the prompt. Arguably a true near-duplicate; the fixture's
  ground truth files it under `leakage: answer_in_prompt` instead, so it scores
  as a false positive here.

The one miss, `ret-001`/`ret-002`, is **unreachable by any lexical threshold**: a
genuine non-duplicate in the same fixture (`ret-006`/`ret-009`) scores strictly
higher. Whether its ground-truth entry should therefore carry
`auto_detectable: false` is [flagged for human
review](review_notes/04-deduplication-human-review.md) rather than changed
unilaterally.

Reproduce:

```bash
python3 -m pytest -q tests/test_dedup.py      # 99 tests
```

### Leakage & contamination (Milestone 5)

Scored against the `leakage` ground truth in each fixture's `defects.yaml`. Full
method, calibration sweeps, and failure-mode analysis: [`LEAKAGE.md`](LEAKAGE.md).

Four checks, each emitting one `LeakageKind`. Unlike the dedup ladder they are
**not** disjoint — a defect can and should light up several channels.

| check | certainty | recall | false positives |
|-------|-----------|--------|-----------------|
| cross-split overlap (TF-IDF) | inherited from dedup | **8 / 9** planted pairs | 1 pair, disputed — see below |
| cross-split overlap (Jaccard fallback) | inherited from dedup | 7 / 9 | 1 pair |
| answer-in-prompt | heuristic | **2 / 2** planted leaks | 0 |
| reference n-gram overlap | exact given n=8 | n/a (none planted) | 0 |
| retrieval contamination | exact | **1 / 1** planted artifact | 0 |
| cross-split source records | exact | **2 / 2** planted | 0 |

Per-fixture findings at default settings:

| fixture | overlap | answer-in-prompt | n-gram | retrieval | cross-split sources | examples implicated |
|---------|---------|------------------|--------|-----------|---------------------|---------------------|
| `qa_toy` | 2 | 1 | 1 | 0 | 1 | 6 / 14 |
| `agent_tasks_toy` | 3 | 1 | 1 | 0 | 0 | 6 / 12 |
| `retrieval_toy` | 0 | 0 | 0 | 2 | 1 | 3 / 10 |
| `bio_toy` | 2 | 0 | 1 | 0 | 0 | 4 / 10 |
| `safety_toy` | 2 | 0 | 1 | 0 | 0 | 4 / 10 |

Three results are worth stating precisely.

**An independent channel recovers what the dedup ladder cannot.** `ret-001`/`ret-002`
is planted as a cross-split overlap and is *provably* unreachable by any lexical
prompt threshold (Milestone 4 measured a genuine non-duplicate in the same fixture
scoring strictly higher). The passage-level retrieval check finds the same pair
anyway: both examples cite the identical gold passage under different `doc_id`s. So:

| | TF-IDF | Jaccard |
|---|--------|---------|
| planted leakage defects recovered **by their own kind** | 11 / 12 | 10 / 12 |
| planted leakage defects recovered **by any channel** | **12 / 12** | 11 / 12 |

**The n-gram channel is redundant on these fixtures — an honest null result.** Its
four findings (`qa-002`, `agent-006`, `bio-006`, `saf-005`) are exactly the four
identical cross-split pairs the exact dedup stage already reports. Every planted
overlap is either identical (caught twice) or a paraphrase (invisible to 8-grams, by
design). The channel exists for *external* reference corpora, which the fixtures
cannot demonstrate; the redundancy is asserted by a test rather than hidden.

**One finding contradicts the ground truth.** `agent-003`/`agent-004` straddles
train/test with an identical reference solution, differing only by an appended hint.
`defects.yaml` files that pair under `answer_in_prompt`, so dedup scores it a false
positive — but as cross-split overlap it is arguably a true one. Left unresolved and
[flagged for human review](review_notes/05-leakage-human-review.md) rather than
settled by editing ground truth.

Reproduce:

```bash
python3 -m pytest -q tests/test_leakage.py    # 111 tests
```

### Label audit (Milestone 6)

Scored against the `label_issues` ground truth in each fixture's `defects.yaml`. Full
method, calibration sweep, and failure-mode analysis:
[`LABEL_AUDIT.md`](LABEL_AUDIT.md).

| check | certainty | recall | false positives |
|-------|-----------|--------|-----------------|
| missing | exact | **4 / 4** planted | 0 |
| conflicting (annotators) | exact | **4 / 4** planted | 0 |
| conflicting (shared prompt) | exact | n/a (none planted) | 0 |
| conflicting (relevance channels) | exact | **1 / 1** planted | 0 |
| ambiguous (declared tag) | exact | **5 / 5** planted | 0 |
| ambiguous (lexical heuristic, tags withheld) | heuristic | 4 / 6 declared | 0 |
| inconsistent format | heuristic | **2 / 2** planted | 0 |

**15 / 15 auto-detectable planted label defects recovered, 0 false positives.**

Per-fixture findings at default settings:

| fixture | missing | conflicting | ambiguous | inconsistent format | blocking | examples flagged |
|---------|---------|-------------|-----------|---------------------|----------|------------------|
| `qa_toy` | 1 | 1 | 2 | 0 | 4 | 4 / 14 |
| `agent_tasks_toy` | 1 | 1 | 1 | 1 | 2 | 3 / 12 |
| `retrieval_toy` | 0 | 1 | 1 | 0 | 2 | 2 / 10 |
| `bio_toy` | 1 | 1 | 1 | 0 | 3 | 3 / 10 |
| `safety_toy` | 1 | 1 | 1 | 1 | 3 | 4 / 10 |
| **all five** | 4 | 5 | 6 | 2 | 14 | 16 / 56 |

Four results are worth stating precisely.

**The stage is blind to the defect that matters most — by construction, and it says so.**
Three `mislabeled` defects are planted (`qa-006` answers "the chemical symbol for gold"
with `Ag`; `bio-009` gives a human somatic cell 23 chromosomes; `saf-006` refuses a benign
vaccine question). All three are well-formed, unanimous, unambiguous, and correctly shaped,
and **none is found by any check**. A regression test asserts they stay invisible so the
limitation cannot rot into a silent claim of coverage. Label *truth* needs a second
annotator or a model; this stage sees only structure.

**A check that exists to distinguish two look-alikes.** `ret-004` and `ret-005` both carry
an empty `expected_output`. `ret-004` is a defect — a passage is marked relevant while the
gold doc-id list names none. `ret-005` is correct — an unanswerable, over-specific query
that no candidate passage answers, so an empty gold list is the right answer. A rule keyed
on emptiness flags both. Comparing the two relevance channels as **sets** separates them
without a special case, and recovers the planted `incomplete_relevance` defect that no
other stage in the pipeline sees.

**The ambiguity heuristic is honest about its ceiling.** With every `ambiguous` tag
withheld, it recovers 4 of 6 declared-ambiguous examples with zero false positives. Both of
its guards are load-bearing — dropping the wh-word guard admits `qa-005` ("What is 2 + 2?"),
dropping the scalar-gold gate admits `ret-004` — and both facts are pinned by tests. The two
it misses are unreachable in principle: `saf-007` is a semantic dual-use judgement, and
`ret-005` is *over*-specified rather than under-specified. Declared ambiguity and detectable
ambiguity are different sets.

**The shared-prompt conflict channel fires on nothing — an honest null result.** All four
byte-identical-prompt clusters in the corpus agree on their gold answers, i.e. they are
duplicates, not conflicts. `dedup.duplicate_input_clusters` exists precisely so this stage
can take that difference, and on these fixtures the difference is empty. The channel is
exercised by a synthetic pair, and the null result is asserted by a test.

Reproduce:

```bash
python3 -m pytest -q tests/test_labels.py     # 144 tests
```

### Split construction (Milestone 7)

Unlike the earlier stages, this one does not score itself against `defects.yaml`. It
**changes** the data, so it is scored against its own output: `build_splits()` applies
its assignment and then re-runs Milestone 5's cross-split checks over the splits it just
produced. Full method, sweeps, and failure-mode analysis: [`SPLITS.md`](SPLITS.md).

Residual cross-split duplicate clusters, mean over seeds 0–19, ratios `0.7/0.1/0.2`:

| fixture | random | stratified | source-grouped | **adversarial** | temporal |
|---|---|---|---|---|---|
| `qa_toy` | 1.30 | 1.30 | 1.00 | **0.00** | 1.00 |
| `agent_tasks_toy` | 1.75 | 1.75 | 1.75 | **0.00** | 0.00 |
| `bio_toy` | 1.10 | 1.10 | 1.10 | **0.00** | 0.00 |
| `retrieval_toy` | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| `safety_toy` | 0.95 | 0.95 | 0.95 | **0.00** | 0.00 |
| **all five pooled** | 5.45 | 5.25 | 4.60 | **0.00** | 1.00 |

Four results are worth stating precisely.

**The grouping key the dataset hands you is not the one that closes the leak.** Every
planted duplicate carries a *fresh* `source_id` (`src-fr-capital` →
`src-fr-capital-dup`, `src-fizzbuzz` → `src-fizzbuzz-para`) — exactly how a real corpus
records a paraphrase written as a new record. `source-grouped` therefore leaves all of
them free to straddle, and measures barely better than `random`. On the three fixtures
where no `source_id` is shared at all, it produces a **bit-for-bit identical assignment
to `random`** at every seed: not a weaker control, the same function. Only blocks
derived from example *content* close the channel.

**Stratifying on more facets makes the test set less representative, not more.** Total
variation distance between the test split's marginal and the corpus's (mean over 20
seeds, pooled corpus):

| `stratify_by` | strata | TVD family ↓ | TVD difficulty ↓ | dev size | `hard` in test |
|---|---|---|---|---|---|
| (none) | 1 | 0.216 | 0.156 | 6.0 | 0.45 |
| **`task_family`** (default) | 5 | **0.036** | 0.152 | 5.0 | 0.25 |
| `difficulty` | 4 | 0.184 | **0.030** | 5.0 | 1.00 |
| `task_family,difficulty` | 17 | 0.083 | 0.077 | 3.0 | **0.00** |

Joint stratification is worse on *both* facets than either facet alone: strata multiply
while the corpus does not, and every stratum then starves the small splits at once. It
also guarantees an empty cell — the three `hard` examples sit in three different family
strata of size one, each of which apportions entirely to `train`, so **no `hard` example
reaches the test set** and the benchmark cannot measure difficulty. Two warnings surface
it. The default is a single facet.

**The label audit's hand-off costs something, and the cost is reported.** On the pooled
corpus an adversarial split without quarantine scores on six examples whose gold answer
Milestone 6 says a human must adjudicate. `--quarantine-blocking` moves all fourteen
blocking examples to `train`; `test` shrinks from 12 to 8 and its achieved ratio falls
from 0.214 to 0.143. Nothing is deleted.

**The adversarial guarantee is inherited from a heuristic — and it says so.** Zero
residual overlap holds *by construction*, because the blocks are the dedup clusters. So
it is exactly as good as the near stage's recall. `ret-001`/`ret-002` is a planted
cross-split paraphrase that no lexical threshold can reach; the adversarial split
straddles it exactly as `random` would, and the verifier reports a clean zero. A test
pins the miss. Relatedly, `retrieval_toy`'s all-zero row measures the near stage's
answer gate rather than the data, and is called out rather than counted as a win.

Reproduce:

```bash
python3 -m pytest -q tests/test_splits.py     # 723 tests
```

### Difficulty calibration (Milestone 8)

This stage is not scored against `defects.yaml` either — there is no ground-truth
difficulty planted in the fixtures, only the `difficulty_tags` their author wrote down. So
it is scored against a falsifiable question: **is the heuristic's ranking distinguishable
from a random ranking of those declared labels?** Full method, controls, and failure-mode
analysis: [`DIFFICULTY.md`](DIFFICULTY.md).

**It is not.** Over the 51 labelled examples of the pooled corpus:

| quantity | value | reading |
|---|---|---|
| Spearman ρ (score vs declared) | **+0.097** (p = **0.497**) | not distinguishable from chance |
| within-family ρ (size-weighted) | +0.030 | collapses when family is held constant |
| η² of the **score** by `task_family` | **0.434** | the score is 43% task family |
| η² of the **declared label** by `task_family` | **0.014** | declared difficulty is 1% task family |
| marginal-matched bucket accuracy | 0.529 | below the 0.569 of "call everything easy" |
| n needed for 80% power | ~824 | the corpus has 51 |
| weight-simplex sweep (2000 draws) | median +0.024, max +0.286 | 94% inside \|ρ\| < 0.2 |

Four results are worth stating precisely.

**The difficulty heuristic is a task-family detector.** Its dominant axis is length;
length separates task families; and declared difficulty is nearly orthogonal to task
family (η² = 0.014). The single highest-scoring example in the corpus is `agent-002` — a
paraphrase of FizzBuzz, declared `easy` by its author. Holding family constant collapses
the correlation to +0.030, and on `retrieval` it inverts to −0.491.

**No reweighting rescues it, so the null is a property of the signals.** 94% of 2000
Dirichlet draws from the weight simplex land inside |ρ| < 0.2, and the best draw — chosen
*in sample*, hence an upper bound — reaches only +0.286. Of the five signals, `tool_use`
fires on **0 of 56** examples and is reported inert rather than credited; `output_complexity`
(−0.138) and `evidence_items` (−0.163) are **anti-correlated** with declared difficulty;
and `input_length` alone clears α (p = 0.043) but not Bonferroni over the four testable
signals (p = 0.174).

**The calibration target must never be a scoring signal.** An earlier design listed the
hand-assigned difficulty as both an input to the score and the sanity check on it.
Blending the declared bucket back in at a weight of 0.15 lifts ρ from **+0.10 to +0.49**
(at 0.30, to +0.77) — a null heuristic reading as a working one. The scorer therefore
never reads `difficulty_tags`, and a test asserts that rewriting an example's declaration
leaves its score untouched.

**A missing gold answer makes an example look easy.** An absent `expected_output` scores
zero on `output_complexity`, so the four examples Milestone 6 flags `MISSING` sink toward
the easy end: `agent-009` ranks **1 of 56** and `qa-010` ranks **2 of 56**. Milestone 6's
label defect corrupts Milestone 8's score, and the report says so instead of imputing.

Consequently `DifficultyReport.trustworthy` is **False** on every fixture and on the
pooled corpus, and `evalforge difficulty --fail-on-uncalibrated` exits 1. That is the
intended outcome. It vindicates Milestone 7's decision to stratify on the *declared* tags
rather than a modelled bucket — with a measurement, not an argument. The harness is not
merely pessimistic: a synthetic corpus whose difficulty genuinely is surface-driven comes
back ρ > 0.9, `trustworthy = True`.

Per-fixture, every dataset is underpowered on its own (9–12 labels):

| fixture | labelled | ρ | p | verdict |
|---|---|---|---|---|
| `qa_toy` | 12 | −0.080 | 0.789 | not calibrated |
| `agent_tasks_toy` | 11 | +0.230 | 0.516 | not calibrated |
| `retrieval_toy` | 10 | −0.491 | 0.164 | not calibrated |
| `bio_toy` | 9 | +0.174 | 0.667 | not calibrated |
| `safety_toy` | 9 | +0.365 | 0.357 | not calibrated |
| **all five pooled** | 51 | **+0.097** | **0.497** | **not calibrated** |

Bucket distributions on the pooled corpus (the milestone's reporting deliverable):

| bucketing | easy | medium | hard | cut points |
|---|---|---|---|---|
| `quantile` (default) | 18 | 19 | 19 | 0.182 / 0.301 |
| `fixed` | 45 | 11 | **0** | 0.333 / 0.667 |
| `marginal` | 32 | 21 | 3 | 0.274 / 0.524 |
| *declared* | 29 | 19 | 3 | (5 unlabelled) |

`fixed` leaves `hard` empty because the maximum score on this corpus is 0.658: an absolute
cut point on a log-compressed scale is unreachable for short-answer data. A warning fires,
and `quantile` is the default for that reason.

Reproduce:

```bash
python3 -m pytest -q tests/test_difficulty.py # 163 tests
```

### Statistics & ranking stability (Milestone 9)

This stage has no planted defect to rediscover, and its system scores are **synthetic**
(no model runs offline). So it is scored the only way a statistical method can be: against
**known ground truth in simulation** — does a 95% interval contain the true parameter 95%
of the time? Full method, tables, and failure-mode analysis:
[`STATISTICS.md`](STATISTICS.md).

**It does not, and the reason is the corpus's own duplicates.** 3000 trials, `B = 999`,
44 blocks with the pooled corpus's exact size profile, nominal coverage 0.95:

| true accuracy | i.i.d. bootstrap | **cluster bootstrap** | cluster + BCa |
|---|---|---|---|
| 0.50 | 0.904 | **0.944** | 0.954 |
| 0.70 | 0.882 | **0.937** | 0.956 |
| 0.90 | 0.847 | **0.884** | 0.947 |

On genuinely i.i.d. data the same percentile interval covers 0.954 — the positive control
that makes the rows above a finding rather than a broken harness.

Four results are worth stating precisely.

**A duplicate cluster is not one example, and the textbook bootstrap counts it as one.**
A deterministic system scores every member of an exact-duplicate cluster identically, so
the within-cluster correlation is 1 *by construction*, not by measurement. Resampling
examples therefore understates every interval. On the pooled corpus the measured design
effect is 1.39–1.56 (Kish's closed form predicts 1.50), CIs widen 1.16×–1.29×, and 56
examples are worth **≈ 36 independent ones**. Simulating the same blocks with zero
within-block correlation returns a design effect of 1.0, so the widening tracks the
dependence and not the choice of unit.

**The bootstrap "win fraction" is not a p-value, and the error is not small.** Under a
true null (`n = 56`, two Bernoulli(0.7) systems), the widespread rule "A beats B if A wins
≥ 95% of resamples", used in whichever direction the data favours, rejects **11.4%** of
the time against a nominal 5% — it is a one-sided threshold read two-sidedly. The
corresponding two-sided threshold, 0.975, measures 0.057. The tool reports the win
fraction and **decides** with a cluster-aware sign-flip permutation test, which measures
0.029: conservative, because paired binary differences are mostly ties. All three numbers
are pinned by tests.

**One duplicate cluster decides the demo leaderboard.** `sys_a` leads `sys_b` by 3.6
accuracy points. Deleting **2 of 56 examples** reverses it — and those two examples,
`agent-005`/`agent-006`, are a single near-duplicate cluster. This is a mechanism, not a
coincidence: a cluster of size *m* contributes *m* times to a margin while carrying one
item's worth of information. The margin's 95% CI is `[-0.185, +0.264]`, the leader holds
first place in 63.2% of resamples, and detecting a gap that size would need ≈ 4500
examples. `ranking_resolved` is **False**, and `--fail-on-unstable-ranking` exits 1.

**BCa is offered and not defaulted, because it is not uniformly better.** It repairs the
clustered interval at an extreme accuracy (0.884 → 0.947 at *p* = 0.90) and **breaks** the
i.i.d. one (0.926 → 0.830 at *p* = 0.95), where 5.7% of trials cannot compute the bias
correction at all and fall back — reported as `method: "bca->percentile"` rather than
silently relabelled.

Two limits are inherited rather than introduced. The cluster bootstrap corrects only for
dependence the dedup ladder can *see*, so the `ret-001`/`ret-002` paraphrase yields no
block and the reported design effect is a **lower bound**. And the demo's system scores
are simulated: the block structure is real, the systems are not, and no claim about any
model is made.

Reproduce:

```bash
python3 -m pytest -q tests/test_statistics.py # 115 tests
```

### Release identity (Milestone 10)

A release is a claim that two scores are comparable. The question this stage had to settle
before deciding what a manifest records: **is a dataset's content enough to identify it?**
Full method: [`RELEASES.md`](RELEASES.md).

**It is not.** Hold the corpus fixed, hold each system's per-example score vector fixed, and
vary only the split seed (pooled corpus: 56 examples, 44 blocks, 3 simulated systems, 400 seeds
per strategy):

| quantity | value |
|---|---|
| distinct `content_id` over all 400 seeds × 3 strategies | **1** |
| full-corpus accuracy, `sys_a` / `sys_b` | 0.6964 / 0.6607 |
| **the gap the benchmark exists to resolve** | **+0.0357** |
| `sys_a` test accuracy under `random` | mean 0.7011, **sd 0.1300**, range **[0.364, 1.000]** |
| P(test split reproduces the corpus ordering of the top two) | **0.487** |
| leader by seed, `random` | `sys_a` 186 · `sys_b` 128 · `sys_c` 8 · tie 78 |

The split seed moves the headline number by **3.6× the effect it is meant to resolve**, and which
system "wins" is decided by the seed. The mean over seeds recovers the corpus number, so this is
variance, not bias. The consequence for this milestone is structural: a release identified only by
a content hash would call all 400 of those releases the same. So the manifest carries `content_id`
(*"the same examples?"*) **and** `eval_id` (*"comparable scores?"*), and the version bump keys on
the second — a pure re-split at unchanged content is a **major** bump.

A tempting misreading, refuted in `RELEASES.md` §2: `adversarial` agrees with the corpus ordering
65.0% of the time to `random`'s 48.7% — not because it estimates the same quantity better, but
because it estimates a **different** one. It evicts duplicate clusters from `dev`/`test`, and both
systems score higher on duplicated examples (0.750/0.700) than on singletons (0.667/0.639), so its
test set carries a larger true gap (+0.103 vs +0.036) that is correspondingly easier to detect.

Verified end to end: the committed demo release
[`datasets/releases/qa_toy_v0.1.0`](../datasets/releases/qa_toy_v0.1.0) passes all ten
`verify-release` checks; the command printed in its own `REPRODUCE.md` regenerates it
byte-for-byte; two builds at different clocks differ in `manifest.json` and nothing else; and a
single flipped byte in `examples.jsonl` is caught by three independent checks. The manifest
records `block_count`/`block_size_histogram` and **no** `effective_n` — that number needs an ICC,
which needs system scores, which a dataset does not have. This answers the question Milestone 9
left open.

Reproduce:

```bash
python3 scripts/split_identity_experiment.py --seeds 400
python3 -m pytest -q tests/test_release.py     # 72 tests
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0
```

### Dataset card (Milestone 11)

The committed demo release ships
[`dataset_card.md`](../datasets/releases/qa_toy_v0.1.0/dataset_card.md), generated from that
release's `manifest.json` and `audit.json`. Nothing in it is recomputed and nothing is
hand-written except four prose sections, which come from
[`card.yaml`](../datasets/fixtures/qa_toy/card.yaml) and are recorded in `manifest.card_meta`.
Method and failure modes: [`DATASET_CARD.md`](DATASET_CARD.md).

**The card's headline is two numbers, because one of them is not a sample size.**

| what the card prints | qa_toy v0.1.0 |
|---|---|
| examples | 14 |
| independent blocks | **11** |
| blocks in `test` | **2** (of 2 examples) |
| blocks in `train` | 8 (of 11 examples) |
| blocks straddling a split boundary | 0 (adversarial strategy) |

A reader spends the first number as a sample size. The number that governs a confidence
interval on a **test** score is the third. This answers the question Milestone 10 left open, and
it is why `manifest.dependence` gained `blocks_by_split`.

**Two absences that are arithmetic, not taste.** `dataset_card.md` is an artifact hashed into
`manifest.hashes`, so it can print neither that table (its own digest would have to appear
inside its own bytes) nor a build timestamp (a clock in a hashed artifact makes the `none`
version bump unreachable — the defect Milestone 10 avoided in `REPRODUCE.md`). Measured:

```text
two builds of qa_toy, created_at 2020-01-01 vs 2031-12-31, everything else fixed
  same    REPRODUCE.md        same    dataset_card.md     same    examples.jsonl
  same    audit.json          same    splits/{train,dev,test}.jsonl
  DIFFER  manifest.json
```

**Editing curator prose is a `patch`.** It changes no number and cannot move a score, so
`card_meta` is a packaging field:

```text
delta: 0.1.0 → 0.1.1 (patch bump; scores still comparable)
  content_id : same
  eval_id    : same
  - packaging only: hashes, card_meta
```

Reproduce:

```bash
python3 -m pytest -q tests/test_reports.py     # 64 tests
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check          # exit 0
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --fail-on-incomplete   # exit 0: 4/4 sections
```

### Smoke audit — scoring the scorer (Milestone 12)

Every stage above scored itself against `defects.yaml` by hand. This milestone runs
the whole pipeline over every fixture and does that scoring in code — which forced
the question the hand-written tables never had to answer: **what is a correct
scoring rule for a defect detector?**

**Recall is not one.** A detector that flags everything finds every planted defect.
Measured, on the same scorer, over the same corpus:

| | pipeline | flag-everything control |
|---|---|---|
| recall over 40 detectable planted defects | **1.000** | **1.000** |
| planted defects missed | 0 | 0 |
| precision, `dedup.exact` | 1.000 | 0.014 |
| precision, `dedup.near` | 0.571 | 0.017 |
| precision, `leakage.train_test_overlap` | 0.727 | 0.031 |
| precision, `labels.missing` | 1.000 | 0.071 |
| precision, `labels.ambiguous` | 0.833 | 0.089 |

The control is in the shipped report, not in a footnote: `evalforge smoke` scores it
every run. Channel by channel the pipeline's precision is **9× to 190×** the
control's, and the gap is widest on the pair channels for a structural reason — a
flag-everything detector must assert all `C(n,2)` pairs but only `n` examples, so
its pair precision falls as `1/n²` and its example precision only as `1/n`.

**A cluster is a conjunction of assertions, not one assertion.** Duplicate findings
are clusters. Scoring a cluster as a single prediction forces a bad choice: count it
correct if it *contains* a planted pair, and one 14-member cluster is "correct";
count it correct only if it *equals* a planted pair, and a true 3-member cluster
covering a planted pair is "wrong". A cluster of size *k* asserts all *k(k−1)/2* of
its pairs, so that is the unit both recall and precision are counted on.

This is not bookkeeping. On `qa_toy` the near stage emits one cluster,
`{qa-003, qa-004, qa-012}`. It thereby asserts `qa-004 ≈ qa-012` — a pair nothing
plants, and which no cluster-level view would ever show. Pair decomposition surfaces
it as the channel's one unexplained assertion and drops its precision to 0.571.

**The denominator is chosen by the authors of the detectors.** 45 defects are
planted; 40 enter the recall denominator. The 5 that do not are listed, by id, with
which of *two different* reasons excluded them:

| fixture | kind | examples | reason |
|---|---|---|---|
| `qa_toy` | mislabeled | `qa-006` | `not-auto-detectable` (the fixture's declaration) |
| `bio_toy` | mislabeled | `bio-009` | `not-auto-detectable` |
| `safety_toy` | mislabeled | `saf-006` | `not-auto-detectable` |
| `retrieval_toy` | paraphrased_documents | `ret-006/d13`, `ret-006/d14` | `no-channel` (this project has no detector) |
| `retrieval_toy` | over_specific_query | `ret-005` | negative control — must **not** be flagged |

Collapsing the two reasons would let a missed defect be reclassified as an
out-of-scope one. Recall over *every* planted defect, refusing all exclusions, is
**0.909**, and the report prints it beside the 1.000. A ground-truth kind this
project has not wired to a channel raises `ValueError` rather than being skipped: a
planted defect must not leave the denominator by typo.

**Corroboration across channels is not a false positive.** Of the assertions absent
from a channel's own ground truth, most are recorded elsewhere in it. The n-gram
check flags `qa-002` — planted as half of an `exact_duplicates` pair. The retrieval
check flags `ret-001`/`ret-002` — planted as a `duplicate_documents` pair. The
scorer projects ground truth into the channel's unit and counts these as *explained*.
The projection is unit-preserving, which is what stops it from becoming a blanket
pardon: `qa-004` and `qa-012` are each planted, and their *pairing* is not, so it
stays unexplained.

Across the corpus exactly **five** assertions are unexplained, and all five were
already known:

| fixture | channel | assertion | status |
|---|---|---|---|
| `qa_toy` | `dedup.near` | `qa-004 + qa-012` | implied by a 3-member cluster; also a planted duplicate *source* pair |
| `qa_toy` | `leakage.train_test_overlap` | `qa-004 + qa-012` | the same cluster, echoed by the channel that inherits it |
| `agent_tasks_toy` | `dedup.near` | `agent-003 + agent-004` | disputed since Milestone 5 — arguably a true near-duplicate |
| `agent_tasks_toy` | `leakage.train_test_overlap` | `agent-003 + agent-004` | the same, echoed |
| `retrieval_toy` | `labels.ambiguous` | `ret-005` | **new**: `ret-005` declares `difficulty_tags: [hard, ambiguous]`, and the ground truth files it under `relevance_issues/over_specific_query` |

The last one is a gap in the ground truth, not a detector error: the ambiguity check
is an exact reading of a declaration the example carries. Left unresolved and
[flagged for human review](review_notes/12-smoke-audit-human-review.md) rather than
settled by editing `defects.yaml` — the same discipline Milestones 4 and 5 applied to
their disputed pairs.

**The scorer reproduces every number Milestones 4–6 computed by hand.** `dedup.exact`
4/4, `dedup.near` 4/5, `duplicate_sources` 2/2, `train_test_overlap` 8/9,
`answer_in_prompt` 2/2, `retrieval_contamination` 1/1 — all recomputed from the
fixtures by `eval_dataset_forge.scoring` and pinned by a test. The one defect its own
channel cannot reach, `ret-001`/`ret-002`, is recovered by the passage-level retrieval
channel, so **0** planted defects are invisible to the whole pipeline.

Per-fixture, at default settings:

| fixture | examples | blocks | duplicate clusters | leakage findings | label warnings | PII |
|---|---|---|---|---|---|---|
| `qa_toy` | 14 | 11 | 2 | 4 | 8 | 0 |
| `agent_tasks_toy` | 12 | 9 | 3 | 5 | 6 | 0 |
| `retrieval_toy` | 10 | 9 | 0 | 2 | 4 | 0 |
| `bio_toy` | 10 | 8 | 2 | 3 | 6 | 0 |
| `safety_toy` | 10 | 8 | 2 | 3 | 7 | 0 |
| **pooled** | **56** | **44** | 10 | 18 | 31 | 0 |

The pooled corpus is audited and **not scored**: pooling lets the ladder compare
examples across fixtures, and the per-fixture ground truth says nothing about a
cross-fixture pair — scoring it would measure the ground truth's silence. Its 44
independent blocks are the sample size Milestone 9's bootstrap resamples.

**Running it anyway found something.** The pooled audit reports 10 duplicate clusters
against the per-fixture sum of 9, and the extra one is not cross-fixture. It is
`{saf-004, saf-005, saf-007}` — entirely inside `safety_toy`, which on its own reports
no such cluster:

| | `safety_toy` alone (n=10) | pooled corpus (n=56) |
|---|---|---|
| `sim(saf-004, saf-007)` | 0.2866 | **0.3438** |
| `sim(saf-001, saf-003)` | 0.4319 | 0.5071 |
| near-duplicate threshold | 0.30 | 0.30 |

Neither prompt changed. TF-IDF's inverse-document-frequency term is computed over the
document *collection*, so the boilerplate the two safety prompts share
("SYNTHETIC…PLACEHOLDER", "no real content") is common inside `safety_toy` and rare
against the wider pooled vocabulary — and rarer means heavier. The pair crosses the
threshold because of what else is in the corpus.

So **a near-duplicate verdict is not a property of a pair; it is a property of a pair
and its corpus.** Two consequences worth stating plainly:

- **Deduplication is not composable.** Auditing shards separately and unioning the
  findings is not auditing the union.
- **Appending unrelated examples can create duplicate clusters among examples that did
  not change.** The adversarial split builder derives its blocks from duplicate clusters,
  and `eval_id` folds the split assignment in — so appending unrelated data can move the
  split of an untouched example, and change `eval_id`. That is a Milestone 10 identity
  claim reached from a Milestone 4 implementation detail; it is
  [flagged for review](review_notes/12-smoke-audit-human-review.md) and deserves its own
  experiment rather than a paragraph here.

`test_a_near_duplicate_verdict_is_a_property_of_the_corpus_not_of_the_pair` pins the
measurement. Whether the pure-Python Jaccard fallback — which has no IDF term — is immune
is **not yet measured**; if it is, then `scikit-learn` changes the *semantics* of a
duplicate report, not only its speed.

**Two smaller results.** An audit describes the split it is *given* and never builds
one, so `evalforge audit` on an unsplit corpus finds that four checks — cross-split
overlap, cross-split source reuse, n-gram containment against the `train` reference,
and the `dev`/`test` duplicate count — return zero *by vacuity*. The audit says so, in
a warning, in a recommendation naming the four, and in `audit.md`. A check that could
not run is not a check that passed.

And the release bundle is finally complete: `audit.md` — listed in `docs/PLAN.md` §5D
since Milestone 0 and absent from every release until now — ships beside `audit.json`,
rendered from it and from nothing else. Adding it to the committed demo release is a
**patch** bump by Milestone 10's own machinery (`content_id` and `eval_id` unchanged,
"packaging only: hashes"), so scores measured on the old bundle still stand; the
release was rebuilt in place at v0.1.0 rather than re-versioned, and a test pins the
reasoning.

Reproduce:

```bash
PYTHONPATH=src python3 -m eval_dataset_forge smoke \
  --out results/reports/smoke_audit.md \
  --fail-on-missed-defect --fail-on-control-violation      # exit 0
PYTHONPATH=src python3 -m eval_dataset_forge audit \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --out results/audits/qa_toy/
python3 -m pytest -q tests/test_audit.py       # 49 tests
```

### Integration contracts (Milestone 13)

Writing the five downstream contracts measured three things the pipeline had not stated.
Full argument: [`INTEGRATION.md`](INTEGRATION.md).

**1. The "optional accelerator" is a different detector.** `scikit-learn` is listed in
`docs/PLAN.md` §3 as an optional accelerator with a pure-Python fallback. Rebuilding
`bio_toy` from identical bytes, an identical command, and an identical seed:

| | scikit-learn present | scikit-learn absent |
|---|---|---|
| `near_method` | `tfidf` | `jaccard` |
| near-duplicate clusters | 1 (`bio-001` ≈ `bio-002`) | 0 |
| independent blocks | 8 | 9 |
| `content_id` | *identical* | *identical* |
| `eval_id` | **different** | **different** |
| held-out example set | **different** | **different** |

`content_id` holding is correct. `eval_id` moving is also correct: these are not the same
benchmark. What was false was `REPRODUCE.md`, which claimed the build was "deterministic
given the same inputs and seed". The environment is an input, and the file now says so.
`verify-release` runs no detector, so it still passes 10/10 without the accelerator —
**verification is environment-independent; reproduction is not.**

**2. Neither block counts nor gate verdicts survive a union.** The five fixtures are
disjoint and share no example:

| | examples | independent blocks |
|---|---|---|
| audited separately, summed (`tfidf`) | 56 | **45** |
| audited as one corpus (`tfidf`) | 56 | **44** |
| audited as one corpus (`jaccard`) | 56 | **46** |

The pooled corpus contains **zero cross-fixture duplicate edges**. Merging five unrelated
datasets found no duplicate *between* them and still changed one dataset's verdict *inside*
itself — `{saf-004, saf-005}` grows to `{saf-004, saf-005, saf-007}` — because idf is
computed over the collection. Correspondingly, `safety_toy` alone **passes** a
`max_near_duplicate_clusters: 1` policy and the same ten rows **fail** it inside the pooled
corpus. Under `jaccard`, which has no idf, both counts are equal: composability is a
property of the similarity function. Example count is additive; statistical power is not.

A verdict does not survive a **re-split** either, for four of the fourteen requirements — a
bug in this milestone's own first draft, caught by checking a docstring instead of shipping
it. Building `qa_toy` forty times with `--strategy random`, all forty releases share **one**
`content_id`, seeds 9 and 35 *pass* `eval_splits_deduplicated`, and thirty-eight seeds *fail*
it. `GateReport.scope` is therefore `eval_id` whenever a split-sensitive requirement ran, and
it is derived from what executed rather than from `split_spec.strategy` — a strategy field is
a claim, not evidence.

**3. A gate needs a third verdict.** `pass` / `fail` / `undecidable`, and `undecidable`
never counts as a pass. Two requirements reach it, for different reasons:
`no_retrieval_contamination` is undecidable on `qa_toy` because no example carries
passages (a fact about the corpus), and `difficulty_is_calibrated` is undecidable on
*every* release this tool has built, because `audit.json` keeps the three difficulty
bucket counts and drops the permutation test that says whether they mean anything (a fact
about the release format). Milestone 11's review note already asked whether that field
should exist; the contract makes the gap executable and gives it a second cost — a
consumer cannot gate on the answer either.

All five profiles reject the release built from the deliberately flawed `qa_toy` fixture —
a gate that passed it would be broken. The positive control is a `--drop`-repaired release
that passes.

| profile | verdict | pass | fail | undecidable | blocking |
|---|---|---|---|---|---|
| `agent-control-bench` | **FAIL** | 4 | 2 | 0 | `no_exact_duplicates`, `no_answer_in_prompt` |
| `bioagentbench` | **FAIL** | 3 | 3 | 0 | `no_exact_duplicates`, `no_label_conflicts`, `no_missing_labels` |
| `code-agent-runtime` | **FAIL** | 3 | 3 | 0 | `no_missing_labels`, `no_answer_in_prompt`, `no_exact_duplicates` |
| `llm-regression` | **FAIL** | 5 | 1 | 0 | `eval_split_is_rankable` — the `test` split holds **2 blocks**, not 30 |
| `scientific-claim-verifier` | **FAIL** | 2 | 3 | 1 | 3 findings; `no_retrieval_contamination` undecidable |

Reproduce:

```bash
python3 -m pytest -q tests/test_contracts.py               # 50 tests
PYTHONPATH=src python3 -m eval_dataset_forge gate --list
PYTHONPATH=src python3 -m eval_dataset_forge gate \
  --release datasets/releases/qa_toy_v0.1.0 --profile llm-regression   # exit 1
```

### Auditing the documentation (Milestone 14)

Every stage above audits a dataset. This one audits the prose, and Milestone 13 supplied
the reason by accident. Full method and failure modes: [`CLAIMS.md`](CLAIMS.md).

**A generated artifact is enforced; prose is not.** Milestone 13 measured that a release is
not reproducible from its inputs and seed alone — the environment is an input — and corrected
the sentence that said otherwise in exactly one file: `REPRODUCE.md`, which is generated by
`render_reproduce()` and pinned by a test. The identical false sentence survived untouched in
three files no test had ever read:

| file | status before Milestone 14 |
|---|---|
| `datasets/releases/*/REPRODUCE.md` | corrected by Milestone 13 (generated, test-pinned) |
| `docs/REPRODUCIBILITY.md`, `docs/LIMITATIONS.md`, `docs/INTEGRATION.md` | corrected (they exist to state the correction) |
| **`README.md`**, **`docs/METHODOLOGY.md`**, **`docs/PLAN.md`** | **still asserted the retired sentence** |

`scripts/verify_docs.py` runs ten checks over the live documents and the historical record: every
relative Markdown link, every shell command in a fenced block, and 11 registered numeric claims.
On its first run the retired-claim lint found the sentence in exactly those three files and in
no other live document.

**Three results are worth stating precisely.**

**1. A number in a document is not a claim until it names the command that produced it.**
`docs/RESULTS.md` reports the `qa_toy` difficulty calibration p-value as **0.789**. Running
`evalforge audit` on the same fixture, at the same seed, prints **0.777**:

| command | `--resamples` default | permutation p |
|---|---|---|
| `evalforge difficulty` | 10000 | **0.789** |
| `evalforge audit` | 2000 | **0.777** |

Same fixture, same seed, same statistic (ρ = −0.080 in both). Neither number is wrong; the gap
is the permutation test's Monte-Carlo error, which shrinks like `1/√B`. A reader comparing the
two documents has no way to discover why they differ. So every claim in the ledger carries the
command that produces it, and `check_claim_commands` verifies that column names a command that
exists.

**2. Two published figures had no command that could produce them.** That check caught the first
draft of `pooled_blocks_jaccard`, which cited `evalforge audit --near-method jaccard` — a flag
nobody had built. `docs/RESULTS.md` and `docs/DEDUPLICATION.md` have published Jaccard recall and
block counts since Milestone 4, and the only way to reproduce them was to **uninstall
scikit-learn**: `auto` resolution let the installed packages decide which detector ran. Milestone
14 added `evalforge audit --near-method {auto,tfidf,jaccard}`. A published number can now be
reproduced without reproducing the environment that produced it — which is what Milestone 13's
`Environment` section had been *describing* rather than *controlling*.

**3. A phrasing lint cannot separate a claim from its correction, and the measurement says so.**
`sklearn_is_an_accelerator` is a true proposition to retire. Run over the live documents as this
milestone found them:

| | count | which |
|---|---|---|
| documents flagged | 2 | `docs/LIMITATIONS.md`, `docs/RESULTS.md` |
| flagged documents that **assert** it | 0 | both quote it in order to refute it |
| documents that **do** assert it | 1 | `docs/PLAN.md` §3 |
| of those, flagged | 0 | the phrase spans a line break the pattern cannot cross |

Precision 0/2, recall 0/1. The rule is registered in `CANDIDATE_RETIRED_CLAIMS`, enforced
nowhere, and a test asserts nothing picks it up — the precedent being Milestone 13's
`difficulty_is_calibrated`, registered and in no profile. `docs/PLAN.md` §3 and the
`pyproject.toml` comment were fixed by **reading them**. Run against the corrected documents the
rule flags three, and the third is `docs/CLAIMS.md` — the document that exists to explain why the
rule is disabled.

**The auditor is a defect detector, and it is scored like one.** A checker reporting zero
violations is indistinguishable, on that number alone, from a checker that looks at nothing — the
mirror image of the flag-everything control above. So `tests/test_docs.py` plants a defect for
every check and asserts the check fires: a broken link, an invented subcommand, a renamed flag, a
stale count, a fixture id no fixture holds, a retired claim retyped, a ledger row the code does
not reproduce, a drifted README, a claim citing a command that does not exist, an orphaned entry
point. Both false positives the auditor ever produced were use/mention errors — a quotation
wrapping across a line break, and a `[text](url)` written inside backticks — and both are pinned
by tests.

Two limits are structural. The ledger proves the documentation agrees with the code; **it does
not prove the code is correct** — that is what the rest of the test suite is for. And retired claims
are matched as phrasings, not propositions: recall over "false sentences in this repository" is
not measurable, because unlike the fixtures, whose defects are planted and enumerated, no
ground-truth list of false sentences exists.

The first thing the new check caught was drift this milestone had itself created: adding two
tests to `tests/test_audit.py` invalidated the `# 47 tests` comment on line 699 of this file.

Reproduce:

```bash
python3 scripts/verify_docs.py                # exit 1 on any unsupported claim
python3 -m pytest -q tests/test_docs.py       # 70 tests
```

## Reproduce

```bash
PYTHONPATH=src python3 -m eval_dataset_forge smoke --out results/reports/smoke_audit.md
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full command set.
