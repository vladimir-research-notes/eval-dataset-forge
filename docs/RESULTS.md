# Results

> Status: pending Milestone 12 (Smoke audit & results). This file is populated
> when the full audit pipeline runs over the committed fixtures. It currently
> describes what will be reported.

The smoke audit (`evalforge smoke --out results/reports/smoke_audit.md`) runs the
entire pipeline over every fixture dataset, fully offline, and this document
summarizes the outcome. Because the fixtures are hand-authored with deliberate
defects, the audit's job is to **rediscover the planted defects** — that is the
demonstration that the checks work.

The five fixtures landed in Milestone 2 (`datasets/fixtures/`, 56 examples). Each
carries a machine-readable `defects.yaml` recording its planted defects as ground
truth, so the audit can be scored for **recall** here (found ÷ planted, excluding
defects marked `auto_detectable: false`) rather than reported as a bare count. The
fixtures already validate structurally (`python3 scripts/validate_fixtures.py`).

## Expected findings per fixture (to be confirmed by the run)

| fixture | planted defects the audit should surface |
|---------|-------------------------------------------|
| `qa_toy` | duplicate + paraphrased questions, mislabeled answers, ambiguous questions, train/test overlap |
| `agent_tasks_toy` | near-duplicate tasks, leaked solutions, ambiguous scoring |
| `retrieval_toy` | duplicated docs, query–document leakage, incomplete relevance labels |
| `bio_toy` | entity-label conflicts, evidence-label mismatches |
| `safety_toy` | over-/under-refusal and ambiguous-boundary label conflicts |

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

## What will be reported here

- Per-fixture counts of exact/near duplicates, leakage flags, label warnings, PII
  flags.
- Difficulty distributions per fixture and per split.

## Reproduce

```bash
PYTHONPATH=src python3 -m eval_dataset_forge smoke --out results/reports/smoke_audit.md
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full command set.
