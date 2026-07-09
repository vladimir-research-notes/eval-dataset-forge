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

## What will be reported here

- Per-fixture counts of exact/near duplicates, leakage flags, label warnings, PII
  flags.
- Difficulty distributions per fixture and per split.
- At least one built release (`datasets/releases/…`) with manifest + hashes.
- At least one generated dataset card.
- A bootstrap / ranking-stability demo on synthetic fixture system scores,
  including any small-N instability warnings.

## Reproduce

```bash
PYTHONPATH=src python3 -m eval_dataset_forge smoke --out results/reports/smoke_audit.md
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full command set.
