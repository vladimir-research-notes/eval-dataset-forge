# Split construction

> Status: implemented in Milestone 7 (`src/eval_dataset_forge/splits/`).

A split is a modelling decision, not a coin flip. Each strategy encodes an
assumption about how the benchmark will be used, and the split builder records a
`SplitSpec` (strategy, seed, ratios, group key, constraints) so the rationale is
reproducible and auditable.

Every strategy in this package is **the same algorithm** — deal indivisible units
into splits until each split's quota is full — and the strategies differ only in
what a unit is. That unit is a `Block`, and the choice of block is the choice of
which dependence between examples the split refuses to cut through.

| strategy | block | channel it closes | tradeoff |
|---|---|---|---|
| `random` | one example | none (the i.i.d. baseline) | leaks near-dups and grouped items |
| `stratified` | one example | distribution shift by facet | needs enough N per stratum |
| `source-grouped` | a `source_id` group | provenance dependence | coarser ratio control |
| `adversarial` | a duplicate cluster ∪ its source group | content dependence | smallest, hardest eval splits |
| `temporal` | one example, in time order | none — it orders, it does not group | needs trustworthy timestamps |

## Verification, not assertion

`build_splits()` applies its assignment and then **re-runs Milestone 5's
cross-split leakage checks against the splits it just produced**. `residual_overlap`
is therefore a measurement of the output, not a restatement of the strategy's
intent, and the comparison below is an experiment rather than a description.

Two numbers are reported, and they mean different things:

- **residual overlap** — duplicate clusters *straddling* a split boundary. The
  classic train/test contamination.
- **duplicates in eval** — examples in `dev` or `test` that belong to a duplicate
  cluster anywhere. A cluster sitting entirely inside `test` leaks nothing across
  splits but still makes two correlated examples look like two independent ones,
  which quietly narrows every confidence interval Milestone 9 computes.

## Measured: residual cross-split duplicate clusters

Mean over seeds 0–19, ratios `0.7/0.1/0.2`, TF-IDF near-duplicate stage.
`temporal` is seedless.

| fixture | random | stratified | source-grouped | **adversarial** | temporal |
|---|---|---|---|---|---|
| `qa_toy` | 1.30 | 1.30 | 1.00 | **0.00** | 1.00 |
| `agent_tasks_toy` | 1.75 | 1.75 | 1.75 | **0.00** | 0.00 |
| `bio_toy` | 1.10 | 1.10 | 1.10 | **0.00** | 0.00 |
| `retrieval_toy` | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| `safety_toy` | 0.95 | 0.95 | 0.95 | **0.00** | 0.00 |
| **all five pooled** | 5.45 | 5.25 | 4.60 | **0.00** | 1.00 |

## Measured: duplicate-cluster examples landing in `dev`/`test`

| fixture | random | stratified | source-grouped | **adversarial** | temporal |
|---|---|---|---|---|---|
| `qa_toy` | 1.70 | 1.70 | 1.10 | **0.00** | 1.00 |
| `agent_tasks_toy` | 2.40 | 2.40 | 2.40 | **0.00** | 0.00 |
| `bio_toy` | 1.40 | 1.40 | 1.40 | **0.00** | 0.00 |
| `retrieval_toy` | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| `safety_toy` | 1.20 | 1.20 | 1.20 | **0.00** | 0.00 |
| **all five pooled** | 6.65 | 6.25 | 5.55 | **0.00** | 6.00 |

The two tables disagree about `temporal` on the pooled corpus — 1.00 residual
overlap but 6.00 duplicates in eval — and the disagreement is the point. Temporal
ordering happens to keep most duplicate pairs on the same side of the cut, then
drops whole clusters into `dev`/`test` together.

## The result worth reading twice

**The grouping key a dataset hands you is not the one that closes the leak.**

Every planted duplicate in the fixtures carries a *fresh* `source_id`:
`src-fr-capital` → `src-fr-capital-dup`, `src-fizzbuzz` → `src-fizzbuzz-para`,
`src-photosynthesis` → `src-photosynthesis-para`. That is exactly how a real corpus
records a paraphrase written as a new record. So `source-grouped`, whose entire
purpose is to stop leakage through shared provenance, leaves **every one of them**
free to straddle train and test, and measures barely better than `random`
(4.60 vs 5.45 pooled).

Worse — and sharper — on the three fixtures where no `source_id` is shared at all
(`agent_tasks_toy`, `bio_toy`, `safety_toy`), `source-grouped` produces a
**bit-for-bit identical assignment to `random`** at every seed. It is not a weaker
control on those fixtures; it is *the same function*. A test asserts this.

Only blocks derived from example **content** close the channel. Provenance metadata
describes where an example came from; it does not know what the example says.

## The second result worth reading twice

**Stratifying on more facets makes the test set less representative, not more.**

Total variation distance between the test split's marginal and the corpus's, mean
over 20 seeds, pooled 56-example corpus:

| `stratify_by` | strata | TVD family ↓ | TVD difficulty ↓ | dev size | `hard` in test |
|---|---|---|---|---|---|
| (none) | 1 | 0.216 | 0.156 | 6.0 | 0.45 |
| **`task_family`** (default) | 5 | **0.036** | 0.152 | 5.0 | 0.25 |
| `difficulty` | 4 | 0.184 | **0.030** | 5.0 | 1.00 |
| `task_family,difficulty` | 17 | 0.083 | 0.077 | 3.0 | **0.00** |

Joint stratification is worse on *both* facets than the single-facet stratification
of either. Strata multiply when facets are combined while the corpus does not: five
families × four difficulty levels gives 17 non-empty strata over 56 examples, a mean
of 3.3 each. A stratum of three cannot fill a `dev` split at ratio `0.1` — its whole
quota rounds to zero — so every stratum starves the small splits at once.

It also guarantees an empty cell. The corpus's three `hard` examples (`agent-008`,
`qa-014`, `ret-005`) sit in three *different* family strata, each of size one, and a
stratum of one apportions entirely to `train`. **No `hard` example reaches the test
set at all**, so the benchmark cannot measure difficulty. Two warnings surface this:
one when the average stratum cannot fill the smallest split, one for each facet value
missing from `test`.

The default is therefore `stratify_by=("task_family",)`. Stratify on both only when
N per joint cell is large enough to fill every split.

## The label-audit hand-off

Milestone 6 produces `blocking_example_ids`: examples whose gold answer no automated
check can resolve. Scoring a system against a label that needs adjudication measures
the label, not the system.

On the pooled corpus at seed 0, an adversarial split **without** quarantine scores on
six of them (`agent-009`, `bio-007`, `qa-010`, `ret-004`, `ret-005`, `saf-008`).
Passing `quarantine_ids=audit_labels(examples).blocking_example_ids` moves all fourteen
to `train`. The cost is real and reported rather than hidden: `test` shrinks from 12 to
8 examples and its achieved ratio falls from 0.214 to 0.143, at which point the builder
warns that `test` can no longer measure the `unlabelled` difficulty bucket.

Nothing is deleted. Quarantined and duplicate examples move to `train`; removing an
example from a dataset is the release builder's decision (Milestone 10), recorded in a
manifest a reader can audit.

## Apportionment

Quotas become **integers before any block moves**, by the largest-remainder rule:
floor each quota, then hand the leftovers to the largest fractional parts.

Rounding afterwards — greedily chasing a float deficit — starves the small splits,
because a stratum of three examples always shows `dev` a deficit of `0.3` that no
assignment can ever close, and the greedy keeps trying: at n=3 and `0.7/0.1/0.2` it
lands on `1/1/1`. Apportioning first gives `2/0/1`, and the unfillable `dev` quota
becomes a warning instead of a silent theft from `train`.

With singleton blocks the deal reproduces the apportioned targets exactly (asserted by
tests over every fixture × 20 seeds). With larger blocks it cannot — three examples do
not divide `0.7/0.1/0.2` — so blocks are dealt largest-first, leaving the singletons to
absorb the rounding error.

## What the adversarial strategy guarantees

After an adversarial build, `cross_split_clusters()` returns nothing and no `dev`/`test`
example belongs to a duplicate cluster. Both hold **by construction**: the blocks *are*
the clusters.

So the guarantee is exactly as good as the clusters, and the clusters are only as good
as the near-duplicate stage's recall.

## Known limitations

- **The guarantee is inherited from a heuristic.** `ret-001`/`ret-002` is a planted
  cross-split paraphrase that no lexical threshold can reach ([`DEDUPLICATION.md`](DEDUPLICATION.md)
  measures a genuine non-duplicate scoring strictly higher). The adversarial split
  cannot see it either, straddles it exactly as `random` would, and the verifier
  reports a clean zero. A test pins this so the heuristic guarantee cannot be quietly
  upgraded to a promise.
- **`retrieval_toy`'s clean row is vacuous.** Its gold answers are document-id lists
  that differ, so the near stage's answer gate rejects every pair before a threshold is
  consulted. Zero residual overlap for *every* strategy on that fixture measures the
  gate, not the data. Pass `near_require_same_answer=False`
  (`--ignore-answer-gate`) for retrieval-style datasets.
- **`stratified` is a no-op on every committed fixture.** Each holds exactly one
  `task_family`, so the default stratification has one stratum. It only bites on the
  pooled corpus. Asserted by a test, so the strategy is never credited with work it did
  not do.
- **`eval_splits=("test",)` changes nothing on this corpus.** Duplicate blocks are dealt
  first and `train`'s target dwarfs `dev`'s, so they never reach `dev` even when
  allowed to. The knob is exercised by a synthetic corpus in which duplicates exhaust
  the train quota.
- **Stratification is best-effort under small N.** Exact proportions are frequently
  unreachable; the builder reports the achieved distribution and warns rather than
  silently forcing it.
- **A block can span strata.** `qa-003` (`easy`) and `qa-012` (`medium`) sit in one
  adversarial block, so it belongs to no single stratum. `block_stratum()` resolves by
  plurality; the alternative — cutting the block — would reintroduce the leakage the
  block exists to prevent.
- **Temporal splits require trustworthy `created_at` values.** Unparseable timestamps
  are treated as absent (`"2024-1-3"` sorts after `"2024-01-30"`), and absent ones
  forward-fill from the previous example, keeping their position; every such example is
  named in a warning.
- **Temporal splitting leaks less than random on these fixtures by accident.**
  Paraphrases were authored on adjacent days. That is a fact about how the fixtures were
  written, not a property of temporal splitting. On the pooled corpus it is actively
  harmful: the fixtures were authored family by family, so time-ordering them nearly
  sorts them by `task_family`, and the resulting test split contains only `safety` and
  one `bio` example.

## Reproduce

```bash
python3 -m pytest -q tests/test_splits.py

# A leakage-free, label-clean evaluation split:
PYTHONPATH=src python3 -m eval_dataset_forge build-splits \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --strategy adversarial --quarantine-blocking --fail-on-leakage \
  --out results/processed/qa_toy_adversarial

# The same dataset split randomly still leaks, and the gate catches it (exit 1):
PYTHONPATH=src python3 -m eval_dataset_forge build-splits \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --strategy random --fail-on-leakage \
  --out results/processed/qa_toy_random
```
