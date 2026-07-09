# Milestone 12 — Smoke audit & results

Full offline audit over every fixture, written to `results/`, scored against each
fixture's planted ground truth, and summarized in [`docs/RESULTS.md`](../RESULTS.md).

## What landed

| file | role |
|---|---|
| `src/eval_dataset_forge/audit.py` | the audit pipeline over a dataset *as it stands*; the single `assemble_audit` both `evalforge audit` and the release builder call |
| `src/eval_dataset_forge/reports/audit_report.py` | `audit.md` — the human-readable rendering `docs/PLAN.md` §5D has listed since Milestone 0 |
| `src/eval_dataset_forge/scoring.py` | matches audit findings to `defects.yaml` ground truth; defines the scoring rule |
| `src/eval_dataset_forge/smoke.py` | `evalforge smoke`: every fixture + the pooled corpus, audited, scored, reported |
| `tests/test_audit.py` | 47 tests, of which the load-bearing ones test the *scoring rule*, not the detectors |

CLI: `evalforge audit` and `evalforge smoke` are implemented. The `_PENDING` stub table
is gone — every subcommand in the PLAN's §8 contract now has a real handler, and a test
asserts none is a placeholder.

## The question this milestone had to settle

Milestones 4–6 each scored themselves against `defects.yaml` by hand and wrote the
number into `RESULTS.md`. Doing that in code forced the question those tables never had
to answer: **what is a correct scoring rule for a defect detector?**

### 1. Recall is not a scoring rule

A detector that flags everything finds every planted defect. This is not a hypothetical:
`scoring.degenerate_predictions` builds it, `evalforge smoke` scores it every run, and it
returns **recall 1.000** on all five fixtures — the same recall the pipeline earns.

No matching rule repairs this. Subset matching, exact-set matching, and the pairwise rule
all award the degenerate detector full recall, *because full recall is what it has*.
Recall bounds what a detector misses and says nothing about what it invents. So every
recall this repository prints is printed beside a precision, and the control ships inside
the report rather than in a footnote.

### 2. A cluster is a conjunction of assertions

Duplicate findings are clusters, and a cluster is not one prediction. Scoring it as one
forces a bad choice:

- *contains a planted pair* → one 14-member cluster is "correct";
- *equals a planted pair* → a true 3-member cluster covering a planted pair is "wrong".

Both rules are wrong because a cluster of size *k* asserts all *k(k−1)/2* of its pairs.
Decomposed that way, recall and precision live on the same unit, a cluster's cost grows
quadratically with its size, and the degenerate detector's precision collapses (0.014 on
`dedup.exact`, against the pipeline's 1.000).

The decomposition earns its keep on real output, not only on the control. On `qa_toy` the
near stage emits `{qa-003, qa-004, qa-012}` as one cluster, and thereby asserts
`qa-004 ≈ qa-012` — a pair nothing plants. A cluster-level view files that finding as
correct and never shows the claim. Pair decomposition surfaces it and charges 0.571
precision for it.

### 3. The denominator is chosen by the people who wrote the detectors

45 defects are planted; 40 enter the recall denominator. The scorer distinguishes two
reasons for the exclusions and never conflates them:

- `not-auto-detectable` — the *fixture's* declaration that no heuristic here should find
  it (three `mislabeled` examples).
- `no-channel` — a fact about the *scorer*: this project wires no detector to that
  ground-truth kind (`paraphrased_documents`).

Collapsing them would let a missed defect be reclassified as an out-of-scope one. Both
are listed by id in the report, and the recall over *all* planted defects (0.909) is
printed beside the recall over detectable ones (1.000). An unmapped ground-truth kind
raises `ValueError` rather than being skipped: a planted defect must not leave the
denominator by typo.

### 4. Corroboration is not a false positive — and the pardon is unit-preserving

An assertion missing from its channel's ground truth may be a defect the fixture records
elsewhere. The n-gram check flags `qa-002`, planted as half of an `exact_duplicates`
pair; the retrieval check flags `ret-001`/`ret-002`, planted as a `duplicate_documents`
pair. `GroundTruth.explained_atoms` projects the ground truth into the channel's unit and
counts those as explained.

The projection preserves the unit, which is what keeps it from pardoning everything. An
*example* atom is explained if the example is named anywhere; a *pair* atom only if that
exact pair is planted. So `qa-004 + qa-012` — endpoints each planted, pairing not —
stays unexplained, which is precisely the finding worth a human's time.

Corpus-wide there are **five** unexplained assertions. Four were already documented as
disputed in Milestones 4–5. The fifth is new (see the review note).

### 5. A check that could not run is not a check that passed

`evalforge audit` describes the split it is *given* and never builds one: re-splitting
inside an audit would make the report describe a dataset that exists nowhere on disk, and
`audit` and `release` would disagree about the same file. The consequence is that on an
unsplit corpus four signals — `leakage.train_test_overlap`, `leakage.ngram_overlap`,
`leakage.cross_split_sources`, `splits.eval_duplicates` — return zero *by vacuity*.

They are named in a warning, in a recommendation, and in `audit.md`, because a report
that printed those zeros beside the substantive ones would be actively misleading. This
is the dataset card's "the check found nothing ≠ there is nothing to find" distinction,
extended to a check that was never asked a question.

## Results

**40 of 40 detectable planted defects recovered by some channel** (38 by their own),
**0 missed**, **5 unexplained assertions**, **0 negative controls violated**. The scorer
reproduces every per-stage number Milestones 4–6 computed by hand, and a test pins them.

Pooled corpus: 56 examples, **44 independent blocks** — the sample size Milestone 9's
bootstrap resamples. Audited but deliberately **not scored**: pooling lets the ladder
compare examples across fixtures, and the per-fixture ground truth says nothing about a
cross-fixture pair.

### The pooled audit found something the per-fixture audits cannot

It reports 10 duplicate clusters against the per-fixture sum of 9, and the extra one is
*not* cross-fixture: `{saf-004, saf-005, saf-007}`, entirely inside `safety_toy`, which on
its own reports no such cluster. `sim(saf-004, saf-007)` is **0.2866** alone and **0.3438**
pooled, straddling the 0.30 threshold. Neither prompt changed.

TF-IDF's IDF term is computed over the document collection. The boilerplate the two safety
prompts share is common within `safety_toy` and rare against the pooled qa/agent/bio
vocabulary — and rarer means heavier. So **a near-duplicate verdict is not a property of a
pair; it is a property of a pair and its corpus.** Deduplication is therefore not
composable, and appending unrelated examples to a benchmark can create duplicate clusters
among the examples that did not change — which, because the adversarial split builder
derives blocks from those clusters and `eval_id` folds the split in, can move an untouched
example's split and change the release's `eval_id`. That is a Milestone 10 identity claim
reached from a Milestone 4 implementation detail; it is flagged for review rather than
resolved here.

Full tables: [`docs/RESULTS.md`](../RESULTS.md).

## The release bundle is complete

`audit.md` was listed in `docs/PLAN.md` §5D from Milestone 0 and shipped in no release
until now; `REPRODUCE.md` said so rather than letting the gap pass. It is now a hashed
artifact of every release, rendered from the `AuditResult` alone — no examples, no
manifest, no filesystem — so it cannot describe a defect the `audit.json` beside it does
not. It renders before `dataset_card.md` in the builder because, unlike the card, its
bytes do not depend on a manifest.

The committed demo release was **rebuilt in place at v0.1.0** rather than re-versioned.
Milestone 10's own `compare()` calls the change a `patch` — `content_id` and `eval_id`
unchanged, "packaging only: hashes" — so a score measured on the old bundle still stands
on the new one. `test_adding_audit_md_is_a_packaging_change_not_a_data_change` pins that
reasoning rather than leaving it in this note. All ten `verify-release` checks pass and
`dataset-card --check` still regenerates the shipped card byte-for-byte.

## Refactor

`build_audit`, `_dependence`, and `_recommendations` moved out of
`datasets/release.py` into `audit.py`. There is now exactly one place an `AuditResult` is
assembled, and it is parameterized by `SplitFacts` — a narrow view a constructed
`SplitReport` and a split read off disk can both supply. The two callers differ in
`split_statistics["source"]` (`constructed` vs `observed`) and nowhere else, so a reader
can tell a `residual_overlap: 0` earned by the adversarial strategy from one a curator
handed us. The certainty vocabulary and per-kind counters moved from `dataset_card.py`
into `reports/audit_report.py`; the card imports them, so the two findings tables cannot
drift.

## Limitations

- **Every number is in-sample.** The fixtures' defects were planted by the repository
  that wrote the detectors, and the near-duplicate and answer-in-prompt thresholds were
  selected by sweeps over these same fixtures. Recall is a resubstitution estimate. There
  is no held-out corpus.
- **"Precision" is a lower bound.** `defects.yaml` records what was planted, not an
  exhaustive labelling of every pair, so an unexplained assertion may be a genuine defect
  nobody wrote down.
- **The fixtures are not a sample of anything.** They were written to exercise the checks.
- **The scorer cannot see a defect no fixture plants and no check looks for.** By
  construction, neither can the audit.
- `evalforge smoke` scores only what the stage reports carry. `relevance_issues` and
  `paraphrased_documents` are read from the ground truth and mapped explicitly; a new
  ground-truth section raises rather than being silently unscored.

## Commands

```bash
python3 -m pytest -q                                        # 1706 tests
python3 -m pytest -q tests/test_audit.py                    # 47 tests

PYTHONPATH=src python3 -m eval_dataset_forge smoke \
  --out results/reports/smoke_audit.md \
  --fail-on-missed-defect --fail-on-control-violation       # exit 0

PYTHONPATH=src python3 -m eval_dataset_forge audit \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --out results/audits/qa_toy/

PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0                 # 10/10 checks
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check         # exit 0
```

## Next

Milestone 13 — Integration contracts: documented contracts for code-agent-runtime,
BioAgentBench, LLM regression/observability, scientific-claim-verifier, and
agent-control-bench.
