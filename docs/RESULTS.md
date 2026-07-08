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
