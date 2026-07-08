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
