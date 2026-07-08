# Results

> Status: pending Milestone 12 (Smoke audit & results). This file is populated
> when the full audit pipeline runs over the committed fixtures. It currently
> describes what will be reported.

The smoke audit (`evalforge smoke --out results/reports/smoke_audit.md`) runs the
entire pipeline over every fixture dataset, fully offline, and this document
summarizes the outcome. Because the fixtures are hand-authored with deliberate
defects, the audit's job is to **rediscover the planted defects** — that is the
demonstration that the checks work.

## Expected findings per fixture (to be confirmed by the run)

| fixture | planted defects the audit should surface |
|---------|-------------------------------------------|
| `qa_toy` | duplicate + paraphrased questions, mislabeled answers, ambiguous questions, train/test overlap |
| `agent_tasks_toy` | near-duplicate tasks, leaked solutions, ambiguous scoring |
| `retrieval_toy` | duplicated docs, query–document leakage, incomplete relevance labels |
| `bio_toy` | entity-label conflicts, evidence-label mismatches |
| `safety_toy` | over-/under-refusal and ambiguous-boundary label conflicts |

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
