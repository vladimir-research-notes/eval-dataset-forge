# Reproducing qa_toy v0.1.0

Built by eval-dataset-forge 0.0.0 against schema 0.1.0.
CPU-only, offline, and deterministic given the same inputs and seed.

## Command

```bash
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset-id qa_toy \
  --strategy adversarial \
  --seed 0 \
  --ratios 0.7,0.1,0.2 \
  --stratify-by task_family \
  --quarantine-blocking \
  --license-notes 'Synthetic fixture data, CC0-1.0. No real personal, clinical, or patient data.' \
  --card-meta datasets/fixtures/qa_toy/card.yaml \
  --version 0.1.0 \
  --out datasets/releases/qa_toy_v0.1.0
```

Every artifact above is byte-identical on a rebuild. `manifest.json` additionally records
`created_at`; add `--created-at <that value>` to reproduce it too. The timestamp is
provenance, not identity — it enters no hash and neither `content_id` nor `eval_id`.

## Identity

| field | value |
|---|---|
| `content_id` | `sha256:a271768e2254cd6992cdc9a34dadce4c310cf784b4d53997547aaf536010af2d` |
| `eval_id` | `sha256:a1fb253b7597a336e1e7ed3757db40c6ab9083c6672662a1fa6a8d528958808a` |
| `source_content_id` | `sha256:a271768e2254cd6992cdc9a34dadce4c310cf784b4d53997547aaf536010af2d` |
| examples | 14 (dev=1, test=2, train=11) |

`content_id` is an order-independent digest of the examples with `split` excluded — it
answers *"are these the same examples?"*. `eval_id` folds each example's split in — it
answers *"are two scores measured on these releases comparable?"*. Rebuilding with a
different `--seed` preserves the first and changes the second, which is the entire reason
both exist.

## Verify without rebuilding

```bash
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0
```

This recomputes every file hash and both identities from the release's own contents. It is
an **integrity** check, not an authenticity one: `manifest.json` cannot hash itself, so
anyone able to rewrite the data can rewrite the digests that describe it.

## Excluded from the source corpus

None. Every source example is in this release.

## Not in this bundle yet

- `audit.md`, the human-readable rendering of `audit.json` — Milestone 12.

`dataset_card.md` is present, and is a rendering of `manifest.json` + `audit.json`. Check that
this tool version still reproduces it:

```bash
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check
```

## Source files

- `datasets/fixtures/qa_toy/examples.jsonl`
