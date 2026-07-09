# Reproducing qa_toy v0.1.0

Built by eval-dataset-forge 0.0.0 against schema 0.1.0.
CPU-only and offline. Deterministic given the same inputs, the same seed, **and the same
build environment** — see below.

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

Every artifact above is byte-identical on a rebuild in the environment recorded below.
`manifest.json` additionally records `created_at`; add `--created-at <that value>` to
reproduce it too. The timestamp is provenance, not identity — it enters no hash and neither
`content_id` nor `eval_id`.

## Environment

| input | value |
|---|---|
| near-duplicate method | `tfidf` (scikit-learn installed) |
| tool version | `0.0.0` |
| schema version | `0.1.0` |

**The command is not sufficient.** `near` resolves to `tfidf` when scikit-learn is
importable and to `jaccard` otherwise. The two methods disagree about which examples are
duplicates; the adversarial split refuses to cut through a duplicate block; and `eval_id`
folds the split in. So rebuilding this release without the "optional accelerator" can
produce different splits and a different `eval_id` from the same bytes and the same seed.
Reproduce with the method above, or accept that you have built a different benchmark.

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
both exist. Rebuilding with a different `near_method` does the same.

## Verify without rebuilding

```bash
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0
```

This recomputes every file hash and both identities from the release's own contents. It
runs no detector, so — unlike the rebuild above — its answer does not depend on which
optional packages are installed. Verification is environment-independent; reproduction is
not.

It is an **integrity** check, not an authenticity one: `manifest.json` cannot hash itself,
so anyone able to rewrite the data can rewrite the digests that describe it.

## Excluded from the source corpus

None. Every source example is in this release.

## Derived artifacts

`audit.md` is a rendering of `audit.json`, and `dataset_card.md` a rendering of `manifest.json`
+ `audit.json`. Neither recomputes anything, so neither can disagree with the evidence it reads.
Check that this tool version still reproduces the card:

```bash
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check
```

## Source files

- `datasets/fixtures/qa_toy/examples.jsonl`
