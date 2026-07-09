# Dataset card — qa_toy v0.1.0

Generated from this release's `manifest.json` and `audit.json`. Every count below is read from the audit that shipped with the data, never recomputed, so the card and the evidence beside it cannot disagree. Prose sections are supplied by the curator and are the only unverified claims here.

| field | value |
|---|---|
| dataset | `qa_toy` |
| version | `0.1.0` |
| size | 14 examples in 11 independent blocks |
| splits | dev 1 · test 2 · train 11 |
| split strategy | `adversarial` (seed 0) |
| `content_id` | `sha256:a271768e2254cd6992cdc9a34dadce4c310cf784b4d53997547aaf536010af2d` |
| `eval_id` | `sha256:a1fb253b7597a336e1e7ed3757db40c6ab9083c6672662a1fa6a8d528958808a` |
| schema | `0.1.0` |
| built by | `eval-dataset-forge 0.0.0` |
| license | Synthetic fixture data, CC0-1.0. No real personal, clinical, or patient data. |

> This release holds 14 examples but only 11 independent blocks: 3 example(s) are duplicates or share a source record with another. Sample size for an interval or a significance test is the block count, not the example count. Resample blocks — `evalforge stats --resample-unit block`. No block straddles a split boundary.

## Purpose

Exercise and regression-test the eval-dataset-forge audit pipeline. Every defect this corpus contains was planted deliberately and is recorded as ground truth in `defects.yaml`, so each audit stage can be scored for recall against a known answer rather than reported as a bare count. The dataset exists to make the checks falsifiable; it does not exist to measure a system.

## Construction

Fourteen hand-authored English question-answer pairs, synthetic and public-safe, with nine planted defect classes: an exact duplicate, a paraphrase near-duplicate, a duplicated source record, a mislabeled answer, two ambiguous questions, a missing gold answer, conflicting annotator labels, train/test overlap, and an answer leaked into its own prompt. Splits are built by the adversarial strategy, which keeps duplicate clusters out of `dev`/`test`; examples whose gold answer is unsettled are quarantined in `train` rather than deleted. No example derives from a real person, patient, or private record.

| field | value |
|---|---|
| source files | `datasets/fixtures/qa_toy/examples.jsonl` |
| examples excluded from source | 0 |
| `source_content_id` | `sha256:a271768e2254cd6992cdc9a34dadce4c310cf784b4d53997547aaf536010af2d` |
| split strategy | `adversarial` |
| split seed | 0 |
| requested ratios | train=0.7, dev=0.1, test=0.2 |
| near-duplicate method | `tfidf` |

## Schema

Each row is a `DatasetExample` under schema `0.1.0`, with fields `example_id`, `source_id`, `task_family`, `input`, `expected_output`, `metadata`, `labels`, `difficulty_tags`, `split`, `provenance`, `created_at`, `version`. Unknown fields are rejected at load time. Task families present: `qa` (14). The formal definition is `docs/DATASET_SCHEMA.md`.

## Splits

| split | examples | independent blocks | share |
|---|---|---|---|
| `dev` | 1 | 1 | 7.1% |
| `test` | 2 | 2 | 14.3% |
| `train` | 11 | 8 | 78.6% |

*A block is a set of examples that must share a split: a duplicate cluster, or a source group. It is the unit a bootstrap should resample.*

## Audit findings

| dimension | count | certainty |
|---|---|---|
| duplicates: exact | 1 | exact |
| duplicates: normalized | 0 | exact after normalization |
| duplicates: near | 1 | heuristic (tfidf) |
| leakage: train/test overlap | 0 | inherits the dedup ladder (exact or heuristic) |
| leakage: answer in prompt | 1 | heuristic |
| leakage: n-gram overlap | 0 | exact given the n-gram width |
| leakage: retrieval contamination | 0 | exact |
| labels: missing gold answer | 1 | exact |
| labels: conflicting annotations | 1 | exact |
| labels: ambiguous | 2 | declaration, plus a heuristic |
| labels: inconsistent format | 0 | exact |
| labels: needs adjudication | 4 | derived from the rows above |
| PII-like patterns | 0 | heuristic (pattern match) |

*Every heuristic count is a lower bound. A zero on a heuristic row means the check found nothing, not that there is nothing to find.*

## Known issues

- 1 exact duplicate cluster(s) — identical input and gold answer: `qa-001 + qa-002`.
- 1 near-duplicate cluster(s) above the tfidf threshold: `qa-003 + qa-004 + qa-012`. Heuristic — a paraphrase below the threshold is not here.
- 1 answer-in-prompt finding(s) covering 1 example(s): `qa-007`.
- 1 example(s) with a missing gold answer: `qa-010`.
- 1 example(s) with conflicting annotator labels: `qa-011`.
- 2 example(s) with an ambiguous expected output: `qa-008`, `qa-009`.
- 4 example(s) with a label that needs adjudication: `qa-008`, `qa-009`, `qa-010`, `qa-011`.
- 4 example(s) are quarantined in `train` because their gold answer is missing, conflicting, or unadjudicated. They are not deleted.

## Recommendations

- 1 example(s) contain their gold answer in the prompt; they measure copying, not the task.
- 1 exact/normalized duplicate cluster(s) remain in the dataset; `evalforge release --drop exact,normalized` removes the redundant members and records each removal in manifest.excluded.
- 4 example(s) still lack a settled gold answer. They are quarantined in train, not deleted, so an adjudication pass can restore them to the evaluation splits without rebuilding the corpus.
- the difficulty heuristic's ranking is not distinguishable from a random ranking of the declared tags on this corpus; do not read its buckets as difficulty labels (docs/DIFFICULTY.md).
- 14 examples form 11 independent block(s); a bootstrap that resamples rows will under-cover. Use `evalforge stats --resample-unit block` (the default).

## Difficulty distribution

| bucket | examples | share |
|---|---|---|
| `easy` | 4 | 28.6% |
| `medium` | 5 | 35.7% |
| `hard` | 5 | 35.7% |

> Buckets come from a transparent, model-free heuristic over input length, reasoning steps, evidence count, and ambiguity tags — never from measured model performance. Whether the heuristic's ranking is distinguishable from chance on this corpus is stated under Limitations and Recommendations above; see `docs/DIFFICULTY.md`.

## Limitations

- Near-duplicate detection is lexical (tfidf). A paraphrase below the threshold is not detected, so every duplicate, leakage, and block count here is a lower bound.
- manifest.json is not self-hashed. verify-release detects accidental corruption of the data, not deliberate rewriting of the manifest that describes it.
- No effective sample size is quoted. It depends on an intraclass correlation, which depends on the systems being scored; manifest.dependence carries the block structure needed to compute it.
- The difficulty buckets are an uncalibrated heuristic on this corpus: their ranking is not distinguishable from a random ranking of the declared tags.

## Recommended use

Testing dataset-audit tooling: as a fixture for this repository's test suite, as a worked example of the release format, and as a demonstration that the audit stages rediscover defects a curator planted on purpose. Read the audit findings before reading the data.

## Non-goals

- Not a measure of question-answering capability. Fourteen examples with three in the evaluation splits cannot distinguish two systems; the confidence interval on any score computed here spans most of the unit interval.

- Not a certification that the corpus is clean. It is deliberately dirty, and the audit reports what its heuristics detect — a lower bound, never a guarantee.

- Not a benchmark. It has no leaderboard, no held-out set worth holding out, and no claim to represent any task distribution.

- Not medical, legal, or security guidance, and not a source of factual knowledge.


## Where these numbers came from

Generated facts are read from `manifest.json` and `audit.json`; prose is the curator's, recorded in `manifest.card_meta`. 4 of 4 curator sections are supplied.

- This card prints no file hashes. It is itself hashed into `manifest.hashes`, so a digest of its own bytes cannot appear inside them. Run `evalforge verify-release --release <dir>` to check every artifact against the manifest, this card included.
- This card prints no build timestamp. A clock inside a hashed artifact would make each rebuild differ from the last, and a rebuild that changes nothing must be able to say so.
- `content_id` answers whether two releases hold the same examples; `eval_id` answers whether two scores measured on them are comparable. Re-splitting a corpus preserves the first and changes the second.

> Regenerate this card, or check it against the release it shipped in, with `evalforge dataset-card --release <dir> --check`.
