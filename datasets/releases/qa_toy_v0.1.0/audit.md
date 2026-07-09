# Audit report — qa_toy @ 0.1.0

A rendering of this dataset's `audit.json`. Every number below is read from that file, never recomputed, so the report and the machine-readable evidence beside it cannot disagree. Nothing here is corrected, deleted, or scored — an audit produces evidence for a reviewer.

| field | value |
|---|---|
| dataset | `qa_toy` |
| version | `0.1.0` |
| `audit_id` | `audit-a271768e2254cd69` |
| examples | 14 |
| independent blocks | 11 |
| splits | train 11 · dev 1 · test 2 |
| split source | `constructed` |
| split strategy | `adversarial` (seed 0) |
| near-duplicate method | `tfidf` |

## Findings

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

## Duplicate clusters

| kind | examples | similarity | method |
|---|---|---|---|
| `exact` | `qa-001` + `qa-002` | — | `sha256(input+expected_output)` |
| `near` | `qa-003` + `qa-004` + `qa-012` | 0.343 | `tfidf(word_1gram)@0.3` |

*2 duplicate cluster(s); showing 2. An `exact` or `normalized` cluster is identical by construction and carries no similarity.*

## Leakage

| kind | examples | score | evidence |
|---|---|---|---|
| `answer_in_prompt` | `qa-007` | 1.000 | expected_output is 100% contained in the prompt at n=1; longest shared token span: 'berlin' |

*1 leakage finding(s); showing 1. The four channels are not disjoint: one defect can and should light up several, and agreement between them is corroboration rather than double counting.*

## Label warnings

| kind | example | detail |
|---|---|---|
| `missing` | `qa-010` | expected_output is an empty string — no gold answer to score against |
| `conflicting` | `qa-011` | annotators disagree (a1='1945', a2='1944'); no majority among 2 |
| `ambiguous` | `qa-008` | declared ambiguous by difficulty_tags \['ambiguous']; prompt is 4 tokens with no wh-word, so its referents are unbound: 'Name a primary color.'; prompt asks for one instance of a category ('Name a') while the gold answer records a single response |
| `ambiguous` | `qa-009` | declared ambiguous by difficulty_tags \['ambiguous']; prompt is 3 tokens with no wh-word, so its referents are unbound: 'Is it good?' |
| `needs_adjudication` | `qa-010` | 1 issue(s) (missing) with no resolution the audit can propose |
| `needs_adjudication` | `qa-011` | 1 issue(s) (conflicting) with no resolution the audit can propose |
| `needs_adjudication` | `qa-008` | 1 issue(s) (ambiguous) with no resolution the audit can propose |
| `needs_adjudication` | `qa-009` | 1 issue(s) (ambiguous) with no resolution the audit can propose |

*8 label finding(s); showing 8. Nothing is corrected. `needs_adjudication` is derived from the rows above it and names no example they do not.*

## PII-like patterns

The pattern scan flagged nothing.

## Splits and dependence

| split | examples | independent blocks touched |
|---|---|---|
| `train` | 11 | 8 |
| `dev` | 1 | 1 |
| `test` | 2 | 2 |

*A block is a set of examples that must share a split: a duplicate cluster, or a source group. A block straddling a boundary is counted in every split it reaches, so these do not sum to the corpus block count.*

> This dataset holds 14 examples but only 11 independent blocks: 3 example(s) are duplicates or share a source record with another. The sample size of an interval or a significance test is the block count, not the example count. Resample blocks — `evalforge stats --resample-unit block`. No block straddles a split boundary. No effective sample size is quoted here: that needs an intraclass correlation, which needs system scores, which a dataset does not have.

## Difficulty distribution

| bucket | examples | share |
|---|---|---|
| `easy` | 4 | 28.6% |
| `medium` | 5 | 35.7% |
| `hard` | 5 | 35.7% |

## Recommendations

- 1 example(s) contain their gold answer in the prompt; they measure copying, not the task.
- 1 exact/normalized duplicate cluster(s) remain in the dataset; `evalforge release --drop exact,normalized` removes the redundant members and records each removal in manifest.excluded.
- 4 example(s) still lack a settled gold answer. They are quarantined in train, not deleted, so an adjudication pass can restore them to the evaluation splits without rebuilding the corpus.
- the difficulty heuristic's ranking is not distinguishable from a random ranking of the declared tags on this corpus; do not read its buckets as difficulty labels (docs/DIFFICULTY.md).
- 14 examples form 11 independent block(s); a bootstrap that resamples rows will under-cover. Use `evalforge stats --resample-unit block` (the default).

## What this audit cannot tell you

- Near-duplicate detection is lexical (`tfidf`). A paraphrase below the threshold is not detected, so every duplicate, leakage, and block count above is a lower bound.
- Contamination is only measurable against text this tool was handed. A model's pretraining corpus is not ours to read, so a clean leakage report says nothing about it. Absence of a flag is not evidence of no contamination.
- The label audit sees structure, never truth. A gold answer that is well-formed, unanimous, unambiguous, and correctly shaped is invisible to every check here, however wrong it is. Detecting that needs a second annotator or a model.
- The difficulty buckets come from a transparent, model-free heuristic over surface signals, never from measured model performance. Whether that heuristic's ranking is distinguishable from chance on this corpus is a separate question — `evalforge difficulty` answers it, and the recommendations above report the verdict.
- PII detection is a pattern match. It finds the shapes it knows and no others.

> Regenerate this report from the evidence with `evalforge audit --dataset <path> --out <dir>`. It is a view of `audit.json`; to check a release's bytes against its manifest use `evalforge verify-release` instead.
