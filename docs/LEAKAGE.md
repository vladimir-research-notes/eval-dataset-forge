# Leakage & contamination

> Status: design (implemented in Milestone 5).

Leakage is any path by which a system can score well without demonstrating the
capability the benchmark intends to measure. We check four kinds.

## 1. Train/test overlap

Exact or normalized-text duplicates that straddle two splits (e.g. the same
example in both `train` and `test`). Reuses the deduplication machinery
([`DEDUPLICATION.md`](DEDUPLICATION.md)) but restricts attention to
cross-split pairs. Exact for identical/normalized items; heuristic for
near-duplicates.

## 2. Answer-in-prompt leakage

The `expected_output` (or a close variant) appears verbatim inside `input`. A
system can then "solve" the task by copying. Method: normalized substring / token
n-gram containment of the answer within the prompt, above a threshold. Reported
with the matched span as evidence.

Caveat: legitimate tasks sometimes restate the answer (e.g. "extract the date
2020 from: … 2020 …"). Findings are flagged for review, not auto-removed.

## 3. High n-gram overlap (contamination signal)

For each test example, compute the maximum n-gram (default n=8..13) overlap
against a supplied reference corpus (e.g. the train split, or a public-safe
reference file). High overlap is a contamination signal in the style of standard
decontamination pipelines. We compute overlap **only against text we are given**
— we cannot see private pretraining corpora, and we say so explicitly.

## 4. Retrieval-document contamination

For `retrieval`-family examples, check whether a retrieved document
(`metadata.retrieval.docs`) contains a label artifact — e.g. the gold answer
string embedded in a passage, or a document that is itself the labelled answer.
This catches retrieval benchmarks where the "hard" retrieval step is trivialized
by a leaked passage.

## Reporting

Each `LeakageFinding` records its `kind`, involved `example_ids`, a numeric
`score` (overlap/containment), and `evidence` (the matched text/span). The audit
summarizes counts per kind and issues prioritized recommendations.

## Known limitations

- Contamination against unseen corpora cannot be detected — only overlap with
  provided reference text is measured.
- n-gram overlap has false positives on formulaic text (boilerplate, templates);
  thresholds are tunable and the matched span is always reported for review.
- Answer-in-prompt detection is lexical; it will miss answers that are entailed
  but not stated.
