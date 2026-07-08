# Limitations & non-goals

Being explicit about what this tool does *not* do is part of its value.

## Fixtures are synthetic

The committed datasets are small, hand-authored fixtures designed to contain
specific defects. They demonstrate that the audits fire; they are **not**
evidence about any real benchmark's quality. Results reported in
[`RESULTS.md`](RESULTS.md) are about the fixtures.

## Most checks are heuristic

- Near-duplicate detection (token/TF-IDF similarity) misses deep paraphrases and
  can over-flag similar-but-distinct items.
- Answer-in-prompt detection is lexical: it sees a *quoted* answer and misses an
  *entailed* one, and it cannot distinguish a leak from a legitimate restatement
  ("extract the year 2020 from: … in 2020 …").
- Cross-split overlap inherits every near-duplicate blind spot: a paraphrase the
  dedup ladder cannot see, it cannot see either.
- The retrieval artifact marker is a curated list of annotation conventions
  (`answer:`, `label:`, `ground truth:`), not a general detector. A corpus leaking
  its labels under another convention passes clean.
- Difficulty scoring is a surface-feature proxy, not ground truth.
- Ambiguity and label-conflict detection are advisory; a human adjudicates.

Each heuristic documents its failure modes in its own doc, and each documented
failure is asserted by a test, so a future improvement cannot silently invalidate
the documentation.

## Contamination we cannot see

n-gram/contamination checks only compare against text we are **given** (e.g. the
train split or a supplied reference). We cannot detect contamination against
private pretraining corpora. Absence of a flag is not proof of no contamination.

## Statistics quantify sampling noise only

Bootstrap CIs and ranking-stability checks describe uncertainty over the *given*
examples. They say nothing about whether those examples are representative of the
real task, and they are simple percentile/resampling methods without coverage
corrections we cannot justify on this data.

## Small datasets

The toy fixtures are tiny by design. Some methods (stratification, ranking
stability) are inherently limited at small N; the tool reports achieved
distributions and instability warnings rather than pretending otherwise.

## Non-goals

- Not a certifier of universal benchmark quality.
- Not a semantic dedup / embedding service (out of scope for the offline build).
- Not a medical, legal, or security authority.
- Not a data-labeling platform or annotation UI (a possible future direction).
- Not a replacement for human judgement on ambiguous labels.
