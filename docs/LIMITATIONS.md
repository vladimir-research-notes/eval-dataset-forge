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
- Ambiguity detection is lexical: it reaches under-specified prompts and misses
  ambiguity that is semantic (`saf-007`) or *over*-specified (`ret-005`). Its
  wh-word guard also misses short wh-questions ("What is it?").
- Label-conflict detection compares values lexically. Two annotators who write
  "a drug" and "pharmaceutical" are recorded as disagreeing.
- Ambiguity and label-conflict detection are advisory; a human adjudicates.

Each heuristic documents its failure modes in its own doc, and each documented
failure is asserted by a test, so a future improvement cannot silently invalidate
the documentation.

## Label truth is invisible to us

The label audit sees **structure**, never truth. A mislabeled gold answer that is
well-formed, unanimous, unambiguous, and correctly shaped is invisible to every
check in the stage — and it is the label defect that does the most damage to a
benchmark. Three such defects are planted in the fixtures (`qa-006`, `bio-009`,
`saf-006`), marked `auto_detectable: false`, and asserted by a regression test to
stay undetected. Catching them needs a second annotator or a model. See
[`LABEL_AUDIT.md`](LABEL_AUDIT.md).

Relatedly, the audit never rewrites a label. Where it can propose a resolution
(a strict annotator majority; a format normalization) the proposal is a string in
`adjudication_queue.jsonl`, never an edit.

## Contamination we cannot see

n-gram/contamination checks only compare against text we are **given** (e.g. the
train split or a supplied reference). We cannot detect contamination against
private pretraining corpora. Absence of a flag is not proof of no contamination.

## A split guarantee is only as good as the clusters it is built from

The adversarial strategy drives residual cross-split overlap to zero **by
construction**: its indivisible blocks *are* the deduplication ladder's clusters, so no
cluster can straddle a boundary. That makes the guarantee exactly as strong as the near
stage's recall, and no stronger. `ret-001`/`ret-002` is a planted cross-split paraphrase
that no lexical threshold can reach ([`DEDUPLICATION.md`](DEDUPLICATION.md)); the
adversarial split straddles it exactly as a random split would, and the post-build
verifier — which reuses the same ladder — reports a clean zero. A test pins the miss so
the heuristic guarantee cannot be quietly upgraded to a promise. See
[`SPLITS.md`](SPLITS.md).

Relatedly, a strategy that closes the channel its *metadata* describes may close nothing
at all. Every planted duplicate in the fixtures carries a fresh `source_id`, so the
source-grouped strategy leaves all of them free to straddle — and on three of the five
fixtures produces a bit-for-bit identical assignment to `random`.

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
