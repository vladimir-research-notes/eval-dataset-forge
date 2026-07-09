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

## The difficulty score does not measure difficulty

On the committed fixtures the heuristic difficulty proxy's ranking is **not
distinguishable from a random ranking** of the hand-assigned `difficulty_tags`: Spearman
ρ = +0.097, permutation p = 0.497 over 51 labelled examples. Sweeping the whole weight
simplex does not rescue it — 94% of 2000 draws land inside |ρ| < 0.2. What the score does
measure is `task_family`, which explains 43% of its variance against 1% of the declared
label's. The highest-scoring example in the corpus is a paraphrase of FizzBuzz, declared
`easy`.

The score is therefore shipped, reported, and explicitly **not certified**:
`DifficultyReport.trustworthy` is `False`, `evalforge difficulty --fail-on-uncalibrated`
exits 1, and the heuristic bucket must not be used as a difficulty label. Difficulty
stratification in the split builder reads the *declared* tags for exactly this reason.

Two honest caveats on the null itself. It is consistent with "the heuristic is wrong"
*and* with "the tags are noisy" — the corpus has three `hard` examples and cannot separate
the two. And the corpus is too small either way: detecting an effect of the observed size
at 80% power would need roughly 824 labelled examples. See [`DIFFICULTY.md`](DIFFICULTY.md).

Relatedly, this stage inherits a defect from the label audit: an example whose gold answer
is missing scores at the floor of the `output_complexity` signal and therefore looks
*easy*. The two lowest-scoring examples in the corpus are both examples with an absent
gold answer.

## The i.i.d. bootstrap is wrong on a corpus with duplicates, and ours still is — a little

The textbook bootstrap resamples examples and assumes they are independent. A
deterministic system scores every member of an exact-duplicate cluster identically, so the
within-cluster correlation is 1 *by construction*. Measured against known ground truth
(3000 trials, nominal 0.95): the i.i.d. percentile interval covers **0.882** at a true
accuracy of 0.70; resampling the duplicate blocks instead recovers **0.937**. So the
cluster bootstrap is the default, and the pooled corpus's 56 examples are reported as worth
≈ 36 independent ones.

The correction reaches only as far as the dedup ladder can see. `ret-001`/`ret-002` is a
paraphrase pair no lexical threshold detects ([`DEDUPLICATION.md`](DEDUPLICATION.md)), so it
yields no block edge and is resampled as if independent. **The reported design effect is a
lower bound and the intervals remain slightly too narrow.** Milestone 4's recall bound is
Milestone 9's correction bound.

Two further caveats. The percentile bootstrap under-covers at extreme accuracies whatever
the resampling unit (0.910 at *p* = 0.90 on 56 i.i.d. examples), and BCa does not reliably
repair it — at *p* = 0.95 it makes coverage *worse* (0.926 → 0.830), which is why it is
offered but not the default. And `minimum_detectable_effect` / `required_n` are normal
approximations to a trinomial quantity: a scale, not a promise. See
[`STATISTICS.md`](STATISTICS.md).

## A bootstrap "win fraction" is not a p-value

The widespread rule *"A beats B if A wins ≥ 95% of the bootstrap resamples"* is a
one-sided threshold, applied in whichever direction the data happened to favour. Its
measured type-I error under a true null is **11.4%** against a nominal 5%. This tool reports
the win fraction because readers expect it and **decides** with a Holm-corrected,
cluster-aware sign-flip permutation test — which is itself conservative on paired binary
scores (0.029 at nominal 0.05), because most examples are ties.

A bootstrap CI can exclude zero while that test declines to reject. The two use different
nulls; the report warns about the disagreement rather than settling it by quoting the
friendlier number.

## Statistics quantify sampling noise only

Bootstrap CIs and ranking-stability checks describe uncertainty over the *given*
examples. They say nothing about whether those examples are representative of the
real task. No amount of resampling can recover that; the rest of the audit exists
for it.

The demo's system scores are **synthetic**, produced by a Rasch simulator so the machinery
can run offline and so the coverage tests have a true parameter to check against. The block
structure is real; the systems are not. No claim about any model is made from them.

## Small datasets

The toy fixtures are tiny by design. Some methods (stratification, ranking
stability) are inherently limited at small N; the tool reports achieved
distributions and instability warnings rather than pretending otherwise.

Concretely, on the pooled corpus the smallest system gap detectable at 80% power is **0.321
accuracy points**, the demo's observed top-2 gap is 0.036, and deleting **two examples** —
one duplicate cluster — reverses the leaderboard. `ranking_resolved` is False and
`evalforge stats --fail-on-unstable-ranking` exits 1, rather than printing an ordering the
data cannot support.

## A release manifest detects accident, not an adversary

`verify-release` recomputes every file hash and both content identities from a release's own
contents. That is an **integrity** check. It is not an authenticity check, and the gap is
structural rather than an oversight: `manifest.json` cannot contain its own digest, so anyone
able to rewrite `examples.jsonl` can rewrite the hashes that describe it. Signing would close
this and is out of scope and out of the dependency budget.

`tests/test_release.py` asserts the hole directly — a consistently rewritten manifest verifies
clean — so the limitation cannot quietly stop being true.

Everything else the manifest reports is a **lower bound**, for the same reason every count
upstream is: `block_count`, the release's duplicate clusters, and its leakage flags all inherit
the near-duplicate stage's lexical recall bound. `content_id` and `eval_id` are the only exact
quantities in the file.

## A version bump is a claim about scores, not about validity

The bump rule keys on what an evaluation split can see: anything added to, removed from, edited
in, or reassigned across `dev`/`test` forces a `major`. A change confined to `train` is a `minor`,
because it cannot move a *fixed* system's test score.

It can, however, introduce train→test leakage that changes what that score **means** for a system
trained on the release. The delta classifies; it does not re-audit. Score comparability and
benchmark validity are different properties and only the first is decidable from a diff, so every
`minor` bump emits a warning pointing back at the leakage section of `audit.json`.

Two smaller edges: the delta joins releases by `example_id`, so renumbering ids without touching
content reads as a full `removed` + `added` and forces a `major` (conservative, and noisy); and
`eval_id` is blind to a change of *metric*, because a dataset does not carry one.

## The split seed is an experimental parameter, and a large one

On the pooled 56-example corpus, re-running the split builder with a different seed leaves the
dataset byte-identical as a set of examples — `content_id` is unchanged across all 400 seeds
tested — and moves the reported test accuracy of a fixed system with a standard deviation of
**0.130**, against a top-two system gap of **0.036**. Under `random`, the test split reproduces
the corpus's own ordering of the top two systems **48.7%** of the time.

This is not a defect the tool can fix; it is what a 11–12 example evaluation split is worth. The
tool's response is to make it *visible*: `eval_id` records the arrangement, `split_spec` records
how it was produced, and a re-split is a `major` version bump rather than a silent one. See
[`RELEASES.md`](RELEASES.md) §2.

Note also that the split strategies do not estimate the same quantity, so their reported numbers
are not interchangeable even at identical content — `adversarial` deliberately evaluates on a
duplicate-free sub-population, on which the true system gap is larger.

## Non-goals

- Not a certifier of universal benchmark quality.
- Not a semantic dedup / embedding service (out of scope for the offline build).
- Not a signing / attestation system. Release manifests detect corruption, not tampering.
- Not a medical, legal, or security authority.
- Not a data-labeling platform or annotation UI (a possible future direction).
- Not a replacement for human judgement on ambiguous labels.
