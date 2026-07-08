# Milestone 5 — Leakage checks

## Summary

Land the third stage of the audit pipeline: four leakage/contamination checks —
cross-split overlap, answer-in-prompt containment, reference n-gram overlap, and
retrieval-document contamination — each emitting exactly one `LeakageKind`. This is
audit dimension #4: *can a system score well here without doing the task?*

The distinguishing work was not writing four detectors. It was (a) **calibrating the
one heuristic detector against planted ground truth** instead of asserting a
threshold, (b) discovering that **the four channels corroborate each other**, so a
defect the deduplication ladder is provably blind to is still recovered, and (c)
reporting two honest negatives — a redundant channel and a finding that contradicts
the fixture ground truth — rather than burying them.

## Deliverables

- `src/eval_dataset_forge/leakage/` package:
  - `ngram_overlap.py` — the shared primitive. `containment()` (asymmetric: what
    fraction of the *needle*'s n-grams are in the haystack) with a width clamp for
    short needles; `longest_shared_run()` for evidence; `ReferenceIndex` (a corpus
    shingled lazily and memoized per width); `reference_overlap()` /
    `split_reference_overlap()` producing `NGRAM_OVERLAP` findings.
  - `answer_in_prompt.py` — `answer_candidates()` (walks a rubric dict / doc-id list
    into `(field_path, text)` leaves), `label_vocabulary()` (the classification-dataset
    guard), `leaked_answers()` producing `ANSWER_IN_PROMPT` findings.
  - `train_test_overlap.py` — `cross_split_clusters()` filters the dedup ladder's
    clusters to those spanning ≥2 assigned splits; `cross_split_sources()` reports
    source-level dependence on its own channel; `split_counts()`.
  - `retrieval_overlap.py` — `ARTIFACT_MARKER` (curated annotation-marker regex),
    `document_clusters()`, `cross_split_documents()`, `retrieval_contamination()`.
  - `__init__.py` — `find_leakage(...) → LeakageReport` orchestrator, `to_dict`,
    `format_report`.
- `tests/test_leakage.py` — 111 tests: primitives, both calibration sweeps, the label
  guard demonstrated on a synthetic classification set, recall against every fixture's
  `defects.yaml`, a corpus-wide false-positive check on the artifact marker, and
  regression tests pinning each documented failure mode.
- Docs: `docs/LEAKAGE.md` rewritten around the measured calibration tables;
  `docs/RESULTS.md` gains its leakage section; `docs/LIMITATIONS.md` updated;
  `docs/PLAN.md` §4 package layout corrected. Website: `leakage.html` rewritten;
  status updates to `audit-pipeline.html`, `milestones.html`, `index.html`,
  `results.html`, `limitations.html`.

## Design decisions

- **One module, one `LeakageKind`.** Enforced by
  `test_each_channel_emits_only_its_own_kind`, and a second test asserts no kind is
  declared in the schema without a check that can emit it.
- **The four channels are deliberately *not* disjoint** — the opposite of the dedup
  ladder. A verbatim cross-split copy is both an exact duplicate and a full n-gram
  overlap. Agreement between independent channels is corroboration, so `counts()`
  reports per-kind counts *and* the number of distinct examples implicated.
- **Containment, not similarity.** Answer-in-prompt divides by the *answer's* n-gram
  count, so a short answer quoted inside a long prompt scores 1.0 where a symmetric
  Jaccard would score near zero. Leakage is asymmetric; duplication is not.
- **The width clamp.** A needle shorter than `n` is compared at its own width. Without
  it, an 8-gram check silently reports "no contamination" for exactly the short prompts
  most likely to be copied verbatim. Every finding records the width it fired at.
- **`n=2` for answer containment, not the design doc's unspecified width.** Measured:
  `n=1` admits 4 false positives at the default threshold (a prompt shares its answer's
  vocabulary — every well-posed question does); `n=3` loses a planted leak. `n=2` is the
  only width with 2/2 recall and zero false positives, and its threshold sits on a
  plateau: `[0.4, 0.6]` all give the identical answer.
- **`n=8` for reference overlap, not `n=5`.** At `n=5` the standard "any shared n-gram"
  rule fires on shared boilerplate ("what is the capital of", "chocolate chip cookies at
  home") — four spurious findings. At `n=8`, none.
- **A label-vocabulary guard, gated on coverage rather than repetition.** Repetition
  alone would classify `qa_toy`'s `"William Shakespeare"` (3 occurrences) as a class
  label. Requiring the candidate set to cover ≥50% of the dataset fires on `safety_toy`
  (`{comply, refuse}`, 8/10) and not on `qa_toy` (21%). Nothing is suppressed silently:
  the chosen vocabulary is on the report and printed by `format_report`.
- **Source-level dependence is not a `LeakageFinding`.** "Dependent" and "duplicated"
  call for different remedies — a source-grouped split versus a deletion — so
  `cross_split_sources` is its own channel, as `duplicate_sources` is in dedup.
- **Shared *distractors* are not contamination.** A cross-split passage cluster counts
  only when the passage is relevant for at least one example it appears in. Reusing an
  irrelevant passage is ordinary corpus construction.
- **`dev`/`test` overlap is the same defect as `train`/`test`**; `unassigned` is
  excluded, since an example not yet placed in a split cannot leak across one.
- **No CLI subcommand.** Leakage surfaces through `evalforge audit` (Milestone 12),
  keeping the `PLAN.md` §8 CLI contract stable — same call as Milestone 4.

## Deviations from the design doc

Both are documented in `docs/LEAKAGE.md` and were driven by the code, not by taste.

- **`PLAN.md` §4 listed `leakage/contamination.py`.** Replaced by
  `answer_in_prompt.py`, so that each module maps 1:1 onto a `LeakageKind` — a property
  a test can check, which "contamination" could not. `PLAN.md` updated to match the
  shipped tree rather than leaving the contract describing a file that does not exist.
- **`LEAKAGE.md` §4 proposed flagging "the gold answer string embedded in a passage".**
  That check is *wrong* for retrieval data: a relevant passage containing the answer is
  the definition of relevance, so it would fire on every correctly labelled example and
  discriminate nothing. Dropped, with the reasoning written down. The prompt-side
  analogue is real and check 2 owns it.

## The result worth reading twice

`ret-001`/`ret-002` is planted as a cross-split overlap and Milestone 4 *proved* it is
unreachable by any lexical prompt threshold: a genuine non-duplicate in the same
fixture scores strictly higher, so no threshold admits one without the other.

The passage-level retrieval check finds the pair anyway — both examples cite the
identical gold passage under different `doc_id`s. Two independent channels, one defect,
one of them blind to it. Pinned by
`test_retrieval_channel_recovers_what_the_dedup_ladder_cannot`.

| | TF-IDF | Jaccard |
|---|--------|---------|
| planted leakage defects recovered **by their own kind** | 11 / 12 | 10 / 12 |
| planted leakage defects recovered **by any channel** | **12 / 12** | 11 / 12 |

## Two honest negatives

- **The n-gram channel is redundant on these fixtures.** Its four findings are exactly
  the four identical cross-split pairs the exact dedup stage already reports; every
  planted overlap is either identical (caught twice) or a paraphrase (invisible to
  8-grams, which is the point of an 8-gram). Asserted by
  `test_ngram_overlap_adds_nothing_beyond_the_exact_overlap_channel`. The channel exists
  for *external* reference corpora, which the fixtures cannot demonstrate.
- **One finding contradicts the ground truth.** `agent-003`/`agent-004` straddles
  train/test with an identical reference solution, the prompts differing only by an
  appended hint. `defects.yaml` files that pair under `answer_in_prompt`, so dedup scores
  it a false positive — but as cross-split overlap it is arguably a true one. Recorded by
  `test_overlap_inherits_the_dedup_false_positive` rather than resolved by editing ground
  truth. This is the same pair Milestone 4's review note raised.

## Results

| check | certainty | recall | false positives |
|-------|-----------|--------|-----------------|
| cross-split overlap (TF-IDF) | inherited from dedup | 8 / 9 | 1 pair, disputed |
| cross-split overlap (Jaccard) | inherited from dedup | 7 / 9 | 1 pair |
| answer-in-prompt | heuristic | 2 / 2 | 0 |
| reference n-gram overlap | exact given n=8 | n/a (none planted) | 0 |
| retrieval contamination | exact | 1 / 1 | 0 |
| cross-split source records | exact | 2 / 2 | 0 |

The artifact-marker regex matches **exactly once across every passage and every prompt
in all five fixtures** — on the planted `[ANSWER: Mars]`. A broader pattern would have
fired on prose; that specificity is a test, not a claim.

## Validation

```bash
python3 -m pytest -q                       # 378 passed (267 prior + 111 leakage)
python3 -m pytest -q tests/test_leakage.py # 111 passed
python3 scripts/validate_fixtures.py       # all 5 fixtures valid, exit 0
ruff check .                                # all checks passed

# Reproduce the four channels on a fixture:
PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.leakage import find_leakage, format_report
examples = loader.load_examples(loader.FIXTURES_ROOT / 'qa_toy' / 'examples.jsonl')
print(format_report(find_leakage(examples)))
"
```

The pure-Python fallback was verified end to end by blocking `sklearn` at the import
hook, not merely by monkeypatching the capability check: `bio_toy` correctly drops from
2 cross-split findings to 1, every other channel is unchanged, and `sklearn` never
enters `sys.modules`.

## Open question for review

The label-vocabulary guard is the only place in this milestone where the tool decides
*not* to look at something. It is measured (it changes nothing on the fixtures, and
suppresses 8 of 8 false positives on a synthetic classification set) and it is loud (the
ignored vocabulary is printed), but its coverage threshold of 50% is a judgement call
made on five small fixtures. Raised in the review note.

## Status

Complete. Next: Milestone 6 — Label audit (missing / conflicting / ambiguous / format
checks, adjudication queue export), which will consume `dedup.duplicate_input_clusters`
for shared prompts with differing answers and the `labels.annotations` blocks the
fixtures already carry.
