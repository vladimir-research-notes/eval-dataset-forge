# Leakage & contamination

> Status: implemented (Milestone 5). Module: `src/eval_dataset_forge/leakage/`.
> Tests: `tests/test_leakage.py`. All numbers below are measured on the committed
> fixtures and reproducible with the commands in [§Reproduce](#reproduce).

Leakage is any path by which a system scores well without demonstrating the
capability the benchmark intends to measure. Four checks look for four distinct
paths, and each emits exactly one `LeakageKind`:

| check | module | kind | certainty |
|-------|--------|------|-----------|
| 1. Cross-split overlap | `train_test_overlap.py` | `train_test_overlap` | inherited from dedup |
| 2. Answer-in-prompt | `answer_in_prompt.py` | `answer_in_prompt` | **heuristic** |
| 3. Reference n-gram overlap | `ngram_overlap.py` | `ngram_overlap` | exact *given* the width |
| 4. Retrieval contamination | `retrieval_overlap.py` | `retrieval_contamination` | exact |

Unlike the [deduplication ladder](DEDUPLICATION.md), whose three stages are
disjoint by construction, **these four channels overlap on purpose**. A test
example copied verbatim from train is a cross-split exact duplicate *and* a full
n-gram overlap against the train reference. Agreement between independent channels
is corroboration, not double counting, and the payoff is measurable: the one
planted defect the dedup ladder provably cannot see is recovered by the retrieval
channel (see [§Corroboration](#corroboration)).

Nothing is removed and nothing is a verdict. `find_leakage()` produces evidence;
the split builder (Milestone 7) decides what to do about it.

## 1. Cross-split overlap

Filters the deduplication ladder's clusters down to those whose members sit in two
or more assigned splits. It therefore inherits each stage's certainty exactly:
exact and normalized clusters yield exact findings, near-duplicate clusters yield
heuristic ones, and the reported `score` is the dedup `similarity` (`None` where
the pair is identical by construction).

`dev`/`test` overlap counts as the same defect as `train`/`test` — tuning on an
item, then scoring on it. `unassigned` is excluded: an example not yet placed in a
split cannot leak across one, and treating it as a leak would flag every raw
dataset before its splits are built.

**Source-level dependence is a separate channel.** Two examples derived from one
`source_id` are not duplicates — the fixtures' `src-shakespeare` yields distinct
questions about *Romeo and Juliet* and about *Hamlet* — but they are statistically
dependent, and scattering them across train and test leaks source-level
information. `cross_split_sources()` reports it as its own signal rather than as a
`LeakageFinding`, because "dependent" and "duplicated" call for different remedies:
the first wants a source-grouped split, the second wants a deletion. Both planted
`duplicate_sources` records straddle a split boundary (`src-shakespeare`,
`src-boiling`); the audit finds both.

### Recall against planted ground truth

Nine `train_test_overlap` defects are planted across the five fixtures.

| fixture | planted pairs | Jaccard | TF-IDF |
|---------|---------------|---------|--------|
| `qa_toy` | `qa-001`~`qa-002`, `qa-003`~`qa-004` | 2 / 2 | 2 / 2 |
| `agent_tasks_toy` | `agent-001`~`agent-002`, `agent-005`~`agent-006` | 2 / 2 | 2 / 2 |
| `bio_toy` | `bio-001`~`bio-002`, `bio-003`~`bio-006` | 1 / 2 | 2 / 2 |
| `safety_toy` | `saf-001`~`saf-003`, `saf-004`~`saf-005` | 2 / 2 | 2 / 2 |
| `retrieval_toy` | `ret-001`~`ret-002` | 0 / 1 | 0 / 1 |
| | **total** | **7 / 9** | **8 / 9** |

The misses are inherited, not new. `bio-001`~`bio-002` needs idf weighting (Jaccard
scores it 0.182); `ret-001`~`ret-002` is unreachable by *any* lexical threshold,
because a genuine non-duplicate in the same fixture scores strictly higher. Both are
analyzed in [`DEDUPLICATION.md`](DEDUPLICATION.md).

**One finding is not in the ground truth.** `agent-003`~`agent-004` straddles
train/test with an identical reference solution and prompts that differ only by an
appended hint. `defects.yaml` files that pair under `leakage: answer_in_prompt`, so
the dedup layer scores it as a false positive — but as *cross-split overlap* it is
arguably a true one. The disagreement is recorded rather than resolved
(`test_overlap_inherits_the_dedup_false_positive`) and is
[flagged for human review](review_notes/05-leakage-human-review.md).

## 2. Answer-in-prompt

The gold answer, quoted inside its own prompt. Measured as **containment** — the
fraction of the answer's word n-grams that occur in the prompt — which is
deliberately asymmetric: it asks "how much of the answer is in the prompt?", not
"how similar are they?". A one-sentence answer fully quoted inside a thousand-word
prompt scores 1.0, where a symmetric Jaccard would score near zero.

`expected_output` may be a string, a scalar, a doc-id list, or a rubric dict.
Serializing a rubric to JSON and searching for that blob would never match, so the
value is walked and each leaf string is checked with its field path recorded. This
is what finds `agent-004`: the leak is in the `reference_solution` field of a
two-key dict, not in the answer as a whole.

Answers shorter than `n` tokens are compared at their own width
(`min(n, len(answer_tokens))`), so a one-word gold answer like `"Berlin"` is found
by a unigram match while longer answers keep bigram discipline. Every finding
records the width it fired at.

### Calibration

`DEFAULT_NGRAM = 2`, `DEFAULT_THRESHOLD = 0.5`. Both were selected by sweeping the
56 committed fixture examples against the 2 planted `answer_in_prompt` defects,
counted per example:

| n-gram width | threshold 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| 1 | 2 TP / 7 FP | 2 / 7 | 2 / 4 | 2 / 2 | 2 / 1 | 2 / 1 |
| **2** | 2 / 1 | 2 / 0 | **2 / 0** | 2 / 0 | 1 / 0 | 1 / 0 |
| 3 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |

`n=2` is the only width that recovers both planted leaks without a false positive,
and the default threshold sits in the middle of a plateau rather than on a knife
edge — the whole band `[0.4, 0.6]` gives the identical answer, because the weakest
true positive scores **0.600** and the strongest non-leak **0.333**.

The rejected widths fail for opposite reasons, and both failures are pinned by
tests:

- **`n=1` cannot tell a quoted answer from a prompt that shares its vocabulary** —
  which every well-posed question does. `agent-011`'s answer `"return sorted(arr)"`
  scores 0.667 against a prompt that merely *describes* returning a sorted `arr`;
  `qa-013`'s `"Pacific Ocean"` scores 0.5 because the prompt contains the word
  *ocean*; `agent-007`'s `"Returns correct index or -1"` scores 0.8 against a prompt
  that says "returns the index of target … or -1".
- **`n=3` loses `agent-004`** (0.25). A paraphrased quotation keeps its bigrams long
  after its trigrams are gone.

| fixture | planted leak | containment | evidence |
|---------|--------------|-------------|----------|
| `qa_toy` | `qa-007` | 1.000 (n=1, clamped) | span `berlin` |
| `agent_tasks_toy` | `agent-004` | 0.600 (n=2) | `reference_solution`, span `return s 1` |
| | **recall** | **2 / 2**, 0 false positives | |

### The label-vocabulary guard

In a classification dataset the gold answer *is* a class name, and a prompt that
names its own label set ("should the assistant comply or refuse?") would be flagged
on every row. `label_vocabulary()` detects that case and excludes those answers.

A candidate must be short (≤ 2 tokens) **and** repeated (≥ 3 times) **and** — the
load-bearing condition — the set of such candidates must cover ≥ 50% of the
dataset. Repetition alone is not enough: without the coverage test, `qa_toy`'s
`"William Shakespeare"` (3 occurrences, 21% coverage) would be dismissed as a class
label. On the fixtures the guard fires on `safety_toy` only, taking
`{comply, refuse}` (8 of 10 examples), and changes no result there.

Its value is shown on data the fixtures do not contain: on a synthetic sentiment set
whose prompts read "Is review #N positive or negative?", the guard takes the
finding count from **8 of 8 examples to zero**
(`test_label_guard_suppresses_a_whole_classification_dataset`).

Nothing is suppressed silently — the chosen vocabulary is carried on the report and
printed by `format_report`. The guard's cost is a true negative: a genuinely leaked
class label in a genuine classification dataset goes unreported. Pass
`ignore_answers=frozenset()` to disable it.

### What it misses

The check is lexical. It sees a *quoted* answer and misses an *entailed* one
("Which country's capital is Berlin?" → `Germany`). And it cannot distinguish a leak
from a legitimate restatement: "extract the year 2020 from: … in 2020 …" is a
correctly posed extraction task that scores 1.0. Findings are for review, never for
removal.

## 3. Reference n-gram overlap

The standard decontamination signal: flag an evaluation prompt that shares a word
n-gram with a reference corpus. The default reference is the `train` split and the
default targets are `dev` and `test`, which makes it an in-dataset contamination
check; `reference_overlap()` accepts any external corpus instead.

`DEFAULT_NGRAM = 8` — the low end of the 8–13 band published pipelines use — and the
rule is the standard one: **any single shared 8-gram is the flag**, with the
containment fraction reported as the finding's strength. Prompts shorter than `n`
are compared at their own width, and the finding says so; without that clamp an
8-gram check would silently return "no contamination" for exactly the short items
most likely to be duplicated verbatim.

### Why n=8 and not n=5

At `n=5` the "any shared n-gram" rule fires on shared boilerplate:

| flagged at n=5 | containment | why |
|---|---|---|
| `qa-002`, `agent-006`, `bio-006`, `saf-005` | 1.000 | genuine verbatim duplicates |
| `qa-004` | 0.167 | shares *"who … the play romeo and juliet"* with `qa-003` |
| `qa-007` | 0.167 | shares *"what is the capital of"* with `qa-001` |
| `saf-003` | 0.125 | shares *"chocolate chip cookies at home"* with `saf-001` |
| `agent-004` | 0.083 | shares a code fragment with `agent-003` |

Four spurious findings, each a shared template rather than a shared item. At `n=8`
only the four verbatim duplicates survive, all at containment 1.000. Pinned by
`test_width_eight_rejects_the_boilerplate_that_width_five_admits`.

### An honest null result

**On these fixtures the n-gram channel finds nothing the exact overlap channel does
not.** Its four findings — `qa-002`, `agent-006`, `bio-006`, `saf-005` — are exactly
the four identical cross-split pairs. Every planted overlap is either identical
(caught by the exact dedup stage too) or a paraphrase (invisible to 8-grams, which
is the point of an 8-gram). Asserted by
`test_ngram_overlap_adds_nothing_beyond_the_exact_overlap_channel`.

The channel earns its place on datasets with an **external** reference corpus, where
no dedup stage can help because the reference is not part of the dataset. That is
the use it is built for, and the fixtures cannot demonstrate it.

## 4. Retrieval contamination

Keyed on the presence of `metadata.retrieval.docs`, not on `task_family ==
"retrieval"`, so a QA item shipping retrieved context is audited too. Two exact
signals:

**Annotation artifacts in passage text.** A passage carrying `[ANSWER: Mars]`,
`label:`, or `ground truth =` has had the annotation pipeline leak into the corpus;
a retriever can rank by the marker instead of by relevance. The pattern is
curated — a label word immediately followed by `:` or `=`, which is the shape of a
machine-written annotation rather than of an English sentence. Over **every passage
and every prompt in all five fixtures it matches exactly once**, on the planted
`[ANSWER: Mars]` in `ret-003`/`d5` (`test_artifact_marker_fires_exactly_once…`).

**A gold passage shared across splits.** The same passage text in a training example
and a test example — under different `doc_id`s, which is how it evades id-level
checks — means the held-out evidence was never held out. `ret-001`/`d1` and
`ret-002`/`d3` carry identical text across train and test. Distractors are *not*
flagged: reusing an irrelevant passage across examples is normal corpus
construction, so a cluster counts only when the passage is marked relevant for at
least one example it appears in.

Passage comparison is exact after `normalize_text`, so casing and punctuation cannot
hide a duplicate.

### What is deliberately not checked

An earlier draft of this document proposed flagging "the gold answer string embedded
in a passage". For retrieval data that check is **wrong**: a relevant passage
containing the answer is the *definition* of relevance, so it would fire on every
correctly labelled example and discriminate nothing. The prompt-side analogue — the
answer quoted in the query — is real, and check 2 owns it.

## Corroboration

`ret-001`~`ret-002` is planted as a cross-split overlap and is **provably
unreachable** by check 1: the queries are a deep paraphrase, and Milestone 4
established that a genuine non-duplicate in the same fixture scores strictly higher,
so no threshold admits one without the other.

The **passage-level** check recovers the same pair anyway, because both examples cite
the identical gold passage under different ids. Two independent channels, one
defect, one of them blind to it. This is the argument for keeping the channels
non-disjoint, and it is pinned by
`test_retrieval_channel_recovers_what_the_dedup_ladder_cannot`.

The consequence for recall accounting:

| | TF-IDF | Jaccard |
|---|--------|---------|
| planted leakage defects recovered **by their own kind** | 11 / 12 | 10 / 12 |
| planted leakage defects recovered **by any channel** | **12 / 12** | 11 / 12 |

## Summary of findings on the fixtures

Default settings, TF-IDF near-duplicates.

| fixture | overlap | answer-in-prompt | n-gram | retrieval | cross-split sources | examples implicated |
|---------|---------|------------------|--------|-----------|---------------------|---------------------|
| `qa_toy` | 2 | 1 | 1 | 0 | 1 | 6 / 14 |
| `agent_tasks_toy` | 3 | 1 | 1 | 0 | 0 | 6 / 12 |
| `retrieval_toy` | 0 | 0 | 0 | 2 | 1 | 3 / 10 |
| `bio_toy` | 2 | 0 | 1 | 0 | 0 | 4 / 10 |
| `safety_toy` | 2 | 0 | 1 | 0 | 0 | 4 / 10 |

`examples implicated` is smaller than the sum of the findings because the channels
overlap by design; one contaminated example is named by several.

## Known limitations

Each is asserted by a test, so a future improvement cannot silently invalidate this
section.

**Contamination against unseen corpora is undetectable.** Overlap is measured only
against text we are handed. A model's pretraining data is not ours to read, so a
clean report says nothing about it. **Absence of a flag is not evidence of no
contamination.** This is a property of the problem, not a gap in the implementation.

**Cross-split overlap inherits every dedup blind spot.** A cross-split paraphrase
invisible to the near-duplicate stage is invisible here. On `retrieval_toy` that
costs the one planted overlap check 1 misses.

**Answer-in-prompt is lexical.** It misses entailed answers and cannot distinguish a
leak from a legitimate restatement.

**The label-vocabulary guard trades a false positive for a false negative.** A
genuinely leaked class label in a genuine classification dataset is not reported. The
guard prints what it ignored.

**Paraphrased duplicate passages are not detected.** `retrieval_toy`'s `d13`/`d14`
state one boiling-point fact two ways. Passage matching is exact after
normalization, and no threshold separates those two from unrelated passages without
semantics. Pinned by `test_paraphrased_passages_are_not_detected`; the fixture's
`paraphrased_documents` ground truth stands as a known miss.

**The artifact marker is a curated list, not a detector.** It catches the annotation
conventions we thought of (`answer:`, `label:`, `ground truth:`, `relevant=`). A
corpus that leaks its labels under a different convention passes clean.

**Comparison is brute-force.** `O(|targets| × |references|)` set intersections for
the n-gram channel and `O(n²)` inside the dedup ladder. At fixture scale that is
microseconds; the blocking step a large corpus would need is deliberately deferred
rather than claimed.

## Reproduce

```bash
python3 -m pytest -q tests/test_leakage.py       # 111 tests

PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.leakage import find_leakage, format_report
examples = loader.load_examples(loader.FIXTURES_ROOT / 'qa_toy' / 'examples.jsonl')
print(format_report(find_leakage(examples)))
"
```

Leakage findings are surfaced through `evalforge audit` (Milestone 12); there is no
standalone `leakage` subcommand, to keep the CLI contract in [`PLAN.md`](PLAN.md) §8
stable.
