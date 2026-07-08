# Deduplication

> Status: implemented (Milestone 4). Module: `src/eval_dataset_forge/dedup/`.
> Tests: `tests/test_dedup.py`. All numbers below are measured on the committed
> fixtures and reproducible with the command in [§Reproduce](#reproduce).

Duplicates inflate apparent sample size, reward memorization, and break the
independence assumption that bootstrap confidence intervals (Milestone 9) rest
on: a benchmark holding forty copies of one item has fewer *effective* examples
than its row count claims. Detection runs as a ladder of three stages, each
strictly more tolerant — and strictly less certain — than the one before.

| stage | module | certainty | catches |
|-------|--------|-----------|---------|
| 1. Exact | `dedup/exact.py` | exact | byte-identical prompt **and** gold answer |
| 2. Normalized | `dedup/fuzzy.py` | exact *given the normalization* | prompts differing only in case / whitespace / punctuation / Unicode form |
| 3. Near | `dedup/near_duplicate.py` | **heuristic** | paraphrases, edits, reorderings |

The stages are **disjoint by construction**: a normalized finding always merges
at least two distinct raw contents, and a near finding never covers a pair the
earlier stages resolved. The three lists partition the evidence instead of
triple-counting it.

Nothing is ever deleted. `find_duplicates()` reports clusters plus the method and
similarity behind each; picking a survivor is a policy decision owned by the
split builder (Milestone 7) and release builder (Milestone 10), which need to
know *which* copy sits in the test split.

## 1. Exact duplicates

An example's dedup identity is the pair `(input, expected_output)` — not
`example_id`, not `split`, not `metadata`. Two records with different ids posing
the same question with the same gold answer are duplicates no matter which split
they landed in.

Method: `sha256(input + NUL + canonical_output(expected_output))`. The NUL
separator makes `("ab", "c")` and `("a", "bc")` hash differently.
`canonical_output` is compact JSON with sorted keys, so `{"a":1,"b":2}` and
`{"b":2,"a":1}` agree while the string `"4"` and the integer `4` do **not** —
mixing those is an inconsistent-format defect for the label audit, not a
duplicate to silently merge.

Exact, deterministic, no false positives. Repeated `example_id`s are a
*structural* error owned by the [validation layer](VALIDATION.md), which reports
them against file ordinals before any example object exists.

Two related, deliberately separate signals:

- **`duplicate_input_clusters`** — a shared prompt with *differing* answers. Not a
  duplicate to drop but a label conflict to adjudicate (Milestone 6).
- **`duplicate_sources`** — several examples derived from one `source_id`.
  Informational, not an error: the fixtures' `src-shakespeare` legitimately yields
  distinct questions about *Romeo and Juliet* and *Hamlet*. But they are
  statistically dependent, so a split that scatters them across train and test
  leaks source-level information, and a bootstrap that resamples them
  independently overstates its precision. Milestone 7's source-grouped strategy
  consumes this.

## 2. Normalized-text duplicates

`normalize_text` applies Unicode NFKC (smart quotes, full-width forms,
ligatures), case folding, punctuation → space, and whitespace-run collapsing.
`"  Who wrote “Hamlet”? "` and `"who wrote hamlet"` agree.

Normalization is applied to the **prompt only**; the gold answer must still match
exactly (canonically). Punctuation-folding a structured answer — a code reference
solution, a JSON rubric — would happily merge two materially different answers,
and a pair sharing a prompt but not an answer is a label conflict, not a
duplicate. This keeps stage 2 conservative: it only ever merges examples that are
the same question *and* the same answer.

Documented risk: normalization is lossy and can merge genuinely distinct short
items whose only difference is punctuation (`C++` vs `C`). That is why it never
touches answers.

## 3. Near-duplicates (heuristic)

Stages 1–2 ask "are these the same string?". Stage 3 asks "are these the same
*item*?", which no lexical method decides correctly. Its output is a ranked
**review queue**, never a verdict.

Two CPU-only methods share one feature space — the set of word n-grams of the
prompt — and differ only in weighting:

- **`jaccard`** — unweighted set overlap `|A ∩ B| / |A ∪ B|`. Pure Python, no
  dependencies, and *dataset-independent*: a pair's score never changes when other
  examples are added. Every n-gram counts the same, so prompts sharing a template
  ("Who wrote the play *X*?") — or merely sharing stopwords — score high on
  boilerplate alone.
- **`tfidf`** — cosine similarity of idf-weighted n-gram vectors (scikit-learn,
  optional). Down-weights n-grams common across the corpus and up-weights the
  distinctive ones, which is exactly the discrimination Jaccard lacks. The cost:
  idf is a property of the **corpus**, so a pair's score depends on what else is in
  the dataset and is *not comparable across datasets*. Falls back to `jaccard`
  when scikit-learn is absent; the resolved method is recorded on every finding.

`near_method="auto"` (the default) prefers TF-IDF because it **dominates** Jaccard
on the fixtures — never fewer true positives, never more false ones.

### The answer gate

By default a candidate pair must also agree on its gold answer
(`require_same_answer=True`). A paraphrased prompt with a *different* gold answer
is not a duplicate: it is either a mislabeling or two genuinely different
questions, and both are the label audit's business.

The gate is not cosmetic. Without it, the two highest-scoring non-duplicate pairs
in the fixtures each outrank a real planted duplicate:

| pair | similarity (TF-IDF) | gold answers | verdict |
|------|--------------------|--------------|---------|
| `qa-001` ~ `qa-010` | 0.620 | `"Paris"` vs `""` | suppressed by gate |
| `saf-004` ~ `saf-009` | 0.544 | `"refuse"` vs `""` | suppressed by gate |
| `bio-001` ~ `bio-002` *(planted)* | 0.305 | `"pancreas"` both | kept |

**Known blind spot.** For retrieval-style data whose gold answer is a list of
document *ids*, two examples can cite identical document *text* under different
ids and be judged to disagree. Such datasets should pass
`require_same_answer=False`.

### Calibration

Thresholds were selected on the 56 committed fixture examples by sweeping and
taking the knee of the precision/recall curve, with the answer gate on. Both
methods land at **0.30**; they are kept as a per-method mapping because the two
scores are *not* on a common scale — an idf-weighted cosine and a raw set overlap
only coincide here — and a different corpus will move them apart.

Threshold sweep, all five fixtures pooled, counted **pairwise** (before
single-linkage clustering) against the 5 planted near-duplicate pairs:

| threshold | Jaccard TP / FP | TF-IDF TP / FP |
|-----------|-----------------|----------------|
| 0.25 | 3 / 5 | 4 / 5 |
| **0.30** | **3 / 2** | **4 / 3** |
| 0.35 | 2 / 2 | 3 / 2 |
| 0.45 | 2 / 1 | 2 / 2 |
| 0.60 | 1 / 0 | 1 / 0 |
| 0.80 | 0 / 0 | 0 / 0 |

The defaults **favor recall**: these findings feed a review queue and the
near-duplicate-aware split strategy, where a missed duplicate silently leaks
across splits while a false one merely pins two examples into the same split.

The n-gram width is the other knob. `DEFAULT_NGRAM = 1` (bag of words), because a
paraphrase reorders and substitutes words, destroying contiguous n-grams long
before it destroys the vocabulary. Widening to n=2 cuts Jaccard recall from 3 of
the 5 planted pairs to 1.

> **These thresholds are tuned on 56 synthetic examples.** That is a small,
> hand-authored corpus, and the numbers above are in-sample. Re-calibrate on your
> own data before trusting the defaults. In particular, Jaccard at 0.30 is
> permissive for short, stopword-heavy prompts (no stopword list is applied);
> prefer TF-IDF, whose idf term becomes *more* selective as the corpus grows.

### Recall against planted ground truth

Each fixture plants exactly one paraphrase pair in its `defects.yaml`. At the
default threshold, on the prompt, with the answer gate:

| fixture | planted pair | Jaccard | TF-IDF |
|---------|--------------|---------|--------|
| `qa_toy` | `qa-003` ~ `qa-004` | ✅ 0.600 | ✅ 0.763 |
| `agent_tasks_toy` | `agent-001` ~ `agent-002` | ✅ 0.485 | ✅ 0.549 |
| `bio_toy` | `bio-001` ~ `bio-002` | ❌ 0.182 | ✅ 0.305 |
| `safety_toy` | `saf-001` ~ `saf-003` | ✅ 0.313 | ✅ 0.432 |
| `retrieval_toy` | `ret-001` ~ `ret-002` | ❌ 0.214 | ❌ 0.286 |
| | **recall** | **3 / 5** | **4 / 5** |

Exact duplicates, by contrast, are recovered at **4 / 4** with zero false
positives (`retrieval_toy` plants none).

## Known limitations

Each is asserted by a test, so a future improvement cannot silently invalidate
this section.

**`ret-001` ~ `ret-002` is unreachable by any lexical threshold.** The planted
paraphrase ("Which organelle carries out photosynthesis in plant cells?" vs
"Where does photosynthesis take place in a plant cell?") scores 0.286, while
`ret-006` ~ `ret-009` — boiling point vs freezing point of water, *not* a
duplicate — scores 0.504. The non-duplicate ranks **above** the duplicate, so no
threshold, with or without the answer gate, admits the first without the second.
Separating them requires semantics. Pinned by
`test_retrieval_paraphrase_is_unreachable_by_any_threshold`; if it ever fails,
the detector improved and this section must be rewritten.

**Template questions are a standing false positive.** `qa-012` ("Who wrote the
play Hamlet?") joins the `qa-003`/`qa-004` Romeo-and-Juliet cluster: same
template, same gold answer `"William Shakespeare"`, so neither the lexical score
nor the answer gate can exclude it. It *is* independently flagged as a duplicate
**source** record (`src-shakespeare`), which is the signal a reviewer should act
on. The two channels agreeing is the intended design.

**Single-linkage clustering is transitive.** If A~B and B~C clear the threshold,
all three surface as one cluster even when A~C does not. That is the right
direction for a review queue, but a long chain of pairwise-similar items can merge
into a cluster whose endpoints are dissimilar. Mitigation: the reported
`similarity` is the **weakest** edge in the cluster, not the strongest — an honest
lower bound on how alike the members are.

**Deep paraphrases with low lexical overlap are invisible**, and not because the
threshold is set wrong. Embedding-based semantic dedup is intentionally out of
scope for the offline, CPU-only, no-paid-API build; it is listed as a future
direction in [`PAPER_PATH.md`](PAPER_PATH.md).

**Comparison is brute-force O(n²)** over the prompt set. At fixture scale (tens of
examples) that is microseconds. The blocking / MinHash bucketing step a large
corpus would need is deliberately **deferred, not claimed** — see
[`roadmap`](../site/roadmap.html).

**No stopword list and no stemming.** `produces` and `producing` are distinct
tokens; `bake` and `baking` are distinct tokens. This is why `bio_toy` and
`safety_toy` score as low as they do, and why TF-IDF (which learns that `the` is
worthless) outperforms raw overlap.

## Reproduce

```bash
python3 -m pytest -q tests/test_dedup.py       # 99 tests

PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup import find_duplicates, format_report
examples = loader.load_examples(loader.FIXTURES_ROOT / 'qa_toy' / 'examples.jsonl')
print(format_report(find_duplicates(examples)))
"
```

Dedup findings are surfaced through `evalforge audit` (Milestone 12); there is no
standalone `dedup` subcommand, to keep the CLI contract in
[`PLAN.md`](PLAN.md) §8 stable.
