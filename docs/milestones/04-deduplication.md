# Milestone 4 — Deduplication

## Summary

Land the second stage of the audit pipeline: a three-level deduplication ladder
(exact → normalized → heuristic near-duplicate) that reports duplicate clusters
with the method and measured similarity behind each. This is audit dimension #2.
Duplicates inflate apparent sample size, reward memorization, and break the
independence assumption the Milestone 9 bootstrap rests on.

The distinguishing work here was not writing the three detectors — it was
**calibrating the heuristic one against the fixtures' planted ground truth and
reporting honestly where it fails**. The design doc's placeholder default
(word-trigram Jaccard ≥ 0.8) turned out to detect *none* of the five planted
paraphrase pairs; the measured trigram similarity for `qa-003`/`qa-004` is 0.30.

## Deliverables

- `src/eval_dataset_forge/dedup/` package:
  - `exact.py` — `canonical_output` (sorted-key JSON; strings encoded too, so the
    string `"4"` and the integer `4` do not collide), `content_signature`
    (`sha256(input + NUL + canonical_output)`), `exact_duplicates`,
    `duplicate_input_clusters` (shared prompt, differing answers → label conflict,
    not a duplicate), `duplicate_sources` (provenance dependence, informational).
  - `fuzzy.py` — `normalize_text` (NFKC, case fold, punctuation → space,
    whitespace collapse), `tokenize`, `word_ngrams`, `jaccard`,
    `normalized_duplicates` (prompt normalized, answer still exact).
  - `near_duplicate.py` — `jaccard` and optional `tfidf` cosine over a shared
    word-n-gram feature space, `auto` resolution with a recorded scikit-learn
    fallback, an **answer-agreement gate**, single-linkage clustering reporting the
    weakest edge as `similarity`.
  - `__init__.py` — `find_duplicates(...) → DedupReport` orchestrator,
    `redundant_example_ids` (exact clusters only), `to_dict`, `format_report`.
- `tests/test_dedup.py` — 99 tests: primitives, exact recall scored against every
  fixture's `defects.yaml`, stage disjointness, the answer gate, the sklearn
  fallback, determinism, degenerate inputs, and regression tests pinning the two
  documented failure modes.
- Docs: `docs/DEDUPLICATION.md` rewritten around the measured calibration table;
  `docs/RESULTS.md` gains its first real numbers. Website: `deduplication.html`
  rewritten; status updates to `audit-pipeline.html`, `milestones.html`,
  `index.html`, `limitations.html`.

## Design decisions

- **Dedup identity is `(input, expected_output)`.** Not `example_id`, not `split`.
  Two records posing the same question with the same answer are duplicates
  wherever they landed — which is exactly the planted `qa-001`/`qa-002` defect.
  Repeated ids stay a *structural* error owned by the validation layer.
- **The three stages are disjoint by construction.** A normalized finding must
  merge ≥2 distinct raw contents; a near finding never covers a pair the cheaper
  stages resolved. So `counts()` adds up instead of triple-counting one duplicate.
- **Normalization touches the prompt, never the answer.** Punctuation-folding a
  code reference solution or a JSON rubric would merge materially different
  answers. A shared prompt with differing answers is a label conflict (Milestone 6).
- **An answer-agreement gate on near-duplicates.** Measured, not assumed: without
  it the two highest-scoring non-duplicate pairs in the fixtures (`qa-001`/`qa-010`
  at 0.620, `saf-004`/`saf-009` at 0.544) each outrank a real planted duplicate.
  Documented blind spot: retrieval data whose gold answer is a list of doc *ids*
  disagrees by construction — such callers pass `require_same_answer=False`.
- **Thresholds are calibrated, not asserted.** A pooled sweep puts the knee of
  both methods at 0.30. Defaults favor recall because these findings feed a review
  queue and the near-dup-aware split strategy: a missed duplicate leaks across
  splits, a false one merely pins two examples together.
- **`n=1` (bag of words), not the design doc's `n=3`.** A paraphrase destroys
  contiguous n-grams long before it destroys the vocabulary; widening to n=2 cuts
  Jaccard recall from 3 planted pairs to 1.
- **TF-IDF preferred by `auto` because it dominates.** On this corpus it never has
  fewer true positives nor more false ones than Jaccard, and it is the only method
  that recovers `bio-001`/`bio-002` (0.305 vs Jaccard's 0.182). Falls back cleanly
  when scikit-learn is absent; the finding records what actually ran.
- **Never auto-drop on a heuristic.** `redundant_example_ids` is restricted to
  exact clusters. Near clusters are surfaced for review.
- **No CLI subcommand.** Dedup surfaces through `evalforge audit` (Milestone 12),
  keeping the `PLAN.md` §8 CLI contract stable.

## A bug the tests caught

The first implementation let strings pass through `canonical_output` unencoded, so
`canonical_output("4") == canonical_output(4) == "4"`. A dataset mixing `"4"` and
`4` — precisely the `inconsistent_format` defect this project exists to surface —
would have had those examples silently merged as exact duplicates. Fixed by
JSON-encoding every value; `test_canonical_output_distinguishes_types` and
`test_differently_typed_answers_are_not_exact_duplicates` pin it.

## Results

| stage | certainty | recall vs. planted ground truth | false positives |
|-------|-----------|--------------------------------|-----------------|
| exact | exact | 4 / 4 | 0 |
| normalized | exact given normalization | n/a (none planted) | 0 |
| near (TF-IDF) | heuristic | 4 / 5 | 2 |
| near (Jaccard) | heuristic | 3 / 5 | 2 |
| duplicate sources | exact | 2 / 2 | 0 |

Both false positives are explained in `docs/RESULTS.md`. The single miss,
`ret-001`/`ret-002`, is unreachable by *any* lexical threshold — a genuine
non-duplicate in the same fixture scores strictly higher — and that inequality is
itself asserted by `test_retrieval_paraphrase_is_unreachable_by_any_threshold`, so
a future improvement forces a documentation update rather than drifting silently.

## Validation

```bash
python3 -m pytest -q                     # 267 passed (168 prior + 99 dedup)
python3 -m pytest -q tests/test_dedup.py # 99 passed
python3 scripts/validate_fixtures.py     # all 5 fixtures valid, exit 0
ruff check .                             # all checks passed

# Reproduce the ladder on a fixture:
PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup import find_duplicates, format_report
examples = loader.load_examples(loader.FIXTURES_ROOT / 'qa_toy' / 'examples.jsonl')
print(format_report(find_duplicates(examples)))
"
```

The pure-Python fallback was verified end to end by blocking `sklearn` at the
import hook, not merely by monkeypatching the capability check.

## Open question for review

`ret-001`/`ret-002` is planted as a near-duplicate but is provably undetectable by
any heuristic in this project. `defects.yaml` has an `auto_detectable: false` flag
for exactly this case, and `RESULTS.md` excludes such defects from recall. The
flag was **not** added: editing ground truth to improve one's own score is the kind
of change that deserves a second pair of eyes. Raised in the review note.

## Status

Complete. Next: Milestone 5 — Leakage checks (train/test overlap, answer-in-prompt
containment, n-gram overlap warnings, retrieval-document contamination), which
will reuse `dedup.fuzzy`'s tokenizer and n-gram primitives and `duplicate_sources`
for cross-split dependence.
