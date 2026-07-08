# Human review: Milestone 4 — Deduplication

TL;DR
The second audit stage landed: a `dedup/` package with a three-level ladder —
sha256 exact matching on `(input, expected_output)`, normalized-text matching
(prompt only), and heuristic near-duplicate detection over word n-grams (Jaccard,
or TF-IDF cosine when scikit-learn is present, with a recorded fallback). The
stages are disjoint by construction and nothing is auto-dropped except from exact
clusters. Thresholds are **calibrated against the fixtures' planted ground truth**
rather than asserted: TF-IDF recovers 4 of 5 planted paraphrases, Jaccard 3 of 5,
exact 4 of 4 with zero false positives. Full suite: 267 passed; fixtures validate;
ruff clean.

What changed
- Added `src/eval_dataset_forge/dedup/{__init__,exact,fuzzy,near_duplicate}.py`.
- Added `tests/test_dedup.py` (99 tests), including regression tests that pin the
  two documented failure modes.
- Rewrote `docs/DEDUPLICATION.md` around the measured calibration sweep; added the
  first real results section to `docs/RESULTS.md`.
- Rewrote `site/deduplication.html`; status updates on `audit-pipeline.html`,
  `milestones.html`, `index.html`, `limitations.html`.
- No CLI change: dedup surfaces through `evalforge audit` (Milestone 12), keeping
  the `PLAN.md` §8 CLI contract stable.

**Deviations from the design doc**, both driven by measurement, both now documented:
- `docs/DEDUPLICATION.md` previously specified word-**trigram** Jaccard at
  threshold **0.8**. That configuration detects **none** of the five planted
  paraphrase pairs (`qa-003`/`qa-004` measures 0.30 under it). Implemented default
  is `n=1` at `0.30`.
- The doc claimed a MinHash/blocking candidate-generation step. It is *not*
  implemented; comparison is brute-force O(n²). The doc now says so rather than
  claiming a feature that does not exist.

What should be checked later
- **The ground-truth question (most important).** `ret-001`/`ret-002` is planted as
  a near-duplicate but is provably undetectable by any lexical threshold: the
  non-duplicate pair `ret-006`/`ret-009` scores *higher* (0.504 vs 0.286). The
  `defects.yaml` schema has an `auto_detectable: false` flag for exactly this, and
  `RESULTS.md` excludes such defects from recall. **I did not add the flag** —
  editing ground truth to improve one's own recall number deserves a second pair of
  eyes. Please decide whether it should be marked, or whether the honest 4/5 stands.
- **Threshold tuning is in-sample.** 0.30 is the knee of a sweep over the same 56
  synthetic examples the recall table reports on. Sanity-check on any real corpus
  before trusting it. Jaccard at 0.30 is permissive for short, stopword-heavy
  prompts (no stopword list, no stemming).
- **The answer gate default.** `require_same_answer=True` suppresses two
  high-scoring non-duplicates, but it also suppresses any genuine paraphrase pair
  whose gold answers merely differ in surface form beyond `normalize_text` — and,
  by construction, retrieval data keyed on doc ids. Is on-by-default right, or
  should the release builder flip it?
- **`agent-003`/`agent-004` is scored as a false positive** against `defects.yaml`,
  but it is arguably a true near-duplicate (identical reference solution; the
  prompts differ only by an appended hint). The fixture files it under
  `leakage: answer_in_prompt`. Worth deciding whether ground truth should record it
  in both categories — Milestone 5 will touch this pair anyway.
- **Single-linkage transitivity.** `qa-012` joins the `qa-003`/`qa-004` cluster via
  a chain. The reported `similarity` is the weakest edge, which is the conservative
  choice, but confirm a 3-member cluster is the output Milestone 7's adversarial
  split wants (vs. a set of pairs).
- **`DedupReport` shape.** It is an ephemeral dataclass, not a versioned artifact;
  `to_dict()` was written to slot into `AuditResult.exact_duplicates` /
  `near_duplicates`. Verify that is what Milestone 12 expects. Note that
  `normalized` findings have no home in `AuditResult` today — currently a third
  list, likely to be folded under `exact_duplicates` or given its own field.

Technical risks
- Low. Pure Python plus an optional scikit-learn accelerator; no network, no paid
  API, no filesystem writes, no mutation of input examples. The fallback was
  verified end to end by blocking `sklearn` at the import hook, not merely by
  monkeypatching the capability check.
- The real risk is a *false sense of completeness*: near-duplicate detection is
  lexical and misses deep paraphrases. Stated in `DEDUPLICATION.md`, on the
  Deduplication and Limitations pages, and enforced by
  `test_retrieval_paraphrase_is_unreachable_by_any_threshold` — if the detector
  ever improves, that test fails and forces the docs to be rewritten rather than
  silently drifting out of date.
- TF-IDF similarity depends on the corpus (idf), so a pair's score is not stable
  across datasets or after adding examples. Documented; Jaccard is the stable
  alternative.
- One real bug was found and fixed by the tests: `canonical_output` originally let
  strings through unencoded, so the gold answers `"4"` and `4` hashed identically
  and would have been merged as exact duplicates. Now every value is JSON-encoded.

Commands to verify
```
python3 -m pytest -q
python3 -m pytest -q tests/test_dedup.py
python3 scripts/validate_fixtures.py
ruff check .

PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup import find_duplicates, format_report
examples = loader.load_examples(loader.FIXTURES_ROOT / 'qa_toy' / 'examples.jsonl')
print(format_report(find_duplicates(examples)))
"
```

Human feedback
<!-- Human will fill this later. -->
