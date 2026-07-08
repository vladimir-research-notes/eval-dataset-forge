# Deduplication

> Status: design (implemented in Milestone 4). This document states the intended
> methods and their failure modes; the module and tests refine it.

Duplicates inflate apparent sample size and reward memorization. We detect them
at three increasing levels of tolerance.

## 1. Exact duplicates (exact check)

- **Duplicate IDs** — two examples share an `example_id`. Always an error.
- **Identical input** — byte-identical `input`.
- **Identical expected output** — byte-identical `expected_output` (informational
  for classification-style data where repeats are expected; flagged, not errored).
- **Duplicate source records** — same `source_id` producing identical content.

Method: hashing (sha256) of the relevant field(s). Exact, deterministic, no false
positives.

## 2. Normalized-text duplicates (near-exact)

Two examples that differ only in formatting. Normalization: lowercase, collapse
whitespace, strip punctuation-only differences, unicode NFKC. Then hash-compare.

Catches: trailing spaces, casing, "  " vs " ", smart quotes. Misses: real
paraphrases (handled below). Documented risk: over-normalization can merge
genuinely distinct short items — normalization is conservative by default.

## 3. Near-duplicates (heuristic)

Two methods, both CPU-only:

- **Token n-gram Jaccard.** Represent each example's text as a set of word
  n-grams (default n=3); flag pairs with Jaccard similarity ≥ threshold
  (default 0.8). Pure-Python, no dependencies. Good for reordered/edited text.
- **TF-IDF cosine (optional).** If `scikit-learn` is available, build a TF-IDF
  matrix and flag pairs with cosine similarity ≥ threshold. Better for
  paraphrases with vocabulary overlap. Falls back to Jaccard when scikit-learn
  is absent.

To keep pairwise comparison tractable on CPU, candidate generation uses a cheap
blocking step (shared rare n-grams / MinHash-style bucketing) before exact
similarity is computed on candidates only. For the toy fixtures (tens to low
hundreds of examples), brute-force O(n²) is also acceptable and used as the
reference implementation.

### Thresholds and reporting

Thresholds are configurable and reported in the finding. Each
`DeduplicationFinding` records `kind`, the involved `example_ids`, the measured
`similarity`, and the `method` used, so a reviewer can judge each cluster.

### Known limitations

- Near-duplicate detection is a heuristic: it will miss deep paraphrases with
  low lexical overlap and may flag legitimately similar-but-distinct items.
- Semantic (embedding-based) duplicate detection is intentionally **out of
  scope** for the default offline build (it needs a model); it is listed as a
  future direction in [`PAPER_PATH.md`](PAPER_PATH.md).
