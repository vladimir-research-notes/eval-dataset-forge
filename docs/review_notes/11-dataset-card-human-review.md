# Human review: Milestone 11 — Dataset card generator

TL;DR

A dataset card is now generated from a release's `manifest.json` + `audit.json`, rendered to
Markdown (in the bundle) and optionally to a standalone HTML page. Nothing is recomputed, so the
card's "known issues" cannot drift from what the pipeline found. The card turns out to be an
*artifact of the release it describes* — hashed into `manifest.hashes` — which makes two fields
every real dataset card carries impossible: the hash table (its own digest cannot appear inside its
own bytes) and `created_at` (a clock in a hashed artifact would make Milestone 10's `none` version
bump unreachable). That forces the release builder to assemble its manifest twice, and the
correctness of that construction rests on `build_card` being blind to `hashes`. Milestone 10's
handoff question is answered: the card prints `14 examples in 11 independent blocks`, and per split,
because the number bounding a test-set claim is the *test split's* block count.

1659 tests pass (1595 prior + 64 new). `ruff check .` clean.

What changed

- New `src/eval_dataset_forge/reports/` package: `document.py` (syntax-free document model),
  `markdown.py` and `html.py` (two backends, one document), `dataset_card.py` (`build_card`, layout,
  `check_card`).
- `schemas.py`: new `CardMeta` model; `DatasetManifest.card_meta`; `DatasetCard` gains `identity`,
  `provenance`, `missing_curator_sections()`. Additive with defaults — `SCHEMA_VERSION` unchanged.
- `datasets/release.py`: two-pass manifest assembly; `dataset_card.md` joins the artifact set;
  `_dependence()` gains `blocks_by_split` and `straddling_block_count`; `REPRODUCE.md` no longer
  lists the card among the missing artifacts.
- `datasets/versioning.py`: `card_meta` added to `_PACKAGING_FIELDS` — editing prose is a `patch`.
- `cli.py`: `evalforge dataset-card --release [--out] [--html] [--check] [--fail-on-incomplete]
  [--json]`; `evalforge release --card-meta`.
- `datasets/fixtures/qa_toy/card.yaml` (curator prose) and a regenerated
  `datasets/releases/qa_toy_v0.1.0/` whose bundle now contains a real generated card.
- `tests/test_reports.py` (64 tests); `tests/test_schemas.py` updated for the new model.
- Docs: new `docs/DATASET_CARD.md`; updates to `RESULTS.md`, `LIMITATIONS.md`, `DATASET_SCHEMA.md`,
  `RELEASES.md`, `REPRODUCIBILITY.md`, `README.md`. Website: `dataset-card.html` rewritten; status
  and content updates across seven other pages.

What should be checked later

- **The two-pass manifest.** `build_release.assemble()` calls `build_manifest` twice. The claim is
  that the two manifests differ only in `hashes`, and that `build_card` reads neither `hashes` nor
  `created_at`. `test_card_is_invariant_to_manifest_hashes` asserts the blindness directly rather
  than only through the release path. Worth an independent read: if any future card field derives
  from a manifest field that the card's own presence changes, the shipped card silently describes a
  discarded manifest.
- **Whether `card_meta` belongs in the manifest.** It makes `manifest.json` grow with the length of a
  curator's non-goals. The alternative (a hashed `card_meta.yaml` artifact) keeps the manifest an
  inventory and adds a ninth file. Both satisfy the constraint that the card must be regenerable from
  the release alone. This is a judgement call and I picked one.
- **The certainty column's wording.** Each audit dimension is labelled `exact` / `heuristic` /
  `exact given the n-gram width` / `derived from the rows above`. These map onto claims the upstream
  modules make in their own docstrings, but the mapping is hand-written in `_CERTAINTY` and nothing
  enforces that it stays true if a stage's method changes.
- **`--fail-on-incomplete` measures non-emptiness.** A curator satisfies it with "TBD". Is a gate
  that can be trivially satisfied better than no gate?
- **The demo release was regenerated in place at `0.1.0`** rather than bumped to `0.1.1`. Its
  `content_id` and `eval_id` are unchanged; only the bundle grew an artifact. A published release
  would have taken a `patch`. Reasonable for a pre-1.0 demonstration artifact regenerated from
  fixtures, but it is the one place the repo does not follow its own bump rule, and it should be
  called out if that bothers a reviewer.

Technical risks

- **Escaping is the security surface.** The card interpolates untrusted text (curator prose,
  `license_notes`, example ids from someone else's JSONL). Markdown escapes ``\ ` | * < [``
  uniformly; HTML escapes every literal span. Intraword `_` is deliberately *not* escaped, on the
  grounds that CommonMark forbids intraword `_` emphasis. If that reading of the spec is wrong, a
  crafted `license_notes` could italicise part of a card. It cannot inject HTML or a link.
  `test_curator_prose_cannot_inject_markup` covers the `<script>` and `|` cases.
- **A real bug was found here and fixed:** `escape_leading` escaped the digit in a line-initial
  `1.` (`\1.`), which CommonMark renders as a literal backslash and which still opens an ordered
  list. The escape belongs on the dot. Other line-initial markers (`- + # >`) were correct.
- **`check_card` is tool-version dependent.** By design — it asks whether *this* renderer still
  produces the shipped card, not whether the bytes are intact. It is deliberately not folded into
  `verify-release`, because an intact release must not fail verification the day the card's wording
  improves. The consequence is that a third party who receives a release cannot distinguish a
  generated card from a hand-written one. Nothing in the bundle can.
- **`_dependence()` gained keys inside a free-form dict.** No schema change, but any consumer that
  round-tripped `manifest.dependence` through a strict model would now reject it. Nothing in this
  repo does.
- **`reports` imports `datasets.manifest` lazily** (inside `load_release`/`check_card`) while
  `datasets.release` imports `reports.dataset_card` at module level. There is no cycle today because
  `datasets/__init__.py` does not import `release`. That invariant is load-bearing and undocumented
  outside `datasets/__init__.py`'s docstring.

Commands to verify

```bash
python3 -m pytest -q                          # 1659 passed
python3 -m pytest -q tests/test_reports.py    # 64 passed
ruff check .                                  # all checks passed
python3 scripts/validate_fixtures.py          # exit 0
python3 scripts/make_demo_scores.py --check   # exit 0

# The committed demo release regenerates its own card, byte for byte.
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check

# The bundle, card included, verifies against its own manifest (ten checks).
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0

# 4 of 4 curator sections supplied.
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --fail-on-incomplete

# The card, plus a standalone HTML rendering (no scripts, no network).
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 \
  --out results/reports/qa_toy_dataset_card.md \
  --html results/reports/qa_toy_dataset_card.html

# Rewriting curator prose changes no number and derives a patch bump. Hold every data-determining
# flag at the demo release's value and drop only --card-meta; otherwise the corpus re-splits and
# the bump is (correctly) major.
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --dataset-id qa_toy \
  --strategy adversarial --seed 0 --ratios 0.7,0.1,0.2 --stratify-by task_family \
  --quarantine-blocking \
  --license-notes 'Synthetic fixture data, CC0-1.0. No real personal, clinical, or patient data.' \
  --version auto --previous datasets/releases/qa_toy_v0.1.0 --out /tmp/qa_toy_prose
#   delta: 0.1.0 → 0.1.1 (patch bump; scores still comparable)
#     - packaging only: hashes, card_meta

# The card carries no clock: two builds, eleven years apart, differ in manifest.json alone.
PYTHONPATH=src python3 -m eval_dataset_forge release --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset-id qa_toy --created-at 2020-01-01T00:00:00+00:00 --out /tmp/clk_a
PYTHONPATH=src python3 -m eval_dataset_forge release --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset-id qa_toy --created-at 2031-12-31T23:59:59+00:00 --out /tmp/clk_b
diff -r /tmp/clk_a /tmp/clk_b     # only manifest.json
```

Human feedback
<!-- Human will fill this later. -->
