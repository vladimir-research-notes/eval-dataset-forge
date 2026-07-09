# Milestone 11 — Dataset card generator

## Summary

Land the reporting layer: a syntax-free document model, a Markdown backend, an HTML backend, a
`DatasetCard` built from a release's `manifest.json` + `audit.json`, `dataset_card.md` inside the
release bundle, and the `evalforge dataset-card` command.

The distinguishing work was not the layout. It was (a) noticing that a dataset card is an
**artifact of the release it describes**, hashed into `manifest.hashes` like any other, which makes
two fields every real dataset card carries — the hash table and `created_at` — *arithmetically*
impossible rather than merely unwise; (b) discovering that this forces the release builder to
assemble its manifest **twice**, and that the correctness of that construction rests on
`build_card` being blind to exactly the field the second pass adds; (c) answering Milestone 10's
handoff question with a number the manifest could not yet supply, which is why
`manifest.dependence` gained `blocks_by_split`; and (d) finding a real CommonMark escaping bug —
`\1.` — that a test caught and no amount of eyeballing would have.

## Deliverables

- `src/eval_dataset_forge/reports/` — the package `PLAN.md` §4 specifies, plus one module:
  - `document.py` — `Span`, `inline_spans()`, `Heading`, `Paragraph`, `Bullets`, `Table`, `Note`,
    `Code`, `Document`, `strings_of()`, `document_strings()`, `blocks_of()`, `rows()`.
  - `markdown.py` — `escape()`, `escape_leading()`, `inline()`, `cell()`, `render_block()`,
    `render()`, `to_bytes()`.
  - `html.py` — `STYLESHEET`, `inline()`, `render_block()`, `render()`, `to_bytes()`.
  - `dataset_card.py` — `build_card()`, `known_issues()`, `card_document()`, `render_markdown()`,
    `render_html()`, `load_card_meta()`, `load_release()`, `card_from_release()`, `check_card()`,
    `write_card()`, `card_json()`, `format_card()`.
- `schemas.py` — new `CardMeta`; `DatasetManifest.card_meta`; `DatasetCard` gains `identity`,
  `provenance`, and `missing_curator_sections()`; `CURATOR_SECTIONS`. All additive with defaults;
  `SCHEMA_VERSION` unchanged.
- `datasets/release.py` — two-pass manifest assembly, `dataset_card.md` in the artifact set,
  `card_meta` parameter, `_dependence()` gains `blocks_by_split` + `straddling_block_count`,
  `REPRODUCE.md` no longer lists the card as missing.
- `datasets/versioning.py` — `card_meta` added to `_PACKAGING_FIELDS`.
- `cli.py` — `evalforge dataset-card` (`--release`, `--out`, `--html`, `--check`,
  `--fail-on-incomplete`, `--json`) and `release --card-meta`.
- `datasets/fixtures/qa_toy/card.yaml` — the demo release's curator prose.
- `datasets/releases/qa_toy_v0.1.0/dataset_card.md` — a real generated card, regenerated in place.
- `tests/test_reports.py` — 64 tests.
- Docs: new `docs/DATASET_CARD.md`; `RESULTS.md` gains its card section; `LIMITATIONS.md` gains one;
  `DATASET_SCHEMA.md` documents `CardMeta`, `DatasetCard`, `card_meta`, and the new `dependence`
  keys; `RELEASES.md` and `REPRODUCIBILITY.md` updated; `README.md` CLI status corrected. Website:
  `dataset-card.html` rewritten; status/content updates to `milestones.html`, `index.html`,
  `results.html`, `limitations.html`, `releases.html`, `schema.html`, `audit-pipeline.html`.

## Design decisions

- **The card is a view, and a view does not recompute its own source.** `build_card` reads
  `manifest.json` and `audit.json` and nothing else. Re-running the dedup ladder at render time
  would let a threshold move between the audit and the card, and the card would then disagree with
  the `audit.json` beside it in the same directory.
- **A card cannot print the hash table that contains it.** `manifest.hashes["dataset_card.md"]` is
  the digest of the card's bytes. For the card to print it, that digest would have to appear inside
  the bytes it is the digest of. Not a style choice — a preimage requirement.
- **A card cannot print `created_at`.** Every dataset card in the wild says "Created: …". If this
  one did, the clock would enter a hashed artifact, every rebuild would differ in `hashes`, and
  Milestone 10's `none` version bump would become unreachable. `REPRODUCE.md` already omitted the
  timestamp for this reason; the card inherits the constraint. Both absences are pinned by tests.
- **Therefore the manifest is assembled twice.** Build a manifest; render the card from it; rebuild
  the manifest with the card's bytes in `artifacts`. The two differ in `hashes` alone — the field
  `build_card` refuses to read. The invariance is not a nicety that permits the two-pass build; it
  is the property that makes it *correct*, and `test_card_is_invariant_to_manifest_hashes` asserts
  it directly rather than through the release path that depends on it.
- **`--check` is not `verify-release`, and is deliberately not folded into it.** `verify-release`
  asks whether the bytes on disk are the bytes the manifest describes; any reader with any version
  of the tool can answer it, and it already covers the card's integrity because the card is in
  `hashes`. `--check` asks whether *this* tool version still renders the shipped card — a question
  about the repository, not the release. Merging them would make an intact release fail verification
  the day the card's wording improved. This is the same objection Milestone 10 raised against
  `verify-release` re-running the audit, applied to itself.
- **Print `example_count` *and* `block_count`, and both per split.** A reader spends the headline as
  a sample size, and 14 is not the sample size of anything. Neither is 11, for a claim about `test`.
- **Curator prose lives in the manifest, not in a side file.** A card whose text lived outside the
  release could not be checked against the release: `--check` would compare the shipped card against
  whatever `card.yaml` happened to be on the checker's disk.
- **A blank section is named, never omitted.** A reader must be able to tell "no restrictions" from
  "nobody wrote any down". `--fail-on-incomplete` lets CI insist.
- **Editing prose is a `patch`.** `card_meta` joins `_PACKAGING_FIELDS` even though a rewritten card
  already moves `hashes` — the redundancy covers a bundle built *without* a card, which would
  otherwise report `none`, "nothing but the clock", and that would be false.
- **Zeros in the table, never in the bullets.** A bullet reading "0 duplicates" is a reassurance; a
  table row reading `0` beside `heuristic (tfidf)` is a statement about what was checked and what
  the check is worth. A card listing only what was found is indistinguishable from a card whose
  pipeline never ran that check.
- **One document, two backends, not two templates.** Two templates are two places a number is
  written down, and the second is where it goes stale. A test walks the document and asserts every
  string reaches both renderings.
- **The inline vocabulary is one item long: a code span.** Bold, links, and nested emphasis would
  need a Markdown parser in each backend. The dependency budget has no room for one. The cost is
  that a card cannot link out.
- **Escaping is uniform, not contextual.** A contextual escaper must know which block it is in, and
  the one time it guesses wrong is on a dataset whose `license_notes` contains a `|`.
- **HTML is a CLI output, not a bundle artifact.** `PLAN.md` §5D lists `dataset_card.md` and no HTML.
  A second rendering of the same facts inside a hashed bundle doubles the surface that can go stale.

## Deviations from the design doc

- **`document.py` is a fourth module** beyond the three `PLAN.md` §4 lists for `reports/`. It holds
  the document model that `markdown.py` and `html.py` render, and that Milestone 12's `audit.md`
  will reuse. Without it, `markdown.py` and `html.py` would each be a template.
- **`--out` is optional.** `PLAN.md` §8 shows `dataset-card --release … --out …`. Without an output
  flag the card goes to stdout, which is what a reader trying the tool for the first time wants.
- **Four flags were added:** `--html`, `--check`, `--fail-on-incomplete`, `--json`.
- **`DatasetManifest` gained an eighth field** (`card_meta`) and `DatasetCard` gained two
  (`identity`, `provenance`). Additive, with defaults; `SCHEMA_VERSION` unchanged; recorded in
  `docs/DATASET_SCHEMA.md`.
- **`_dependence()` gained two keys.** `blocks_by_split` and `straddling_block_count` are Milestone
  10 data that Milestone 11 needed and Milestone 10 did not record. `dependence` is a free-form dict,
  so this is not a schema change.
- **The demo release was regenerated in place at `0.1.0`** rather than bumped. Its `content_id` and
  `eval_id` are unchanged; only the bundle's contents grew. A *published* release gaining a card
  would take a `patch` — and the validation section below runs exactly that command to show the tool
  deriving `0.1.0 → 0.1.1` from a prose change, which is the same class of difference.

## The question Milestone 10 left open

> *"It will need to decide whether the card prints `block_count` next to `example_count` — a card
> that says '14 examples' when the corpus holds 11 independent blocks is not lying, but it is not
> helping."*

**It prints both — and then a third number, because neither of the first two is the one a reader
needs.** A confidence interval on a *test* score is governed by the independent blocks in the *test
split*, and the manifest recorded only a corpus-wide `block_count`. So `_dependence()` now computes
`blocks_by_split`, and the card's split table reads:

| split | examples | independent blocks | share |
|---|---|---|---|
| `dev` | 1 | 1 | 7.1% |
| `test` | 2 | 2 | 14.3% |
| `train` | 11 | 8 | 78.6% |

A block straddling a boundary is counted in every split it reaches, so the per-split counts sum to
at least `block_count`; `straddling_block_count` says by how much. Under `adversarial` it is zero —
which is the split builder's guarantee, restated in the units a bootstrap uses. Under `random` it is
not, and it is the same defect `residual_overlap` reports, counted differently.

## The result worth reading twice

**A dataset card is an artifact of the release it describes, and that fact removes two fields from
it that every real dataset card has.**

`dataset_card.md` is hashed into `manifest.hashes` beside `examples.jsonl`. So:

1. It cannot print the hash table. `H(card_bytes)` would have to appear inside `card_bytes`.
2. It cannot print `created_at`. A clock inside a hashed artifact makes every rebuild differ from
   every other, and `versioning.compare()` would report at least a `patch` for a build that changed
   nothing. Milestone 10 spent a paragraph keeping the timestamp out of `REPRODUCE.md`; a card with
   a "Created:" line would have quietly undone it.

Measured, on the committed fixture — two builds, same data, clocks eleven years apart:

```text
  same    REPRODUCE.md        same    dataset_card.md     same    examples.jsonl
  same    audit.json          same    splits/{train,dev,test}.jsonl
  DIFFER  manifest.json
```

And (2) is the constraint that makes (1) *load-bearing* rather than merely true. The card must be
rendered before the manifest that hashes it can exist. So the builder assembles the manifest twice
and hands the first one to `build_card`. If `build_card` read `hashes`, the shipped card would
describe the provisional manifest — the one that was thrown away — and
`dataset-card --check` would fail on every release the tool ever built. The blindness is not a
convenience. It is what makes the two-pass construction sound.

## A bug the tests found and reading would not have

`escape_leading()` escaped a line-initial ordered-list marker as `\1.`, escaping the **digit**.
CommonMark only recognizes a backslash escape before ASCII *punctuation*, so `\1.` renders as a
literal backslash followed by `1.` — which still opens an ordered list. The escape had to fall on
the dot: `1\.`. Caught by `test_a_leading_marker_does_not_open_a_list`, which asserted the output
rather than that "escaping happened".

The neighbouring decision went the other way. Escaping `_` uniformly turned the most-read line of
every card into `# Dataset card — qa\_toy v0.1.0`. CommonMark permits intraword emphasis with `*`
and **forbids** it with `_`, so `qa_toy` and `effective_n` cannot open emphasis and need no escape;
`_emphasis_` at a word boundary can, and does. `*` stays escaped everywhere, because `a*b*c` is
emphasis.

## Honest negatives

- **The card is a rendering of evidence, not evidence.** `audit.json` is what a tool should read.
  When the two disagree the card is wrong by construction, and `--check` says so.
- **Its counts inherit every upstream bound.** A heuristic zero is what the check found, not what is
  there. `ret-001`/`ret-002` is invisible to the audit and therefore invisible to the card. The
  findings table's caption says this; a reader who skips captions will read a zero as an absence.
- **Four sections are unverified claims.** Nothing checks whether the curator's prose is *true*. A
  release whose curator wrote "suitable for clinical use" would render that sentence faithfully. The
  card labels them as the only unverified text on the page, which is the most the tool can do.
- **The card cannot conditionally warn about the difficulty buckets.** `AuditResult` records the
  difficulty *distribution* but not the calibration verdict, so the note under the table is
  unconditional. The verdict does reach the reader, two sections away, through `known_limitations`
  and `recommendations` printed verbatim. Fixing this properly means a schema change made for a
  rendering convenience, and it was not made.
- **`--fail-on-incomplete` measures non-emptiness.** A curator can satisfy it with "TBD".
- **`check_card` is tool-version dependent, by design, and therefore useless to a third party.** A
  reader who receives a release from elsewhere can verify its integrity and cannot verify that its
  card was generated rather than hand-written. Nothing in the bundle distinguishes the two.
- **A card cannot link out.** No link node in the inline vocabulary; adding one needs a URL policy
  for a file read variously from a release directory, a repository, and a web page.
- **The demo release's card describes a 14-example corpus with a two-example test split.** It
  demonstrates the format. Milestone 9's small-N arithmetic applies with force, and the card says so
  in three places.

## Validation

```bash
python3 -m pytest -q                          # 1659 passed (1595 prior + 64 reports)
python3 -m pytest -q tests/test_reports.py    # 64 passed
python3 scripts/validate_fixtures.py          # all 5 fixtures valid, exit 0
python3 scripts/make_demo_scores.py --check   # fixture matches its generator, exit 0
ruff check .                                  # all checks passed

# The committed demo release still renders its own card, byte for byte (exit 0).
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check

# 4 of 4 curator sections supplied (exit 0); a release built without --card-meta exits 1.
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --fail-on-incomplete

# The card, and a standalone HTML rendering of it.
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 \
  --out results/reports/qa_toy_dataset_card.md \
  --html results/reports/qa_toy_dataset_card.html

# The whole bundle, card included, still verifies against its own manifest (ten checks, exit 0).
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0

# Rewriting curator prose changes no number and derives a patch bump (exit 0). Every flag that
# determines the *data* is held at the demo release's value; only --card-meta is dropped.
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --dataset-id qa_toy \
  --strategy adversarial --seed 0 --ratios 0.7,0.1,0.2 --stratify-by task_family \
  --quarantine-blocking \
  --license-notes 'Synthetic fixture data, CC0-1.0. No real personal, clinical, or patient data.' \
  --version auto --previous datasets/releases/qa_toy_v0.1.0 --out /tmp/qa_toy_prose
#   delta: 0.1.0 → 0.1.1 (patch bump; scores still comparable)
#     - packaging only: hashes, card_meta

# Drop --quarantine-blocking from that command and the same corpus re-splits: five examples cross
# the dev/test boundary, eval_id moves, content_id does not, and the bump is major. Milestone 10's
# headline, reachable from the committed release in one command.
```

Verified end to end, not merely unit-tested: the committed release passes all ten `verify-release`
checks with the card in `hashes`; the card regenerated from that release's `manifest.json` +
`audit.json` is byte-identical to the one on disk; two builds eleven years apart differ in
`manifest.json` and nothing else; corrupting one byte of `dataset_card.md` trips `hashes_match`;
editing the card *and* its recorded hash leaves `verify-release` green and makes `--check` red, with
a message saying the renderer changed rather than that the release is corrupt; `<script>alert(1)</script>`
in a curator's `purpose` renders as text in both formats; and `--drop exact` dissolves the duplicate
cluster so the shipped card reports `duplicates: exact | 0` rather than the source corpus's `1`.

## Open questions for review

- **Should `AuditResult` record the difficulty calibration verdict, not just the distribution?** It
  would let the card's note become a conditional warning that fires only when the heuristic failed
  its permutation test. It would also be a schema change made for a rendering convenience, and
  `AuditResult` has so far recorded *findings* rather than *verdicts about methods*.
- **Should the card quote the `examples.jsonl` digest?** Only the card's *own* digest is impossible;
  quoting another artifact's is not circular, and it would let a reader holding just the card verify
  the data. It would also re-introduce a file hash into a document whose whole thesis is that file
  hashes answer the wrong question.
- **Nothing in a release distinguishes a generated card from a hand-written one.** `--check` answers
  it for someone who has this repository at this version. A third party cannot. A `generated_by`
  line would be a claim, not a proof, and the honest fix is signing — which is out of scope for the
  same reason it was in Milestone 10.
- **Is `card_meta` in the manifest the right home for prose?** It makes `manifest.json` grow with the
  length of a curator's non-goals, and a manifest has so far been an inventory. The alternative — a
  `card_meta.yaml` artifact inside the bundle, hashed — keeps the manifest lean and adds a ninth
  file, and it is not obviously worse.
- **`--fail-on-incomplete` cannot distinguish "TBD" from a real answer,** and no rule can. Is a gate
  that measures non-emptiness better than no gate, or does it launder a box-ticking exercise as a
  quality check?
- **Should `dataset-card --check` run inside `verify-release` when the tool versions match?**
  `manifest.tool_version` records the builder. Comparing it against `__version__` and only then
  re-rendering would give a third party the check when it is meaningful and skip it when it is not.
  It would also make `verify-release`'s behaviour depend on which version wrote the release, which is
  exactly the kind of conditional a verifier should not have.

## Status

Complete. Next: Milestone 12 — Smoke audit & results (full offline audit over all fixtures →
`results/` + `docs/RESULTS.md`). It lands `evalforge audit` and `evalforge smoke`, and it inherits
`reports/document.py`: `audit.md` is the second document this package renders, and the first one
whose Markdown and HTML must agree about a table nobody has written yet. It will also have to decide
whether `audit.md` belongs in the release bundle — the last artifact `REPRODUCE.md` still lists as
missing — and, if so, whether a bundle that hashes both a card and an audit report is a bundle with
two renderings of one truth.
