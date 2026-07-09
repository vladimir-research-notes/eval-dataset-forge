# Dataset cards

> Milestone 11. Implemented in `src/eval_dataset_forge/reports/`; exercised by
> `tests/test_reports.py` (64 tests) and by the committed demo release
> [`datasets/releases/qa_toy_v0.1.0/dataset_card.md`](../datasets/releases/qa_toy_v0.1.0/dataset_card.md).

A dataset card is the document a reader consults before trusting a benchmark. The
failure mode it exists to prevent is not absence — most datasets ship a README — it
is **drift**: a card written by hand in week one, describing a corpus that has been
re-cut four times since, with a "known issues" section that lists the issues someone
remembered.

So this card is **derived**. `build_card` takes exactly two inputs, a release's
`manifest.json` and its `audit.json`, and every count it prints is read from them.

```bash
# Render a release's card. --out is optional; without it the Markdown goes to stdout.
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 \
  --out results/reports/qa_toy_dataset_card.md \
  --html results/reports/qa_toy_dataset_card.html

# Does this tool version still render the card the release shipped? (CI gate; exit 1 on drift)
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check

# Did the curator leave a section unwritten? (CI gate; exit 1 if so)
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --fail-on-incomplete
```

---

## 1. Nothing is recomputed

The card does not re-run the deduplication ladder, the leakage checks, or the label
audit. It reads `audit.json`.

That is a deliberate constraint, not laziness. Recomputing would let a threshold move
between the moment the audit ran and the moment the card was rendered, and the card
would then disagree with the `audit.json` sitting beside it in the same directory. A
card is a **view** of the evidence, and a view that recomputes its own source is not a
view — it is a second, unversioned audit.

`test_card_counts_match_the_audit_exactly` pins this. `test_dropping_a_duplicate_changes_the_card_the_release_ships`
pins the consequence: `--drop exact` dissolves the duplicate cluster, the release's
audit reports zero exact duplicates, and so does its card.

## 2. Two things the card cannot print, by arithmetic

`dataset_card.md` is an artifact of the release, hashed into `manifest.hashes` along
with `examples.jsonl` and the split files. Two conventional card fields therefore
become impossible rather than merely undesirable.

**It cannot print the file-hash table.** The digest of the card's bytes would have to
appear inside the card's bytes. That is a preimage requirement, not a design choice.
`build_card` never reads `manifest.hashes`; `test_card_is_invariant_to_manifest_hashes`
replaces every digest with `sha256:ffff…` and asserts the rendered card is unchanged.

**It cannot print a build timestamp.** Every dataset card in the wild says "Created:
2026-07-09". If this one did, the clock would reach a hashed artifact, every rebuild
would differ from every other in `manifest.hashes`, and the `none` version bump that
[`versioning.py`](../src/eval_dataset_forge/datasets/versioning.py) defines — *"nothing
changed but the clock"* — would become unreachable. Milestone 10 kept `created_at` out
of `REPRODUCE.md` for exactly this reason; the card inherits the constraint.
`test_two_clocks_differ_in_the_manifest_and_nothing_else` builds the same release at
2020 and at 2031 and asserts `manifest.json` is the only artifact that moves.

What the card prints instead are `content_id` and `eval_id`, which are functions of the
*examples* rather than of the card, and a pointer to `verify-release` for byte integrity.

### The order this forces on the release builder

The card is rendered *from* a manifest, and its digest belongs *in* that manifest. So
[`release.py`](../src/eval_dataset_forge/datasets/release.py) builds the manifest twice:

1. build a manifest over `{examples.jsonl, splits/*, audit.json, REPRODUCE.md}`;
2. render the card from it;
3. rebuild the manifest with `dataset_card.md` in the artifact set.

The two manifests differ in `hashes` and nothing else — which is precisely the field the
card does not read. Were that not true, the shipped card would describe a manifest that
was thrown away. `test_a_shipped_card_regenerates_byte_for_byte_from_the_release` is the
end-to-end check.

## 3. `--check` and `verify-release` answer different questions

They are deliberately not merged.

| command | question | depends on |
|---|---|---|
| `verify-release` | Are the bytes on disk the bytes the manifest describes? | the release directory alone |
| `dataset-card --check` | Does *this* tool version still render the card that shipped? | the release **and** the renderer |

`verify-release` already covers the card's integrity: `dataset_card.md` is in
`manifest.hashes`, so a corrupted card fails `hashes_match` like any other artifact
(`test_verify_release_catches_a_corrupted_card`).

Folding the render check into `verify-release` would make an old, perfectly intact
release fail verification on the day the card's wording improved — turning a byte
integrity check that any reader can run into a claim about this repository's renderer.
`test_check_card_reports_drift_without_calling_the_release_corrupt` constructs exactly
that case: a card edited *and* its hash updated. `verify-release` passes; `--check` fails
and says the renderer changed, not that the release is corrupt.

## 4. The headline number is two numbers

Milestone 10 left this open:

> *"It will need to decide whether the card prints `block_count` next to `example_count`
> — a card that says '14 examples' when the corpus holds 11 independent blocks is not
> lying, but it is not helping."*

**It prints both.** A reader spends the headline number as a sample size, and `14` is not
the sample size of anything. The demo card's first table reads *"14 examples in 11
independent blocks"*, followed by a note naming the three examples that are duplicates
or share a source record.

And it goes one step further, because neither number bounds the claim a reader usually
wants to make. A confidence interval on a **test** score is governed by the number of
independent blocks in the **test split**. So `manifest.dependence` gained
`blocks_by_split` and `straddling_block_count`, and the card's split table has an
`independent blocks` column:

| split | examples | independent blocks | share |
|---|---|---|---|
| `dev` | 1 | 1 | 7.1% |
| `test` | 2 | 2 | 14.3% |
| `train` | 11 | 8 | 78.6% |

A block whose members straddle a split boundary is counted in **every** split it reaches,
so the per-split counts sum to at least `block_count`; `straddling_block_count` says by
how much. Under the adversarial strategy it is zero. Under `random` it is not, and it is
the same defect `residual_overlap` reports, counted differently.

## 5. Zeros go in the table, never in the bullets

The findings table lists every audit dimension, including the ones that fired zero times,
with a **certainty** column:

| dimension | count | certainty |
|---|---|---|
| duplicates: exact | 1 | exact |
| duplicates: near | 1 | heuristic (tfidf) |
| leakage: n-gram overlap | 0 | exact given the n-gram width |
| leakage: answer in prompt | 1 | heuristic |
| labels: needs adjudication | 4 | derived from the rows above |
| PII-like patterns | 0 | heuristic (pattern match) |

Two facts a bare count cannot carry:

- **A zero on a heuristic row is not a guarantee.** It says the check found nothing. The
  near-duplicate stage is lexical, and `ret-001`/`ret-002` in the retrieval fixture is a
  genuine paraphrase pair that no lexical threshold can reach without also flagging a
  genuine non-duplicate (see [`DEDUPLICATION.md`](DEDUPLICATION.md)). Every heuristic
  count in a card is a lower bound, and the table's caption says so.
- **A zero must be visible at all.** A card listing only what was found is
  indistinguishable from a card whose pipeline never ran that check.

The "Known issues" bullets carry only the non-zero findings, with example ids. A bullet
reading *"0 duplicates"* is a reassurance; a table row reading `0` is a statement about
what was checked. When *every* dimension is zero the card says so without claiming the
data is clean (`test_an_empty_audit_says_so_without_claiming_the_data_is_clean`).

## 6. Curator prose: absent, never invented

Four sections cannot be computed from any audit, because they are claims about intent:
`purpose`, `construction`, `recommended_use`, `non_goals`.

They are supplied in a YAML file, validated against `CardMeta` (which forbids unknown
fields — a `non_goal:` typo raises rather than silently dropping the curator's non-goals),
and **recorded in `manifest.card_meta`**:

```bash
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --dataset-id qa_toy \
  --card-meta datasets/fixtures/qa_toy/card.yaml ...
```

The prose lives in the manifest rather than being read from a side file at render time,
because a card whose text lived outside the release could not be checked against the
release: `--check` would compare the shipped card against a card rendered from whatever
`card.yaml` happened to be on the checker's disk.

A section the curator left blank is **named** in the rendered card, not quietly omitted.
A reader must be able to tell *"no restrictions"* from *"nobody wrote any down"*, and
`--fail-on-incomplete` lets CI insist on the difference. The card's last section reports
`4 of 4 curator sections are supplied` — or names the missing ones.

### Editing prose is a `patch`

Rewriting a non-goal changes no number and cannot move a score, so it is packaging.
`card_meta` was added to `_PACKAGING_FIELDS`, which makes the bump `patch` even for a
bundle built without a card — otherwise a release whose stated purpose had been rewritten
would report `none`, *"nothing but the clock"*, and that would be false.

```text
delta: 0.1.0 → 0.1.1 (patch bump; scores still comparable)
  content_id : same
  eval_id    : same
  - packaging only: hashes, card_meta
```

## 7. One document, two backends

Markdown and HTML are not two templates. A report is built once as a `Document` — a flat
list of headings, paragraphs, bullets, tables, notes, and code blocks — and rendered
twice. Two templates are two places a number is written down, and the second one is where
it goes stale. `test_every_string_reaches_the_markdown_rendering` and its HTML twin walk
the document and assert no backend drops content; no template pair can offer that.

The inline vocabulary is one item long: a **code span**. Bold, links, and nested emphasis
would need a Markdown parser in each backend, and the dependency budget (`pydantic`,
`numpy`, `pyyaml`) has no room for one. The cost is that a card cannot link out.

### Escaping is a correctness property, not decoration

The card interpolates untrusted text: `license_notes`, curator prose, and example ids
from a JSONL file someone else wrote.

- **Markdown** escapes `\`, `` ` ``, `|`, `*`, `<`, `[` uniformly — in every literal span,
  not only inside table cells. A contextual escaper has to know which block it is in, and
  the one time it guesses wrong is on a dataset whose `license_notes` contains a `|`, at
  which point the card's table silently grows a column. `<` and `[` are escaped because
  CommonMark passes raw HTML through and `[text](url)` would become a live link the
  curator never wrote.
- **Intraword `_` is *not* escaped.** CommonMark permits intraword emphasis with `*` and
  forbids it with `_`, so `qa_toy` and `effective_n` cannot open emphasis, and escaping
  them would put a backslash through the most-read line of every card. `_emphasis_` at a
  word boundary still can, and still is.
- **A line-initial `1.` escapes the *dot*, not the digit.** `\1.` would render as a literal
  backslash and still open an ordered list: CommonMark only recognizes a backslash escape
  before ASCII punctuation. This was a real bug, caught by
  `test_a_leading_marker_does_not_open_a_list`.
- **HTML** escapes every literal span with `html.escape(quote=True)`, and `<title>` gets a
  *plain* rendering — a `<code>` element inside `<title>` is not markup, it is six literal
  characters in the browser tab.

The generated HTML page is standalone: one file, inline stylesheet, no scripts, no fonts,
no network. A dataset card should not require a server, and it should not be able to
phone one.

## 8. What the card does not do

- **It is not evidence.** It is a rendering of evidence. `audit.json` is the artifact a
  tool should read; the card is the artifact a person should read. When they disagree, the
  card is wrong by construction, and `--check` says so.
- **It states no calibration verdict for the difficulty buckets.** `AuditResult` records
  the difficulty *distribution* but not whether the heuristic's ranking was distinguishable
  from chance, so the card cannot conditionally warn. It carries an unconditional note
  instead, and the verdict does reach the reader — through `known_limitations` and
  `recommendations`, which the card prints verbatim. Adding a boolean to `AuditResult`
  would let the note become conditional; it would also be a schema change made for a
  rendering convenience.
- **It cannot detect a dishonest curator.** `purpose` and `non_goals` are unverified
  claims, and the card labels them as the only unverified claims on the page. A release
  whose curator wrote *"suitable for clinical use"* would render that sentence faithfully.
- **HTML is not part of the release bundle.** `docs/PLAN.md` §5D lists `dataset_card.md`
  and no HTML. A second rendering of the same facts inside a hashed bundle doubles the
  surface that can go stale and adds no information, so `--html` writes wherever the caller
  asks and nothing hashes it.
- **Every heuristic count it prints is a lower bound**, inherited from
  [`DEDUPLICATION.md`](DEDUPLICATION.md) and [`LEAKAGE.md`](LEAKAGE.md). The card repeats
  the bound; it does not narrow it.

## Files

| path | role |
|---|---|
| `src/eval_dataset_forge/reports/document.py` | syntax-free document model, inline spans |
| `src/eval_dataset_forge/reports/markdown.py` | CommonMark backend + escaping |
| `src/eval_dataset_forge/reports/html.py` | standalone-page backend + escaping |
| `src/eval_dataset_forge/reports/dataset_card.py` | `build_card`, layout, `--check` |
| `datasets/fixtures/qa_toy/card.yaml` | the demo release's curator prose |
| `datasets/releases/qa_toy_v0.1.0/dataset_card.md` | a real generated card |
| `tests/test_reports.py` | 64 tests |
