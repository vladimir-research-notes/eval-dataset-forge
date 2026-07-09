# Releases: identity, hashes, and version bumps

*Milestone 10. Implemented in `src/eval_dataset_forge/datasets/{writer,versioning,manifest,release}.py`.*

A benchmark release is a claim. Not *"here are some rows"* — that is a file. The claim is:

> A score measured on this release is comparable to a score measured on that one.

A filename does not support the claim. A hand-typed version string does not support it either;
in practice a hand-typed version string is a number someone forgot to increment. This milestone
computes the claim instead of accepting it.

---

## 1. A file hash answers the wrong question

The obvious way to identify a dataset is to `sha256sum examples.jsonl`. It fails in both
directions at once:

- **It changes when the dataset does not.** Re-serialize the same records with sorted JSON
  keys, or with `ensure_ascii=True`, or in a different row order, and the digest moves. Nothing
  about the benchmark changed.
- **It stays put when the dataset's *meaning* does.** It cannot distinguish a release whose
  test split holds one set of examples from one whose test split holds another, if the file
  merely lists them all.

So the builder computes three things, and each answers a different question.

| identity | question it answers | ignores | sees |
|---|---|---|---|
| `hashes[<file>]` | did these bytes survive the trip? | nothing | every byte |
| `content_id` | are these the same examples? | row order, key order, `split` | example content |
| `eval_id` | are two reported scores comparable? | row order, key order | example content **and** `split` |

`content_id` is a sha256 over the sorted per-example digests, each digest taken over the
example's canonical JSON with `split` removed. `eval_id` is the same aggregate over sorted
`digest split` pairs. Both are domain-separated by a tag and by `SCHEMA_VERSION`, so an id
computed under one schema can never collide with an id computed under another, and a content id
can never be read as an eval id.

Two properties are worth stating because they are choices, not accidents:

- **Multiplicity counts.** A corpus that lists an example twice is not the corpus that lists it
  once. The aggregate sorts the digests; it does not deduplicate them. The deduplication stage
  exists precisely because that difference matters.
- **`eval_id` hashes the *assignment*, not the strategy.** Two builds that reach the same split
  by different seeds share an `eval_id`, and should: their numbers are comparable. `split_spec`
  is recorded for provenance; identity is taken from the result.

---

## 2. Why `eval_id` has to exist — the measurement

Hold the corpus fixed. Hold the systems fixed: each system's per-example score vector is
generated once and never re-drawn. Vary **only the split seed**. Anything that moves is caused by
the split, which is the one thing `content_id` cannot see.

On the pooled fixture corpus (56 examples, 44 independent blocks, 3 simulated systems,
400 seeds — `scripts/split_identity_experiment.py`):

| | value |
|---|---|
| `content_id` values observed across all 400 seeds × 3 strategies | **1** |
| `eval_id` values observed | 400 per strategy |
| full-corpus accuracy: `sys_a` / `sys_b` | 0.6964 / 0.6607 |
| **the gap the benchmark exists to resolve** | **+0.0357** |
| `sys_a` test accuracy, `random` split | mean 0.7011, **sd 0.1300**, range **[0.364, 1.000]** |
| P(test split reproduces the corpus ordering of the top two) | **0.487** |
| P(the top two tie on test) | 0.172 |
| which system leads, by seed (`random`) | `sys_a` 186 · `sys_b` 128 · `sys_c` 8 · tie 78 |

**The split seed moves the headline number by 3.6× the effect the benchmark was built to
measure, and which system "wins" is decided by the seed rather than by the systems.**

The mean over seeds recovers the corpus number (0.7011 vs 0.6964), so this is variance, not
bias — the test split is an honest estimator of a quantity it is simply too small to pin down.
That is Milestone 9's territory. What belongs to Milestone 10 is the consequence: **a release
identified only by its content would call two of those releases the same.** They are not, and
`eval_id` is what says so.

### The strategies do not estimate the same quantity

A tempting misreading of the same experiment: `adversarial` agrees with the corpus ordering
65.0% of the time versus `random`'s 48.7%, therefore `adversarial` is the better estimator.
It is not — it is estimating something else.

| estimand | `sys_a` | `sys_b` | gap |
|---|---|---|---|
| whole corpus (n=56) | 0.6964 | 0.6607 | +0.0357 |
| singletons only (n=36) | 0.6667 | 0.6389 | +0.0278 |
| duplicate-cluster examples only (n=20) | 0.7500 | 0.7000 | +0.0500 |

| strategy | mean test gap over 400 seeds | sd |
|---|---|---|
| `random` | +0.0409 | 0.1890 |
| `stratified` | +0.0504 | 0.1648 |
| `adversarial` | **+0.1029** | 0.1452 |

`adversarial` keeps duplicate clusters out of `dev`/`test`, and the duplicated examples are not
a random subset — both systems score higher on them. Its test set therefore measures a
sub-population on which the true gap is larger, and a larger true effect is easier to detect.
Its higher agreement rate is a consequence of a **different estimand**, not of a lower-variance
estimate of the same one. The tool records `split_spec` and `eval_id` so a reader can tell which
question a number answers; it does not rank the strategies here.

---

## 3. The bump rule follows from the measurement

`evalforge release --previous <dir> --version auto` derives the version by comparing the two
example sets. It does not read the previous manifest's stored ids: a manifest is a summary, and a
summary that disagrees with its data is `verify-release`'s problem, not the delta's.

| bump | what changed | scores comparable? |
|---|---|---|
| `major` | anything visible to an evaluation split (`dev` ∪ `test`) — an example added, removed, edited, or reassigned across the boundary — or a `schema_version` change | **no** |
| `minor` | content changed, but only within `train` | yes |
| `patch` | no content and no split changed; artifacts, provenance, or audit counts did | yes |
| `none` | nothing but the clock | yes |

Examples are joined across releases by `example_id`. Re-issuing an id with different content
reads as `changed`; renumbering ids without touching content reads as `removed` + `added`. Both
are conservative in the direction that matters: a leaderboard keyed by `example_id` really does
break under either.

**A pure re-split is a `major` bump.** `content_id` is unchanged, `eval_id` is not, and §2 is why.
This is the one verdict a curator will find surprising and the one the measurement was run to
justify.

### `minor` rests on an assumption, and the tool says so

A train-only change cannot move a *fixed* system's test score. It can introduce train→test
leakage that changes what that score **means** for a system trained on the release. The delta
classifies; it does not re-audit. Every `minor` bump therefore emits a warning pointing back at
`audit.json`'s leakage section. Score comparability and benchmark validity are different
properties, and only the first one is decidable from the diff.

### No `0.x` special case

The common convention — *"anything goes below 1.0, so bump the minor"* — collapses `major` and
`minor` into one number exactly on the datasets most likely to be re-cut weekly. The one bit this
module exists to publish is whether yesterday's scores still stand. `0.1.0` + `major` is `1.0.0`.

---

## 4. What a release contains

```text
datasets/releases/<dataset_id>_v<version>/
  examples.jsonl     # the released data, in source order
  manifest.json      # DatasetManifest — the only artifact nothing hashes
  audit.json         # AuditResult, computed over the data that shipped
  dataset_card.md    # DatasetCard rendered from manifest.json + audit.json
  REPRODUCE.md       # the command, the identities, and the gaps
  splits/{train,dev,test}.jsonl
```

`audit.md` (Milestone 12) is not written yet. `REPRODUCE.md` says so in every release rather
than letting the reader assume the bundle is complete.

`dataset_card.md` (Milestone 11) is hashed like any other artifact, which forces the builder to
assemble the manifest **twice**: once so the card has a manifest to render from, once more with
the card's bytes in the artifact set. The two differ only in `hashes` — the one field
`build_card` refuses to read, because a card quoting its own digest would need that digest to
appear inside the bytes it is the digest of. The card carries no timestamp for the same reason
`REPRODUCE.md` does not. See [`DATASET_CARD.md`](DATASET_CARD.md).

### The audit describes the data that shipped

Exclusions are applied first, and the deduplication, leakage, label, and difficulty stages then
run **again** over the survivors. Dropping one member of an exact-duplicate pair dissolves the
cluster; a report computed before the drop would describe a defect the release does not have.
Running the ladder twice on a 56-example corpus costs milliseconds. A release whose `audit.json`
described a different dataset would cost the reader their reason to trust any of it.

### Deletion is a decision, so it is recorded

Milestone 7 promised that *"removing an example is a decision for the release builder, which
records it in a manifest a reader can audit."* Nothing is dropped by default. `--drop` accepts
`exact`, `normalized`, `near`, and `blocking-labels`; every removed example lands in
`manifest.excluded` with its reason and the id of the cluster member that was kept, and
`manifest.source_content_id` pins the pre-exclusion corpus so the deletion is checkable against
the source file rather than taken on faith.

Within a cluster the survivor is the member **first in input order**. That is a convention, not a
judgement: a later member may carry richer metadata and the builder cannot tell. Both ids are
recorded, so the choice is inspectable and reversible.

`--drop near` deletes data on the say-so of a lexical heuristic with known false positives. It is
available. It is not the default. The manifest names every row it took.

### `created_at` is provenance, not identity

It enters no hash and neither id. `REPRODUCE.md` is a hashed artifact, so it carries no timestamp
— embedding the clock there would make every rebuild differ from every other and reduce the
`none` verdict to decoration. The timestamp lives in `manifest.json` alone, the one file nothing
hashes.

The consequence is testable and tested: two builds at a fixed `--created-at` are byte-identical in
every artifact, and two builds at different clocks differ in `manifest.json` **and nothing else**.
This holds across processes and `PYTHONHASHSEED` values, which is the only way to catch a `set`
leaking into a hashed field.

### The manifest records block structure, not `effective_n`

Milestone 9 left the question open. The answer is no. Effective sample size is
`n / (1 + (m̄ − 1)·ICC)`, and the ICC is a property of a **scored comparison** — it needs system
scores, which a dataset does not have. A manifest quoting one would be quoting an assumed ICC.

So `manifest.dependence` records what the computation needs and stops there:

```json
{
  "block_count": 11,
  "block_size_histogram": {"1": 9, "2": 1, "3": 1},
  "duplicate_block_count": 2,
  "near_method": "tfidf"
}
```

Resample those blocks, not the rows. `evalforge stats --resample-unit block` already does.

---

## 5. Verification is integrity, not authenticity

```bash
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0
```

Ten checks, using nothing but the release directory: the manifest parses; every hashed artifact
exists; every file hash matches; the example count matches; `content_id` and `eval_id` recompute
from `examples.jsonl`; the split files partition it exactly; each split file's rows carry the
split they are filed under and agree with `examples.jsonl`; `audit.json` parses and names this
release. Failures are collected rather than raised, so one pass reports every defect. Passing
checks are printed too — a verifier that only prints failures cannot be told apart from one that
did nothing.

**What this does not do.** `manifest.json` cannot hash itself. Anyone able to rewrite the data can
rewrite the digests that describe it. The check detects a truncated download, a stray editor save,
a split file that drifted out of sync with `examples.jsonl` — it does not detect an adversary.
Signing is out of scope and out of the dependency budget. `tests/test_release.py` asserts the hole
rather than leaving it implied: a manifest rewritten consistently verifies clean.

---

## 6. Honest negatives

- **Every count in a release is a lower bound.** `content_id` is exact; `block_count`,
  `audit.json`'s duplicate clusters, and the leakage flags all inherit the near-duplicate stage's
  lexical recall bound. `ret-001`/`ret-002` is a paraphrase pair no threshold reaches, and it is
  invisible here too.
- **`eval_id` is sensitive to a re-split and blind to a re-scoring.** It says two numbers are
  incomparable when the arrangement moved. It cannot say whether the *metric* changed, because a
  dataset does not carry one.
- **The `minor` bump is about scores, not validity.** See §3.
- **The delta's join key is `example_id`.** A curator who renumbers ids gets `removed` + `added`
  for every row and a `major` bump. Conservative, and noisy.
- **The demo release's test split holds two examples.** `qa_toy` has 14, `--quarantine-blocking`
  moves four unadjudicated ones to `train`, and 20% of the remainder rounds to two. The release
  warns that its test split cannot measure `hard`. It is a format demonstration, not a benchmark;
  Milestone 9's small-N arithmetic applies with force.
- **`--drop near` can delete a real example.** The near stage has false positives — on
  `agent-003`/`agent-004` the "false positive" is itself disputed (Milestone 5). The flag exists,
  defaults off, and records what it took.

---

## 7. Reproduce

```bash
# Build the committed demo release (byte-identical to what is in git).
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset-id qa_toy --strategy adversarial --seed 0 --quarantine-blocking \
  --created-at 2026-07-09T00:00:00+00:00 \
  --license-notes "Synthetic fixture data, CC0-1.0. No real personal, clinical, or patient data." \
  --version 0.1.0 --out datasets/releases/qa_toy_v0.1.0

# Check it without rebuilding it.
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0

# A re-split at the same content: content_id holds, eval_id moves, the bump is major.
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset-id qa_toy --seed 7 --version auto \
  --previous datasets/releases/qa_toy_v0.1.0 --out /tmp/qa_toy_next

# The measurement behind all of it.
python3 scripts/split_identity_experiment.py --seeds 400

python3 -m pytest -q tests/test_release.py    # 72 tests
```
