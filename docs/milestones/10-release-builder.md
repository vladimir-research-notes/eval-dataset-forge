# Milestone 10 — Release builder

## Summary

Land the release layer: `examples.jsonl`, `manifest.json`, `audit.json`, `splits/*.jsonl`,
`REPRODUCE.md`, content hashes, a versioned release directory, and the `evalforge release` /
`evalforge verify-release` commands.

The distinguishing work was not the packaging. It was (a) noticing that a file hash answers the
wrong question — it changes when the dataset does not, and holds still when the dataset's
*meaning* moves; (b) measuring, on this repository's own corpus, that the split seed moves the
reported headline number with a standard deviation of **0.130** against a top-two system gap of
**0.036**, so that under `random` the test split reproduces the corpus's own ordering **48.7%** of
the time — a coin flip; (c) concluding from that measurement, rather than from taste, that a
release needs **two** identities and that a pure re-split is a `major` version bump; and (d)
refusing to write `effective_n` into a manifest, which answers the question Milestone 9 left open.

## Deliverables

- `src/eval_dataset_forge/datasets/` gains the four modules `PLAN.md` §4 specifies:
  - `writer.py` — `canonical_bytes()` (sorted-key, compact, UTF-8 — the *hashed* form),
    `example_line()` / `render_jsonl()` / `render_json()` (schema-order, human-readable — the
    *written* form), `sha256_bytes()`, `sha256_file()`, `write_bytes()`, `write_examples()`.
  - `versioning.py` — `example_digest()`, `content_id()`, `eval_id()`, `digests_by_id()`,
    `parse_version()`, `next_version()`, `ReleaseDelta`, `compare()`, `format_delta()`.
  - `manifest.py` — `build_manifest()`, `load_manifest()`, `load_audit()`, `VerifyReport`,
    `verify_release()` (ten checks), `format_verify()`.
  - `release.py` — `Exclusion`, `plan_exclusions()`, `build_audit()`, `render_reproduce()`,
    `build_release()`, `write_release()`, `ReleaseReport`, `format_report()`.
- `schemas.py` — `DatasetManifest` gains `content_id`, `eval_id`, `source_content_id`,
  `tool_version`, `split_spec`, `excluded`, `dependence` (additive, with defaults; old records
  still load; `SCHEMA_VERSION` unchanged).
- `cli.py` — `evalforge release` (repeatable `--dataset`, `--dataset-id`, `--version` incl.
  `auto`, `--previous`, `--strategy`, `--seed`, `--ratios`, `--stratify-by`, `--drop`,
  `--quarantine-blocking`, `--ignore-answer-gate`, `--created-at`, `--license-notes`, `--out`,
  `--json`, `--fail-on-leakage`) and `evalforge verify-release` (`--release`, `--json`).
- `scripts/split_identity_experiment.py` — the reproducible harness behind the headline.
- `datasets/releases/qa_toy_v0.1.0/` — a committed demo release; its own `REPRODUCE.md` command
  regenerates it byte-for-byte.
- `tests/test_release.py` — 72 tests.
- Docs: new `docs/RELEASES.md`; `docs/RESULTS.md` gains its release section; `docs/LIMITATIONS.md`
  gains three sections; `docs/DATASET_SCHEMA.md` documents the new manifest fields and the release
  layout; `docs/REPRODUCIBILITY.md` gains the two commands and the determinism argument. Website:
  new `site/releases.html`; status updates to `milestones.html`, `index.html`,
  `audit-pipeline.html`, `results.html`, `limitations.html`, `schema.html`, and the nav on every
  page.

## Design decisions

- **Three identities, because they answer three questions.** `hashes[<file>]` — *did these bytes
  survive the trip?* `content_id` — *are these the same examples?* (order-independent, `split`
  excluded). `eval_id` — *are two reported scores comparable?* (the same digest with each
  example's split folded in). A manifest with only the first cannot support any comparison at all.
- **Canonicalization is not serialization.** The bytes that get hashed sort their keys and strip
  whitespace; the bytes that get written preserve schema field order because a human reads them.
  Hashing the presentation form would make a dataset's identity depend on how its writer felt
  about field order.
- **`eval_id` hashes the assignment, not the strategy.** Two builds that reach the same split by
  different seeds share an `eval_id`, and should — their numbers are comparable. `split_spec` is
  provenance.
- **Multiplicity counts.** `content_id` sorts the per-example digests; it does not deduplicate
  them. A corpus listing an example twice is not the corpus listing it once, which is the entire
  premise of Milestone 4.
- **Domain separation.** Each aggregate binds a purpose tag *and* `SCHEMA_VERSION`, so a content
  id can never be read as an eval id and a schema bump changes every id by construction.
- **The bump rule keys on the evaluation splits.** `major` for anything `dev`∪`test` can see;
  `minor` for train-only content changes; `patch` for packaging; `none` for the clock. Derived
  from the two example sets, never from the previous manifest's stored ids — a summary that
  disagrees with its data is `verify-release`'s problem.
- **A pure re-split is `major`.** The one verdict a curator finds surprising, and the reason the
  400-seed experiment exists.
- **No `0.x` special case.** The usual "below 1.0 anything goes" convention collapses `major` and
  `minor` exactly on the datasets most likely to be re-cut weekly. `0.1.0` + `major` = `1.0.0`.
- **`created_at` reaches no hashed artifact.** `REPRODUCE.md` carries no timestamp; the clock
  lives in `manifest.json`, the only file nothing hashes. Without this, `none` would be
  unreachable and `patch` meaningless.
- **The version cannot influence the bump that chose it.** `audit.json` and `REPRODUCE.md` embed
  the version string, so the candidate manifest used for the delta is assembled at the *previous*
  version. Otherwise every rebuild would report at least a `patch` on hash churn alone.
- **The audit describes the data that shipped.** Exclusions first, then the dedup/leakage/label/
  difficulty ladder runs *again* over the survivors. Dropping one member of an exact pair
  dissolves the cluster; a report computed before the drop describes a defect the release does not
  have.
- **Deletion is recorded, never silent.** `manifest.excluded` names every removed example, its
  reason, and the id of the member kept; `source_content_id` pins the pre-exclusion corpus. This
  keeps the promise made in `splits/__init__.py`'s module docstring at Milestone 7.
- **The survivor of a cluster is the first member in input order.** A convention, not a judgement
  — a later member may carry richer metadata. Both ids are recorded, so it is reversible.
- **Nothing is dropped by default,** and `--drop near` (which deletes on a lexical heuristic with
  known false positives) is offered rather than imposed.
- **`manifest.json` is excluded from `hashes`.** A file cannot contain its own digest. Said out
  loud in the module, in `REPRODUCE.md`, in `verify-release`'s output, and in a test.
- **`verify_release` collects problems instead of raising,** so one pass reports every defect, and
  prints the checks that *passed* — a verifier that only prints failures is indistinguishable from
  one that did nothing.
- **The manifest records block structure, not `effective_n`.** See below.

## Deviations from the design doc

- **A CLI subcommand pair was added.** `PLAN.md` §8 lists `build-splits --out
  datasets/releases/qa_toy_v0.1/`, which conflates constructing splits with publishing a release.
  `build-splits` keeps its job (construct + verify, writing `splits/` and `split_report.json`) and
  `release` publishes. `verify-release` is additive and is what makes the manifest worth writing:
  a reader who re-runs the pipeline has only proved the pipeline is deterministic.
- **Versions are `MAJOR.MINOR.PATCH`.** `PLAN.md` §8 writes `qa_toy_v0.1`. A two-component version
  cannot express a patch, and this milestone's whole subject is what a version communicates.
- **`release.py` is a fifth module** beyond the four `PLAN.md` §4 lists for `datasets/`. It holds
  the orchestration and is deliberately **not** re-exported from `datasets/__init__.py`: it
  imports every audit stage, and `from eval_dataset_forge.datasets import loader` — which the CLI
  does for its cheap subcommands — must not drag the dedup ladder and numpy into the process to
  read a JSONL file.
- **`DatasetManifest` gained seven fields.** Additive, with defaults, so `SCHEMA_VERSION` is
  unchanged and pre-existing manifests load. Recorded in `docs/DATASET_SCHEMA.md`.
- **`--dataset` is repeatable,** matching `stats`. A benchmark assembled from several files is the
  normal case for a release.
- **`audit.md` and `dataset_card.md` are not written.** They belong to Milestones 12 and 11.
  Rather than leave the reader to notice, every `REPRODUCE.md` says which two artifacts the bundle
  is missing and which milestone lands them.

## The result worth reading twice

**Two releases can hold identical data and produce incomparable numbers, and the difference is
larger than the effect the benchmark exists to measure.**

Hold the corpus fixed. Hold each system's per-example score vector fixed — generated once, never
re-drawn. Vary only the split seed. Anything that moves is caused by the split, which is precisely
what a content hash cannot see. Pooled corpus: 56 examples, 44 independent blocks, 3 simulated
systems, 400 seeds per strategy.

| quantity | value |
|---|---|
| distinct `content_id` across all 400 seeds × 3 strategies | **1** |
| full-corpus accuracy, `sys_a` / `sys_b` | 0.6964 / 0.6607 |
| **the gap the benchmark exists to resolve** | **+0.0357** |
| `sys_a` test accuracy under `random` | mean 0.7011, **sd 0.1300**, range **[0.364, 1.000]** |
| P(test reproduces the corpus ordering of the top two) | **0.487** |
| P(the top two tie on test) | 0.172 |
| leader by seed | `sys_a` 186 · `sys_b` 128 · `sys_c` 8 · tie 78 |

The split seed is worth **3.6× the effect the benchmark measures**, and which system "wins" is
decided by the seed rather than by the systems. The mean over seeds recovers the corpus number
(0.7011 vs 0.6964), so this is variance, not bias — an honest estimator of a quantity 11 examples
cannot pin down.

That is Milestone 9's problem. Milestone 10's problem is the corollary: **a release identified only
by its content would call all 400 of those releases the same.** So the manifest carries both ids,
and the version bump keys on `eval_id`.

## A claim the measurements killed

The first draft of `RELEASES.md` said the `adversarial` strategy produces a *better* test split,
and pointed at its agreement rate: it reproduces the corpus ordering 65.0% of the time against
`random`'s 48.7%. The sub-population accuracies say otherwise.

| estimand | `sys_a` | `sys_b` | gap |
|---|---|---|---|
| whole corpus (n=56) | 0.6964 | 0.6607 | +0.0357 |
| singletons only (n=36) | 0.6667 | 0.6389 | +0.0278 |
| duplicate-cluster examples only (n=20) | 0.7500 | 0.7000 | +0.0500 |

| strategy | mean test gap (400 seeds) | sd |
|---|---|---|
| `random` | +0.0409 | 0.1890 |
| `stratified` | +0.0504 | 0.1648 |
| `adversarial` | **+0.1029** | 0.1452 |

`adversarial` evicts duplicate clusters from `dev`/`test`, and the duplicated examples are not a
random subset — both systems score higher on them. Its test set therefore measures a
sub-population whose **true gap is nearly three times larger**, and a larger true effect is easier
to detect. Its higher agreement rate is a consequence of a *different estimand*, not of a
lower-variance estimate of the same one. The tool records `split_spec` and `eval_id` so a reader
can tell which question a number answers. It does not rank the strategies.

## The question Milestone 9 left open

> *"It will need to decide whether a manifest records `effective_n` alongside `example_count`."*

**No.** `n_eff = n / (1 + (m̄ − 1)·ICC)`, and the ICC is a property of a *scored comparison* — it
needs system scores, which a dataset does not have. A manifest quoting an `effective_n` would be
quoting an assumed ICC and passing it off as a property of the data.

So `manifest.dependence` records exactly what the computation needs and stops:
`block_count: 11`, `block_size_histogram: {1: 9, 2: 1, 3: 1}`, `duplicate_block_count: 2`,
`near_method: tfidf`, and a note pointing at `--resample-unit block`. Milestone 9's other open
question — whether `audit.json` carries `BootstrapResult` records — answers itself the same way:
it does not, because a bootstrap result is a statement about systems, and a release audit is a
statement about data.

## Honest negatives

- **`verify-release` detects accident, not an adversary.** `manifest.json` cannot hash itself, so
  anyone able to rewrite the data can rewrite the digests describing it. Signing is out of scope
  and out of the dependency budget. `test_verify_release_cannot_detect_a_rewritten_manifest`
  asserts the hole rather than leaving it implied.
- **Every count in a release except the ids is a lower bound.** `block_count`, the duplicate
  clusters, and the leakage flags all inherit Milestone 4's lexical recall bound.
  `ret-001`/`ret-002` is invisible here too.
- **The `minor` bump is about score comparability, not benchmark validity.** A train-only addition
  cannot move a fixed system's test score, but it *can* introduce train→test leakage that changes
  what that score measures for a system trained on the release. Every `minor` emits a warning
  saying so; the delta classifies and does not re-audit.
- **The delta joins on `example_id`.** Renumbering ids without changing content reads as a full
  `removed` + `added` and forces a `major`. Conservative, and noisy.
- **`eval_id` is blind to a change of metric.** It can say the arrangement moved. It cannot say
  the scoring function did, because a dataset does not carry one.
- **The committed demo release has a two-example test split.** `qa_toy` holds 14 examples,
  `--quarantine-blocking` moves four unadjudicated ones to `train`, and 20% of the remainder
  rounds to two. The release warns that its test split cannot measure `hard`. It demonstrates the
  format; it is not a benchmark, and Milestone 9's small-N arithmetic applies with force.
- **`--drop near` can delete a real example.** The near stage has false positives — on
  `agent-003`/`agent-004` the "false positive" is itself disputed (Milestone 5).
- **The experiment's system scores are synthetic.** No model runs offline. The block structure and
  the split strategies are real, and the claim is about the *estimator* rather than about any
  model, so the simulator is the right instrument: it is the only way to know the corpus-level
  truth each test split is trying to estimate. Nothing here is a statement about any real system.

## Validation

```bash
python3 -m pytest -q                          # 1595 passed (1523 prior + 72 release)
python3 -m pytest -q tests/test_release.py    # 72 passed
python3 scripts/validate_fixtures.py          # all 5 fixtures valid, exit 0
python3 scripts/make_demo_scores.py --check   # fixture matches its generator, exit 0
ruff check .                                  # all checks passed

# The committed demo release, and the check that needs nothing but the release (exit 0):
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0

# Same content, different seed: content_id holds, eval_id moves, the bump is major (exit 0).
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset-id qa_toy --seed 7 --version auto \
  --previous datasets/releases/qa_toy_v0.1.0 --out /tmp/qa_toy_next

# A version below the bump the content requires is refused (exit 2).
PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --dataset-id qa_toy --seed 7 --version 0.1.1 \
  --previous datasets/releases/qa_toy_v0.1.0 --out /tmp/qa_toy_stale

# The measurement behind eval_id.
python3 scripts/split_identity_experiment.py --seeds 400
```

Verified end to end, not merely unit-tested: the committed release passes all ten checks; the
command printed inside its own `REPRODUCE.md` regenerates it byte-for-byte; rebuilding it under
`PYTHONHASHSEED` ∈ {1, 2, 12345, random} in four separate interpreters yields identical bytes in
all seven artifacts; two builds at different clocks differ in `manifest.json` and nothing else; a
single flipped byte in `examples.jsonl` trips three independent checks; and dropping an exact
duplicate reduces the released `audit.json`'s `exact_duplicates` count to zero rather than
reporting the source corpus's.

## Open questions for review

- **Is `major` too blunt for an *addition* to `test`?** Adding ten test examples does not
  invalidate an old score on the old test set; it makes the two numbers describe different sets.
  Today both are `major`. A separate level ("superset") would let a reader know an old score is
  still valid *on the subset*. It would also be a level nobody knows how to act on.
- **Should `--drop near` exist?** It deletes real data on a heuristic this repo has already
  measured to have disputed false positives. It is off by default and fully recorded, but nothing
  stops a curator reaching for it because it makes the audit look clean.
- **`content_id` covers `metadata` and `provenance`.** Editing a note in `metadata` changes the
  content id, hence a `minor` (or `major`, if the example is in `dev`/`test`). Arguably a comment
  is not data. Arguably `metadata.retrieval.docs` is very much data, and no rule distinguishes
  them without a schema that says which keys are load-bearing.
- **The delta reads the previous release from disk.** A release published elsewhere and never
  fetched cannot be compared against, so `--version auto` is unavailable exactly when the previous
  release is hardest to obtain. A manifest-only comparison would need `example_digests` in the
  manifest, which is O(n) and would make the manifest scale with the dataset.
- **Should `verify-release` re-run the audit?** It checks that `audit.json` names the release; it
  does not check that the findings are *correct* for the data. Doing so would make verification
  cost a full audit and would silently re-run heuristics whose thresholds may have moved between
  tool versions — turning a byte-integrity check into a claim about method stability.
- **Is a two-example test split an acceptable demo?** It is the honest output of the documented
  defaults on a 14-example fixture, and the release says the split cannot measure `hard`. It is
  also an unflattering advertisement for the format.

## Status

Complete. Next: Milestone 11 — Dataset card generator (Markdown + optional HTML cards). It should
render `DatasetCard` from the manifest and `audit.json` so the card's "known issues" cannot drift
from what the pipeline found, and it will need to decide whether the card prints `block_count`
next to `example_count` — a card that says "14 examples" when the corpus holds 11 independent
blocks is not lying, but it is not helping.
