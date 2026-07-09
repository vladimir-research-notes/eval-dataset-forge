# Human review: Milestone 10 — Release builder

## TL;DR

A release is now a versioned, verifiable bundle: `examples.jsonl`, `manifest.json`, `audit.json`,
`splits/*.jsonl`, and a `REPRODUCE.md` whose own command regenerates the bundle byte-for-byte.
`evalforge release` builds it; `evalforge verify-release` checks it with nothing but the release
itself.

The packaging was the easy half. The milestone's actual finding is that **a file hash answers the
wrong question**. It changes when the dataset does not (reorder rows, rebuild on a different clock)
and holds still when the dataset's *meaning* moves, because re-cutting the splits leaves every byte
of the corpus in place. A 400-seed experiment on the pooled fixtures — corpus fixed, each system's
per-example score vector fixed, only the split seed varying — shows the reported headline number
moving with **sd 0.130** against a top-two system gap of **0.036**. Under `random` the test split
reproduces the corpus's own ordering **48.7%** of the time: a coin flip. All 400 releases share one
`content_id`.

So a release carries **two** identities — `content_id` (*are these the same examples?*) and
`eval_id` (*are two reported scores comparable?*) — and a pure re-split is a `major` version bump.
That last verdict is the one a curator will find surprising, and it is the reason the experiment
exists.

It also answers the question Milestone 9 left open: **the manifest does not record `effective_n`.**
`n_eff = n / (1 + (m̄ − 1)·ICC)`, and an ICC is a property of a *scored comparison*, not of a
dataset. A manifest quoting one would be quoting an assumed ICC and passing it off as a property of
the data. `manifest.dependence` records the block structure the computation needs, and stops.

An early draft of `RELEASES.md` claimed the `adversarial` strategy produces a *better* test split,
citing its 65.0% agreement rate against `random`'s 48.7%. **The sub-population accuracies killed
the claim.** `adversarial` evicts duplicate clusters from `dev`/`test`, and both systems score
higher on duplicated examples (0.750 / 0.700) than on singletons (0.667 / 0.639), so its test set
measures a sub-population whose true gap is nearly three times larger (+0.1029 vs +0.0357). A larger
true effect is easier to detect. Its agreement rate is a consequence of a *different estimand*, not
of a lower-variance estimate of the same one. The tool records `split_spec` and does not rank the
strategies.

The reviewer's job here is mostly to decide whether `major`-on-re-split is a rule curators will
actually follow rather than route around, and whether `--drop near` should exist at all.

## What changed

- `src/eval_dataset_forge/datasets/` gains four modules:
  - `writer.py` — `canonical_bytes()` (sorted-key, compact — the *hashed* form) vs
    `example_line()` / `render_jsonl()` (schema-order — the *written* form); `sha256_bytes()`,
    `sha256_file()`, `write_examples()`.
  - `versioning.py` — `example_digest()`, `content_id()`, `eval_id()`, `parse_version()`,
    `next_version()`, `ReleaseDelta`, `compare()`, `format_delta()`.
  - `manifest.py` — `build_manifest()`, `load_manifest()`, `verify_release()` (ten checks),
    `format_verify()`.
  - `release.py` — `plan_exclusions()`, `build_audit()`, `render_reproduce()`, `build_release()`,
    `write_release()`. A fifth module beyond the four `PLAN.md` §4 lists, deliberately **not**
    re-exported from `datasets/__init__.py`: it imports every audit stage, and the CLI's cheap
    subcommands must not drag the dedup ladder and numpy into the process to read a JSONL file.
- `schemas.py`: `DatasetManifest` gains `content_id`, `eval_id`, `source_content_id`,
  `tool_version`, `split_spec`, `excluded`, `dependence` — additive, with defaults, so
  `SCHEMA_VERSION` is unchanged and older manifests still load.
- `cli.py`: `evalforge release` (repeatable `--dataset`, `--version` incl. `auto`, `--previous`,
  `--strategy`, `--seed`, `--ratios`, `--stratify-by`, `--drop`, `--quarantine-blocking`,
  `--created-at`, `--license-notes`, `--json`, `--fail-on-leakage`) and `evalforge verify-release`.
- `scripts/split_identity_experiment.py` — the reproducible harness behind the headline.
- `datasets/releases/qa_toy_v0.1.0/` — a committed demo release (the one artifact under
  `datasets/releases/` that `CLAUDE.md` permits).
- `tests/test_release.py`, 72 tests.
- Docs: new `docs/RELEASES.md`; release sections in `docs/RESULTS.md`; three new sections in
  `docs/LIMITATIONS.md`; new manifest fields and release layout in `docs/DATASET_SCHEMA.md`; the two
  new commands and the determinism argument in `docs/REPRODUCIBILITY.md`.
- Website: new `site/releases.html`; a `Releases` nav link on all 16 pages; status updates on
  `milestones.html`, `index.html` (whose status badge had also drifted, still reading "Milestone 8"),
  `audit-pipeline.html`, `results.html`, `limitations.html`, `schema.html`.

## What should be checked later

1. **Is `major`-on-re-split a rule people will follow, or route around?** It is correct — a
   re-split makes old and new scores incomparable — but it means a curator who re-cuts weekly
   burns a major version weekly, and `0.1.0` + `major` = `1.0.0` with no `0.x` grace period. The
   rule's correctness and its adoptability are different questions and only the first has been
   settled. **This is the milestone's most consequential unexamined assumption.**
2. **Should `--drop near` exist?** It deletes real data on a lexical heuristic this repo has already
   measured to have false positives — and on `agent-003`/`agent-004` the "false positive" is itself
   disputed (Milestone 5). It is off by default and every deletion is recorded in
   `manifest.excluded`. Nothing stops a curator reaching for it because it makes the audit look
   clean, which is precisely the failure mode this project exists to surface.
3. **Is `major` too blunt for an *addition* to `test`?** Adding ten test examples does not
   invalidate an old score on the old test set; it makes the two numbers describe different sets. A
   separate "superset" level would tell a reader the old score is still valid on the subset. It
   would also be a level nobody knows how to act on.
4. **`content_id` covers `metadata` and `provenance`.** Editing a note in `metadata` changes the
   content id, hence at least a `minor`. Arguably a comment is not data; arguably
   `metadata.retrieval.docs` very much is. No rule distinguishes them without a schema that declares
   which keys are load-bearing.
5. **The delta reads the previous release from disk**, so `--version auto` is unavailable exactly
   when the previous release is hardest to obtain. A manifest-only comparison would need
   `example_digests` in the manifest — O(n), making the manifest scale with the dataset.
6. **Should `verify-release` re-run the audit?** It checks that `audit.json` names the release; it
   does not check that the findings are *correct* for the data. Doing so would make verification
   cost a full audit and would silently re-run heuristics whose thresholds may have moved between
   tool versions — turning a byte-integrity check into a claim about method stability.
7. **Is a two-example test split an acceptable demo?** `qa_toy` holds 14 examples,
   `--quarantine-blocking` moves four unadjudicated ones to `train`, and 20% of the remainder rounds
   to two. It is the honest output of the documented defaults, and the release warns that its test
   split cannot measure `hard`. It is also an unflattering advertisement for the format, and
   Milestone 9's small-N arithmetic applies with force.
8. **The survivor of a duplicate cluster is the first member in input order.** A convention, not a
   judgement — a later member may carry richer metadata. Both ids are recorded, so it is reversible,
   but nothing warns that the choice was arbitrary.

## Technical risks

- **`verify-release` detects accident, not an adversary — and this was confirmed by
  construction, not merely asserted.** `manifest.json` cannot contain its own digest. Rewriting
  `examples.jsonl` *and* recomputing the hashes, `content_id`, and `eval_id` with the tool's own
  functions produces a release that `verify-release` reports as internally consistent, exit 0. The
  only thing that catches it is a reader who already knows the published `content_id`. Signing is
  out of scope and out of the dependency budget.
  `test_verify_release_cannot_detect_a_rewritten_manifest` asserts the hole rather than leaving it
  implied. **A reader who takes a passing `verify-release` as evidence of authenticity is being
  misled**, and the CLI's own success line says so.
- **Every count in a release except the ids is a lower bound.** `block_count`, the duplicate
  clusters, and the leakage flags all inherit Milestone 4's lexical recall bound;
  `ret-001`/`ret-002` is invisible here too. A `dependence` block that under-reports clustering
  makes Milestone 9's intervals *narrower* — the direction nobody notices.
- **The version cannot influence the bump that chose it**, and this is load-bearing and subtle.
  `audit.json` and `REPRODUCE.md` embed the version string, so the candidate manifest used for the
  delta must be assembled at the *previous* version; otherwise every rebuild reports at least a
  `patch` on hash churn alone. A future refactor that "simplifies" this ordering would reintroduce
  the bug silently, and the only thing standing in the way is a test.
- **`created_at` must reach no hashed artifact.** It lives only in `manifest.json`, the one file
  nothing hashes. Without that, `none` would be unreachable and `patch` meaningless. Any future
  artifact that embeds a timestamp breaks the bump lattice.
- **The delta joins on `example_id`.** Renumbering ids without changing content reads as a full
  removal plus addition and forces a `major`. Conservative, and noisy.
- **`eval_id` is blind to a change of metric.** It can say the arrangement moved. It cannot say the
  scoring function did, because a dataset does not carry one. Two releases with the same `eval_id`
  scored under different metrics are not comparable, and nothing in the manifest will say so.
- **The `minor` bump is a claim about scores, not about validity.** A train-only addition cannot
  move a fixed system's test score, but it *can* introduce train→test leakage that changes what
  that score measures for a system trained on the release. Every `minor` emits a warning; the delta
  classifies and does not re-audit.
- **The experiment's system scores are synthetic.** No model runs offline. The block structure and
  the split strategies are real, and the claim is about the *estimator* rather than any model — the
  simulator is the only instrument that knows the corpus-level truth each test split is estimating.
  Nothing in `RELEASES.md` is a statement about any real system.

## Commands to verify

```bash
python3 -m pytest -q                          # 1595 passed (1523 prior + 72 release)
python3 -m pytest -q tests/test_release.py    # 72 passed
python3 scripts/validate_fixtures.py          # all 5 fixtures valid, exit 0
python3 scripts/make_demo_scores.py --check   # fixture matches its generator, exit 0
ruff check .                                  # all checks passed

# The committed demo release passes ten checks using nothing but itself (exit 0).
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

# The measurement behind eval_id (~1 min).
python3 scripts/split_identity_experiment.py --seeds 400
```

The claims most worth reproducing by hand:

```bash
# 1. The REPRODUCE.md command regenerates the committed release byte-for-byte, and the
#    bytes do not depend on the interpreter's hash seed.
PYTHONHASHSEED=999 PYTHONPATH=src python3 -m eval_dataset_forge release \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --dataset-id qa_toy \
  --strategy adversarial --seed 0 --ratios 0.7,0.1,0.2 --stratify-by task_family \
  --quarantine-blocking --version 0.1.0 --created-at 2026-07-09T00:00:00+00:00 \
  --license-notes 'Synthetic fixture data, CC0-1.0. No real personal, clinical, or patient data.' \
  --out /tmp/rel_check
diff -r datasets/releases/qa_toy_v0.1.0 /tmp/rel_check && echo identical

# 2. One flipped byte in examples.jsonl trips three independent checks (exit 1).
python3 -m pytest -q tests/test_release.py -k "flipped or tamper or hashes_match"

# 3. The hole is asserted, not hidden: a fully rewritten manifest verifies clean.
python3 -m pytest -q tests/test_release.py -k "cannot_detect_a_rewritten_manifest"

# 4. content_id ignores the split; eval_id does not.
python3 -m pytest -q tests/test_release.py -k "content_id or eval_id"
```

## Human feedback

<!-- Human will fill this later. -->
