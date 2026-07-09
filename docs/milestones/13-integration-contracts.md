# Milestone 13 — Integration contracts

Executable contracts for five downstream consumers — `code-agent-runtime`, `BioAgentBench`,
an LLM regression/observability suite, `scientific-claim-verifier`, and `agent-control-bench`
— plus `evalforge gate`, the command that turns one into an exit code. None of the five is a
dependency, and none exists in this repository.

## What landed

| file | role |
|---|---|
| `src/eval_dataset_forge/contracts/reader.py` | `ReleaseHandle`: the read-only consumption surface, and the **declared read set** — the only manifest/audit keys this package takes |
| `src/eval_dataset_forge/contracts/requirements.py` | 14 named requirements over three verdicts, each carrying the claim its pass does *not* license |
| `src/eval_dataset_forge/contracts/gate.py` | `GatePolicy`, `GateReport`, `CONTRACT_VERSION`, exit-code semantics, and the `content_id` scope check |
| `src/eval_dataset_forge/contracts/profiles.py` | the five consumer profiles, each with what it `assumes` and what it `does_not_check` |
| `src/eval_dataset_forge/contracts/report.py` | the report as a `Document`, rendered by Milestone 11's two backends |
| `tests/test_contracts.py` | 50 tests, of which the load-bearing ones are the four measurements below |

`evalforge gate --release <dir> --profile <name>` is the thirteenth subcommand. Exit 0 pass,
1 fail, 2 usage error.

## The question this milestone had to settle

An audit has no opinion. It reports that `qa-007` quotes its gold answer inside its own prompt
and leaves the reader to decide whether that disqualifies the dataset. A gate must decide. So:
**where does the opinion live, and what is it an opinion about?**

The answer to the first half is easy — with the consumer, in a reviewable `GatePolicy`, never
in the release. A release that carried its own pass/fail verdict would be a dataset grading
itself. The second half is where the work was, and it produced four results — the last of them
a bug in this milestone's own first draft, found by checking a docstring rather than shipping
it.

### 1. A gate needs a third verdict

Milestone 12 established that on an unsplit corpus four leakage signals return zero *by
vacuity*, and printed a warning beside them. A warning is advice. A boolean gate that maps
`leakage_flags == [] → PASS` silently converts vacuity into permission. So `Verdict` has three
values, `UNDECIDABLE` never counts as a pass, and a policy that proceeds anyway must set
`allow_undecidable: true` where a reviewer can see it. Milestone 12's sentence became an exit
code.

Two requirements reach it, for two different reasons that must not be conflated:

- **`no_retrieval_contamination` on `qa_toy`** — no example carries `metadata.retrieval.docs`,
  so the check inspected nothing. Decidable on `retrieval_toy`. *A fact about the corpus.*
- **`difficulty_is_calibrated` on every release this tool has ever built** — `audit.json`
  records `difficulty_distribution` (three bucket counts) and drops the permutation test that
  says whether those buckets mean anything. *A fact about the release format.* Milestone 11's
  review note already asked whether the field should exist; this does not discover the gap, it
  makes it **executable** and adds a second cost — besides forcing the dataset card to print an
  unconditional caveat, it denies every consumer a verdict. The requirement is in the registry,
  in **no** profile, and a test asserts no profile can pick it up.

The general form: **`audit.json` records what each check found, never what it was given.** Zero
findings over zero passages and zero findings over a thousand clean passages serialize
identically. That is why `no_retrieval_contamination` is the one requirement that reads
`examples.jsonl` — the evidence bundle cannot answer the question it asks.

### 2. The "optional accelerator" is a different detector

`docs/PLAN.md` §3 lists `scikit-learn` as an optional accelerator with a pure-Python fallback.
It is not an accelerator. `resolve_near_method("auto")` returns `tfidf` iff scikit-learn is
importable and `jaccard` otherwise; the two disagree about which examples are duplicates;
duplicates become blocks; the adversarial split will not cut through a block; and `eval_id`
folds the split in.

Rebuilding `bio_toy` from identical bytes, an identical command, and an identical seed:

| | scikit-learn present | scikit-learn absent |
|---|---|---|
| `near_method` | `tfidf` | `jaccard` |
| near-duplicate clusters | 1 (`bio-001` ≈ `bio-002`) | 0 |
| independent blocks | 8 | 9 |
| `content_id` | *identical* | *identical* |
| `eval_id` | **different** | **different** |
| held-out example set | **different** | **different** |

`content_id` holding is correct — the examples *are* the same examples. `eval_id` moving is
also correct, and is Milestone 10's identity design telling the truth: these are not the same
benchmark. What was false was `REPRODUCE.md`, which read *"deterministic given the same inputs
and seed"*. **The environment is an input.** `render_reproduce` now emits an `Environment`
section naming the method that ran, and `dependence.near_method` already recorded it.

The counterweight is what makes the contract usable: **`verify-release` runs no detector.** It
recomputes file hashes and both identities from bytes, so a release built with the accelerator
verifies at 10/10 on a machine without it. *Verification is environment-independent;
reproduction is not.* This also answers an open question Milestone 10 left on
`site/releases.html` — should `verify-release` re-run the audit? No: it would make integrity
depend on the verifier's `pip list`.

### 3. Two properties that do not hold under union

Both follow from idf being computed over the document collection, which Milestone 12 measured
on `sim(saf-004, saf-007)`: 0.2866 inside `safety_toy`, 0.3438 pooled, across a 0.30 threshold
with neither prompt changed. This milestone turns that observation into two consequences a
consumer can trip over.

**A gate verdict is not preserved under union.** `safety_toy` alone reports one near-duplicate
cluster and passes a `max_near_duplicate_clusters: 1` policy. Concatenated with the four other
fixtures — which share *not one duplicate* with it — the same ten rows report two clusters and
fail. `GateReport.applies_to()` compares `content_id` and `assert_applies_to()` raises.

**Independent-block count is not additive, although example count is.**

| the five fixtures, disjoint | examples | independent blocks |
|---|---|---|
| audited separately, summed (`tfidf`) | 56 | **45** |
| audited as one corpus (`tfidf`) | 56 | **44** |
| audited as one corpus (`jaccard`) | 56 | **46** |

The pooled corpus contains **zero cross-fixture duplicate edges**. Merging five unrelated
datasets found no duplicate *between* them and still changed one dataset's verdict *inside*
itself. Under `jaccard`, which has no idf, both counts agree — so **composability is a property
of the similarity function**, and the optional dependency is exactly what removes it. This
settles Milestone 12's open review question 7, which flagged the Jaccard path as "presumably
immune. **Unverified.**"

A regression suite that doubles its example count has therefore not necessarily bought `√2` of
anything, and `evalforge stats --resample-unit block` must be re-derived after any merge.

### 4. A verdict does not survive a re-split either — a bug this milestone wrote, then measured

The first draft of `applies_to` compared `content_id` alone, on the argument that no finding a
gate reads depends on the split assignment. Checking the argument rather than shipping it:
building `qa_toy` forty times with `--strategy random`, one seed per build, **all forty releases
share one `content_id`**, seeds 9 and 35 *pass* `eval_splits_deduplicated`, and the other
thirty-eight *fail* it. The first draft would have handed seed 9's pass to seed 0's release.

Four of the fourteen requirements read the split — `no_train_test_overlap`,
`no_retrieval_contamination`, `eval_splits_deduplicated`, `eval_split_is_rankable` — and are now
marked `split_sensitive`. `GateReport.scope` is `eval_id` when one of them ran and `content_id`
otherwise, derived from what **executed** rather than from `split_spec.strategy`: a strategy
field is a claim, not evidence. The adversarial strategy does hold all four at zero for every
seed, and a test pins that fact while the gate declines to depend on it.

`integrity` sits outside the scheme: it is a statement about the bytes of one directory, and two
releases can share both identities while differing in `created_at`. A report is never evidence
about a directory other than its own `release_dir`.

## The declared read set

A consumer that calls `json.load(open("manifest.json"))` depends on every key in the file,
including the ones it does not read, because nothing tells a reader which those are. Milestone
12 grew `split_statistics` by two keys under an unchanged `SCHEMA_VERSION`: legal, and
invisible. So `MANIFEST_FIELDS`, `AUDIT_FIELDS`, `DEPENDENCE_FIELDS`, `SPLIT_STAT_FIELDS`, and
`SPLIT_COUNT_FIELDS` name the entire surface, and tests assert each name exists both on the
pydantic model and in the committed demo release. The compatibility rule is then checkable:

> Adding a key is a `patch`. Removing or retyping a declared key is a `major`, and
> `CONTRACT_VERSION` moves with it.

## Results

**All five profiles reject the demo release**, which is the only correct answer: `qa_toy` is a
deliberately flawed fixture, and a gate that passed it would be broken. The positive control is
a `--drop exact,normalized,blocking-labels` release that passes, so `not report.ok` everywhere
does not merely prove the gate is capable of saying no.

| profile | verdict | pass / fail / undecidable | blocking |
|---|---|---|---|
| `agent-control-bench` | FAIL | 4 / 2 / 0 | duplicates, answer-in-prompt |
| `bioagentbench` | FAIL | 3 / 3 / 0 | duplicates, conflicting + missing labels |
| `code-agent-runtime` | FAIL | 3 / 3 / 0 | missing label, answer-in-prompt, duplicates |
| `llm-regression` | FAIL | 5 / 1 / 0 | `eval_split_is_rankable` — `test` holds **2 blocks**, not 30 |
| `scientific-claim-verifier` | FAIL | 2 / 3 / 1 | 3 findings; `no_retrieval_contamination` undecidable |

`llm-regression` reads `dependence.blocks_by_split`, never `example_count`. A duplicate cluster
contributes *m* rows and one unit's worth of information (Milestone 9), so a test split of 200
rows in 3 duplicate clusters is a test split of 3.

Full tables: [`docs/RESULTS.md`](../RESULTS.md). Full argument:
[`docs/INTEGRATION.md`](../INTEGRATION.md).

## The demo release was rebuilt in place, again

`REPRODUCE.md` carried a false determinism claim, so its bytes had to change. Milestone 10's
`compare()` classifies the result: `patch`, `0.1.0 → 0.1.1`, reasons `("packaging only:
hashes",)`, `content_id` / `eval_id` / `schema_version` all equal, zero examples added, removed,
changed, or reassigned, and exactly one artifact hash moved. A score measured on the old bundle
stands on this one.

The release was therefore rebuilt at v0.1.0 rather than re-versioned, following Milestone 12's
precedent and its reasoning (never distributed; a build artifact of the repo; regenerated by its
own `REPRODUCE.md`). `test_rewriting_reproduce_md_is_a_packaging_change_not_a_data_change` pins
the classification rather than leaving it in this note. The risk Milestone 12 flagged — git
history now holds three byte-sets called `qa_toy v0.1.0` — is compounded, not resolved, and is
raised again in the review note.

## What no contract here can supply

Whether the model under test was trained on the benchmark. Contamination is measurable only
against text the auditor holds, and nobody holds a frontier model's training set. Every leakage
check in this repository is intra-corpus. Also absent: construct validity, label correctness,
authenticity (`manifest.json` cannot hash itself), and coverage. Each is stated in the profile
that would otherwise be read as promising it, and every requirement prints, beside its verdict,
the claim its pass does **not** establish — because a gate report is read by whoever most wants
it to say the dataset is fine.

## Limitations

- **The five profiles are written from the outside.** None of the named projects has reviewed
  the contract written on its behalf. Each is a plausible reading of what that consumer needs,
  not an agreed interface.
- **`min_eval_blocks = 30` is a convention**, inherited from Milestone 9's `MIN_EFFECTIVE_N`. On
  a corpus with a 3.6-point margin, Milestone 9 measured that ≈4500 examples were needed.
- **A release records `near_method`; nothing in it can verify the record.** Verifying it means
  rebuilding, which needs the build environment.
- **Every count a gate reads is a lower bound**, inheriting the near stage's lexical recall
  bound. A `PASS` on `no_exact_duplicates` says the normalized strings differ.
- **The gate reads a release, not a dataset.** `evalforge audit` output is not gateable, by
  design: the requirements that matter are about what shipped.

## Commands

```bash
python3 -m pytest -q                                        # 1756 tests
python3 -m pytest -q tests/test_contracts.py                # 50 tests
ruff check .

PYTHONPATH=src python3 -m eval_dataset_forge gate --list
PYTHONPATH=src python3 -m eval_dataset_forge gate \
  --release datasets/releases/qa_toy_v0.1.0 \
  --profile llm-regression                                  # exit 1, by design

# The two properties that do not hold, and the dependency that removes them:
python3 -m pytest -q tests/test_contracts.py \
  -k "additive or union or optional_dependency or environment_independent"

# The rebuilt demo release still verifies and still regenerates its card.
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0                 # 10/10 checks
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check         # exit 0
```

## Next

Milestone 14 — Final portfolio report: bring README and docs to portfolio quality (problem,
architecture, example audit, detected issues, dataset-card example, reproduction commands,
limitations, research path).
