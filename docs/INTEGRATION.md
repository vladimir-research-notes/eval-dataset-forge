# Integration contracts: what a downstream project may assume

*Milestone 13. Implemented in `src/eval_dataset_forge/contracts/`.*

Five adjacent projects want to consume audited releases: `code-agent-runtime`,
`BioAgentBench`, an LLM regression/observability suite, `scientific-claim-verifier`, and
`agent-control-bench`. None of them is a dependency of this package and none of them exists
in this repository. What is committed here is the **shape** of the question each one asks —
executable, tested against the fixtures, and versioned.

Writing those five contracts turned up four properties of this pipeline that nothing in
Milestones 0–12 had stated. Two are defects in claims the repository was already making; one
was a defect in the contract API's own first draft, caught by checking a docstring rather
than shipping it. They are the substance of this document; the API is at the end.

---

## 1. An audit has no opinion. A gate does, and the opinion is not the dataset's

`audit.json` reports that `qa-007` quotes its gold answer inside its own prompt. It does not
say whether that is disqualifying, because the answer depends on who is asking. A retrieval
benchmark measuring passage ranking may not care. A code-agent runtime about to gate a
release on that benchmark cares a great deal.

So the threshold lives with the **consumer**, in a reviewable `GatePolicy`, never in the
release. A release that carried its own pass/fail verdict would be a dataset grading itself.

```bash
evalforge gate --release datasets/releases/qa_toy_v0.1.0 --profile code-agent-runtime
```

Exit 0 when the gate passes, 1 when it does not, 2 on a usage error.

### Three verdicts, not two

| verdict | meaning |
|---|---|
| `pass` | the check ran on data it could act on, and found nothing |
| `fail` | the check ran and found something |
| `undecidable` | the check could not run, or the release does not carry the evidence |

Milestone 12 established that on an unsplit corpus four leakage signals return zero *by
vacuity*, and printed a warning beside them. A warning is advice. A boolean gate that maps
`leakage_flags == [] → pass` silently converts vacuity into permission. `undecidable` is
therefore a distinct value, it never counts as a pass, and a policy that proceeds anyway
must set `allow_undecidable: true` where a reviewer can see it.

Two requirements reach `undecidable` on real releases, for two different reasons, and the
difference is worth keeping:

- **`no_retrieval_contamination` on `qa_toy`** — no example carries `metadata.retrieval.docs`,
  so the check inspected nothing. The check is fine; the corpus gave it no work. It is
  decidable on `retrieval_toy`. **This is a fact about the corpus.**
- **`difficulty_is_calibrated` on every release this tool has ever built** — Milestone 8 runs
  a permutation test asking whether the difficulty heuristic's ranking beats a random ranking
  of the declared tags. Milestone 10 renders the answer into an English sentence in
  `audit.recommendations`. `audit.json` keeps `difficulty_distribution` — three bucket counts
  — and drops the verdict. Buckets exist whether or not they mean anything, so a consumer
  reading that field cannot tell a calibrated heuristic from noise, and can recover the truth
  only by pattern-matching prose. **This is a fact about the release format.** Milestone 11's
  review note already asked whether `AuditResult` should carry the verdict; the contract does
  not discover the gap but makes it *executable*, and adds a second reason to close it —
  besides forcing the dataset card to print an unconditional caveat, it denies every consumer
  a verdict. The requirement ships in the registry, in no profile, and is not fixed here.

`audit.json` records what each check **found**. It does not record what each check was
**given**. Zero findings over zero passages and zero findings over a thousand clean passages
serialize identically.

---

## 2. Three identities, three consumer-side scopes

Milestone 10 gave a release three identities because a file hash answers the wrong question.
Each turns out to be the scope of exactly one downstream claim.

| identity | scopes | a consumer that ignores it |
|---|---|---|
| `content_id` | a **content-scoped gate verdict** | reuses a verdict on a corpus it was not computed on |
| `eval_id` | a **score**, and a **split-sensitive gate verdict** | compares two numbers measured on different held-out sets |
| `dependence.near_method` | a **rebuild** | cannot reproduce the release it depends on |

The third row is what this milestone had to discover; the second row turned out to scope more
than Milestone 10 assigned to it (§4).

### The "optional accelerator" is a different detector

`docs/PLAN.md` §3 calls `scikit-learn` an *optional accelerator* with a *pure-Python
fallback*. It is not an accelerator. `resolve_near_method("auto")` returns `tfidf` if and only
if scikit-learn is importable, and `jaccard` otherwise, and the two disagree about which
examples are duplicates. Duplicates become blocks; the adversarial split refuses to cut
through a block; `eval_id` folds the split in.

Rebuilding `bio_toy` from identical bytes, an identical command, and an identical seed:

| | scikit-learn present | scikit-learn absent |
|---|---|---|
| `near_method` | `tfidf` | `jaccard` |
| near-duplicate clusters | 1 (`bio-001`≈`bio-002`) | 0 |
| independent blocks | 8 | 9 |
| `content_id` | *identical* | *identical* |
| `eval_id` | **different** | **different** |
| held-out example set | **different** | **different** |

`content_id` holding is correct — the examples *are* the same examples. `eval_id` moving is
also correct, and it is the tool telling the truth: these are not the same benchmark. What
was false was `REPRODUCE.md`, which read *"deterministic given the same inputs and seed"*.
**The environment is an input.** Every release now records the method that actually ran, and
`REPRODUCE.md` carries an `Environment` section.

`bio-001`/`bio-002` clears the 0.30 threshold under idf weighting and not under raw set
overlap. Nothing pathological is required — one pair, one threshold.

### Verification is environment-independent; reproduction is not

`verify-release` recomputes every file hash, `content_id`, and `eval_id` from the release's
own bytes. It runs no detector. So a release built with scikit-learn verifies at 10/10 on a
machine without it, and integrity travels where reproducibility does not. This is the
property that makes the whole contract usable, and it is pinned by
`test_verification_is_environment_independent`.

---

## 3. Two properties that do not hold under union

Both follow from one fact: TF-IDF's idf term is computed over the document collection, so a
similarity is a property of a **pair and its corpus**. Milestone 12 measured this on
`sim(saf-004, saf-007)`: 0.2866 inside `safety_toy`, 0.3438 inside the pooled corpus,
straddling the 0.30 threshold, with neither prompt changed.

### A gate verdict is not preserved under union

`safety_toy` alone reports one near-duplicate cluster and **passes** a
`max_near_duplicate_clusters: 1` policy. Concatenate it with the four other fixtures — which
share **not one duplicate** with it — and the same ten rows report two clusters and **fail**.

Merging two gated releases yields an ungated corpus. `GateReport.applies_to()` compares
`content_id` and refuses to answer for any other; `assert_applies_to()` raises rather than
warns.

### Independent-block count is not additive, although example count is

The five fixtures, disjoint, sharing no example:

| | examples | independent blocks |
|---|---|---|
| sum of the five audited separately | 56 | **45** |
| the five audited as one corpus (`tfidf`) | 56 | **44** |
| the five audited as one corpus (`jaccard`) | 56 | **46** = 45 + 1 |

Two observations, both uncomfortable.

The pooled corpus contains **zero cross-fixture duplicate edges**. Merging five unrelated
datasets found no duplicate *between* them and still changed one dataset's verdict *inside*
itself: `{saf-004, saf-005}` grows to `{saf-004, saf-005, saf-007}`. Appending unrelated data
can move an untouched example's block, hence its split, hence the release's `eval_id`.

Under `jaccard` the counts are equal, and 46 rather than 45 because `jaccard` misses
`bio-001`≈`bio-002` entirely. Composability is a property of the **similarity function**, not
of this repository's data — and the optional dependency is exactly what removes it. So a
regression suite that doubles its example count has not necessarily bought `√2` of anything,
and `evalforge stats --resample-unit block` must be re-derived after any merge.

Dedup findings do not restrict to subsets either, for the same reason and in the same
direction. **Audit the corpus you will score on.**

---

## 4. A third property that does not hold, for a different reason

Union is not the only operation a verdict fails to survive. **Four of the fourteen
requirements read the split assignment** — `no_train_test_overlap`,
`no_retrieval_contamination`, `eval_splits_deduplicated`, and `eval_split_is_rankable` — so
their answers move when the seed does, while `content_id` stands still. This has nothing to do
with idf; it is simply that `content_id` was designed to ignore `split`, and these four do not.

The witness is measured, not hypothetical. Building `qa_toy` forty times with
`--strategy random`, one seed per build:

| | value |
|---|---|
| distinct `content_id`s across all 40 releases | **1** |
| seeds that **pass** `eval_splits_deduplicated` | 9, 35 |
| seeds that **fail** it | the other 38 |

The first draft of `applies_to` compared `content_id` alone, on the argument that no finding a
gate reads depends on the split. It would have handed seed 9's pass to seed 0's release. So a
report is scoped to `eval_id` whenever a split-sensitive requirement ran, and to `content_id`
otherwise; `GateReport.scope` says which, and the identity table in every rendered report names
it.

The scope is derived from the requirements that **actually executed**, never from the split
strategy someone claims to have used. The adversarial strategy does hold all four at zero for
every seed — a test pins that — but relying on it would mean trusting `split_spec.strategy` to
be true, and a strategy field is a claim, not evidence.

`integrity` sits outside the scheme entirely: it is a statement about the bytes of one
directory. Two releases can share both identities and differ in `created_at` or in a card's
wording, so a report is never evidence about a directory other than its own `release_dir`.

---

## 5. The five contracts

Each profile names its requirements and — at equal weight — two lists nothing else in this
repository writes down: what the consumer must establish **for itself**, and what the profile
**deliberately does not check**. An omission that goes unstated is indistinguishable from an
oversight.

`evalforge gate --list` prints all of them. In summary:

| profile | consumer | the question it turns into an exit code |
|---|---|---|
| `code-agent-runtime` | code-agent-runtime | are these tasks sound enough to gate a run on? |
| `bioagentbench` | BioAgentBench | is this synthetic biomedical set free of PII and label conflicts? |
| `llm-regression` | LLM regression / observability | can this suite rank two models at all? |
| `scientific-claim-verifier` | scientific-claim-verifier | are the claim/evidence labels consistent and unleaked? |
| `agent-control-bench` | agent-control-bench | is this control set clean without being sanitized? |

Every profile begins with `integrity` and `schema_compatible`. A finding read out of a
release whose bytes do not match its manifest is not a finding.

Two design points are worth stating explicitly.

**`llm-regression` counts blocks, never rows.** `eval_split_is_rankable` reads
`dependence.blocks_by_split`, because a duplicate cluster contributes *m* rows and one unit's
worth of information (Milestone 9). A test split of 200 rows in 3 duplicate clusters is a
test split of 3. `min_eval_blocks = 30` is a convention — Milestone 9's `MIN_EFFECTIVE_N` —
and not a theorem; on a corpus with a 3.6-point margin, Milestone 9 measured that ≈4500
examples were needed.

**`agent-control-bench` does not require unambiguous labels.** An over-refusal /
under-refusal set exists to probe the ambiguous boundary, so `LabelIssue.AMBIGUOUS` is a
feature of the corpus. A profile requiring zero ambiguous labels would reject every
well-constructed control benchmark, `safety_toy` included.

### What the demo release scores

`qa_toy` is a deliberately flawed fixture, so **all five profiles reject the release built
from it**, and a gate that passed it would be broken. The positive control lives in the test
suite: a release built with `--drop exact,normalized,blocking-labels` passes.

| profile | verdict | pass | fail | undecidable | blocking |
|---|---|---|---|---|---|
| `agent-control-bench` | **FAIL** | 4 | 2 | 0 | `no_exact_duplicates`, `no_answer_in_prompt` |
| `bioagentbench` | **FAIL** | 3 | 3 | 0 | `no_exact_duplicates`, `no_label_conflicts`, `no_missing_labels` |
| `code-agent-runtime` | **FAIL** | 3 | 3 | 0 | `no_missing_labels`, `no_answer_in_prompt`, `no_exact_duplicates` |
| `llm-regression` | **FAIL** | 5 | 1 | 0 | `eval_split_is_rankable` (2 blocks < 30) |
| `scientific-claim-verifier` | **FAIL** | 2 | 3 | 1 | 3 findings; `no_retrieval_contamination` undecidable |

---

## 6. The API

```python
from eval_dataset_forge.contracts import GatePolicy, ReleaseHandle, evaluate_gate

handle = ReleaseHandle.open("datasets/releases/qa_toy_v0.1.0")   # verifies on open
report = evaluate_gate(handle, GatePolicy.for_profile("llm-regression"))

report.ok                # False
report.failed            # (RequirementResult(name='eval_split_is_rankable', …),)
report.applies_to(other) # False — a different corpus
handle.blocks_in("test") # 2, not len(handle.split("test"))
handle.near_method       # 'tfidf'
```

A consumer that calls `json.load(open("manifest.json"))` has taken a dependency on every key
in the file, including the ones it does not read, because nothing tells a reader which those
are. So the reads are **declared**: `MANIFEST_FIELDS`, `AUDIT_FIELDS`, `DEPENDENCE_FIELDS`,
`SPLIT_STAT_FIELDS`, `SPLIT_COUNT_FIELDS` are the entire surface, and a test asserts each
name exists both on the pydantic model and in the committed demo release.

The compatibility rule is then checkable rather than aspirational:

> **Adding a key is a `patch`. Removing or retyping a declared key is a `major`, and
> `CONTRACT_VERSION` moves with it.**

Milestone 12 grew `split_statistics` by two keys under an unchanged `SCHEMA_VERSION`. That
was legal and invisible. The test that would have caught the *reverse* now exists.

### A policy is a file

```yaml
# policy.yaml — thresholds belong to the consumer, not to the dataset
profile: llm-regression
requirements: [integrity, schema_compatible, near_method_recorded, eval_split_is_rankable]
thresholds:
  min_eval_blocks: 30
  eval_split: test
  expected_near_method: tfidf     # pin it, or compare two different benchmarks
allow_undecidable: false
```

```bash
evalforge gate --release <dir> --policy policy.yaml --out results/reports/gate.md
```

---

## 7. What no contract here can supply

- **Whether the model under test was trained on the benchmark.** Contamination is measurable
  only against text the auditor holds, and nobody holds a frontier model's training set.
  Every leakage check in this repository is *intra-corpus*.
- **Whether the dataset measures what the consumer wants measured.** Construct validity is not
  a checkable property of a JSONL file.
- **Whether a gold label is correct.** `no_missing_labels` asks whether a gold answer exists.
  `no_label_conflicts` asks whether annotators agreed. Neither asks whether they were right.
- **Authenticity.** `manifest.json` cannot hash itself. Verification detects accidental
  corruption, not deliberate rewriting.
- **Coverage.** No artifact in a release carries a taxonomy, so no gate can say a task set
  covers the behaviors it claims to.

A gate report prints, beside every requirement, the claim its pass does **not** establish.
That column is not a footnote: a gate report is read by whoever most wants it to say the
dataset is fine.

---

## Reproduce

```bash
python3 -m pytest -q tests/test_contracts.py            # 50 tests

evalforge gate --list
evalforge gate --release datasets/releases/qa_toy_v0.1.0 --profile llm-regression   # exit 1

# The two properties that do not hold, measured:
python3 -m pytest -q tests/test_contracts.py \
  -k "not_additive or additive_under_jaccard or not_preserved_under_union"

# The optional dependency that changes the answer:
python3 -m pytest -q tests/test_contracts.py \
  -k "optional_dependency or environment_independent"
```

## See also

- [`RELEASES.md`](RELEASES.md) — the three identities and the version-bump rule.
- [`DEDUPLICATION.md`](DEDUPLICATION.md) — the near stage, its thresholds, and its failure modes.
- [`STATISTICS.md`](STATISTICS.md) — why blocks and not rows.
- [`LIMITATIONS.md`](LIMITATIONS.md) — the honest negatives, collected.
