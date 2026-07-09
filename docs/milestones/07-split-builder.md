# Milestone 7 — Split builder

## Summary

Land the fifth stage of the pipeline and the first one that **changes** the data rather
than describing it: five split strategies (random, stratified, source-grouped,
adversarial, temporal) over one algorithm, plus the `evalforge build-splits` CLI. This is
audit dimension #7.

The distinguishing work was not writing five strategies. It was (a) noticing that all
five are **the same call with a different definition of an indivisible unit**, which
collapses the package to one assigner and five block constructors; (b) discovering by
measurement that **the strategy designed to stop leakage through provenance stops none of
it**, and that on three of five fixtures it is bit-for-bit identical to a random split;
(c) discovering that **stratifying on two facets is worse on both facets than stratifying
on either alone**, and that it guarantees the test set contains no `hard` example — which
changed the default; and (d) being explicit that the adversarial strategy's zero is
inherited from a heuristic, with a test that pins the paraphrase it cannot see.

## Deliverables

- `src/eval_dataset_forge/splits/` package:
  - `blocks.py` — `Block` (`example_ids`, `reason`, `duplicate`), union-find
    `_components()`, and the three constructors: `singleton_blocks()`, `source_blocks()`,
    `duplicate_blocks()` (edges from the whole dedup ladder ∪ `duplicate_sources`).
  - `assignment.py` — `apportion()` (largest-remainder), `assign_blocks()` (seeded,
    stratum- and eligibility-aware), `assign_in_order()` (temporal), `normalize_ratios()`.
  - `random.py` — `random_splits()`, `source_grouped_splits()`. Together, because they
    are the same call with a different block.
  - `stratified.py` — `declared_bucket()`, `facet_value()`, `stratum_of()`,
    `block_stratum()`, `distribution()`, `stratified_splits()`.
  - `adversarial.py` — `adversarial_splits()`: duplicate blocks, eval-ineligibility,
    label-audit quarantine.
  - `temporal.py` — `parse_timestamp()`, `temporal_order()` (forward-fill), `temporal_splits()`.
  - `__init__.py` — `build_splits() → SplitReport`, `apply_splits()`, `verify_splits()`,
    `write_splits()`, `format_report()`.
- `cli.py` — `evalforge build-splits` with `--seed`, `--ratios`, `--stratify-by`,
  `--quarantine-blocking`, `--ignore-answer-gate`, `--fail-on-leakage`, `--json`.
  Removed from `_PENDING`.
- `tests/test_splits.py` — 723 tests: apportionment, block algebra, the assignment
  invariants, the two strategy equivalences, the leakage comparison across 20 seeds × 6
  corpora × 5 strategies, the stratification sweep, the label hand-off, four honest
  negatives, determinism, byte-identical IO, and the CLI.
- Docs: `docs/SPLITS.md` rewritten around the measurements; `docs/RESULTS.md` gains its
  splits section; `docs/LIMITATIONS.md` gains "a split guarantee is only as good as the
  clusters it is built from". Website: `splits.html` rewritten; status updates to
  `milestones.html`, `audit-pipeline.html`, `index.html`, `results.html`,
  `limitations.html`.

## Design decisions

- **A strategy is a choice of block.** `random` and `source-grouped` are literally
  `assign_blocks(singleton_blocks(x))` and `assign_blocks(source_blocks(x))`, which is why
  they share a module. The insight is load-bearing rather than cosmetic: it is what makes
  the headline result visible at all.
- **Apportion to integers before dealing.** Rounding afterwards starves the small splits:
  a stratum of three always shows `dev` a deficit of 0.3 that no assignment can close, and
  a float-deficit greedy keeps chasing it to `1/1/1` instead of `2/0/1`. Largest-remainder
  first, greedy after. With singleton blocks the deal reproduces the targets exactly.
- **Two metrics, because they disagree.** `residual_overlap` counts clusters *straddling*
  a boundary; `eval_duplicates` counts duplicate-cluster examples *inside* `dev`/`test`.
  A cluster wholly inside `test` leaks nothing yet still narrows every Milestone 9
  confidence interval. Pooled `temporal` scores 1.00 and 6.00 respectively.
- **`eval_duplicates` reads the dedup report, never the blocks.** Only the adversarial
  strategy builds duplicate blocks, so a block-derived count would report a flattering
  zero for the four strategies that never looked. Caught by a test written before the
  fix; see "A bug the tests found".
- **`dev` and `test` are both evaluation splits.** Tuning on an item and then scoring on
  its twin is the same defect as training on it. So an adversarial duplicate block lands
  in `train`, which is strictly stronger than "no cluster straddles train and test": it
  also keeps two correlated items out of the test set itself.
- **A pure source group is dependent, not duplicated,** and stays eligible for an
  evaluation split. `ret-006`/`ret-009` share `src-boiling`; they are two questions from
  one source, not two copies of one question.
- **Verification reuses Milestone 5 unchanged** and knows nothing about the strategy that
  produced the assignment. A test asserts `verify_splits()` reproduces
  `SplitReport.residual_overlap` from the outside.
- **Nothing is deleted.** Duplicate and quarantined examples move to `train`. Removing an
  example is the release builder's decision (Milestone 10), recorded in a manifest.
- **Quarantine is honoured by `adversarial` only.** Applying it silently to `random` would
  make random not random, and the comparison table would stop measuring what it claims.
- **Difficulty stratification reads the *declared* tags,** not a modelled score. Milestone
  8 calibrates difficulty; stratifying on a score this package computed would let a
  heuristic define the test set it is later evaluated on.

## Deviations from the design doc

`PLAN.md` §4 sketches `splits/{random,stratified,adversarial,temporal}.py`, and §10
requires five strategies. Two shared modules were added rather than smearing their content
across the four:

- **`blocks.py`** — the block algebra all five strategies consume, and the home of
  source-grouped's block constructor.
- **`assignment.py`** — the two assigners. Folding these into `random.py` would have left
  `temporal.py` importing its core from a module named `random`.

`source_grouped_splits()` lives in `random.py` rather than in a module of its own,
because it *is* the random assigner over a different block, and putting the two side by
side is the clearest statement of the package's thesis.

## The result worth reading twice

**The grouping key a dataset hands you is not the one that closes the leak.**

Every planted duplicate carries a *fresh* `source_id` — `src-fr-capital` →
`src-fr-capital-dup`, `src-fizzbuzz` → `src-fizzbuzz-para`,
`src-photosynthesis` → `src-photosynthesis-para` — exactly as a real corpus records a
paraphrase written as a new record. So `source-grouped`, whose entire purpose is to stop
provenance leakage, leaves every one of them free to straddle train and test.

Mean residual cross-split duplicate clusters over seeds 0–19:

| fixture | random | stratified | source-grouped | **adversarial** | temporal |
|---|---|---|---|---|---|
| `qa_toy` | 1.30 | 1.30 | 1.00 | **0.00** | 1.00 |
| `agent_tasks_toy` | 1.75 | 1.75 | 1.75 | **0.00** | 0.00 |
| `bio_toy` | 1.10 | 1.10 | 1.10 | **0.00** | 0.00 |
| `retrieval_toy` | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |
| `safety_toy` | 0.95 | 0.95 | 0.95 | **0.00** | 0.00 |
| **all five pooled** | 5.45 | 5.25 | 4.60 | **0.00** | 1.00 |

Sharper than the near-tie: on `agent_tasks_toy`, `bio_toy`, and `safety_toy` no
`source_id` is shared at all, so `source_blocks()` returns singletons and
`source-grouped` produces a **bit-for-bit identical assignment to `random`** at every
seed. It is not a weaker control there; it is the same function. Pinned by
`test_source_grouped_is_bit_for_bit_random_without_a_shared_source`.

Only blocks derived from example *content* close the channel. Provenance metadata
describes where an example came from; it does not know what the example says.

## The second result worth reading twice

**Stratifying on more facets makes the test set less representative, not more.**

Total variation distance between the test split's marginal and the corpus's, mean over
20 seeds, pooled corpus:

| `stratify_by` | strata | TVD family ↓ | TVD difficulty ↓ | dev size | `hard` in test |
|---|---|---|---|---|---|
| (none) | 1 | 0.216 | 0.156 | 6.0 | 0.45 |
| **`task_family`** (default) | 5 | **0.036** | 0.152 | 5.0 | 0.25 |
| `difficulty` | 4 | 0.184 | **0.030** | 5.0 | 1.00 |
| `task_family,difficulty` | 17 | 0.083 | 0.077 | 3.0 | **0.00** |

Joint stratification is worse on *both* facets than either facet alone. Strata multiply
when facets are combined while the corpus does not: 5 × 4 gives 17 non-empty strata over
56 examples, a mean of 3.3 each, and a stratum of three cannot fill a `dev` split at
ratio 0.1.

The empty cell is exact, not statistical: the three `hard` examples (`agent-008`,
`qa-014`, `ret-005`) sit in three *different* family strata, each of size one, and
`apportion(1, …)` returns `{train: 1, dev: 0, test: 0}`. **No `hard` example can reach the
test set**, so the benchmark cannot measure difficulty.

This changed the default from `("task_family", "difficulty")` to `("task_family",)`, and
added two warnings — one when the average stratum cannot fill the smallest split, one for
each facet value missing from `test`. Pinned by
`test_joint_stratification_is_worse_on_both_facets_than_either_alone` and
`test_joint_stratification_puts_no_hard_example_in_the_test_split`.

## A bug the tests found

`SplitReport.eval_duplicates` originally read the `duplicate` flag off the report's
blocks. It returned **0.00 for every strategy on every fixture** — including `random` —
and the number looked like a clean sweep rather than a bug.

The cause: only the adversarial strategy *builds* duplicate blocks. Random, stratified,
source-grouped, and temporal assign singletons, whose `duplicate` flag is always `False`.
The metric was measuring which strategy had bothered to look for duplicates, not which
strategy had let one into an evaluation split.

Fixed by reading the duplicate example ids off the `DedupReport` — the same object the
verifier uses — and by fixing the definition to `dev ∪ test` for every strategy, so the
number means the same thing in every row of the comparison table. Pinned by
`test_eval_duplicates_is_read_from_the_dedup_report_not_from_the_blocks`. Without the fix
the entire second results table would have read `0.00` and been quietly wrong.

## Results

Every planted duplicate cluster, on every fixture, at all 20 seeds: `adversarial` leaves
**zero** residual cross-split overlap and **zero** duplicates in `dev`/`test`. `random`
and `source-grouped` leave both non-zero on every fixture that has a detectable duplicate.

The label-audit hand-off Milestone 6 promised: on the pooled corpus at seed 0, an
adversarial split *without* quarantine scores on six examples whose gold answer a human
must adjudicate (`agent-009`, `bio-007`, `qa-010`, `ret-004`, `ret-005`, `saf-008`).
`--quarantine-blocking` moves all fourteen blocking examples to `train`. The cost:
`test` shrinks from 12 to 8 examples, its achieved ratio falls from 0.214 to 0.143, and
the builder warns that `test` can no longer measure the `unlabelled` difficulty bucket.

Milestone 4's review note asked whether the adversarial split wants the three-member
`qa-003`/`qa-004`/`qa-012` cluster or a set of pairs. **One cluster.** A pair
decomposition would let `qa-004` and `qa-012` land in different splits.

## Four honest negatives

- **The adversarial zero is inherited from a heuristic.** It holds *by construction* —
  the blocks are the dedup clusters — so it is exactly as good as the near stage's
  recall. `ret-001`/`ret-002` is a planted cross-split paraphrase no lexical threshold can
  reach; with the answer gate disabled the ladder finds the fixture's other two clusters,
  the adversarial split keeps them intact, and `ret-001`/`ret-002` straddles `train`/`test`
  anyway while the verifier reports a clean zero. Pinned by
  `test_the_adversarial_guarantee_is_inherited_from_a_heuristic`.
- **`retrieval_toy`'s clean row is the answer gate, not the data.** Its gold answers are
  document-id lists that differ, so the near stage rejects every pair before a threshold
  is consulted, and `find_duplicates()` returns *no findings at all*. Zero residual
  overlap for every strategy on that fixture is vacuous. Asserted by
  `test_retrieval_toys_clean_row_is_the_answer_gate_not_an_absence_of_leakage`.
- **`stratified` is bit-for-bit `random` on every committed fixture.** Each holds exactly
  one `task_family`, so the default stratification has one stratum. It only bites on the
  pooled corpus.
- **`eval_splits=("test",)` changes nothing on this corpus.** Duplicate blocks are dealt
  first and `train`'s target dwarfs `dev`'s, so they never reach `dev` even when allowed
  to. Exercised for real by a synthetic corpus of four identical pairs, where the strict
  setting collapses everything into `train` (`dev=0`, `test=0`, two warnings, nothing
  dropped) and the relaxed one puts a pair in `dev`.

## Validation

```bash
python3 -m pytest -q                      # 1245 passed (522 prior + 723 splits)
python3 -m pytest -q tests/test_splits.py # 723 passed
python3 scripts/validate_fixtures.py      # all 5 fixtures valid, exit 0
ruff check .                              # all checks passed

# A leakage-free, label-clean evaluation split (exit 0):
PYTHONPATH=src python3 -m eval_dataset_forge build-splits \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --strategy adversarial --quarantine-blocking --fail-on-leakage \
  --out results/processed/qa_toy_adversarial

# The same dataset split randomly still leaks, and the gate catches it (exit 1):
PYTHONPATH=src python3 -m eval_dataset_forge build-splits \
  --dataset datasets/fixtures/qa_toy/examples.jsonl \
  --strategy random --fail-on-leakage \
  --out results/processed/qa_toy_random
```

Verified end to end, not merely unit-tested: the two commands above exit 0 and 1
respectively; running either twice produces byte-identical `splits/*.jsonl` and
`split_report.json`; and the artifacts land under the gitignored `results/` tree.

## Open questions for review

- **Does `eval_splits` earn its surface area?** It is a documented degree of freedom in
  the guarantee, but it is a no-op on the entire committed corpus and is exercised only
  synthetically.
- **A block spanning strata is resolved by plurality**, which distorts the facet by
  exactly the block's minority members. Cutting the block would reintroduce the leakage the
  block exists to prevent, so the residue is reported rather than removed — but the
  tradeoff is a judgement call.
- **Should difficulty stratification wait for Milestone 8's calibrated bucket?** Using it
  would let a difficulty heuristic define the test set that heuristic is later evaluated
  on. The declared tags avoid the circularity but leave 5 of 56 examples `unlabelled`.
- **`temporal` on the pooled corpus is pathological and the tool only warns.** The
  fixtures were authored family by family, so time-ordering nearly sorts by `task_family`
  and the test split contains only `safety` plus one `bio` example. Whether a strategy
  should ever *refuse* to build is left open.

## Status

Complete. Next: Milestone 8 — Difficulty calibration (transparent heuristics + difficulty
distribution reports), which will supply the calibrated bucket this milestone deliberately
declined to stratify on, and whose per-split distributions `SplitReport.distributions`
already reports for the declared tags.
