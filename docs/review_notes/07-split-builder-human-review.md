# Human review: Milestone 7 — Split builder

TL;DR
The fifth pipeline stage landed, and the first that **changes** the data rather than
describing it: a `splits/` package with five strategies (random, stratified,
source-grouped, adversarial, temporal) built on one assigner, plus the `evalforge
build-splits` CLI. The strategies are the *same call with a different definition of an
indivisible unit*, which is what made two results visible. **(1) The strategy designed to
stop provenance leakage stops none of it** — every planted duplicate carries a fresh
`source_id`, and on three of five fixtures `source-grouped` is bit-for-bit identical to
`random`. **(2) Stratifying on two facets is worse on both facets than either alone**, and
provably puts no `hard` example in the test split; this changed the default from
`(task_family, difficulty)` to `(task_family,)`. `adversarial` reaches **0.00 residual
cross-split overlap on every fixture at every one of 20 seeds** — and that zero is
verified, not asserted: every build re-runs Milestone 5's checks over its own output. It
is also honestly labelled as inherited from a heuristic. Full suite: 1245 passed;
fixtures validate; ruff clean.

What changed
- Added `src/eval_dataset_forge/splits/{__init__,blocks,assignment,random,stratified,adversarial,temporal}.py`.
- Added `tests/test_splits.py` (723 tests): apportionment, block algebra, assignment
  invariants, the two strategy equivalences, the 20-seed × 6-corpus × 5-strategy leakage
  comparison, the stratification sweep, the label hand-off, four honest negatives,
  determinism, byte-identical IO, and the CLI.
- Wired `evalforge build-splits` (removed from `_PENDING`) with `--seed`, `--ratios`,
  `--stratify-by`, `--quarantine-blocking`, `--ignore-answer-gate`, `--fail-on-leakage`,
  `--json`. It writes `<out>/splits/{train,dev,test}.jsonl` + `<out>/split_report.json`;
  manifest/audit/card remain Milestone 10's.
- Rewrote `docs/SPLITS.md` around the measured results; added the splits section to
  `docs/RESULTS.md`; added "A split guarantee is only as good as the clusters it is built
  from" to `docs/LIMITATIONS.md`.
- Rewrote `site/splits.html`; status updates on `milestones.html`, `audit-pipeline.html`,
  `index.html`, `results.html`, `limitations.html`. (`index.html`'s status paragraph was
  also missing Milestone 6; added.)

**Two deviations from `PLAN.md` §4**, both additive plumbing, neither a contract change:
- `blocks.py` — the block algebra all five strategies consume, and the home of
  source-grouped's block constructor. §4 sketches four strategy modules but §10 requires
  five strategies, so grouping needed a home.
- `assignment.py` — the two assigners. Folding these into `random.py` would have left
  `temporal.py` importing its core from a module named `random`.

`source_grouped_splits()` lives in `random.py` rather than a module of its own, because it
*is* the random assigner over a different block. I did not edit `PLAN.md`.

**A bug the tests found.** `SplitReport.eval_duplicates` read the `duplicate` flag off the
report's blocks, and returned **0.00 for every strategy on every fixture** — including
`random`. It looked like a clean sweep. The cause: only `adversarial` *builds* duplicate
blocks; the other four assign singletons whose `duplicate` flag is always `False`. The
metric was measuring which strategy had bothered to look for duplicates, not which had let
one into an evaluation split. Fixed by reading the ids off the `DedupReport` — the same
object the verifier uses — and fixing the definition to `dev ∪ test` for every strategy so
the number means the same thing in every row. Without the fix the second results table
would have read all zeros and been quietly wrong.

**Answering Milestone 4's open question.** The review note asked whether the adversarial
split wants the three-member `qa-003`/`qa-004`/`qa-012` single-linkage cluster or a set of
pairs. **One cluster** — a pair decomposition would let `qa-004` and `qa-012` land in
different splits. Pinned by a test.

What should be checked later
- **Does `eval_splits` earn its surface area? (most important design call)** The knob
  relaxes "no duplicate in `dev` or `test`" to "…in `test`". It is a **no-op on the entire
  committed corpus**: duplicate blocks are dealt first and `train`'s target dwarfs `dev`'s,
  so they never reach `dev` even when permitted. It is exercised only by a synthetic corpus
  of four identical pairs. Decide whether a documented degree of freedom that no real
  dataset here exercises should ship, or whether the guarantee should simply be fixed at
  `dev ∪ test`.
- **`hard` cannot reach the test split under joint stratification, and this is exact, not
  statistical.** The three `hard` examples sit in three different family strata of size
  one, and `apportion(1, ...)` returns `{train: 1}`. I changed the default rather than the
  algorithm. The alternative — letting a small stratum "borrow" from a large one, or
  apportioning globally and stratifying only as a tiebreak — would preserve marginals
  better at small N. Worth a decision before Milestone 8 introduces a calibrated bucket
  that someone will want to stratify on.
- **Should difficulty stratification wait for Milestone 8?** I deliberately stratify on the
  *declared* `difficulty_tags`, not a modelled score, because stratifying on a score this
  codebase computes would let a difficulty heuristic define the test set it is later
  evaluated on. The cost: 5 of 56 examples fall into an `unlabelled` stratum. Confirm the
  circularity argument, and decide whether Milestone 8's bucket should ever be a legal
  `stratify_by` facet.
- **A block spanning strata is resolved by plurality.** `qa-003` is `easy`, `qa-012` is
  `medium`, and they share an adversarial block, so the block distorts whichever facet it
  is counted under by exactly its minority members. Cutting the block would reintroduce the
  leakage the block exists to prevent, so I report the residue in the achieved distribution
  rather than removing it. The tradeoff is a judgement call.
- **`temporal` on the pooled corpus is pathological and the tool only warns.** The fixtures
  were authored family by family, so time-ordering nearly sorts by `task_family`: the test
  split ends up containing only `safety` plus one `bio` example. The builder emits "test
  contains no task_family=…" warnings and proceeds. Decide whether a strategy should ever
  *refuse* to build.
- **The temporal forward-fill rule is a choice, not a derivation.** `docs/SPLITS.md`
  originally said "missing timestamps fall back to insertion order with a warning", which is
  ambiguous between per-example and whole-dataset. I implemented per-example forward-fill (an
  undated example inherits its predecessor's timestamp, keeping its position) because it
  discards no good timestamps and sends leading undated examples to `train` rather than
  `test`. Two other readings are defensible.
- **The 0.7/0.1/0.2 default ratio is unswept.** Inherited from nothing; it is simply
  train-heavy. `dev` at 0.1 of a 10-example fixture is one example, which is what makes the
  small-stratum warnings fire so readily. Sanity-check on a real corpus.
- **`SplitReport` shape.** An ephemeral dataclass, consistent with `DedupReport`,
  `LeakageReport`, and `LabelAuditReport`. `to_dict()` was written to slot into
  `AuditResult.split_statistics`. Verify that is what Milestone 12 expects. `SplitSpec` *is*
  a versioned pydantic model and is populated in full (strategy, seed, ratios, group_key,
  constraints incl. the resolved `near_method`), so a release can state what its splits
  protect. `split_report.json` is a report artifact under the gitignored `results/` tree;
  confirm Milestone 10 wants the split files themselves (`splits/*.jsonl`, which
  `write_splits()` already emits in the `PLAN.md` §5D layout) and generates the rest.
- **Quarantine is honoured by `adversarial` only.** Applying it silently to `random` would
  make random not random, and the comparison table would stop measuring what it claims. But
  it means `evalforge build-splits --strategy random --quarantine-blocking` silently ignores
  the flag rather than erroring. Confirm that is the behaviour you want.

Technical risks
- Low. Pure Python; no new dependencies, no network, no paid API. Input examples are never
  mutated (`apply_splits()` returns `model_copy` results; a test asserts it). The only
  filesystem writes are under the caller's explicit `--out`. The optional scikit-learn
  accelerator is reached only through the `DedupReport` this stage consumes.
- **The real risk is a guarantee that reads stronger than it is.** `adversarial` leaves
  zero residual cross-split overlap *by construction* — its blocks **are** the dedup
  ladder's clusters — so it is exactly as good as the near stage's recall and no better.
  `ret-001`/`ret-002` is a planted cross-split paraphrase no lexical threshold can reach
  (measured in Milestone 4: a genuine non-duplicate scores strictly higher). The adversarial
  split straddles it exactly as random would, and the post-build verifier, which reuses the
  same ladder, reports a clean zero. `test_the_adversarial_guarantee_is_inherited_from_a_heuristic`
  pins this. Stated in `SPLITS.md`, `RESULTS.md`, `LIMITATIONS.md`, the website, and the
  module docstring.
- **`retrieval_toy`'s clean row is vacuous.** Its gold answers are document-id lists that
  differ, so the near stage's answer gate rejects every pair before a threshold is consulted
  and `find_duplicates()` returns *no findings at all*. Zero residual overlap for every
  strategy on that fixture measures the gate, not the data. `--ignore-answer-gate` is the
  escape hatch, and the vacuity is asserted by a test — but a user who reads only the table
  could easily miss it. This is the third milestone in a row where `retrieval_toy`'s
  doc-id answers force a special case.
- **Verification shares the detector's blind spots by construction.** `verify_splits()`
  reuses `cross_split_clusters()` and the same `DedupReport`, so a zero means "no
  *detectable* cross-split duplicate", never "no duplicate". An independent verifier (a
  different similarity method, or embeddings) would be a genuine improvement and is an
  explicit non-goal of the offline build.
- Nothing in this milestone deletes an example. Duplicate and quarantined examples move to
  `train`. Removing an example from a dataset is Milestone 10's decision, recorded in a
  manifest a reader can audit.
- `Random(seed).shuffle` fixes the block order. Determinism therefore depends on CPython's
  Mersenne Twister, which is stable across versions but is not a documented cross-language
  contract. Same exposure as any seeded numpy/stdlib pipeline; noted, not fixed.

Commands to verify
```
python3 -m pytest -q
python3 -m pytest -q tests/test_splits.py
python3 scripts/validate_fixtures.py
ruff check .

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

# The headline result, in one command: source-grouped is the same function as random
# on the three fixtures where no source_id is shared.
PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.dedup import find_duplicates
from eval_dataset_forge.splits import build_splits
examples = loader.load_examples(loader.FIXTURES_ROOT / 'agent_tasks_toy' / 'examples.jsonl')
report = find_duplicates(examples)
a = build_splits(examples, strategy='source-grouped', seed=3, dedup_report=report)
b = build_splits(examples, strategy='random', seed=3, dedup_report=report)
print('identical assignment:', a.assignment == b.assignment)
print('residual overlap    :', len(a.assignment) and len(a.residual_overlap))
"
```

Human feedback
<!-- Human will fill this later. -->
