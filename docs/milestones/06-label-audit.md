# Milestone 6 — Label audit

## Summary

Land the fourth stage of the audit pipeline: five label-quality checks — missing,
conflicting, ambiguous, inconsistent format, and a derived adjudication flag — emitting all
five `LabelIssue` kinds, plus a deterministic adjudication-queue export. This is audit
dimension #5: *is the gold answer trustworthy?*

The distinguishing work was not writing five detectors. It was (a) noticing that **two
fixture examples with identical empty gold answers have opposite verdicts**, and finding
the check that separates them without a special case; (b) **calibrating the one fuzzy check
against withheld ground truth** — stripping the `ambiguous` tags and asking what a lexical
heuristic would have found on its own; (c) reducing the adjudication queue to **one rule**
rather than a table of special cases; and (d) reporting three honest negatives, including
the fact that the stage is structurally blind to the label defect that matters most.

## Deliverables

- `src/eval_dataset_forge/labels/` package (the tree `PLAN.md` §4 specifies):
  - `disagreement.py` — the three `CONFLICTING` sources. `annotator_conflicts()` (records
    whether a strict majority exists), `shared_prompt_conflicts()` (consumes
    `dedup.duplicate_input_clusters`), `relevance_conflicts()` (set-compares an example's
    gold doc-id list against its own passage `relevant` flags). Plus `label_key()`,
    `majority_value()`, and `agreement()` → `AgreementSummary`.
  - `audit.py` — `missing_labels()`, `ambiguous_labels()` (declaration + two lexical
    signals), `inconsistent_formats()` (shape minority + vocabulary outlier),
    `family_expectations()` → `FamilyExpectation`, `has_label()`, `answer_shape()`.
  - `adjudication.py` — `AdjudicationItem`, `conflict_resolutions()`, `format_resolution()`,
    `build_queue()`, `needs_adjudication()`, `write_queue()` (JSONL).
  - `__init__.py` — `audit_labels(...) → LabelAuditReport`, `to_dict`, `format_report`.
- `tests/test_labels.py` — 144 tests: primitives and their traps, the ambiguity calibration
  sweep with both ablations, recall against every fixture's `defects.yaml`, the queue's
  determinism and byte-identical export, and regression tests pinning each documented
  failure mode and honest negative.
- Docs: `docs/LABEL_AUDIT.md` rewritten around the measured results; `docs/RESULTS.md` gains
  its label-audit section; `docs/LIMITATIONS.md` gains a "Label truth is invisible to us"
  section. Website: `label-audit.html` rewritten; status updates to `audit-pipeline.html`,
  `milestones.html`, `index.html`, `results.html`, `limitations.html`.

## Design decisions

- **One check function, one `LabelIssue`.** Enforced by `test_each_check_emits_only_its_own_kind`,
  and a second test asserts no kind is declared in the schema without a check that can emit
  it. (The M5 property was per-*module*; here two kinds share `audit.py`, so the property
  moves down a level rather than being dropped.)
- **Stage order is load-bearing**, unlike the leakage channels. `missing` runs first: an
  absent answer has no format, and letting `""` into the shape census would both
  double-flag it and skew the census every other answer is measured against.
- **Emptiness is `answer_candidates()`, reused from Milestone 5.** Its docstring already
  said "an absent gold answer is a label defect (Milestone 6)" — a deliberate hand-off. It
  also already gets the traps right: `False`, `0`, and `0.0` are labels, and `if not value`
  deletes all three.
- **`missing` does not own an empty *container* when the example labels relevance twice.**
  For such an example an empty gold list is a *claim* that can be checked, not an absence
  that can be asserted. Check 2c owns it. This is the whole `ret-004` / `ret-005`
  distinction (below).
- **Relevance conflict is set equality, not an emptiness rule.** Two empty sets agree. The
  legitimacy of `ret-005` falls out of the comparison rather than from a guard.
- **`label_key` folds a string and *then* JSON-encodes it.** Fold alone maps the string
  `"4"` and the integer `4` onto the same key — see "A bug the tests found" below.
- **Ambiguity separates declaration from inference.** A tag is exact; the lexical signals
  are heuristic; `detail` names which fired; `heuristic_ambiguity=False` keeps only
  declarations.
- **Format checks shape, not key set.** `agent_tasks_toy`'s coding tasks carry
  `{reference_solution, success_criteria}` and its research tasks `{reference_answer}` — a
  legitimate per-subtype schema the fixture's own description declares. A key-set check
  would report three false positives.
- **The vocabulary check only flags a *rare* out-of-vocabulary answer.** `label_vocabulary`
  admits answers of ≤2 tokens, so a repeated longer label is outside the vocabulary through
  an artifact of that cap, not through malformation.
- **Adjudication is one rule**: blocking ⟺ at least one issue has no resolution the audit
  can propose. Each kind's blocking-ness follows, so there is no severity table to argue
  with. A relevance conflict blocks despite an obvious-looking fix, because it is equally
  possible the *flag* is wrong as the gold list, and guessing would silently rewrite the
  benchmark.
- **No CLI subcommand.** The label audit surfaces through `evalforge audit` (Milestone 12),
  keeping the `PLAN.md` §8 CLI contract stable — same call as Milestones 4 and 5.
  `adjudication_queue.jsonl` is a report artifact, not part of a release directory
  (`PLAN.md` §5D), so it is a plain dataclass rather than a pydantic on-disk model.

## Deviations from the design doc

None to `PLAN.md` — the shipped tree is exactly `labels/{audit,disagreement,adjudication}.py`.
One refinement to `LABEL_AUDIT.md`'s original design:

- **`LABEL_AUDIT.md` §1 defined missing as "`expected_output` … absent or empty".** Taken
  literally that rule is *wrong* on retrieval data: `ret-005`'s empty gold list is the
  correct answer to an unanswerable query. Emptiness alone cannot decide, so the check was
  narrowed and the empty-container case handed to the relevance comparison, which has the
  evidence to decide it. Written down rather than papered over.

## The result worth reading twice

`ret-004` and `ret-005` both carry `expected_output: []`. **A rule keyed on emptiness flags
both, and is wrong about one of them.**

| example | gold list | passages marked relevant | verdict |
|---------|-----------|--------------------------|---------|
| `ret-004` | `[]` | `{d9}` | **conflict** — an incomplete relevance set |
| `ret-005` | `[]` | `{}` | clean — an unanswerable query, correctly labelled |

Retrieval examples label relevance *twice* — once as `expected_output`, once as
`metadata.retrieval.docs[].relevant`. Comparing the two channels as **sets** separates the
pair with no special case, and recovers `ret-004`, the planted `incomplete_relevance`
defect that **no other stage in the pipeline sees**. Milestone 5's `_is_relevant` noticed
the contradiction and left a docstring handing it here; this milestone collects it.

Pinned by `test_relevance_conflict_recovers_the_planted_incomplete_relevance` and
`test_the_over_specific_query_passes_clean_by_set_equality`.

## The second result worth reading twice

`qa-008` ("Name a primary color.") and `ret-004` ("Name a noble gas.") are **the same
sentence pattern**. The fixture declares the first ambiguous and not the second. *No rule
keyed on prompt text alone can flag one and not the other* — separating them lexically
would need to know that primary colors and noble gases are both enumerable, which is world
knowledge, not form.

What separates them is the **shape of the gold answer**: a list-valued answer can enumerate
every acceptable response, so an open-set prompt answered by a set is not under-specified.
`test_qa_008_and_ret_004_are_separated_only_by_the_shape_of_the_answer` swaps only the
answer shape between the two prompts and shows both verdicts flip with it.

## A bug the tests found

`test_label_key_folds_strings_but_not_types` failed on the first run. `label_key` folded
strings through `normalize_text` and sent everything else through `canonical_output`, so
the string `"4"` and the integer `4` both keyed to `4` — and two annotators who wrote
different things would have been recorded as **agreeing**. Precisely the collision
`canonical_output` exists to prevent (it JSON-encodes strings for this reason), reintroduced
by bypassing it.

Fixed by folding *then* encoding. The module docstring had asserted the correct behaviour
before the code implemented it; a test written to check the docstring's claim caught the
gap. Pinned at the level a user would hit it, by
`test_annotators_writing_different_types_are_not_recorded_as_agreeing`.

## Results

| check | certainty | recall | false positives |
|-------|-----------|--------|-----------------|
| missing | exact | 4 / 4 | 0 |
| conflicting (annotators) | exact | 4 / 4 | 0 |
| conflicting (shared prompt) | exact | n/a (none planted) | 0 |
| conflicting (relevance channels) | exact | 1 / 1 | 0 |
| ambiguous (declared tag) | exact | 5 / 5 | 0 |
| ambiguous (lexical, tags withheld) | heuristic | 4 / 6 declared | 0 |
| inconsistent format | heuristic | 2 / 2 | 0 |

**15 / 15 auto-detectable planted label defects recovered, 0 false positives.** 16 of 56
examples flagged; 14 blocking; only `agent-011` and `saf-010` carry a proposed resolution.

Ambiguity calibration (all `ambiguous` tags withheld from all 56 examples):

| max tokens | flagged | recovered (of 6 declared) | false positives |
|---|---|---|---|
| 3 | 2 | 2 | 0 |
| **4 (default)** | **4** | **4** | **0** |
| 5 – 7 | 4 | 4 | 0 |
| 8 | 5 | 4 | 1 (`bio-004`) |

Both guards are load-bearing, each preventing exactly one false positive: dropping the
wh-word guard admits `qa-005` ("What is 2 + 2?"); dropping the scalar-gold gate admits
`ret-004`.

## Three honest negatives

- **The stage is blind to the defect that matters most.** `qa-006` (the chemical symbol for
  gold given as `Ag`), `bio-009` (23 chromosomes in a human somatic cell), and `saf-006` (a
  benign vaccine question labelled `refuse`) are well-formed, unanimous, unambiguous, and
  correctly shaped. **None is found by any check.** `defects.yaml` marks all three
  `auto_detectable: false`, and `test_mislabeled_examples_are_invisible_to_every_check`
  asserts they stay invisible, so the limitation cannot rot into a silent claim of coverage.
- **The shared-prompt conflict channel fires on nothing.** All four byte-identical-prompt
  clusters agree on their gold answers, i.e. they are duplicates, not conflicts. The
  difference `dedup.duplicate_input_clusters` exists to expose is empty on this corpus.
  Asserted by `test_shared_prompt_conflicts_fire_on_nothing_in_the_corpus` and exercised by
  a synthetic pair. (`open_set_request` is likewise redundant at the default width — the one
  example it flags is already caught by the dangling-reference signal.)
- **The widths 4–7 "plateau" is fixture sparsity, not a robustness basin.** No fixture prompt
  of 5–7 tokens passes the guards, so those widths agree by accident of the corpus. The
  default is therefore the *smallest* width attaining full reachable recall, not the middle
  of a basin we have evidence for. Stated in the docstring, the doc, and
  `test_widths_four_through_seven_are_indistinguishable_on_this_corpus`.

## Validation

```bash
python3 -m pytest -q                      # 522 passed (378 prior + 144 labels)
python3 -m pytest -q tests/test_labels.py # 144 passed
python3 scripts/validate_fixtures.py      # all 5 fixtures valid, exit 0
ruff check .                              # all checks passed

# Audit one fixture and export its adjudication queue:
PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.labels import audit_labels, format_report, write_queue
examples = loader.load_examples(loader.FIXTURES_ROOT / 'safety_toy' / 'examples.jsonl')
report = audit_labels(examples)
print(format_report(report))
write_queue(report.queue, 'results/audits/safety_toy/adjudication_queue.jsonl')
"
```

The queue export was verified end to end, not merely unit-tested: running the command twice
produces byte-identical `adjudication_queue.jsonl` files, and the artifact lands under the
gitignored `results/` tree.

## Open question for review

At `max_tokens=8` the ambiguity heuristic flags `bio-004` ("Classify the entity 'aspirin'
into a single category."), which `defects.yaml` files under `conflicting` — its annotators
split on *drug* vs *chemical compound*, arguably **because** the prompt never says which
taxonomy. Under that reading it is a true positive filed under a different heading, and the
"false positive" at width 8 is an artifact of the ground truth's taxonomy rather than of the
detector. The default width is not tuned on it either way, and the ground truth was not
edited. Raised in the review note, alongside the same seam in `retrieval_toy`, whose two
findings map onto its `relevance_issues` block rather than `label_issues`.

## Status

Complete. Next: Milestone 7 — Split builder (random, stratified by family/difficulty,
source-grouped, adversarial), which will consume `dedup.redundant_example_ids` and the
near-duplicate clusters to keep paraphrases out of the test split, `dedup.duplicate_sources`
for source-grouped splits, and this milestone's `blocking_example_ids` to decide which
examples are fit to ship in an evaluation split at all.
