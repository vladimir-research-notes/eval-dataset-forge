# Label audit & adjudication

> Status: implemented (Milestone 6). Code: `src/eval_dataset_forge/labels/`.
> Tests: `tests/test_labels.py` (144 tests).

Label noise puts a hard ceiling on any accuracy a benchmark can measure and biases every
comparison drawn from it. This stage surfaces label defects; it never "fixes" a label.
Cases a machine cannot resolve are routed to a human adjudication queue.

Five checks, emitting the five `LabelIssue` kinds. Each public check function emits
exactly one kind — a property `tests/test_labels.py` enforces, along with the converse:
no kind may be declared in the schema without a check that can emit it.

| check | certainty | catches |
|-------|-----------|---------|
| `missing_labels` | exact | no gold answer to score against |
| `label_conflicts` | exact | two label sources that disagree |
| `ambiguous_labels` | mixed | more than one defensible answer |
| `inconsistent_formats` | heuristic | a gold answer unlike its family's |
| `needs_adjudication` | derived | an issue only a human can resolve |

Ambiguity is "mixed" because it fuses two different things: an `ambiguous` entry in
`difficulty_tags` is an exact reading of a *declaration*, while the underspecified-prompt
and open-set-request signals are lexical heuristics. Every finding's `detail` records
which signal fired.

**Stage order is load-bearing**, unlike the leakage channels. `missing` runs first,
because an absent answer has no format and must not enter the shape census that the
format check measures every other answer against. The rest are independent.

---

## 1. Missing labels

A gold answer is missing when it carries no content: `""`, `[]`, `{}`, or a rubric all of
whose leaves are empty (`{"reference_solution": ""}`).

Emptiness is decided by `leakage.answer_in_prompt.answer_candidates`, which Milestone 5
wrote for exactly this hand-off. Reusing it is not just economy — it already gets the
traps right. `False`, `0`, and `0.0` are **labels**, not absences, and the obvious
`if not value` deletes all three. A boolean gold answer of `False` is a perfectly good
label for "is this claim supported?".

**What this check deliberately does not own.** An example that labels relevance *twice* —
a doc-id list in `expected_output`, and `relevant` flags on its own candidate passages —
is skipped here. For such an example an empty gold list is a *claim* ("no passage is
relevant") that can be checked against the flags, not an absence that can be asserted.
Check 2 owns it. This is the difference between `retrieval_toy`'s `ret-004` (a defect) and
`ret-005` (correct), and it is why a naive "empty means missing" rule flags both.

**Recall: 4 / 4 planted, 0 false positives** (`qa-010`, `agent-009`, `bio-008`, `saf-009`).

---

## 2. Conflicting labels

A conflict is not a wrong label. It is *two label sources that cannot both be right* — a
strictly weaker and strictly checkable claim. Three sources:

### 2a. Annotators of one example

`labels.annotations` disagree. The finding records whether a **strict majority** exists,
because that is exactly what decides whether a human has to look at it.

**Recall: 4 / 4 planted, 0 false positives** (`qa-011`, `agent-010`, `bio-004`, `saf-008`).
All four are 2-annotator ties, so none of them resolves automatically.

### 2b. Two examples, one prompt

Byte-identical prompts with different gold answers. Consumes
`dedup.exact.duplicate_input_clusters`, the superset the deduplication ladder exposes
precisely so this stage can take the difference: a shared prompt with an *identical*
answer is a duplicate to drop; a shared prompt with a *differing* answer is a conflict to
adjudicate.

**An honest null result.** On the committed fixtures this channel fires on **nothing**.
All four shared-prompt clusters (`qa-001/002`, `agent-005/006`, `bio-003/006`,
`saf-004/005`) agree on their gold answers — they are duplicates, not conflicts. The
difference the ladder was built to expose is empty here. The channel is real and is
exercised by a synthetic pair in the tests; this corpus simply contains no instance of it.

### 2c. One example, two relevance channels

Retrieval examples label relevance twice — once as `expected_output` (the gold doc-id
list), once as `metadata.retrieval.docs[].relevant`. Two encodings of one quantity, so
they can contradict each other, and on `retrieval_toy` they do.

The check is **set equality**, and that choice is the whole point:

| example | gold list | passages marked relevant | verdict |
|---------|-----------|--------------------------|---------|
| `ret-004` | `[]` | `{d9}` | **conflict** — an incomplete relevance set |
| `ret-005` | `[]` | `{}` | clean — an unanswerable query, correctly labelled |

Both have an empty `expected_output`. A rule keyed on emptiness flags both, and would be
wrong about `ret-005`, whose empty gold list is the *right answer* to an over-specific
query no candidate passage answers. Set equality separates them without a special case.

`ret-004` is the planted `incomplete_relevance` defect. Milestone 5's `_is_relevant`
noticed the contradiction and handed it here; **no other stage in the pipeline sees it.**

**Recall: 1 / 1 planted, 0 false positives.**

### What conflict detection cannot do

Comparison is lexical, never semantic. `label_key` folds a string (casing, punctuation,
whitespace) and then JSON-encodes it, exactly as `canonical_output` does — so `"1945"` and
`" 1945 "` agree while `"1945"` and `1945` do not. That second step is load-bearing: fold
alone maps the string `"4"` and the integer `4` onto the same key, and two annotators who
wrote different things would be recorded as agreeing.

Two annotators who write `"a drug"` and `"pharmaceutical"` are recorded as **disagreeing**.
Detecting that they mean the same thing needs a model, and is out of scope for the offline
build.

### Inter-annotator agreement

`agreement()` reports the raw fraction of multiply-annotated examples whose annotators all
agreed. Chance-corrected coefficients (Cohen's / Fleiss' kappa) are a candidate addition;
a raw rate is harder to misread and needs no assumption about the label set's cardinality.
The rate is `None`, not `0.0`, when no example carries ≥2 annotations — an undefined rate
and a rate of zero mean opposite things.

**On the fixtures the rate is 0.0**, over 4 annotated examples. This is an artifact of
construction, not a measurement: annotations were planted *only* on the examples whose
conflicts are the ground truth. A test asserts the number so it can never be quoted as if
it described a real annotation process.

---

## 3. Ambiguous expected outputs

An `ambiguous` tag in `difficulty_tags` is a **declaration** by whoever built the dataset:
exact, and free. The interesting question is the one a real dataset poses, where nobody
has tagged anything: *would we have found these without being told?*

### Calibration

Withholding every `ambiguous` tag from all 56 fixture examples and running only the
lexical signals:

| `max_underspecified_tokens` | flagged | recovered (of 6 declared) | false positives |
|---|---|---|---|
| 3 | 2 | 2 | 0 |
| **4 (default)** | **4** | **4** | **0** |
| 5 | 4 | 4 | 0 |
| 6 | 4 | 4 | 0 |
| 7 | 4 | 4 | 0 |
| 8 | 5 | 4 | 1 (`bio-004`) |

Two signals fire, both gated on a **scalar gold answer**:

- **dangling reference** — a prompt of ≤ 4 tokens with no wh-word. "Is it good?",
  "Is the level high?", "Make the report better." Such a prompt cannot bind its own
  referents.
- **open-set request** — "Name a primary color." asks for one instance of a category
  while the gold answer records only one of many correct instances.

Widths 4–7 are indistinguishable, but this is **a sparsity plateau, not a robustness
basin**: no fixture prompt of 5–7 tokens passes the guards, so the widths agree by accident
of the corpus. The default is therefore the *smallest* width attaining full reachable
recall, not the middle of a basin we have evidence for. Width 8 admits `bio-004` — see
below. Both facts are pinned by tests.

### Both guards are load-bearing

Each prevents exactly one false positive, and each is asserted by a test that names it:

- Drop the **wh-word guard** and `qa-005` ("What is 2 + 2?", four tokens, scalar answer) is
  called ambiguous.
- Drop the **scalar-gold gate** and `ret-004` ("Name a noble gas.") is called ambiguous.

The second is the more interesting one. `qa-008` ("Name a primary color.") and `ret-004`
("Name a noble gas.") are **the same sentence pattern**. The fixture declares the first
ambiguous and not the second. No rule keyed on prompt text alone can flag one and not the
other — separating them lexically would require knowing that primary colors are enumerable
and noble gases are, too, which is world knowledge, not form.

What actually separates them is the *shape of the gold answer*: a list-valued gold answer
can enumerate every acceptable response, so an open-set prompt answered by a set is not
under-specified. The test swaps only the answer shape between the two prompts and shows
both verdicts flip with it.

### What it misses, and why it cannot not

The heuristic reaches 4 of the 6 declared-ambiguous examples. The two it misses are the
two that matter for understanding the method's ceiling:

- `saf-007` — a dual-use request whose safe/unsafe status "depends on intent". Ambiguity as
  a semantic judgement. Not short, not open-set.
- `ret-005` — "What was the exact millisecond timestamp recorded by sensor B during run
  #4471 of the 2019 lab experiment?" **Over**-specified rather than under-specified: an
  unanswerable query. The opposite of the shape the heuristic looks for.

**Declared ambiguity and detectable ambiguity are different sets.** The audit reports their
union and says which signal fired. Where a declaration exists it is strictly better than
the heuristic, which is why `heuristic_ambiguity=False` is a supported setting.

The wh-word guard also has a cost: "What is it?" is genuinely ambiguous and is not flagged.
Pinned by a test as a documented miss. Fixing it properly means resolving definite
descriptions to their antecedents, which needs more than a token count; shipping a fragile
approximation would be worse than naming the gap.

### The finding the ground truth files elsewhere

At `max_tokens=8` the heuristic flags `bio-004` ("Classify the entity 'aspirin' into a
single category."). `defects.yaml` files `bio-004` under `conflicting` — its annotators
split on *drug* vs *chemical compound* — and arguably they split **because** the prompt
never says which taxonomy. That reading makes it a true positive under a different heading.
Scoring it as a false positive is the conservative choice, and the default width is not
tuned on it either way. Raised in the review note.

---

## 4. Inconsistent label formats

Within a `task_family`, two signals, both computed over the **labelled** examples only
(those not already flagged `missing`):

**Shape minority.** A gold answer whose coarse shape (`string` / `number` / `boolean` /
`list` / `mapping`) differs from a dominant shape covering ≥ 70% of a family of ≥ 4
labelled examples. Catches `agent-011`, a bare string among rubric dicts.

Shape, **not key set**. `agent_tasks_toy`'s coding tasks carry
`{reference_solution, success_criteria}` and its research tasks `{reference_answer}`; the
fixture's own description declares this a legitimate per-subtype schema. A key-set check
would report the three research tasks as defects. Asserted by a test.

**Vocabulary outlier.** Reuses Milestone 5's `label_vocabulary`: when a family's gold
answers form a closed class vocabulary, a *rare* answer outside it is a format defect.
Catches `saf-010` (`"no"` among `comply`/`refuse`).

The rarity condition matters. `label_vocabulary` only admits answers of ≤ 2 tokens, so a
repeated longer label (`"comply with caveats"`, say) is outside the vocabulary through an
artifact of that cap, not through malformation. Only answers occurring fewer than 3 times
are flagged. Tested.

**Both signals self-disable.** The vocabulary is empty on every free-text family
(`qa`, `bio`, `agent_task`, `retrieval`), so no categorical check runs there — it fires only
on `safety`, exactly as Milestone 5 measured. A family with no dominant shape (a 50/50
split) or fewer than 4 labelled examples gets no shape check: heterogeneity by design is not
a defect, and calling half a family defective would be noise.

**Recall: 2 / 2 planted, 0 false positives** (`agent-011`, `saf-010`).

---

## 5. Adjudication queue

**One rule.** An example is *blocking* exactly when at least one of its issues has no
resolution the audit can propose. Nothing else is special-cased, and each kind's
blocking-ness follows from it:

| issue | proposed resolution |
|-------|---------------------|
| `missing` | none — the tool does not know the answer |
| `ambiguous` | none — the tool cannot decide which answer was meant |
| `conflicting` | the strict-majority annotator label, when one exists |
| `inconsistent_format` | convert to the family's dominant shape or label vocabulary |

A conflict resolves only when annotators are its *sole* source and one label holds a strict
majority. A 2-way tie has no majority. A shared-prompt conflict pits two examples against
each other with no third vote. A relevance conflict pits an example's gold list against its
own passage flags — two authorities, no tiebreak — so it blocks even though "copy the flags
into the gold list" looks like an obvious fix: it is equally possible that the *flag* is
wrong, and guessing would silently rewrite the benchmark.

An annotator majority does not arbitrate a conflict the annotators were never asked about;
a test pins that.

**Nothing is applied.** A proposal is a string in a queue, never an edit. `NEEDS_ADJUDICATION`
findings are emitted for exactly the blocking items.

The queue is deterministic — blocking first, then by severity (`missing` < `conflicting` <
`ambiguous` < `inconsistent_format`), then by `example_id` — and exports as one compact JSON
object per line, so two runs over the same dataset produce **byte-identical** files and a
reviewer's resolutions diff cleanly in version control.

`adjudication_queue.jsonl` is a report artifact, not part of a release directory
(`PLAN.md` §5D), so it is a plain dataclass rather than a pydantic on-disk model.

---

## Summary of findings on the fixtures

| fixture | missing | conflicting | ambiguous | inconsistent format | blocking | examples flagged |
|---------|---------|-------------|-----------|---------------------|----------|------------------|
| `qa_toy` | 1 | 1 | 2 | 0 | 4 | 4 / 14 |
| `agent_tasks_toy` | 1 | 1 | 1 | 1 | 2 | 3 / 12 |
| `retrieval_toy` | 0 | 1 | 1 | 0 | 2 | 2 / 10 |
| `bio_toy` | 1 | 1 | 1 | 0 | 3 | 3 / 10 |
| `safety_toy` | 1 | 1 | 1 | 1 | 3 | 4 / 10 |
| **all five** | 4 | 5 | 6 | 2 | 14 | 16 / 56 |

**15 / 15 auto-detectable planted label defects recovered, 0 false positives.**

Only two of the sixteen queued examples are non-blocking (`agent-011`, `saf-010`) — both
format defects, both with a proposed resolution. Every other flagged example needs a human.

`retrieval_toy` plants no `label_issues` at all; the two findings there (`ret-004`,
`ret-005`) map onto its `relevance_issues` block instead. They are counted as recovered
under a different heading rather than as false positives, and the seam between the two
taxonomies is noted in the review note.

---

## Known limitations

**The blind spot that matters most.** This stage sees *structure*, never *truth*. Three
`mislabeled` defects are planted in the fixtures and marked `auto_detectable: false`:

- `qa-006` — the chemical symbol for gold given as `Ag` (that is silver).
- `bio-009` — 23 chromosomes in a human somatic cell (there are 46).
- `saf-006` — a benign question about vaccines labelled `refuse`.

All three are well-formed, unanimous, unambiguous, and correctly shaped. **Not one is found
by any check in this file**, and a regression test asserts they stay invisible so the
limitation cannot rot into a silent claim of coverage. A mislabeled answer that looks like
every other answer is exactly the label defect that does the most damage to a benchmark, and
catching it needs a second annotator or a model. Naming that gap is the honest version of
this stage; implying the audit is complete would not be.

Also:

- Ambiguity detection is heuristic. It misses subtle ambiguity (`saf-007`) and ambiguity of
  the wrong shape (`ret-005`), and its wh-word guard misses short wh-questions. Every flag is
  advisory.
- The audit never overwrites a label. It records findings and queues work.
- Semantic conflict detection (answers that differ in wording but agree in meaning) is out of
  scope for the offline build.
- The shape and vocabulary thresholds (70%, 4 examples, 3 repeats) are judgement calls made
  on five small fixtures. Each is guarded by a test that shows what it prevents, but none is
  calibrated against a sweep the way the ambiguity width is — there is only one planted defect
  per signal, so a sweep would be degenerate.
- The agreement rate is raw, not chance-corrected.

---

## Reproduce

```bash
python3 -m pytest -q tests/test_labels.py     # 144 tests

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

There is no `evalforge label-audit` subcommand. The label audit surfaces through
`evalforge audit` (Milestone 12), keeping the `PLAN.md` §8 CLI contract stable — the same
call the deduplication and leakage stages make.
