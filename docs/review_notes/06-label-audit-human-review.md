# Human review: Milestone 6 — Label audit

TL;DR
The fourth audit stage landed: a `labels/` package with five checks — missing, conflicting
(three sources), ambiguous (a declaration plus two lexical signals), inconsistent format
(shape plus vocabulary), and a derived `needs_adjudication` flag — plus a deterministic
`adjudication_queue.jsonl` export. Each check emits exactly one `LabelIssue`, enforced by a
test, as is the converse (no kind without a producer). **15/15 auto-detectable planted label
defects recovered, 0 false positives.** The fuzzy check is calibrated against *withheld*
ground truth: strip every `ambiguous` tag and the lexical heuristic recovers 4 of 6 with no
false positives, with both guards shown load-bearing. The queue follows one rule — blocking
iff the audit can propose no resolution — rather than a table of special cases. Full suite:
522 passed; fixtures validate; ruff clean.

What changed
- Added `src/eval_dataset_forge/labels/{__init__,audit,disagreement,adjudication}.py` —
  exactly the tree `PLAN.md` §4 specifies, so no contract edit was needed this milestone.
- Added `tests/test_labels.py` (144 tests), including the calibration sweep, both guard
  ablations, and regression tests pinning every documented failure mode and honest negative.
- Rewrote `docs/LABEL_AUDIT.md` around the measured results; added the label-audit section to
  `docs/RESULTS.md`; added a "Label truth is invisible to us" section to `docs/LIMITATIONS.md`.
- Rewrote `site/label-audit.html`; status updates on `audit-pipeline.html`, `milestones.html`,
  `index.html`, `results.html`, `limitations.html`.
- No CLI change: the label audit surfaces through `evalforge audit` (Milestone 12), keeping
  the `PLAN.md` §8 CLI contract stable — same call as Milestones 4 and 5.

**One refinement to the design doc**, driven by the code:
- `LABEL_AUDIT.md` §1 defined missing as "`expected_output` … absent or empty". Taken
  literally that rule is **wrong** on retrieval data: `ret-005`'s empty gold list is the
  *correct* answer to an unanswerable query. The check was narrowed — an empty container is
  handed to the relevance comparison, which has the evidence to decide it — and the reasoning
  written down rather than papered over. I did not edit `PLAN.md`.

**A bug the tests found.** `label_key` folded strings through `normalize_text` and sent
everything else through `canonical_output`, so the string `"4"` and the integer `4` both keyed
to `4` — two annotators who wrote different things would have been recorded as **agreeing**.
This is precisely the collision `canonical_output` exists to prevent, reintroduced by
bypassing it. The module docstring asserted the correct behaviour before the code implemented
it, and a test written to check that claim caught the gap. Fixed by folding *then* encoding.

What should be checked later
- **`bio-004` and the taxonomy seam (most important).** At `max_tokens=8` the ambiguity
  heuristic flags `bio-004` ("Classify the entity 'aspirin' into a single category."), which
  `defects.yaml` files under `conflicting`. Its annotators split on *drug* vs *chemical
  compound* — arguably **because** the prompt never says which taxonomy. Under that reading it
  is a true positive filed under a different heading, and the "false positive" at width 8 is an
  artifact of the ground truth's categories rather than of the detector. I scored it
  conservatively as a false positive, did not tune the default on it, and did not edit ground
  truth. Please decide whether `defects.yaml` should record a defect under more than one kind.
- **The same seam in `retrieval_toy`.** That fixture plants **no `label_issues` at all**; its
  two label findings (`ret-004`, `ret-005`) map onto its `relevance_issues` block instead. I
  counted them as recovered under a different heading rather than as false positives — a
  judgement call. This is the third milestone in a row to hit a ground-truth taxonomy seam
  (Milestone 4's `ret-001`/`ret-002`, Milestone 5's `agent-003`/`agent-004`); it may be worth
  a single decision about whether `defects.yaml` categories are exclusive.
- **The format thresholds are unswept.** 70% dominant shape share, ≥4 labelled examples per
  family, <3 occurrences for a vocabulary outlier. Each is guarded by a test showing what it
  prevents (a 50/50 family, a tiny family, a repeated out-of-vocabulary label), but there is
  exactly **one planted defect per format signal**, so a calibration sweep would be degenerate
  and I did not fake one. These are the least-evidenced numbers in the milestone. Sanity-check
  them on a real corpus.
- **`AMBIGUOUS` blocks the queue.** The one-rule design ("blocking iff no resolution can be
  proposed") makes every ambiguous example blocking, including ones the dataset author
  *declared* ambiguous and knowingly kept. That is defensible — a reviewer should decide whether
  such an example belongs in a test split — but it means a dataset that honestly tags its
  ambiguity gets a longer blocking queue than one that says nothing. Confirm that is the
  incentive you want, or add a "declared and accepted" state.
- **A relevance conflict blocks despite an obvious-looking fix.** "Copy the flags into the gold
  list" would resolve `ret-004` automatically. I made it blocking because it is equally possible
  the *flag* is wrong as the gold list, and guessing would silently rewrite the benchmark. If
  the release builder wants an auto-repair path, this is where it goes — and it should be
  opt-in.
- **Calibration is in-sample.** `max_underspecified_tokens=4` is the knee of a sweep over the
  same 56 synthetic examples the recall table reports on. Worse than Milestone 5's answer
  threshold, which at least sat mid-plateau: here **widths 4–7 agree only because no fixture
  prompt of 5–7 tokens passes the guards**. That is sparsity, not a basin, and it is stated as
  such in the docstring, the doc, the website, and a test name. Treat the width as unvalidated
  outside these fixtures.
- **The wh-word guard's cost.** "What is it?" is genuinely ambiguous and is not flagged. Fixing
  it properly means resolving definite descriptions to antecedents, which a token count cannot
  do; I pinned the miss with a test rather than shipping a fragile approximation. Worth deciding
  whether the guard should be a *penalty* rather than a veto.
- **`LabelAuditReport` shape.** An ephemeral dataclass, not a versioned artifact. `to_dict()`
  was written to slot into `AuditResult.label_warnings` (a flat list — `report.findings` provides
  it, ordered by kind, with the derived `needs_adjudication` findings last). Verify that is what
  Milestone 12 expects. Note that `agreement`, `expectations`, and `queue` have no home in
  `AuditResult` today, exactly as `cross_split_sources` did not in Milestone 5. The queue is a
  report artifact under the gitignored `results/` tree, not part of a release directory
  (`PLAN.md` §5D) — confirm Milestone 10 agrees, since `LABEL_AUDIT.md` could be read as
  implying the queue ships with a release.
- **`AdjudicationItem` is a plain dataclass, not a pydantic model.** Chosen for consistency with
  `DedupReport`/`LeakageReport` and because the queue is not a release artifact. If it ever
  becomes one, it should move into `schemas.py` (additively — no `SCHEMA_VERSION` bump needed for
  a new model).

Technical risks
- Low. Pure Python; no new dependencies, no network, no paid API, no mutation of input examples.
  The only filesystem write is `write_queue()`, which the caller points at an explicit path. The
  optional scikit-learn accelerator is not reached by this stage at all.
- **The real risk is a false sense of coverage.** This stage sees *structure*, never *truth*. A
  mislabeled gold answer that is well-formed, unanimous, unambiguous, and correctly shaped is
  invisible to every check here — and it is the label defect that does the most damage to a
  benchmark. Three are planted (`qa-006`: the chemical symbol for gold given as `Ag`; `bio-009`:
  23 chromosomes in a human somatic cell; `saf-006`: a benign vaccine question labelled `refuse`),
  all marked `auto_detectable: false`, and **none is found**. A regression test asserts they stay
  invisible so the limitation cannot rot into a silent claim of coverage. Stated in
  `LABEL_AUDIT.md`, `RESULTS.md`, `LIMITATIONS.md`, both website pages, and the package docstring.
- Conflict detection is lexical. Two annotators who write "a drug" and "pharmaceutical" are
  recorded as disagreeing; the `bio-004` finding is exactly this. Semantic agreement needs a model
  and is an explicit non-goal.
- `shared_prompt_conflicts` keys on *byte-identical* prompts (it consumes
  `dedup.duplicate_input_clusters`). A conflict between two examples whose prompts differ only in
  whitespace is invisible to it. Deliberate — the normalized dedup stage owns that folding, and
  folding a prompt before declaring its answers in conflict would merge questions that differ by
  punctuation ("C++" vs "C").
- Nothing in this milestone deletes, rewrites, or auto-corrects a label. A proposed resolution is
  a string in a queue, never an edit.
- `agreement()` reports **0.0** on the fixtures. This is an artifact of construction — annotations
  were planted only where a conflict was planted — not a measurement of any annotation process. A
  test asserts the number so it can never be quoted as if it were one.

Commands to verify
```
python3 -m pytest -q
python3 -m pytest -q tests/test_labels.py
python3 scripts/validate_fixtures.py
ruff check .

PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.labels import audit_labels, format_report, write_queue
examples = loader.load_examples(loader.FIXTURES_ROOT / 'safety_toy' / 'examples.jsonl')
report = audit_labels(examples)
print(format_report(report))
write_queue(report.queue, 'results/audits/safety_toy/adjudication_queue.jsonl')
"
```

Human feedback
<!-- Human will fill this later. -->
