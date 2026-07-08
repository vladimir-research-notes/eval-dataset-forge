# Human review: Milestone 5 — Leakage checks

TL;DR
The third audit stage landed: a `leakage/` package with four channels — cross-split
overlap (the dedup ladder's clusters filtered to those spanning two splits),
answer-in-prompt containment, reference n-gram overlap, and retrieval-document
contamination. Each emits exactly one `LeakageKind`, enforced by a test. Unlike the
dedup ladder the channels are **deliberately non-disjoint**, and it pays off: the one
planted defect the ladder is provably blind to (`ret-001`/`ret-002`, a deep query
paraphrase) is recovered by the passage-level check, because both examples cite the
identical gold passage under different `doc_id`s. Per-kind recall 11/12; across
channels 12/12. Thresholds are **calibrated against planted ground truth**, not
asserted. Full suite: 378 passed; fixtures validate; ruff clean.

What changed
- Added `src/eval_dataset_forge/leakage/{__init__,train_test_overlap,answer_in_prompt,
  ngram_overlap,retrieval_overlap}.py`.
- Added `tests/test_leakage.py` (111 tests), including regression tests that pin every
  documented failure mode and both rejected calibration settings.
- Rewrote `docs/LEAKAGE.md` around the measured sweeps; added the leakage section to
  `docs/RESULTS.md`; updated `docs/LIMITATIONS.md`.
- Rewrote `site/leakage.html`; status updates on `audit-pipeline.html`,
  `milestones.html`, `index.html`, `results.html`, `limitations.html`.
- No CLI change: leakage surfaces through `evalforge audit` (Milestone 12), keeping the
  `PLAN.md` §8 CLI contract stable.

**Deviations from the design doc**, both driven by the code, both now documented:
- `PLAN.md` §4 listed `leakage/contamination.py`. **I edited `PLAN.md`** to name
  `answer_in_prompt.py` instead, so each module maps 1:1 onto a `LeakageKind` — a
  property a test can check, which "contamination" could not. Editing the project
  contract is not something to do quietly; please confirm the rename is what you want
  rather than a `contamination.py` that aggregates two kinds.
- `LEAKAGE.md` §4 proposed flagging "the gold answer string embedded in a passage". That
  check is **wrong** for retrieval data — a relevant passage containing the answer is the
  *definition* of relevance, so it would fire on every correctly labelled example. Dropped
  and the reasoning written down, rather than implemented and quietly never firing.

What should be checked later
- **The label-vocabulary guard (most important).** It is the only place this milestone
  decides *not* to look at something. A gold answer is skipped when it is short (≤2
  tokens), repeated (≥3×), and the set of such answers covers ≥50% of the dataset. The
  coverage condition is load-bearing — without it, `qa_toy`'s `"William Shakespeare"`
  (3 occurrences, 21% coverage) would be dismissed as a class label. It fires on
  `safety_toy` (`{comply, refuse}`) and changes nothing there; on a synthetic sentiment
  set it suppresses 8 of 8 false positives. But **50% is a judgement call made on five
  small fixtures**, and its cost is a true negative: a genuinely leaked class label in a
  genuine classification dataset goes unreported. Should the release builder require the
  vocabulary to be declared explicitly per dataset instead of inferred?
- **`agent-003`/`agent-004`, again.** Milestone 4's review note raised this pair; this
  milestone reports it as a cross-split overlap. It straddles train/test with an identical
  reference solution, prompts differing only by an appended hint. `defects.yaml` files it
  under `leakage: answer_in_prompt`, so it scores as a false positive against *both* the
  dedup and the leakage ground truth — while being, on the merits, a real cross-split
  overlap. I did not edit `defects.yaml`. Please decide whether ground truth should record
  it in both categories.
- **`ret-001`/`ret-002` and the `auto_detectable` flag.** Still unresolved from Milestone
  4, and now more interesting: the pair *is* detected, just not by the channel its ground
  truth names. If the ground-truth schema gains a "detected by another channel" notion,
  the recall accounting in `RESULTS.md` should follow. Today it reports both numbers
  (per-kind 11/12, any-channel 12/12) rather than picking the flattering one.
- **Calibration is in-sample.** `n=2 @ 0.5` for answer containment and `n=8` for
  reference overlap are the knees of sweeps over the same 56 synthetic examples the
  recall table reports on. The answer threshold at least sits mid-plateau (weakest true
  positive 0.600, strongest non-leak 0.333), which is more margin than Milestone 4's
  dedup thresholds have. Sanity-check on a real corpus.
- **The n-gram channel is untested against its actual use case.** It exists for external
  reference corpora; on the fixtures it is provably redundant with the exact overlap
  channel (asserted by a test). Worth deciding whether a small public-safe reference file
  should be committed so the channel is exercised end to end, or whether the redundancy
  test is enough.
- **The artifact marker is a curated regex**, not a detector: `answer:`, `label:`,
  `ground truth:`, `gold answer:`, `relevant=`. It matches exactly once across every
  passage and prompt in all five fixtures — the planted `[ANSWER: Mars]` — which is a good
  specificity signal on 56 examples and no guarantee on a real corpus. A corpus leaking
  labels under another convention passes clean.
- **`LeakageReport` shape.** An ephemeral dataclass, not a versioned artifact.
  `to_dict()` was written to slot into `AuditResult.leakage_flags` (a flat list —
  `report.findings` provides it, ordered by channel). Verify that is what Milestone 12
  expects, and note that `cross_split_sources` and `duplicate_documents` have no home in
  `AuditResult` today, exactly as `normalized` dedup findings do not.
- **Severity.** `dev`/`test` overlap and `train`/`test` overlap share one `LeakageKind`.
  Defensible (both are "tuned on it, then scored on it") but a release gate may want to
  distinguish them.

Technical risks
- Low. Pure Python plus the optional scikit-learn accelerator inherited from dedup; no
  network, no paid API, no filesystem writes, no mutation of input examples. The fallback
  was verified end to end by blocking `sklearn` at the import hook, not merely by
  monkeypatching the capability check: `bio_toy` correctly drops from 2 cross-split
  findings to 1, every other channel is unchanged, and `sklearn` never enters
  `sys.modules`.
- The real risk is a *false sense of safety*. Contamination is measurable only against
  text we are handed; a model's pretraining corpus is not ours to read. **Absence of a
  flag is not evidence of no contamination.** Stated in `LEAKAGE.md`, on the Leakage and
  Limitations pages, and in the package docstring.
- Answer-in-prompt is lexical: it sees a *quoted* answer and misses an *entailed* one
  ("Which country's capital is Berlin?" → `Germany`), and it cannot distinguish a leak
  from a legitimate restatement ("extract the year 2020 from: … in 2020 …"). Findings are
  for review, never for removal — nothing in this milestone deletes or rewrites an
  example.
- Cross-split overlap inherits every dedup blind spot by construction. That is the design
  (one detector, one place to fix it), but it means the leakage layer cannot be more
  complete than the dedup layer on its main channel.
- Single-linkage transitivity leaks through: `qa_toy`'s reported overlap cluster is
  `{qa-003, qa-004, qa-012}`, not the planted pair. Scoring uses coverage ("does a
  reported cluster contain both planted members?"), the same convention Milestone 4 used.
- No bug of the `canonical_output` class was found this milestone. The closest call was
  the module/function name collision (`leakage.answer_in_prompt` as both a module and a
  function would have made `from eval_dataset_forge.leakage import answer_in_prompt`
  ambiguous, and the resolution order surprising); the functions were renamed
  (`leaked_answers`, `reference_overlap`, `cross_split_clusters`) so every module name
  resolves to a module.

Commands to verify
```
python3 -m pytest -q
python3 -m pytest -q tests/test_leakage.py
python3 scripts/validate_fixtures.py
ruff check .

PYTHONPATH=src python3 -c "
from eval_dataset_forge.datasets import loader
from eval_dataset_forge.leakage import find_leakage, format_report
examples = loader.load_examples(loader.FIXTURES_ROOT / 'qa_toy' / 'examples.jsonl')
print(format_report(find_leakage(examples)))
"
```

Human feedback
<!-- Human will fill this later. -->
