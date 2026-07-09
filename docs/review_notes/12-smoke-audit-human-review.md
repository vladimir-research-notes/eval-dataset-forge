# Human review: Milestone 12 — Smoke audit & results

TL;DR

`evalforge audit` and `evalforge smoke` are implemented; the CLI has no stubs left. The smoke
audit runs the whole pipeline over all five fixtures plus the pooled corpus, fully offline, and
scores every finding against each fixture's planted `defects.yaml`. **40 of 40 detectable planted
defects are recovered, 0 missed, 5 unexplained assertions, 0 negative controls violated.**

The interesting part is not that number. Writing the scorer forced the question the hand-written
tables in Milestones 4–6 never had to answer: *what is a correct scoring rule for a defect
detector?* Three answers, each measured rather than asserted:

1. **Recall is not one.** A detector that flags everything scores recall **1.000** — the same as
   the pipeline. That control ships inside the report and is scored on every run. Precision
   separates them by 9× to 190×.
2. **A cluster is a conjunction of assertions.** A *k*-member duplicate cluster asserts *k(k−1)/2*
   pairs, so it is scored on those pairs. This is not bookkeeping: `qa_toy`'s near stage merges
   `{qa-003, qa-004, qa-012}` and thereby asserts `qa-004 ≈ qa-012`, which nothing plants. A
   cluster-level view files that as correct and never shows the claim.
3. **The denominator is chosen by the authors of the detectors.** 45 planted, 40 in the
   denominator. Every exclusion is listed by id with which of two distinct reasons applies
   (`not-auto-detectable` = the fixture's claim; `no-channel` = the scorer's wiring), and recall
   over *all* planted defects (0.909) is printed beside recall over detectable ones (1.000).

`audit.md` — listed in `docs/PLAN.md` §5D since Milestone 0, shipped in no release until now — is
in the bundle. The demo release was rebuilt in place at v0.1.0 (see "Technical risks").

1706 tests pass (1659 prior + 47 new). `ruff check .` clean. No network, no model, no paid API.

What changed

- **New** `src/eval_dataset_forge/audit.py` — the audit pipeline over a dataset *as it stands*.
  `assemble_audit` is now the single place an `AuditResult` is built; `datasets/release.py`
  delegates to it. Parameterized by a new `SplitFacts` view so a constructed `SplitReport` and a
  split read off disk can both feed it. `split_statistics["source"]` records which
  (`constructed` / `observed`).
- **New** `src/eval_dataset_forge/reports/audit_report.py` — `audit.md`, rendered from the
  `AuditResult` alone. The certainty vocabulary and per-kind counters moved here out of
  `dataset_card.py`, which now imports them, so the two findings tables cannot drift.
- **New** `src/eval_dataset_forge/scoring.py` — ground-truth parsing and the scoring rule
  (pair decomposition, unit-preserving corroboration, degenerate control, explicit exclusions).
- **New** `src/eval_dataset_forge/smoke.py` — `evalforge smoke`, its report document, and the
  per-fixture audit writer.
- **Changed** `datasets/release.py` — emits `audit.md`; `_dependence`/`_recommendations`/
  `build_audit` internals moved to `audit.py`; `REPRODUCE.md` no longer names a missing artifact.
- **Changed** `cli.py` — `_PENDING` removed; `audit` and `smoke` implemented with `--fail-on-*`
  gates consistent with the other commands.
- **Rebuilt** `datasets/releases/qa_toy_v0.1.0/` — gains `audit.md`; `REPRODUCE.md`, `audit.json`,
  `dataset_card.md`, `manifest.json` change accordingly. `examples.jsonl` and the split files are
  byte-identical.
- **Docs** `docs/RESULTS.md` rewritten (no longer "pending Milestone 12"); site pages updated.

What should be checked later

1. **`ret-005` — a new ground-truth dispute, deliberately not settled.** `labels.ambiguous` flags
   `ret-005`, and the scorer records it as the corpus's only *new* unexplained assertion. The
   detector is right: `ret-005` carries `difficulty_tags: [hard, ambiguous]`, and the ambiguity
   check is an exact reading of that declaration. But `defects.yaml` files `ret-005` under
   `relevance_issues/over_specific_query`, not under `label_issues/ambiguous`. So the ground truth
   is incomplete, not the detector wrong. **Question for a human:** should `retrieval_toy`'s
   `defects.yaml` gain a `label_issues: [{id: ret-005, kind: ambiguous}]` entry? Doing so would
   move `labels.ambiguous` precision from 0.833 to 1.000 — which is exactly why I did not do it
   unilaterally. Editing ground truth to raise a score is the failure mode this project exists to
   detect.

2. **Is `ret-005` a sound negative control at all?** It is planted so that a rule keyed on empty
   `expected_output` is caught flagging it, and `CONTROL_CHANNELS` scopes that to `labels.missing`
   alone. That scoping is a judgement: "must not be flagged by *anything*" is untestable, but
   naming one channel may be too narrow. Worth a second opinion.

3. **The `agent-003`/`agent-004` dispute is still open**, now surfaced by two channels rather than
   one. Milestone 5 flagged it; the scorer independently marks it unexplained under both
   `dedup.near` and `leakage.train_test_overlap`. Unchanged, still unresolved.

4. **The two-precision presentation.** Each channel reports a strict kind-specific `precision`
   (`None` when nothing of that kind is planted, because `0.0` would be the only value a non-silent
   detector could score) and an `explained` fraction / `unexplained` count. Is that two numbers too
   many? The alternative — one blended number — hides either the honest null results
   (`leakage.ngram_overlap` fires four times on a corpus planting none of its kind, and all four
   are corroborations) or the inventions.

5. **Recall over all planted defects is 0.909, not 1.000.** The gap is four defects: three
   `mislabeled` (declared undetectable by the fixtures, and a regression test pins that they stay
   invisible) and one `paraphrased_documents` (no detector exists). A reader should decide whether
   0.909 or 1.000 is the number they want quoted.

6. **The pooled corpus is unscored**, and running it anyway found something. Justification for not
   scoring: cross-fixture pairs are outside every `defects.yaml`, so scoring would measure the
   ground truth's silence. But the pooled audit reports 10 duplicate clusters against the
   per-fixture sum of 9, and the extra one is **not** cross-fixture — it is
   `{saf-004, saf-005, saf-007}`, entirely inside `safety_toy`. See "A near-duplicate verdict is
   not a property of the pair", below. A human should decide whether that finding changes the
   argument for leaving the pooled corpus unscored (I think it strengthens it: the pooled ladder
   answers a different question about the same rows).

7. **A near-duplicate verdict is not a property of the pair — it is a property of the pair *and
   its corpus*.** `sim(saf-004, saf-007)` is **0.2866** inside `safety_toy` and **0.3438** inside
   the pooled corpus, straddling the 0.30 threshold. Neither prompt changed. TF-IDF's IDF term is
   computed over the document collection, so the boilerplate the two safety prompts share
   ("SYNTHETIC…PLACEHOLDER", "no real content") is common within `safety_toy` and rare against the
   pooled qa/agent/bio/retrieval vocabulary — and rarer means heavier. Consequences a human should
   weigh:
   - **Deduplication is not composable.** Auditing shards and unioning the findings is not the same
     as auditing the union. Nothing in `docs/DEDUPLICATION.md` says so.
   - **Appending unrelated examples to a corpus can create duplicate clusters among the examples
     that did not change.** Since the adversarial split builder derives its blocks from duplicate
     clusters, and `eval_id` folds the split assignment in, appending unrelated data can move the
     split of an untouched example and therefore change `eval_id`. That is a Milestone 10 identity
     claim reached from a Milestone 4 implementation detail, and it deserves a documented
     experiment of its own rather than a paragraph here.
   - The Jaccard fallback has no IDF and is presumably immune. **Unverified.** Worth measuring: if
     so, the "optional accelerator" `scikit-learn` silently changes the *semantics* of a duplicate
     report, not just its speed.
   `test_a_near_duplicate_verdict_is_a_property_of_the_corpus_not_of_the_pair` pins the measurement.

8. **Single-linkage clusters can re-cover a pair the exact stage owns.** `dedup/__init__.py` states
   the three rungs are "disjoint by construction: … a near finding never covers a pair the earlier
   stages already resolved". That is true of the near stage's *edges* (`near_duplicates` skips a
   pair whose normalized signatures match) but not of its *clusters*: in the pooled corpus the near
   cluster `{saf-004, saf-005, saf-007}` is the transitive closure of the edges
   `saf-004–saf-007` and `saf-005–saf-007`, and it contains the pair `saf-004 + saf-005`, which the
   exact stage reports separately. `DedupReport.counts()` still adds up at the *cluster* level;
   at the *pair* level the exact pair is covered twice. This is pre-existing and touches nothing
   scored (no per-fixture near cluster overlaps an exact one), but the docstring's claim is
   stronger than the code's guarantee. **Not changed unilaterally** — the fix could be to the
   sentence or to the clustering, and that is a design decision.

Technical risks

- **The demo release was rebuilt in place at v0.1.0, not bumped to v0.1.1.** Milestone 10's
  `compare()` calls adding `audit.md` a `patch` ("packaging only: hashes"; `content_id` and
  `eval_id` identical), so scores measured on the old bundle stand on the new one. The release has
  never been distributed and is a build artifact of the repo, regenerated by its own `REPRODUCE.md`
  command. **But**: git history now contains two different byte-sets both calling themselves
  `qa_toy v0.1.0`. If the demo release is ever treated as published, this is the wrong call.
  `test_adding_audit_md_is_a_packaging_change_not_a_data_change` pins the reasoning; a human should
  confirm the conclusion.
- **`audit.json` bytes changed** for reasons beyond `audit.md`: `split_statistics` gained `source`
  and `unassigned` keys, and two recommendation strings were reworded (`--drop` → `evalforge
  release --drop`) now that the same text serves both `audit` and `release`. `dataset_card.md`
  changed downstream. All still `patch`-class, but the manifest hash churn is wider than "one new
  file".
- **`SplitFacts.from_split_report` is annotated `report` with no type**, to avoid a circular import
  between `audit.py` and `splits/`. A `Protocol` would be cleaner.
- **`scoring.CHANNEL_OF` maps `relevance_issues/incomplete_relevance` → `labels.conflicting`**, a
  cross-section claim. It is correct (the conflict channel compares the two relevance channels as
  sets) but it is a hard-coded piece of knowledge about *which* check finds *which* planted kind,
  and it will rot if the label audit is restructured. A test would catch it.
- **`degenerate_predictions` asserts `C(n,2)` pairs per pair-channel.** At 56 examples that is
  1540 sets; at 10⁴ examples it is 5×10⁷ and the control becomes unrunnable. It is a fixture-scale
  diagnostic, not a production one, and nothing says so in the code.
- **`evalforge smoke` requires a `defects.yaml` per fixture and raises otherwise.** Intentional —
  an unscored fixture looks exactly like a passing one — but it means the command cannot be pointed
  at an arbitrary dataset directory.

Commands to verify

```bash
python3 -m pytest -q                                        # 1706 tests
python3 -m pytest -q tests/test_audit.py                    # 47 tests
ruff check .

# The smoke audit, with both regression gates armed.
PYTHONPATH=src python3 -m eval_dataset_forge smoke \
  --out results/reports/smoke_audit.md \
  --fail-on-missed-defect --fail-on-control-violation       # exit 0

# The control that makes the recall meaningful. Both score 1.000.
PYTHONPATH=src python3 -c "
from eval_dataset_forge.smoke import run_smoke
r = run_smoke('datasets/fixtures')
print('pipeline  recall:', r.recall(any_channel=True))
print('flag-all  recall:', r.degenerate_recall())
real, ctrl = r.channel_totals(), r.channel_totals(degenerate=True)
for n in ('dedup.exact', 'dedup.near', 'labels.missing'):
    print(f'{n:16s} precision: {real[n].precision:.3f} vs {ctrl[n].precision:.3f}')
"

# An unsplit corpus: four checks return zero by vacuity, and the audit says so.
PYTHONPATH=src python3 -m eval_dataset_forge audit \
  --dataset datasets/fixtures/qa_toy/examples.jsonl --out results/audits/qa_toy/

# The rebuilt demo release still verifies, and its card still regenerates.
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0                 # 10/10 checks
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check         # exit 0
```

Human feedback
<!-- Human will fill this later. -->
