# Human review: Milestone 13 — Integration contracts

TL;DR

Five downstream consumers now have executable contracts, and `evalforge gate --release <dir>
--profile <name>` turns one into an exit code. The policy lives with the consumer, never in the
release: a release carrying its own pass/fail verdict would be a dataset grading itself.

The interesting part is not the command. Writing the contracts down forced four questions, and
the answers are measurements, not assertions:

1. **A gate needs a third verdict.** Milestone 12 showed four leakage checks return zero *by
   vacuity* on an unsplit corpus and printed a warning. A boolean gate that maps
   `leakage_flags == [] → PASS` converts vacuity into permission. `UNDECIDABLE` is now a value,
   it never passes, and `allow_undecidable: true` is a decision a policy reviewer can see.

2. **`scikit-learn` is not an "optional accelerator"; it is a different detector.** Rebuilding
   `bio_toy` from identical bytes, an identical command and an identical seed yields the same
   `content_id` and a **different `eval_id`**, a different held-out set, and 9 blocks instead of
   8, purely because the package was importable. `REPRODUCE.md` claimed the build was
   "deterministic given the same inputs and seed". It now carries an `Environment` section.
   `verify-release` runs no detector, so **verification is environment-independent where
   reproduction is not** — which is the only reason a release built on one machine can be
   trusted on another.

3. **Neither block counts nor gate verdicts survive a union.** The five fixtures hold 45
   independent blocks apart and **44** pooled at 56 examples either way, with **zero
   cross-fixture duplicate edges** — merging five unrelated datasets changed one dataset's
   verdict *inside itself*. `safety_toy` passes a one-cluster near-duplicate budget alone and
   the same ten rows fail it pooled. Under `jaccard` both properties hold, so composability is a
   property of the similarity function. This settles Milestone 12's open review question 7,
   which flagged the Jaccard path as "presumably immune. **Unverified.**"

4. **And a gate verdict does not survive a re-split either, for four of the fourteen
   requirements** — a bug in the first draft of this milestone's own API, caught by measuring a
   claim I had written into a docstring. Forty `--strategy random` builds of `qa_toy` share one
   `content_id`; seeds 9 and 35 pass `eval_splits_deduplicated` and thirty-eight seeds fail it.
   `GateReport.scope` is now `eval_id` whenever a split-sensitive requirement ran. See "What
   should be checked later", item 5.

All five profiles reject the deliberately flawed demo release, which is the only correct answer;
a `--drop`-repaired release is the positive control. 1756 tests pass (1706 prior + 50 new).
`ruff check .` clean. No network, no model, no paid API.

What changed

- **New** `src/eval_dataset_forge/contracts/` — `reader.py` (`ReleaseHandle` + the declared read
  set), `requirements.py` (14 requirements, three verdicts, a `cannot` clause each),
  `gate.py` (`GatePolicy`, `GateReport`, `CONTRACT_VERSION`, exit codes), `profiles.py` (the five
  consumers, each with `assumes` and `does_not_check`), `report.py` (rendered through Milestone
  11's `Document`, so Markdown and HTML cannot disagree).
- **New** `tests/test_contracts.py` — 50 tests. The four measurements above are regression tests.
- **Changed** `cli.py` — `gate` subcommand. Opens the release with `verify=False` on purpose:
  `integrity` is itself a requirement, so a corrupt release earns a report, not a traceback.
- **Changed** `datasets/release.py` — `render_reproduce` takes `near_method` and emits an
  `Environment` section; the "Verify without rebuilding" prose now states the asymmetry.
- **Rebuilt** `datasets/releases/qa_toy_v0.1.0/` — `REPRODUCE.md` and the one manifest hash that
  describes it. Nothing else. See "Technical risks".
- **Docs** new `docs/INTEGRATION.md`; `RESULTS.md`, `LIMITATIONS.md`, `DEDUPLICATION.md`,
  `REPRODUCIBILITY.md`, `README.md` updated. **Site** new `site/integration.html`, nav added to all
  17 pages, `milestones.html` / `roadmap.html` / `results.html` / `limitations.html` /
  `deduplication.html` / `index.html` updated (`index.html` still claimed Milestone 11).

What should be checked later

1. **Should `near_method` default to `jaccard`?** This is the milestone's real open question and
   I did not settle it. TF-IDF dominates Jaccard on the fixtures at every threshold (Milestone
   4's sweep) *and* it is the sole reason a duplicate report does not compose, a gate verdict
   does not survive a union, and a release's `eval_id` depends on the build machine's `pip list`.
   Choosing the strictly weaker detector to buy composability and reproducibility is a real
   trade. Arguments for keeping TF-IDF: it finds `bio-001`≈`bio-002`, which Jaccard misses, and a
   missed duplicate leaks across splits while a false one merely pins two examples together.
   Arguments for Jaccard: a benchmark whose held-out set depends on an optional package is not a
   benchmark. **A human should decide.** My inclination is to keep TF-IDF as the `auto` default
   *and* make `near_method` an explicit, required field of a release's build command — but that
   changes the CLI contract, so I did not do it.

2. **Should `AuditResult` carry the difficulty-calibration verdict?** Milestone 11's review note
   asked this first; I did not resolve it, I made the gap executable. `difficulty_is_calibrated`
   returns `UNDECIDABLE` on every release this tool has ever built, because the permutation test
   is computed at build time and survives only as prose in `recommendations`. It is in the
   registry, in **no** profile, and a test asserts no profile can pick it up. Adding the field is
   an additive schema change (a `patch` under `compare()`), and the test
   `test_difficulty_calibration_is_undecidable_from_any_release` will fail loudly the day it
   lands — deliberately.

3. **Are the five profiles the right five requirements each?** They are written from the outside;
   none of the named projects has reviewed the contract written on its behalf. In particular:
   - `code-agent-runtime` omits `near_duplicates_within_budget` on the argument that a
     paraphrased task is a legitimate second instance of a skill. Is that right for a *gating*
     benchmark, where a duplicated task double-counts?
   - `bioagentbench` omits `no_retrieval_contamination` because a lookup-style bio set would
     return `UNDECIDABLE` and block the gate for a reason unrelated to data quality. That is a
     convenience, and it means a bio set that *does* ship passages is under-checked. A
     `--requirement` override exists; nothing forces its use.
   - `agent-control-bench` deliberately does not require unambiguous labels — the ambiguous
     boundary is what a control benchmark measures. I am confident in the reasoning and less
     confident that a real control benchmark wants `no_answer_in_prompt`, which it does require.

4. **`min_eval_blocks = 30` is a convention**, inherited from Milestone 9's `MIN_EFFECTIVE_N`,
   exactly as `MIN_TOP1_STABILITY = 0.95` and `MIN_CALIBRATION_N = 30` are. The gate prints the
   measured block count so a reader can apply their own bar; `--profile llm-regression` does not
   let them without a policy file. On the demo release the answer is not close (2 blocks), so
   nothing here is decided by the threshold — but it will be on a real suite.

5. **A bug I wrote and then measured, and the fix I chose.** The first draft of `applies_to`
   compared `content_id` alone, on the argument that no finding a gate reads depends on the split
   assignment. That is false for four of the fourteen requirements. Building `qa_toy` forty times
   with `--strategy random`: **all forty releases share one `content_id`**, seeds 9 and 35 pass
   `eval_splits_deduplicated`, and the other thirty-eight fail it. The first draft would have
   handed seed 9's pass to seed 0's release.

   The fix is a `split_sensitive` flag on `Requirement`, set on `no_train_test_overlap`,
   `no_retrieval_contamination`, `eval_splits_deduplicated`, and `eval_split_is_rankable`;
   `GateReport.scope` is `eval_id` when any of them ran and `content_id` otherwise. **Two things
   a human should check.** First, is the flag set on exactly the right four? It is a hand-audited
   claim about which checks read `split`, and nothing tests it mechanically — a new
   split-reading requirement that forgot the flag would silently widen its own scope. Second,
   the scope is derived from the requirements that *ran*, not from `split_spec.strategy`, which
   I think is right (a `strategy` field is a claim, not evidence) but which means an adversarial
   release — where all four are seed-invariant in fact — still gets the stricter `eval_id` scope
   and re-gates unnecessarily. That is conservative and cheap; it is still a choice.

6. **The union experiment used `build_release`, so it is a claim about releases.** The
   subadditivity numbers (45 vs 44) come from `duplicate_blocks(find_duplicates(...))` directly.
   Both paths agree here, but a reader should know the two tests exercise different call stacks.

7. **`no_retrieval_contamination` is the one requirement that reads `examples.jsonl`.** Every
   other requirement is a projection of `manifest.json` + `audit.json`. That asymmetry is
   load-bearing (the evidence bundle records findings, never coverage) and it is also a smell: a
   consumer that fetched only the evidence bundle cannot run the full gate. Should `audit.json`
   grow a `coverage` block — how many examples each check actually inspected? That would make the
   whole gate runnable from the bundle and would subsume open question 2.

Technical risks

- **The demo release was rebuilt in place at v0.1.0 for the second time.** `compare()` classifies
  it `patch` → `0.1.1` (`content_id`, `eval_id`, `schema_version` all equal; zero examples added,
  removed, changed or reassigned; reasons `("packaging only: hashes",)`; one artifact hash moved),
  and I overrode the recommendation as Milestone 12 did, for its reasons: never distributed, a
  build artifact of the repo, regenerated by its own `REPRODUCE.md`. **Git history now contains
  three different byte-sets all calling themselves `qa_toy v0.1.0`.** Milestone 12 flagged this
  once; doing it again compounds it. A bump would have rewritten 61 references across README,
  docs, site, tests, and three *historical* milestone notes, which is its own kind of dishonesty.
  If the demo release is ever treated as published, both calls were wrong. **A human should
  decide whether Milestone 15 re-cuts it as `v0.1.1` and freezes it.**
- **`CONTRACT_VERSION` is a string with no machinery behind it.** Nothing compares a report's
  `contract_version` against the running package's, and nothing enforces the stated bump rule.
  The declared-read-set tests catch a *removal* from the artifact; they do not catch a requirement
  quietly changing its verdict for the same evidence, which the rule says is a `major`.
- **`GatePolicy.thresholds` is `dict[str, Any]`.** `resolve_thresholds` rejects unknown *keys* and
  validates nothing about *values*: `min_eval_blocks: "thirty"` raises a `ValueError` deep inside
  `int()` rather than at policy load. A typed threshold model would be cleaner.
- **`_no_sklearn` monkeypatches `sklearn_available` rather than uninstalling scikit-learn.** The
  suite therefore passes with and without the package present, but it tests the *resolution
  branch*, not a genuinely sklearn-free interpreter. A no-sklearn CI job would test the real
  thing; the dependency-free guarantee is currently asserted at one function.
- **`test_the_committed_release_rebuilds_byte_for_byte` compares `examples.jsonl`, `audit.json`,
  `audit.md`, and the three split files — not `manifest.json`, `dataset_card.md`, or
  `REPRODUCE.md`.** The last three embed `created_at`, curator prose, and the invoking command
  respectively. That is defensible, and it means the test does *not* prove `REPRODUCE.md`'s exact
  command reproduces `REPRODUCE.md`.
- **The `integrity` requirement runs `verify_release`, which re-reads and re-hashes every file.**
  Any profile therefore pays a full hash pass. At fixture scale this is microseconds; at 10⁶
  examples it is not, and nothing says so.
- **`ReleaseHandle.examples` is a `cached_property` while `verify()` re-reads from disk.** A
  handle held across a rebuild would report stale examples and fresh hashes. Handles are meant to
  be short-lived; nothing enforces it.

Commands to verify

```bash
python3 -m pytest -q                                        # 1756 tests
python3 -m pytest -q tests/test_contracts.py                # 50 tests
ruff check .

PYTHONPATH=src python3 -m eval_dataset_forge gate --list
PYTHONPATH=src python3 -m eval_dataset_forge gate \
  --release datasets/releases/qa_toy_v0.1.0 --profile llm-regression    # exit 1

# The demo release fails all five profiles, and the reasons are the planted defects.
PYTHONPATH=src python3 -c "
from eval_dataset_forge.contracts import PROFILES, gate_release
for name in sorted(PROFILES):
    r = gate_release('datasets/releases/qa_toy_v0.1.0', name)
    print(f'{name:26s} ok={r.ok}  fail={[x.name for x in r.failed]}')
"

# An optional dependency decides which examples are held out.
python3 -m pytest -q tests/test_contracts.py -k optional_dependency

# …and yet the committed release verifies with or without it.
python3 -m pytest -q tests/test_contracts.py -k environment_independent

# 45 blocks apart, 44 pooled, 56 examples either way — and 46/46 under jaccard.
python3 -m pytest -q tests/test_contracts.py -k additive

# A verdict does not survive a union, and refuses to say it did.
python3 -m pytest -q tests/test_contracts.py -k union

# The rebuilt demo release still verifies and still regenerates its card.
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0                 # 10/10 checks
PYTHONPATH=src python3 -m eval_dataset_forge dataset-card \
  --release datasets/releases/qa_toy_v0.1.0 --check         # exit 0
PYTHONPATH=src python3 -m eval_dataset_forge smoke \
  --out results/reports/smoke_audit.md \
  --fail-on-missed-defect --fail-on-control-violation       # exit 0
```

Human feedback
<!-- Human will fill this later. -->
