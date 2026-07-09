# Human review: Milestone 15 — Final cleanup

TL;DR

The final cleanup gate runs clean — 1852 tests, `ruff`, fixture validation, the smoke audit, the
demo-release rebuild, `verify-release` at 10/10, and the documentation auditor all green, offline.
Running it exposed two defects, both the shape Milestone 14 named (a claim in a document with no way
to fail), one level down in the repo's description of its own code:

1. **The `docs/PLAN.md` §4 architecture diagram declared an `adapters/` package — `mock_llm.py`,
   `api_stub.py` — that was never built.** The pipeline is entirely heuristic and calls no model, so
   no labeler adapter was ever needed. The ten existing documentation checks read what the docs say,
   never what the tree is, so the false lines passed every check since the Milestone 0 scaffold. Added
   `check_package_layout` (the eleventh check), which caught exactly those three lines; corrected the
   diagram and the three docs that echoed it rather than building dead code.
2. **`docs/SAFETY.md` referenced "the hygiene check in Milestone 15", which did not exist.** Built
   `scripts/hygiene_scan.py` (`eval_dataset_forge.hygiene`): an allowlist-free audit of the
   do-not-commit contract against the tracked tree.

Plus a sharper reproduction result: rebuilding the committed `qa_toy` release without scikit-learn
preserves `content_id`, `eval_id`, and every split byte — on this release the two detectors agree
about the split, so only the four artifacts that *name* the detector move. Pinned by a new test.

What changed

- **New:** `src/eval_dataset_forge/hygiene.py`, `scripts/hygiene_scan.py`, `tests/test_hygiene.py`
  (12 tests).
- **`src/eval_dataset_forge/docs_audit.py`:** added `check_package_layout`, `observe_package_layout`,
  `CHECK_NAMES` (single source of the check count), and a `docs_check_count` numeric claim; wired the
  eleventh check into `audit_docs` and two new stats (`checks`, `declared_modules`).
- **`tests/test_docs.py`:** +11 tests (70 → 81) — controls for the layout check (declared-but-missing
  module/package, wrapped continuation line, one-directionality, the disk observer), and the
  parametrised clean-check coverage now iterates `CHECK_NAMES`.
- **`tests/test_contracts.py`:** +1 test (50 → 51) — `qa_toy` no-scikit-learn reproduction pin.
- **Corrected false claims:** `docs/PLAN.md` §3 ("mock adapters are first-class"), §4 (the
  `adapters/` line), §9 (the whole adapters section); `docs/METHODOLOGY.md` ("mock adapters stand in
  for any model call").
- **Updated:** `docs/CLAIMS.md` ("Nine checks" → 11, `package-layout` row, `docs_check_count` ledger
  row, a new subsection); `docs/SAFETY.md` (names the hygiene scan); `docs/RESULTS.md` (M15 section;
  test-count comments 70 → 81, 50 → 51); `docs/INTEGRATION.md` (50 → 51);
  `site/{index,results,limitations,milestones,claims}.html`.

What should be checked later

1. **The hygiene scan is a second concrete deliverable in a milestone whose brief is "run the
   gates".** It is justified twice — the brief lists "hygiene scan," and `docs/SAFETY.md` already
   referenced it — but confirm that building a new module + script + test file was the right call
   versus making `SAFETY.md` point at the existing `.gitignore` + settings deny-list. My reasoning:
   a promised check that does not exist is exactly this milestone's defect class, so building it is
   the consistent move.
2. **`check_package_layout` is one-directional, and that is a real limit, not just a design note.** It
   cannot catch a real module the diagram omits. If you want the diagram to be exhaustive, this is the
   wrong tool and a different (noisier, and arguably dishonest) check would be needed. Confirm the
   asymmetry is the behaviour you want.
3. **The layout parser is regex over a fenced `text` block.** It joins continuation lines (the
   `leakage/` wrap) and reads `·`-separated `.py` tokens. A future reformat of the §4 diagram — say to
   a real tree with `├──` glyphs — would silently parse to nothing and the check would pass vacuously.
   `test_the_audit_actually_looked_at_something` guards the global counter (`declared_modules >= 40`),
   but not this specific block. Worth deciding whether to assert the §4 block yields a non-empty
   declared set specifically.
4. **The hygiene scan reads filenames, not contents.** A real secret pasted into a `.py` or `.md`
   file is not caught. This is deliberate (content detection lives in `validation.pii`, tested on
   fixtures that hold fake secrets), but it means "hygiene scan is green" is not "there are no secrets
   in the repo." Documented in the module and in `docs/LIMITATIONS`-equivalent site copy.
5. **`MAX_TRACKED_BYTES` is 512 KiB, chosen against today's largest tracked file (~50 KiB).** It is a
   heuristic threshold; a legitimately large committed doc later would trip it and need the constant
   raised. There is no allowlist by design, so the escape hatch is editing the cap, which is visible.
6. **`CLAUDE.md` was left unchanged.** It still calls scikit-learn an "optional accelerator" (the
   characterization Milestone 13 falsified) and says "Prefer deterministic mock adapters
   (`MockLLMLabeler`)". The second is imperative policy, not an existence claim, so it is arguably
   fine; the first is the same stale characterization Milestone 14 also left alone. Both were left
   deliberately — it is the operator's instruction file, and rewriting one's own instructions
   unprompted is the wrong default. A human should decide.
7. **The demo release was not rebuilt.** No code path that produces it changed behaviour
   (`near_method` still resolves `auto`), and `verify-release` reports 10/10. The M15 reproduction
   test confirms a no-scikit-learn rebuild preserves identity on `qa_toy`; the "three byte-sets called
   `qa_toy v0.1.0` in git history" problem from Milestones 12–13 is neither worsened nor closed here.
8. **`docs_check_count` binds to `len(CHECK_NAMES)`, and its "produced by" command is
   `scripts/verify_docs.py --json`** (which emits `checks` in its stats). A reader may call that
   circular — the number is a property of the auditor's own source, verified by the auditor. The
   argument for it: the auditor should not be able to misstate its own size in the doc that describes
   it, which is the drift `docs/CLAIMS.md` already had ("Nine checks" over a ten-row table).

Technical risks

- **The docs auditor is a build gate that fails on unrelated work.** Adding a test to `test_docs`,
  `test_contracts`, or any file whose count a live doc cites now fails `verify_docs.py` until the doc
  is updated — as it did for this milestone's own additions (70 → 81, 50 → 51). Intended friction.
- **`scan_repository` shells out to `git ls-files` and `git check-ignore` from inside the test
  suite.** Fast and offline, but a subprocess-and-git dependency inside `tests/test_hygiene.py`
  (`test_the_tracked_tree_honours_the_contract`, `test_no_tracked_file_is_git_ignored`). In a build
  environment without git or outside a work tree those two tests would error; the pure-function
  controls would still run.
- **`observe_package_layout` walks the real `src/eval_dataset_forge` tree on every audit.** Cheap
  today; it grows with the package. `audit_docs` already runs the deduplication ladder twice and five
  audits, so this is not the dominant cost.
- **No release artifact or `eval_id` changed.** `near_method` defaults to `auto`; the correction was
  to prose and to two new checks. The demo release bytes are untouched.

Commands to verify

```bash
python3 -m pytest -q                                        # 1852 tests
ruff check .

python3 scripts/verify_docs.py                              # 11 checks, exit 0
python3 scripts/hygiene_scan.py                             # exit 0
python3 scripts/validate_fixtures.py

# The controls: every documentation and hygiene check fires on a planted defect.
python3 -m pytest -q tests/test_docs.py -k "package_layout or catches or fires"
python3 -m pytest -q tests/test_hygiene.py

# The three results this milestone measured.
python3 -m pytest -q tests/test_contracts.py -k without_sklearn
python3 -m pytest -q tests/test_docs.py -k "check_names or looked_at_something"

# The demo release still rebuilds and verifies.
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0                 # 10/10
```

Human feedback
<!-- Human will fill this later. -->
