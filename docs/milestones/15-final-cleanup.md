# Milestone 15 — Final cleanup

The last milestone runs the whole gate and confirms a clean, reproducible repository — and, running
it, finds the same defect Milestone 14 named, twice, one level down from prose: **a document that
describes code that was never written.** The full suite passes (1852 tests), `ruff` is clean, the
fixtures validate, the smoke audit runs offline, the committed demo release rebuilds, and the
documentation auditor is green. What the milestone adds is the two checks that were missing for the
gate to be honest.

## The gate, run clean

| gate | result |
|---|---|
| `python3 -m pytest -q` | 1852 passed |
| `ruff check .` | clean |
| `python3 scripts/validate_fixtures.py` | 5 fixtures valid |
| `PYTHONPATH=src … smoke` | 0 missed, 0 control violations, 56 examples / 44 blocks |
| `… verify-release --release datasets/releases/qa_toy_v0.1.0` | 10/10 |
| `… dataset-card --release … --check` | card regenerates |
| `python3 scripts/verify_docs.py` | 11 checks, 0 violations |
| `python3 scripts/hygiene_scan.py` | 197 tracked files, 0 violations |

## The defect that ran the milestone: a diagram naming code that does not exist

The documentation auditor from Milestone 14 has ten checks. Every one reads what the docs *say* —
links, commands, cited counts, cited ids, retired phrasings, numbers bound to a command. None reads
what the tree *is*. So a package could be drawn in the `docs/PLAN.md` §4 architecture diagram, and
never built, and the auditor would report zero violations.

That is not hypothetical. §4 declared an `adapters/` package with `mock_llm.py` and `api_stub.py`,
and §9 described a `MockLLMLabeler` "deterministic, offline, used by tests" and an `ApiLabelerStub`.
Git history shows `src/eval_dataset_forge/adapters/` never existed. The label audit (Milestone 6)
turned out to be deterministic heuristics over the committed annotations, and no stage calls a model,
so there was never a model call for a mock to stand in for. The lines sat in the diagram from the
Milestone 0 scaffold, describing a subsystem the implementation never needed.

`check_package_layout` is the eleventh check. It parses the §4 layout block and verifies every module
it names against the tree. Its first run:

```text
[package-layout]
  docs/PLAN.md:142: the layout declares package src/eval_dataset_forge/adapters/, which does not exist
  docs/PLAN.md:142: the layout declares module src/eval_dataset_forge/adapters/mock_llm.py, which does not exist
  docs/PLAN.md:142: the layout declares module src/eval_dataset_forge/adapters/api_stub.py, which does not exist
```

Three violations, and nothing else — the diagram omits `contracts/`, `smoke.py`, and `scoring.py`,
all real, and the check said nothing about them. It is **one-directional on purpose**: it flags a
module the diagram declares and the tree lacks; it is silent about a module the tree holds and the
diagram omits. A layout diagram is a curated sketch, so demanding that it enumerate every module
would make the check assert something false in order to check a true thing. It proves the diagram
names no vaporware; it does not prove the diagram is complete, and it does not measure recall over
"structural claims the repository makes" for the same reason `retired-claims` cannot measure recall
over false sentences.

**The fix was a correction, not a rebuild.** Building an `adapters/` package to satisfy the diagram
would be dead code — nothing would import it. So `docs/PLAN.md` §4 dropped the line (and gained a
sentence saying the diagram is not exhaustive), §3 and §9 stopped asserting a labeler that was never
built, and `docs/METHODOLOGY.md` stopped claiming mock adapters "stand in for any model call in
tests" when there are no model calls. The requirements any *future* paid-API adapter must satisfy —
disabled by default, dry-run, cache, cost estimate, max-cost guard — are kept as a forward contract,
because those are policy and remain true.

## The second defect: a hygiene scan the safety docs promised and did not ship

`docs/SAFETY.md` said "See `.gitignore` and the hygiene check in Milestone 15." No such check
existed. The milestone's own deliverable list names a "hygiene scan," so this is both a promised
artifact and a required one.

`scripts/hygiene_scan.py` (`eval_dataset_forge.hygiene`) audits the do-not-commit contract against
the tree git actually tracks:

- **`ignored-but-tracked`** — a file `.gitignore` would ignore, force-added past the rule.
- **`forbidden-path`** — a tracked `*.pem`, `*.key`, `id_rsa`, `.env`, or model-weight file.
- **`oversized`** — a tracked file above 512 KiB (the largest today is a ~50 KiB test module).

It is **allowlist-free**, the deliberate constraint: an exemption list grows quietly until it exempts
the violation you cared about. And it checks *which files exist*, not their contents. Content
secret-detection is the [validation layer](../VALIDATION.md)'s job, and that layer is tested against
fixtures that hold fake AWS keys and private-key headers on purpose — a repo-wide secret grep would
flag those very fixtures, and silencing it would require the allowlist this refuses to keep. A real
secret pasted into a source file is therefore outside this scan's reach, and that is written down
rather than papered over.

## A sharper reproduction result than REPRODUCE.md claims

Milestone 13 measured that on `bio_toy` scikit-learn's presence changes `eval_id`. On the committed
`qa_toy` release the two detectors *agree* about the split — the near cluster `qa-003` + `qa-004` +
`qa-012` is one block under both `tfidf` and `jaccard`. So rebuilding the demo without scikit-learn:

| rebuilt qa_toy v0.1.0 without scikit-learn | outcome |
|---|---|
| `content_id`, `eval_id`, `source_content_id` | unchanged |
| `examples.jsonl`, all three split files | byte-identical |
| `audit.{json,md}`, `dataset_card.md`, `REPRODUCE.md` | differ — they name the detector |
| `manifest.json` hashes of those four | differ accordingly |
| `verify-release` | still 10/10 |

The detector's disagreement reaches only the artifacts that record which detector ran — never the
data or either identity. `REPRODUCE.md` warns a no-scikit-learn rebuild "can produce different splits
and a different `eval_id`"; that "can" is the general case, and `test_contracts.py` now pins that on
this release it does not. Five tracked files would show as modified in `git status`, and the release
would still verify — the gap between reproduction (environment-dependent) and verification (not).

## What landed

| file | role |
|---|---|
| `src/eval_dataset_forge/docs_audit.py` | `check_package_layout`, `observe_package_layout`, `CHECK_NAMES`, the `docs_check_count` claim |
| `src/eval_dataset_forge/hygiene.py` | the do-not-commit contract audit; three allowlist-free checks |
| `scripts/hygiene_scan.py` | the runner; exit 1 on any violation |
| `tests/test_docs.py` | +11 tests (81); controls for the layout check, `CHECK_NAMES` drives the parametrised coverage |
| `tests/test_hygiene.py` | 12 tests; a planted violation per check |
| `tests/test_contracts.py` | +1 test (51); the qa_toy no-scikit-learn reproduction pin |
| `docs/PLAN.md` §3/§4/§9, `docs/METHODOLOGY.md`, `docs/SAFETY.md` | false structural claims corrected; the hygiene scan named |
| `docs/CLAIMS.md` | "Nine checks" → 11, `package-layout` row, `docs_check_count` ledger row |

## What no check here can supply

- **The layout check is one-directional.** It cannot notice a real module the diagram fails to draw;
  a diagram is a sketch, and completeness is not measurable.
- **The hygiene scan reads filenames, not file contents.** A secret pasted into a `.py` file is not
  caught here.
- **`CLAUDE.md` was left unchanged.** It still calls scikit-learn an "optional accelerator" and
  names `MockLLMLabeler` as a preference. It is the operator's instruction file, not project
  documentation, and rewriting one's own instructions unprompted is the wrong default — the same
  call Milestone 14 made. Flagged for a human in the review note.

## Commands

```bash
python3 -m pytest -q                              # 1852 tests
ruff check .

python3 scripts/verify_docs.py                    # 11 checks, exit 0
python3 scripts/hygiene_scan.py                   # exit 0
python3 scripts/validate_fixtures.py

# The controls: every documentation and hygiene check fires on a planted defect.
python3 -m pytest -q tests/test_docs.py -k "package_layout or catches"
python3 -m pytest -q tests/test_hygiene.py

# The reproduction pin: identity survives a no-scikit-learn rebuild of the demo.
python3 -m pytest -q tests/test_contracts.py -k without_sklearn
```

## Next

None. This is the final milestone. The repository is complete against the §11 quality gates: tests
pass, fixtures validate, the smoke audit runs offline, duplicates / leakage / label conflicts are
detected, a release and a dataset card are built, the ranking-stability demo runs, the docs explain
methodology and limitations and reproduction, no paid API is needed anywhere, and nothing is pushed.
