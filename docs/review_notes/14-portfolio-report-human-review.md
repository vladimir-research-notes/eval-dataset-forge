# Human review: Milestone 14 — Final portfolio report

TL;DR

The README and docs were brought to portfolio quality, and writing them exposed why they needed
it. Milestone 13 measured that the build environment is an input to a release, corrected the
sentence that said otherwise in `REPRODUCE.md` — which is generated and pinned by a test — and
left the identical false sentence in `README.md`, `docs/METHODOLOGY.md`, and `docs/PLAN.md`, three
files no test had ever read. A generated artifact is enforced; prose is not.

So this milestone added `scripts/verify_docs.py`: ten checks over the documentation, run in CI-able
form, exit 1 on any unsupported claim. Its first run found the retired sentence in exactly those
three files. Two further results came out of building it. **A number is not a claim until it names
the command that produced it** — the `qa_toy` calibration p-value is 0.789 under `evalforge
difficulty` and 0.777 under `evalforge audit`, same fixture, same seed, same statistic, different
`--resamples` default, both correct. And **two published figures had no command that could produce
them**: the Jaccard block counts required uninstalling scikit-learn, so `evalforge audit
--near-method {auto,tfidf,jaccard}` was added.

What changed

- **New:** `src/eval_dataset_forge/docs_audit.py` (ten checks, 11 registered numeric claims, the
  retired-claim ledger), `scripts/verify_docs.py`, `tests/test_docs.py` (70 tests),
  `docs/CLAIMS.md`, `site/claims.html`.
- **Code change beyond docs:** `run_audit(near_method=...)` threaded to `find_duplicates` and
  `find_leakage`; `evalforge audit --near-method {auto,tfidf,jaccard}` added. `auto` remains the
  default and is asserted to change nothing. Two tests added to `tests/test_audit.py` (47 → 49).
- **Corrected false claims:** `README.md`, `docs/METHODOLOGY.md` §Determinism, `docs/PLAN.md` §3 and
  §4, `pyproject.toml`'s `[project.optional-dependencies]` comment.
- **Rewritten:** `README.md` (problem, architecture, worked example audit with real output, what it
  finds, dataset card, reproduction commands, limitations, research path).
- **Updated:** `docs/RESULTS.md` (M14 section; `# 47 tests` → `# 49 tests`), `docs/LIMITATIONS.md`
  (new section "Prose has no way to fail, so it rots"), `docs/REPRODUCIBILITY.md`,
  `site/{index,results,limitations,roadmap,milestones}.html`, nav on all 18 site pages.

What should be checked later

1. **The `--near-method` flag is a code change inside a documentation milestone.** It is additive,
   defaults to `auto`, and is covered by two tests, but it was not in the milestone's brief. The
   justification is that the milestone's own brief includes "reproduction commands", and two
   published tables had none. Confirm that trade was the right call, or split it out.
2. **`evalforge release` still resolves `auto`.** `--near-method` pins an *audit*, not a *release*.
   Pinning a release's detector means deciding whether the method enters `eval_id`'s inputs — a
   Milestone 10 identity question. Until then `REPRODUCE.md`'s `Environment` section still describes
   a constraint the CLI cannot enforce.
3. **The retired-claim ledger holds exactly one enforced rule.** That is honest (the second candidate
   measured precision 0/2) but thin. Is one rule worth the mechanism? The argument for yes is that
   the mechanism is what makes the *next* retired claim cheap; the argument for no is that a
   one-entry registry is a lint with a ceremony around it.
4. **`_asserted` treats a paragraph with unbalanced double quotes as inverted after the imbalance.**
   Documented, not defended. A stray quote in a long paragraph would suppress a real violation after
   it. Worth deciding whether to warn on unbalanced quotes.
5. **The claim `echoes` patterns are regexes over rendered sentences.** Rewording an echo site fails
   the check rather than silently passing, which is the right default — but the fix is sometimes to
   update the pattern, and a reviewer must confirm the sentence still says what the claim means. This
   is the one check where a green result depends on a human having read the regex.
6. **`site/*.html` is unaudited.** It restates every claim in the docs, including the numbers, and
   nothing verifies the restatement. The HTML I wrote this milestone is hand-checked only.
7. **`CLAUDE.md` still says "`pandas` and `scikit-learn` are optional accelerators with pure-Python
   fallbacks."** That is the same characterization Milestone 13 falsified. It was left alone
   deliberately: it is the operator's instruction file, not project documentation, and rewriting my
   own instructions unprompted seemed the wrong call. A human should decide.
8. **`pooled_blocks_jaccard` and `summed_fixture_blocks_tfidf` name `scripts/verify_docs.py` as their
   producer**, because no CLI command computes a corpus-level block count under a chosen detector
   (`audit` takes one dataset; `smoke` pools but exposes no `--near-method`). The ledger says so
   rather than naming a command that does not exist, but a reader may reasonably call that circular.

Technical risks

- **The docs auditor is now a build gate that can fail on unrelated work.** Adding a test to any file
  whose count is cited in `docs/RESULTS.md` fails `verify_docs.py` until the doc is updated. That is
  the intent, and it will be friction. `python3 -m pytest -q --collect-only` is the cost (~0.6 s).
- **`observe_numbers()` runs deduplication over the pooled corpus twice (tfidf and jaccard) plus five
  per-fixture audits and two 2000/10000-resample permutation tests** on every `verify_docs.py` run and
  once per `tests/test_docs.py` session. Measured at a few seconds; it will grow with the fixtures.
- **`check_test_counts` shells out to `pytest --collect-only` from inside pytest.** It works and is
  fast, but it is a subprocess dependency inside the test suite.
- **The retired-claim regex is a phrasing matcher.** It cannot catch a paraphrase, and a future
  contributor may read a green check as "the docs contain no false claims". `docs/CLAIMS.md` says
  otherwise in as many words; whether that is enough is a judgement call.
- **No release artifact changed.** `datasets/releases/qa_toy_v0.1.0` was not rebuilt: no code path
  that produces it changed behaviour (`near_method` defaults to `auto`), and `verify-release` still
  reports 10/10. Unlike Milestones 12 and 13, this milestone does not compound the "three byte-sets
  called `qa_toy v0.1.0` in git history" problem — but that earlier problem is still open.

Commands to verify

```bash
python3 -m pytest -q                                        # 1828 tests
python3 -m pytest -q tests/test_docs.py                     # 70 tests
ruff check .

python3 scripts/verify_docs.py                              # exit 0
python3 scripts/validate_fixtures.py

# The controls: every check must fire on a planted defect.
python3 -m pytest -q tests/test_docs.py -k "catches"

# The three results this milestone measured.
python3 -m pytest -q tests/test_docs.py -k "p_values or candidate_rule or claim_command"

# The new flag, and that it changes nothing by default.
python3 -m pytest -q tests/test_audit.py -k near_method
PYTHONPATH=src python3 -m eval_dataset_forge audit \
  --dataset datasets/fixtures/bio_toy/examples.jsonl \
  --out results/audits/bio_toy/ --near-method jaccard

# The demo release is untouched and still verifies.
PYTHONPATH=src python3 -m eval_dataset_forge verify-release \
  --release datasets/releases/qa_toy_v0.1.0                 # 10/10
```

Human feedback
<!-- Human will fill this later. -->
