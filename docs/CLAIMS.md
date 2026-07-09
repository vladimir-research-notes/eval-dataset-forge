# Claims — what the documentation asserts, and what checks it

Every earlier milestone audits a dataset. This one audits the prose, because Milestone
13 supplied the reason by accident.

Milestone 13 measured that a release is **not** reproducible from its inputs and seed
alone: the build environment is an input, because `scikit-learn`'s presence resolves
near-duplicate detection to TF-IDF rather than Jaccard, the two disagree about which
examples are duplicates, duplicates become blocks, the adversarial split will not cut
through a block, and `eval_id` folds the split in. It then corrected the sentence that
said otherwise — in exactly one file.

`REPRODUCE.md` is *generated*, by `render_reproduce()`, and pinned by a test. It was
fixed. The same false sentence sat untouched in `README.md`, `docs/METHODOLOGY.md`, and
`docs/PLAN.md`: three files no test had ever read.

> A generated artifact is enforced. Prose is not. A claim written in Markdown has no way
> to fail, so it does not fail, so it rots.

The repository's own quality bar says every major claim must be backed by a test, a
reproducible command, a result table, a documented limitation, or an explicit non-goal.
Until Milestone 14 nothing checked that the backing existed.

## The auditor

```bash
python3 scripts/verify_docs.py          # exit 1 if any documented claim is unsupported
python3 scripts/verify_docs.py --json
python3 -m pytest -q tests/test_docs.py
```

Nine checks. Each takes the world it needs as an argument — a parser's option strings, a
mapping of collected test counts, a set of example ids, a mapping of computed values —
rather than reaching for the repository. That is not a stylistic preference: it is the
only way `tests/test_docs.py` can plant a defect and prove the check catches it.

| check | reads | asserts |
|---|---|---|
| `links` | every doc | each relative Markdown link resolves to a file |
| `cli-commands` | every doc | each `evalforge` command in a fenced block names a real subcommand and real options, checked against the live `argparse` parser |
| `referenced-paths` | every doc | each `scripts/*.py` and `tests/*.py` named in a command exists |
| `test-counts` | live docs | each `# N tests` comment is the number `pytest` collects |
| `example-ids` | every doc | each cited `qa-006`-style id exists in a committed fixture |
| `retired-claims` | live docs | no document *asserts* a sentence a milestone measured false |
| `ledger` | live docs | every registered claim has a row here, and the code reproduces its value |
| `echoes` | live docs | documents that repeat a claim's value repeat the current one |
| `claim-commands` | the registry | each claim's "produced by" command exists |
| `entry-point` | `pyproject.toml` | the `evalforge` console script points at something callable |

### Live documents and historical records are different things

`README.md` and `docs/*.md` state what is true now. `docs/milestones/*.md` and
`docs/review_notes/*.md` state what was true at a commit — `docs/milestones/12-smoke-audit.md`
records `# 1706 tests` and is right about the commit it describes. Checking counts against
history would force rewriting the record to match the present, which is the one thing a
record must never do.

So counts and retired claims are checked against live docs only. Links, commands, and
referenced paths must work wherever they appear, and are checked everywhere.

### Use is not mention

`docs/LIMITATIONS.md` contains the retired sentence on purpose, in order to say it is
false. A lint that cannot tell an assertion from a quotation would either flag the
correction or need an allowlist naming every file permitted to discuss the topic.

The rule is the one English uses: a retired claim inside double quotes is **mentioned**;
anywhere else it is **asserted**. Two refinements were forced by real false positives,
each found by running the checker rather than by reasoning about it:

- The unit is the **paragraph**, not the line. `docs/RESULTS.md` opens its quotation on
  one line and closes it on the next, and the first draft — which counted quotes per line —
  flagged the continuation as an assertion. It produced its only false positive on the
  document most careful to disown the claim.
- Code spans are masked first. `docs/DATASET_CARD.md` contains a literal `[text](url)`
  inside backticks, as an example of the Markdown the card generator must escape. The link
  checker chased it to a file named `url`. That is the same use/mention error one level
  down, for links rather than sentences.

`RetiredClaim.cited_in` — the escape hatch that names files allowed to assert a retired
claim anyway — is **empty**, and an allowlist that stays empty is one nobody can quietly
grow.

## The retired-claim ledger

| claim | retired by | because | superseded by |
|---|---|---|---|
| `determinism_omits_environment` | Milestone 13 | the build environment is an input; scikit-learn's presence changes the blocks, the adversarial split, and `eval_id` | `docs/REPRODUCIBILITY.md` § The environment is an input |

On its first run this rule found the sentence asserted in `README.md`, `docs/METHODOLOGY.md`,
and `docs/PLAN.md` — and in no other live document. All three are fixed.

**The lint does not verify the fact.** `tests/test_contracts.py` does, by building the same
corpus under both detectors and comparing `eval_id`. The lint only stops the retired phrasing
from being typed again into a file no test reads.

### One rule, registered and not enforced

`sklearn_is_an_accelerator` is a true proposition to retire: `docs/PLAN.md` §3 called
scikit-learn an *optional accelerator with a pure-Python fallback*, and Milestone 13 showed
it is a different detector. The rule is in `CANDIDATE_RETIRED_CLAIMS`, in no enforced check,
and a test asserts nothing picks it up — following Milestone 13's precedent, which registered
`difficulty_is_calibrated` and put it in no profile.

The reason is measured, not asserted. Run against the live documents **as Milestone 14 found
them**, before the fix:

| | count | which |
|---|---|---|
| documents flagged | 2 | `docs/LIMITATIONS.md`, `docs/RESULTS.md` |
| flagged documents that **assert** the claim | 0 | both quote it in order to refute it |
| documents that **do** assert it | 1 | `docs/PLAN.md` §3 |
| of those, flagged | 0 | the phrase spans a line break the pattern cannot cross |

Precision 0/2, recall 0/1. The rule flags exactly the two documents that disown the claim and
misses the one that makes it, because a correction must use the same words as the error.
`docs/PLAN.md` §3 and the `pyproject.toml` comment were fixed by **reading them**, which is
what actually found them — not by the lint.

Run against the documents as they now stand, it flags three, and the third is **this one**: the
document that exists to explain why the rule is disabled is flagged by the rule. A check with
that property does not fail closed; it trains a reader to ignore it.

This is Milestone 12's result restated for a different detector. Recall bounds what a checker
misses and says nothing about what it invents, and a rule that cannot separate a claim from its
correction has no precision to spend.

## The numeric ledger

Every number below is recomputed from the code and the committed artifacts by
`observe_numbers()`, which reads no document. `scripts/verify_docs.py` compares each row
against its computed value and fails on disagreement. The `echoes` check additionally
verifies that documents repeating a value repeat the current one — a ledger can be right
while the README is stale.

| claim | value | produced by |
|---|---|---|
| `fixture_count` | 5 | `python3 scripts/validate_fixtures.py` |
| `pooled_examples` | 56 | `evalforge smoke` |
| `pooled_blocks_tfidf` | 44 | `evalforge smoke` |
| `pooled_blocks_jaccard` | 46 | `python3 scripts/verify_docs.py --json` |
| `summed_fixture_blocks_tfidf` | 45 | `python3 scripts/verify_docs.py --json` |
| `subcommand_count` | 13 | `evalforge --help` |
| `verify_release_checks` | 10 | `evalforge verify-release` |
| `demo_block_count` | 11 | `evalforge verify-release` |
| `demo_test_blocks` | 2 | `evalforge dataset-card` |
| `qa_toy_calibration_p_b2000` | 0.777 | `evalforge audit` |
| `qa_toy_calibration_p_b10000` | 0.789 | `evalforge difficulty` |

### A number is not a claim until it names the command that produced it

The last two rows are the same statistic, on the same fixture, at the same seed, and they
disagree.

`docs/RESULTS.md` reports the `qa_toy` difficulty calibration permutation p-value as
**0.789**. `evalforge audit` on that fixture prints **0.777**. Neither is wrong:
`evalforge difficulty` defaults to `--resamples 10000` and `evalforge audit` defaults to
`--resamples 2000`, and the gap is the permutation test's Monte-Carlo error, which shrinks
like `1/√B`.

Nothing in either document was mistaken, and a reader comparing them has no way to discover
why they differ. **A number in a document is not a claim until it names the command that
produced it**, because without one there is nothing to disagree with. Hence the third column,
and hence `check_claim_commands`, which verifies that column names a command that exists.

That check earned its keep immediately. The first draft of `pooled_blocks_jaccard` cited
`evalforge audit --near-method jaccard` — a flag that did not exist. `docs/RESULTS.md` and
`docs/DEDUPLICATION.md` have published Jaccard recall and block counts since Milestone 4, and
**no command could produce them**: `auto` resolution meant the only way to see the Jaccard
numbers was to uninstall scikit-learn. Milestone 14 added `evalforge audit --near-method
{auto,tfidf,jaccard}`, so a published number can now be reproduced without reproducing the
environment that produced it.

Two corpus-level claims still have no CLI producer: `audit` takes one dataset, and `smoke`
pools the five fixtures but exposes no `--near-method`. Their `produced by` column says so
rather than naming a command that does not exist.

## What this does not check

- **It does not check that the code is correct.** The ledger proves the documentation agrees
  with the code. That the code is right is what the rest of the test suite is for. A number can
  be wrong in both places and pass every check here.
- **Retired claims are matched as phrasings, not propositions.** A paraphrase evades the
  pattern. Recall over "false sentences in this repository" is not measurable, because no
  ground-truth list of false sentences exists — unlike the fixtures, whose defects are planted
  and enumerated in `defects.yaml`.
- **Only fenced `bash`/`sh`/`shell`/`console` blocks are parsed for commands.** An
  `evalforge` invocation written inline in prose is not checked. Inline commands inside the
  tool's own recommendation strings are emitted by code, so linting them would test the code
  against itself.
- **`evalforge` is never executed.** The documentation's primary spelling is `evalforge <sub>`,
  which exists on `PATH` only after `pip install -e .`; nothing in the offline suite installs it.
  `check_entry_point` verifies the declared target imports and is callable, which catches a
  rename in `cli.py` that orphans `pyproject.toml`. It does not prove the binary runs.
- **Prose that states no number and quotes no retired claim is unchecked.** Most of the
  documentation is of that kind. This is a lint over the checkable surface, not a proof of
  honesty.
- **`echoes` is a regex against a rendered sentence.** Rewording an echo site to something the
  pattern does not match is reported as a violation, not silently ignored — but the fix is
  sometimes to update the pattern, and a reviewer must confirm that the sentence still says what
  the claim means.

A checker that reports zero violations is indistinguishable, on that number alone, from a
checker that looks at nothing. `tests/test_docs.py` therefore plants a defect for every check
and asserts the check fires — the mirror image of the flag-everything control in
[`RESULTS.md`](RESULTS.md#smoke-audit--scoring-the-scorer-milestone-12).

## Reproduce

```bash
python3 scripts/verify_docs.py
python3 -m pytest -q tests/test_docs.py
```
