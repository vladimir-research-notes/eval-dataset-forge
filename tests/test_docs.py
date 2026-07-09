"""The documentation auditor, and the controls that prove it can fail (Milestone 14).

Two halves, and the second is the one that matters.

The first half runs :func:`~eval_dataset_forge.docs_audit.audit_docs` over this repository and
asserts every check is clean. On its own that assertion is worth nothing: a checker that reports
zero violations is indistinguishable, on that number alone, from a checker that looks at nothing.
It is the mirror image of the flag-everything detector that Milestone 12 puts through the smoke
audit's scorer — there, a detector that asserts everything earns recall 1.000; here, a detector
that asserts nothing earns zero violations. Both numbers are free.

So the second half plants a defect for every check and asserts the check fires. A broken link, an
invented subcommand, a renamed flag, a stale test count, a fixture id no fixture holds, a retired
claim retyped, a ledger row the code does not reproduce, a README that has drifted from the
ledger, a claim citing a command that does not exist, an entry point pointing at nothing.

The use/mention tests are not padding. Both false positives the auditor ever produced were
use/mention errors, and both were found by running it rather than by reasoning about it: a
quotation that wrapped across a line break, and a ``[text](url)`` written inside backticks as an
example of Markdown. Each has a test here, because the next person to touch ``_asserted`` or
``mask_code`` will otherwise reintroduce them.
"""

from __future__ import annotations

import pathlib

import pytest

from eval_dataset_forge.docs_audit import (
    CANDIDATE_RETIRED_CLAIMS,
    CLAIMS,
    LEDGER_PATH,
    RETIRED_CLAIMS,
    Claim,
    Doc,
    RetiredClaim,
    audit_docs,
    check_claim_commands,
    check_cli_commands,
    check_echoes,
    check_entry_point,
    check_example_ids,
    check_ledger,
    check_links,
    check_referenced_paths,
    check_retired_claims,
    check_test_counts,
    extract_commands,
    live_docs,
    mask_code,
    observe_cli_flags,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: A synthetic retired claim, so the control tests do not depend on which real claims are retired.
_TOY_RETIRED = (
    RetiredClaim(
        claim_id="toy",
        pattern=r"the moon is made of cheese",
        retired_by="Milestone 0",
        because="it is not",
        superseded_by="docs/LIMITATIONS.md",
    ),
)


# --------------------------------------------------------------------------- #
# The repository, audited
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def report():
    """One full audit of this repository. Expensive: it collects the suite and re-derives every
    registered claim from the fixtures."""
    return audit_docs(ROOT)


def test_the_repository_documentation_has_no_violations(report) -> None:
    assert report.ok, "\n".join(str(v) for v in report.violations)


@pytest.mark.parametrize(
    "check",
    [
        "links",
        "cli-commands",
        "referenced-paths",
        "test-counts",
        "example-ids",
        "retired-claims",
        "ledger",
        "echoes",
        "claim-commands",
        "entry-point",
    ],
)
def test_each_check_is_clean_on_this_repository(report, check: str) -> None:
    """Localises a failure to one check rather than to one 300-line assertion."""
    grouped = report.by_check()
    assert grouped.get(check, []) == [], "\n".join(str(v) for v in grouped.get(check, []))


def test_the_audit_actually_looked_at_something(report) -> None:
    """A zero-violation report over zero documents is the checker that looks at nothing."""
    assert report.stats["live_docs"] >= 15
    assert report.stats["historical_docs"] >= 20
    assert report.stats["links"] >= 80
    assert report.stats["cli_commands"] >= 100
    assert report.stats["claims"] == len(CLAIMS)


def test_every_registered_claim_has_a_ledger_row() -> None:
    ledger = (ROOT / LEDGER_PATH).read_text()
    for claim in CLAIMS:
        assert f"`{claim.claim_id}`" in ledger, f"{claim.claim_id} is registered but not documented"


def test_every_claim_names_a_command() -> None:
    for claim in CLAIMS:
        assert claim.command.strip(), f"{claim.claim_id} states a number with no producer"


# --------------------------------------------------------------------------- #
# Controls: each check must fire on a planted defect
# --------------------------------------------------------------------------- #


def test_link_check_catches_a_broken_link(tmp_path: pathlib.Path) -> None:
    doc = Doc("README.md", "See [the plan](docs/NOPE.md).")
    violations = check_links([doc], tmp_path)
    assert len(violations) == 1
    assert "docs/NOPE.md" in violations[0].message


def test_link_check_accepts_a_link_that_resolves(tmp_path: pathlib.Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PLAN.md").write_text("x")
    doc = Doc("README.md", "See [the plan](docs/PLAN.md).")
    assert check_links([doc], tmp_path) == []


def test_link_check_ignores_a_link_inside_a_code_span(tmp_path: pathlib.Path) -> None:
    """``docs/DATASET_CARD.md`` demonstrates Markdown escaping with a literal ``[text](url)``.

    The first draft of the link checker chased it to a file named ``url``. A link inside backticks
    is mentioned, not used.
    """
    doc = Doc("docs/DATASET_CARD.md", "Markdown escapes `[` in `[text](url)` uniformly.")
    assert check_links([doc], tmp_path) == []


def test_link_check_ignores_links_inside_fenced_blocks(tmp_path: pathlib.Path) -> None:
    doc = Doc("README.md", "before\n\n```text\n[a](docs/NOPE.md)\n```\n\nafter")
    assert check_links([doc], tmp_path) == []


def test_link_check_skips_absolute_urls(tmp_path: pathlib.Path) -> None:
    doc = Doc("README.md", "[home](https://example.com) [mail](mailto:a@b.c)")
    assert check_links([doc], tmp_path) == []


def test_cli_check_catches_an_unknown_subcommand() -> None:
    doc = Doc("README.md", "```bash\nevalforge frobnicate --dataset x.jsonl\n```")
    violations = check_cli_commands([doc], {"audit": {"--dataset"}})
    assert len(violations) == 1
    assert "unknown subcommand" in violations[0].message


def test_cli_check_catches_an_unknown_flag() -> None:
    doc = Doc("README.md", "```bash\nevalforge audit --dataset x.jsonl --turbo\n```")
    violations = check_cli_commands([doc], {"audit": {"--dataset"}})
    assert len(violations) == 1
    assert "--turbo" in violations[0].message


def test_cli_check_treats_both_spellings_as_the_same_program() -> None:
    """``evalforge audit`` and ``python3 -m eval_dataset_forge audit`` are one entry point."""
    flags = {"audit": {"--dataset"}}
    bare = Doc("a.md", "```bash\nevalforge audit --nope\n```")
    module = Doc("b.md", "```bash\nPYTHONPATH=src python3 -m eval_dataset_forge audit --nope\n```")
    assert len(check_cli_commands([bare], flags)) == 1
    assert len(check_cli_commands([module], flags)) == 1


def test_cli_check_accepts_flag_equals_value_and_line_continuations() -> None:
    doc = Doc(
        "README.md",
        "```bash\nevalforge audit --dataset=x.jsonl \\\n  --out results/\n```",
    )
    assert check_cli_commands([doc], {"audit": {"--dataset", "--out"}}) == []


def test_cli_check_ignores_non_evalforge_commands() -> None:
    doc = Doc("README.md", "```bash\nruff check .\npython3 -m pytest -q\n```")
    assert check_cli_commands([doc], {}) == []


def test_cli_check_ignores_help_invocations() -> None:
    doc = Doc("README.md", "```bash\nevalforge --help\n```")
    assert check_cli_commands([doc], {"audit": set()}) == []


def test_every_documented_flag_exists_on_the_real_parser(report) -> None:
    """The check above, wired to the live parser: renaming a flag in cli.py fails the docs."""
    flags = observe_cli_flags()
    assert "--near-method" in flags["audit"], "Milestone 14 added it; the ledger cites it"
    assert check_cli_commands(live_docs(ROOT), flags) == []


def test_referenced_path_check_catches_a_missing_script(tmp_path: pathlib.Path) -> None:
    doc = Doc("README.md", "```bash\npython3 scripts/nope.py\n```")
    violations = check_referenced_paths([doc], tmp_path)
    assert len(violations) == 1
    assert "scripts/nope.py" in violations[0].message


def test_referenced_path_check_accepts_a_script_that_exists() -> None:
    doc = Doc("README.md", "```bash\npython3 scripts/verify_docs.py\n```")
    assert check_referenced_paths([doc], ROOT) == []


def test_test_count_check_catches_a_stale_count() -> None:
    block = "```bash\npython3 -m pytest -q tests/test_dedup.py  # 5 tests\n```"
    violations = check_test_counts([Doc("docs/RESULTS.md", block)], {"test_dedup.py": 99})
    assert len(violations) == 1
    assert "claims 5 tests, pytest collects 99" in violations[0].message


def test_test_count_check_accepts_a_current_count() -> None:
    block = "```bash\npython3 -m pytest -q tests/test_dedup.py  # 99 tests\n```"
    assert check_test_counts([Doc("docs/RESULTS.md", block)], {"test_dedup.py": 99}) == []


def test_test_count_check_catches_a_test_file_that_does_not_exist() -> None:
    doc = Doc("docs/RESULTS.md", "```bash\npython3 -m pytest -q tests/test_gone.py  # 3 tests\n```")
    violations = check_test_counts([doc], {"test_dedup.py": 99})
    assert len(violations) == 1
    assert "no such test file" in violations[0].message


def test_test_counts_are_not_checked_against_the_historical_record() -> None:
    """``docs/milestones/12-smoke-audit.md`` says ``# 1706 tests`` and is right about its commit.

    A record that is rewritten to match the present is not a record. The count check runs over
    live docs only, and this test pins that the historical note still carries its own number.
    """
    note = (ROOT / "docs" / "milestones" / "12-smoke-audit.md").read_text()
    assert "# 1706 tests" in note
    assert all(doc.path != "docs/milestones/12-smoke-audit.md" for doc in live_docs(ROOT))


def test_example_id_check_catches_an_invented_id() -> None:
    doc = Doc("docs/RESULTS.md", "The pair `qa-999` is interesting.")
    violations = check_example_ids([doc], {"qa-001"})
    assert len(violations) == 1
    assert "qa-999" in violations[0].message


def test_example_id_check_accepts_a_real_id() -> None:
    assert check_example_ids([Doc("d.md", "`qa-001`")], {"qa-001"}) == []


# --------------------------------------------------------------------------- #
# Retired claims: use is not mention
# --------------------------------------------------------------------------- #


def test_retired_claim_check_catches_an_assertion() -> None:
    doc = Doc("README.md", "Everyone knows the moon is made of cheese.")
    violations = check_retired_claims([doc], _TOY_RETIRED)
    assert len(violations) == 1
    assert "toy" in violations[0].message


def test_retired_claim_check_ignores_an_inline_quotation() -> None:
    doc = Doc("docs/LIMITATIONS.md", 'The README used to say "the moon is made of cheese".')
    assert check_retired_claims([doc], _TOY_RETIRED) == []


def test_retired_claim_check_ignores_a_quotation_that_wraps_across_a_line() -> None:
    """The auditor's one false positive, pinned.

    ``docs/RESULTS.md`` opens its quotation on one line and closes it on the next. A line-scoped
    rule reads the continuation as an assertion and flags the document most careful to disown the
    claim. Quotations wrap across lines; they do not wrap across a blank line.
    """
    text = 'What was false was `REPRODUCE.md`, which claimed "the moon\nis made of cheese".'
    assert check_retired_claims([Doc("docs/RESULTS.md", text)], _TOY_RETIRED) == []


def test_retired_claim_check_scopes_quotations_to_the_paragraph() -> None:
    """A quotation in an earlier paragraph must not license an assertion in a later one."""
    doc = Doc(
        "README.md",
        'Someone wrote "the moon is made of cheese".\n\nIn fact the moon is made of cheese.',
    )
    violations = check_retired_claims([doc], _TOY_RETIRED)
    assert len(violations) == 1
    assert violations[0].line == 3


def test_retired_claim_check_ignores_fenced_blocks_and_code_spans() -> None:
    fenced = Doc("README.md", "```text\nthe moon is made of cheese\n```")
    inline = Doc("README.md", "The string `the moon is made of cheese` is retired.")
    assert check_retired_claims([fenced], _TOY_RETIRED) == []
    assert check_retired_claims([inline], _TOY_RETIRED) == []


def test_retired_claim_check_reports_the_right_line() -> None:
    doc = Doc("README.md", "one\n\ntwo\n\nthe moon is made of cheese\n")
    violations = check_retired_claims([doc], _TOY_RETIRED)
    assert [v.line for v in violations] == [5]


def test_the_cited_in_allowlist_is_empty() -> None:
    """An allowlist that stays empty is one nobody can quietly grow.

    Quoting a retired claim to refute it is a mention, which ``_asserted`` already permits, so no
    document needs an exemption.
    """
    for claim in RETIRED_CLAIMS:
        assert claim.cited_in == (), f"{claim.claim_id} needs an exemption; the rule is too coarse"


def test_the_retired_determinism_claim_is_registered() -> None:
    ids = {claim.claim_id for claim in RETIRED_CLAIMS}
    assert "determinism_omits_environment" in ids


def test_every_retired_claim_points_at_a_document_that_exists() -> None:
    """``superseded_by`` is the reader's next step; a dangling one is worse than none."""
    for claim in (*RETIRED_CLAIMS, *CANDIDATE_RETIRED_CLAIMS):
        path = claim.superseded_by.split("§")[0].strip().rstrip(",")
        assert (ROOT / path).exists(), f"{claim.claim_id} points at {path}"


def test_the_lint_does_not_verify_the_fact_it_lints() -> None:
    """The retired claim's *fact* is pinned by a behavioural test, not by a regex.

    ``tests/test_contracts.py`` builds the same corpus under both detectors and compares
    ``eval_id``. If that test disappears, the lint becomes a rule with nothing behind it.
    """
    contracts = (ROOT / "tests" / "test_contracts.py").read_text()
    assert "def test_near_method_resolves_on_sklearn_availability" in contracts
    assert "def test_verification_is_environment_independent" in contracts


# --------------------------------------------------------------------------- #
# The candidate rule: registered, measured, and not enforced
# --------------------------------------------------------------------------- #


def test_the_candidate_rule_is_not_enforced() -> None:
    enforced = {claim.claim_id for claim in RETIRED_CLAIMS}
    candidates = {claim.claim_id for claim in CANDIDATE_RETIRED_CLAIMS}
    assert candidates
    assert enforced.isdisjoint(candidates)


def test_the_candidate_rule_would_flag_only_documents_that_refute_the_claim() -> None:
    """Why ``sklearn_is_an_accelerator`` ships disabled — measured, not asserted.

    It fires on the documents that *correct* the error, because a correction must use the same
    words as the error. One of them is ``docs/CLAIMS.md``, the document that exists to explain why
    the rule is disabled. It never fired on ``docs/PLAN.md`` §3, which is where the claim actually
    lived: the phrase there spanned a line break the pattern could not cross, and a human reading
    the file is what found it.
    """
    docs = live_docs(ROOT)
    violations = check_retired_claims(docs, CANDIDATE_RETIRED_CLAIMS)
    flagged = {v.path for v in violations}

    assert len(flagged) >= 2, "a rule that fires on nothing proves nothing about precision"
    assert "docs/CLAIMS.md" in flagged
    assert "docs/PLAN.md" not in flagged


def test_the_candidate_rule_contributes_nothing_to_the_real_audit(report) -> None:
    for violation in report.violations:
        assert "sklearn_is_an_accelerator" not in violation.message


# --------------------------------------------------------------------------- #
# The numeric ledger
# --------------------------------------------------------------------------- #


def _ledger(rows: str) -> list[Doc]:
    return [Doc(LEDGER_PATH, f"| claim | value |\n|---|---|\n{rows}")]


def test_ledger_check_accepts_a_row_the_code_reproduces() -> None:
    claims = (Claim("n", "count", "evalforge info"),)
    assert check_ledger(_ledger("| `n` | 56 |\n"), {"n": 56}, claims) == []


def test_ledger_check_catches_a_row_the_code_does_not_reproduce() -> None:
    claims = (Claim("n", "count", "evalforge info"),)
    violations = check_ledger(_ledger("| `n` | 55 |\n"), {"n": 56}, claims)
    assert len(violations) == 1
    assert "states 55" in violations[0].message and "produces 56" in violations[0].message


def test_ledger_check_catches_a_missing_row() -> None:
    claims = (Claim("n", "count", "evalforge info"),)
    violations = check_ledger(_ledger(""), {"n": 56}, claims)
    assert len(violations) == 1
    assert "has no ledger row" in violations[0].message


def test_ledger_check_catches_an_orphan_row() -> None:
    """A ledger that can hold an unverified number is a document, not a ledger."""
    violations = check_ledger(_ledger("| `ghost` | 7 |\n"), {}, ())
    assert len(violations) == 1
    assert "unregistered claim row" in violations[0].message


def test_ledger_check_catches_a_missing_ledger() -> None:
    violations = check_ledger([Doc("README.md", "")], {}, ())
    assert len(violations) == 1
    assert "missing" in violations[0].message


def test_ledger_check_honours_tolerance_for_monte_carlo_values() -> None:
    """The permutation p-values are seeded and exact to three places, not to machine epsilon."""
    strict = (Claim("p", "p", "evalforge audit"),)
    loose = (Claim("p", "p", "evalforge audit", tolerance=5e-4),)
    doc = _ledger("| `p` | 0.777 |\n")
    assert len(check_ledger(doc, {"p": 0.7772}, strict)) == 1
    assert check_ledger(doc, {"p": 0.7772}, loose) == []


def test_the_two_p_values_are_the_same_statistic_at_different_resample_counts() -> None:
    """The measurement that made ``Claim.command`` a required field.

    ``evalforge difficulty`` defaults to ``--resamples 10000`` and ``evalforge audit`` to
    ``--resamples 2000``. Same fixture, same seed, same statistic, two published p-values, both
    correct. A number is not a claim until it names the command that produced it.
    """
    from eval_dataset_forge.datasets.loader import load_examples
    from eval_dataset_forge.difficulty.calibration import calibrate

    examples = load_examples(ROOT / "datasets" / "fixtures" / "qa_toy" / "examples.jsonl")
    low = calibrate(examples, n_resamples=2_000, seed=0)
    high = calibrate(examples, n_resamples=10_000, seed=0)

    assert low.rho == pytest.approx(high.rho), "the statistic is identical; only its p-value moves"
    assert round(low.p_value, 3) == 0.777
    assert round(high.p_value, 3) == 0.789
    assert low.p_value != high.p_value


# --------------------------------------------------------------------------- #
# Echoes
# --------------------------------------------------------------------------- #


def test_echo_check_catches_a_stale_readme() -> None:
    """The ledger can be right while the README is stale. This is the check that notices."""
    claims = (Claim("n", "count", "evalforge info", echoes=(("README.md", r"(\d+) examples"),)),)
    docs = [Doc("README.md", "a corpus of 55 examples")]
    violations = check_echoes(docs, {"n": 56}, claims)
    assert len(violations) == 1
    assert "states 55" in violations[0].message


def test_echo_check_accepts_a_current_readme() -> None:
    claims = (Claim("n", "count", "evalforge info", echoes=(("README.md", r"(\d+) examples"),)),)
    assert check_echoes([Doc("README.md", "56 examples")], {"n": 56}, claims) == []


def test_echo_check_catches_a_reworded_claim() -> None:
    """Deleting the sentence must not silently delete the check on it."""
    claims = (Claim("n", "count", "evalforge info", echoes=(("README.md", r"(\d+) examples"),)),)
    violations = check_echoes([Doc("README.md", "lots of data")], {"n": 56}, claims)
    assert len(violations) == 1
    assert "no text matches" in violations[0].message


def test_echo_check_catches_a_missing_echo_site() -> None:
    claims = (Claim("n", "count", "evalforge info", echoes=(("GONE.md", r"(\d+)"),)),)
    violations = check_echoes([], {"n": 56}, claims)
    assert len(violations) == 1
    assert "missing" in violations[0].message


# --------------------------------------------------------------------------- #
# Claim commands and the entry point
# --------------------------------------------------------------------------- #


def test_claim_command_check_catches_a_flag_that_does_not_exist() -> None:
    """It caught the first draft of ``pooled_blocks_jaccard``, which cited a flag nobody had built.

    That is how Milestone 14 discovered ``docs/RESULTS.md`` had been publishing Jaccard figures no
    command could produce.
    """
    claims = (Claim("x", "x", "evalforge audit --dataset a.jsonl --near-method jaccard"),)
    violations = check_claim_commands(ROOT, {"audit": {"--dataset"}}, claims)
    assert len(violations) == 1
    assert "--near-method" in violations[0].message


def test_claim_command_check_catches_an_unknown_subcommand() -> None:
    claims = (Claim("x", "x", "evalforge frobnicate"),)
    violations = check_claim_commands(ROOT, {"audit": set()}, claims)
    assert len(violations) == 1
    assert "unknown subcommand" in violations[0].message


def test_claim_command_check_catches_a_missing_script() -> None:
    claims = (Claim("x", "x", "python3 scripts/nope.py"),)
    violations = check_claim_commands(ROOT, {}, claims)
    assert len(violations) == 1
    assert "missing path" in violations[0].message


def test_claim_command_check_accepts_the_real_registry() -> None:
    assert check_claim_commands(ROOT, observe_cli_flags(), CLAIMS) == []


def test_entry_point_resolves_on_this_repository() -> None:
    assert check_entry_point(ROOT) == []


def test_entry_point_check_catches_a_missing_console_script(tmp_path: pathlib.Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    violations = check_entry_point(tmp_path)
    assert len(violations) == 1
    assert "no `evalforge` console script" in violations[0].message


def test_entry_point_check_catches_a_target_that_is_not_callable(tmp_path: pathlib.Path) -> None:
    """The failure mode: a rename in ``cli.py`` that orphans ``pyproject.toml``."""
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\nevalforge = "eval_dataset_forge.cli:no_such_function"\n'
    )
    violations = check_entry_point(tmp_path)
    assert len(violations) == 1
    assert "not callable" in violations[0].message


# --------------------------------------------------------------------------- #
# Extraction primitives
# --------------------------------------------------------------------------- #


def test_mask_code_preserves_offsets_and_newlines() -> None:
    text = "a `b` c\n```\nfenced\n```\nd"
    masked = mask_code(text)
    assert len(masked) == len(text)
    assert masked.count("\n") == text.count("\n")
    assert "b" not in masked and "fenced" not in masked
    assert masked.startswith("a ") and masked.endswith("d")


def test_extract_commands_joins_line_continuations() -> None:
    doc = Doc("d.md", "```bash\nevalforge audit \\\n  --dataset x.jsonl\n```")
    commands = extract_commands(doc)
    assert len(commands) == 1
    assert commands[0].tokens == ("evalforge", "audit", "--dataset", "x.jsonl")


def test_extract_commands_strips_comments_and_env_prefixes() -> None:
    doc = Doc("d.md", "```bash\n# a comment\nPYTHONPATH=src evalforge info  # trailing\n```")
    commands = extract_commands(doc)
    assert len(commands) == 1
    assert commands[0].tokens == ("evalforge", "info")


def test_extract_commands_reports_the_line_within_the_file() -> None:
    doc = Doc("d.md", "intro\n\n```bash\nevalforge info\n```")
    assert extract_commands(doc)[0].line == 4


def test_extract_commands_ignores_prose_and_non_shell_fences() -> None:
    doc = Doc("d.md", "run `evalforge info`\n\n```python\nevalforge info\n```")
    assert extract_commands(doc) == []
