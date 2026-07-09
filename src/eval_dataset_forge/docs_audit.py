"""Auditing this repository's own documentation (Milestone 14).

Every earlier milestone audits a *dataset*. This one audits the *prose*, for a reason that
Milestone 13 supplied by accident.

Milestone 13 measured that a release is not reproducible from its inputs and seed alone — the
build environment is an input, because ``scikit-learn``'s presence flips near-duplicate detection
from ``tfidf`` to ``jaccard`` and moves ``eval_id``. It then corrected the sentence that said
otherwise. It corrected it in exactly one place: ``REPRODUCE.md``, which is *generated* by
:func:`~eval_dataset_forge.datasets.release.render_reproduce` and pinned by a test. The same
false sentence survived, untouched, in ``README.md``, ``docs/METHODOLOGY.md``, and
``docs/PLAN.md`` — three files no test had ever read.

That is the asymmetry this module exists to remove. **A generated artifact is enforced; prose is
not.** A claim written in Markdown has no way to fail, so it does not fail, so it rots. The repo's
own quality bar says every major claim must be backed by a test, a reproducible command, a result
table, a documented limitation, or an explicit non-goal — and until now, nothing checked that the
backing existed.

The auditor runs six checks. Each takes the world it needs as an argument (a parser's flags, a
collection of test counts, a set of example ids, a mapping of computed values) rather than
reaching for the repository, so every check can be run against synthetic documents. That is not a
stylistic preference: it is the only way ``tests/test_docs.py`` can plant a defect and prove the
check catches it.

**This module is a defect detector, and Milestone 12's scoring discipline applies to it.**
A doc checker that reports zero violations is indistinguishable, on that number alone, from a
checker that looks at nothing — the mirror image of the flag-everything detector
:mod:`~eval_dataset_forge.scoring` uses as a control. So the tests ship the opposite control:
each check is shown to *fire* on a planted defect. And the recall of :data:`RETIRED_CLAIMS` over
"false sentences in this repository" is **not measurable**, because no ground-truth list of false
sentences exists. It catches phrasings, not propositions. A paraphrase evades it. What pins the
*fact* is a behavioural test (``tests/test_contracts.py``); the lint only stops the retired
phrasing from being typed again.

Two structural decisions carry weight.

**Live docs and historical records are different things.** ``README.md`` and ``docs/*.md`` state
what is true now. ``docs/milestones/*.md`` and ``docs/review_notes/*.md`` state what was true at a
commit, and ``docs/milestones/12-smoke-audit.md`` correctly records ``# 1706 tests`` when the suite
now holds more. Checking counts against history would force rewriting the record to match the
present, which is the one thing a record must never do. So counts and retired claims are checked
against :data:`LIVE_DOCS` only, while links, commands, and referenced paths — which must work
wherever they appear — are checked everywhere.

**Use is not mention.** ``docs/LIMITATIONS.md`` contains the retired sentence, on purpose, in
order to say it is false. A lint that cannot tell an assertion from a quotation would either flag
the correction or need an allowlist naming every file permitted to discuss it. :func:`_asserted`
draws the line where English draws it: a retired claim inside double quotes is *mentioned*, and
anywhere else it is *asserted*. This keeps :attr:`RetiredClaim.cited_in` empty, and an allowlist
that stays empty is one nobody can quietly grow.
"""

from __future__ import annotations

import pathlib
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Document surfaces
# --------------------------------------------------------------------------- #

#: Files whose claims are about the present. Drift here is a defect.
LIVE_DOC_GLOBS: tuple[str, ...] = ("README.md", "docs/*.md")

#: Dated records. Their numbers were true when written and must not be rewritten.
HISTORICAL_DOC_GLOBS: tuple[str, ...] = ("docs/milestones/*.md", "docs/review_notes/*.md")

#: The claim ledger: one row per registered numeric claim.
LEDGER_PATH = "docs/CLAIMS.md"

#: Fenced blocks the command checks read. Prose backticks are *not* parsed: an inline
#: ``evalforge release --drop exact,normalized`` inside a recommendation string is emitted by the
#: code, not written by hand, so linting it would test the code against itself.
_FENCE = re.compile(r"```(?:bash|sh|shell|console)\n(.*?)```", re.S)

_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: Code spans, at both levels. ``docs/DATASET_CARD.md`` contains a literal ``[text](url)`` inside
#: backticks, as an example of the Markdown the card generator must escape. It is a link that is
#: *mentioned*, not *used*, and following it is a category error — the same use/mention confusion
#: :func:`_asserted` resolves one level up, for sentences rather than links.
_FENCED_ANY = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"``[^`]*``|`[^`\n]*`")

#: ``qa-006``, ``agent-003``, ``ret-001``, ``saf-007``, ``bio-009``.
_EXAMPLE_ID = re.compile(r"\b((?:qa|agent|bio|ret|saf)-\d{3})\b")

#: ``python3 -m pytest -q tests/test_dedup.py      # 99 tests``
_TEST_COUNT = re.compile(r"pytest[^\n#]*?tests/(test_\w+\.py)[^\n#]*#\s*(\d+)\s+tests?")

_ENV_PREFIX = re.compile(r"^[A-Z_][A-Z0-9_]*=")


def mask_code(text: str) -> str:
    """Blank out fenced blocks and inline code spans, preserving every offset and newline.

    Line numbers computed against the masked text still point at the original file.
    """
    masked = text
    for regex in (_FENCED_ANY, _INLINE_CODE):
        chars = list(masked)
        for match in regex.finditer(masked):
            for index in range(match.start(), match.end()):
                if chars[index] != "\n":
                    chars[index] = " "
        masked = "".join(chars)
    return masked


@dataclass(frozen=True)
class Doc:
    """One documentation file, read once."""

    path: str
    text: str

    def line_of(self, offset: int) -> int:
        return self.text.count("\n", 0, offset) + 1

    @property
    def prose(self) -> str:
        """``text`` with every code span blanked, for the checks that read English."""
        return mask_code(self.text)


@dataclass(frozen=True)
class Violation:
    """One documented claim that the repository does not support."""

    check: str
    path: str
    line: int
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.path}:{self.line}: [{self.check}] {self.message}"


# --------------------------------------------------------------------------- #
# The claim ledger
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Claim:
    """A number the documentation states, bound to the command that produces it.

    The ``command`` field is the load-bearing one, and it exists because of a measurement this
    milestone made. ``docs/RESULTS.md`` reports the ``qa_toy`` difficulty calibration p-value as
    ``0.789``. ``evalforge audit`` on the same fixture, at the same seed, prints ``0.777``.
    Neither is wrong: ``evalforge difficulty`` defaults to ``--resamples 10000`` and
    ``evalforge audit`` defaults to ``--resamples 2000``, and the gap is the permutation test's
    Monte-Carlo error. **A number in a document is not a claim until it names the command that
    produced it**, because without one there is nothing to disagree with.

    ``echoes`` names the other documents that repeat the value. A claim can be right in the ledger
    and stale in the README; the echo check is what makes that a test failure.
    """

    claim_id: str
    description: str
    command: str
    tolerance: float = 0.0
    echoes: tuple[tuple[str, str], ...] = ()


#: Every registered claim is verified twice: its ledger row must match the computed value, and
#: each echo site must repeat it. ``observe_numbers`` is the only place a value is computed.
CLAIMS: tuple[Claim, ...] = (
    Claim(
        "fixture_count",
        "committed fixture datasets",
        "python3 scripts/validate_fixtures.py",
        echoes=(("README.md", r"(\d+) committed fixture datasets"),),
    ),
    Claim(
        "pooled_examples",
        "examples in the five fixtures pooled",
        "PYTHONPATH=src python3 -m eval_dataset_forge smoke --out results/reports/smoke_audit.md",
        echoes=(("README.md", r"pooled corpus of (\d+) examples"),),
    ),
    Claim(
        "pooled_blocks_tfidf",
        "independent blocks in the pooled corpus (near_method=tfidf)",
        "PYTHONPATH=src python3 -m eval_dataset_forge smoke --out results/reports/smoke_audit.md",
        echoes=(("README.md", r"(\d+) independent blocks in the pooled corpus"),),
    ),
    # No CLI command produces these two. `audit` takes one dataset and `smoke` pools the five but
    # exposes no --near-method, so a corpus-level Jaccard block count has no user-facing producer.
    # The gap is real and recorded rather than papered over with a command that does not exist.
    Claim(
        "pooled_blocks_jaccard",
        "independent blocks in the pooled corpus (near_method=jaccard)",
        "python3 scripts/verify_docs.py --json",
    ),
    Claim(
        "summed_fixture_blocks_tfidf",
        "independent blocks summed over the five fixtures audited separately",
        "python3 scripts/verify_docs.py --json",
    ),
    Claim(
        "subcommand_count",
        "evalforge subcommands",
        "PYTHONPATH=src python3 -m eval_dataset_forge --help",
        echoes=(("README.md", r"All (\d+) subcommands"),),
    ),
    Claim(
        "verify_release_checks",
        "integrity checks run by verify-release",
        "PYTHONPATH=src python3 -m eval_dataset_forge verify-release "
        "--release datasets/releases/qa_toy_v0.1.0",
        echoes=(("README.md", r"all (\d+) `verify-release` checks"),),
    ),
    Claim(
        "demo_block_count",
        "independent blocks in the committed demo release",
        "PYTHONPATH=src python3 -m eval_dataset_forge verify-release "
        "--release datasets/releases/qa_toy_v0.1.0",
        echoes=(("README.md", r"\| independent blocks \| \*\*(\d+)\*\* \|"),),
    ),
    Claim(
        "demo_test_blocks",
        "independent blocks in the demo release's test split",
        "PYTHONPATH=src python3 -m eval_dataset_forge dataset-card "
        "--release datasets/releases/qa_toy_v0.1.0",
        echoes=(("README.md", r"\| blocks in `test` \| \*\*(\d+)\*\* \|"),),
    ),
    Claim(
        "qa_toy_calibration_p_b2000",
        "qa_toy difficulty calibration permutation p, as `audit` reports it",
        "PYTHONPATH=src python3 -m eval_dataset_forge audit "
        "--dataset datasets/fixtures/qa_toy/examples.jsonl --out results/audits/qa_toy/",
        tolerance=5e-4,
    ),
    Claim(
        "qa_toy_calibration_p_b10000",
        "qa_toy difficulty calibration permutation p, as `difficulty` reports it",
        "PYTHONPATH=src python3 -m eval_dataset_forge difficulty "
        "--dataset datasets/fixtures/qa_toy/examples.jsonl",
        tolerance=5e-4,
    ),
)

CLAIMS_BY_ID: dict[str, Claim] = {claim.claim_id: claim for claim in CLAIMS}


# --------------------------------------------------------------------------- #
# Retired claims
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetiredClaim:
    """A sentence a milestone measured false, banned from live prose.

    The lint does not verify the *fact* — ``retired_by`` names the milestone that measured it and
    ``pinned_by`` names the test that keeps it measured. This rule exists only so that the retired
    phrasing cannot be typed again, in a file no test reads, and survive review.
    """

    claim_id: str
    pattern: str
    retired_by: str
    because: str
    superseded_by: str
    #: Files permitted to *assert* the claim anyway. Empty, and meant to stay empty: quoting a
    #: retired claim in order to refute it is a mention, not an assertion, and :func:`_asserted`
    #: already allows it.
    cited_in: tuple[str, ...] = ()


RETIRED_CLAIMS: tuple[RetiredClaim, ...] = (
    RetiredClaim(
        claim_id="determinism_omits_environment",
        # Covers "deterministic given a fixed input and seed" and the inverted
        # "Given a fixed input and seed, the pipeline is reproducible."
        pattern=r"given (?:a|the same) (?:fixed )?inputs?,? and (?:a |an )?seed",
        retired_by="Milestone 13",
        because=(
            "the build environment is an input: scikit-learn's presence resolves near-duplicate "
            "detection to tfidf rather than jaccard, which changes the blocks, the adversarial "
            "split, and eval_id"
        ),
        superseded_by="docs/REPRODUCIBILITY.md § The environment is an input",
    ),
)

#: Rules that were written, measured, and **not enforced**, following the precedent of Milestone
#: 13's ``difficulty_is_calibrated`` requirement — registered, in no profile, and pinned by a test
#: asserting nothing picks it up.
#:
#: ``sklearn_is_an_accelerator`` is a true proposition to retire: ``docs/PLAN.md`` §3 called
#: scikit-learn an *optional accelerator with a pure-Python fallback*, and Milestone 13 showed it
#: is a different detector. But the phrasing is the same phrasing every document that *corrects*
#: the error must use, and the correction is not a quotation, so :func:`_asserted` cannot separate
#: them. Enforcing it needs an allowlist naming most of the live docs — at which point the rule
#: asserts nothing. It is kept here, with its measured cost, rather than shipped at a precision
#: that would train a reader to ignore it.
CANDIDATE_RETIRED_CLAIMS: tuple[RetiredClaim, ...] = (
    RetiredClaim(
        claim_id="sklearn_is_an_accelerator",
        pattern=r"optional accelerators?[^.\n]{0,80}pure-Python fallbacks?",
        retired_by="Milestone 13",
        because="scikit-learn changes which examples are duplicates, not how fast they are found",
        superseded_by="docs/LIMITATIONS.md § An \"optional accelerator\" changes the answer",
    ),
)


# --------------------------------------------------------------------------- #
# Reading documents
# --------------------------------------------------------------------------- #


def _read(root: pathlib.Path, globs: Iterable[str]) -> list[Doc]:
    docs: list[Doc] = []
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                docs.append(Doc(path.relative_to(root).as_posix(), path.read_text()))
    return docs


def live_docs(root: pathlib.Path) -> list[Doc]:
    """``README.md`` and ``docs/*.md`` — claims about the present."""
    return _read(root, LIVE_DOC_GLOBS)


def historical_docs(root: pathlib.Path) -> list[Doc]:
    """Milestone and review notes — dated records, never rewritten to match the present."""
    return _read(root, HISTORICAL_DOC_GLOBS)


def all_docs(root: pathlib.Path) -> list[Doc]:
    return [*live_docs(root), *historical_docs(root)]


# --------------------------------------------------------------------------- #
# Command extraction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Command:
    """A shell command lifted from a fenced block, with the line it starts on."""

    doc: str
    line: int
    text: str
    tokens: tuple[str, ...]


def extract_commands(doc: Doc) -> list[Command]:
    """Every shell command in ``doc``'s fenced blocks, with continuations joined.

    Comment-only lines and trailing ``# ...`` comments are dropped, and leading ``VAR=value``
    environment prefixes are stripped, so ``PYTHONPATH=src python3 -m eval_dataset_forge audit``
    and ``evalforge audit`` reduce to the same token stream.
    """
    commands: list[Command] = []
    for block in _FENCE.finditer(doc.text):
        base_line = doc.line_of(block.start(1))
        raw_lines = block.group(1).split("\n")
        i = 0
        while i < len(raw_lines):
            start = i
            line = raw_lines[i]
            while line.rstrip().endswith("\\") and i + 1 < len(raw_lines):
                i += 1
                line = line.rstrip()[:-1] + " " + raw_lines[i]
            i += 1
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            text = re.sub(r"\s+#.*$", "", text).strip()
            if not text:
                continue
            try:
                tokens = shlex.split(text)
            except ValueError:
                continue
            while tokens and _ENV_PREFIX.match(tokens[0]):
                tokens.pop(0)
            if not tokens:
                continue
            commands.append(Command(doc.path, base_line + start, text, tuple(tokens)))
    return commands


def _cli_invocation(tokens: Sequence[str]) -> tuple[str, ...] | None:
    """Return the arguments after the ``evalforge`` entry point, or ``None``.

    Both spellings are the same program: ``[project.scripts]`` maps ``evalforge`` to
    ``eval_dataset_forge.cli:main``. The bare form only runs once the package is installed, which
    is why :func:`check_entry_point` exists separately.
    """
    if tokens and tokens[0] == "evalforge":
        return tuple(tokens[1:])
    if list(tokens[:3]) == ["python3", "-m", "eval_dataset_forge"]:
        return tuple(tokens[3:])
    return None


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_links(docs: Sequence[Doc], root: pathlib.Path) -> list[Violation]:
    """Every relative Markdown link resolves to a file that exists.

    Reads ``doc.prose``, so a ``[text](url)`` written inside backticks as an example of Markdown
    is not chased to a file named ``url``.
    """
    violations: list[Violation] = []
    for doc in docs:
        parent = (root / doc.path).parent
        for match in _MD_LINK.finditer(doc.prose):
            target = match.group(1).split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (parent / target).exists():
                violations.append(
                    Violation(
                        "links",
                        doc.path,
                        doc.line_of(match.start()),
                        f"link target does not exist: {target}",
                    )
                )
    return violations


def check_cli_commands(docs: Sequence[Doc], flags: Mapping[str, set[str]]) -> list[Violation]:
    """Every documented ``evalforge`` command names a real subcommand and real options.

    ``flags`` comes from the live ``argparse`` parser, so renaming a flag in ``cli.py`` fails the
    documentation rather than silently orphaning it.
    """
    violations: list[Violation] = []
    for doc in docs:
        for command in extract_commands(doc):
            rest = _cli_invocation(command.tokens)
            if rest is None:
                continue
            rest = tuple(token for token in rest if token not in ("-h", "--help"))
            if not rest or rest[0].startswith("-"):
                continue
            sub, *args = rest
            if sub not in flags:
                violations.append(
                    Violation(
                        "cli-commands",
                        doc.path,
                        command.line,
                        f"unknown subcommand {sub!r}: {command.text}",
                    )
                )
                continue
            for token in args:
                if not token.startswith("--"):
                    continue
                option = token.split("=", 1)[0]
                if option not in flags[sub]:
                    violations.append(
                        Violation(
                            "cli-commands",
                            doc.path,
                            command.line,
                            f"unknown option {option!r} for {sub!r}: {command.text}",
                        )
                    )
    return violations


def check_referenced_paths(docs: Sequence[Doc], root: pathlib.Path) -> list[Violation]:
    """Scripts and test files named in documented commands exist."""
    violations: list[Violation] = []
    for doc in docs:
        for command in extract_commands(doc):
            for token in command.tokens:
                candidate = token.lstrip("./")
                looks_like_repo_path = candidate.startswith(("scripts/", "tests/")) and (
                    candidate.endswith((".py", ".sh"))
                )
                if looks_like_repo_path and not (root / candidate).exists():
                    violations.append(
                        Violation(
                            "referenced-paths",
                            doc.path,
                            command.line,
                            f"referenced path does not exist: {token}",
                        )
                    )
    return violations


def check_test_counts(docs: Sequence[Doc], counts: Mapping[str, int]) -> list[Violation]:
    """``pytest -q tests/test_x.py  # N tests`` states the number pytest collects.

    Applied to live docs only. ``docs/milestones/12-smoke-audit.md`` says ``# 1706 tests`` and is
    right about the commit it describes.
    """
    violations: list[Violation] = []
    for doc in docs:
        for match in _TEST_COUNT.finditer(doc.text):
            name, claimed = match.group(1), int(match.group(2))
            actual = counts.get(name)
            line = doc.line_of(match.start())
            if actual is None:
                violations.append(
                    Violation("test-counts", doc.path, line, f"no such test file: {name}")
                )
            elif actual != claimed:
                violations.append(
                    Violation(
                        "test-counts",
                        doc.path,
                        line,
                        f"{name} claims {claimed} tests, pytest collects {actual}",
                    )
                )
    return violations


def check_example_ids(docs: Sequence[Doc], known: Iterable[str]) -> list[Violation]:
    """Every fixture example id cited in prose exists in a committed fixture."""
    known = set(known)
    violations: list[Violation] = []
    for doc in docs:
        for match in _EXAMPLE_ID.finditer(doc.text):
            if match.group(1) not in known:
                violations.append(
                    Violation(
                        "example-ids",
                        doc.path,
                        doc.line_of(match.start()),
                        f"cites an example id no fixture contains: {match.group(1)}",
                    )
                )
    return violations


def _asserted(text: str, start: int) -> bool:
    """True when the span beginning at ``start`` is *used*, not *mentioned*.

    A retired claim wrapped in double quotes is being reported, not believed — ``LIMITATIONS.md``,
    ``INTEGRATION.md``, and ``RESULTS.md`` all quote the sentence in order to say it is false.
    Outside quotes it is asserted.

    The unit is the **paragraph**, not the line. ``docs/RESULTS.md`` opens its quotation on one
    line and closes it on the next, and a line-scoped rule reads the continuation as an assertion —
    which is precisely what the first draft of this function did, producing a false positive on the
    one document most careful to disown the claim. Quotations wrap across lines; they do not wrap
    across a blank line.

    Failure mode: an unbalanced double quote earlier in the same paragraph inverts the verdict for
    everything after it. Apostrophes are not counted, so the usual culprit is absent.
    """
    paragraph_start = text.rfind("\n\n", 0, start) + 1
    prefix = text[paragraph_start:start]
    quotes = prefix.count('"') + prefix.count("“") + prefix.count("”")
    return quotes % 2 == 0


def check_retired_claims(
    docs: Sequence[Doc], retired: Sequence[RetiredClaim] = RETIRED_CLAIMS
) -> list[Violation]:
    """No live document *asserts* a sentence a milestone measured false.

    Reads ``doc.prose``: a retired claim inside a fenced block or a code span is a transcript, not
    a belief.
    """
    violations: list[Violation] = []
    for doc in docs:
        prose = doc.prose
        for claim in retired:
            if doc.path in claim.cited_in:
                continue
            for match in re.finditer(claim.pattern, prose, re.I):
                if not _asserted(prose, match.start()):
                    continue
                violations.append(
                    Violation(
                        "retired-claims",
                        doc.path,
                        doc.line_of(match.start()),
                        f"asserts {claim.claim_id!r}, retired by {claim.retired_by} "
                        f"({claim.because}); see {claim.superseded_by}",
                    )
                )
    return violations


def _ledger_rows(ledger: Doc) -> dict[str, tuple[int, str]]:
    """``| `claim_id` | value | ... |`` → ``{claim_id: (line, value)}``."""
    rows: dict[str, tuple[int, str]] = {}
    for number, line in enumerate(ledger.text.split("\n"), 1):
        match = re.match(r"\s*\|\s*`([a-z0-9_]+)`\s*\|\s*\*{0,2}([0-9.]+)\*{0,2}\s*\|", line)
        if match:
            rows[match.group(1)] = (number, match.group(2))
    return rows


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance + 1e-12


def check_ledger(
    docs: Sequence[Doc],
    computed: Mapping[str, float],
    claims: Sequence[Claim] = CLAIMS,
) -> list[Violation]:
    """Every registered claim has a ledger row whose value the code reproduces.

    Orphan rows are violations too: a ledger that can hold an unverified number is a document, not
    a ledger.
    """
    by_path = {doc.path: doc for doc in docs}
    ledger = by_path.get(LEDGER_PATH)
    if ledger is None:
        return [Violation("ledger", LEDGER_PATH, 0, "the claim ledger is missing")]

    rows = _ledger_rows(ledger)
    violations: list[Violation] = []
    for claim in claims:
        if claim.claim_id not in rows:
            violations.append(
                Violation("ledger", LEDGER_PATH, 0, f"claim {claim.claim_id!r} has no ledger row")
            )
            continue
        line, raw = rows[claim.claim_id]
        if claim.claim_id not in computed:
            violations.append(
                Violation("ledger", LEDGER_PATH, line, f"claim {claim.claim_id!r} is not computed")
            )
            continue
        stated, actual = float(raw), float(computed[claim.claim_id])
        if not _close(stated, actual, claim.tolerance):
            violations.append(
                Violation(
                    "ledger",
                    LEDGER_PATH,
                    line,
                    f"{claim.claim_id} states {raw}, `{claim.command}` produces {actual:g}",
                )
            )
    registered = {claim.claim_id for claim in claims}
    for claim_id, (line, _) in rows.items():
        if claim_id not in registered:
            violations.append(
                Violation("ledger", LEDGER_PATH, line, f"unregistered claim row: {claim_id!r}")
            )
    return violations


def check_echoes(
    docs: Sequence[Doc],
    computed: Mapping[str, float],
    claims: Sequence[Claim] = CLAIMS,
) -> list[Violation]:
    """Documents that repeat a claim's value repeat the current one.

    The ledger can be right while the README is stale; this is the check that notices.
    """
    by_path = {doc.path: doc for doc in docs}
    violations: list[Violation] = []
    for claim in claims:
        for path, pattern in claim.echoes:
            doc = by_path.get(path)
            if doc is None:
                violations.append(
                    Violation("echoes", path, 0, f"echo site for {claim.claim_id} is missing")
                )
                continue
            match = re.search(pattern, doc.text)
            if match is None:
                violations.append(
                    Violation(
                        "echoes",
                        path,
                        0,
                        f"{claim.claim_id}: no text matches /{pattern}/ — the claim was reworded "
                        "or removed, so nothing checks it here any more",
                    )
                )
                continue
            stated = float(match.group(1))
            actual = float(computed[claim.claim_id])
            if not _close(stated, actual, claim.tolerance):
                violations.append(
                    Violation(
                        "echoes",
                        path,
                        doc.line_of(match.start()),
                        f"{claim.claim_id} states {match.group(1)}, "
                        f"`{claim.command}` produces {actual:g}",
                    )
                )
    return violations


def check_claim_commands(
    root: pathlib.Path,
    flags: Mapping[str, set[str]],
    claims: Sequence[Claim] = CLAIMS,
) -> list[Violation]:
    """Each claim's ``command`` is a command that exists.

    Without this, the ledger's most important column is the one nothing reads. It caught the first
    draft of ``pooled_blocks_jaccard``, which cited ``evalforge audit --near-method jaccard`` — a
    flag that did not exist. That is how Milestone 14 discovered ``docs/RESULTS.md`` had been
    publishing Jaccard figures no command could produce.
    """
    violations: list[Violation] = []
    for claim in claims:
        tokens = [t for t in shlex.split(claim.command) if not _ENV_PREFIX.match(t)]
        for token in tokens:
            candidate = token.lstrip("./")
            if candidate.startswith(("scripts/", "tests/")) and not (root / candidate).exists():
                violations.append(
                    Violation(
                        "claim-commands",
                        LEDGER_PATH,
                        0,
                        f"{claim.claim_id}: command names a missing path: {token}",
                    )
                )
        rest = _cli_invocation(tokens)
        if rest is None:
            continue
        rest = tuple(token for token in rest if token not in ("-h", "--help"))
        if not rest or rest[0].startswith("-"):
            continue
        sub, *args = rest
        if sub not in flags:
            violations.append(
                Violation(
                    "claim-commands",
                    LEDGER_PATH,
                    0,
                    f"{claim.claim_id}: unknown subcommand {sub!r}",
                )
            )
            continue
        for token in args:
            if token.startswith("--") and token.split("=", 1)[0] not in flags[sub]:
                violations.append(
                    Violation(
                        "claim-commands",
                        LEDGER_PATH,
                        0,
                        f"{claim.claim_id}: unknown option {token.split('=', 1)[0]!r} for {sub!r}",
                    )
                )
    return violations


def check_entry_point(root: pathlib.Path) -> list[Violation]:
    """``[project.scripts]`` maps ``evalforge`` to a target that imports.

    The documentation's primary spelling is ``evalforge <sub>``, which only exists on ``PATH``
    after the package is installed. Nothing in the offline test suite installs it, so the binary
    itself cannot be exercised. What *can* be checked is that the declared target resolves — the
    failure mode being a rename in ``cli.py`` that leaves ``pyproject.toml`` pointing at nothing.
    """
    pyproject = (root / "pyproject.toml").read_text()
    match = re.search(r"^\s*evalforge\s*=\s*\"([^\"]+)\"", pyproject, re.M)
    if match is None:
        return [Violation("entry-point", "pyproject.toml", 0, "no `evalforge` console script")]
    target = match.group(1)
    module, _, attribute = target.partition(":")
    try:
        imported = __import__(module, fromlist=[attribute or "__doc__"])
    except ImportError as exc:  # pragma: no cover - import failure is environmental
        return [Violation("entry-point", "pyproject.toml", 0, f"{target} does not import: {exc}")]
    if attribute and not callable(getattr(imported, attribute, None)):
        return [
            Violation("entry-point", "pyproject.toml", 0, f"{target} is not callable")
        ]
    return []


# --------------------------------------------------------------------------- #
# Observing the world
# --------------------------------------------------------------------------- #


def observe_cli_flags() -> dict[str, set[str]]:
    """``{subcommand: {option strings}}`` from the live parser."""
    from eval_dataset_forge.cli import build_parser

    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "_name_parser_map")
    )
    flags: dict[str, set[str]] = {}
    for name, sub in subparsers.choices.items():
        options: set[str] = set()
        for action in sub._actions:
            options.update(action.option_strings)
        flags[name] = options
    return flags


def observe_test_counts(root: pathlib.Path) -> dict[str, int]:
    """``{test_file: tests pytest collects}`` — one collection, no execution."""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    counts: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if "::" not in line or not line.startswith("tests/"):
            continue
        name = pathlib.PurePosixPath(line.split("::", 1)[0]).name
        counts[name] = counts.get(name, 0) + 1
    return counts


def observe_example_ids(root: pathlib.Path) -> set[str]:
    from eval_dataset_forge.datasets.loader import discover_fixtures, load_examples

    ids: set[str] = set()
    for fixture in discover_fixtures(root / "datasets" / "fixtures"):
        ids.update(example.example_id for example in load_examples(fixture / "examples.jsonl"))
    return ids


def observe_numbers(root: pathlib.Path) -> dict[str, float]:
    """Recompute every registered claim from the code and the committed artifacts.

    Nothing here reads a document. That is the point: the ledger is compared against a value this
    function derived, not against a value someone typed.
    """
    import json

    from eval_dataset_forge.datasets.loader import discover_fixtures, load_examples
    from eval_dataset_forge.datasets.manifest import verify_release
    from eval_dataset_forge.dedup import find_duplicates
    from eval_dataset_forge.difficulty.calibration import calibrate
    from eval_dataset_forge.splits.blocks import duplicate_blocks

    fixtures = discover_fixtures(root / "datasets" / "fixtures")
    per_fixture = {path.name: load_examples(path / "examples.jsonl") for path in fixtures}
    pooled = [example for name in sorted(per_fixture) for example in per_fixture[name]]

    tfidf = find_duplicates(pooled, near_method="tfidf")
    jaccard = find_duplicates(pooled, near_method="jaccard")

    summed = 0
    for examples in per_fixture.values():
        summed += len(duplicate_blocks(examples, find_duplicates(examples, near_method="tfidf")))

    release_dir = root / "datasets" / "releases" / "qa_toy_v0.1.0"
    verify = verify_release(release_dir)
    manifest = json.loads((release_dir / "manifest.json").read_text())
    dependence = manifest.get("dependence", {})

    qa_toy = per_fixture["qa_toy"]

    return {
        "fixture_count": len(fixtures),
        "pooled_examples": len(pooled),
        "pooled_blocks_tfidf": len(duplicate_blocks(pooled, tfidf)),
        "pooled_blocks_jaccard": len(duplicate_blocks(pooled, jaccard)),
        "summed_fixture_blocks_tfidf": summed,
        "subcommand_count": len(observe_cli_flags()),
        "verify_release_checks": len(verify.checks),
        "demo_block_count": dependence["block_count"],
        "demo_test_blocks": dependence["blocks_by_split"]["test"],
        "qa_toy_calibration_p_b2000": calibrate(qa_toy, n_resamples=2_000, seed=0).p_value,
        "qa_toy_calibration_p_b10000": calibrate(qa_toy, n_resamples=10_000, seed=0).p_value,
    }


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


@dataclass
class DocsAuditReport:
    """What the six checks found, and how much they looked at."""

    violations: list[Violation] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def by_check(self) -> dict[str, list[Violation]]:
        grouped: dict[str, list[Violation]] = {}
        for violation in self.violations:
            grouped.setdefault(violation.check, []).append(violation)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stats": dict(self.stats),
            "violations": [
                {
                    "check": v.check,
                    "path": v.path,
                    "line": v.line,
                    "message": v.message,
                }
                for v in self.violations
            ],
        }


def audit_docs(root: pathlib.Path) -> DocsAuditReport:
    """Run every check against the repository at ``root``."""
    live = live_docs(root)
    every = all_docs(root)

    flags = observe_cli_flags()
    counts = observe_test_counts(root)
    ids = observe_example_ids(root)
    numbers = observe_numbers(root)

    violations = [
        *check_links(every, root),
        *check_cli_commands(every, flags),
        *check_referenced_paths(every, root),
        *check_test_counts(live, counts),
        *check_example_ids(every, ids),
        *check_retired_claims(live),
        *check_ledger(live, numbers),
        *check_echoes(live, numbers),
        *check_claim_commands(root, flags),
        *check_entry_point(root),
    ]

    stats = {
        "live_docs": len(live),
        "historical_docs": len(every) - len(live),
        "links": sum(len(_MD_LINK.findall(doc.prose)) for doc in every),
        "commands": sum(len(extract_commands(doc)) for doc in every),
        "cli_commands": sum(
            1
            for doc in every
            for command in extract_commands(doc)
            if _cli_invocation(command.tokens) is not None
        ),
        "claims": len(CLAIMS),
        "retired_claims": len(RETIRED_CLAIMS),
        "candidate_retired_claims": len(CANDIDATE_RETIRED_CLAIMS),
    }
    return DocsAuditReport(violations=violations, stats=stats)


def format_report(report: DocsAuditReport) -> str:
    """A short, greppable summary; violations grouped by check."""
    lines = ["docs audit"]
    for key in sorted(report.stats):
        lines.append(f"  {key:26s}: {report.stats[key]}")
    if report.ok:
        lines.append("\nno violations: every documented claim is backed by the repository")
        return "\n".join(lines)
    grouped = report.by_check()
    lines.append(f"\n{len(report.violations)} violation(s) across {len(grouped)} check(s):")
    for check in sorted(grouped):
        lines.append(f"\n[{check}]")
        for violation in grouped[check]:
            lines.append(f"  {violation.path}:{violation.line}: {violation.message}")
    return "\n".join(lines)


__all__ = [
    "CANDIDATE_RETIRED_CLAIMS",
    "CLAIMS",
    "CLAIMS_BY_ID",
    "Claim",
    "Command",
    "Doc",
    "DocsAuditReport",
    "LEDGER_PATH",
    "RETIRED_CLAIMS",
    "RetiredClaim",
    "Violation",
    "all_docs",
    "audit_docs",
    "check_claim_commands",
    "check_cli_commands",
    "check_echoes",
    "check_entry_point",
    "check_example_ids",
    "check_ledger",
    "check_links",
    "check_referenced_paths",
    "check_retired_claims",
    "check_test_counts",
    "extract_commands",
    "format_report",
    "historical_docs",
    "live_docs",
    "observe_cli_flags",
    "observe_example_ids",
    "observe_numbers",
    "observe_test_counts",
]
