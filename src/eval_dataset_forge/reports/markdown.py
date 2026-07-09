"""Markdown backend for :class:`~eval_dataset_forge.reports.document.Document` (Milestone 11).

The only module in the package that knows what a pipe character means. Everything it emits is
CommonMark, LF-terminated, and a deterministic function of the document — a card rendered
twice is byte-identical, which is what lets ``dataset-card --check`` compare a regenerated
card against the one that shipped.

**Escaping is uniform rather than contextual.** Every literal span is escaped the same way
wherever it appears, not only inside table cells. A contextual escaper has to know which
block it is inside, and the one time it guesses wrong is on a dataset whose ``license_notes``
happens to contain a ``|`` — at which point the card's table silently grows a column.
CommonMark renders ``\\|`` as ``|`` outside a table too, so uniformity costs nothing.

The escaped set includes ``<`` and ``[``, which are not decoration: CommonMark passes raw
HTML through, so a curator note containing ``<script>`` would otherwise become a script tag
in every renderer that displays the card, and ``[text](url)`` would become a live link the
curator never wrote.

The set excludes intraword ``_``. CommonMark permits intraword emphasis with ``*`` and forbids
it with ``_``, so ``qa_toy`` and ``effective_n`` cannot open emphasis and escaping them would
only put backslashes through the most-read line of every card. ``_emphasis_`` at a word
boundary still can, and still is escaped.

Newlines inside a table cell collapse to spaces. Markdown has no line break there that
CommonMark and GitHub agree on, and a real newline would end the row.
"""

from __future__ import annotations

import re

from eval_dataset_forge.reports.document import (
    CODE,
    Block,
    Bullets,
    Code,
    Document,
    Heading,
    Note,
    Paragraph,
    Span,
    Table,
    inline_spans,
)

# Characters a literal span must not be allowed to spend as syntax. ``\`` leads: escaping it
# after the others would double-escape the backslashes they introduced. ``_`` is absent — it
# is handled by :data:`_WORD_BOUNDARY_UNDERSCORE`, below.
_ESCAPES = ("\\", "`", "|", "*", "<", "[")

# CommonMark forbids intraword emphasis with ``_`` (but permits it with ``*``), so ``qa_toy``
# and ``effective_n`` are already literal and escaping them would put a backslash in the most
# frequently read line of every card. Only a boundary underscore can open emphasis.
_WORD_BOUNDARY_UNDERSCORE = re.compile(r"(?<![0-9A-Za-z])_|_(?![0-9A-Za-z])")

# Markers that only bite at the start of a line, where they would open a list, a heading, or
# a blockquote nobody wrote.
_LEADING_MARKERS = ("-", "+", "#", ">")

# ``1.`` / ``1)`` at line start opens an ordered list. The *punctuation* is escaped, never the
# digit: CommonMark only recognizes a backslash escape before ASCII punctuation, so ``\1.``
# would render as a literal backslash and still open the list.
_LEADING_ORDINAL = re.compile(r"^(\s*)(\d+)([.)])")


def escape(text: str) -> str:
    """Escape a literal span so no character in it is read as Markdown syntax."""
    for character in _ESCAPES:
        text = text.replace(character, "\\" + character)
    return _WORD_BOUNDARY_UNDERSCORE.sub(r"\\\g<0>", text)


def escape_leading(text: str) -> str:
    """Escape a line-initial list/heading/quote marker. Applied after :func:`escape`."""
    stripped = text.lstrip()
    if stripped[:1] in _LEADING_MARKERS:
        indent = text[: len(text) - len(stripped)]
        return f"{indent}\\{stripped}"
    return _LEADING_ORDINAL.sub(r"\1\2\\\3", text)


def _span(span: Span) -> str:
    if span.kind == CODE:
        # A code span is verbatim by definition. A backtick inside one would close it early;
        # CommonMark's fence-widening rule is not worth implementing for a card, whose only
        # code spans are identifiers, digests, and CLI flags.
        return "`" + span.text.replace("`", "") + "`"
    return escape(span.text)


def inline(text: str) -> str:
    """Render inline text: code spans verbatim in backticks, everything else escaped."""
    return "".join(_span(span) for span in inline_spans(text))


def cell(text: str) -> str:
    """Render one table cell. Whitespace collapses; the pipe is escaped by :func:`inline`."""
    return " ".join(inline(text).split())


def _table(block: Table) -> str:
    lines = [
        "| " + " | ".join(cell(header) for header in block.headers) + " |",
        "|" + "|".join("---" for _ in block.headers) + "|",
        *("| " + " | ".join(cell(value) for value in row) + " |" for row in block.rows),
    ]
    if block.caption:
        lines += ["", f"*{inline(block.caption)}*"]
    return "\n".join(lines)


def render_block(block: Block) -> str:
    """Render one block. Blocks are joined by a blank line; none carries its own."""
    match block:
        case Heading(text=text, level=level):
            return f"{'#' * level} {inline(text)}"
        case Paragraph(text=text):
            return escape_leading(inline(text))
        case Bullets(items=items):
            return "\n".join(f"- {escape_leading(inline(item))}" for item in items)
        case Note(text=text):
            return f"> {escape_leading(inline(text))}"
        case Code(text=text, language=language):
            body = text.rstrip("\n")
            return f"```{language}\n{body}\n```"
        case Table():
            return _table(block)
    raise TypeError(f"unknown block type: {type(block).__name__}")


def render(document: Document) -> str:
    """Render ``document`` as Markdown text, ending in exactly one newline."""
    parts = [f"# {inline(document.title)}"]
    parts += [rendered for block in document.blocks if (rendered := render_block(block))]
    return "\n\n".join(parts) + "\n"


def to_bytes(document: Document) -> bytes:
    """The exact bytes of a rendered Markdown artifact. UTF-8, LF, deterministic."""
    return render(document).encode("utf-8")


__all__ = ["cell", "escape", "escape_leading", "inline", "render", "render_block", "to_bytes"]
