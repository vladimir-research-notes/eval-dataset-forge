"""A minimal document model shared by every rendered report (Milestone 11).

A dataset card has to be Markdown, because that is what a release bundle can carry and what
a code host renders. It also has to be HTML, because that is what a reader opens. The
tempting implementation is two string templates.

Two templates are two places a number is written down, and the second one is where the
number goes stale. So a report is built once as a :class:`Document` — a flat list of
headings, paragraphs, bullet lists, tables, notes, and code blocks — and rendered twice.
:mod:`eval_dataset_forge.reports.markdown` and :mod:`eval_dataset_forge.reports.html` are
the only modules that know a syntax; nothing that computes a fact knows one. A test walks
the document and asserts that every string it carries survives into both renderings, which
is a property no template pair can offer.

**The inline vocabulary is one item long: a code span.** ``inline_spans`` splits text on
backticks and everything else is literal, escaped by whichever backend is running. Bold,
links, and nested emphasis are deliberately absent: supporting them in two backends means
writing a Markdown parser, and the dependency budget (`pydantic`, `numpy`, `pyyaml`) has no
room for one. The cost is that a card cannot link out. The benefit is that a license note
containing ``<script>`` or a ``|`` cannot break the table it sits in, in either format —
the escaping is the renderer's job, and there is exactly one renderer per format.

Blocks are frozen dataclasses in a tuple. A document is therefore a value: two calls that
compute the same card compare equal, which is what makes ``dataset-card --check`` a
comparison of *meaning* rather than of bytes that happen to match.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# Span kinds. ``CODE`` renders as a Markdown code span and an HTML ``<code>``; ``TEXT`` is
# literal and is escaped by the backend.
TEXT = "text"
CODE = "code"


@dataclass(frozen=True)
class Span:
    """One run of inline text: literal prose, or a code span."""

    kind: str
    text: str


def inline_spans(text: str) -> tuple[Span, ...]:
    """Split ``text`` into literal and code spans on paired backticks.

    An *odd* number of backticks means at least one is unpaired, and an unpaired backtick is
    far more often a literal one (a shell prompt, a quoted character) than the start of a
    span the author forgot to close. Such a string is returned as a single literal span, and
    the backend escapes the backticks. The rule is total — every string has a rendering —
    and it never silently swallows text, which a "close it at the end" rule would do.
    """
    parts = text.split("`")
    if len(parts) % 2 == 0:  # an even part count means an odd backtick count
        return (Span(TEXT, text),)
    return tuple(
        Span(CODE if index % 2 else TEXT, part)
        for index, part in enumerate(parts)
        if part  # an empty run carries no text in either format
    )


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Heading:
    """A section heading. ``level`` is 2..6; level 1 belongs to :attr:`Document.title`."""

    text: str
    level: int = 2

    def __post_init__(self) -> None:
        if not 2 <= self.level <= 6:
            raise ValueError(f"heading level must be 2..6, got {self.level}")


@dataclass(frozen=True)
class Paragraph:
    text: str


@dataclass(frozen=True)
class Bullets:
    """An unordered list. An empty list is legal and renders as nothing."""

    items: tuple[str, ...] = ()


@dataclass(frozen=True)
class Table:
    """A table with a header row. Every row must be as wide as ``headers``."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...] = ()
    caption: str = ""

    def __post_init__(self) -> None:
        if not self.headers:
            raise ValueError("a table needs at least one column")
        for row in self.rows:
            if len(row) != len(self.headers):
                raise ValueError(
                    f"row {row!r} has {len(row)} cells, expected {len(self.headers)}"
                )


@dataclass(frozen=True)
class Note:
    """An aside: a caveat about the block above it. Blockquote in md, ``<aside>`` in html."""

    text: str


@dataclass(frozen=True)
class Code:
    """A verbatim block. Never escaped, never wrapped, never parsed for inline spans."""

    text: str
    language: str = ""


Block = Heading | Paragraph | Bullets | Table | Note | Code


@dataclass(frozen=True)
class Document:
    """A rendered report before it has a syntax."""

    title: str
    blocks: tuple[Block, ...] = field(default_factory=tuple)


def strings_of(block: Block) -> tuple[str, ...]:
    """Every author-supplied string in ``block``, in reading order.

    Used by the renderers' shared test to assert that no backend drops content. ``Code`` is
    included as a single opaque string because that is exactly how both backends treat it.
    """
    match block:
        case Heading(text=text) | Paragraph(text=text) | Note(text=text) | Code(text=text):
            return (text,)
        case Bullets(items=items):
            return tuple(items)
        case Table(headers=headers, rows=rows, caption=caption):
            cells = tuple(cell for row in rows for cell in row)
            return (*headers, *cells, *((caption,) if caption else ()))
    raise TypeError(f"unknown block type: {type(block).__name__}")


def document_strings(document: Document) -> tuple[str, ...]:
    """Every author-supplied string in ``document``, title first."""
    return (document.title, *(text for block in document.blocks for text in strings_of(block)))


def blocks_of(*items: Block | Iterable[Block] | None) -> tuple[Block, ...]:
    """Flatten block builders into a tuple, dropping ``None``.

    Lets a report say ``blocks_of(Heading("Splits"), _split_table(card), _caveats(card))``
    where a helper returns nothing when it has nothing to say.
    """
    out: list[Block] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, Heading | Paragraph | Bullets | Table | Note | Code):
            out.append(item)
        else:
            out.extend(item)
    return tuple(out)


def rows(values: Iterable[Sequence[object]]) -> tuple[tuple[str, ...], ...]:
    """Coerce an iterable of sequences into table rows of strings."""
    return tuple(tuple(str(cell) for cell in row) for row in values)


__all__ = [
    "CODE",
    "TEXT",
    "Block",
    "Bullets",
    "Code",
    "Document",
    "Heading",
    "Note",
    "Paragraph",
    "Span",
    "Table",
    "blocks_of",
    "document_strings",
    "inline_spans",
    "rows",
    "strings_of",
]
