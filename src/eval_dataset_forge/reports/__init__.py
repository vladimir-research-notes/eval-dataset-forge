"""Rendered reports: dataset cards now, the human-readable audit report at Milestone 12.

The package is arranged around one rule: **a module that computes a fact does not know a
syntax, and a module that knows a syntax does not compute a fact.**

- :mod:`.document` — the syntax-free document model (headings, tables, notes, code spans).
- :mod:`.markdown`, :mod:`.html` — the two backends. Each escapes its own metacharacters;
  neither decides what the report says.
- :mod:`.dataset_card` — builds a :class:`~eval_dataset_forge.schemas.DatasetCard` from a
  release's ``manifest.json`` and ``audit.json``, and lays it out as a ``Document``.

The alternative — a Markdown template and an HTML template — writes every number twice, and
the second copy is the one that goes stale. Here a card is rendered from one document, and a
test walks that document asserting every string it carries reaches both outputs.

Importing this package pulls in pydantic. The CLI imports it lazily, inside the command
handler, so ``evalforge check-env`` keeps working before the scientific stack is installed.
"""

from __future__ import annotations

from eval_dataset_forge.reports.dataset_card import (
    CARD_FILE,
    build_card,
    card_document,
    card_from_release,
    card_json,
    check_card,
    format_card,
    known_issues,
    load_card_meta,
    load_release,
    render_html,
    render_markdown,
    write_card,
)
from eval_dataset_forge.reports.document import (
    Bullets,
    Code,
    Document,
    Heading,
    Note,
    Paragraph,
    Table,
    document_strings,
    inline_spans,
)

__all__ = [
    "CARD_FILE",
    "Bullets",
    "Code",
    "Document",
    "Heading",
    "Note",
    "Paragraph",
    "Table",
    "build_card",
    "card_document",
    "card_from_release",
    "card_json",
    "check_card",
    "document_strings",
    "format_card",
    "inline_spans",
    "known_issues",
    "load_card_meta",
    "load_release",
    "render_html",
    "render_markdown",
    "write_card",
]
