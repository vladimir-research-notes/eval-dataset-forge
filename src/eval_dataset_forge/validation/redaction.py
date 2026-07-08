"""Deterministic redaction of detected PII/secret spans (Milestone 3).

Redaction replaces each span found by :mod:`.pii` with a typed placeholder
(``<EMAIL>``, ``<TOKEN>``, …). It is deterministic and reversible only in the
sense that the *kind* of the removed value is preserved; the value itself is
gone. Two guarantees the rest of the tool relies on:

* replacing spans right-to-left keeps earlier indices valid, so the output is
  independent of how many spans are redacted;
* :func:`redacted_context` produces a short, human-readable snippet in which the
  raw value is already replaced, so findings can be surfaced without leaking the
  secret they flag.

This is a sanitizer for *this* project's own fixtures and demos, not a general
DLP guarantee — it can only remove what :mod:`.pii` detects.
"""

from __future__ import annotations

from eval_dataset_forge.validation.pii import ALL_KINDS, PiiSpan, scan

# Typed placeholder per detected kind. Kept in sync with pii.ALL_KINDS by a test.
PLACEHOLDERS: dict[str, str] = {
    "email": "<EMAIL>",
    "phone": "<PHONE>",
    "ipv4": "<IP>",
    "aws_access_key": "<AWS_KEY>",
    "private_key": "<PRIVATE_KEY>",
    "secret_assignment": "<SECRET>",
    "token": "<TOKEN>",
}

# Fallback for any span kind without an explicit placeholder.
_DEFAULT_PLACEHOLDER = "<REDACTED>"

assert ALL_KINDS <= set(PLACEHOLDERS), "PLACEHOLDERS is missing a pii kind"


def placeholder_for(kind: str) -> str:
    """Return the typed placeholder string for a span kind."""
    return PLACEHOLDERS.get(kind, _DEFAULT_PLACEHOLDER)


def redact(text: str, spans: list[PiiSpan] | None = None) -> str:
    """Return ``text`` with every detected PII/secret span replaced.

    When ``spans`` is ``None`` the text is scanned first. Replacement is applied
    from the rightmost span leftward so span indices stay valid throughout.
    """
    if not isinstance(text, str) or not text:
        return text
    if spans is None:
        spans = scan(text)
    out = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        out = out[: span.start] + placeholder_for(span.kind) + out[span.end :]
    return out


def redacted_context(text: str, span: PiiSpan, width: int = 12) -> str:
    """Return a redacted window around ``span``, safe to print in a report.

    The window is ``width`` characters either side of the span; the span itself
    is replaced with its placeholder. Ellipses mark truncation. The raw matched
    value never appears in the result.
    """
    lo = max(0, span.start - width)
    hi = min(len(text), span.end + width)
    window = text[lo:hi]
    local = PiiSpan(span.kind, span.start - lo, span.end - lo, span.text)
    body = redact(window, [local]).replace("\n", " ")
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(text) else ""
    return f"{prefix}{body}{suffix}"


def _redact_tree(node):
    """Recursively redact every string leaf in a metadata/labels tree."""
    if isinstance(node, str):
        return redact(node)
    if isinstance(node, dict):
        return {k: _redact_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact_tree(v) for v in node]
    return node


def redact_example(example):
    """Return a copy of ``example`` with PII/secrets redacted in its text fields.

    Redacts ``input``, a textual ``expected_output``, and every string leaf under
    ``metadata`` and ``labels``. Identity/timestamp fields are left untouched. The
    original object is not mutated.
    """
    from eval_dataset_forge.schemas import DatasetExample

    data = example.model_dump()
    data["input"] = redact(data["input"])
    if isinstance(data["expected_output"], (str, dict, list)):
        data["expected_output"] = _redact_tree(data["expected_output"])
    data["metadata"] = _redact_tree(data["metadata"])
    data["labels"] = _redact_tree(data["labels"])
    return DatasetExample(**data)


__all__ = [
    "PLACEHOLDERS",
    "placeholder_for",
    "redact",
    "redacted_context",
    "redact_example",
]
