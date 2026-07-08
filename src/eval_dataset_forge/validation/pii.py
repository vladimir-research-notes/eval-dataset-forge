"""Heuristic PII-like and secret-like string detection (Milestone 3).

The detectors here answer one narrow question: *does a span of text look like a
personal identifier or a credential?* They are deliberately pattern- and
entropy-based, not a machine-learned NER model, so they are cheap, deterministic,
and offline — and, like every heuristic in this project, they miss things and
raise false positives. The trade-offs are documented per detector below and in
``docs/VALIDATION.md``.

Two severity classes are distinguished so the caller can decide what blocks a
release (see :mod:`eval_dataset_forge.validation`):

* **secret** — an unambiguous credential shape (an AWS-style access key, a PEM
  private-key header, a ``password = ...`` assignment). A benchmark should never
  ship one, so these are treated as errors by default.
* **pii** — a personal-identifier shape (email, phone-like number, IPv4) or a
  bare high-entropy token. These are ambiguous enough to be warnings by default
  and are only promoted to errors under an explicit ``--fail-on-pii`` policy.

Nothing here mutates data; redaction lives in :mod:`.redaction`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Span type + severity vocabulary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PiiSpan:
    """A single detected span within one string.

    ``start``/``end`` are Python slice indices into the scanned text and ``text``
    is the raw matched substring. ``text`` is retained so :mod:`.redaction` can
    replace exactly this span; callers that surface findings to a human should
    redact it first (see :func:`.redaction.redacted_context`) rather than print
    it raw.
    """

    kind: str
    start: int
    end: int
    text: str


# Unambiguous credential shapes: never belong in a public benchmark.
SECRET_KINDS = frozenset({"aws_access_key", "private_key", "secret_assignment"})

# Personal-identifier / low-confidence shapes: reported, not fatal by default.
PII_KINDS = frozenset({"email", "phone", "ipv4", "token"})

# All placeholder-able kinds, for validation of the redaction table.
ALL_KINDS = SECRET_KINDS | PII_KINDS


# --------------------------------------------------------------------------- #
# Named patterns (high precision)
# --------------------------------------------------------------------------- #

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Dotted quad with per-octet range validation so 999.1.1.1 is not matched.
_IPV4_RE = re.compile(
    r"(?<![\w.])"
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?![\w.])"
)

# Classic AWS access-key id shape (AKIA/ASIA + 16 upper-alphanumerics).
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")

# PEM private-key header — an exact, unambiguous credential marker.
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")

# ``keyword = value`` / ``keyword: value`` assignments. The value (group 1) is
# post-filtered by ``_plausible_secret_value`` to avoid matching prose like
# ``password: required``.
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token"
    r"|client[_-]?secret|password|passwd)\b\s*[:=]\s*"
    r"['\"]?([A-Za-z0-9!@#$%^&*()/+_\-.]{6,})['\"]?"
)

# Phone-like run: an optional country code then 8+ separator/digit chars. The
# digit count is range-checked afterwards to cut down on false positives.
_PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().\-]{8,}\d(?![\w.])")

# Candidate for a bare high-entropy token (API-key-like string with no keyword).
_TOKEN_CANDIDATE_RE = re.compile(r"(?<![\w/+])[A-Za-z0-9+/_\-]{20,}(?![\w/+])")

# Common non-secret words that syntactically fit ``keyword: value``.
_SECRET_VALUE_STOPWORDS = frozenset(
    {
        "required",
        "optional",
        "true",
        "false",
        "none",
        "null",
        "yes",
        "no",
        "string",
        "password",
        "passwd",
        "secret",
        "token",
        "changeme",
        "example",
        "redacted",
    }
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def shannon_entropy(text: str) -> float:
    """Shannon entropy of ``text`` in bits per character (0.0 for empty)."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _valid_phone(candidate: str) -> bool:
    """A phone match is kept only if it carries 10–15 digits (E.164-ish range)."""
    digits = sum(ch.isdigit() for ch in candidate)
    return 10 <= digits <= 15


def _looks_like_secret_token(token: str) -> bool:
    """Heuristic for a bare credential-looking token.

    Requires length ≥ 20 and either three of {lowercase, uppercase, digit} with
    entropy ≥ 3.5 bits/char, or a very long, very high-entropy string. This is
    tuned to fire on API-key-like values while ignoring hex hashes, English
    prose, and dotted/underscored identifiers (which rarely mix three classes at
    high entropy). It is the noisiest detector; see ``docs/VALIDATION.md``.
    """
    if len(token) < 20:
        return False
    classes = sum(
        (
            any(c.islower() for c in token),
            any(c.isupper() for c in token),
            any(c.isdigit() for c in token),
        )
    )
    entropy = shannon_entropy(token)
    if classes >= 3 and entropy >= 3.5:
        return True
    return len(token) >= 32 and entropy >= 4.2


def _plausible_secret_value(value: str) -> bool:
    """Filter ``keyword: value`` hits so prose values do not read as secrets."""
    if len(value) < 6 or value.lower() in _SECRET_VALUE_STOPWORDS:
        return False
    has_alpha = any(c.isalpha() for c in value)
    has_digit = any(c.isdigit() for c in value)
    return has_alpha and (has_digit or len(value) >= 12)


def _dedupe_overlaps(spans: list[PiiSpan]) -> list[PiiSpan]:
    """Sort by position and drop spans overlapping an already-accepted one.

    Earlier start wins; on a tie the longer span wins. Because named-pattern
    spans are collected before token candidates, a precise match (e.g. an AWS
    key) suppresses the generic token candidate covering the same characters.
    """
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[PiiSpan] = []
    last_end = -1
    for span in ordered:
        if span.start >= last_end:
            kept.append(span)
            last_end = span.end
    return kept


# --------------------------------------------------------------------------- #
# Public scan
# --------------------------------------------------------------------------- #


def scan(text: str) -> list[PiiSpan]:
    """Return non-overlapping PII/secret spans in ``text``, left to right.

    Deterministic: the same input always yields the same spans. An empty or
    non-string input yields ``[]``.
    """
    if not text or not isinstance(text, str):
        return []

    spans: list[PiiSpan] = []
    for m in _EMAIL_RE.finditer(text):
        spans.append(PiiSpan("email", m.start(), m.end(), m.group()))
    for m in _IPV4_RE.finditer(text):
        spans.append(PiiSpan("ipv4", m.start(), m.end(), m.group()))
    for m in _AWS_KEY_RE.finditer(text):
        spans.append(PiiSpan("aws_access_key", m.start(), m.end(), m.group()))
    for m in _PRIVATE_KEY_RE.finditer(text):
        spans.append(PiiSpan("private_key", m.start(), m.end(), m.group()))
    for m in _SECRET_ASSIGN_RE.finditer(text):
        if _plausible_secret_value(m.group(1)):
            spans.append(PiiSpan("secret_assignment", m.start(1), m.end(1), m.group(1)))
    for m in _PHONE_RE.finditer(text):
        if _valid_phone(m.group()):
            spans.append(PiiSpan("phone", m.start(), m.end(), m.group()))
    for m in _TOKEN_CANDIDATE_RE.finditer(text):
        if _looks_like_secret_token(m.group()):
            spans.append(PiiSpan("token", m.start(), m.end(), m.group()))

    return _dedupe_overlaps(spans)


# --------------------------------------------------------------------------- #
# Example-level scanning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PiiFinding:
    """A PII/secret hit located on one example.

    ``field`` is a dotted path into the example (``input``,
    ``metadata.retrieval.docs[0].text`` …). ``snippet`` is a *redacted* window of
    surrounding text — safe to print — so the raw value never leaks into a report
    or log. ``is_secret`` flags the unambiguous-credential class.
    """

    example_id: str | None
    field: str
    kind: str
    snippet: str

    @property
    def is_secret(self) -> bool:
        return self.kind in SECRET_KINDS


def _walk_text(path: str, node: object):
    """Yield ``(dotted_path, string)`` for every string leaf under ``node``."""
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_text(f"{path}.{key}", value)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_text(f"{path}[{i}]", value)


def iter_text_fields(example) -> list[tuple[str, str]]:
    """Return ``(field_path, text)`` pairs for the scannable fields of an example.

    Covers ``input``, ``expected_output`` (when textual), and every string leaf
    inside ``metadata`` and ``labels``. Identity/timestamp fields are excluded:
    they are structural, not free text.
    """
    out: list[tuple[str, str]] = []
    out.extend(_walk_text("input", example.input))
    out.extend(_walk_text("expected_output", example.expected_output))
    out.extend(_walk_text("metadata", example.metadata))
    out.extend(_walk_text("labels", example.labels))
    return out


def scan_example(example) -> list[PiiFinding]:
    """Scan every text field of one :class:`DatasetExample` for PII/secrets."""
    from eval_dataset_forge.validation.redaction import redacted_context

    findings: list[PiiFinding] = []
    for field, text in iter_text_fields(example):
        for span in scan(text):
            findings.append(
                PiiFinding(
                    example_id=example.example_id,
                    field=field,
                    kind=span.kind,
                    snippet=redacted_context(text, span),
                )
            )
    return findings


def scan_examples(examples) -> list[PiiFinding]:
    """Scan a sequence of examples; flat list of findings in example order."""
    findings: list[PiiFinding] = []
    for example in examples:
        findings.extend(scan_example(example))
    return findings


__all__ = [
    "PiiSpan",
    "PiiFinding",
    "SECRET_KINDS",
    "PII_KINDS",
    "ALL_KINDS",
    "shannon_entropy",
    "scan",
    "iter_text_fields",
    "scan_example",
    "scan_examples",
]
