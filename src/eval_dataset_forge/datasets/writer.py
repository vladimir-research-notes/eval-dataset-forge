"""Deterministic serialization and content hashing for release artifacts (Milestone 10).

Two serializations live here and they are not the same function, which is the whole point
of the module:

*Presentation* (:func:`example_line`, :func:`render_jsonl`) is what lands in
``examples.jsonl`` and ``splits/*.jsonl``. It preserves the schema's field order because a
human reads these files, and ``example_id`` belongs on the left. It is byte-identical
across runs and platforms — LF endings, no trailing whitespace, ``ensure_ascii=False`` so a
UTF-8 corpus round-trips as itself rather than as ``\\uXXXX`` escapes.

*Canonicalization* (:func:`canonical_bytes`) is what gets hashed. It sorts keys, strips
insignificant whitespace, and never sees a file. Two records that differ only in key order
canonicalize to the same bytes; the files that hold them do not. Hashing the presentation
form would make a dataset's identity depend on how its writer felt about field order — the
defect this module exists to prevent.

Both are pure functions of the data. Nothing here touches the clock, the filesystem, or the
environment, so a hash computed in memory and a hash computed over a file agree by
construction rather than by convention.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eval_dataset_forge.schemas import DatasetExample

# Digests are stored self-describing so a manifest never has to be read alongside a
# separate note saying which algorithm produced its hex strings.
HASH_PREFIX = "sha256:"


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def canonical_bytes(value: Any) -> bytes:
    """Serialize ``value`` to the canonical form used for every content hash.

    Sorted keys, no insignificant whitespace, UTF-8. ``ensure_ascii=False`` keeps the
    bytes a faithful encoding of the text rather than of its escape sequence, so the
    digest of a record does not change when a writer decides to escape non-ASCII.

    Only JSON-native values are accepted (the models are dumped with ``mode="json"``
    before they reach here), so this never has to guess how to encode a ``datetime``.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """``sha256:<hex>`` of ``data``."""
    return HASH_PREFIX + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """``sha256:<hex>`` of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return HASH_PREFIX + digest.hexdigest()


# --------------------------------------------------------------------------- #
# Presentation: JSONL and JSON artifacts
# --------------------------------------------------------------------------- #


def example_line(example: DatasetExample) -> str:
    """One JSONL line for ``example``, in schema field order, LF-terminated."""
    return json.dumps(example.model_dump(mode="json"), ensure_ascii=False) + "\n"


def render_jsonl(examples: Iterable[DatasetExample]) -> bytes:
    """The exact bytes of an ``examples.jsonl``. Input order is preserved."""
    return "".join(example_line(example) for example in examples).encode("utf-8")


def render_json(value: Any) -> bytes:
    """The exact bytes of a pretty-printed JSON artifact (``manifest.json``, ``audit.json``).

    Indented and *not* key-sorted: these files are read by people, and the pydantic models
    already declare a field order chosen for reading. Their bytes are still deterministic,
    because ``model_dump`` emits fields in declaration order.
    """
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_bytes(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to ``path``, creating parent directories. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_examples(examples: Iterable[DatasetExample], path: str | Path) -> Path:
    """Write ``examples`` as JSONL. Byte-identical across runs for a fixed input order."""
    return write_bytes(path, render_jsonl(examples))


__all__ = [
    "HASH_PREFIX",
    "canonical_bytes",
    "example_line",
    "render_json",
    "render_jsonl",
    "sha256_bytes",
    "sha256_file",
    "write_bytes",
    "write_examples",
]
