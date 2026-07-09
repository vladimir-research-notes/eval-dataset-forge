"""Dataset I/O, identity, and release packaging.

The loader (Milestone 2) reads schema-validated examples. The writer, versioning, and
manifest modules (Milestone 10) turn a set of examples into a **versioned release**: a
deterministic serialization, two computed identities (``content_id`` for *"the same
examples?"*, ``eval_id`` for *"comparable scores?"*), a hashed inventory, and a verifier
that checks a release against its own manifest without rebuilding it.

Orchestration lives in :mod:`eval_dataset_forge.datasets.release` and is **not** re-exported
here. It imports every audit stage, and ``from eval_dataset_forge.datasets import loader``
— which the CLI does for its cheap subcommands — must not drag the deduplication ladder,
the leakage checks, and numpy into the process to read a JSONL file.
"""

from __future__ import annotations

from eval_dataset_forge.datasets.loader import (
    FIXTURES_ROOT,
    FixtureError,
    discover_fixtures,
    load_defects,
    load_examples,
    read_records,
)
from eval_dataset_forge.datasets.manifest import (
    VerifyReport,
    build_manifest,
    format_verify,
    load_manifest,
    verify_release,
)
from eval_dataset_forge.datasets.versioning import (
    ReleaseDelta,
    compare,
    content_id,
    eval_id,
    example_digest,
    format_delta,
    next_version,
    parse_version,
)
from eval_dataset_forge.datasets.writer import (
    canonical_bytes,
    render_jsonl,
    sha256_bytes,
    sha256_file,
    write_examples,
)

__all__ = [
    # loader
    "FIXTURES_ROOT",
    "FixtureError",
    "discover_fixtures",
    "load_defects",
    "load_examples",
    "read_records",
    # writer
    "canonical_bytes",
    "render_jsonl",
    "sha256_bytes",
    "sha256_file",
    "write_examples",
    # versioning
    "ReleaseDelta",
    "compare",
    "content_id",
    "eval_id",
    "example_digest",
    "format_delta",
    "next_version",
    "parse_version",
    # manifest
    "VerifyReport",
    "build_manifest",
    "format_verify",
    "load_manifest",
    "verify_release",
]
