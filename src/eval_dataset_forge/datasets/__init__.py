"""Dataset I/O for eval-dataset-forge.

At Milestone 2 this package holds only a minimal fixture/dataset loader
(:mod:`eval_dataset_forge.datasets.loader`). The full ``validate`` CLI (schema
enforcement, ID-uniqueness reporting, PII checks) lands in Milestone 3 and the
writer / manifest / versioning modules in Milestone 10; both build on the loader.
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

__all__ = [
    "FIXTURES_ROOT",
    "FixtureError",
    "discover_fixtures",
    "load_defects",
    "load_examples",
    "read_records",
]
