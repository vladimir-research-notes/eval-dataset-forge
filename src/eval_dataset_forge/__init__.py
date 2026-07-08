"""eval-dataset-forge: a CPU-first framework for auditing evaluation datasets.

Raw examples go in; audited, versioned benchmark releases come out — with schema
validation, deduplication, leakage/contamination checks, label audits, difficulty
calibration, split construction, bootstrap uncertainty estimates, dataset cards,
and reproducible release manifests.

The public surface is intentionally small at scaffold stage: subsystems land
milestone by milestone (see ``docs/PLAN.md``).
"""

__version__ = "0.0.0"

# Schema version for dataset examples / manifests / audit results. Bumped when a
# breaking change is made to the on-disk formats defined in docs/DATASET_SCHEMA.md.
SCHEMA_VERSION = "0.1.0"

__all__ = ["__version__", "SCHEMA_VERSION"]
