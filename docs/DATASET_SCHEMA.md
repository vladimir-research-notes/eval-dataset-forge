# Dataset schema

This document defines the on-disk formats. The authoritative implementation is
`src/eval_dataset_forge/schemas.py` (pydantic v2 models, landed in Milestone 1);
this document and that module must stay in sync. `SCHEMA_VERSION` in
`src/eval_dataset_forge/__init__.py` is bumped on any breaking change.

## Formats

- **Examples**: JSON Lines (`.jsonl`), one `DatasetExample` per line. YAML is
  accepted for hand-authored fixtures and normalized to JSONL on load.
- **Manifests, audits, cards**: JSON (`manifest.json`, `audit.json`) and Markdown
  (`audit.md`, `dataset_card.md`).

## DatasetExample

| field | type | required | notes |
|-------|------|----------|-------|
| `example_id` | str | yes | stable, unique within a dataset |
| `source_id` | str | yes | id of the originating record/file (provenance) |
| `task_family` | str | yes | `qa` \| `agent_task` \| `retrieval` \| `bio` \| `safety` \| … |
| `input` | str | yes | prompt / query / task statement |
| `expected_output` | str \| obj | yes | gold answer / label / relevance judgement |
| `metadata` | obj | no | free-form; e.g. retrieval docs, notes, source url |
| `labels` | obj | no | structured labels; may hold multiple annotator labels |
| `difficulty_tags` | list[str] | no | e.g. `easy`, `hard`, `ambiguous` |
| `split` | str | no | `train` \| `dev` \| `test` \| `unassigned` (default) |
| `provenance` | str | no | `synthetic` \| `adapted` \| … |
| `created_at` | str (ISO-8601) | no | creation timestamp |
| `version` | str | no | example content/schema version |

Notes:
- `example_id` uniqueness is an **exact** audit check (Milestone 3).
- `labels` supports a `annotations: [{annotator, value}]` shape so the label
  audit can detect inter-annotator disagreement (Milestone 6).
- `metadata.retrieval` (for `retrieval` family) may carry `docs: [{doc_id, text,
  relevant}]` used by retrieval-contamination checks (Milestone 5).

## DatasetManifest (one per release)

| field | type | notes |
|-------|------|-------|
| `dataset_id` | str | logical dataset name |
| `version` | str | semantic release version, e.g. `0.1.0` |
| `created_at` | str | ISO-8601 |
| `source_files` | list[str] | inputs used to build the release |
| `example_count` | int | number of examples |
| `schema_version` | str | matches `SCHEMA_VERSION` |
| `hashes` | obj | content hashes (sha256) of examples + split files |
| `splits` | obj | split name → count and file path |
| `audit_results` | obj | summary counts from the `AuditResult` |
| `known_limitations` | list[str] | carried into the dataset card |
| `license_notes` | str | licensing / usage notes |

## AuditResult

| field | type | notes |
|-------|------|-------|
| `audit_id` | str | unique id for this audit run |
| `dataset_id` | str | dataset audited |
| `version` | str | dataset version audited |
| `exact_duplicates` | list[DeduplicationFinding] | exact/normalized duplicates |
| `near_duplicates` | list[DeduplicationFinding] | heuristic near-duplicates |
| `leakage_flags` | list[LeakageFinding] | leakage/contamination findings |
| `label_warnings` | list[LabelAuditFinding] | label-quality findings |
| `pii_flags` | list[obj] | PII-like matches (redaction candidates) |
| `difficulty_distribution` | obj | bucket → count |
| `split_statistics` | obj | per-split sizes and distributions |
| `recommendations` | list[str] | prioritized, human-readable actions |

### Finding types

- **DeduplicationFinding** — `kind` (exact/normalized/near), `example_ids`,
  `similarity`, `method`.
- **LeakageFinding** — `kind` (train_test_overlap / answer_in_prompt /
  ngram_overlap / retrieval_contamination), `example_ids`, `score`, `evidence`.
- **LabelAuditFinding** — `kind` (missing / conflicting / ambiguous /
  inconsistent_format / needs_adjudication), `example_id`, `detail`.
- **SplitSpec** — `strategy`, `seed`, `ratios`, `group_key`, `constraints`.
- **DifficultyScore** — `example_id`, `score`, `bucket`, `components` (per-signal
  contributions), `method`.
- **BootstrapResult** — `metric`, `point_estimate`, `ci_low`, `ci_high`,
  `n_resamples`, `seed`.
- **DatasetCard** — structured card fields (purpose, construction, schema,
  splits, known issues, audit findings, limitations, recommended use, non-goals).

## Release directory

```text
datasets/releases/<dataset_id>_v<version>/
  examples.jsonl        # the data (validated)
  manifest.json         # DatasetManifest
  audit.json            # AuditResult (machine-readable)
  audit.md              # human-readable audit report
  dataset_card.md       # DatasetCard rendered to Markdown
  splits/
    train.jsonl
    dev.jsonl
    test.jsonl
  REPRODUCE.md          # exact commands to rebuild this release
```

All fields above are defined as pydantic models so that malformed data fails
loudly at load time rather than silently propagating into an audit.
