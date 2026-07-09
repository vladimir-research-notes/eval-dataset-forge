# Dataset schema

This document defines the on-disk formats. The authoritative implementation is
`src/eval_dataset_forge/schemas.py` (pydantic v2 models, landed in Milestone 1);
this document and that module must stay in sync. `SCHEMA_VERSION` in
`src/eval_dataset_forge/__init__.py` is bumped on any breaking change.

**Status (Milestone 1).** All eleven models below are implemented and tested
(`tests/test_schemas.py`). What the models enforce:

- **Unknown fields are rejected** (`extra="forbid"`). A misspelled key such as
  `expcted_output` fails at load time rather than being silently dropped;
  free-form extension belongs in the explicit `metadata` / `labels` dicts.
- **String-valued enums.** `split`, and the finding `kind` fields, are
  `StrEnum`s, so `model_dump(mode="json")` produces plain JSONL with no custom
  encoder and reloads exactly.
- **Identity fields are non-empty** (`example_id`, `source_id`, `task_family`,
  `dataset_id`, `version`, `audit_id`, `metric`).
- **Cross-field validators** reject structurally impossible records: split
  `ratios` (when present) must be non-negative and sum to 1.0; a
  `BootstrapResult` must satisfy `ci_low ≤ ci_high` with `n_resamples ≥ 1`; a
  `DatasetRelease` must have `manifest.example_count == len(examples)` once
  examples are attached.

These are **structural** checks. Semantic defects (a wrong gold answer, a leaked
answer, a label conflict) are the job of the later audit stages, not the schema.
`task_family` is deliberately an open string vocabulary; the validation layer
(Milestone 3) may *warn* on families outside `KNOWN_TASK_FAMILIES` but the schema
does not reject them.

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
| `created_at` | str | ISO-8601. **Provenance, not identity**: enters no hash and neither id |
| `source_files` | list[str] | inputs used to build the release |
| `example_count` | int | number of examples |
| `schema_version` | str | matches `SCHEMA_VERSION` |
| `hashes` | obj | `sha256:<hex>` per release file. `manifest.json` is absent — a file cannot hash itself |
| `splits` | obj | split name → `{count, path, sha256}` |
| `audit_results` | obj | summary counts from the `AuditResult` |
| `known_limitations` | list[str] | carried into the dataset card |
| `license_notes` | str | licensing / usage notes |
| `content_id` | str | **Milestone 10.** Order-independent digest of the examples, `split` excluded: *"are these the same examples?"* |
| `eval_id` | str | **Milestone 10.** The same digest with each example's split folded in: *"are two reported scores comparable?"* |
| `source_content_id` | str | **Milestone 10.** `content_id` of the corpus **before** `excluded` was applied |
| `tool_version` | str | **Milestone 10.** the `evalforge` version that built the release |
| `split_spec` | obj \| null | **Milestone 10.** the `SplitSpec` the splits were built from (strategy, seed, ratios, constraints) |
| `excluded` | list[obj] | **Milestone 10.** `{example_id, reason, kept_id, method}` per removed example |
| `dependence` | obj | **Milestone 10.** `{block_count, block_size_histogram, duplicate_block_count, near_method}`, plus **Milestone 11**'s `{blocks_by_split, straddling_block_count}` |
| `card_meta` | obj | **Milestone 11.** the `CardMeta` prose the dataset card renders |

The Milestone 10 and 11 fields are **additive with defaults**, so `SCHEMA_VERSION` did not move
and a pre-existing manifest still loads.

`dependence` deliberately carries **no `effective_n`**. Effective sample size is
`n / (1 + (m̄ − 1)·ICC)` and the ICC is a property of a *scored comparison*, not of a dataset —
it needs system scores. The block structure is what a reader needs to compute it for their own
systems. See [`RELEASES.md`](RELEASES.md) §4 and [`STATISTICS.md`](STATISTICS.md).

`blocks_by_split` counts the independent blocks each split *touches*, because a corpus-wide
block count does not bound a claim about a **test** score — the test split's own block count
does. A block straddling a boundary is counted in every split it reaches, so the per-split
counts sum to at least `block_count`, and `straddling_block_count` says by how much.

## CardMeta (Milestone 11)

The parts of a dataset card no audit can compute: claims about intent. Supplied via
`release --card-meta <file.yaml>`, validated (unknown fields rejected), and stored in
`manifest.card_meta` so a card regenerated from a release directory alone reproduces the card
that shipped in it.

| field | type | notes |
|-------|------|-------|
| `purpose` | str | what the dataset is for |
| `construction` | str | how it was built |
| `recommended_use` | str | what a reader may reasonably do with it |
| `non_goals` | list[str] | what it must not be used for |

A blank section is **named** in the rendered card, never quietly omitted: a reader must be able
to tell *"no restrictions"* from *"nobody wrote any down"*. `dataset-card --fail-on-incomplete`
gates on it. `card_meta` is a *packaging* field for versioning — rewriting a non-goal changes no
number, so it forces a `patch` and never a `minor`.

## DatasetCard (Milestone 11)

Built by `build_card(manifest, audit)` from exactly those two artifacts; nothing is recomputed.
Rendered to `dataset_card.md` in the release, and optionally to a standalone HTML page.

| field | type | notes |
|-------|------|-------|
| `dataset_id`, `version` | str | must match the manifest **and** the audit, or the build refuses |
| `purpose`, `construction`, `recommended_use`, `non_goals` | — | from `manifest.card_meta` |
| `schema_summary` | str | generated: fields, schema version, task families present |
| `splits` | obj | `{total_examples, by_split{examples, blocks, share}, blocks{…}, strategy}` |
| `known_issues` | list[str] | generated from the audit's non-zero findings, with example ids |
| `audit_findings` | obj | counts per dimension, difficulty distribution, recommendations, certainty |
| `limitations` | list[str] | `manifest.known_limitations`, verbatim |
| `identity` | obj | `content_id`, `eval_id`, `source_content_id`, `schema_version`, `audit_id` |
| `provenance` | obj | source files, exclusions, tool version, license, near-duplicate method |

A card prints **no file hashes** and **no timestamp**, and both absences are arithmetic rather
than taste: `dataset_card.md` is itself hashed into `manifest.hashes`, so its own digest cannot
appear inside it, and a clock in a hashed artifact would make `versioning.py`'s `none` bump
unreachable. See [`DATASET_CARD.md`](DATASET_CARD.md).

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
  `n_resamples`, `seed`, plus `system`, `method`, `resample_unit` (added in
  Milestone 9; additive, with defaults, so records written before it still load).
  The last two are load-bearing rather than decorative: a percentile interval and a
  BCa interval over the same resamples are different intervals, and an interval that
  resampled *examples* assumes an independence that one resampling *duplicate
  blocks* does not. A stored result omitting them could not be compared with
  another. `method` records what actually ran — a BCa request that hit a degenerate
  case is written `"bca->percentile"`. See [`STATISTICS.md`](STATISTICS.md).
- **DatasetCard** — structured card fields (purpose, construction, schema,
  splits, known issues, audit findings, limitations, recommended use, non-goals).

### System scores (Milestone 9)

Not part of a release; an *input* to `evalforge stats`. JSONL, one record per line,
in either shape (they may be mixed):

```json
{"example_id": "qa-001", "system": "sys_a", "score": 1.0}
{"example_id": "qa-001", "scores": {"sys_a": 1.0, "sys_b": 0}}
```

Alignment against the dataset is strict in both directions — a score for an unknown
example, or an example no system scored, raises rather than being imputed.
`datasets/fixtures/qa_toy/system_scores.jsonl` is a committed, **synthetic** example
of the format.

## Release directory

```text
datasets/releases/<dataset_id>_v<version>/
  examples.jsonl        # the data (validated), in source order
  manifest.json         # DatasetManifest — the only artifact nothing hashes
  audit.json            # AuditResult, computed over the data that shipped
  audit.md              # human-readable audit report          (Milestone 12 — not written yet)
  dataset_card.md       # DatasetCard rendered to Markdown, from manifest.json + audit.json
  splits/
    train.jsonl
    dev.jsonl
    test.jsonl
  REPRODUCE.md          # the command, the identities, and the gaps
```

Written by `evalforge release` (Milestone 10) and checked by `evalforge verify-release`, which
recomputes every file hash and both identities from the release's own contents. That is an
**integrity** check, not an authenticity one: `manifest.json` cannot hash itself, so anyone able
to rewrite the data can rewrite the digests describing it.

Two builds at a fixed `--created-at` are byte-identical in every artifact; two builds at
different clocks differ in `manifest.json` and nothing else.

All fields above are defined as pydantic models so that malformed data fails
loudly at load time rather than silently propagating into an audit.

## Fixture datasets (Milestone 2)

Five synthetic fixtures under `datasets/fixtures/` are instances of this schema
with defects planted on purpose (they are structurally valid but semantically
flawed):

- `qa_toy`, `agent_tasks_toy`, `retrieval_toy`, `bio_toy`, `safety_toy`
  (56 examples total).

Each fixture directory contains:

- `examples.jsonl` — the examples (one `DatasetExample` per line).
- `defects.yaml` — **machine-readable ground truth**: the defects deliberately
  planted, keyed by category (`exact_duplicates`, `near_duplicates`, `leakage`,
  `label_issues`, `duplicate_sources`, and family-specific categories). This lets
  the audit stages be scored for *recall* against known defects, not merely
  “it found something”. Defects no heuristic in this project can catch — e.g. a
  well-formed but factually wrong gold answer — are marked
  `auto_detectable: false` and excluded from recall.
- `README.md` — a human-readable table of the planted defects.

The loader (`eval_dataset_forge.datasets.loader`) reads and schema-validates
fixtures; `scripts/validate_fixtures.py` checks structural health (schema, unique
ids, known `task_family`, valid ground-truth references) and
`tests/test_fixtures.py` additionally verifies the ground truth matches the data.
`.jsonl` is the committed format; the loader also accepts a YAML list for
hand-authored inputs.
