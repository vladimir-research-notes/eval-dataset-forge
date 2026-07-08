# Human review: Milestone 2 — Fixture datasets

TL;DR
Five synthetic fixture datasets (`qa_toy`, `agent_tasks_toy`, `retrieval_toy`,
`bio_toy`, `safety_toy`; 56 examples) landed under `datasets/fixtures/`, each with
deliberately planted defects, a machine-readable `defects.yaml` ground-truth file,
and a README. A minimal loader (`eval_dataset_forge.datasets.loader`), a
validation script (`scripts/validate_fixtures.py`), and a 72-test suite
(`tests/test_fixtures.py`) check both structural health and that the planted
defects are actually present. Full suite: 121 passed; ruff clean.

What changed
- Added `datasets/fixtures/{qa_toy,agent_tasks_toy,retrieval_toy,bio_toy,safety_toy}/`
  each with `examples.jsonl`, `defects.yaml`, `README.md`.
- Added `src/eval_dataset_forge/datasets/__init__.py` and `.../loader.py`
  (JSONL/YAML loading + schema validation + defects loading + fixture discovery).
- Added `scripts/validate_fixtures.py` and `tests/test_fixtures.py`.
- Updated `docs/DATASET_SCHEMA.md`, `docs/RESULTS.md`, and the website
  (`site/schema.html` new fixtures section, `site/milestones.html` status → done,
  `site/index.html` status banner + text).

What should be checked later
- Whether the planted-defect *categories* in `defects.yaml` line up with what the
  audit stages will actually emit (Milestone 4 dedup, 5 leakage, 6 label audit),
  so recall scoring in Milestone 12 is apples-to-apples. Retrieval uses
  family-specific categories (`duplicate_documents`, `paraphrased_documents`,
  `relevance_issues`) that no enum covers yet.
- Whether the `auto_detectable: false` set is drawn correctly — i.e. that the
  audits are not later expected to catch those cases.
- Fixture sizes: 10–14 examples each is enough to plant every defect once or
  twice; some checks (near-duplicate thresholds, bootstrap CIs) may want larger or
  additional fixtures later.
- The loader is intentionally minimal; confirm Milestone 3's `validate` CLI builds
  on it rather than diverging.

Technical risks
- Low. Data + a small loader + a script + tests; no network, no paid API, no
  runtime mutation of the repo. Generated artifacts are still gitignored; only the
  committed fixtures are added.
- The safety fixtures contain synthetic placeholder "unsafe" items only; a test
  enforces that no operational content is present. Worth a human eyeball anyway.

Commands to verify
```
python3 -m pytest -q
python3 scripts/validate_fixtures.py
ruff check .
```

Human feedback
<!-- Human will fill this later. -->
