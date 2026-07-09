# qa_toy — question-answering fixture

A small, synthetic QA dataset (14 examples, `task_family: qa`) with defects
planted on purpose so the audit stages have real problems to find. All content is
hand-authored general knowledge — no personal or clinical data.

Machine-readable ground truth is in [`defects.yaml`](defects.yaml); the audit
milestones are scored against it. Structural validity (schema, unique ids) is
checked by `scripts/validate_fixtures.py` and `tests/test_fixtures.py`.

## Planted defects

| defect | example(s) | notes |
|--------|------------|-------|
| exact duplicate | `qa-001` / `qa-002` | identical question + answer |
| near duplicate (paraphrase) | `qa-003` / `qa-004` | "Who wrote" vs "Who is the author of" |
| mislabeled answer | `qa-006` | gold is "Ag" (silver); gold symbol is "Au" — **not auto-detectable** |
| ambiguous | `qa-008`, `qa-009` | many valid answers / underspecified |
| missing label | `qa-010` | empty `expected_output` |
| conflicting labels | `qa-011` | annotators disagree (1945 vs 1944) |
| train/test overlap | `qa-001`↔`qa-002`, `qa-003`↔`qa-004` | duplicate/near-duplicate across splits |
| answer in prompt | `qa-007` | "Berlin" appears verbatim in the question |
| duplicate source record | `qa-003`, `qa-012` | share `source_id: src-shakespeare` |

## Splits

Pre-assigned `train` / `dev` / `test` so cross-split leakage is detectable before
the split builder (Milestone 7) runs. A `mislabeled` answer is included as a
documented limitation: a well-formed but factually wrong gold answer is generally
not detectable by the label-audit heuristics, and is marked
`auto_detectable: false` in `defects.yaml`.

## `system_scores.jsonl` (Milestone 9)

A committed, inspectable example of the per-example score file that
`evalforge stats --scores` consumes: three systems × 14 examples, one JSON record
per `(example, system)` pair.

**The scores are synthetic and describe no real system.** They come from the Rasch
simulator in `eval_dataset_forge.statistics.simulate` at seed 0, and exist so a
reader can see the input format without running a model. No conclusion about any
model may be drawn from them.

Regenerate with `python3 scripts/make_demo_scores.py` (deterministic and
byte-identical); `python3 scripts/make_demo_scores.py --check` and
`tests/test_statistics.py` fail if the file drifts from its generator.
