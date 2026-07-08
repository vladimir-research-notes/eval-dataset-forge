# bio_toy — biomedical-style fixture (non-clinical)

A small, synthetic biomedical-style dataset (10 examples, `task_family: bio`) of
textbook lookup and entity-classification tasks with defects planted on purpose.

**Safety scope.** All content is non-clinical and non-patient-specific: generic
textbook biology with no medical advice, no dosing, and no patient data. See
[`docs/SAFETY.md`](../../../docs/SAFETY.md). Ground truth is in
[`defects.yaml`](defects.yaml).

## Planted defects

| defect | example(s) | notes |
|--------|------------|-------|
| exact duplicate | `bio-003` / `bio-006` | identical DNA-bases Q + A |
| near duplicate (paraphrase) | `bio-001` / `bio-002` | paraphrased insulin question |
| conflicting labels | `bio-004` | "drug" vs "chemical compound" for aspirin |
| ambiguous | `bio-007` | no reference range / context |
| missing label | `bio-008` | empty `expected_output` |
| mislabeled answer | `bio-009` | gold "23"; a somatic cell has 46 — **not auto-detectable** |
| train/test overlap | `bio-001`↔`bio-002`, `bio-003`↔`bio-006` | duplicates across splits |

`bio-004` carries a two-annotator `labels.annotations` list so inter-annotator
disagreement (Milestone 6) has a concrete example.
