# retrieval_toy — retrieval fixture

A small, synthetic retrieval dataset (10 examples, `task_family: retrieval`) with
defects planted on purpose. Each example is a query plus a set of candidate
documents under `metadata.retrieval.docs` (`{doc_id, text, relevant}`);
`expected_output` is the list of relevant `doc_id`s. All content is hand-authored
general knowledge. Ground truth is in [`defects.yaml`](defects.yaml).

## Planted defects

| defect | example(s) | notes |
|--------|------------|-------|
| near-duplicate query | `ret-001` / `ret-002` | paraphrased query |
| duplicated document | `ret-001` `d1` ≡ `ret-002` `d3` | identical doc text across examples |
| paraphrased documents | `ret-006` `d13`/`d14` | two relevant docs, same fact |
| retrieval contamination | `ret-003` `d5` | doc carries label artifact `[ANSWER: Mars]` |
| incomplete relevance labels | `ret-004` | relevant doc `d9` omitted from `expected_output` |
| over-specific / unanswerable query | `ret-005` | no candidate doc can answer it |
| duplicate source record | `ret-006`, `ret-009` | share `source_id: src-boiling` |
| train/test overlap | `ret-001`↔`ret-002` | near-duplicate query across splits |

`ret-007`, `ret-008`, `ret-010` are clean. The over-specific query and the
label-artifact contamination are heuristic-detectable signals — they are proxies,
not proof, of the underlying problem, and are documented as such on the
[Leakage](../../../site/leakage.html) page.
