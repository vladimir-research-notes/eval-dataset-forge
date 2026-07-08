# agent_tasks_toy — agent/coding-task fixture

A small, synthetic agent-task dataset (12 examples, `task_family: agent_task`)
mixing coding and research tasks, with defects planted on purpose. All content is
hand-authored and generic — no personal or proprietary data.

Coding tasks use a dict `expected_output` of `{reference_solution,
success_criteria}`; research tasks use `{reference_answer}`. One coding task
(`agent-011`) deliberately uses a bare string instead, to plant an
inconsistent-format defect. Ground truth is in [`defects.yaml`](defects.yaml).

## Planted defects

| defect | example(s) | notes |
|--------|------------|-------|
| exact duplicate | `agent-005` / `agent-006` | identical research task + answer |
| near duplicate (paraphrase) | `agent-001` / `agent-002` | paraphrased fizzbuzz, same solution |
| leaked solution in prompt | `agent-004` | prompt quotes `return s[::-1]` |
| ambiguous scoring | `agent-009` | no success criteria |
| missing gold output | `agent-009` | empty `expected_output` |
| conflicting labels | `agent-010` | disagreement on the `is_prime(1)` boundary |
| inconsistent format | `agent-011` | bare string where peers use a dict rubric |
| train/test overlap | `agent-001`↔`agent-002`, `agent-005`↔`agent-006` | duplicates across splits |

`agent-007` (binary search) and `agent-008` (Dijkstra) are clean, harder tasks so
the difficulty signal (Milestone 8) has range.
