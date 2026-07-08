# Label audit & adjudication

> Status: design (implemented in Milestone 6).

Label noise puts a hard ceiling on any accuracy a benchmark can measure, and
biases comparisons between systems. The label audit surfaces defects; it does not
"fix" labels automatically — ambiguous cases are routed to a human adjudication
queue.

## Checks

1. **Missing labels** — `expected_output` (or a required `labels` field) is
   absent or empty.
2. **Conflicting labels** — multiple annotator labels disagree
   (`labels.annotations`), or two examples with identical inputs carry different
   expected outputs.
3. **Ambiguous expected outputs** — the example is tagged `ambiguous`, or the
   gold answer is under-specified relative to the question (heuristic: very short
   answer to an open question, multiple plausible answers listed, etc.).
4. **Inconsistent label formats** — within a `task_family`, expected outputs mix
   incompatible shapes (e.g. some booleans, some free text; inconsistent
   casing/units for categorical labels).
5. **Needs adjudication** — any example flagged by the above that a human must
   resolve.

## Inter-annotator disagreement

When examples carry multiple annotator labels, we compute simple, transparent
agreement summaries (raw agreement rate, and per-example disagreement flags).
More elaborate agreement coefficients (e.g. Cohen's/Fleiss' kappa) are a
candidate future addition; we prefer to start with quantities that are hard to
misread.

## Adjudication queue export

Flagged examples are exported to an `adjudication_queue.jsonl` with the reason,
the conflicting values, and a stable `example_id`, so a human (or a future
LLM-assisted adjudicator, disabled by default) can resolve them and feed
decisions back. The queue is deterministic and diff-friendly.

## Known limitations

- Ambiguity detection is heuristic; it will both miss subtle ambiguity and
  over-flag terse-but-correct answers. Every flag is advisory.
- The audit never overwrites a label. It records findings and queues work.
- Semantic conflict detection (answers that differ in wording but agree in
  meaning) is out of scope for the offline build.
