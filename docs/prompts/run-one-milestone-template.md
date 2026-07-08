Read CLAUDE.md and docs/PLAN.md. Inspect `git status`, `git log --oneline -5`,
docs/milestones/, and docs/review_notes/. Determine the next incomplete milestone
unless a specific milestone number is provided.

Implement exactly one milestone. Do not implement later milestones. Follow the
per-milestone deliverables in docs/PLAN.md §10 and the detailed note in
docs/milestones/ if present.

For that one milestone:

1. Implement the code and tests. Keep everything CPU-only and offline; do not add
   networked or heavy dependencies; do not use any paid API.
2. Run `python3 -m pytest -q` and the milestone's validation commands. They must
   pass.
3. Update the static website under site/ (including milestones.html status).
4. Update docs/RESULTS.md or relevant docs if results changed.
5. Create/update docs/milestones/NN-<slug>.md and
   docs/review_notes/NN-<slug>-human-review.md.
6. Verify no secrets, private data, caches, virtualenvs, large datasets, or
   cached API responses are staged.
7. Commit locally. Do not push.

Stop after exactly one milestone and print: milestone title, commit hash, tests
run, website pages updated, review placeholder path, and the next milestone.
