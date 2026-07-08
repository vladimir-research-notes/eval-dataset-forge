# Initial build prompt for Claude Code

You are working in the `eval-dataset-forge` repository.

Read:

- `CLAUDE.md`
- `docs/PLAN.md`

Then execute the plan milestone by milestone (see `docs/PLAN.md` §10).

## Important instructions

- Do not push. Commit locally after each coherent milestone.
- Do not wait for human review. Create human-review placeholders, then continue.
- Keep all default tests CPU-only and fully offline. No paid API calls, ever, in
  this version.
- Keep the dependency set small: hard deps are `pydantic`, `numpy`, `pyyaml`;
  `pandas`/`scikit-learn` are optional with pure-Python fallbacks.
- Do not commit secrets, credentials, `.claude/settings.local.json`,
  virtualenvs, caches, `node_modules`, model weights, large datasets, or cached
  API responses.
- Use only synthetic / public-safe fixtures. No real personal or clinical data.
- Update the multi-page website at every milestone.
- Keep documentation professional and precise; avoid learning-diary tone; be
  explicit about limitations and non-goals.

Scaffold (Milestone 0) already exists. Start with the next incomplete milestone.

After each milestone, print: milestone title, commit hash, tests run, website
pages updated, review placeholder path, next milestone.

Continue automatically unless a blocker would risk secrets, paid API spend, data
loss, or repository corruption.
