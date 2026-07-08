# Human review: Milestone 0 — Scaffold & project contract

TL;DR
Repository scaffold, full build plan (`docs/PLAN.md`), operating contract
(`CLAUDE.md`), CPU-only test suite, multi-page website, and the milestone-runner
script are in place. Milestones 1–15 are launched by the user via
`scripts/run_all_milestones_with_claude.sh`.

What changed
Created packaging, package skeleton with a working CLI stub, docs set, static
website, environment check, milestone runner + prompt templates, and first tests.

What should be checked later
- That the dependency strategy (hard: pydantic/numpy/pyyaml; optional:
  pandas/scikit-learn) still holds once schemas and dedup land.
- That the website stays in sync with implemented behavior each milestone.
- Port assignment 9004 (see ~/claude/ports.md).

Technical risks
- PEP-668 environment requires `--break-system-packages` for installs; documented
  but worth confirming on a fresh instance.
- Scaffold subcommands are stubs that exit 2; they must be wired to real handlers
  by their milestones.

Commands to verify
```
python3 -m pytest -q
python3 scripts/00_check_environment.py
PYTHONPATH=src python3 -m eval_dataset_forge info
```

Human feedback
<!-- Human will fill this later. -->
