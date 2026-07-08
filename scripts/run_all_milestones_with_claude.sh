#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run the eval-dataset-forge milestone plan with Claude Code, one milestone per
# `claude -p` invocation. Between milestones the script enforces: a new commit
# was created, the working tree is clean, and the test suite passes. It stops on
# the first failure so you can inspect.
#
# Usage:
#   ./scripts/run_all_milestones_with_claude.sh [START] [END]
#     START  first milestone to run (default 1 — Milestone 0 is the committed scaffold)
#     END    last milestone to run  (default 15)
#
#   # run milestones 1..15:
#   ./scripts/run_all_milestones_with_claude.sh
#   # resume from milestone 6:
#   ./scripts/run_all_milestones_with_claude.sh 6 15
#   # single milestone:
#   ./scripts/run_all_milestones_with_claude.sh 4 4
# ---------------------------------------------------------------------------

START="${1:-1}"
END="${2:-15}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TEMPLATE="docs/prompts/run-one-milestone-template.md"

# ---------------------------------------------------------------------------
# Cost / usage guard (best-effort).
# ---------------------------------------------------------------------------
# Set WEEKLY_BUDGET_USD to enable. Before each milestone the script stops if
# cumulative estimated cost has reached COST_FRACTION (default 0.80) of it.
# Cost comes from Claude Code's own `total_cost_usd` estimate (--output-format
# json). For Pro/Max subscriptions this is only an APPROXIMATION of weekly
# rate-limit usage — the authoritative figure is the interactive `/usage`
# command. The guard can only stop BETWEEN milestones; use MAX_TURNS to bound a
# single milestone.
WEEKLY_BUDGET_USD="${WEEKLY_BUDGET_USD:-}"
COST_FRACTION="${COST_FRACTION:-0.80}"
MAX_TURNS="${MAX_TURNS:-}"

# Extra flags passed to every `claude -p` call. The committed
# .claude/settings.json already sets `defaultMode: acceptEdits` plus a safe
# allow/deny list, so runs should not prompt. This env var lets you override or
# add flags without editing the script, e.g.:
#   CLAUDE_EXTRA_ARGS="--permission-mode acceptEdits" ./scripts/run_all_milestones_with_claude.sh
# Do NOT put --dangerously-skip-permissions here; the settings.json denylist
# (git push, sudo, curl/wget, rm, secret reads) is the safety net.
CLAUDE_EXTRA_ARGS="${CLAUDE_EXTRA_ARGS:---permission-mode acceptEdits}"

LOG_DIR="logs"
COST_LOG="${LOG_DIR}/milestone-costs.log"
CUMULATIVE_USD=0

if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: missing $TEMPLATE"
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: the 'claude' CLI is not on PATH. Install Claude Code first."
  exit 1
fi

mkdir -p "$LOG_DIR"

THRESHOLD_USD=""
if [ -n "$WEEKLY_BUDGET_USD" ]; then
  THRESHOLD_USD="$(python3 -c "print(f'{float(${WEEKLY_BUDGET_USD}) * float(${COST_FRACTION}):.4f}')")"
  echo "Cost guard : ON"
  echo "  weekly budget : \$${WEEKLY_BUDGET_USD}"
  echo "  stop fraction : ${COST_FRACTION}"
  echo "  stop at >=    : \$${THRESHOLD_USD} cumulative estimated cost"
else
  echo "Cost guard : OFF  (set WEEKLY_BUDGET_USD to enable a ${COST_FRACTION} stop)"
fi
[ -n "$MAX_TURNS" ] && echo "Per-milestone turn cap : ${MAX_TURNS}"
echo "Permissions : .claude/settings.json (acceptEdits + allow/deny list)"
echo "Extra flags : ${CLAUDE_EXTRA_ARGS:-<none>}"

echo "Running milestones $START..$END with Claude Code (one claude -p run per milestone)."
echo "Stops on: Claude failure, no new commit, dirty tree, pytest failure, or cost guard."
echo

for MILESTONE in $(seq "$START" "$END"); do
  echo "======================================================================"
  echo "Milestone $MILESTONE"
  echo "======================================================================"

  # Pre-flight cost guard (only stops between milestones).
  if [ -n "$THRESHOLD_USD" ]; then
    OVER="$(python3 -c "print(1 if float(${CUMULATIVE_USD}) >= float(${THRESHOLD_USD}) else 0)")"
    if [ "$OVER" = "1" ]; then
      echo "COST GUARD TRIPPED: cumulative \$${CUMULATIVE_USD} >= \$${THRESHOLD_USD}."
      echo "Stopping BEFORE Milestone $MILESTONE."
      echo "Resume later with: ./scripts/run_all_milestones_with_claude.sh $MILESTONE $END"
      exit 0
    fi
  fi

  BEFORE_COMMIT="$(git rev-parse HEAD)"
  RUN_LOG="${LOG_DIR}/milestone-$(printf '%02d' "$MILESTONE").log"

  CLAUDE_ARGS=(-p "Use docs/prompts/run-one-milestone-template.md. Implement Milestone ${MILESTONE} only. Do not implement later milestones. Do not push. Commit locally when the milestone is complete. Stop after exactly this one milestone." --output-format json)
  # Split CLAUDE_EXTRA_ARGS on whitespace into individual argv entries.
  if [ -n "$CLAUDE_EXTRA_ARGS" ]; then
    read -ra _EXTRA <<< "$CLAUDE_EXTRA_ARGS"
    CLAUDE_ARGS+=("${_EXTRA[@]}")
  fi
  [ -n "$MAX_TURNS" ] && CLAUDE_ARGS+=(--max-turns "$MAX_TURNS")

  # Run Claude; capture the JSON result to a log (gitignored via logs/).
  claude "${CLAUDE_ARGS[@]}" | tee "$RUN_LOG"
  echo

  # Surface the agent's final summary text.
  python3 -c "import json; d=json.load(open('${RUN_LOG}')); print(d.get('result',''))" 2>/dev/null || true

  # Accumulate estimated cost.
  RUN_USD="$(python3 -c "import json; d=json.load(open('${RUN_LOG}')); print(d.get('total_cost_usd') or d.get('cost_usd') or 0)" 2>/dev/null || echo 0)"
  CUMULATIVE_USD="$(python3 -c "print(f'{float(${CUMULATIVE_USD}) + float(${RUN_USD}):.4f}')")"
  echo "$(date -u +%FT%TZ) milestone=$MILESTONE run_usd=$RUN_USD cumulative_usd=$CUMULATIVE_USD" >> "$COST_LOG"

  AFTER_COMMIT="$(git rev-parse HEAD)"

  if [ "$BEFORE_COMMIT" = "$AFTER_COMMIT" ]; then
    echo "ERROR: Milestone $MILESTONE did not create a new commit. Stopping."
    git status
    exit 1
  fi

  if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean after Milestone $MILESTONE. Stopping."
    git status
    exit 1
  fi

  echo
  echo "Milestone $MILESTONE completed."
  echo "Commit        : $AFTER_COMMIT"
  echo "Run est. cost : \$${RUN_USD}"
  echo "Cumulative    : \$${CUMULATIVE_USD}"
  echo

  echo "Running smoke validation after Milestone $MILESTONE..."
  python3 -m pytest -q || {
    echo "ERROR: pytest failed after Milestone $MILESTONE. Stopping."
    exit 1
  }

  echo
done

echo "======================================================================"
echo "All requested milestones completed: $START..$END"
echo "Total estimated cost: \$${CUMULATIVE_USD}"
echo "======================================================================"
git log --oneline -20
