#!/usr/bin/env bash
# preflight.sh <PR_NUMBER>
#
# Validates env and initializes per-PR resolve state. Idempotent: round 1
# does cold init (env checks, TASK_ROOT resolution from PR body, mkdir,
# meta.json), round 2+ scans existing state and returns it without rework.
# The skill body assumes preflight has succeeded.
#
# Stdout: absolute path to the run dir's meta.json (single line).
# Stderr: human-readable status / errors.
# Exit 0: state ready. Non-zero: env or arg error.

set -euo pipefail

PR="${1:?usage: preflight.sh <PR_NUMBER>}"
[[ "$PR" =~ ^[0-9]+$ ]] || { echo "preflight: PR must be a positive integer" >&2; exit 1; }

REPO="tile-ai/TileOPs"
# Dependency checks first — both branches below use jq/gh.
command -v gh >/dev/null 2>&1 || { echo "preflight: missing gh" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "preflight: missing jq" >&2; exit 1; }

REPO_PATH="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || { echo "preflight: not in a git repo" >&2; exit 1; }

# Round 2+ fast path: an existing run dir's meta.json already pins this PR.
META=""
for m in "$REPO_PATH/.foundry/runs"/*/resolve/meta.json; do
  [[ -f "$m" ]] || continue
  if [[ "$(jq -r '.pr_number' "$m" 2>/dev/null)" = "$PR" ]]; then
    META="$m"
    break
  fi
done

if [[ -n "$META" ]]; then
  echo "preflight: state exists for PR #$PR at $META" >&2
  echo "$META"
  exit 0
fi

# Round 1 cold start: validate repo remote, resolve TASK_ROOT, create state.
git -C "$REPO_PATH" remote -v \
  | awk '/tile-ai\/TileOPs(\.git)?[[:space:]]+\(fetch\)/ {found=1; exit} END{exit !found}' \
  || { echo "preflight: no git remote in $REPO_PATH points to tile-ai/TileOPs" >&2; exit 1; }

PR_BODY=$(gh pr view "$PR" --repo "$REPO" --json body --jq .body) \
  || { echo "preflight: gh pr view failed for PR #$PR (auth? missing?)" >&2; exit 1; }
ISSUE=$(printf '%s' "$PR_BODY" \
  | grep -oiE '(Closes|Fixes|Resolves)[[:space:]]+#[0-9]+' \
  | head -1 \
  | grep -oE '[0-9]+' \
  || true)
if [[ -n "$ISSUE" ]]; then
  TASK_ROOT="$REPO_PATH/.foundry/runs/issue-$ISSUE"
else
  TASK_ROOT="$REPO_PATH/.foundry/runs/pr-$PR"
fi
RUN_DIR="$TASK_ROOT/resolve"
META="$RUN_DIR/meta.json"

mkdir -p "$RUN_DIR/rounds" "$RUN_DIR/inbox-history"
: > "$RUN_DIR/inbox.md"
jq -n --arg pr "$PR" --arg repo "$REPO" '{
  pr_number:($pr|tonumber), repo:$repo,
  status:"active",
  round:0, max_rounds:15,
  last_processed_review_id:0,
  last_processed_review_comment_id:0,
  last_pushed_sha:null,
  consecutive_idle:0, max_idle:20
}' > "$META"

echo "preflight: state initialized for PR #$PR at $META" >&2
echo "$META"
