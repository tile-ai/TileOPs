#!/usr/bin/env bash
# round-post.sh — post-codex hook for the review loop.
#
# Implements Rule 2 (same-path 5-strike monitor): track per-path
# consecutive-blocker streaks across rounds. When any path's streak
# reaches >= 5 consecutive rounds with at least one blocker on it,
# ensure the PR carries the GitHub label "agent-stuck" so a human
# can take a look.
#
# Monitoring-only: this hook MUST NOT modify the round's blockers,
# the unresolved-threads file, the new-review-comments.json file,
# the PR title, or post any comment. Its only side-effects on the
# PR are GitHub label operations (idempotent).
#
# Inputs (env or positional, env wins):
#   RUN_DIR        — review/ root for this loop run.
#   ROUND          — integer round number that just finished.
#   COMMENTS_JSON  — path to round-NN.codex-blockers.json (the post-
#                    codex snapshot of review comments authored by
#                    REVIEWER_LOGIN this round; used as the source of
#                    blocker paths). The pre-codex
#                    round-NN.new-review-comments.json is the WRONG
#                    artifact — it explicitly filters out the reviewer
#                    and contains human comments instead.
#   REPO           — "owner/repo" string for `gh` operations.
#                    Default: tile-ai/TileOPs.
#   PR             — PR number for `gh pr edit`.
#
# Positional fallback: round-post.sh <RUN_DIR> <ROUND> <COMMENTS_JSON> <REPO> <PR>
#
# Optional env (for testing):
#   GH_BIN         — gh executable to use (default: "gh"). Tests can
#                    point this at a stub. If empty or "none", label
#                    operations are skipped entirely (monitor still
#                    updates region-history.json).
#
# Output: exit 0 on success. Logs to stderr; nothing on stdout.
set -euo pipefail

RUN_DIR="${RUN_DIR:-${1:-}}"
ROUND="${ROUND:-${2:-}}"
COMMENTS_JSON="${COMMENTS_JSON:-${3:-}}"
REPO="${REPO:-${4:-tile-ai/TileOPs}}"
PR="${PR:-${5:-}}"
GH_BIN="${GH_BIN:-gh}"

if [[ -z "$RUN_DIR" || -z "$ROUND" || -z "$COMMENTS_JSON" ]]; then
  echo "round-post.sh: missing args; usage: RUN_DIR=... ROUND=... COMMENTS_JSON=... [REPO=... PR=...] round-post.sh" >&2
  exit 2
fi

HISTORY="$RUN_DIR/region-history.json"
THRESHOLD=5
LABEL="agent-stuck"
LABEL_DESC="Autonomous loop appears stuck and needs human attention"
LABEL_COLOR="FBCA04"

# --- 1. Extract blocker paths from this round's review comments ---
#
# Source-of-truth contract: this round's codex-blockers.json is the
# post-codex snapshot of review comments authored by the reviewer login
# (i.e. comments codex itself just posted as part of this review).
# Entries with severity=blocker (or category=blocker) carry the .path
# the reviewer flagged. We treat any entry whose .severity or .category
# is "blocker" as a blocker. If neither field exists (older format or
# free-form review), every entry counts. Round-post is monitoring-only:
# the only side-effect is the `agent-stuck` label, which is itself
# idempotent and safe to apply early. Over-counting accelerates the
# threshold (i.e. the label may land sooner than a strict-blocker count
# would warrant); it never produces an unsafe action.
PATHS_THIS_ROUND=()
if [[ -f "$COMMENTS_JSON" ]]; then
  # tolerate empty / malformed file by falling back to []
  if ! jq empty "$COMMENTS_JSON" >/dev/null 2>&1; then
    echo "round-post.sh: $COMMENTS_JSON is not valid JSON; treating as empty" >&2
  elif ! jq -e 'type == "array"' "$COMMENTS_JSON" >/dev/null 2>&1; then
    # `jq empty` accepts ANY valid JSON, including objects. If the
    # artifact is e.g. `{path:"x"}` (a single comment object, or an
    # error-wrapper produced by a failed gh call), `.[]` would extract
    # its values and miscount them as blocker paths. Require an array
    # before iterating; non-array input is treated as no blockers this
    # round (counters will reset, label stays unchanged).
    echo "round-post.sh: $COMMENTS_JSON is not a JSON array; treating as empty" >&2
  else
    while IFS= read -r p; do
      [[ -n "$p" && "$p" != "null" ]] && PATHS_THIS_ROUND+=("$p")
    done < <(jq -r '
      [
        .[]
        | select(
            (.severity // "" | ascii_downcase) == "blocker"
            or (.category // "" | ascii_downcase) == "blocker"
            or ((.severity // "") == "" and (.category // "") == "")
          )
        | .path // empty
      ]
      | unique
      | .[]' "$COMMENTS_JSON")
  fi
fi

# --- 2. Update region-history.json: increment present, reset absent ---
#
# Schema (stable across rounds; loaded by future tools):
#   {
#     "counters": { "<path>": <int>, ... },
#     "events":   [ {"round": int, "path": str, "comment_ids": [int]}, ... ]
#   }
#
# Backward compat: missing file → start from empty state.
if [[ ! -f "$HISTORY" ]]; then
  echo '{"counters":{},"events":[]}' > "$HISTORY"
fi

# Build a JSON array of paths-this-round for jq.
PATHS_JSON=$(printf '%s\n' "${PATHS_THIS_ROUND[@]:-}" \
  | jq -R . \
  | jq -s 'map(select(. != ""))')

NEW_HISTORY=$(jq \
  --argjson paths "$PATHS_JSON" \
  '
  . as $h
  | (.counters // {}) as $cur
  # increment counters for paths present this round; reset absent.
  | (
      ($paths | map({key:., value: (($cur[.] // 0) + 1)}) | from_entries)
    ) as $next
  | .counters = $next
  | .events = (.events // [])
  ' "$HISTORY")

printf '%s\n' "$NEW_HISTORY" > "$HISTORY.tmp" && mv "$HISTORY.tmp" "$HISTORY"

# --- 3. Determine which paths just hit the threshold this round ---
TRIGGERED_PATHS=()
while IFS= read -r p; do
  [[ -n "$p" ]] && TRIGGERED_PATHS+=("$p")
done < <(jq -r --argjson t "$THRESHOLD" '
  .counters
  | to_entries
  | map(select(.value == $t))
  | .[].key' "$HISTORY")

# --- 4. Append events for triggered paths (postmortem trail) ---
if [[ "${#TRIGGERED_PATHS[@]}" -gt 0 ]]; then
  # collect comment ids (best-effort) so the event log carries enough
  # info to triangulate which findings drove the strike.
  for path in "${TRIGGERED_PATHS[@]}"; do
    if [[ -f "$COMMENTS_JSON" ]] \
        && jq empty "$COMMENTS_JSON" >/dev/null 2>&1 \
        && jq -e 'type == "array"' "$COMMENTS_JSON" >/dev/null 2>&1; then
      IDS_JSON=$(jq --arg p "$path" '[.[]|select(.path==$p)|.id // empty]' "$COMMENTS_JSON" 2>/dev/null || echo '[]')
    else
      IDS_JSON='[]'
    fi
    UPDATED=$(jq \
      --argjson r "$ROUND" \
      --arg p "$path" \
      --argjson ids "$IDS_JSON" \
      '.events += [{round:$r, path:$p, comment_ids:$ids}]' \
      "$HISTORY")
    printf '%s\n' "$UPDATED" > "$HISTORY.tmp" && mv "$HISTORY.tmp" "$HISTORY"
  done
fi

# --- 5. Apply the agent-stuck label if any path triggered ---
#
# Idempotent: gh label create is no-op when the label exists; gh pr
# edit --add-label is no-op when the PR already has it. We swallow
# benign "already exists" errors so a transient gh failure doesn't
# crash the loop (Rule 2 is monitor-only by design).
if [[ "${#TRIGGERED_PATHS[@]}" -gt 0 && -n "$PR" && "$GH_BIN" != "none" && -n "$GH_BIN" ]]; then
  if ! command -v "$GH_BIN" >/dev/null 2>&1 && [[ ! -x "$GH_BIN" ]]; then
    echo "round-post.sh: $GH_BIN not found; skipping label application" >&2
  else
    "$GH_BIN" label create "$LABEL" \
      --repo "$REPO" \
      --description "$LABEL_DESC" \
      --color "$LABEL_COLOR" \
      >/dev/null 2>&1 || true
    "$GH_BIN" pr edit "$PR" \
      --repo "$REPO" \
      --add-label "$LABEL" \
      >/dev/null 2>&1 || true
  fi
fi

exit 0
