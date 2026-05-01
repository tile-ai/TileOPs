#!/usr/bin/env bash
# round-pre.sh <PR_NUMBER>
#
# Pre-round work for resolve-tileops: locate state, snapshot the current
# PR view, decide action (continue/idle/terminate). On 'continue', gather
# this round's input snapshots and archive any inbox.
#
# Prerequisite: preflight.sh must have already initialized state for this
# PR. round-pre.sh does NOT init; it errors if state is missing.
#
# Stdout: single JSON object describing the action and (on continue) the
#         snapshot prefix the skill body should read.
# Stderr: human-readable status / errors.
# Exit 0: action ready (skill body should branch on .action).
# Exit non-zero: missing state or upstream failure.

set -euo pipefail

PR="${1:?usage: round-pre.sh <PR_NUMBER>}"
[[ "$PR" =~ ^[0-9]+$ ]] || { echo "round-pre: PR must be a positive integer" >&2; exit 1; }

REPO="tile-ai/TileOPs"
REVIEWER_LOGIN="${RESOLVE_REVIEWER_LOGIN:-Ibuki-wind}"
REPO_PATH="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || { echo "round-pre: not in a git repo" >&2; exit 1; }

# Locate state. preflight.sh must have created it.
META=""
for m in "$REPO_PATH/.foundry/runs"/*/resolve/meta.json; do
  [[ -f "$m" ]] || continue
  if [[ "$(jq -r '.pr_number' "$m" 2>/dev/null)" = "$PR" ]]; then
    META="$m"
    break
  fi
done
[[ -n "$META" ]] \
  || { echo "round-pre: no state for PR #$PR — run preflight.sh first" >&2; exit 1; }

RUN_DIR=$(dirname "$META")

ROUND=$(jq -r '.round' "$META")
MAX_ROUNDS=$(jq -r '.max_rounds' "$META")
LAST_REVIEW_ID_PREV=$(jq -r '.last_processed_review_id' "$META")
LAST_REVIEW_COMMENT_ID_PREV=$(jq -r '.last_processed_review_comment_id' "$META")

PR_JSON=$(gh pr view "$PR" --repo "$REPO" --json state,headRefOid,isDraft 2>/dev/null) \
  || { echo "round-pre: gh pr view failed" >&2; exit 1; }
PR_STATE=$(echo "$PR_JSON" | jq -r .state)
HEAD_SHA=$(echo "$PR_JSON" | jq -r .headRefOid)

# Reviews + inline comments — paginate so PRs with >1 page don't
# silently lose the latest IDs / state.
LATEST_REVIEWER_STATE=$(gh api --paginate --slurp "repos/$REPO/pulls/$PR/reviews" \
  --jq "[.[][]|select(.user.login==\"$REVIEWER_LOGIN\")] | sort_by(.submitted_at) | last | .state // \"NONE\"")
LATEST_REVIEW_ID=$(gh api --paginate --slurp "repos/$REPO/pulls/$PR/reviews" \
  --jq "[.[][]|select(.user.login==\"$REVIEWER_LOGIN\")|.id]|max // 0")
LATEST_REVIEW_COMMENT_ID=$(gh api --paginate --slurp "repos/$REPO/pulls/$PR/comments" \
  --jq "[.[][]|select(.user.login==\"$REVIEWER_LOGIN\")|.id]|max // 0")

# Unresolved review thread count — paginate via cursor so PRs with
# >100 threads don't undercount.
count_unresolved() {
  local cursor='null' total=0 page page_unresolved has_next
  while :; do
    page=$(gh api graphql -f query='
      query($owner:String!,$repo:String!,$pr:Int!,$after:String){
        repository(owner:$owner,name:$repo){
          pullRequest(number:$pr){
            reviewThreads(first:100, after:$after){
              nodes{ isResolved }
              pageInfo{ hasNextPage endCursor }
            }
          }
        }
      }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
        ${cursor:+-f after="$cursor"})
    page_unresolved=$(printf '%s' "$page" \
      | jq '[.data.repository.pullRequest.reviewThreads.nodes[]|select(.isResolved==false)]|length')
    total=$((total + page_unresolved))
    has_next=$(printf '%s' "$page" \
      | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
    [[ "$has_next" == "true" ]] || break
    cursor=$(printf '%s' "$page" \
      | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
  done
  echo "$total"
}
UNRESOLVED=$(count_unresolved)

# Decide action — first match wins. PR_STATE==DRAFT does not stop.
ACTION=""; MESSAGE=""
case "$PR_STATE" in
  MERGED|CLOSED) ACTION="terminate-external"; MESSAGE="PR #$PR is $PR_STATE — stopping." ;;
esac
if [[ -z "$ACTION" && "$ROUND" -ge "$MAX_ROUNDS" ]]; then
  ACTION="terminate-diverged"
  MESSAGE="Reached max rounds ($MAX_ROUNDS) — human attention needed."
fi
if [[ -z "$ACTION" \
      && "$UNRESOLVED" -eq 0 \
      && "$LATEST_REVIEWER_STATE" == "APPROVED" \
      && "$LATEST_REVIEW_COMMENT_ID" == "$LAST_REVIEW_COMMENT_ID_PREV" ]]; then
  # Approve + nothing actionable left → exit immediately. The APPROVE
  # review itself doesn't need "processing" — no threads, no inline
  # comments to reply to. Symmetric with review-tileops/loop.sh's
  # converge condition (state + sha + comment-id stable).
  ACTION="terminate-success"
  MESSAGE="PR #$PR converged — all threads resolved, reviewer approved."
fi

# Idle gate: only sleep when there's nothing to do. Unresolved threads
# (from any source) override idle — the dev should still process them
# even if the canonical reviewer hasn't posted new activity.
if [[ -z "$ACTION" \
      && "$UNRESOLVED" -eq 0 \
      && "$LATEST_REVIEW_ID" == "$LAST_REVIEW_ID_PREV" \
      && "$LATEST_REVIEW_COMMENT_ID" == "$LAST_REVIEW_COMMENT_ID_PREV" ]]; then
  ACTION="idle"
  MESSAGE="No new review feedback for PR #$PR — sleeping."
fi

[[ -z "$ACTION" ]] && ACTION="continue"

NEXT_ROUND=$((ROUND + 1))
SNAP_PREFIX=""

if [[ "$ACTION" == "continue" ]]; then
  N=$(printf '%02d' "$NEXT_ROUND")
  SNAP_PREFIX="$RUN_DIR/rounds/round-$N"
  mkdir -p "$RUN_DIR/rounds"

  gh api --paginate --slurp "repos/$REPO/pulls/$PR/reviews" \
    --jq "[.[][]|select(.user.login==\"$REVIEWER_LOGIN\" and .id>$LAST_REVIEW_ID_PREV)|{id,state,body,submitted_at}]" \
    > "$SNAP_PREFIX.new-reviews.json"

  gh api --paginate --slurp "repos/$REPO/pulls/$PR/comments" \
    --jq "[.[][]|select(.user.login==\"$REVIEWER_LOGIN\" and .id>$LAST_REVIEW_COMMENT_ID_PREV)|{id,path,line,body,in_reply_to_id,created_at}]" \
    > "$SNAP_PREFIX.new-inline-comments.json"

  # Snapshot ALL unresolved threads (paginated) — agent acts on these
  # regardless of which reviewer raised them.
  : > "$SNAP_PREFIX.unresolved-threads.json"
  echo '[' > "$SNAP_PREFIX.unresolved-threads.json"
  cursor='null'; first_page=1
  while :; do
    page=$(gh api graphql -f query='
      query($owner:String!,$repo:String!,$pr:Int!,$after:String){
        repository(owner:$owner,name:$repo){
          pullRequest(number:$pr){
            reviewThreads(first:100, after:$after){
              nodes{
                id isResolved
                comments(first:100){ nodes{ databaseId author{login} body path line } }
              }
              pageInfo{ hasNextPage endCursor }
            }
          }
        }
      }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
        ${cursor:+-f after="$cursor"})
    items=$(printf '%s' "$page" \
      | jq -c '.data.repository.pullRequest.reviewThreads.nodes|map(select(.isResolved==false))[]')
    if [[ -n "$items" ]]; then
      while IFS= read -r line; do
        [[ "$first_page" -eq 1 ]] && first_page=0 || echo ',' >> "$SNAP_PREFIX.unresolved-threads.json"
        echo -n "$line" >> "$SNAP_PREFIX.unresolved-threads.json"
      done <<< "$items"
    fi
    has_next=$(printf '%s' "$page" \
      | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
    [[ "$has_next" == "true" ]] || break
    cursor=$(printf '%s' "$page" \
      | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
  done
  echo ']' >> "$SNAP_PREFIX.unresolved-threads.json"

  gh pr checks "$PR" --repo "$REPO" --json name,state,conclusion \
    > "$SNAP_PREFIX.ci.json" 2>/dev/null || echo '[]' > "$SNAP_PREFIX.ci.json"

  # Archive inbox for this round, then clear it. Skill body reads the
  # archived copy if it wants this round's guidance.
  if [[ -s "$RUN_DIR/inbox.md" ]]; then
    mv "$RUN_DIR/inbox.md" "$RUN_DIR/inbox-history/round-$N.md"
    : > "$RUN_DIR/inbox.md"
  fi

  # Persist baseline so round-post.sh can compute deltas without
  # re-querying. Critically, persist LATEST_REVIEW_ID and
  # LATEST_REVIEW_COMMENT_ID so round-post advances the watermark to the
  # PRE-round max — items that arrive mid-round get picked up next round.
  jq -n --arg sha "$HEAD_SHA" \
    --argjson unresolved "$UNRESOLVED" \
    --arg state "$LATEST_REVIEWER_STATE" \
    --argjson rid "$LATEST_REVIEW_ID" \
    --argjson cid "$LATEST_REVIEW_COMMENT_ID" \
    '{head_sha:$sha, unresolved_before:$unresolved, reviewer_state_before:$state,
      latest_review_id:$rid, latest_review_comment_id:$cid}' \
    > "$RUN_DIR/.round-pre.json"
fi

jq -n \
  --arg action "$ACTION" \
  --arg run_dir "$RUN_DIR" \
  --arg snap_prefix "$SNAP_PREFIX" \
  --argjson round "$NEXT_ROUND" \
  --arg message "$MESSAGE" \
  '{action:$action, round:$round, run_dir:$run_dir, snap_prefix:$snap_prefix, message:$message}'
