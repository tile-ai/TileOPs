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

LATEST_REVIEWER_STATE=$(gh api "repos/$REPO/pulls/$PR/reviews" \
  --jq '[.[]|select(.user.login=="Ibuki-wind")] | sort_by(.submitted_at) | last | .state // "NONE"')
LATEST_REVIEW_ID=$(gh api "repos/$REPO/pulls/$PR/reviews" \
  --jq '[.[]|select(.user.login=="Ibuki-wind")|.id]|max // 0')
LATEST_REVIEW_COMMENT_ID=$(gh api "repos/$REPO/pulls/$PR/comments" \
  --jq '[.[]|select(.user.login=="Ibuki-wind")|.id]|max // 0')

UNRESOLVED=$(gh api graphql -f query='
  query($owner:String!,$repo:String!,$pr:Int!){
    repository(owner:$owner,name:$repo){
      pullRequest(number:$pr){
        reviewThreads(first:100){ nodes{ isResolved } }
      }
    }
  }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
  --jq '[.data.repository.pullRequest.reviewThreads.nodes[]|select(.isResolved==false)]|length')

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
  # converge condition (state + sha + comment-id stable; no review-id
  # equality requirement).
  ACTION="terminate-success"
  MESSAGE="PR #$PR converged — all threads resolved, reviewer approved."
fi

# Idle gate: no new reviewer activity since last processed round.
if [[ -z "$ACTION" \
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

  gh api "repos/$REPO/pulls/$PR/reviews" \
    --jq "[.[]|select(.user.login==\"Ibuki-wind\" and .id>$LAST_REVIEW_ID_PREV)|{id,state,body,submitted_at}]" \
    > "$SNAP_PREFIX.new-reviews.json"

  gh api "repos/$REPO/pulls/$PR/comments" \
    --jq "[.[]|select(.user.login==\"Ibuki-wind\" and .id>$LAST_REVIEW_COMMENT_ID_PREV)|{id,path,line,body,in_reply_to_id,created_at}]" \
    > "$SNAP_PREFIX.new-inline-comments.json"

  gh api graphql -f query='
    query($owner:String!,$repo:String!,$pr:Int!){
      repository(owner:$owner,name:$repo){
        pullRequest(number:$pr){
          reviewThreads(first:100){
            nodes{
              id isResolved
              comments(first:20){ nodes{ databaseId author{login} body path line } }
            }
          }
        }
      }
    }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
    --jq '.data.repository.pullRequest.reviewThreads.nodes|map(select(.isResolved==false))' \
    > "$SNAP_PREFIX.unresolved-threads.json"

  gh pr checks "$PR" --repo "$REPO" --json name,state,conclusion \
    > "$SNAP_PREFIX.ci.json" 2>/dev/null || echo '[]' > "$SNAP_PREFIX.ci.json"

  # Archive inbox for this round, then clear it. Skill body reads the
  # archived copy if it wants this round's guidance.
  if [[ -s "$RUN_DIR/inbox.md" ]]; then
    mv "$RUN_DIR/inbox.md" "$RUN_DIR/inbox-history/round-$N.md"
    : > "$RUN_DIR/inbox.md"
  fi

  # Persist baseline so round-post.sh can compute deltas without
  # re-querying.
  jq -n --arg sha "$HEAD_SHA" \
    --argjson unresolved "$UNRESOLVED" \
    --arg state "$LATEST_REVIEWER_STATE" \
    '{head_sha:$sha, unresolved_before:$unresolved, reviewer_state_before:$state}' \
    > "$RUN_DIR/.round-pre.json"
fi

jq -n \
  --arg action "$ACTION" \
  --arg run_dir "$RUN_DIR" \
  --arg snap_prefix "$SNAP_PREFIX" \
  --argjson round "$NEXT_ROUND" \
  --arg message "$MESSAGE" \
  '{action:$action, round:$round, run_dir:$run_dir, snap_prefix:$snap_prefix, message:$message}'
