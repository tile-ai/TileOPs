#!/usr/bin/env bash
# round-post.sh <PR_NUMBER>
#
# Post-round work for resolve-tileops: detect what changed during the
# round (push, threads resolved), write the round summary, advance meta.
# Reads baseline from $RUN_DIR/.round-pre.json (left by round-pre.sh).
#
# Stdout: single JSON line summary {round, pushed_sha, threads_resolved,
#         open_after}.
# Stderr: human-readable status / errors.
# Exit 0: round finalized. Non-zero: missing state.

set -euo pipefail

PR="${1:?usage: round-post.sh <PR_NUMBER>}"
[[ "$PR" =~ ^[0-9]+$ ]] || { echo "round-post: PR must be a positive integer" >&2; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "round-post: missing gh" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "round-post: missing jq" >&2; exit 1; }

REPO="tile-ai/TileOPs"
# Anchor state lookup to the main checkout (see preflight.sh).
REPO_PATH="$(git worktree list --porcelain 2>/dev/null | head -n 1 | sed 's/^worktree //')" \
  || { echo "round-post: not in a git repo" >&2; exit 1; }

META=""
for m in "$REPO_PATH/.foundry/runs"/*/resolve/meta.json; do
  [[ -f "$m" ]] || continue
  if [[ "$(jq -r '.pr_number' "$m" 2>/dev/null)" = "$PR" ]]; then
    META="$m"
    break
  fi
done
[[ -n "$META" ]] || { echo "round-post: no state for PR #$PR" >&2; exit 1; }

RUN_DIR=$(dirname "$META")
PRE_JSON="$RUN_DIR/.round-pre.json"
[[ -f "$PRE_JSON" ]] \
  || { echo "round-post: missing $PRE_JSON — was round-pre.sh run with action=continue?" >&2; exit 1; }

HEAD_SHA_BEFORE=$(jq -r '.head_sha' "$PRE_JSON")
UNRESOLVED_BEFORE=$(jq -r '.unresolved_before' "$PRE_JSON")
REVIEWER_STATE_BEFORE=$(jq -r '.reviewer_state_before' "$PRE_JSON")
# Use the PRE-round watermark when advancing meta. If the reviewer added
# new comments mid-round (after round-pre snapshotted but before
# round-post), they'd be lost if we used the post-round max. Reading from
# the baseline guarantees mid-round feedback is processed next round.
PRE_LATEST_REVIEW_ID=$(jq -r '.latest_review_id' "$PRE_JSON")
PRE_LATEST_REVIEW_COMMENT_ID=$(jq -r '.latest_review_comment_id' "$PRE_JSON")

NEW_HEAD_SHA=$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)
PUSHED_SHA="none"
[[ "$NEW_HEAD_SHA" != "$HEAD_SHA_BEFORE" ]] && PUSHED_SHA="$NEW_HEAD_SHA"

# Paginate reviewThreads — same rationale as round-pre.sh.
NEW_UNRESOLVED=0
cursor=''
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
  NEW_UNRESOLVED=$((NEW_UNRESOLVED + page_unresolved))
  has_next=$(printf '%s' "$page" \
    | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
  [[ "$has_next" == "true" ]] || break
  cursor=$(printf '%s' "$page" \
    | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
done

# Clamp at 0 — new threads opened during the round can make the raw
# delta negative, but reporting "-3 threads_resolved" is misleading.
THREADS_RESOLVED_RAW=$(( UNRESOLVED_BEFORE - NEW_UNRESOLVED ))
if (( THREADS_RESOLVED_RAW < 0 )); then
  THREADS_RESOLVED=0
else
  THREADS_RESOLVED=$THREADS_RESOLVED_RAW
fi

ROUND=$(jq -r '.round' "$META")
NEXT_ROUND=$((ROUND + 1))
N=$(printf '%02d' "$NEXT_ROUND")
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Round summary file
jq -n --argjson r "$NEXT_ROUND" --arg now "$NOW" \
  --arg sha_b "$HEAD_SHA_BEFORE" --arg sha_a "$NEW_HEAD_SHA" \
  --arg pushed "$PUSHED_SHA" \
  --argjson resolved "$THREADS_RESOLVED" \
  --argjson unresolved_after "$NEW_UNRESOLVED" \
  --arg reviewer_state "$REVIEWER_STATE_BEFORE" \
  '{round:$r, finished_at:$now, head_sha_before:$sha_b, head_sha_after:$sha_a,
    pushed_sha:$pushed, threads_resolved:$resolved,
    unresolved_after:$unresolved_after, reviewer_state_before:$reviewer_state}' \
  > "$RUN_DIR/rounds/round-$N.json"

# Advance meta. last_pushed_sha is sticky: only touched when a push
# actually happened this round, preserving its prior value (null on a
# fresh state, or the last real sha) otherwise. Watermarks come from the
# PRE-round baseline so mid-round reviewer activity is picked up next
# round.
if [[ "$PUSHED_SHA" != "none" ]]; then
  jq --argjson r "$NEXT_ROUND" \
     --argjson rid "$PRE_LATEST_REVIEW_ID" \
     --argjson cid "$PRE_LATEST_REVIEW_COMMENT_ID" \
     --arg pushed "$PUSHED_SHA" \
    '.round=$r | .last_processed_review_id=$rid
     | .last_processed_review_comment_id=$cid
     | .last_pushed_sha=$pushed' \
    "$META" > "$META.tmp" && mv "$META.tmp" "$META"
else
  jq --argjson r "$NEXT_ROUND" \
     --argjson rid "$PRE_LATEST_REVIEW_ID" \
     --argjson cid "$PRE_LATEST_REVIEW_COMMENT_ID" \
    '.round=$r | .last_processed_review_id=$rid
     | .last_processed_review_comment_id=$cid' \
    "$META" > "$META.tmp" && mv "$META.tmp" "$META"
fi

rm -f "$PRE_JSON"

jq -n \
  --argjson r "$NEXT_ROUND" \
  --arg pushed "$PUSHED_SHA" \
  --argjson resolved "$THREADS_RESOLVED" \
  --argjson open_after "$NEW_UNRESOLVED" \
  '{round:$r, pushed_sha:$pushed, threads_resolved:$resolved, open_after:$open_after}'
