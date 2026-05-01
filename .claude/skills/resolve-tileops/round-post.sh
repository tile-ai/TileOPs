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

REPO="tile-ai/TileOPs"
REPO_PATH="$(git rev-parse --show-toplevel 2>/dev/null)" \
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

NEW_HEAD_SHA=$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)
PUSHED_SHA="none"
[[ "$NEW_HEAD_SHA" != "$HEAD_SHA_BEFORE" ]] && PUSHED_SHA="$NEW_HEAD_SHA"

NEW_UNRESOLVED=$(gh api graphql -f query='
  query($owner:String!,$repo:String!,$pr:Int!){
    repository(owner:$owner,name:$repo){
      pullRequest(number:$pr){
        reviewThreads(first:100){ nodes{ isResolved } }
      }
    }
  }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
  --jq '[.data.repository.pullRequest.reviewThreads.nodes[]|select(.isResolved==false)]|length')
THREADS_RESOLVED=$(( UNRESOLVED_BEFORE - NEW_UNRESOLVED ))

LATEST_REVIEW_ID=$(gh api "repos/$REPO/pulls/$PR/reviews" \
  --jq '[.[]|select(.user.login=="Ibuki-wind")|.id]|max // 0')
LATEST_REVIEW_COMMENT_ID=$(gh api "repos/$REPO/pulls/$PR/comments" \
  --jq '[.[]|select(.user.login=="Ibuki-wind")|.id]|max // 0')

ROUND=$(jq -r '.round' "$META")
NEXT_ROUND=$((ROUND + 1))
N=$(printf '%02d' "$NEXT_ROUND")
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LAST_PUSHED_SHA_PREV=$(jq -r '.last_pushed_sha // ""' "$META")

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

# Advance meta. last_pushed_sha is sticky: only updated when a push
# actually happened this round.
PUSHED_FOR_META="$LAST_PUSHED_SHA_PREV"
[[ "$PUSHED_SHA" != "none" ]] && PUSHED_FOR_META="$PUSHED_SHA"
jq --argjson r "$NEXT_ROUND" \
   --argjson rid "$LATEST_REVIEW_ID" \
   --argjson cid "$LATEST_REVIEW_COMMENT_ID" \
   --arg pushed "$PUSHED_FOR_META" \
  '.round=$r | .last_processed_review_id=$rid
   | .last_processed_review_comment_id=$cid
   | .last_pushed_sha=$pushed' \
  "$META" > "$META.tmp" && mv "$META.tmp" "$META"

rm -f "$PRE_JSON"

jq -n \
  --argjson r "$NEXT_ROUND" \
  --arg pushed "$PUSHED_SHA" \
  --argjson resolved "$THREADS_RESOLVED" \
  --argjson open_after "$NEW_UNRESOLVED" \
  '{round:$r, pushed_sha:$pushed, threads_resolved:$resolved, open_after:$open_after}'
