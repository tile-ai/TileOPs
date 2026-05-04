#!/usr/bin/env bash
# round-pre.sh — pre-codex hook for the review loop.
#
# Implements Rule 1 (same-SHA APPROVE skip): if any prior round in
# this loop run already produced an APPROVE on the *current* HEAD
# sha, skip the codex invocation entirely. Reuse the prior APPROVE
# verbatim by writing a marker round-NN.json that carries the same
# codex_event/blockers as the prior round and points back at it
# via the "approve_reused_from" field.
#
# Inputs (env or positional, env wins):
#   RUN_DIR           — review/ root for this loop run; must contain rounds/.
#   NEXT_ROUND        — integer round number about to run (e.g. 5).
#   HEAD_SHA          — full git sha of the PR HEAD this round would review.
#   LATEST_ISSUE_ID   — current max id of any non-reviewer top-level PR
#                       (issue) comment.
#   LATEST_REVIEW_ID  — current max id of any non-reviewer review/inline
#                       comment (incl. thread replies).
#                       Each is independent. When supplied, the reuse
#                       decision requires the prior approving round's
#                       counterpart watermark to match. A caller may
#                       supply only one — the unsupplied dimension is
#                       not constrained. If a NEW human comment of
#                       either kind landed on the same SHA after the
#                       prior APPROVE, the matched watermark mismatches
#                       and this hook returns "proceed" so codex re-
#                       reviews and ingests the comment. When BOTH are
#                       absent or empty, the watermark guard is skipped
#                       entirely (legacy behavior, used by tests that
#                       pre-date the watermark fields).
#
# Positional fallback:
#   round-pre.sh <RUN_DIR> <NEXT_ROUND> <HEAD_SHA> [LATEST_ISSUE_ID] [LATEST_REVIEW_ID]
#
# Output:
#   - exit 0 + stdout "skip"     → loop should skip codex; round-NN.json is
#                                  already written by this script.
#   - exit 0 + stdout "proceed"  → no prior APPROVE on this SHA, OR a
#                                  recoverable condition was hit (missing
#                                  args, missing rounds dir, malformed
#                                  prior round file, etc.). The loop's
#                                  ``|| echo "proceed"`` fallback expects
#                                  this contract: round-pre.sh is a
#                                  monitor and MUST NOT exit non-zero on
#                                  recoverable conditions, otherwise
#                                  corrupt state would be silently
#                                  re-mapped to "proceed" and could
#                                  re-enable a same-SHA re-review.
#   - non-zero exit              → reserved for truly unexpected errors
#                                  (e.g. jq missing). Currently unused.
#
# Determinism: pure file scan. No gh / git / network. No LLM.
set -uo pipefail

RUN_DIR="${RUN_DIR:-${1:-}}"
NEXT_ROUND="${NEXT_ROUND:-${2:-}}"
HEAD_SHA="${HEAD_SHA:-${3:-}}"
LATEST_ISSUE_ID="${LATEST_ISSUE_ID:-${4:-}}"
LATEST_REVIEW_ID="${LATEST_REVIEW_ID:-${5:-}}"

if [[ -z "$RUN_DIR" || -z "$NEXT_ROUND" || -z "$HEAD_SHA" ]]; then
  # Missing args is a recoverable misconfiguration: emit "proceed" so the
  # loop continues with codex rather than failing fast. (See contract
  # note above.)
  echo "round-pre.sh: missing args; usage: RUN_DIR=... NEXT_ROUND=... HEAD_SHA=... round-pre.sh" >&2
  echo "proceed"
  exit 0
fi

ROUNDS_DIR="$RUN_DIR/rounds"
if [[ ! -d "$ROUNDS_DIR" ]]; then
  # No prior rounds → nothing to reuse; loop proceeds normally.
  echo "proceed"
  exit 0
fi

# Scan prior round-*.json files for an APPROVE on the same head_sha_after.
# Pure jq filter; oldest-first sort iterates rounds chronologically.
#
# Selection contract: when LATEST_ISSUE_ID and/or LATEST_REVIEW_ID are
# provided, we MUST iterate ALL same-SHA APPROVE candidates and reuse the
# one whose recorded watermarks BOTH match the current values. Breaking
# on the first SHA match would cause this failure mode:
#
#   round 5: APPROVE on sha=X, issue_wm=100, review_wm=50
#   <human comments land; watermarks → 200, 60>
#   round 6: re-APPROVE on sha=X, issue_wm=200, review_wm=60
#   round 7: must skip, but old code found round 5 first, saw stale
#            watermark=100 != 200, returned "proceed" — re-running codex
#            even though round 6 is a perfectly reusable APPROVE.
#
# So: keep scanning. Prefer a same-SHA APPROVE whose BOTH watermarks
# match the supplied LATEST_*_ID values; otherwise (none supplied, or no
# prior round recorded watermarks) fall back to legacy SHA-only reuse
# using the earliest APPROVE.
PRIOR_FILE=""
PRIOR_ROUND=""
PRIOR_BLOCKERS=0
PRIOR_ISSUE_ID=""
PRIOR_REVIEW_ID=""
# Track the earliest same-SHA APPROVE as a fallback for the legacy
# (no-watermark) path so pre-watermark round files keep skipping.
FALLBACK_FILE=""
FALLBACK_ROUND=""
FALLBACK_BLOCKERS=0
FALLBACK_ISSUE_ID=""
FALLBACK_REVIEW_ID=""

# Use a sorted glob so iteration order is stable across filesystems.
# Iterate via while-read against a NUL-safe stream so paths with spaces
# or other shell-meta characters survive intact (paths come from the
# RUN_DIR caller-supplied root). Malformed JSON in any candidate file is
# treated as "not an approval" and skipped silently — corrupt round
# files must never crash the monitor; downstream consumers are
# responsible for surfacing them.
shopt -s nullglob
while IFS= read -r -d '' f; do
  [[ -f "$f" ]] || continue
  ev=$(jq -r '.codex_event // empty' "$f" 2>/dev/null) || ev=""
  sha=$(jq -r '.head_sha_after // empty' "$f" 2>/dev/null) || sha=""
  [[ "$ev" == "APPROVE" && "$sha" == "$HEAD_SHA" ]] || continue

  cand_round=$(jq -r '.round // empty' "$f" 2>/dev/null) || cand_round=""
  cand_blockers=$(jq -r '.blockers_after // 0' "$f" 2>/dev/null) || cand_blockers=0
  # Backward-compat: pre-namespace-split round files only carry
  # last_human_comment_id; treat it as the issue-comment seed (the legacy
  # max() across both endpoints was dominated by issue ids in practice).
  cand_issue_id=$(jq -r '.last_issue_comment_id // .last_human_comment_id // empty' "$f" 2>/dev/null) || cand_issue_id=""
  cand_review_id=$(jq -r '.last_review_comment_id // empty' "$f" 2>/dev/null) || cand_review_id=""
  # Defensive numeric validation: --argjson rejects non-JSON-numeric
  # values, which would crash marker generation below. A corrupt prior
  # round file (e.g. blockers_after: "many", round: null) must not
  # break the monitor — coerce to safe defaults so the marker still
  # writes and reuse can proceed. Empty strings from `// empty` above
  # are normalized here too.
  [[ "$cand_round" =~ ^[0-9]+$ ]] || cand_round=0
  [[ "$cand_blockers" =~ ^[0-9]+$ ]] || cand_blockers=0
  [[ "$cand_issue_id" =~ ^[0-9]+$ ]] || cand_issue_id=""
  [[ "$cand_review_id" =~ ^[0-9]+$ ]] || cand_review_id=""

  # Remember the earliest same-SHA APPROVE for the legacy fallback.
  if [[ -z "$FALLBACK_FILE" ]]; then
    FALLBACK_FILE="$f"
    FALLBACK_ROUND="$cand_round"
    FALLBACK_BLOCKERS="$cand_blockers"
    FALLBACK_ISSUE_ID="$cand_issue_id"
    FALLBACK_REVIEW_ID="$cand_review_id"
  fi

  # Watermark-aware reuse: pick this candidate only if every watermark
  # the caller actually supplied matches its recorded counterpart. A
  # caller that provides only LATEST_ISSUE_ID does not constrain the
  # review-id dimension, so we must NOT silently require the recorded
  # review id to be 0. Continue scanning otherwise so a later same-SHA
  # APPROVE with fresher watermarks can still win.
  ok=true
  if [[ -n "$LATEST_ISSUE_ID" \
        && "${cand_issue_id:-0}" != "$LATEST_ISSUE_ID" ]]; then
    ok=false
  fi
  if [[ -n "$LATEST_REVIEW_ID" \
        && "${cand_review_id:-0}" != "$LATEST_REVIEW_ID" ]]; then
    ok=false
  fi
  if [[ ( -n "$LATEST_ISSUE_ID" || -n "$LATEST_REVIEW_ID" ) \
        && "$ok" == "true" ]]; then
    PRIOR_FILE="$f"
    PRIOR_ROUND="$cand_round"
    PRIOR_BLOCKERS="$cand_blockers"
    PRIOR_ISSUE_ID="$cand_issue_id"
    PRIOR_REVIEW_ID="$cand_review_id"
    break
  fi
done < <(find "$ROUNDS_DIR" -maxdepth 1 -type f -name 'round-*.json' -print0 2>/dev/null | sort -z)

# Resolve which candidate (if any) wins.
if [[ -z "$PRIOR_FILE" ]]; then
  if [[ -z "$FALLBACK_FILE" ]]; then
    # No same-SHA APPROVE at all → nothing to reuse.
    echo "proceed"
    exit 0
  fi
  if [[ -n "$LATEST_ISSUE_ID" || -n "$LATEST_REVIEW_ID" ]]; then
    # Caller supplied watermarks and NO same-SHA APPROVE matched them.
    # That means a fresh human comment (issue or review) has landed
    # since every prior APPROVE on this SHA — codex must re-review.
    echo "proceed"
    exit 0
  fi
  # Legacy path: no watermarks from caller → reuse the earliest match.
  PRIOR_FILE="$FALLBACK_FILE"
  PRIOR_ROUND="$FALLBACK_ROUND"
  PRIOR_BLOCKERS="$FALLBACK_BLOCKERS"
  PRIOR_ISSUE_ID="$FALLBACK_ISSUE_ID"
  PRIOR_REVIEW_ID="$FALLBACK_REVIEW_ID"
fi

# Found a prior APPROVE on this SHA. Emit a marker round-NN.json that
# reuses the prior outcome and clearly notes the reuse so postmortem
# tools can distinguish skipped rounds from genuinely-reviewed rounds.
N=$(printf '%02d' "$NEXT_ROUND")
MARKER="$ROUNDS_DIR/round-$N.json"
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Carry the watermarks forward so a downstream reuse decision (or
# postmortem) can tell whether new human comments have landed since the
# *original* approving round. Use the caller-provided LATEST_*_ID when
# present; otherwise inherit the prior round's recorded watermark.
MARKER_ISSUE_ID="${LATEST_ISSUE_ID:-${PRIOR_ISSUE_ID:-0}}"
MARKER_REVIEW_ID="${LATEST_REVIEW_ID:-${PRIOR_REVIEW_ID:-0}}"
[[ -z "$MARKER_ISSUE_ID" ]] && MARKER_ISSUE_ID=0
[[ -z "$MARKER_REVIEW_ID" ]] && MARKER_REVIEW_ID=0

# Final numeric validation before --argjson. Any of the values below
# could still be non-numeric here (e.g. caller-supplied NEXT_ROUND or
# LATEST_*_ID came in malformed). Coerce or bail to "proceed" so the
# loop falls back to a real codex run rather than silently emitting
# "skip" without writing a marker.
[[ "$NEXT_ROUND" =~ ^[0-9]+$ ]] || { echo "round-pre.sh: NEXT_ROUND not numeric ($NEXT_ROUND); proceeding" >&2; echo "proceed"; exit 0; }
[[ "$PRIOR_BLOCKERS" =~ ^[0-9]+$ ]] || PRIOR_BLOCKERS=0
[[ "$PRIOR_ROUND" =~ ^[0-9]+$ ]] || PRIOR_ROUND=0
[[ "$MARKER_ISSUE_ID" =~ ^[0-9]+$ ]] || MARKER_ISSUE_ID=0
[[ "$MARKER_REVIEW_ID" =~ ^[0-9]+$ ]] || MARKER_REVIEW_ID=0

if ! jq -n \
  --argjson r "$NEXT_ROUND" \
  --arg now "$NOW" \
  --arg sha "$HEAD_SHA" \
  --arg ev "APPROVE" \
  --argjson bl "$PRIOR_BLOCKERS" \
  --argjson prior_round "$PRIOR_ROUND" \
  --argjson iid "$MARKER_ISSUE_ID" \
  --argjson rid "$MARKER_REVIEW_ID" \
  '{round:$r, finished_at:$now,
    head_sha_before:$sha, head_sha_after:$sha,
    codex_event:$ev, blockers_after:$bl,
    last_issue_comment_id:$iid,
    last_review_comment_id:$rid,
    approve_reused_from: $prior_round,
    skipped_codex: true}' \
  > "$MARKER" 2>/dev/null; then
  # Marker generation failed (jq error, disk full, etc.). Without a
  # marker we MUST NOT emit "skip" — the loop would skip codex *and*
  # have no recorded round file, leaving the postmortem trail broken.
  # Force a real codex re-run instead.
  rm -f "$MARKER" 2>/dev/null || true
  echo "round-pre.sh: marker generation failed; forcing codex re-run" >&2
  echo "proceed"
  exit 0
fi

# Defensive: if jq exited 0 but the marker is missing or empty, also
# fall back to "proceed" (e.g. redirection failure on a read-only FS).
if [[ ! -s "$MARKER" ]]; then
  rm -f "$MARKER" 2>/dev/null || true
  echo "round-pre.sh: marker missing/empty after jq; forcing codex re-run" >&2
  echo "proceed"
  exit 0
fi

echo "skip"
exit 0
