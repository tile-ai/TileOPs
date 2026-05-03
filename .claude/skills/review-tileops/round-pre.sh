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
#   RUN_DIR          — review/ root for this loop run; must contain rounds/.
#   NEXT_ROUND       — integer round number about to run (e.g. 5).
#   HEAD_SHA         — full git sha of the PR HEAD this round would review.
#   LATEST_HUMAN_ID  — current max id of any non-reviewer comment on the PR.
#                      Optional; when provided, the reuse decision also
#                      requires the prior approving round's recorded
#                      ``last_human_comment_id`` to match. If a NEW human
#                      comment landed on the same SHA after the prior
#                      APPROVE, this hook returns "proceed" so codex re-
#                      reviews and ingests the comment. When absent or
#                      empty, this guard is skipped (legacy behavior, used
#                      by tests that pre-date the watermark field).
#
# Positional fallback: round-pre.sh <RUN_DIR> <NEXT_ROUND> <HEAD_SHA> [LATEST_HUMAN_ID]
#
# Output:
#   - exit 0 + stdout "skip"     → loop should skip codex; round-NN.json is
#                                  already written by this script.
#   - exit 0 + stdout "proceed"  → no prior APPROVE on this SHA; loop runs
#                                  codex normally.
#   - non-zero exit              → unexpected error; loop should fail-fast.
#
# Determinism: pure file scan. No gh / git / network. No LLM.
set -euo pipefail

RUN_DIR="${RUN_DIR:-${1:-}}"
NEXT_ROUND="${NEXT_ROUND:-${2:-}}"
HEAD_SHA="${HEAD_SHA:-${3:-}}"
LATEST_HUMAN_ID="${LATEST_HUMAN_ID:-${4:-}}"

if [[ -z "$RUN_DIR" || -z "$NEXT_ROUND" || -z "$HEAD_SHA" ]]; then
  echo "round-pre.sh: missing args; usage: RUN_DIR=... NEXT_ROUND=... HEAD_SHA=... round-pre.sh" >&2
  exit 2
fi

ROUNDS_DIR="$RUN_DIR/rounds"
if [[ ! -d "$ROUNDS_DIR" ]]; then
  # No prior rounds → nothing to reuse; loop proceeds normally.
  echo "proceed"
  exit 0
fi

# Scan prior round-*.json files for an APPROVE on the same head_sha_after.
# Pure jq filter; oldest-first sort so "prior" matches the earliest
# approving round (deterministic when multiple match).
PRIOR_FILE=""
PRIOR_ROUND=""
PRIOR_BLOCKERS=0
PRIOR_HUMAN_ID=""

# Use a sorted glob so iteration order is stable across filesystems.
shopt -s nullglob
for f in $(printf '%s\n' "$ROUNDS_DIR"/round-*.json | sort); do
  [[ -f "$f" ]] || continue
  ev=$(jq -r '.codex_event // empty' "$f" 2>/dev/null || echo "")
  sha=$(jq -r '.head_sha_after // empty' "$f" 2>/dev/null || echo "")
  if [[ "$ev" == "APPROVE" && "$sha" == "$HEAD_SHA" ]]; then
    PRIOR_FILE="$f"
    PRIOR_ROUND=$(jq -r '.round // empty' "$f")
    PRIOR_BLOCKERS=$(jq -r '.blockers_after // 0' "$f")
    PRIOR_HUMAN_ID=$(jq -r '.last_human_comment_id // empty' "$f")
    break
  fi
done

if [[ -z "$PRIOR_FILE" ]]; then
  echo "proceed"
  exit 0
fi

# Human-comment watermark guard. The same-SHA reuse path must NOT skip
# codex when a new non-reviewer comment has landed since the prior
# APPROVE — that comment is fresh feedback the loop is contractually
# obligated to surface. Compare watermarks only when the caller passed
# LATEST_HUMAN_ID *and* the prior round file recorded its own
# last_human_comment_id; otherwise fall back to legacy SHA-only reuse so
# pre-watermark round files (and the existing test suite) keep working.
if [[ -n "$LATEST_HUMAN_ID" && -n "$PRIOR_HUMAN_ID" \
      && "$LATEST_HUMAN_ID" != "$PRIOR_HUMAN_ID" ]]; then
  echo "proceed"
  exit 0
fi

# Found a prior APPROVE on this SHA. Emit a marker round-NN.json that
# reuses the prior outcome and clearly notes the reuse so postmortem
# tools can distinguish skipped rounds from genuinely-reviewed rounds.
N=$(printf '%02d' "$NEXT_ROUND")
MARKER="$ROUNDS_DIR/round-$N.json"
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Carry the human watermark forward so a downstream reuse decision (or
# postmortem) can tell whether new human comments have landed since the
# *original* approving round. Use the caller-provided LATEST_HUMAN_ID
# when present; otherwise inherit the prior round's recorded watermark.
MARKER_HUMAN_ID="${LATEST_HUMAN_ID:-${PRIOR_HUMAN_ID:-0}}"
[[ -z "$MARKER_HUMAN_ID" ]] && MARKER_HUMAN_ID=0

jq -n \
  --argjson r "$NEXT_ROUND" \
  --arg now "$NOW" \
  --arg sha "$HEAD_SHA" \
  --arg ev "APPROVE" \
  --argjson bl "$PRIOR_BLOCKERS" \
  --argjson prior_round "${PRIOR_ROUND:-0}" \
  --argjson hid "$MARKER_HUMAN_ID" \
  '{round:$r, finished_at:$now,
    head_sha_before:$sha, head_sha_after:$sha,
    codex_event:$ev, blockers_after:$bl,
    last_human_comment_id:$hid,
    approve_reused_from: $prior_round,
    skipped_codex: true}' \
  > "$MARKER"

echo "skip"
exit 0
