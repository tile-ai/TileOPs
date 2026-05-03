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
LATEST_HUMAN_ID="${LATEST_HUMAN_ID:-${4:-}}"

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
# Pure jq filter; oldest-first sort so "prior" matches the earliest
# approving round (deterministic when multiple match).
PRIOR_FILE=""
PRIOR_ROUND=""
PRIOR_BLOCKERS=0
PRIOR_HUMAN_ID=""

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
  if [[ "$ev" == "APPROVE" && "$sha" == "$HEAD_SHA" ]]; then
    PRIOR_FILE="$f"
    PRIOR_ROUND=$(jq -r '.round // empty' "$f" 2>/dev/null) || PRIOR_ROUND=""
    PRIOR_BLOCKERS=$(jq -r '.blockers_after // 0' "$f" 2>/dev/null) || PRIOR_BLOCKERS=0
    PRIOR_HUMAN_ID=$(jq -r '.last_human_comment_id // empty' "$f" 2>/dev/null) || PRIOR_HUMAN_ID=""
    # Defensive numeric validation: --argjson rejects non-JSON-numeric
    # values, which would crash marker generation below. A corrupt prior
    # round file (e.g. blockers_after: "many", round: null) must not
    # break the monitor — coerce to safe defaults so the marker still
    # writes and reuse can proceed. Empty strings from `// empty` above
    # are normalized here too.
    [[ "$PRIOR_ROUND" =~ ^[0-9]+$ ]] || PRIOR_ROUND=0
    [[ "$PRIOR_BLOCKERS" =~ ^[0-9]+$ ]] || PRIOR_BLOCKERS=0
    [[ "$PRIOR_HUMAN_ID" =~ ^[0-9]+$ ]] || PRIOR_HUMAN_ID=""
    break
  fi
done < <(find "$ROUNDS_DIR" -maxdepth 1 -type f -name 'round-*.json' -print0 2>/dev/null | sort -z)

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

# Final numeric validation before --argjson. Any of the four values
# below could still be non-numeric here (e.g. caller-supplied
# NEXT_ROUND or LATEST_HUMAN_ID came in malformed). Coerce or bail to
# "proceed" so the loop falls back to a real codex run rather than
# silently emitting "skip" without writing a marker.
[[ "$NEXT_ROUND" =~ ^[0-9]+$ ]] || { echo "round-pre.sh: NEXT_ROUND not numeric ($NEXT_ROUND); proceeding" >&2; echo "proceed"; exit 0; }
[[ "$PRIOR_BLOCKERS" =~ ^[0-9]+$ ]] || PRIOR_BLOCKERS=0
[[ "$PRIOR_ROUND" =~ ^[0-9]+$ ]] || PRIOR_ROUND=0
[[ "$MARKER_HUMAN_ID" =~ ^[0-9]+$ ]] || MARKER_HUMAN_ID=0

if ! jq -n \
  --argjson r "$NEXT_ROUND" \
  --arg now "$NOW" \
  --arg sha "$HEAD_SHA" \
  --arg ev "APPROVE" \
  --argjson bl "$PRIOR_BLOCKERS" \
  --argjson prior_round "$PRIOR_ROUND" \
  --argjson hid "$MARKER_HUMAN_ID" \
  '{round:$r, finished_at:$now,
    head_sha_before:$sha, head_sha_after:$sha,
    codex_event:$ev, blockers_after:$bl,
    last_human_comment_id:$hid,
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
