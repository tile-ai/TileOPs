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
#   RUN_DIR       — review/ root for this loop run; must contain rounds/.
#   NEXT_ROUND    — integer round number about to run (e.g. 5).
#   HEAD_SHA      — full git sha of the PR HEAD this round would review.
#
# Positional fallback: round-pre.sh <RUN_DIR> <NEXT_ROUND> <HEAD_SHA>
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
    break
  fi
done

if [[ -z "$PRIOR_FILE" ]]; then
  echo "proceed"
  exit 0
fi

# Found a prior APPROVE on this SHA. Emit a marker round-NN.json that
# reuses the prior outcome and clearly notes the reuse so postmortem
# tools can distinguish skipped rounds from genuinely-reviewed rounds.
N=$(printf '%02d' "$NEXT_ROUND")
MARKER="$ROUNDS_DIR/round-$N.json"
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

jq -n \
  --argjson r "$NEXT_ROUND" \
  --arg now "$NOW" \
  --arg sha "$HEAD_SHA" \
  --arg ev "APPROVE" \
  --argjson bl "$PRIOR_BLOCKERS" \
  --argjson prior_round "${PRIOR_ROUND:-0}" \
  '{round:$r, finished_at:$now,
    head_sha_before:$sha, head_sha_after:$sha,
    codex_event:$ev, blockers_after:$bl,
    approve_reused_from: $prior_round,
    skipped_codex: true}' \
  > "$MARKER"

echo "skip"
exit 0
