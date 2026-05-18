# signals.sh — pure helpers that compute the per-poll "fresh round" signal.
# Sourced, never executed; no shebang on purpose.
#
# The review loop fires a fresh round whenever the PR's externally
# observable state has materially changed. Sourced by `loop.sh`.
# Pure: no gh / git / network calls; callers pass the raw GitHub
# fields in.
#
# Public functions:
#
#   sha256_text <text>
#     Print the lowercase sha256 of <text> (truncated to 16 hex chars to
#     keep meta.json compact). Single-line output, no newline-terminated
#     trailing bytes via printf '%s'.
#
#   pr_body_hash <body>
#     Stable hash of a PR body. Normalizes CRLF to LF so a body edited on
#     GitHub web (CRLF) and locally (LF) compares equal when textually
#     identical. Empty body → fixed sentinel hash of empty string.
#
#   pr_labels_hash <labels_json_array>
#     Stable hash of a labels set. <labels_json_array> is the raw
#     `.labels` field from `gh pr view --json labels` — a JSON array of
#     `{name,...}` objects. Hash is order-independent (labels are a set,
#     not a list).
#
#   signature_diff_reason \
#       <head_now> <head_prev> \
#       <body_now> <body_prev> \
#       <labels_now> <labels_prev> \
#       <issue_id_now> <issue_id_prev> \
#       <review_id_now> <review_id_prev> \
#       <inbox_present>
#     Print a single short reason string naming which signal fired, or
#     the empty string if no signal changed. Reasons are stable tokens
#     (used in log lines and asserted by tests):
#       - "head changed"
#       - "body changed"
#       - "labels changed"
#       - "issue comment"
#       - "review comment"
#       - "inbox prompt"
#     When multiple signals change in one tick, the first one in the
#     priority order above wins — every signal still triggers exactly
#     one round, so picking one stable token avoids double-firing.

set -uo pipefail

sha256_text() {
  # printf '%s' avoids the implicit trailing newline of echo which would
  # otherwise make `sha256_text ""` differ from a true-empty hash.
  # Prefer GNU `sha256sum`; fall back to BSD/macOS `shasum -a 256`.
  local out
  if command -v sha256sum >/dev/null 2>&1; then
    out=$(printf '%s' "$1" | sha256sum)
  else
    out=$(printf '%s' "$1" | shasum -a 256)
  fi
  printf '%s' "$out" | cut -c1-16
}

pr_body_hash() {
  local body="${1-}"
  # Normalize CRLF → LF so web-edit and CLI-edit bodies compare equal.
  body="${body//$'\r'/}"
  sha256_text "$body"
}

pr_labels_hash() {
  local labels_json="${1:-[]}"
  # Sort label names so the hash is order-independent. Serialize as
  # compact JSON (not join(",")) so a label name containing a comma
  # cannot collide with two labels split on commas. `jq -e` would exit
  # non-zero on empty input; default to "[]" so the empty-labels case
  # yields a stable, deterministic hash.
  #
  # ``agent-stuck`` is loop-owned (round-post.sh applies it); excluding
  # it keeps the loop from observing its own write as a fresh label
  # signal on the next poll.
  local sorted
  sorted=$(printf '%s' "$labels_json" \
    | jq -c 'if type=="array" then [.[].name | select(. != "agent-stuck")] | sort else [] end' \
    2>/dev/null) || sorted="[]"
  sha256_text "$sorted"
}

signature_diff_reason() {
  local head_now="$1" head_prev="$2"
  local body_now="$3" body_prev="$4"
  local labels_now="$5" labels_prev="$6"
  local issue_now="$7" issue_prev="$8"
  local review_now="$9" review_prev="${10}"
  local inbox_present="${11:-0}"

  if [[ "$head_now" != "$head_prev" ]]; then
    printf '%s' "head changed"
    return 0
  fi
  if [[ "$body_now" != "$body_prev" && -n "$body_prev" ]]; then
    printf '%s' "body changed"
    return 0
  fi
  if [[ "$labels_now" != "$labels_prev" && -n "$labels_prev" ]]; then
    printf '%s' "labels changed"
    return 0
  fi
  if [[ "$issue_now" != "$issue_prev" ]]; then
    printf '%s' "issue comment"
    return 0
  fi
  if [[ "$review_now" != "$review_prev" ]]; then
    printf '%s' "review comment"
    return 0
  fi
  if [[ "$inbox_present" -eq 1 ]]; then
    printf '%s' "inbox prompt"
    return 0
  fi
  printf '%s' ""
}
