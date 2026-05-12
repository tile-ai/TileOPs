#!/usr/bin/env bash
# Unit tests for signals.sh — the helpers behind the review-loop fresh-
# round trigger. Pure bash; runs without GitHub or network.
#
# Run: bash .claude/skills/review-tileops/tests/test_review_signals.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../signals.sh
source "$HERE/../signals.sh"

FAILS=0
PASSES=0

assert_eq() {
  local name="$1" want="$2" got="$3"
  if [[ "$want" == "$got" ]]; then
    PASSES=$((PASSES + 1))
    printf '  ok   %s\n' "$name"
  else
    FAILS=$((FAILS + 1))
    printf '  FAIL %s\n    want: %q\n    got:  %q\n' "$name" "$want" "$got"
  fi
}

# ---- pr_body_hash ----------------------------------------------------------

H_EMPTY=$(pr_body_hash "")
H_HELLO=$(pr_body_hash "hello world")
assert_eq "body_hash empty != hello" 1 \
  "$([[ "$H_EMPTY" != "$H_HELLO" ]] && echo 1 || echo 0)"

H_LF=$(pr_body_hash $'line1\nline2')
H_CRLF=$(pr_body_hash $'line1\r\nline2')
assert_eq "body_hash CRLF normalizes to LF" "$H_LF" "$H_CRLF"

H_HELLO2=$(pr_body_hash "hello world")
assert_eq "body_hash deterministic" "$H_HELLO" "$H_HELLO2"

# ---- pr_labels_hash --------------------------------------------------------

L1=$(pr_labels_hash '[{"name":"bug"},{"name":"skill"}]')
L2=$(pr_labels_hash '[{"name":"skill"},{"name":"bug"}]')
assert_eq "labels_hash order-independent" "$L1" "$L2"

L3=$(pr_labels_hash '[{"name":"bug"}]')
assert_eq "labels_hash differs when set differs" 1 \
  "$([[ "$L1" != "$L3" ]] && echo 1 || echo 0)"

L_EMPTY1=$(pr_labels_hash '[]')
L_EMPTY2=$(pr_labels_hash '')
assert_eq "labels_hash empty array == empty input" "$L_EMPTY1" "$L_EMPTY2"

# ---- signature_diff_reason -------------------------------------------------

# helper: call with positional args matching the function signature.
diff_reason() {
  signature_diff_reason "$@"
}

# No change at all → empty reason.
R=$(diff_reason "sha1" "sha1" "bh1" "bh1" "lh1" "lh1" 42 42 99 99 0)
assert_eq "no change → empty" "" "$R"

# HEAD changed wins over everything else.
R=$(diff_reason "sha2" "sha1" "bh2" "bh1" "lh2" "lh1" 99 42 100 99 1)
assert_eq "head wins priority" "head changed" "$R"

# Body changed (head unchanged, prior hash known).
R=$(diff_reason "sha1" "sha1" "bh2" "bh1" "lh1" "lh1" 42 42 99 99 0)
assert_eq "body changed fires" "body changed" "$R"

# Body changed but no prior hash (first-time-after-upgrade) → no fire.
R=$(diff_reason "sha1" "sha1" "bh2" "" "lh1" "lh1" 42 42 99 99 0)
assert_eq "body unset → no fire" "" "$R"

# Labels changed.
R=$(diff_reason "sha1" "sha1" "bh1" "bh1" "lh2" "lh1" 42 42 99 99 0)
assert_eq "labels changed fires" "labels changed" "$R"

# Issue comment delta.
R=$(diff_reason "sha1" "sha1" "bh1" "bh1" "lh1" "lh1" 50 42 99 99 0)
assert_eq "issue comment fires" "issue comment" "$R"

# Review (inline) comment delta — AC-2: author reply on inline thread.
R=$(diff_reason "sha1" "sha1" "bh1" "bh1" "lh1" "lh1" 42 42 105 99 0)
assert_eq "review comment fires" "review comment" "$R"

# Inbox prompt alone.
R=$(diff_reason "sha1" "sha1" "bh1" "bh1" "lh1" "lh1" 42 42 99 99 1)
assert_eq "inbox prompt fires" "inbox prompt" "$R"

# AC-3 regression: empty arguments / unchanged state stays empty.
R=$(diff_reason "" "" "" "" "" "" 0 0 0 0 0)
assert_eq "all-zero state → empty" "" "$R"

# ---- report ----------------------------------------------------------------

printf '\n%d passed, %d failed\n' "$PASSES" "$FAILS"
exit $(( FAILS > 0 ? 1 : 0 ))
