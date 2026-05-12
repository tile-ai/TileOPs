#!/usr/bin/env bash
# Integration-ish test for the loop.sh idle decision. Stubs the gh /
# git / codex calls and runs the loop body's signal computation +
# branching, then asserts the log line names the right trigger.
#
# Run: bash .claude/skills/review-tileops/tests/test_review_idle_decision.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../signals.sh
source "$HERE/../signals.sh"

FAILS=0
PASSES=0

assert_contains() {
  local name="$1" needle="$2" haystack="$3"
  if [[ "$haystack" == *"$needle"* ]]; then
    PASSES=$((PASSES + 1))
    printf '  ok   %s\n' "$name"
  else
    FAILS=$((FAILS + 1))
    printf '  FAIL %s\n    needle: %q\n    in:     %s\n' "$name" "$needle" "$haystack"
  fi
}

# Re-create the loop's per-poll decision inline. This mirrors the
# branching logic added to loop.sh; if we ever refactor that branch
# out into a helper, swap the inline form for a function call.
decide() {
  local round="$1"
  local head_now="$2" head_prev="$3"
  local body_now="$4" body_prev="$5"
  local labels_now="$6" labels_prev="$7"
  local issue_now="$8" issue_prev="$9"
  local review_now="${10}" review_prev="${11}"
  local inbox_present="${12}"
  # Optional: simulate the APPROVE convergence branch. When provided,
  # last_event=APPROVE + gh_state=APPROVED triggers converge-vs-rereview
  # based on the same TRIGGER_REASON signature used by the idle path.
  local last_event="${13:-DISMISSED}"
  local gh_state="${14:-DISMISSED}"

  local trigger_reason
  trigger_reason=$(signature_diff_reason \
    "$head_now" "$head_prev" \
    "$body_now" "$body_prev" \
    "$labels_now" "$labels_prev" \
    "$issue_now" "$issue_prev" \
    "$review_now" "$review_prev" \
    "$inbox_present")

  local head_unchanged=0
  [[ "$head_now" == "$head_prev" ]] && head_unchanged=1

  # APPROVE convergence guard: reuses TRIGGER_REASON so any fresh
  # signal (body / labels / comments / head / inbox) blocks convergence.
  if [[ "$last_event" == "APPROVE" ]]; then
    if [[ "$gh_state" == "APPROVED" && -z "$trigger_reason" ]]; then
      echo "CONVERGE"
      return 0
    fi
    # Fall through to fresh review.
    local fired_a="${trigger_reason:-head changed}"
    echo "FIRE: $fired_a"
    return 0
  fi

  if [[ "$round" -gt 0 && "$head_unchanged" -eq 1 \
        && "$inbox_present" -eq 0 && -z "$trigger_reason" ]]; then
    echo "IDLE"
    return 0
  fi

  local fired
  if [[ "$round" -eq 0 ]]; then
    fired="first round"
  else
    fired="${trigger_reason:-head changed}"
  fi
  echo "FIRE: $fired"
}

# AC-1: body edit on unchanged HEAD → fires within one poll.
R=$(decide 3 "shaX" "shaX" "bodyHASH2" "bodyHASH1" "lh1" "lh1" 42 42 99 99 0)
assert_contains "AC-1 body edit fires" "FIRE: body changed" "$R"

# AC-2: new author reply (inline review comment) on unchanged HEAD → fires.
R=$(decide 3 "shaX" "shaX" "bh1" "bh1" "lh1" "lh1" 42 42 105 99 0)
assert_contains "AC-2 review comment fires" "FIRE: review comment" "$R"

# AC-2b: top-level issue comment.
R=$(decide 3 "shaX" "shaX" "bh1" "bh1" "lh1" "lh1" 55 42 99 99 0)
assert_contains "AC-2b issue comment fires" "FIRE: issue comment" "$R"

# AC-3: no observable change → idle.
R=$(decide 3 "shaX" "shaX" "bh1" "bh1" "lh1" "lh1" 42 42 99 99 0)
assert_contains "AC-3 no change → idle" "IDLE" "$R"

# AC-3b: ROUND==0 always fires (first poll, no prior state).
R=$(decide 0 "shaX" "" "bh1" "" "lh1" "" 42 0 99 0 0)
assert_contains "AC-3b first round always fires" "FIRE: first round" "$R"

# Existing: HEAD change still fires (regression guard for AC-5).
R=$(decide 3 "shaY" "shaX" "bh1" "bh1" "lh1" "lh1" 42 42 99 99 0)
assert_contains "AC-5 head sha change still fires" "FIRE: head changed" "$R"

# Existing: inbox prompt still fires.
R=$(decide 3 "shaX" "shaX" "bh1" "bh1" "lh1" "lh1" 42 42 99 99 1)
assert_contains "AC-5 inbox prompt still fires" "FIRE: inbox prompt" "$R"

# Labels change fires.
R=$(decide 3 "shaX" "shaX" "bh1" "bh1" "lh2" "lh1" 42 42 99 99 0)
assert_contains "labels change fires" "FIRE: labels changed" "$R"

# No prior body hash recorded (legacy meta.json upgrade) → body diff does
# NOT spuriously fire on first post-upgrade poll.
R=$(decide 3 "shaX" "shaX" "newbody" "" "lh1" "lh1" 42 42 99 99 0)
assert_contains "legacy meta no spurious body fire" "IDLE" "$R"

# APPROVE convergence guard: prior round was APPROVE and GH still shows
# APPROVED. With no fresh signal → converge. With a body edit on the
# same HEAD → must fall through to a fresh review pass, NOT converge.

# Baseline: nothing changed since approval → converge.
R=$(decide 3 "shaX" "shaX" "bh1" "bh1" "lh1" "lh1" 42 42 99 99 0 APPROVE APPROVED)
assert_contains "APPROVE + no signal → converge" "CONVERGE" "$R"

# Bug regression: body edit after APPROVE → fresh review, NOT converge.
R=$(decide 3 "shaX" "shaX" "bodyHASH2" "bodyHASH1" "lh1" "lh1" 42 42 99 99 0 APPROVE APPROVED)
assert_contains "APPROVE + body edit → fresh review" "FIRE: body changed" "$R"

# Labels change after APPROVE → fresh review.
R=$(decide 3 "shaX" "shaX" "bh1" "bh1" "lh2" "lh1" 42 42 99 99 0 APPROVE APPROVED)
assert_contains "APPROVE + labels change → fresh review" "FIRE: labels changed" "$R"

# New review comment after APPROVE → fresh review (pre-existing guard).
R=$(decide 3 "shaX" "shaX" "bh1" "bh1" "lh1" "lh1" 42 42 105 99 0 APPROVE APPROVED)
assert_contains "APPROVE + review comment → fresh review" "FIRE: review comment" "$R"

printf '\n%d passed, %d failed\n' "$PASSES" "$FAILS"
exit $(( FAILS > 0 ? 1 : 0 ))
