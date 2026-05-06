#!/usr/bin/env bash
# Fixture tests for auto-resolve-stale.sh.
#
# Drives the classifier in --dry-run against fixtures under fixtures/
# and asserts the emitted action plan + side-effect artifacts.
#
# Exit 0: all cases pass.
# Exit 1: any case fails (full diff printed).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$HERE")"
CLASSIFIER="$SKILL_DIR/auto-resolve-stale.sh"
BOTS="$SKILL_DIR/known-bots.json"
FIX="$HERE/fixtures"

[[ -x "$CLASSIFIER" ]] || { echo "FAIL: $CLASSIFIER not found / not executable" >&2; exit 1; }
[[ -f "$BOTS" ]] || { echo "FAIL: $BOTS not found" >&2; exit 1; }

PASS=0
FAIL=0

run_case() {
  local name="$1" fixture="$2" round="$3"
  local expected_resolve="$4" expected_unknown="$5" expected_skip="$6"
  local expected_unknown_artifact="$7"  # path to expected artifact json (or "none")
  local tmp; tmp=$(mktemp -d)
  local plan
  if ! plan=$("$CLASSIFIER" \
        --threads "$fixture" \
        --bots "$BOTS" \
        --run-dir "$tmp" \
        --round "$round" \
        --dry-run); then
    echo "FAIL [$name]: classifier exited non-zero" >&2
    FAIL=$((FAIL+1)); return
  fi
  local got_resolve got_unknown got_skip
  got_resolve=$(printf '%s' "$plan" | jq '[.resolve[].thread_id]')
  got_unknown=$(printf '%s' "$plan" | jq '[.unknown_bot_like[].thread_id]')
  got_skip=$(printf '%s' "$plan" | jq '[.skip[].thread_id]')

  if [[ "$got_resolve" != "$expected_resolve" ]]; then
    echo "FAIL [$name]: resolve mismatch" >&2
    echo "  expected: $expected_resolve" >&2
    echo "  got:      $got_resolve" >&2
    FAIL=$((FAIL+1)); return
  fi
  if [[ "$got_unknown" != "$expected_unknown" ]]; then
    echo "FAIL [$name]: unknown_bot_like mismatch" >&2
    echo "  expected: $expected_unknown" >&2
    echo "  got:      $got_unknown" >&2
    FAIL=$((FAIL+1)); return
  fi
  if [[ "$got_skip" != "$expected_skip" ]]; then
    echo "FAIL [$name]: skip mismatch" >&2
    echo "  expected: $expected_skip" >&2
    echo "  got:      $got_skip" >&2
    FAIL=$((FAIL+1)); return
  fi

  # Reply text contract: every resolve entry must carry the literal text.
  local reply_texts
  reply_texts=$(printf '%s' "$plan" | jq -r '[.resolve[].reply] | unique | .[]')
  if [[ -n "$reply_texts" && "$reply_texts" != "Not assessed on latest HEAD" ]]; then
    echo "FAIL [$name]: reply text mismatch — got '$reply_texts'" >&2
    FAIL=$((FAIL+1)); return
  fi

  # Artifact contract.
  local artifact="$tmp/round-${round}.unknown-bot-like.json"
  if [[ "$expected_unknown_artifact" == "none" ]]; then
    if [[ -f "$artifact" ]]; then
      echo "FAIL [$name]: unexpected artifact at $artifact" >&2
      cat "$artifact" >&2
      FAIL=$((FAIL+1)); return
    fi
  else
    if [[ ! -f "$artifact" ]]; then
      echo "FAIL [$name]: missing artifact $artifact" >&2
      FAIL=$((FAIL+1)); return
    fi
    local got_logins
    got_logins=$(jq '[.[].login]' "$artifact")
    if [[ "$got_logins" != "$expected_unknown_artifact" ]]; then
      echo "FAIL [$name]: artifact logins mismatch" >&2
      echo "  expected: $expected_unknown_artifact" >&2
      echo "  got:      $got_logins" >&2
      FAIL=$((FAIL+1)); return
    fi
  fi

  echo "PASS [$name]"
  PASS=$((PASS+1))
  rm -rf "$tmp"
}

# Case 1: known bot, stale anchor → auto-resolve, reply text matches.
run_case "stale-bot-thread" \
  "$FIX/stale-bot-thread.json" "01" \
  '[
  "PRT_kwDO_stale_bot"
]' \
  '[]' \
  '[]' \
  "none"

# Case 2: human reviewer, stale anchor → no reply, no resolve, recorded as skip.
run_case "stale-human-thread" \
  "$FIX/stale-human-thread.json" "02" \
  '[]' \
  '[]' \
  '[
  "PRT_kwDO_stale_human"
]' \
  "none"

# Case 3: bot-like login not on list → recorded to unknown-bot-like.json.
run_case "unknown-bot-thread" \
  "$FIX/unknown-bot-thread.json" "03" \
  '[]' \
  '[
  "PRT_kwDO_unknown_bot"
]' \
  '[]' \
  '[
  "shiny-new-reviewer[bot]"
]'

# Case 4: known bot anchored at current HEAD → no action at all.
run_case "current-head-bot-thread" \
  "$FIX/current-head-bot-thread.json" "04" \
  '[]' \
  '[]' \
  '[
  "PRT_kwDO_current_head_bot"
]' \
  "none"

echo
echo "Results: $PASS passed, $FAIL failed."
[[ "$FAIL" -eq 0 ]] || exit 1
