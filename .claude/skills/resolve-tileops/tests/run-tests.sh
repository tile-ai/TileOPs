#!/usr/bin/env bash
# Fixture tests for auto-resolve-stale.sh.
#
# Drives the classifier in --dry-run against fixtures under fixtures/
# and asserts the emitted action plan + side-effect artifacts. On a
# mismatch each case prints the expected and actual JSON strings (or
# the offending artifact's contents) so the failing comparison can be
# read off the test log directly.
#
# Exit 0: all cases pass.
# Exit 1: any case fails.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$HERE")"
CLASSIFIER="$SKILL_DIR/auto-resolve-stale.sh"
BOTS="$SKILL_DIR/known-bots.json"
FIX="$HERE/fixtures"

[[ -x "$CLASSIFIER" ]] || { echo "FAIL: $CLASSIFIER not found / not executable" >&2; exit 1; }
[[ -f "$BOTS" ]] || { echo "FAIL: $BOTS not found" >&2; exit 1; }

# Reply text contract — kept here as the single source the assertion
# checks against. The classifier owns the literal string; this constant
# must match auto-resolve-stale.sh's REPLY_TEXT exactly.
STALE_REPLY_TEXT="Not assessed on latest HEAD"

PASS=0
FAIL=0

run_case() {
  local name="$1" fixture="$2" round="$3"
  local expected_resolve="$4" expected_unknown="$5" expected_skip="$6"
  # expected_unknown_artifact: either the literal string "none" (expect
  # no artifact file) or a JSON-array string (compared against the
  # artifact's [.[].login] projection).
  local expected_unknown_artifact="$7"
  local tmp; tmp=$(mktemp -d)
  # Cleanup on every return path, including failures. Without the trap
  # the early `return` branches below would leak the tmpdir.
  trap 'rm -rf "$tmp"' RETURN
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
  if [[ -n "$reply_texts" && "$reply_texts" != "$STALE_REPLY_TEXT" ]]; then
    echo "FAIL [$name]: reply text mismatch — got '$reply_texts' want '$STALE_REPLY_TEXT'" >&2
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

# Case 5: GitHub App with a bare prefix (no -bot / -reviewer / -code-assist
# before [bot]) is still classified as bot-like. The literal "[bot]"
# suffix is reserved by GitHub for Apps and is sufficient on its own.
run_case "unknown-bare-bot-thread" \
  "$FIX/unknown-bare-bot-thread.json" "05" \
  '[]' \
  '[
  "PRT_kwDO_unknown_bare_bot"
]' \
  '[]' \
  '[
  "random-app[bot]"
]'

# Case 6: live-mode reply failure → the thread must NOT be marked resolved.
# Drives the classifier without --dry-run against an injected mock gh
# that fails the addPullRequestReviewThreadReply mutation. The thread id
# must show up in executed.reply_failed and must NOT show up in
# executed.resolved.
run_reply_failure_case() {
  local name="reply-failure-leaves-thread-open"
  local tmp; tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN
  local mockbin="$tmp/bin"
  mkdir -p "$mockbin"
  cat > "$mockbin/gh" <<'MOCK'
#!/usr/bin/env bash
# Mock gh: fail any GraphQL mutation that posts a reply, succeed
# everything else. Detect the reply mutation by scanning the args for
# the addPullRequestReviewThreadReply marker.
for a in "$@"; do
  case "$a" in
    *addPullRequestReviewThreadReply*) exit 1 ;;
  esac
done
exit 0
MOCK
  chmod +x "$mockbin/gh"
  local plan
  if ! plan=$(GH_BIN="$mockbin/gh" "$CLASSIFIER" \
        --threads "$FIX/stale-bot-thread.json" \
        --bots "$BOTS" \
        --run-dir "$tmp" \
        --round "06" 2>/dev/null); then
    echo "FAIL [$name]: classifier exited non-zero" >&2
    FAIL=$((FAIL+1)); return
  fi
  local resolved reply_failed
  resolved=$(printf '%s' "$plan" | jq -c '.executed.resolved')
  reply_failed=$(printf '%s' "$plan" | jq -c '.executed.reply_failed')
  if [[ "$resolved" != '[]' ]]; then
    echo "FAIL [$name]: thread was resolved despite reply failure: $resolved" >&2
    FAIL=$((FAIL+1)); return
  fi
  if [[ "$reply_failed" != '["PRT_kwDO_stale_bot"]' ]]; then
    echo "FAIL [$name]: reply_failed list mismatch: $reply_failed" >&2
    FAIL=$((FAIL+1)); return
  fi
  echo "PASS [$name]"
  PASS=$((PASS+1))
}
run_reply_failure_case

# Case 7: live-mode happy path with mock gh → thread ends up in
# executed.resolved. Guards against regressions where the new gating
# logic accidentally short-circuits the success path too.
run_reply_success_case() {
  local name="reply-success-resolves-thread"
  local tmp; tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN
  local mockbin="$tmp/bin"
  mkdir -p "$mockbin"
  cat > "$mockbin/gh" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK
  chmod +x "$mockbin/gh"
  local plan
  if ! plan=$(GH_BIN="$mockbin/gh" "$CLASSIFIER" \
        --threads "$FIX/stale-bot-thread.json" \
        --bots "$BOTS" \
        --run-dir "$tmp" \
        --round "07" 2>/dev/null); then
    echo "FAIL [$name]: classifier exited non-zero" >&2
    FAIL=$((FAIL+1)); return
  fi
  local resolved reply_failed
  resolved=$(printf '%s' "$plan" | jq -c '.executed.resolved')
  reply_failed=$(printf '%s' "$plan" | jq -c '.executed.reply_failed')
  if [[ "$resolved" != '["PRT_kwDO_stale_bot"]' ]]; then
    echo "FAIL [$name]: resolved list mismatch: $resolved" >&2
    FAIL=$((FAIL+1)); return
  fi
  if [[ "$reply_failed" != '[]' ]]; then
    echo "FAIL [$name]: unexpected reply_failed: $reply_failed" >&2
    FAIL=$((FAIL+1)); return
  fi
  echo "PASS [$name]"
  PASS=$((PASS+1))
}
run_reply_success_case

# Case 8: known bot whose first comment has no commit oid → skipped with
# reason known_bot_missing_commit_oid (NOT known_bot_at_head). The
# missing oid means we can't tell whether the comment is anchored to
# HEAD, so the safe action is to leave it for human triage.
run_missing_oid_case() {
  local name="known-bot-missing-commit-oid"
  local tmp; tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN
  local plan
  if ! plan=$("$CLASSIFIER" \
        --threads "$FIX/known-bot-missing-oid-thread.json" \
        --bots "$BOTS" \
        --run-dir "$tmp" \
        --round "08" \
        --dry-run); then
    echo "FAIL [$name]: classifier exited non-zero" >&2
    FAIL=$((FAIL+1)); return
  fi
  local got_skip_reason
  got_skip_reason=$(printf '%s' "$plan" \
    | jq -r '[.skip[] | select(.thread_id=="PRT_kwDO_missing_oid_bot") | .reason] | first // ""')
  if [[ "$got_skip_reason" != "known_bot_missing_commit_oid" ]]; then
    echo "FAIL [$name]: skip reason mismatch — got '$got_skip_reason' want 'known_bot_missing_commit_oid'" >&2
    FAIL=$((FAIL+1)); return
  fi
  local got_resolve
  got_resolve=$(printf '%s' "$plan" | jq -c '[.resolve[].thread_id]')
  if [[ "$got_resolve" != '[]' ]]; then
    echo "FAIL [$name]: resolve must be empty for missing-oid bot, got $got_resolve" >&2
    FAIL=$((FAIL+1)); return
  fi
  echo "PASS [$name]"
  PASS=$((PASS+1))
}
run_missing_oid_case

echo
echo "Results: $PASS passed, $FAIL failed."
[[ "$FAIL" -eq 0 ]] || exit 1
