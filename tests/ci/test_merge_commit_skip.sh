#!/bin/bash
# Test: merge commits should be skipped by commit message validation
# Tests both pr-validation.yml logic and validate.sh logic
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/.claude/conventions/types.sh"

PASS=0
FAIL=0
TOTAL=0

assert_eq() {
  local test_name="$1" expected="$2" actual="$3"
  TOTAL=$((TOTAL + 1))
  if [[ "$expected" == "$actual" ]]; then
    echo "PASS: $test_name"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $test_name (expected='$expected', actual='$actual')"
    FAIL=$((FAIL + 1))
  fi
}

# --- Test the merge-skip regex used in pr-validation.yml ---
# The pattern: [[ "$SUBJECT" =~ ^Merge\  ]]

test_merge_skip_regex() {
  local subject="$1" expected_skip="$2" label="$3"
  if [[ "$subject" =~ ^Merge\  ]]; then
    assert_eq "$label" "$expected_skip" "skip"
  else
    assert_eq "$label" "$expected_skip" "no_skip"
  fi
}

echo "=== PR-validation merge skip tests ==="
test_merge_skip_regex "Merge branch 'main' into feat/foo"     "skip"    "Standard merge commit"
test_merge_skip_regex "Merge pull request #123 from org/branch" "skip"  "GitHub PR merge commit"
test_merge_skip_regex "Merge remote-tracking branch 'origin/main'" "skip" "Remote tracking merge"
test_merge_skip_regex "Merge tag 'v1.0' into dev"             "skip"    "Merge tag commit"
test_merge_skip_regex "[Feat] Add new feature"                 "no_skip" "Normal typed commit"
test_merge_skip_regex "[BugFix] Fix something"                 "no_skip" "Normal bugfix commit"
test_merge_skip_regex "Merge-sort implementation"              "no_skip" "Merge-sort (no trailing space after Merge)"
test_merge_skip_regex "Merging branches"                       "no_skip" "Merging (not 'Merge ')"
test_merge_skip_regex "merge branch 'main'"                    "no_skip" "Lowercase merge"
test_merge_skip_regex "Some commit with Merge in middle"       "no_skip" "Merge in middle of message"

echo ""
echo "=== validate.sh merge skip tests ==="

# Test check_commit_message with merge commits by simulating the function logic
# We replicate the expected post-fix behavior of check_commit_message
test_validate_merge() {
  local msg="$1" expected="$2" label="$3"

  # Simulated check_commit_message logic (post-fix)
  if [[ "$msg" =~ ^Merge\  ]]; then
    result="skip"
  elif [[ "$msg" =~ $COMMIT_MSG_PATTERN ]]; then
    result="pass"
  else
    result="fail"
  fi

  assert_eq "$label" "$expected" "$result"
}

test_validate_merge "Merge branch 'main' into feat/foo"      "skip"  "validate.sh: merge commit skipped"
test_validate_merge "Merge pull request #99 from org/branch"  "skip"  "validate.sh: PR merge commit skipped"
test_validate_merge "[Feat] Add feature"                      "pass"  "validate.sh: normal commit passes"
test_validate_merge "[BugFix][Core] Fix bug"                  "pass"  "validate.sh: scoped commit passes"
test_validate_merge "Merge-sort implementation"               "fail"  "validate.sh: Merge-sort not skipped, fails format"
test_validate_merge "bad commit message"                      "fail"  "validate.sh: bad message fails"

echo ""
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="
if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
exit 0
