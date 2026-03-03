#!/bin/bash
# Run tests and benchmarks specified in a context.json file
# Usage: run-affected-tests.sh <context.json>
#
# Reads test_targets and bench_targets from the JSON file,
# executes each via pytest, and outputs structured JSON results.
#
# Designed to be called by lifecycle-issue-fixer and lifecycle-pull-request skills.

set -euo pipefail

# --- Argument validation ---

if [[ $# -lt 1 ]]; then
  echo '{"status":"error","message":"Usage: run-affected-tests.sh <context.json>"}' >&1
  exit 1
fi

CONTEXT_FILE="$1"

if [[ ! -f "$CONTEXT_FILE" ]]; then
  echo "{\"status\":\"error\",\"message\":\"Context file not found: ${CONTEXT_FILE}\"}" >&1
  exit 1
fi

# Verify jq is available
if ! command -v jq &>/dev/null; then
  echo '{"status":"error","message":"jq not found. Install jq or ensure it is on PATH."}' >&1
  exit 1
fi

# --- Parse context.json ---

# Validate JSON structure
if ! jq -e '.' "$CONTEXT_FILE" &>/dev/null; then
  echo '{"status":"error","message":"Invalid JSON in context file"}' >&1
  exit 1
fi

# Extract test_targets (required)
TEST_TARGETS=$(jq -r '.test_targets // empty' "$CONTEXT_FILE")
if [[ -z "$TEST_TARGETS" ]]; then
  echo '{"status":"error","message":"context.json missing required field: test_targets"}' >&1
  exit 1
fi

# Extract bench_targets (optional, defaults to empty array)
BENCH_TARGETS=$(jq -r '.bench_targets // "[]"' "$CONTEXT_FILE")

# Logging (stderr only)
log_info() {
  echo "[run-affected-tests] $1" >&2
}

# --- Run tests ---

TEST_RESULTS="[]"
BENCH_RESULTS="[]"
WARNINGS="[]"
TOTAL_TESTS=0
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_BENCH=0
BENCH_PASSED=0
BENCH_FAILED=0

run_pytest_file() {
  local file="$1"
  local category="$2"  # "test" or "bench"

  if [[ ! -f "$file" ]]; then
    WARNINGS=$(echo "$WARNINGS" | jq --arg w "${file} not found, skipped" '. + [$w]')
    log_info "WARNING: $file not found, skipping"
    return
  fi

  log_info "Running: $file"

  # Run pytest and capture output
  local pytest_output
  local exit_code=0
  pytest_output=$(PYTHONPATH="$PWD" python -m pytest -v "$file" --tb=short -q 2>&1) || exit_code=$?

  # Parse pytest output for pass/fail/error counts
  # pytest summary line format: "X passed, Y failed, Z errors" or subsets thereof
  local passed=0
  local failed=0
  local errors=0

  if echo "$pytest_output" | grep -qE '[0-9]+ passed'; then
    passed=$(echo "$pytest_output" | grep -oE '[0-9]+ passed' | tail -1 | grep -oE '[0-9]+')
  fi
  if echo "$pytest_output" | grep -qE '[0-9]+ failed'; then
    failed=$(echo "$pytest_output" | grep -oE '[0-9]+ failed' | tail -1 | grep -oE '[0-9]+')
  fi
  if echo "$pytest_output" | grep -qE '[0-9]+ error'; then
    errors=$(echo "$pytest_output" | grep -oE '[0-9]+ error' | tail -1 | grep -oE '[0-9]+')
  fi

  # Extract failure/error names if any
  local failures="[]"
  if [[ $failed -gt 0 ]] || [[ $errors -gt 0 ]]; then
    failures=$(echo "$pytest_output" | grep -E '^(FAILED |ERROR )' | sed 's/^FAILED //' | sed 's/^ERROR //' | sed 's/ - .*$//' | jq -R -s 'split("\n") | map(select(. != ""))')
  fi

  # Determine status: non-zero exit code OR any failed/error count means failure
  # This catches collection errors (e.g., ModuleNotFoundError) where pytest exits
  # non-zero but reports no "failed" count in the summary line.
  local status="pass"
  if [[ $exit_code -ne 0 ]] || [[ $failed -gt 0 ]] || [[ $errors -gt 0 ]]; then
    status="fail"
    # If pytest exited non-zero but no failed/errors were parsed,
    # this is a collection/import error — count it as 1 error
    if [[ $failed -eq 0 ]] && [[ $errors -eq 0 ]]; then
      errors=1
      failures=$(echo "$pytest_output" | tail -5 | jq -R -s 'split("\n") | map(select(. != ""))')
    fi
  fi

  # Include errors in failed count for aggregation
  local total_failed=$((failed + errors))

  local result
  result=$(jq -n \
    --arg file "$file" \
    --arg status "$status" \
    --argjson passed "$passed" \
    --argjson failed "$total_failed" \
    --argjson failures "$failures" \
    '{file: $file, status: $status, passed: $passed, failed: $failed, failures: $failures}')

  if [[ "$category" == "test" ]]; then
    TEST_RESULTS=$(echo "$TEST_RESULTS" | jq --argjson r "$result" '. + [$r]')
    TOTAL_TESTS=$((TOTAL_TESTS + passed + total_failed))
    TOTAL_PASSED=$((TOTAL_PASSED + passed))
    TOTAL_FAILED=$((TOTAL_FAILED + total_failed))
  else
    BENCH_RESULTS=$(echo "$BENCH_RESULTS" | jq --argjson r "$result" '. + [$r]')
    TOTAL_BENCH=$((TOTAL_BENCH + passed + total_failed))
    BENCH_PASSED=$((BENCH_PASSED + passed))
    BENCH_FAILED=$((BENCH_FAILED + total_failed))
  fi
}

# Run test targets
log_info "=== Running test targets ==="
for file in $(echo "$TEST_TARGETS" | jq -r '.[]'); do
  run_pytest_file "$file" "test"
done

# Run bench targets
log_info "=== Running benchmark targets ==="
for file in $(echo "$BENCH_TARGETS" | jq -r '.[]'); do
  run_pytest_file "$file" "bench"
done

# --- Determine overall status ---

OVERALL_STATUS="pass"
if [[ $TOTAL_FAILED -gt 0 ]] || [[ $BENCH_FAILED -gt 0 ]]; then
  OVERALL_STATUS="fail"
fi

# Check for partial: some passed, some failed
if [[ "$OVERALL_STATUS" == "fail" ]] && [[ $TOTAL_PASSED -gt 0 || $BENCH_PASSED -gt 0 ]]; then
  OVERALL_STATUS="partial"
fi

# --- Output structured JSON ---

jq -n \
  --arg status "$OVERALL_STATUS" \
  --argjson test_total "$TOTAL_TESTS" \
  --argjson test_passed "$TOTAL_PASSED" \
  --argjson test_failed "$TOTAL_FAILED" \
  --argjson test_results "$TEST_RESULTS" \
  --argjson bench_total "$TOTAL_BENCH" \
  --argjson bench_passed "$BENCH_PASSED" \
  --argjson bench_failed "$BENCH_FAILED" \
  --argjson bench_results "$BENCH_RESULTS" \
  --argjson warnings "$WARNINGS" \
  '{
    status: $status,
    tests: {
      total: $test_total,
      passed: $test_passed,
      failed: $test_failed,
      results: $test_results
    },
    benchmarks: {
      total: $bench_total,
      passed: $bench_passed,
      failed: $bench_failed,
      results: $bench_results
    },
    warnings: $warnings
  }'
