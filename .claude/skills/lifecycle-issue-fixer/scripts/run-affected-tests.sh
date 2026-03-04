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

# Check jq availability first (before any jq usage)
if ! command -v jq &>/dev/null; then
  echo '{"status":"error","message":"jq not found. Install jq or ensure it is on PATH."}' >&1
  exit 1
fi

if [[ $# -lt 1 ]]; then
  jq -n '{status: "error", message: "Usage: run-affected-tests.sh <context.json>"}' >&1
  exit 1
fi

CONTEXT_FILE="$1"

if [[ ! -f "$CONTEXT_FILE" ]]; then
  jq -n --arg file "$CONTEXT_FILE" '{status: "error", message: "Context file not found: \($file)"}' >&1
  exit 1
fi

# --- Parse context.json ---

# Validate JSON structure
if ! jq -e '.' "$CONTEXT_FILE" &>/dev/null; then
  echo '{"status":"error","message":"Invalid JSON in context file"}' >&1
  exit 1
fi

# Extract test_targets (required, must be JSON array)
TEST_TARGETS=$(jq -c '.test_targets // empty' "$CONTEXT_FILE")
if [[ -z "$TEST_TARGETS" ]]; then
  jq -n '{status: "error", message: "context.json missing required field: test_targets"}' >&1
  exit 1
fi
if ! jq -e 'type == "array"' <<<"$TEST_TARGETS" >/dev/null 2>&1; then
  jq -n '{status: "error", message: "context.json field test_targets must be a JSON array"}' >&1
  exit 1
fi

# Extract bench_targets (optional, defaults to empty array, must be JSON array)
BENCH_TARGETS=$(jq -c '.bench_targets // []' "$CONTEXT_FILE")
if ! jq -e 'type == "array"' <<<"$BENCH_TARGETS" >/dev/null 2>&1; then
  jq -n '{status: "error", message: "context.json field bench_targets must be a JSON array"}' >&1
  exit 1
fi

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

  # Reject absolute paths and path traversal attempts
  if [[ "$file" == /* ]] || [[ "$file" == *../* ]]; then
    WARNINGS=$(echo "$WARNINGS" | jq --arg w "${file} rejected: absolute or traversal path" '. + [$w]')
    log_info "WARNING: $file rejected (path security check), skipping"
    return
  fi

  if [[ ! -f "$file" ]]; then
    WARNINGS=$(echo "$WARNINGS" | jq --arg w "${file} not found, skipped" '. + [$w]')
    log_info "WARNING: $file not found, skipping"
    return
  fi

  log_info "Running: $file"

  # Run pytest and capture output
  local pytest_output
  local exit_code=0
  pytest_output=$(PYTHONPATH="$PWD" python -m pytest -v --tb=short -q -- "$file" 2>&1) || exit_code=$?

  # Parse pytest summary counts (e.g., "3 passed, 1 failed, 2 errors")
  local passed failed errors
  passed=$(echo "$pytest_output" | grep -oE '[0-9]+ passed' | tail -1 | grep -oE '[0-9]+' || echo 0)
  failed=$(echo "$pytest_output" | grep -oE '[0-9]+ failed' | tail -1 | grep -oE '[0-9]+' || echo 0)
  errors=$(echo "$pytest_output" | grep -oE '[0-9]+ error' | tail -1 | grep -oE '[0-9]+' || echo 0)

  # Extract failure/error names if any
  local failures="[]"
  if [[ $failed -gt 0 ]] || [[ $errors -gt 0 ]]; then
    failures=$(echo "$pytest_output" | grep -E '^(FAILED |ERROR )' | sed -E 's/^(FAILED |ERROR )//; s/ - .*$//' | jq -R -s 'split("\n") | map(select(. != ""))')
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

# Run test targets (process substitution avoids subshell — variable updates preserved)
log_info "=== Running test targets ==="
while IFS= read -r file; do
  run_pytest_file "$file" "test"
done < <(jq -r '.[]' <<<"$TEST_TARGETS")

# Run bench targets
log_info "=== Running benchmark targets ==="
while IFS= read -r file; do
  run_pytest_file "$file" "bench"
done < <(jq -r '.[]' <<<"$BENCH_TARGETS")

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
