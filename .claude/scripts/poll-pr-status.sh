#!/bin/bash
# Poll GitHub PR status (CI checks + review comments) and return structured JSON
# Usage: poll-pr-status.sh <owner/repo> <pr_number> [--interval 30] [--timeout 600]
#
# Designed to run via Claude Code's Bash tool with run_in_background: true.
# Consumes zero Claude API tokens during the wait.

set -euo pipefail

# Defaults
INTERVAL=30
TIMEOUT=600
OWNER_REPO=""
PR_NUMBER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      INTERVAL="$2"
      if [[ ! "$INTERVAL" =~ ^[0-9]+$ ]]; then
        echo '{"status":"error","message":"INTERVAL must be a positive integer"}'
        exit 1
      fi
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      if [[ ! "$TIMEOUT" =~ ^[0-9]+$ ]]; then
        echo '{"status":"error","message":"TIMEOUT must be a positive integer"}'
        exit 1
      fi
      shift 2
      ;;
    *)
      if [[ -z "$OWNER_REPO" ]]; then
        OWNER_REPO="$1"
        if [[ ! "$OWNER_REPO" =~ ^[^/]+/[^/]+$ ]]; then
          echo '{"status":"error","message":"OWNER_REPO must be in owner/repo format"}'
          exit 1
        fi
      elif [[ -z "$PR_NUMBER" ]]; then
        PR_NUMBER="$1"
        if [[ ! "$PR_NUMBER" =~ ^[0-9]+$ ]]; then
          echo '{"status":"error","message":"PR_NUMBER must be a positive integer"}'
          exit 1
        fi
      fi
      shift
      ;;
  esac
done

# Validate required arguments
if [[ -z "$OWNER_REPO" ]] || [[ -z "$PR_NUMBER" ]]; then
  echo '{"status":"error","message":"Usage: poll-pr-status.sh <owner/repo> <pr_number> [--interval 30] [--timeout 600]"}'
  exit 1
fi

# Verify required tools
if ! command -v gh &> /dev/null; then
  echo '{"status":"error","message":"gh CLI not found"}'
  exit 1
fi

if ! command -v jq &> /dev/null; then
  echo '{"status":"error","message":"jq not found. Install jq or ensure it is on PATH."}'
  exit 1
fi

# Split owner/repo
IFS='/' read -r OWNER REPO <<< "$OWNER_REPO"
if [[ -z "$OWNER" ]] || [[ -z "$REPO" ]]; then
  echo '{"status":"error","message":"Invalid owner/repo format"}'
  exit 1
fi

# Logging function (to stderr only)
log_error() {
  echo "[poll-pr-status] ERROR: $1" >&2
}

log_info() {
  echo "[poll-pr-status] $1" >&2
}

# Initialize state tracking
START_TIME=$(date +%s)
PR_CREATED_AT=""
PR_AUTHOR=""
HEAD_SHA=""

# --- Core functions ---

# Fetch PR info once at startup.
# HEAD_SHA is intentionally not refreshed during polling — each re-invocation
# (after a fix-push cycle) will fetch the new SHA from scratch.
fetch_pr_info() {
  local pr_json
  pr_json=$(gh api "repos/${OWNER}/${REPO}/pulls/${PR_NUMBER}" 2>&1) || { log_error "gh api failed: $pr_json"; return 1; }

  PR_CREATED_AT=$(echo "$pr_json" | jq -r '.created_at') || { log_error "Failed to parse .created_at from PR JSON"; return 1; }
  PR_AUTHOR=$(echo "$pr_json" | jq -r '.user.login') || { log_error "Failed to parse .user.login from PR JSON"; return 1; }
  HEAD_SHA=$(echo "$pr_json" | jq -r '.head.sha') || { log_error "Failed to parse .head.sha from PR JSON"; return 1; }

  if [[ -z "$PR_CREATED_AT" || -z "$PR_AUTHOR" || -z "$HEAD_SHA" ]]; then
    return 1
  fi
  return 0
}

# Fetch CI check runs (returns JSON object with .check_runs array)
# Returns non-zero on failure so the caller's if-guard can log and retry.
fetch_ci_checks() {
  local result
  result=$(gh api "repos/${OWNER}/${REPO}/commits/${HEAD_SHA}/check-runs" 2>&1) || { log_error "fetch_ci_checks: $result"; return 1; }
  echo "$result"
}

# Fetch inline review comments (returns merged JSON array)
# --paginate can produce concatenated arrays ([...][...]); jq -s 'add' merges them.
fetch_inline_comments() {
  local raw
  raw=$(gh api "repos/${OWNER}/${REPO}/pulls/${PR_NUMBER}/comments" --paginate 2>&1) || { log_error "fetch_inline_comments: $raw"; return 1; }
  echo "$raw" | jq -s 'add // []'
}

# Fetch PR-level reviews (returns merged JSON array)
fetch_pr_reviews() {
  local raw
  raw=$(gh api "repos/${OWNER}/${REPO}/pulls/${PR_NUMBER}/reviews" --paginate 2>&1) || { log_error "fetch_pr_reviews: $raw"; return 1; }
  echo "$raw" | jq -s 'add // []'
}

# Determine CI state from check runs JSON
# Returns: "success", "failure", or "pending"
determine_ci_state() {
  local checks_json="$1"
  echo "$checks_json" | jq -r '
    .check_runs as $runs |
    ($runs | length) as $total |
    [$runs[] | select(.status != "completed")] | length as $pending |
    [$runs[] | select(.conclusion == "failure" or .conclusion == "cancelled")] | length as $failed |
    if $total == 0 then "pending"
    elif $pending > 0 then "pending"
    elif $failed > 0 then "failure"
    else "success"
    end
  ' || echo "pending"
}

# Extract failed checks as JSON array
get_failed_checks() {
  local checks_json="$1"
  echo "$checks_json" | jq -c '[.check_runs[] | select(.conclusion == "failure" or .conclusion == "cancelled") | {name: .name, conclusion: .conclusion, url: .html_url}]' || echo "[]"
}

# Filter inline comments: exclude PR author, only after PR creation
filter_inline_comments() {
  local comments_json="$1"
  echo "$comments_json" | jq -c --arg author "$PR_AUTHOR" --arg created "$PR_CREATED_AT" '
    [.[] | select(
      .user.login != $author and
      .created_at > $created
    ) | {
      id: .id,
      author: .user.login,
      body: .body,
      path: .path,
      line: .line,
      created_at: .created_at
    }]' || echo "[]"
}

# Filter PR reviews: exclude PR author, only non-empty body, after PR creation
filter_pr_reviews() {
  local reviews_json="$1"
  echo "$reviews_json" | jq -c --arg author "$PR_AUTHOR" '
    [.[] | select(
      .user.login != $author and
      .state != "PENDING" and
      .body != null and
      .body != ""
    ) | {
      id: .id,
      author: .user.login,
      body: .body,
      state: .state
    }]' || echo "[]"
}

# Build and output final JSON
output_json() {
  local status="$1"
  local ci_state="$2"
  local failed_checks="$3"
  local inline_comments="$4"
  local review_bodies="$5"
  local unresolved="$6"

  jq -n \
    --arg status "$status" \
    --arg ci_state "$ci_state" \
    --argjson failed_checks "$failed_checks" \
    --argjson inline "$inline_comments" \
    --argjson reviews "$review_bodies" \
    --argjson unresolved "$unresolved" \
    '{
      status: $status,
      ci: {
        state: $ci_state,
        failed_checks: $failed_checks
      },
      reviews: {
        new_inline_comments: $inline,
        new_review_bodies: $reviews,
        unresolved_count: $unresolved
      }
    }'
}

# --- Main polling loop ---

main() {
  log_info "Polling PR #${PR_NUMBER} on ${OWNER}/${REPO} (interval=${INTERVAL}s, timeout=${TIMEOUT}s)"

  # Fetch PR info once
  if ! fetch_pr_info; then
    log_error "Failed to fetch PR #${PR_NUMBER} info. Check PR number and gh auth."
    output_json "error" "unknown" "[]" "[]" "[]" 0
    exit 1
  fi

  log_info "PR author: ${PR_AUTHOR}, HEAD: ${HEAD_SHA:0:7}, created: ${PR_CREATED_AT}"

  local ci_state="pending"
  local failed_checks="[]"
  local inline_comments="[]"
  local review_bodies="[]"
  local unresolved=0

  while true; do
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))

    # Check timeout
    if [[ $elapsed -ge $TIMEOUT ]]; then
      log_info "Timeout reached (${TIMEOUT}s). Returning current state."
      output_json "timeout" "$ci_state" "$failed_checks" "$inline_comments" "$review_bodies" "$unresolved"
      exit 0
    fi

    # Fetch CI status
    local checks_json
    if checks_json=$(fetch_ci_checks); then
      ci_state=$(determine_ci_state "$checks_json")
      failed_checks=$(get_failed_checks "$checks_json")
    else
      log_error "Failed to fetch CI status, will retry"
    fi

    # Fetch inline comments
    local raw_inline
    if raw_inline=$(fetch_inline_comments); then
      inline_comments=$(filter_inline_comments "$raw_inline")
      # Simplified: count filtered inline comments as unresolved
      # (GitHub REST API doesn't expose thread resolution state)
      unresolved=$(echo "$inline_comments" | jq 'length' || echo 0)
    else
      log_error "Failed to fetch inline comments, will retry"
    fi

    # Fetch PR reviews
    local raw_reviews
    if raw_reviews=$(fetch_pr_reviews); then
      review_bodies=$(filter_pr_reviews "$raw_reviews")
    else
      log_error "Failed to fetch PR reviews, will retry"
    fi

    log_info "Elapsed: ${elapsed}s | CI: ${ci_state} | Inline comments: $(echo "$inline_comments" | jq 'length' || echo '?') | Reviews: $(echo "$review_bodies" | jq 'length' || echo '?')"

    # Termination: CI completed (pass or fail) AND at least one interval has passed
    # (wait one interval to give review bots time to post)
    if [[ "$ci_state" != "pending" ]] && [[ $elapsed -ge $INTERVAL ]]; then
      log_info "CI completed (${ci_state}). Returning results."
      output_json "ready" "$ci_state" "$failed_checks" "$inline_comments" "$review_bodies" "$unresolved"
      exit 0
    fi

    sleep "$INTERVAL"
  done
}

main
