#!/bin/bash
# Poll GitHub PR status (CI checks + review comments) and return structured JSON
# Usage: poll-pr-status.sh <owner/repo> <pr_number> [--interval 30] [--timeout 6000]
#
# Designed to run via Claude Code's Bash tool as a blocking call (timeout: 6060000).
# The agent waits for this script to complete before processing the result.

set -euo pipefail

# Defaults
INTERVAL=30
TIMEOUT=6000
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
  echo '{"status":"error","message":"Usage: poll-pr-status.sh <owner/repo> <pr_number> [--interval 30] [--timeout 6000]"}'
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

# Fetch review threads via GraphQL (returns JSON with thread data including node IDs)
# This replaces the REST-based fetch_inline_comments to provide:
# - thread_node_id (PRRT_...) needed for resolveReviewThread mutation
# - isResolved state for accurate unresolved_count
# - Full conversation chain per thread (comments first:100)
fetch_review_threads() {
  local query='query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          pageInfo { hasNextPage endCursor }
          nodes {
            id
            isResolved
            isOutdated
            comments(first: 100) {
              nodes {
                databaseId
                author { login }
                body
                path
                line
                createdAt
              }
            }
          }
        }
      }
    }
  }'

  local result
  result=$(gh api graphql -F owner="$OWNER" -F repo="$REPO" -F pr="$PR_NUMBER" -f query="$query" 2>&1) || { log_error "fetch_review_threads: $result"; return 1; }

  # Validate GraphQL response structure:
  # 1. Reject if response contains non-empty .errors array (semantic GraphQL error)
  # 2. Reject if .data.repository.pullRequest.reviewThreads.nodes is missing/non-array
  #
  # Note: GraphQL can return both .errors AND .data (partial success). We reject
  # any response with .errors present, regardless of whether .data is also valid.
  local error_count
  error_count=$(echo "$result" | jq -r '(.errors // []) | length' 2>/dev/null) || error_count="unknown"
  if [[ "$error_count" != "0" ]]; then
    local err_msg
    err_msg=$(echo "$result" | jq -r '.errors[0].message // "unknown GraphQL error"' 2>/dev/null)
    log_error "fetch_review_threads: GraphQL returned $error_count error(s): $err_msg"
    return 1
  fi

  local nodes_type
  nodes_type=$(echo "$result" | jq -r '.data.repository.pullRequest.reviewThreads.nodes | type' 2>/dev/null) || nodes_type="null"
  if [[ "$nodes_type" != "array" ]]; then
    log_error "fetch_review_threads: GraphQL response missing valid reviewThreads.nodes (got type: $nodes_type)"
    return 1
  fi

  # Treat >100 threads as error to avoid undercounting unresolved threads
  local has_next
  has_next=$(echo "$result" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage' 2>/dev/null || echo "false")
  if [[ "$has_next" == "true" ]]; then
    log_error "fetch_review_threads: PR has >100 review threads; pagination not implemented, aborting to avoid partial data."
    return 1
  fi

  echo "$result"
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

# Filter review threads from GraphQL response
# Extracts all unresolved threads with their node IDs and full comment chains.
# Does NOT exclude author-only threads — all unresolved threads are exposed so
# that unresolved_count and actionable thread list stay consistent.
# The agent (handler) decides which threads need action vs which to auto-resolve.
filter_review_threads() {
  local graphql_json="$1"
  local result
  result=$(echo "$graphql_json" | jq -c '
    [.data.repository.pullRequest.reviewThreads.nodes[]
     | select(.isResolved == false)
     | {
         thread_node_id: .id,
         is_resolved: .isResolved,
         is_outdated: .isOutdated,
         comments: [.comments.nodes[] | {
           id: .databaseId,
           author: .author.login,
           body: .body,
           path: .path,
           line: .line,
           created_at: .createdAt
         }]
       }
    ]') || { log_error "filter_review_threads: jq extraction failed"; return 1; }
  echo "$result"
}

# Count unresolved threads from GraphQL response (based on isResolved field)
# Returns non-zero on parse failure so caller treats cycle as unsuccessful.
count_unresolved_threads() {
  local graphql_json="$1"
  local count
  count=$(echo "$graphql_json" | jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)] | length') || { log_error "count_unresolved_threads: jq count failed"; return 1; }
  echo "$count"
}

# Filter PR reviews: only CHANGES_REQUESTED with non-empty body, exclude PR author.
# COMMENTED and APPROVED reviews are informational and do not require handler action.
# Only CHANGES_REQUESTED is a blocking state that the handler must address.
filter_pr_reviews() {
  local reviews_json="$1"
  echo "$reviews_json" | jq -c --arg author "$PR_AUTHOR" '
    [.[] | select(
      .user.login != $author and
      .state == "CHANGES_REQUESTED" and
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

    # Per-cycle fetch success tracking
    local ci_ok=false
    local threads_ok=false
    local pr_reviews_ok=false

    # Fetch CI status
    local checks_json
    if checks_json=$(fetch_ci_checks); then
      ci_state=$(determine_ci_state "$checks_json")
      failed_checks=$(get_failed_checks "$checks_json")
      ci_ok=true
    else
      log_error "Failed to fetch CI status, will retry"
    fi

    # Fetch review threads (GraphQL)
    # threads_ok requires ALL three steps to succeed: fetch, filter, count.
    # If any step fails (including semantic GraphQL errors or jq parse failures),
    # threads_ok stays false and the cycle cannot return "ready".
    local threads_json
    if threads_json=$(fetch_review_threads); then
      local filtered_threads
      local thread_count
      if filtered_threads=$(filter_review_threads "$threads_json") && \
         thread_count=$(count_unresolved_threads "$threads_json"); then
        inline_comments="$filtered_threads"
        unresolved="$thread_count"
        threads_ok=true
      else
        log_error "Failed to parse review thread data, will retry"
      fi
    else
      log_error "Failed to fetch review threads, will retry"
    fi

    # Fetch PR reviews
    local raw_reviews
    if raw_reviews=$(fetch_pr_reviews); then
      review_bodies=$(filter_pr_reviews "$raw_reviews")
      pr_reviews_ok=true
    else
      log_error "Failed to fetch PR reviews, will retry"
    fi

    log_info "Elapsed: ${elapsed}s | CI: ${ci_state} | Inline comments: $(echo "$inline_comments" | jq 'length' || echo '?') | Reviews: $(echo "$review_bodies" | jq 'length' || echo '?') | Fetches OK: ci=$ci_ok threads=$threads_ok reviews=$pr_reviews_ok"

    # Termination logic — three distinct outcomes:
    #   "actionable" — CI failure or unresolved reviews exist, handler must process
    #   "done"       — CI all success + all threads resolved, handler can skip to Phase 6
    #   (continue)   — CI still pending + nothing to handle, keep polling
    #
    # Must have waited at least one interval (give bots time to post) and all
    # fetches must have succeeded in this cycle (no stale data).
    if [[ $elapsed -ge $INTERVAL ]] && \
       [[ "$ci_ok" == "true" ]] && [[ "$threads_ok" == "true" ]] && [[ "$pr_reviews_ok" == "true" ]]; then

      local inline_count review_count
      inline_count=$(echo "$inline_comments" | jq 'length' 2>/dev/null || echo 0)
      review_count=$(echo "$review_bodies" | jq 'length' 2>/dev/null || echo 0)

      # Case 1: actionable — CI failed or unresolved reviews/threads exist
      if [[ "$ci_state" == "failure" ]] || [[ "$unresolved" -gt 0 ]] || [[ "$review_count" -gt 0 ]]; then
        log_info "Actionable: ci=$ci_state, unresolved=$unresolved, reviews=$review_count. Handler must process."
        output_json "actionable" "$ci_state" "$failed_checks" "$inline_comments" "$review_bodies" "$unresolved"
        exit 0
      fi

      # Case 2: done — CI success + no unresolved threads + no pending reviews
      if [[ "$ci_state" == "success" ]] && [[ "$unresolved" -eq 0 ]] && [[ "$review_count" -eq 0 ]]; then
        log_info "Done: CI success, all threads resolved, no pending reviews."
        output_json "done" "$ci_state" "$failed_checks" "$inline_comments" "$review_bodies" "$unresolved"
        exit 0
      fi

      # Otherwise CI is still pending with nothing actionable — keep polling
    fi

    # If fetches failed, keep polling (don't return with stale data)
    if [[ $elapsed -ge $INTERVAL ]] && \
       { [[ "$ci_ok" != "true" ]] || [[ "$threads_ok" != "true" ]] || [[ "$pr_reviews_ok" != "true" ]]; }; then
      log_info "Fetch(es) incomplete (ci=$ci_ok, threads=$threads_ok, reviews=$pr_reviews_ok). Retrying..."
    fi

    sleep "$INTERVAL"
  done
}

main
