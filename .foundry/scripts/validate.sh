#!/bin/bash
# Validate PR metadata for TileOPs
# Usage: validate.sh <owner/repo> <pr_number>
#
# Checks: title format, body sections, labels, MCP pitfall
# Exit 0 = all checks pass
# Exit 1 = at least one check failed

set -euo pipefail

# Source canonical type definitions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$REPO_ROOT/.claude/conventions/types.sh"

OWNER_REPO="${1:?Usage: validate.sh <owner/repo> <pr_number>}"
PR_NUMBER="${2:?Usage: validate.sh <owner/repo> <pr_number>}"

ERRORS=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1" >&2; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠${NC} $1" >&2; }

# --- Fetch PR data ---

PR_JSON=$(gh pr view "$PR_NUMBER" --repo "$OWNER_REPO" --json title,body,labels 2>&1) || {
  echo -e "${RED}✗ Failed to fetch PR #${PR_NUMBER}: ${PR_JSON}${NC}" >&2
  exit 1
}

TITLE=$(echo "$PR_JSON" | jq -r '.title')
BODY=$(echo "$PR_JSON" | jq -r '.body')
LABEL_COUNT=$(echo "$PR_JSON" | jq '.labels | length')

echo "=== PR #${PR_NUMBER} validation ==="
echo "Title: ${TITLE}"
echo ""

# --- Checks ---

# 1. PR title format
if [[ "$TITLE" =~ $COMMIT_MSG_PATTERN ]]; then
  pass "PR title follows [Type] Description format"
else
  fail "PR title must match [Type] Description (e.g. [Feat] Add forward op)"
fi

# 2. ## Summary section
if echo "$BODY" | grep -q '## Summary'; then
  pass "PR body contains ## Summary section"
else
  fail "PR body must contain ## Summary section"
fi

# 3. ## Test plan section
if echo "$BODY" | grep -q '## Test plan'; then
  pass "PR body contains ## Test plan section"
else
  fail "PR body must contain ## Test plan section"
fi

# 4. At least one label
if [[ "$LABEL_COUNT" -gt 0 ]]; then
  LABELS=$(echo "$PR_JSON" | jq -r '[.labels[].name] | join(", ")')
  pass "PR has ${LABEL_COUNT} label(s): ${LABELS}"
else
  fail "PR must have at least one label"
fi

# 5. No literal \n in body (MCP pitfall)
if echo "$BODY" | grep -q '\\n'; then
  fail "PR body contains literal \\\\n — use actual newlines instead"
else
  pass "PR body uses actual newlines (no MCP pitfall)"
fi

# 6. BugFix should have Regression section (warning only)
if [[ "$TITLE" =~ ^\[BugFix\] ]] && ! echo "$BODY" | grep -q '## Regression'; then
  warn "[BugFix] PR should include ## Regression section"
fi

echo ""
if [[ $ERRORS -gt 0 ]]; then
  echo -e "${RED}FAILED: ${ERRORS} check(s) failed${NC}" >&2
  exit 1
else
  echo -e "${GREEN}ALL CHECKS PASSED${NC}"
  exit 0
fi
