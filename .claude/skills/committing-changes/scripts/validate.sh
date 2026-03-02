#!/bin/bash
# Validate commit readiness for TileOPs
# Usage: validate.sh [--pre | --post]
#
# --pre  : Run before commit (pre-commit + branch name)
# --post : Run after commit (all --pre checks + commit msg + large files)
#
# Exit 0 = all checks pass
# Exit 1 = at least one check failed

set -euo pipefail

# Source canonical type definitions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$REPO_ROOT/.claude/conventions/types.sh"

MODE="${1:---pre}"
ERRORS=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1" >&2; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠${NC} $1" >&2; }

# --- Checks ---

check_precommit() {
  echo "Running pre-commit..."
  if pre-commit run --all-files; then
    pass "pre-commit passed"
  else
    fail "pre-commit failed — fix the issues above and re-run"
  fi
}

check_branch_name() {
  local branch
  branch=$(git branch --show-current)
  if [[ -z "$branch" ]]; then
    fail "Not on a branch (detached HEAD)"
    return
  fi
  if [[ "$branch" == "main" || "$branch" == "master" ]]; then
    fail "Cannot commit directly to '$branch' — create a feature branch first"
    return
  fi
  # Pattern: type/scope/description  (each segment: lowercase, digits, dots, hyphens)
  if [[ "$branch" =~ $BRANCH_NAME_PATTERN ]]; then
    pass "Branch name '${branch}' follows type/scope/description"
  else
    fail "Branch '${branch}' must match type/scope/description (e.g. feat/flash-attn/fwd-kernel)"
  fi
}

check_commit_message() {
  local msg
  msg=$(git log -1 --pretty=%s 2>/dev/null || echo "")
  if [[ -z "$msg" ]]; then
    fail "No commits found on current branch"
    return
  fi
  if [[ "$msg" =~ $COMMIT_MSG_PATTERN ]]; then
    pass "Commit message '${msg}' follows [Type] Description"
  else
    fail "Commit message '${msg}' must match [Type] Description (e.g. [Feat] Add forward op)"
  fi
}

check_large_files() {
  local large_files=""
  while IFS= read -r f; do
    if [[ -f "$f" ]]; then
      local size
      size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
      if [[ "$size" -gt 1048576 ]]; then
        large_files="${large_files}  ${f} ($((size / 1024))KB)\n"
      fi
    fi
  done < <(git diff-tree --no-commit-id -r HEAD --name-only 2>/dev/null)

  if [[ -z "$large_files" ]]; then
    pass "No large files (>1MB) in commit"
  else
    fail "Large files detected (>1MB):\n${large_files}"
  fi
}

# --- Main ---

case "$MODE" in
  --pre)
    echo "=== Pre-commit validation ==="
    check_precommit
    check_branch_name
    ;;
  --post)
    echo "=== Post-commit validation ==="
    check_precommit
    check_branch_name
    check_commit_message
    check_large_files
    ;;
  *)
    echo "Usage: validate.sh [--pre | --post]" >&2
    exit 1
    ;;
esac

echo ""
if [[ $ERRORS -gt 0 ]]; then
  echo -e "${RED}FAILED: ${ERRORS} check(s) failed${NC}" >&2
  exit 1
else
  echo -e "${GREEN}ALL CHECKS PASSED${NC}"
  exit 0
fi
