#!/bin/bash
# types.sh — Single source of truth for TileOPs commit/PR/issue type conventions.
#
# Usage:
#   source .claude/conventions/types.sh
#   # Now use $COMMIT_MSG_PATTERN, $BRANCH_NAME_PATTERN, TYPE_TO_LABEL, etc.
#
# This file is sourced by validation scripts, CI workflows, and skill docs.
# Update HERE first, then all consumers pick up the change automatically.

# ---------------------------------------------------------------------------
# Commit / PR types
# ---------------------------------------------------------------------------

COMMIT_PR_TYPES="Feat|BugFix|Fix|Refactor|Enhancement|Doc|Chore|Bench|CI|Test|Perf"

# Full commit message / PR title regex
#   [Type] description   or   [Type][Scope] description
COMMIT_MSG_PATTERN="^\[(${COMMIT_PR_TYPES})\](\[[a-zA-Z0-9_-]+\])? .+"

# ---------------------------------------------------------------------------
# Branch naming
# ---------------------------------------------------------------------------

BRANCH_PREFIXES="feat|fix|refactor|doc|chore|perf|test|bench"
BRANCH_NAME_PATTERN="^(${BRANCH_PREFIXES})/[a-z0-9._-]+/[a-z0-9._-]+$"

# ---------------------------------------------------------------------------
# Type → GitHub label mapping
# ---------------------------------------------------------------------------

declare -A TYPE_TO_LABEL=(
  [Feat]=feature
  [BugFix]=fix
  [Fix]=fix
  [Refactor]=refactor
  [Enhancement]=enhancement
  [Doc]=docs
  [Chore]=chore
  [Bench]=bench
  [CI]=ci
  [Test]=test
  [Perf]=perf
)

# All type-related labels (used for stale-label cleanup)
ALL_TYPE_LABELS="feature fix refactor enhancement docs chore bench ci test perf"

# Extra labels (not derived from types)
EXTRA_LABELS="all ai powered|breaking change|help wanted|good first issue"

# ---------------------------------------------------------------------------
# Issue types (ALL CAPS, used in issue titles)
# ---------------------------------------------------------------------------

ISSUE_TYPES="FEAT|BUG|PERF|REFACTOR|DOCS|TEST|META"

# Issue type → GitHub label (for issue-label workflow)
declare -A ISSUE_TYPE_TO_LABEL=(
  [FEAT]=feature
  [BUG]=fix
  [PERF]=perf
  [REFACTOR]=refactor
  [DOCS]=docs
  [TEST]=test
  [META]=chore
)

# Issue type → commit/PR type prefix
declare -A ISSUE_TO_COMMIT_TYPE=(
  [BUG]=BugFix
  [FEAT]=Feat
  [PERF]=Enhancement
  [REFACTOR]=Refactor
  [DOCS]=Doc
  [TEST]=Test
  [META]=Chore
)

# Issue type → branch prefix
declare -A ISSUE_TO_BRANCH_PREFIX=(
  [BUG]=fix
  [FEAT]=feat
  [PERF]=perf
  [REFACTOR]=refactor
  [DOCS]=doc
  [TEST]=test
  [META]=chore
)

# Default for unrecognized issue types
ISSUE_DEFAULT_COMMIT_TYPE="Fix"
ISSUE_DEFAULT_BRANCH_PREFIX="fix"
