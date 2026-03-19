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

COMMIT_PR_TYPES="Bench|BugFix|Chore|CI|Doc|Enhancement|Feat|Fix|Maintain|Perf|Refactor|Style|Test"

# Full commit message / PR title regex
#   [Type] description   or   [Type][Scope] description
COMMIT_MSG_PATTERN="^\[(${COMMIT_PR_TYPES})\](\[[a-zA-Z0-9_-]+\])? .+"

# ---------------------------------------------------------------------------
# Branch naming
# ---------------------------------------------------------------------------

BRANCH_PREFIXES="bench|chore|doc|feat|fix|maintain|perf|refactor|test"
BRANCH_NAME_PATTERN="^(${BRANCH_PREFIXES})/[a-z0-9._-]+/[a-z0-9._-]+$"

# ---------------------------------------------------------------------------
# Type → GitHub label mapping
# ---------------------------------------------------------------------------

declare -A TYPE_TO_LABEL=(
  [Bench]=bench
  [BugFix]=fix
  [Chore]=chore
  [CI]=ci
  [Doc]=docs
  [Enhancement]=enhancement
  [Feat]=feature
  [Fix]=fix
  [Maintain]=maintain
  [Perf]=perf
  [Refactor]=refactor
  [Style]=style
  [Test]=test
)

# All type-related labels (used for stale-label cleanup)
ALL_TYPE_LABELS="bench bug chore ci docs enhancement feature fix maintain perf refactor style test"

# Extra labels (not derived from types)
EXTRA_LABELS="all-ai-powered|human-led|breaking change|help wanted|good first issue"

# ---------------------------------------------------------------------------
# Issue types (ALL CAPS, used in issue titles)
# ---------------------------------------------------------------------------

ISSUE_TYPES="BENCHMARK|BUG|DOCS|FEAT|MAINTAIN|META|PERF|REFACTOR|TEST"

# Issue type → GitHub label (for auto-label workflow)
declare -A ISSUE_TYPE_TO_LABEL=(
  [BENCHMARK]=bench
  [BUG]=bug
  [DOCS]=docs
  [FEAT]=feature
  [MAINTAIN]=maintain
  [META]=chore
  [PERF]=perf
  [REFACTOR]=refactor
  [TEST]=test
)

# Issue type → commit/PR type prefix
declare -A ISSUE_TO_COMMIT_TYPE=(
  [BENCHMARK]=Bench
  [BUG]=BugFix
  [DOCS]=Doc
  [FEAT]=Feat
  [MAINTAIN]=Maintain
  [META]=Chore
  [PERF]=Enhancement
  [REFACTOR]=Refactor
  [TEST]=Test
)

# Issue type → branch prefix
declare -A ISSUE_TO_BRANCH_PREFIX=(
  [BENCHMARK]=bench
  [BUG]=fix
  [DOCS]=doc
  [FEAT]=feat
  [MAINTAIN]=maintain
  [META]=chore
  [PERF]=perf
  [REFACTOR]=refactor
  [TEST]=test
)

# Default for unrecognized issue types
ISSUE_DEFAULT_COMMIT_TYPE="Fix"
ISSUE_DEFAULT_BRANCH_PREFIX="fix"
