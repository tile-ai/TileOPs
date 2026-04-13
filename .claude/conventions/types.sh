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

COMMIT_PR_TYPES="Bench|BugFix|Chore|CI|Design|Doc|Enhancement|Feat|Fix|Maintain|Perf|Refactor|Style|Test"

# Full commit message / PR title regex
#   [Type] description   or   [Type][Scope] description
COMMIT_MSG_PATTERN="^\[(${COMMIT_PR_TYPES})\](\[[a-zA-Z0-9_-]+\])? .+"

# ---------------------------------------------------------------------------
# Branch naming
# ---------------------------------------------------------------------------

BRANCH_PREFIXES="bench|chore|design|doc|feat|fix|maintain|perf|refactor|style|test"
BRANCH_NAME_PATTERN="^(${BRANCH_PREFIXES})/[a-z0-9._-]+/[a-z0-9._-]+$"

# ---------------------------------------------------------------------------
# Type → GitHub label mapping
# ---------------------------------------------------------------------------

declare -A TYPE_TO_LABEL=(
  [Bench]=bench
  [BugFix]=fix
  [Chore]=chore
  [CI]=ci
  [Design]=design
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
ALL_TYPE_LABELS="bench bug chore ci design docs enhancement feature fix maintain perf refactor style test"

# ---------------------------------------------------------------------------
# Issue types (ALL CAPS, used in issue titles)
# ---------------------------------------------------------------------------

ISSUE_TYPES="BENCHMARK|BUG|DESIGN|DOCS|FEAT|MAINTAIN|META|PERF|REFACTOR|STYLE|TEST"

# Issue type → GitHub label (for auto-label workflow)
declare -A ISSUE_TYPE_TO_LABEL=(
  [BENCHMARK]=bench
  [BUG]=bug
  [DESIGN]=design
  [DOCS]=docs
  [FEAT]=feature
  [MAINTAIN]=maintain
  [META]=chore
  [PERF]=perf
  [REFACTOR]=refactor
  [STYLE]=style
  [TEST]=test
)

# Issue type → commit/PR type prefix
declare -A ISSUE_TO_COMMIT_TYPE=(
  [BENCHMARK]=Bench
  [BUG]=BugFix
  [DESIGN]=Design
  [DOCS]=Doc
  [FEAT]=Feat
  [MAINTAIN]=Maintain
  [META]=Chore
  [PERF]=Enhancement
  [REFACTOR]=Refactor
  [STYLE]=Style
  [TEST]=Test
)

# Issue type → branch prefix
declare -A ISSUE_TO_BRANCH_PREFIX=(
  [BENCHMARK]=bench
  [BUG]=fix
  [DESIGN]=design
  [DOCS]=doc
  [FEAT]=feat
  [MAINTAIN]=maintain
  [META]=chore
  [PERF]=perf
  [REFACTOR]=refactor
  [STYLE]=style
  [TEST]=test
)

# Default for unrecognized issue types
ISSUE_DEFAULT_COMMIT_TYPE="Fix"
ISSUE_DEFAULT_BRANCH_PREFIX="fix"
