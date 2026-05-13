#!/usr/bin/env bash
# Run once per PR, before the first review round, to verify reviewer-identity
# wiring. Subsequent rounds (whether single-shot or loop-driven) skip these
# checks — the skill itself assumes preflight has passed.
set -euo pipefail

PR="${1:?usage: preflight.sh <PR_NUMBER>}"
REPO="tile-ai/TileOPs"

if ! command -v codex >/dev/null 2>&1; then
  echo "error: codex CLI not found in PATH." >&2
  echo "  the review loop runs each round under Codex; install it from https://github.com/openai/codex" >&2
  exit 1
fi

if [ -z "${TILEOPS_REVIEW_GH_CONFIG_DIR:-}" ]; then
  echo "error: TILEOPS_REVIEW_GH_CONFIG_DIR is not set." >&2
  echo "  export it in your shell rc; see .claude/skills/review-tileops/README.md" >&2
  exit 1
fi

if [ ! -f "$TILEOPS_REVIEW_GH_CONFIG_DIR/hosts.yml" ]; then
  echo "error: $TILEOPS_REVIEW_GH_CONFIG_DIR/hosts.yml not found." >&2
  echo "  run: GH_CONFIG_DIR=$TILEOPS_REVIEW_GH_CONFIG_DIR gh auth login --hostname github.com" >&2
  exit 1
fi

REVIEWER=$(GH_CONFIG_DIR="$TILEOPS_REVIEW_GH_CONFIG_DIR" gh api user --jq .login)
AUTHOR=$(GH_CONFIG_DIR="$TILEOPS_REVIEW_GH_CONFIG_DIR" gh pr view "$PR" --repo "$REPO" --json author --jq .author.login)

if [ "$REVIEWER" = "$AUTHOR" ]; then
  echo "error: reviewer ($REVIEWER) equals PR #$PR author." >&2
  echo "  TILEOPS_REVIEW_GH_CONFIG_DIR points at the author's gh config; reviewer must be distinct." >&2
  exit 1
fi

echo "OK: reviewer=$REVIEWER  PR #$PR author=$AUTHOR  (distinct)"
