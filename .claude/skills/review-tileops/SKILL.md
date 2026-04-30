---
name: review-tileops
description: Single-shot review of a tile-ai/TileOPs PR as a separate GitHub identity from the PR author. Inline comments on code, agent-targeted summary. Enforces design consistency via project review checklists.
---

## Input

`$ARGUMENTS`: `<PR_NUMBER> [extra_prompt]`

- `PR_NUMBER` (required): integer PR number in `tile-ai/TileOPs`.
- `extra_prompt` (optional): free-form text after the PR number; treated as overriding guidance for this review per `criteria.md` §3.

## Step 0: Reviewer-identity preflight (HARD GATE)

```bash
if [ -z "$TILEOPS_REVIEW_GH_CONFIG_DIR" ]; then
  echo "error: TILEOPS_REVIEW_GH_CONFIG_DIR is not set." >&2
  echo "  reviewer identity must be separate from the PR author." >&2
  echo "  see .claude/skills/review-tileops/README.md" >&2
  exit 1
fi

if [ ! -f "$TILEOPS_REVIEW_GH_CONFIG_DIR/hosts.yml" ]; then
  echo "error: $TILEOPS_REVIEW_GH_CONFIG_DIR/hosts.yml not found." >&2
  echo "  run: GH_CONFIG_DIR=$TILEOPS_REVIEW_GH_CONFIG_DIR gh auth login --hostname github.com" >&2
  exit 1
fi

export GH_CONFIG_DIR="$TILEOPS_REVIEW_GH_CONFIG_DIR"
REVIEWER_LOGIN=$(gh api user --jq .login)
```

After Step 1 fetches the PR, also enforce reviewer ≠ author:

```bash
if [ "$REVIEWER_LOGIN" = "$PR_AUTHOR" ]; then
  echo "error: reviewer identity ($REVIEWER_LOGIN) equals PR author." >&2
  echo "  TILEOPS_REVIEW_GH_CONFIG_DIR points at your own gh config; reviewer/author must be distinct." >&2
  exit 1
fi
```

## Step 1: Fetch PR

```bash
gh pr view <N> --repo tile-ai/TileOPs \
  --json number,headRefName,baseRefName,title,body,url,files,headRefOid,state,author
gh pr diff <N> --repo tile-ai/TileOPs
```

Extract `number`, `title`, `author.login` (→ `PR_AUTHOR` for the Step 0 second check), changed file list, full diff.

## Step 2: Load criteria + checklists

Read `criteria.md` (colocated in this skill directory). Then per its §1, load the review checklists matching the PR.

## Step 3: Analyze

Apply criteria.md §2 priority order. If `extra_prompt` was provided in `$ARGUMENTS`, apply criteria.md §3 — `extra_prompt` overrides defaults.

## Step 4: Submit

Submit per criteria.md §4–§6 (one atomic review, inline format, summary format). Follow criteria.md §7 hard rules.
