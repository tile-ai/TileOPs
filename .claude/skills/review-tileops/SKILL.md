---
name: review-tileops
description: Single-shot review of a tile-ai/TileOPs PR as a separate GitHub identity from the PR author. Inline comments on code, agent-targeted summary. Enforces design consistency via project review checklists.
---

## Input

`$ARGUMENTS`: `<PR_NUMBER> [extra_prompt]`

- `PR_NUMBER` (required): integer PR number in `tile-ai/TileOPs`.
- `extra_prompt` (optional): free-form text after the PR number; treated as overriding guidance for this review per `criteria.md` §3.

## Step 0: Identity wiring

The caller must have run `bash .claude/skills/review-tileops/preflight.sh <N>` once for this PR before invoking the skill (env var, `hosts.yml`, and reviewer ≠ author are verified there). Skip if already validated this session.

```bash
export GH_CONFIG_DIR="$TILEOPS_REVIEW_GH_CONFIG_DIR"
```

## Step 1: Fetch PR

```bash
gh pr view <N> --repo tile-ai/TileOPs \
  --json number,headRefName,baseRefName,title,body,url,files,headRefOid,state
gh pr diff <N> --repo tile-ai/TileOPs
```

Extract `number`, `title`, changed file list, full diff.

## Step 2: Load criteria + checklists

Read `criteria.md` (colocated in this skill directory). Then per its §1, load the review checklists matching the PR.

## Step 3: Analyze

Apply criteria.md §2 priority order. If `extra_prompt` was provided in `$ARGUMENTS`, apply criteria.md §3 — `extra_prompt` overrides defaults.

## Step 4: Submit

Submit per criteria.md §4–§6 (one atomic review, inline format, summary format). Follow criteria.md §7 hard rules.
