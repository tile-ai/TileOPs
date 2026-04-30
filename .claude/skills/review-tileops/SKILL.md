---
name: review-tileops
description: Single-shot review of a tile-ai/TileOPs PR as a separate GitHub identity from the PR author. Inline comments on code, agent-targeted summary. Enforces design consistency via project review checklists. For multi-round review until APPROVE, run `bash .claude/skills/review-tileops/loop.sh <PR>` instead.
---

## Input

`$ARGUMENTS`: integer PR number in `tile-ai/TileOPs`. E.g. `1122`.

## Step 0: Identity wiring

Caller must have run `bash .claude/skills/review-tileops/preflight.sh <PR>` once for this PR before invoking the skill (env var, `hosts.yml`, reviewer ≠ author, codex CLI all verified there).

```bash
export GH_CONFIG_DIR="$TILEOPS_REVIEW_GH_CONFIG_DIR"
```

## Step 1: Fetch PR

```bash
gh pr view <PR> --repo tile-ai/TileOPs \
  --json number,title,body,files,headRefOid,state
gh pr diff <PR> --repo tile-ai/TileOPs
```

Extract `title`, changed file list, full diff.

## Step 2: Classify PR → load checklists

Parse the title's first two bracket tokens — `[type][scope] description`. Look both up in `loading.yaml`:

- Always load every entry under `always:`.
- For each token, if it appears as a key under `match:`, load the listed checklist files.
- Multi-match → union load. Both tokens unmatched → no domain checklist; rely on the format spec + general judgment.

Read each checklist file under `.claude/review-checklists/`.

## Step 3: Read criteria + inputs

Read `criteria.md` (output format spec). Read every changed source file in full. Skim unresolved review threads, recent non-reviewer comments, and CI status if relevant.

## Step 4: Submit

Apply checklists to the diff. Compose inline comments + summary per `criteria.md` §2-§3. Submit one atomic review via `gh api` per `criteria.md` §1. Follow `criteria.md` §4 hard rules.

If the diff touches `tests/` and you intend to APPROVE, first read `.claude/review-checklists/approval-gate.md` and run every check it lists; downgrade to `REQUEST_CHANGES` if any fail.
