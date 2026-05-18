---
name: review-tileops
description: Single-shot review of a tile-ai/TileOPs PR as a separate GitHub identity from the PR author. Manual / interactive — for autonomous multi-round review until APPROVE, run `bash .claude/skills/review-tileops/loop.sh <PR>` instead.
---

## Input

`$ARGUMENTS`: integer PR number in `tile-ai/TileOPs`. E.g. `1122`.

## Step 0: Preflight

```bash
bash .claude/skills/review-tileops/preflight.sh <PR> || exit 1
export GH_CONFIG_DIR="$TILEOPS_REVIEW_GH_CONFIG_DIR"
```

## Step 1: Gather inputs

```bash
gh pr view <PR> --repo tile-ai/TileOPs \
  --json number,title,body,files,headRefOid,state
gh pr diff <PR> --repo tile-ai/TileOPs
```

Parse the title's first two bracket tokens — `[type][scope] description`. Look both up in `loading.yaml`:

- Always load entries under `always:`.
- For each token, if it appears as a key under `match:`, load the listed checklist files.
- Multi-match across the two tokens → union. Both unmatched → no domain checklist; rely on general judgment.

Read each loaded checklist file under `.claude/review-checklists/`, plus `criteria.md`. Skim unresolved review threads (`gh api graphql … reviewThreads`), recent non-reviewer comments (`gh api repos/.../issues/<N>/comments` and `…/pulls/<N>/comments`), and CI status (`gh pr checks <N>`) if relevant to the diff.

## Step 2: Execute the review

Follow `procedure.md` end to end. Submit one atomic review at the end.
