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

## Convergence guarantees

The autonomous loop (`loop.sh`) layers two deterministic, non-LLM rules on top of the per-round Codex review. They run in `round-pre.sh` (before Codex) and `round-post.sh` (after Codex) respectively, and do not exit the loop on their own.

- **Rule 1 — same-SHA APPROVE skip** (`round-pre.sh`): before each round, the loop scans prior `round-*.json` entries in this loop run for any APPROVE whose `head_sha_after` matches the current HEAD sha. If it finds one, the round is short-circuited: Codex is not invoked, a marker `round-NN.json` is written that reuses the prior outcome (`approve_reused_from`, `skipped_codex: true`), and the loop continues normally. This eliminates the "re-review the same commit and contradict yourself" failure mode without ever overriding a fresh negative review.
- **Rule 2 — same-path 3-strike monitor** (`round-post.sh`): after each round, the loop extracts blocker paths from `round-NN.codex-blockers.json` (the post-codex snapshot of review comments authored by `REVIEWER_LOGIN` this round, filtered against the previous reviewer-comment watermark) and updates `review/region-history.json`, incrementing per-path counters for paths present this round and resetting counters for paths absent. When any path's counter reaches three consecutive blocker rounds, the loop ensures the PR carries the `agent-stuck` label (created idempotently in the repo if missing). The hook never mutates blockers, threads, comments, or the PR title — the label is the only externally-visible side effect. Missing `region-history.json` is treated as empty state, so this rule is backward compatible with older runs.
