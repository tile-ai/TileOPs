---
name: resolve-tileops
description: Address all reviewer comments on a PR — accept and fix reasonable feedback, reject with rationale otherwise. Reply inline on every comment.
---

## Input

`$ARGUMENTS` is an integer PR number in `tile-ai/TileOPs`. E.g. `1081`.
The PR URL is `https://github.com/tile-ai/TileOPs/pull/<N>`; derive it
when needed.

## Constants

- `REPO=tile-ai/TileOPs`

## Step 1: Fetch PR and checkout branch

```bash
gh pr view "$PR" --repo "$REPO" \
  --json number,headRefName,baseRefName,title,body,url,headRepository,headRepositoryOwner
```

Extract `headRefName`. Checkout the branch locally. For cross-fork PRs
(head repo != base repo), ensure the remote for the fork exists and the
branch is checked out from it.

## Step 2: Collect review threads AND review summaries (single GraphQL call)

```bash
gh api graphql -f query='
  query($owner:String!, $repo:String!, $pr:Int!) {
    repository(owner:$owner, name:$repo) {
      pullRequest(number:$pr) {
        reviews(first:50) {
          nodes {
            databaseId
            author { login }
            body
            state
          }
        }
        reviewThreads(first:100) {
          nodes {
            id
            isResolved
            comments(first:20) {
              nodes {
                databaseId
                author { login }
                body
                path
                line
              }
            }
          }
        }
      }
    }
  }' -f owner=tile-ai -f repo=TileOPs -F pr="$PR"
```

From the result, build:

- **Review summaries**: extract non-empty `body` from `reviews.nodes` (state = `CHANGES_REQUESTED` or `COMMENTED`). Parse each summary for actionable items — bullet points, numbered lists, or imperative statements requesting changes.
- **Thread list**: each thread's `id` (node_id for resolve mutation), `isResolved`, root comment, all replies.
- **comment_id → thread_id map**: for resolving later.
- **Unresolved threads**: filter to threads where `isResolved == false` and no reply from the PR author exists.

Skip threads that are already resolved or already have an author reply.

**Important**: review summaries often raise issues not covered by any inline comment. Treat each actionable item from a summary as an additional feedback item to triage, alongside inline threads.

## Step 3–5: Triage, fix, reply

Follow `.claude/skills/resolve-tileops/procedure.md` end to end for triage → fix → reply → resolve. Reply formats and hard rules live in `.claude/skills/resolve-tileops/criteria.md`.

## Step 6: Report

```
| # | Source   | Reviewer | File | Verdict | Summary |
|---|----------|----------|------|---------|---------|
| 1 | inline   | ...      | ...  | Accept  | ...     |
| 2 | summary  | ...      | —    | Accept  | ...     |
```

Source is `inline` (review thread) or `summary` (review body). File is `—` for summary-only items.

`Pushed N fix(es) in <sha>. M rejected, K deferred.`

## Rules

- Reply inline on the thread. Never post top-level PR comments unless something cross-cutting can't fit inline (per `criteria.md`).
- Do not modify files no reviewer commented on, unless a fix requires it.
- If a suggestion would introduce a bug or regression, reject and explain.
- If the branch has diverged locally, `git pull` first.
- All replies in **English**, concise — conclusion, action, reasoning. No filler.
