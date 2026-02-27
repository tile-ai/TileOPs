---
name: creating-pull-request
description: Create a high-quality PR end-to-end with automated lifecycle loop (pre-checks, branch/commit, PR metadata, review handling, CI fixes) based on TileOPs workflow
---

## When to use

- You need to turn local changes into a clean PR that follows TileOPs conventions.
- You want the full lifecycle automated: create PR → wait for CI/review → fix → done.

______________________________________________________________________

## Workflow

- **Phase 0–2: Create PR** — subagent (sonnet) runs Phase 0 → 1 → 2
  - Phase 0: Pre-checks (lint, tests)
  - Phase 1: Branch + commit + push
  - Phase 2: Create PR + labeling
- **Phase 3–5: Poll & Handle** — main agent loops until done (max 3 rounds)
  - Phase 3: Launch poll script (background, zero tokens)
  - Phase 4: Handle CI failures + review comments
  - Phase 5: Verify done (CI green + reviews handled → **DONE**, notify user)

______________________________________________________________________

## Phase 0–2: Create PR

Dispatch a subagent (sonnet-level or equivalent) to execute Phase 0 → 1 → 2. The subagent handles all pre-checks, branching, committing, pushing, and PR creation.

### Subagent spec

Spawn a general-purpose subagent with model=sonnet and the following prompt:

> You are creating a PR for `{owner}/{repo}`.
>
> **Background:** \{issue_description_or_task_context}
>
> **Steps:** Execute Phase 0 → 1 → 2 from the skill below. Read `docs/CONTRIBUTING.md` first for naming conventions.
>
> **Return format** (report exactly):
>
> - `PR_NUMBER: <number>`
> - `PR_URL: <url>`
> - `BRANCH: <branch-name>`
> - `SUMMARY: <one-line summary of changes>`

### Phase 0: Pre-checks

**Goal**: avoid opening a PR that immediately fails formatting/tests.

0. If it's not clear what the PR is fixing/adding, ask the user what issue/task this PR is for (link or short description).
1. **Read `docs/CONTRIBUTING.md`** to confirm the latest branch naming, commit message, and PR title conventions. Do NOT rely on other files (workflow configs, CI scripts) for naming rules — `CONTRIBUTING.md` is the single source of truth.
1. Confirm repository root and current git state:
   - `git status`
   - `git diff --stat`
1. Confirm Python environment is active (TileOPs example):
   - `conda activate top`
1. Run formatting/lint checks and tests following `Claude.md` → `DEVELOPMENT.md`.

Notes:

- Keep runs reproducible (fixed seed where applicable).
- Don't fix unrelated failures in this PR; report them.

**Pre-Checklist**

- [ ] `pre-commit run --all-files` passes
- [ ] Tests pass (see `Claude.md` for exact commands)
- [ ] docs examples use portable env vars (prefer `PYTHONPATH="$PWD"`, avoid backticks like `` `pwd` ``)

### Phase 1: Branch + commit

**Goal**: keep history clean and easy to review.

1. Sync `main` first:

```bash
git fetch origin
git switch main
git pull --ff-only
```

2. Create a new branch:

```bash
git switch -c <branch-name>
```

3. Stage changes intentionally (list specific files, avoid `git add .`):

```bash
git add <file1> <file2> ...
```

4. Create a focused commit:

```bash
git commit -m "[<Area>] <short summary>"
```

5. Push branch (always pull before push to incorporate any remote changes):

```bash
git pull origin <branch-name> --rebase
git push -u origin <branch-name>
```

Branch naming + commit conventions:

- **Branch name**: `type/scope/description` (e.g. `feat/flash-attn/fwd-kernel`, `fix/mla/parsing-error`)
- **Commit/PR title**: bracket format `[Type] Description` or `[Type][Scope] Description`
- Common types: `[Feat]`, `[BugFix]`, `[Refactor]`, `[Enhancement]`, `[Doc]`, `[Chore]`, `[Bench]`
- Follow `docs/CONTRIBUTING.md` as the single source of truth.

### Phase 2: Create PR + labeling

**Goal**: PR is self-contained and matches project expectations.

1. Create PR using the GitHub MCP tool (`create_pull_request`) or GitHub CLI:

```bash
gh pr create --base main --head <owner>:<branch> --title "<title>"
```

> **MCP pitfall:** When using the GitHub MCP `create_pull_request` or `update_pull_request` tools, the `body` parameter must use **actual newlines** (multi-line string), NOT `\n` escape sequences.

2. PR title guidelines:

- **Must** use bracket format from `docs/CONTRIBUTING.md`: `[Type] Description` or `[Type][Scope] Description`.
- Examples: `[Feat][GEMV] Add forward kernel`, `[CI] Add pr-validation workflow`.
- Keep it under ~80 chars, describe the user-facing change.

3. PR body template (fill in, or leave for user to edit after creation):

```markdown
Closes #<issue-number>

## Summary
- <what was migrated/added/fixed>
- <what was removed/replaced>
- <other notable changes>

## Test plan
- [x] pre-commit passed
- [x] pytest <N> passed
```

4. Add labels (if your repo uses them):

- docs / bug / benchmark / ci

If the PR was fully AI-driven (no user discussion, no user code edits — e.g., selfplay mode), add the `All AI powered` label:

```bash
gh pr edit <PR_NUMBER> --add-label "All AI powered"
```

______________________________________________________________________

## Phase 3–5: Poll & Handle

After the subagent returns `PR_NUMBER`, the main agent enters a **poll-handle loop**:

> **do** { Phase 3 → Phase 4 → Phase 5 } **while** PR is not done
>
> Exit conditions: CI green + all reviews handled, OR timeout, OR max 3 rounds.
>
> **Critical**: Every re-poll (Phase 3) must check **both** CI status **and** review comments. Never check only one. Review bots may post comments at any time — especially after the first CI run completes.

### Phase 3: Poll

**Dependencies**: The poll script requires `gh` (GitHub CLI, authenticated) and `jq`. If either is missing, the script returns a JSON error to stdout — proceed to Phase 4a (do NOT silently work around it).

Launch the poll script in background (**zero token cost** during the wait):

```bash
.claude/scripts/poll-pr-status.sh {owner}/{repo} {pr_number}
# Run this via your tool's background/async execution mode.
```

The script polls CI checks and review comments every 30 seconds for up to 10 minutes, then returns structured JSON:

```json
{
  "status": "ready | timeout | error",
  "ci": {
    "state": "success | failure | pending",
    "failed_checks": [{"name": "...", "conclusion": "failure", "url": "..."}]
  },
  "reviews": {
    "new_inline_comments": [{"id": ..., "author": "...", "body": "...", "path": "...", "line": ...}],
    "new_review_bodies": [{"id": ..., "author": "...", "body": "...", "state": "..."}],
    "unresolved_count": 3
  }
}
```

### Phase 4: Handle polling result

Parse the JSON and follow this decision tree **in order** (first matching branch wins):

#### 4a. Timeout / error → STOP and ask user

If `status == "error"` or `status == "timeout"`:

> "PR #\{pr_number} — poll returned `{status}`: \{message}.
> You can retry later: `.claude/scripts/poll-pr-status.sh {owner}/{repo} {pr_number}`
> Or ask me to continue monitoring."

**You MUST stop and ask the user.** Do NOT silently fall back to manual polling or alternative approaches. The user decides how to proceed.

**Exit the loop.**

#### 4b. CI failure → auto fix

If `ci.state == "failure"`, handle CI first (reviews will be re-checked after fix via re-poll).

1. Identify failing checks from `ci.failed_checks`

1. Reproduce locally when possible:

   - Run the same command CI runs (format/test)
   - Use the same Python environment

1. Fix by severity:

   - **Lint/format failures** (pre-commit, ruff, black, isort):

     - Dispatch a haiku subagent: `pre-commit run --all-files`, commit, push
     - → **re-poll** (back to Phase 3)

   - **Test/build failures**:

     - Main agent analyzes the failure logs (`gh pr checks` → get log URL)
     - Simple fix → fix, commit, push → **re-poll**
     - Complex/unclear → report to user with log summary, ask for guidance

1. Common CI issues:

   - Non-portable commands in docs (use `"$PWD"` instead of backticks)
   - Tool missing from PATH (use `python -m <tool>`)
   - Formatting drift (run `pre-commit run --all-files`)

#### 4c. New reviews → auto handle

If `reviews.new_inline_comments` or `reviews.new_review_bodies` are non-empty:

**Every review comment MUST be replied to individually in its original thread.** Do NOT post a summary comment.

For each comment, classify and handle:

- **Simple/format** (typos, naming, imports, docstring style):

  - Dispatch a haiku subagent to fix and reply in thread
  - → **re-poll** (back to Phase 3)

- **Logic/architecture**:

  - Main agent analyzes the suggestion against the codebase
  - Accept: fix + reply with commit hash
  - Decline: reply with specific technical reason
  - Uncertain: ask the user

- **Invalid/incorrect**:

  - Reply with a specific technical reason why the suggestion is unnecessary or incorrect

**How to reply** to a review comment:

```bash
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies \
  -f body="<reply>"
```

**After replying, resolve the thread** via GraphQL (use the thread's `PRRT_*` node ID, not the comment ID):

```bash
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "<thread_node_id>"}) { thread { isResolved } } }'
```

To get thread IDs, query `repository.pullRequest.reviewThreads`. Every replied-to thread **must** be resolved — an unresolved thread means the PR is not done.

**Reply content rules**:

- **Accepting**: `"Accepted. Fixed X. See {commit_hash}."`
- **Declining (future work)**: `"Declined for this PR — scope is limited to X. Tracked in #{issue}."` — also create a GitHub issue.
- **Declining (invalid)**: `"Declined. {specific technical reason}."`
- If accepting reveals a **novel pattern** (broadly applicable, not a duplicate, actionable): document it in `docs/CONTRIBUTING.md`.

### Phase 5: Verify done

**All** of these must be true to declare done:

1. `ci.state == "success"` — **not** "pending". If any check is still pending (e.g. waiting for a GPU runner), you **must keep waiting**. Print periodic "waiting for CI: \{check_name} still pending…" messages so the user sees progress, and continue polling. Do NOT skip pending checks or declare done early.
1. `reviews.new_inline_comments` is empty (or all replied to **and resolved**)
1. `reviews.new_review_bodies` is empty (or all replied to)

If done:

> "PR #\{pr_number} is ready for human review:
>
> - CI: all checks passed
> - Automated review comments: all addressed and resolved
> - URL: \{pr_url}"

**Exit the loop.**

If `ci.state == "pending"` and no failures/reviews to handle → keep polling (back to Phase 3). Print a waiting message each cycle. There is no max-rounds limit for passive waiting — only fix-push cycles count toward the 3-round limit.

> **Note on `unresolved_count`**: The poll script counts all non-author inline comments (REST API lacks thread resolution state). The agent should verify it has replied to every comment in both `new_inline_comments` and `new_review_bodies` before considering the PR done.

### Loop constraints

- **Max 3 rounds.** After 3 fix-poll cycles still failing → escalate to user with status summary.
- Each "round" = one poll + one fix + one push. Re-poll counts as the next round.

______________________________________________________________________

## Done criteria

A PR is "done" when:

- CI is green (**all** checks passed, none pending)
- All automated review comments have been replied to **and resolved**
- User is notified that PR is ready for human review
