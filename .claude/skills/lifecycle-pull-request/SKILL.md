---
name: lifecycle-pull-request
description: Full PR lifecycle — commit, create PR, monitor CI, handle reviews. Use when you want end-to-end PR automation.
---

## When to use

- You want the full lifecycle automated: commit → create PR → wait for CI/review → fix → done.
- Invoke with: `/lifecycle-pull-request <task description>`

## Workflow

```text
Phase 1:   /committing-changes      →  BRANCH, COMMIT_MSG
Phase 2:   /creating-pull-request   →  PR_NUMBER, PR_URL (draft)
Phase 3-5: Poll-Handle Loop (CI + Gemini review, max 3 fix-push rounds)
Phase 6:   Mark PR ready for review →  triggers Copilot review + human notifications
Phase 7-9: Poll-Handle Loop (Copilot review, max 3 fix-push rounds)
```

### Draft-first strategy

PRs are created as **draft** (Phase 2). This defers Copilot code review and human reviewer notifications until CI passes and Gemini review is addressed. Phase 6 marks the PR ready, triggering the second round of reviews.

______________________________________________________________________

## Phase 1: Commit

Dispatch a **general-purpose subagent** (model=sonnet) via the Agent tool to execute the `committing-changes` skill:

> **Subagent prompt:**
>
> You are committing changes for this repository.
>
> **Background:** {task_context from \$ARGUMENTS}
>
> **Instructions:** Read and execute `.claude/skills/committing-changes/SKILL.md` step by step. Read `docs/CONTRIBUTING.md` first for conventions.
>
> **Return format** (report exactly):
>
> - `BRANCH: <branch-name>`
> - `COMMIT_MSG: <commit message>`
> - `FILES_CHANGED: <comma-separated list>`

Parse the result for: `BRANCH`, `COMMIT_MSG`, `FILES_CHANGED`.

If the subagent fails, report the error to the user and **stop**.

______________________________________________________________________

## Phase 2: Create PR

Dispatch a **general-purpose subagent** (model=sonnet) via the Agent tool to execute the `creating-pull-request` skill:

> **Subagent prompt:**
>
> You are creating a PR for this repository.
>
> **Background:** {task_context from \$ARGUMENTS}
>
> **Instructions:** Read and execute `.claude/skills/creating-pull-request/SKILL.md` step by step. Read `docs/CONTRIBUTING.md` first for conventions.
>
> **Return format** (report exactly):
>
> - `PR_NUMBER: <number>`
> - `PR_URL: <url>`
> - `BRANCH: <branch-name>`
> - `SUMMARY: <one-line summary>`

Parse the result for: `PR_NUMBER`, `PR_URL`.

If the subagent fails, report the error to the user and **stop**.

______________________________________________________________________

## Phase 3–5: Poll & Handle

After obtaining `PR_NUMBER`, enter the **poll-handle loop**.

> **do** { Phase 3 → Phase 4 → Phase 5 } **while** PR is not done
>
> Exit: CI green + all reviews handled, OR timeout/error, OR max 3 fix-push rounds.
>
> **Critical**: Every re-poll must check **both** CI status **and** review comments.

### Phase 3: Poll

Run the poll script as a **blocking** Bash call:

```bash
.claude/skills/lifecycle-pull-request/scripts/poll-pr-status.sh {owner}/{repo} {pr_number}
```

Use Bash tool with `timeout: 6060000` (covers the script's 100-minute timeout plus buffer).

The script returns structured JSON:

```json
{
  "status": "ready | timeout | error",
  "ci": {
    "state": "success | failure | pending",
    "failed_checks": [{"name": "...", "conclusion": "failure", "url": "..."}]
  },
  "reviews": {
    "new_inline_comments": [{"id": "...", "author": "...", "body": "...", "path": "...", "line": "..."}],
    "new_review_bodies": [{"id": "...", "author": "...", "body": "...", "state": "..."}],
    "unresolved_count": 3
  }
}
```

### Phase 4: Handle result

Parse the JSON. Follow this decision tree **in order** (first matching branch wins):

#### 4a. Timeout / error → STOP

If `status == "error"` or `status == "timeout"`:

> "PR #\{pr_number} — poll returned `{status}`: \{message}.
> You can retry later: `.claude/skills/lifecycle-pull-request/scripts/poll-pr-status.sh {owner}/{repo} {pr_number}`
> Or ask me to continue monitoring."

**You MUST stop and ask the user.** Do NOT silently fall back to manual polling. **Exit the loop.**

#### 4b. CI failure → auto fix

If `ci.state == "failure"`:

1. Identify failing checks from `ci.failed_checks`.
1. Reproduce locally when possible.
1. Fix by severity:
   - **Lint/format** (pre-commit, ruff, codespell, mdformat):
     Run `pre-commit run --all-files`, fix, commit, push → **re-poll**
   - **Test/build**:
     Analyze failure logs (`gh pr checks {pr_number} --repo {owner}/{repo}`)
     Simple fix → fix, commit, push → **re-poll**
     Complex → report to user with log summary, ask for guidance
1. Common CI issues:
   - Non-portable commands in docs (use `"$PWD"` not backticks)
   - Formatting drift (run `pre-commit run --all-files`)

#### 4c. New reviews → auto handle

If `reviews.new_inline_comments` or `reviews.new_review_bodies` are non-empty:

**Every comment MUST be replied to individually in its original thread.** Do NOT post a summary comment.

For each comment, classify and handle:

- **Simple/format** (typos, naming, imports): fix + reply with commit hash
- **Logic/architecture**: analyze, accept/decline with specific reason
- **Invalid/incorrect**: decline with specific technical reason

**How to reply** to a review comment:

```bash
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies \
  -f body="<reply>"
```

**After replying, resolve the thread** via GraphQL:

```bash
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "<PRRT_thread_node_id>"}) { thread { isResolved } } }'
```

Reply content rules:

- **Accepting**: `"Accepted. Fixed X. See {commit_hash}."`
- **Declining (future work)**: `"Declined for this PR — scope is limited to X. Tracked in #{issue}."` (also create a GitHub issue)
- **Declining (invalid)**: `"Declined. {specific technical reason}."`

### Phase 5: Verify done

**All** must be true:

1. `ci.state == "success"` — **not** "pending". If pending, keep polling (no round increment). Print "waiting for CI: \{check_name} still pending..." each cycle.
1. `reviews.new_inline_comments` is empty (or all replied to **and resolved**)
1. `reviews.new_review_bodies` is empty (or all replied to)

If done:

> "PR #\{pr_number} — draft phase complete:
>
> - CI: all checks passed
> - Review comments from draft phase: all addressed and resolved
> - Proceeding to Phase 6 (mark ready for review)."

**Exit the loop** and continue to Phase 6.

### Loop constraints

- **Max 3 fix-push rounds.** After 3 cycles still failing → escalate to user.
- Each "round" = one poll + one fix + one push. Passive waiting (pending CI) does NOT count.
- Re-poll after fix counts as the next round.

______________________________________________________________________

## Phase 6: Mark PR Ready for Review

After the draft poll-handle loop (Phase 3–5) completes with CI green and Gemini comments addressed, mark the PR as ready:

```bash
gh pr ready {pr_number} --repo {owner}/{repo}
```

This triggers:

- **Copilot code review** (ruleset: `review_on_push: true`, `review_draft_pull_requests: false`)
- **Required reviewer notifications** (human reviewers get pinged)

Report to the user:

> "PR #\{pr_number} marked as ready for review. CI is green and Gemini comments are addressed.
> Copilot review and human reviewer notifications are now triggered.
> Entering second poll-handle loop for Copilot review..."

______________________________________________________________________

## Phase 7–9: Poll & Handle (Copilot Review)

Re-enter a new poll-handle loop modeled on Phase 3–5, with the same logic but its own independent fix-push counter:

- **Phase 7: Poll** — same as Phase 3
- **Phase 8: Handle** — same as Phase 4 (CI fixes, review comment replies)
- **Phase 9: Verify done** — same as Phase 5

The same loop constraints apply: in this Copilot loop you may perform up to 3 fix-push rounds; passive waiting doesn't count. Do not carry over fix-push rounds spent in Phase 3–5 — reset the counter when starting Phase 7.

When the loop exits successfully:

> "PR #\{pr_number} is ready for human review:
>
> - CI: all checks passed
> - Gemini review: addressed (draft phase)
> - Copilot review: addressed (ready phase)
> - URL: \{pr_url}"
