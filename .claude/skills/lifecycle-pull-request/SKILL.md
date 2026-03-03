---
name: lifecycle-pull-request
description: Full PR lifecycle — commit, create PR, monitor CI, handle reviews. Use when you want end-to-end PR automation.
---

## When to use

- You want the full lifecycle automated: commit → create PR → wait for CI/review → fix → done.
- Invoke with: `/lifecycle-pull-request <task description>`

## Arguments

$ARGUMENTS

## Workflow

```text
Phase 1:   Skill("committing-changes")   →  BRANCH, COMMIT_MSG
Phase 2:   Skill("creating-pull-request") →  PR_NUMBER, PR_URL (draft)
Phase 2b:  Trigger Gemini review          →  comment `/gemini review` on the PR
Phase 3-5: Poll-Handle Loop (CI + Copilot + Gemini reviews, max 3 fix-push rounds)
Phase 6:   Mark PR ready for review       →  triggers human reviewer notifications
```

### Draft-first strategy

PRs are created as **draft** (Phase 2). On draft PRs:

- **Copilot** reviews automatically (configured via `review_on_push: true`, `review_draft_pull_requests: true`).
- **Gemini** does **not** review drafts automatically — it must be triggered by commenting `/gemini review` on the PR (Phase 2b).

The draft phase handles CI + both bot reviews. Phase 6 marks the PR ready, which triggers **human reviewer notifications only** (both bot reviews are already complete).

### Skill invocation rule

Every phase that references another skill **MUST** invoke it via the `Skill` tool. Do NOT use subagent dispatch or attempt to re-implement the skill logic inline.

______________________________________________________________________

## Phase 1: Commit

**You MUST use the `Skill` tool** to invoke `committing-changes`:

```
Skill(skill="committing-changes", args="<task_context from $ARGUMENTS>")
```

The skill will:

1. Detect current branch (skip sync-main if already on feature branch)
1. Stage specific files (never `git add .`)
1. Run pre-commit validation (HARD GATE)
1. Commit with proper `[Type] Description` format
1. Run post-commit validation (HARD GATE)
1. Push to origin

**Capture the output:** `BRANCH`, `COMMIT_MSG`, `FILES_CHANGED`.

If the skill fails, fix the reported issues and re-invoke. Do NOT proceed until push succeeds.

______________________________________________________________________

## Phase 2: Create PR

**You MUST use the `Skill` tool** to invoke `creating-pull-request`:

```
Skill(skill="creating-pull-request", args="<task_context from $ARGUMENTS>")
```

The skill will:

1. Create the PR via `gh pr create` (NOT the GitHub MCP tool)
1. Add required labels
1. Validate the PR (HARD GATE)

**Capture the output:** `PR_NUMBER`, `PR_URL`.

If the skill fails, fix the reported issues and re-invoke. Do NOT proceed until validation passes.

______________________________________________________________________

## Phase 2b: Trigger Gemini Review

Gemini does not automatically review draft PRs. Trigger it by commenting on the PR:

```bash
gh pr comment {pr_number} --repo {owner}/{repo} --body "/gemini review"
```

This must happen **after** the PR is created (Phase 2) and **before** entering the poll-handle loop (Phase 3).

______________________________________________________________________

## Phase 3–5: Poll & Handle

After obtaining `PR_NUMBER`, enter the **poll-handle loop**.

> **do** { Phase 3 → Phase 4 → Phase 5 } **while** PR is not done
>
> Exit: CI green + all reviews handled, OR timeout/error, OR max 3 fix-push rounds.
>
> **Critical**: Every re-poll must check **both** CI status **and** review comments.

### Context recovery

If context has been compressed (long-running workflow), read the context file to recover issue understanding:

```
Read("docs/plans/issue-{N}-context.json")
```

This file contains the issue goal, acceptance criteria, execution route, affected files, and code understanding — everything needed to make informed decisions when handling review feedback.

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
    "new_inline_comments": [
      {
        "thread_node_id": "PRRT_kwDOABC123",
        "is_resolved": false,
        "is_outdated": false,
        "comments": [
          {"id": 12345, "author": "copilot[bot]", "body": "...", "path": "...", "line": 42, "created_at": "..."}
        ]
      }
    ],
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
     Fix the issue, commit, push → **re-poll**
   - **Test/build**:
     Analyze failure logs (`gh pr checks {pr_number} --repo {owner}/{repo}`)
     Simple fix → fix, commit, push → **re-poll**
     Complex → report to user with log summary, ask for guidance

#### 4c. New reviews → auto handle (reply + resolve atomic flow)

If `reviews.new_inline_comments` is non-empty (unresolved threads exist):

**Every unresolved thread MUST be handled with an atomic reply+resolve two-step.**

For each unresolved thread:

1. Read the **full comment chain** in the thread (all entries in `comments` array) to understand the complete conversation — not just the first comment.
2. Classify and decide: accept (fix + commit hash), decline (reason), or defer (create issue).
3. **Step 1 — Reply** to the thread:

```bash
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies \
  -f body="<reply>"
```

Use the `id` from the **last comment** in the thread's `comments` array as `comment_id`.

4. **Step 2 — Resolve** the thread using the `thread_node_id`:

```bash
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "<PRRT_thread_node_id>"}) { thread { isResolved } } }'
```

Reply content rules:

- **Accepting**: `"Accepted. Fixed X. See {commit_hash}."`
- **Declining (future work)**: `"Declined for this PR — scope is limited to X. Tracked in #{issue}."` (also create a GitHub issue)
- **Declining (invalid)**: `"Declined. {specific technical reason}."`

For `reviews.new_review_bodies` (PR-level reviews with body text):

- Read and address the feedback
- Reply via `gh api repos/{owner}/{repo}/issues/{pr_number}/comments -f body="<response>"`

#### 4d. Post-fix verification

After fixing CI or review issues, run local tests before pushing:

```bash
.claude/skills/lifecycle-issue-fixer/scripts/run-affected-tests.sh docs/plans/issue-{N}-context.json
```

If the context file is not available (standalone PR lifecycle without issue-fixer), skip this step.

Then commit and push the fixes.

### Phase 5: Verify done

**All** must be true:

1. `ci.state == "success"` — **not** "pending". If pending, keep polling (no round increment). Print "waiting for CI: \{check_name} still pending..." each cycle.
1. `reviews.unresolved_count == 0` (all threads resolved via the atomic reply+resolve flow)
1. `reviews.new_review_bodies` is empty (or all replied to)

If done:

> "PR #\{pr_number} — draft phase complete:
>
> - CI: all checks passed
> - Copilot review: all comments addressed and resolved
> - Gemini review: all comments addressed and resolved
> - Proceeding to Phase 6 (mark ready for review)."

**Exit the loop** and continue to Phase 6.

### Loop constraints

- **Max 3 fix-push rounds.** After 3 cycles still failing → escalate to user.
- Each "round" = one poll + one fix + one push. Passive waiting (pending CI) does NOT count.
- Re-poll after fix counts as the next round.

______________________________________________________________________

## Phase 6: Mark PR Ready for Review

After the draft poll-handle loop (Phase 3–5) completes with CI green and all review threads resolved, mark the PR as ready:

```bash
gh pr ready {pr_number} --repo {owner}/{repo}
```

This triggers **human reviewer notifications** only — both bot reviews (Copilot and Gemini) were already completed and addressed during the draft phase.

Report to the user:

> "PR #\{pr_number} is ready for human review:
>
> - CI: all checks passed
> - Copilot review: addressed and resolved
> - Gemini review: addressed and resolved
> - URL: \{pr_url}"
