---
name: lifecycle-pull-request
description: Full PR lifecycle — commit, create PR, monitor CI, handle reviews. Use when you want end-to-end PR automation.
---

## When to use

- You want the full lifecycle automated: commit → create PR → wait for CI/review → fix → done.
- Invoke with: `/lifecycle-pull-request <task description>`

## Arguments

\$ARGUMENTS

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

Use Bash tool with `timeout: 18060000` (covers the script's 5-hour timeout plus buffer).

The script returns structured JSON with one of four statuses:

- **`actionable`** — CI failed or unresolved reviews exist. Handler must process, then re-poll.
- **`done`** — CI all success + all threads resolved. Handler skips to Phase 6.
- **`timeout`** — CI still pending, nothing to handle yet. Re-poll in next round.
- **`error`** — Script-level failure (bad args, auth, etc.). Stop and ask user.

```json
{
  "status": "actionable | done | timeout | error",
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

#### 4a. `done` → proceed to Phase 6

If `status == "done"`: CI is green and all threads are resolved. **Skip to Phase 6** (mark PR ready).

#### 4b. Timeout / error → re-poll or STOP

If `status == "timeout"`: CI is still pending. **Re-poll** (this does NOT count as a fix-push round).

If `status == "error"`:

> "PR #\{pr_number} — poll returned error: \{message}.
> You can retry later: `.claude/skills/lifecycle-pull-request/scripts/poll-pr-status.sh {owner}/{repo} {pr_number}`"

**You MUST stop and ask the user.** Do NOT silently fall back to manual polling. **Exit the loop.**

#### 4c. `actionable` with CI failure → auto fix

If `status == "actionable"` and `ci.state == "failure"`:

1. Identify failing checks from `ci.failed_checks`.
1. Reproduce locally when possible.
1. Fix by severity:
   - **Lint/format** (pre-commit, ruff, codespell, mdformat):
     Run `pre-commit run --all-files` to auto-fix, or fix manually. Then commit and push → **re-poll**
   - **Test/build**:
     Analyze failure logs (`gh pr checks {pr_number} --repo {owner}/{repo}`)
     Simple fix → fix, commit, push → **re-poll**
     Complex → report to user with log summary, ask for guidance

#### 4d. `actionable` with unresolved reviews → auto handle (reply + resolve atomic flow)

If `status == "actionable"` and `reviews.new_inline_comments` is non-empty (unresolved threads exist):

The poll script exposes **all** unresolved threads (including author-started ones) so that `unresolved_count` and the actionable thread list stay consistent. Each thread follows a deterministic handling path based on its type.

**For each unresolved thread, execute these steps in order:**

**Step 1 — Read** the full comment chain (all entries in `comments` array). Determine the thread type:

- **Type A — Reviewer thread**: at least one comment is from a non-PR-author (e.g., `copilot[bot]`, `gemini[bot]`, a human reviewer)
- **Type B — Author-only thread**: all comments are from the PR author

**Step 2 — Handle based on type:**

**If Type A (reviewer thread):**

1. Classify the feedback: accept (fix + commit hash), decline (reason), or defer (create issue).
1. **Reply** to the thread:

```bash
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies \
  -f body="<reply>"
```

Use the `id` from the **last comment** in the thread's `comments` array as `comment_id`.

Reply content rules:

- **Accepting**: `"Accepted. Fixed X. See {commit_hash}."`
- **Declining (future work)**: `"Declined for this PR — scope is limited to X. Tracked in #{issue}."` (also create a GitHub issue)
- **Declining (invalid)**: `"Declined. {specific technical reason}."`

**If Type B (author-only thread):**

No reply needed. Proceed directly to Step 3.

**Step 3 — Resolve** the thread (both types):

```bash
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "<PRRT_thread_node_id>"}) { thread { isResolved } } }'
```

**Step 4 — Handle resolve failure**: If the mutation fails:

- Log: "Failed to resolve thread \{thread_node_id}: \{error_message}"
- **Continue processing remaining threads** — do NOT stop the loop
- The unresolved thread will be picked up on the next poll cycle
- If the same thread fails across multiple rounds, escalate to the user

**For `reviews.new_review_bodies`** (PR-level reviews with body text):

- Read and address the feedback
- Reply via `gh api repos/{owner}/{repo}/issues/{pr_number}/comments -f body="<response>"`

#### 4e. Post-fix verification

After fixing CI or review issues, run local tests before pushing:

```bash
.claude/skills/lifecycle-issue-fixer/scripts/run-affected-tests.sh docs/plans/issue-{N}-context.json
```

If the context file is not available (standalone PR lifecycle without issue-fixer), skip this step.

Then commit and push the fixes, and **re-poll** (go back to Phase 3).

### Phase 5: Verify done

The poll script handles completion detection. When it returns `status == "done"`:

- CI: all checks passed
- All review threads resolved
- No pending review bodies

Report:

> "PR #\{pr_number} — draft phase complete:
>
> - CI: all checks passed
> - Copilot review: all comments addressed and resolved
> - Gemini review: all comments addressed and resolved
> - Proceeding to Phase 6 (mark ready for review)."

**Exit the loop** and continue to Phase 6.

### Loop constraints

**No fixed round limit.** The loop continues until `done` or an exit condition is met.

**Script-level exit conditions** (handled by `poll-pr-status.sh`):

- `done` — CI success + all threads resolved → proceed to Phase 6
- `timeout` — 5 hours (18000s) elapsed with CI still pending → report to user
- `error` — 5 consecutive fetch failures → report to user

**Agent-level exit conditions** (you must track across rounds):

- **Same CI check name fails 3 times** and you cannot fix it (e.g., upstream test failure on main) → report to user with evidence that the failure pre-exists on main
- **Same review issue reappears 3 times** after you've addressed it → report to user
- In both cases, explain what was tried and why the issue is beyond this PR's scope

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
