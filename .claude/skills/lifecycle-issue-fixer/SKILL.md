---
name: lifecycle-issue-fixer
description: End-to-end issue resolution — reads a GitHub issue, creates a worktree if needed, explores code, implements via TDD, verifies, and creates a PR with full lifecycle management
---

## When to use

- You have a GitHub issue number and want to resolve it end-to-end
- You want automated TDD implementation + PR creation from an issue

______________________________________________________________________

## Arguments

This skill expects an issue reference as its argument:

- Issue number: `123`
- Full URL: `https://github.com/Tile-AI/TileOPs/issues/123`

If no argument is provided, ask the user for the issue number via `AskUserQuestion`.

______________________________________________________________________

## Workflow Overview

- **Phase 1: Read Issue** — fetch and understand the issue
- **Phase 2: Worktree** — smart detection, create worktree if needed
- **Phase 3: Explore Code** — understand affected code paths
- **Phase 4: TDD Implementation** — `Skill("superpowers:test-driven-development")`
- **Phase 5: Verify** — `Skill("superpowers:verification-before-completion")` + test suite + pre-commit
- **Phase 6: Commit, Create PR & Lifecycle** — `Skill("committing-changes")` → `Skill("creating-pull-request")` → poll CI/reviews

### Skill invocation rule

Every phase that references another skill **MUST** invoke it via the `Skill` tool. Do NOT attempt to re-implement the skill logic inline — the skills contain HARD GATES, validation scripts, and templates that must be executed.

______________________________________________________________________

## Phase 1: Read & Understand Issue

**Goal:** Parse the issue reference, fetch full issue data, and summarize understanding before proceeding.

### 1a. Parse issue reference

Extract the issue number from the skill arguments:

- If argument is a number (e.g. `123`), use it directly
- If argument is a URL (e.g. `https://github.com/Tile-AI/TileOPs/issues/123`), extract the number from the path
- If no argument is provided, ask the user via `AskUserQuestion`: "Which issue number should I work on?"

### 1b. Determine repository owner/repo

Run:

```bash
gh repo view --json nameWithOwner -q '.nameWithOwner'
```

Split the result into `{owner}` and `{repo}`.

### 1c. Fetch issue details

Use the GitHub MCP tool to read the issue — fetch all three in parallel:

```
issue_read(method="get", owner={owner}, repo={repo}, issue_number={number})
issue_read(method="get_comments", owner={owner}, repo={repo}, issue_number={number})
issue_read(method="get_labels", owner={owner}, repo={repo}, issue_number={number})
```

### 1d. Extract key information

From the issue title, body, labels, and comments, extract:

- **Type**: Parse `[TYPE]` from title (BUG, FEAT, PERF, REFACTOR, DOCS, TEST) — see Issue Type Mapping below
- **Component**: Parse `[COMPONENT]` from title if present
- **What**: The bug description or feature request
- **Where**: Any mentioned file paths, function names, error messages, or stack traces
- **Acceptance criteria**: Any checkboxes, expected behavior, or test cases described

### 1e. Summarize and confirm

Print a summary to the user:

> **Issue #\{number}: \{title}**
>
> **Type:** \{type} | **Component:** \{component}
>
> **Understanding:**
>
> - {what needs to be done}
> - {which code areas are affected}
> - {acceptance criteria}
>
> Proceeding to Phase 2.

### 1f. Guard rails

- If the issue is **not found** (404): report error and **stop**.
- If the issue is **closed**: warn the user and ask "This issue is already closed. Do you still want to proceed?"
- If the issue has **linked PRs** (check body for `#NNN` PR references): warn the user "This issue may already have PR #\{pr} addressing it. Proceed anyway?"

______________________________________________________________________

## Phase 2: Worktree (Smart Detection)

**Goal:** Ensure work happens in an isolated environment when appropriate, without creating unnecessary worktrees.

### Decision logic

Run these checks in order:

**Check 1: Already in a worktree?**

```bash
test -f .git
```

If `.git` is a file (not a directory), this is a secondary worktree — **skip** and use current environment.

**Check 2: On a feature branch with clean working tree?**

```bash
BRANCH=$(git branch --show-current)
STATUS=$(git status --porcelain)
```

If `$BRANCH` is not `main` AND `$STATUS` is empty, **skip** — the user is already on a clean feature branch.

**Check 3: On main OR dirty working tree → create worktree**

Use the `EnterWorktree` tool with a descriptive name:

```
EnterWorktree(name="issue-{number}")
```

This creates a worktree at `.claude/worktrees/issue-{number}` with a new branch based on HEAD.

### After worktree creation

Report to the user:

> Working in worktree: `.claude/worktrees/issue-{number}`

**Note:** The `committing-changes` skill (Phase 6a) will create a proper `{type}/scope/description` branch from main. The worktree branch is just a temporary working name.

______________________________________________________________________

## Phase 3: Explore Relevant Code

**Goal:** Build a thorough understanding of the affected code paths before writing any code.

### 3a. Extract search targets

From Phase 1's extracted information, identify:

- File paths mentioned in the issue
- Function/class names mentioned
- Error messages or stack traces
- Component name (e.g., GEMV → search `tileops/kernels/gemv/`, `tileops/ops/gemv.py`)

### 3b. Directed search (fast)

For each concrete search target, use Glob and Grep directly:

```
Glob("tileops/**/*{component}*")
Grep("{function_name}", path="tileops/")
Grep("{error_message}", path="tileops/")
```

### 3c. Deep exploration (if needed)

If the issue is complex or the directed search is insufficient, dispatch an `Agent(Explore)`:

```
Agent(
  subagent_type="Explore",
  prompt="Explore the {component} implementation in this codebase.
  I need to understand:
  1. How {component} is implemented (kernel + op layers)
  2. The data flow from op entry point to kernel execution
  3. Where {specific_problem} might occur
  4. Existing test coverage for {component}
  Focus on: {file_paths_from_issue}"
)
```

### 3d. Summarize findings

Before proceeding to implementation, summarize:

- Which files need to be modified
- The root cause (for bugs) or insertion points (for features)
- Existing test files that should be extended
- Any edge cases or related code that might be affected

______________________________________________________________________

## Phase 4: TDD Implementation

**Goal:** Implement the fix/feature using strict test-driven development.

### Invoke the TDD skill via Skill tool

**You MUST use the `Skill` tool** to invoke `superpowers:test-driven-development`:

```
Skill(skill="superpowers:test-driven-development")
```

Before invoking, print the context so the TDD skill has it in the conversation:

> **Issue:** #\{number} — \{title}
>
> **Problem:** {summary from Phase 1}
>
> **Affected code:** {findings from Phase 3}
>
> **Files to modify:** {list from Phase 3}
>
> **Test files:** {existing test files from Phase 3}

The TDD skill will enforce the red-green-refactor cycle:

1. Write a failing test that captures the expected behavior
1. Run it to confirm it fails
1. Write the minimal implementation to make it pass
1. Run tests to confirm they pass
1. Refactor if needed

**Do NOT commit in this phase** — committing is handled in Phase 6.

### Implementation guidelines

- Follow the project's 2-layer architecture: `kernel` → `op` (see `docs/DEVELOPMENT.md`)
- Keep dependency direction strict: `op -> kernel`, never the reverse
- Prefer minimal, targeted changes — do not refactor unrelated code
- Use the test command pattern: `PYTHONPATH="$PWD" python -m pytest -v {test_file}`

______________________________________________________________________

## Phase 5: Verify

**Goal:** Confirm the implementation is correct and causes no regressions before creating a PR.

### Invoke verification skill via Skill tool

**You MUST use the `Skill` tool** to invoke `superpowers:verification-before-completion`:

```
Skill(skill="superpowers:verification-before-completion")
```

This ensures all claims are backed by evidence — run the checks below as part of that skill.

### 5a. Run the full relevant test suite

Not just the new tests — run the entire test file and any related test files:

```bash
PYTHONPATH="$PWD" python -m pytest -v tests/ops/test_{component}.py
```

If there are kernel-level tests, run those too:

```bash
PYTHONPATH="$PWD" python -m pytest -v tests/kernels/test_{component}.py
```

### 5b. Run pre-commit

```bash
pre-commit run --all-files
```

If pre-commit fails:

- Auto-fix formatting issues (ruff, black, isort are typically auto-fixable)
- Re-run `pre-commit run --all-files` to confirm
- If it fails again on non-auto-fixable issues, fix manually

### 5c. Confirm no regressions

Verify:

- [ ] All pre-existing tests still pass
- [ ] New tests pass
- [ ] `pre-commit run --all-files` passes
- [ ] No untracked files that should be committed are left behind

### 5d. Report

Print a verification summary:

> **Verification complete:**
>
> - Tests: \{N} passed, \{M} failed
> - Pre-commit: {pass/fail}
> - Ready for PR: {yes/no}

If any verification fails, fix the issues before proceeding. If unable to fix after reasonable effort, escalate to the user.

______________________________________________________________________

## Phase 6: Commit, Create PR & Lifecycle

**Goal:** Commit changes, create a high-quality PR that closes the issue, then monitor CI and handle reviews.

This phase directly invokes the project's `committing-changes` and `creating-pull-request` skills, then runs the poll-handle loop from `lifecycle-pull-request`.

### 6a. Commit & Push — invoke `committing-changes` skill

**You MUST use the `Skill` tool** to invoke `committing-changes`:

```
Skill(skill="committing-changes", args="<context>")
```

Pass the following as args:

> Committing fix for issue #\{number}: \{title}.
> Branch type: `{branch_prefix}` (see Issue Type Mapping).
> Commit tag: `{pr_type_prefix}` (see Issue Type Mapping).

The skill will:

1. Sync with main, create a feature branch following `type/scope/description` convention
1. Stage specific files (never `git add .`)
1. Run pre-commit validation (HARD GATE)
1. Commit with proper `[Type] Description` format
1. Run post-commit validation (HARD GATE)
1. Push to origin

**Capture the output:** `BRANCH`, `COMMIT_MSG`, `FILES_CHANGED`.

If the skill fails, fix the reported issues and re-invoke. Do NOT proceed until push succeeds.

### 6b. Create PR — invoke `creating-pull-request` skill

**You MUST use the `Skill` tool** to invoke `creating-pull-request`:

```
Skill(skill="creating-pull-request", args="<context>")
```

Pass the following as args:

> PR for issue #\{number}: \{title}.
> PR title: `{pr_type_prefix} {description}` (see Issue Type Mapping).
> PR body MUST include: `Closes #{number}`.
> Labels: add the type label AND `all ai powered` label.

The skill will:

1. Create the PR via `gh pr create` (NOT the GitHub MCP tool)
1. Add required labels
1. Validate the PR (HARD GATE)

**Capture the output:** `PR_NUMBER`, `PR_URL`, `BRANCH`.

If the skill fails, fix the reported issues and re-invoke. Do NOT proceed until validation passes.

### 6c. Poll & Handle CI/Reviews (max 3 rounds)

After obtaining `PR_NUMBER`, enter the **poll-handle loop**. This logic is inlined from the `lifecycle-pull-request` skill.

> **do** { Poll → Handle → Verify } **while** PR is not done
>
> Exit: CI green + all reviews handled, OR timeout/error, OR max 3 fix-push rounds.
>
> **Critical**: Every re-poll must check **both** CI status **and** review comments.

#### Poll

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

#### Handle result

Parse the JSON. Follow this decision tree **in order** (first matching branch wins):

**Timeout / error → STOP:**
If `status == "error"` or `status == "timeout"`: report to user and **exit the loop**.

**CI failure → auto fix:**
If `ci.state == "failure"`:

1. Identify failing checks from `ci.failed_checks`.
1. **Lint/format** failures: run `pre-commit run --all-files`, fix, commit, push → **re-poll**
1. **Test/build** failures: analyze logs via `gh pr checks {pr_number} --repo {owner}/{repo}`. Simple fix → fix, commit, push → **re-poll**. Complex → escalate to user.

**New reviews → auto handle:**
If `reviews.new_inline_comments` or `reviews.new_review_bodies` are non-empty:

- **Every comment MUST be replied to individually in its original thread.** Do NOT post a summary comment.
- Reply via: `gh api repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies -f body="<reply>"`
- Resolve thread via GraphQL: `gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "<thread_id>"}) { thread { isResolved } } }'`

#### Verify done

**All** must be true:

1. `ci.state == "success"` — **not** "pending". If pending, keep polling (no round increment).
1. `reviews.new_inline_comments` is empty (or all replied to and resolved)
1. `reviews.new_review_bodies` is empty (or all replied to)

**Loop constraints:** Max 3 fix-push rounds. Passive waiting (pending CI) does NOT count as a round.

### Done

When the loop exits successfully, report to the user:

> **Issue #\{number} resolved:**
>
> - PR: \{pr_url}
> - Status: CI green, reviews addressed
> - The PR will auto-close issue #\{number} on merge.

______________________________________________________________________

## Error Handling

| Situation                                       | Behavior                                               |
| ----------------------------------------------- | ------------------------------------------------------ |
| No argument provided                            | Ask user for issue number via `AskUserQuestion`        |
| Issue not found (404)                           | Report error and **stop**                              |
| Issue is already closed                         | Warn user, ask "Proceed anyway?" via `AskUserQuestion` |
| Issue has linked PR already                     | Warn user, ask "Proceed anyway?" via `AskUserQuestion` |
| TDD tests won't pass after reasonable effort    | Escalate to user with what was tried and error details |
| Pre-commit fails after auto-fix retry           | Escalate to user                                       |
| `committing-changes` skill fails (HARD GATE)    | Fix reported issues, re-invoke the skill               |
| `creating-pull-request` skill fails (HARD GATE) | Fix reported issues (title/body/labels), re-invoke     |
| Poll-handle loop exceeds 3 rounds               | Escalate to user with failure summary                  |

______________________________________________________________________

## Issue Type Mapping

Derived from the issue title's `[TYPE]` tag. Used in Phase 6 for PR title prefix and branch naming.

Canonical mappings are defined in `.claude/conventions/types.sh` (`ISSUE_TO_COMMIT_TYPE` / `ISSUE_TO_BRANCH_PREFIX`).

| Issue `[TYPE]` | PR Title Prefix | Branch Prefix |
| -------------- | --------------- | ------------- |
| `BUG`          | `[BugFix]`      | `fix/`        |
| `FEAT`         | `[Feat]`        | `feat/`       |
| `PERF`         | `[Enhancement]` | `perf/`       |
| `REFACTOR`     | `[Refactor]`    | `refactor/`   |
| `DOCS`         | `[Doc]`         | `doc/`        |
| `TEST`         | `[Test]`        | `test/`       |
| `META`         | `[Chore]`       | `chore/`      |
| (not found)    | `[Fix]`         | `fix/`        |
