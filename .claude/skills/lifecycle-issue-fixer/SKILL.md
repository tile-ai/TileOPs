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

- **Phase 1: Read Issue** — fetch, validate template, create context.json
- **Phase 2: Worktree** — smart detection, create worktree if needed
- **Phase 3: Explore Code** — understand affected code paths, update context.json
- **Phase 4: TDD Implementation** — red-green-refactor with HARD GATE enforcement
- **Phase 5: Verify** — run tests + check acceptance criteria
- **Phase 6: PR Lifecycle** — `Skill("lifecycle-pull-request")`

### Skill invocation rule

Every phase that references another skill **MUST** invoke it via the `Skill` tool. Do NOT attempt to re-implement the skill logic inline — the skills contain HARD GATES, validation scripts, and templates that must be executed.

______________________________________________________________________

## Phase 1: Read & Understand Issue

**Goal:** Parse the issue reference, fetch full issue data, validate the template, and create the context file.

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

### 1d. HARD GATE — Validate issue template

The issue body **MUST** contain these sections:

- `## Description` — problem description
- `## Goal` — completion objective
- `## Plan` — execution route (with `proposal` or `fixed` type)
- `## Constraints` — hard constraints (may be empty)
- `## Acceptance Criteria` — verifiable conditions

**If any section is missing: STOP.** Report to the user:

> "Issue #\{number} is missing required sections: \{list}. Please update the issue body with the required template before proceeding."

Do NOT proceed until the template is valid.

### 1e. Extract key information

From the issue title, body, labels, and comments, extract:

- **Type**: Parse `[TYPE]` from title (BUG, FEAT, PERF, REFACTOR, DOCS, TEST, META, BENCHMARK) — see Issue Type Mapping below
- **Component**: Parse `[COMPONENT]` from title if present
- **Plan type**: Parse `<!-- type: proposal | fixed -->` comment from Plan section. Default: `proposal`
- **Goal**: Content of `## Goal` section
- **Execution route**: Steps listed in `## Plan` section
- **Constraints**: Content of `## Constraints` section (may be empty)
- **Acceptance criteria**: Checkboxes from `## Acceptance Criteria` section

### 1f. Create context file

Write `docs/plans/issue-{number}-context.json`:

```json
{
  "issue": {number},
  "type": "{TYPE}",
  "component": "{COMPONENT}",
  "plan_type": "proposal",
  "goal": "{goal text}",
  "acceptance_criteria": ["{AC-1}", "{AC-2}"],
  "execution_route": ["{step-1}", "{step-2}"],
  "constraints": ["{constraint-1}"],
  "affected_files": [],
  "test_targets": [],
  "bench_targets": [],
  "code_understanding": ""
}
```

### 1g. Summarize and confirm

Print a summary to the user:

> **Issue #\{number}: \{title}**
>
> **Type:** \{type} | **Component:** \{component} | **Plan:** \{plan_type}
>
> **Goal:** \{goal}
>
> **Execution route:** \{steps}
>
> **Acceptance criteria:** {AC list}
>
> Proceeding to Phase 2.

### 1h. Claim the issue

Before any work begins, check the issue's assignees:

```bash
gh issue view {number} --repo {owner}/{repo} --json assignees --jq '.assignees[].login'
```

- **No assignees:** Assign yourself by running `gh issue edit {number} --repo {owner}/{repo} --add-assignee @me`, then proceed.
- **Assigned to current user:** Already claimed, proceed.
- **Assigned to someone else:** **STOP.** Report to the user: "Issue #\{number} is assigned to @\{assignee}. Will not claim — another contributor is working on it." Do NOT proceed unless the user explicitly overrides.

### 1i. Guard rails

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

### 3d. Update context file

Update `docs/plans/issue-{number}-context.json` with:

- `affected_files`: list of files that need to be modified
- `code_understanding`: summary of key findings (root cause, insertion points, edge cases)

### 3e. Summarize findings

Before proceeding to implementation, summarize:

- Which files need to be modified
- The root cause (for bugs) or insertion points (for features)
- Existing test files that should be extended
- Any edge cases or related code that might be affected

______________________________________________________________________

## Phase 4: TDD Implementation

**Goal:** Implement the fix/feature using strict test-driven development.

### Plan enforcement

Read `plan_type` from context.json:

- **`fixed`**: Follow `execution_route` steps **strictly and in order**. Do not deviate.
- **`proposal`**: Use `execution_route` as guidance but explore autonomously. The TDD cycle below is still mandatory.

### HARD GATE — TDD red-green-refactor cycle

Implementation **MUST** follow this sequence. Skipping any step is not permitted.

**1. Write or update tests first**

- Create/modify test files for the affected functionality
- Update context.json `test_targets` with the test file paths
- If benchmarks are affected (kernel changes), update `bench_targets` too

**2. Confirm FAIL (red)**

```bash
.claude/skills/lifecycle-issue-fixer/scripts/run-affected-tests.sh docs/plans/issue-{number}-context.json
```

The new/modified tests **MUST fail**. If they pass without any implementation changes, the tests are not testing the right thing — revise them.

**3. Write the minimal implementation**

- Follow the project's 2-layer architecture: `kernel` → `op` (see `docs/DEVELOPMENT.md`)
- Keep dependency direction strict: `op -> kernel`, never the reverse
- Prefer minimal, targeted changes — do not refactor unrelated code

**4. Confirm PASS (green)**

```bash
.claude/skills/lifecycle-issue-fixer/scripts/run-affected-tests.sh docs/plans/issue-{number}-context.json
```

All tests and benchmarks in context.json **MUST pass**.

**5. Refactor if needed**

If code can be improved without changing behavior:

- Refactor
- Run tests again to confirm still PASS

**Do NOT commit in this phase** — committing is handled in Phase 6.

______________________________________________________________________

## Phase 5: Verify (Independent Verification Subagent)

**Goal:** Confirm the implementation is correct and causes no regressions before creating a PR. Verification is performed by an **independent subagent** — the implementer (main agent) does not judge its own work.

### 5a. Dispatch verification subagent

The main agent **MUST** dispatch a subagent to perform verification. The subagent's sole input is `context.json` — it reads `acceptance_criteria`, `test_targets`, and `bench_targets` from this file and independently verifies each criterion.

```
Task(
  subagent_type="general-purpose",
  prompt="You are an independent verification agent. Your job is to verify that
  the implementation satisfies ALL acceptance criteria defined in the context file.

  Context file: docs/plans/issue-{number}-context.json

  Read the context file and verify EACH acceptance criterion independently.
  For each criterion, determine the appropriate verification method and execute it.
  Report pass/fail with concrete evidence for every criterion.

  See the Verification Examples section below for guidance on common verification patterns."
)
```

### 5b. Verification examples

The subagent determines how to verify each AC based on its content. Below are common patterns:

#### Example: Unit test verification

AC: `"Modified files pass unit tests"`

```bash
# Run the test targets listed in context.json
PYTHONPATH="$PWD" python -m pytest -v tests/ops/test_gemv.py --tb=short -q
```

Evidence: `"8 passed, 0 failed"`

#### Example: Benchmark execution and data collection

AC: `"GEMV kernel benchmark runs successfully and performance data is collected"`

```bash
# Run the benchmark target listed in context.json
PYTHONPATH="$PWD" python -m pytest -v benchmarks/ops/bench_gemv.py --tb=short -q 2>&1
```

To extract performance data from benchmark output, look for lines containing metrics like `TFLOPS`, `GB/s`, `latency`, `ms`, `us`, etc. Format results as a markdown table:

```markdown
| Shape | Metric | Value |
|-------|--------|-------|
| M=4096, N=4096, K=4096 | Throughput | 523.4 TFLOPS |
| M=8192, N=8192, K=8192 | Throughput | 498.1 TFLOPS |
```

This table should be included in the verification report so the main agent can embed it in the PR body's `## Benchmark` section.

#### Example: Performance threshold verification

AC: `"GEMV throughput >= 500 TFLOPS on H200"`

Run the benchmark as above, extract the throughput value, and compare against the threshold. Evidence: `"Measured 523.4 TFLOPS >= 500 TFLOPS threshold — PASS"` or `"Measured 480.2 TFLOPS < 500 TFLOPS threshold — FAIL"`.

#### Example: API compatibility verification

AC: `"Public API signature unchanged"`

```bash
# Check that the function signature matches expected
python -c "import inspect; from tileops.ops.gemv import gemv; print(inspect.signature(gemv))"
```

Evidence: `"Signature: (A, B, bias=None) — matches expected"`

#### Other criteria

For criteria not covered by the examples above, the subagent should use its judgment to determine the appropriate verification method — run commands, read files, check outputs, etc. The key requirement is that every criterion must have **concrete evidence** (command output, file content, measured values), not just assertions.

### 5c. HARD GATE — Subagent must pass

The main agent reads the subagent's verification report:

- **ALL criteria pass**: Proceed to Phase 6.
- **Any criterion fails**: Fix the failing items, update context.json if needed, and re-dispatch the verification subagent. Do NOT proceed until all criteria pass.
- **Benchmark data collected**: Save the benchmark table for inclusion in the PR body.

### 5d. Report

Print the verification summary from the subagent:

> **Verification complete (independent subagent):**
>
> - Tests: \{N} passed, \{M} failed
> - Benchmarks: \{N} passed, \{M} failed
> - Acceptance criteria: \{X}/\{Y} met
> - Benchmark data: {table or "N/A"}
> - Ready for PR: {yes/no}

If any verification fails, fix the issues before proceeding. If unable to fix after reasonable effort, escalate to the user.

______________________________________________________________________

## Phase 6: PR Lifecycle

**Goal:** Commit changes, create a high-quality PR that closes the issue, then monitor CI and handle reviews.

Invoke the `lifecycle-pull-request` skill:

```
Skill(skill="lifecycle-pull-request", args="Issue #{number}: {goal}. Context: docs/plans/issue-{number}-context.json")
```

The skill handles: commit → create PR → trigger Gemini → poll-handle loop → mark ready.

### Done

When the lifecycle-pull-request skill completes, report to the user:

> **Issue #\{number} resolved:**
>
> - PR: \{pr_url}
> - Status: CI green, Copilot and Gemini reviews addressed
> - The PR will auto-close issue #\{number} on merge.

______________________________________________________________________

## Error Handling

| Situation                                | Behavior                                               |
| ---------------------------------------- | ------------------------------------------------------ |
| No argument provided                     | Ask user for issue number via `AskUserQuestion`        |
| Issue not found (404)                    | Report error and **stop**                              |
| Issue missing template sections          | Report missing sections and **stop**                   |
| Issue assigned to someone else           | Report and **stop** — do not claim another's work      |
| Issue is already closed                  | Warn user, ask "Proceed anyway?" via `AskUserQuestion` |
| Issue has linked PR already              | Warn user, ask "Proceed anyway?" via `AskUserQuestion` |
| TDD red phase passes unexpectedly        | Revise tests — they are not testing the right thing    |
| Tests won't pass after reasonable effort | Escalate to user with what was tried and error details |
| `lifecycle-pull-request` skill fails     | Fix reported issues, re-invoke the skill               |
| Poll-handle loop exceeds 3 rounds        | Escalate to user with failure summary                  |

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
