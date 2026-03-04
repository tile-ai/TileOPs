---
name: creating-issue
description: Create a high-quality issue which conforms to the rules of TileOPs development
---

## Task

\$ARGUMENTS

## Reference

- Title format: `[TYPE][COMPONENT] short description in lowercase`
- TYPE values: FEAT, BUG, PERF, REFACTOR, DOCS, TEST, META (canonical list in `.claude/conventions/types.sh`)
- COMPONENT: kernel name or subsystem (e.g., GEMV, GEMM, FLASH_ATTN, CI, TOOLING)
- Language: **English** only

## Steps

Execute these steps in order. **Do NOT skip any HARD GATE.**

### Step 1: Determine repository

```bash
gh repo view --json nameWithOwner -q '.nameWithOwner'
```

Split into `{owner}` and `{repo}`.

### Step 2: Parse arguments

Extract structured arguments from `$ARGUMENTS` if present:

- `--plan "<type>: <steps>"` — e.g., `--plan "fixed: 1. Change X 2. Update Y"`
- `--constraints "<text>"` — e.g., `--constraints "Do not modify public API"`
- `--criteria "<text>"` — e.g., `--criteria "Performance improves by 10%"`

Everything else in `$ARGUMENTS` (not prefixed with `--`) is the natural language description.

### Step 3: Build title

Format: `[TYPE][COMPONENT] short description in lowercase`

- Extract TYPE and COMPONENT from the description or ask user if ambiguous
- Keep description concise (under 80 chars total)

**Do NOT use Conventional Commits style** (`feat(scope): ...`).

### Step 4: Build issue body from template

Assemble the body using either parsed arguments or smart defaults:

```markdown
## Description
### Symptom / Motivation
{What is observed or what motivates this change}

### Root Cause Analysis
{Why it happens — file paths, logic errors, missing features, etc. Use "N/A" for feature requests}

### Related Files
{Key files, functions, or configurations involved}

## Goal
{extracted from description, or ask user if unclear}

## Plan
<!-- type: {proposal | fixed} -->
{if --plan provided: use that content}
{if no --plan: default to proposal, infer steps from description}

## Constraints
{if --constraints provided: use that content}
{if no --constraints: leave section empty}

## Acceptance Criteria
- [ ] Modified files pass unit tests
{if --criteria provided: add as additional checkboxes}
```

**Smart defaults when no arguments are provided:**

1. Goal: extracted from description
1. Plan: `proposal` type, steps inferred from description
1. Constraints: empty
1. Acceptance Criteria: `"Modified files pass unit tests"` (always included as default)

### Step 5: HARD GATE — Validate template completeness

Before creating, verify ALL required sections are present and non-empty:

- `## Description` — must contain `### Symptom / Motivation` (non-empty)
- `## Goal` — must not be empty
- `## Plan` — must contain at least one step (line starting with `- ` or `1.`)
- `## Acceptance Criteria` — must contain at least one checkbox (`- [ ]`)

**If any section is missing or empty: STOP.** Fix the body before proceeding.

### Step 6: Create the issue

```bash
gh issue create --repo {owner}/{repo} \
  --title "[TYPE][COMPONENT] description" \
  --body "$(cat <<'ISSUEEOF'
<assembled body from Step 4>
ISSUEEOF
)"
```

### Step 7: Add labels

Add the type label matching the issue TYPE:

| Issue TYPE    | Label         |
| ------------- | ------------- |
| `FEAT`        | `feature`     |
| `BUG`         | `fix`         |
| `PERF`        | `perf`        |
| `REFACTOR`    | `refactor`    |
| `DOCS`        | `docs`        |
| `TEST`        | `test`        |
| `META`        | `chore`       |
| `ENHANCEMENT` | `enhancement` |
| `BENCH`       | `bench`       |
| `CI`          | `ci`          |

```bash
gh issue edit {issue_number} --repo {owner}/{repo} --add-label "{label}"
```

### Step 8: Report

Print:

> **Issue #\{number} created:** \{url}
>
> - Title: \{title}
> - Type: \{TYPE} | Component: \{COMPONENT}
> - Plan: {proposal|fixed}
> - Labels: \{labels}

## Return format

Report exactly:

- `ISSUE_NUMBER: <number>`
- `ISSUE_URL: <url>`
- `TITLE: <title>`

______________________________________________________________________

## Guidelines (for reference)

### When to file an issue before a PR

Always file an issue first when:

- The change involves a non-trivial performance optimization
- The root cause analysis is worth documenting separately from the code diff
- You want upstream visibility before investing in a full PR

For trivial typo fixes or single-line changes, a PR without a prior issue is fine.

### Case log

| Date       | Issue                                                                 | Note                                                                                                     |
| ---------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 2026-02-27 | [[PERF][GEMV] #232](https://github.com/tile-ai/TileOPs/issues/232)    | GEMV uncoalesced B access on H200; title initially filed as `perf(gemv): ...`, corrected post-submission |
| 2026-02-27 | [[META][TOOLING] #233](https://github.com/tile-ai/TileOPs/issues/233) | Upstream issue requesting formal title format documentation in CONTRIBUTING.md                           |
