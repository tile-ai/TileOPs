---
name: creating-pull-request
description: Create a PR with correct title, body, and labels following TileOPs conventions. Use when you have a pushed branch ready for PR creation.
---

## Current state

- Repo: !`gh repo view --json nameWithOwner -q '.nameWithOwner'`
- Branch: !`git branch --show-current`
- Commits on branch: !`git log --oneline main..HEAD 2>/dev/null || echo "(no commits vs main)"`
- Remote: !`git remote -v | head -2`

## Task

\$ARGUMENTS

## Reference

Read `docs/CONTRIBUTING.md` before proceeding. It is the **single source of truth** for PR title conventions.

For PR body format, see [template.md](template.md).

## Steps

Execute these steps in order. **Do NOT skip any HARD GATE.**

### Step 1: Determine owner/repo

```bash
gh repo view --json nameWithOwner -q '.nameWithOwner'
```

Split into `{owner}` and `{repo}`.

### Step 2: Create the PR

Use `gh pr create`. **Do NOT use the GitHub MCP tool** — `gh` CLI avoids the `\n` pitfall.

```bash
gh pr create --base main --head "$(git branch --show-current)" \
  --title "[Type] Description" \
  --body "$(cat <<'PREOF'
<fill in from template.md>
PREOF
)"
```

PR title rules:

- **Must** use bracket format: `[Type] Description` or `[Type][Scope] Description`
- Types: `[Feat]`, `[BugFix]`, `[Fix]`, `[Refactor]`, `[Enhancement]`, `[Doc]`, `[Chore]`, `[Bench]`, `[CI]`
- Keep under ~80 chars

PR body section rules:

- `## Summary` — always required
- `## Test plan` — always required
- `## Benchmark` — required when PR involves performance changes
- `## Regression` — recommended when PR is bugfix or refactor
- `## Additional context` — optional
- **Delete** inapplicable optional sections entirely. Never leave empty headers.

### Step 3: Add labels (MANDATORY)

**At least one label is required.** Select based on PR type:

| PR Type              | Label             |
| -------------------- | ----------------- |
| `[Feat]`             | `feature`         |
| `[BugFix]` / `[Fix]` | `bug` or `fix`    |
| `[Enhancement]`      | `enhancement`     |
| `[Refactor]`         | `refactor`        |
| `[Doc]`              | `documentation`   |
| `[Chore]` / `[CI]`   | `ci` or `chore`   |
| Breaking change      | `breaking change` |

Additional labels (stack as needed):

- `all ai powered` — PR fully AI-generated (no user code edits)
- `help wanted` — needs extra attention
- `good first issue` — suitable for newcomers

```bash
gh pr edit <PR_NUMBER> --add-label "<label1>" --add-label "<label2>"
```

### Step 4: HARD GATE — Validate PR

```bash
.claude/skills/creating-pull-request/scripts/validate.sh <owner/repo> <PR_NUMBER>
```

**If exit code != 0: STOP.** Fix the reported issues:

- Wrong title → `gh pr edit <N> --title "[Type] New title"`
- Missing body section → `gh pr edit <N> --body "$(cat <<'EOF' ... EOF)"`
- Missing labels → `gh pr edit <N> --add-label "<label>"`

Re-run the gate until it passes.

## Return format

Report exactly:

- `PR_NUMBER: <number>`
- `PR_URL: <url>`
- `BRANCH: <branch-name>`
- `SUMMARY: <one-line summary>`
