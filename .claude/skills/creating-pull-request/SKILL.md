---
name: creating-pull-request
description: Create a high-quality PR end-to-end (pre-checks, branch/commit, PR metadata, code review handling, CI fixes) based on TileOPs workflow
---

## When to use

- You have local changes and want Claude to turn them into a clean PR.
- You want a repeatable PR workflow: format/tests -> branch/commit -> PR -> review -> CI.
- You want the PR to match TileOPs conventions (branch naming, commands, and validation).

______________________________________________________________________

## Workflow

### Phase 0: Pre-checks (before making a PR)

**Goal**: avoid opening a PR that immediately fails formatting/tests.

0. If it’s not clear what the PR is fixing/adding, ask the user what issue/task this PR is for (link or short description).
1. **Read `docs/CONTRIBUTING.md`** to confirm the latest branch naming, commit message, and PR title conventions. Do NOT rely on other files (workflow configs, CI scripts) for naming rules — `CONTRIBUTING.md` is the single source of truth.
1. Confirm repository root and current git state:
   - `git status`
   - `git diff --stat`
1. Confirm Python environment is active (TileOPs example):
   - `conda activate top`
1. Run formatting/lint checks and tests following `Claude.md` → `DEVELOPMENT.md`.

Notes:

- Keep runs reproducible (fixed seed where applicable).
- Don’t fix unrelated failures in this PR; report them.

**Pre-Checklist**

- [ ] `pre-commit run --all-files` passes
- [ ] Tests pass (see `Claude.md` for exact commands)
- [ ] docs examples use portable env vars (prefer `PYTHONPATH="$PWD"`, avoid backticks like `pwd`)

______________________________________________________________________

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

3. Stage changes intentionally:

```bash
git add -p
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

- Follow `docs/CONTRIBUTING.md` (the single source of truth for naming conventions).
- Commit/PR titles use bracket format: `[Type] Description` or `[Type][Scope] Description`.
- See `Claude.md` § "Commit & PR Title Convention" for quick reference.

______________________________________________________________________

### Phase 2: Create PR (title/body/labels)

**Goal**: PR is self-contained and matches project expectations.

1. Create PR using the GitHub MCP tool (`create_pull_request`) or GitHub CLI:

```bash
gh pr create --base main --head <owner>:<branch> --title "<title>"
```

PR body:

- Leave the PR body for the user to fill in (interactively or edit after creation).

> **IMPORTANT (MCP tool pitfall):** When using the GitHub MCP `create_pull_request` or `update_pull_request` tools, the `body` parameter must use **actual newlines** (multi-line string), NOT `\n` escape sequences. Using `\n` will render as literal `\n` text in the PR body instead of line breaks.

2. PR title guidelines:

- **Must** use bracket format from `docs/CONTRIBUTING.md`: `[Type] Description` or `[Type][Scope] Description`.
- Examples: `[Feat][GEMV] Add forward kernel`, `[CI] Add pr-validation workflow`.
- Keep it under ~80 chars, describe the user-facing change.

3. PR body should include:

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

4. Add labels:

You can add labels via GitHub CLI or GitHub MCP. Use the following labels based on the PR content:

| Label Name       | Description                                |
| ---------------- | ------------------------------------------ |
| all ai powered   | Issue or Pull request all finished by AI   |
| breaking change  | Introduces backward-incompatible changes   |
| bug              | Something isn't working                    |
| ci               | -                                          |
| docs             | Documentation improvements                 |
| duplicate        | Duplicate of another issue                 |
| enhancement      | Improvement to existing functionality      |
| feature          | New feature or new operator                |
| fix              | Auto-created by issue labeler              |
| good first issue | Good for newcomers                         |
| help wanted      | Extra attention is needed                  |
| invalid          | Not a valid issue                          |
| question         | Further information is requested           |
| refactor         | Code restructuring without behavior change |
| wontfix          | This will not be worked on                 |

For example, to add the "all ai powered" label:

```bash
gh pr edit <PR_NUMBER> --add-label "all ai powered"
```

______________________________________________________________________

### Phase 3: Handling Automated Code Reviews

> **IMPORTANT**: This phase is **mandatory** after creating a PR. Do NOT consider the PR done until all review comments have been addressed.

After PR creation, **wait briefly then fetch all review feedback**. There are two types of comments to check:

1. **Inline review comments** (file-level suggestions):

   ```bash
   gh api repos/tile-ai/TileOPs/pulls/<PR_NUMBER>/comments
   ```

1. **PR-level reviews** (overall review body, may contain high-level feedback):

   ```bash
   gh api repos/tile-ai/TileOPs/pulls/<PR_NUMBER>/reviews
   ```

**Every review comment MUST be replied to individually in its original thread.** Do NOT post a summary comment — reply directly in the review conversation so each finding has a one-to-one traceable response.

For each inline comment (regardless of which bot posted it):

1. **Analyze validity** — compare against existing reference implementations and project conventions
1. **Reply in the original thread** via:
   ```bash
   gh api repos/tile-ai/TileOPs/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
     -f body="<reply>"
   ```
1. **Reply content rules**:
   - **If accepting**: state acceptance, describe what was fixed, reference the commit hash. Example: `Accepted. Added try/except with timeout. See abc1234.`
   - **If declining but valid for future work**: state decline with reason, then create a GitHub issue to track the suggestion. Example: `Declined for this PR — scope is limited to X. Tracked in #<issue>.`
   - **If declining as invalid**: state decline with a specific technical reason why the suggestion is unnecessary or incorrect. Example: `Declined. The API key check at L215 (shell level) runs before the Python heredoc, so the key is already validated.`
   - Never leave a review comment without a reply
1. **If accepting**: apply fixes → `pre-commit run` → `make` (if needed) → commit → push, then reply with the fix commit hash
1. **If the fix requires updating the PR description** (e.g., scope changed, new files added): update the PR body via `gh pr edit <PR_NUMBER> --body "..."`
1. **If accepting AND the finding reveals a novel pattern** (not already covered by existing guidelines): document the new lesson in the appropriate project documentation (e.g., `docs/CONTRIBUTING.md`). Criteria for novelty:
   - Not a duplicate of existing guideline items
   - Broadly applicable (not a one-off PR-specific issue)
   - Actionable (a reviewer can check for it in future code)

______________________________________________________________________

### Phase 4: Handle CI failures

**Goal**: fix CI efficiently without broad refactors.

1. Identify failing checks:

- `gh pr checks` (if available)
- or view logs in CI

2. Reproduce locally when possible:

- Run the same command CI runs (format/test)
- Use the same Python environment

3. Fix only the root cause:

Common doc/CI issues observed in practice:

- Non-portable commands in docs (use `"$PWD"` instead of backticks)
- Tool missing from PATH (use `python -m <tool>`)
- Formatting drift (run `pre-commit run --all-files`)

4. Validate and push:

- Re-run `pre-commit` and the relevant `pytest` subset
- Push fixes to the same branch; CI should rerun

______________________________________________________________________

## Done criteria

A PR is “done” when:

- pre-commit passes locally
- relevant tests pass locally
- PR title/body include validation instructions
- review feedback addressed (or explicitly resolved with rationale)
- CI is green (or failures are proven unrelated and documented)
