---
name: committing-changes
description: Pre-check, branch, commit, and push changes following TileOPs conventions. Use when you need to create a clean commit from working tree changes.
---

## Current git state

- Status: !`git status --short`
- Diff stat: !`git diff --stat`
- Current branch: !`git branch --show-current`

## Task

\$ARGUMENTS

## Reference

Read `docs/CONTRIBUTING.md` before proceeding for branch naming and commit message conventions.

For commit message format, see [template.txt](template.txt).

**Source of truth** for type lists, regex patterns, and label mappings: `.claude/conventions/types.sh`.

## Steps

Execute these steps in order. **Do NOT skip any HARD GATE.**

### Step 1: Sync with main

```bash
git fetch origin
git switch main
git pull --ff-only
```

### Step 2: Create feature branch

Branch name format: `type/scope/description` (all lowercase, hyphens for separators).

```bash
git switch -c <type>/<scope>/<description>
```

Type mapping:

- New feature → `feat/`
- Bug fix → `fix/`
- Refactor → `refactor/`
- Documentation → `doc/`
- CI/build → `chore/`
- Performance → `perf/`
- Tests → `test/`
- Benchmarks → `bench/`

### Step 3: Stage changes

Stage specific files only. **Never use `git add .` or `git add -A`.**

```bash
git add <file1> <file2> ...
```

### Step 4: HARD GATE — Pre-commit validation

```bash
.claude/skills/committing-changes/scripts/validate.sh --pre
```

**If exit code != 0: STOP.** Fix the reported issues, re-stage, and re-run this gate. Do NOT proceed to Step 5 until this passes.

### Step 5: Commit

Use the format from [template.txt](template.txt):

```bash
git commit -m "[Type] Short description"
```

### Step 6: HARD GATE — Post-commit validation

```bash
.claude/skills/committing-changes/scripts/validate.sh --post
```

**If exit code != 0: STOP.** The commit message or branch name is wrong. Fix with `git commit --amend` or rename the branch, then re-run this gate.

### Step 7: Push

```bash
git push -u origin <branch-name>
```

## Return format

Report exactly:

- `BRANCH: <branch-name>`
- `COMMIT_MSG: <commit message>`
- `FILES_CHANGED: <comma-separated list>`
