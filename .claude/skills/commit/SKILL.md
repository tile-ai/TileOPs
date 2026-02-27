---
name: commit
description: Create a compliant git commit for TileOPs — validates commit message format, runs pre-commit, and commits cleanly.
---

## When to use

- You have staged changes and want to commit following TileOPs conventions.
- You want to ensure `pre-commit` passes before the commit is recorded.
- You want the commit message validated against the TileLang Commit Convention.

______________________________________________________________________

## Commit Message Convention

Source of truth: `docs/CONTRIBUTING.md` § "Step 3: Commit".

Format: `[Type] Description` or `[Type][Scope] Description`

**Valid Types** (case-sensitive, PascalCase):

| Type          | Use                                    |
| ------------- | -------------------------------------- |
| `Feat`        | New features or operators              |
| `BugFix`      | Bug fixes                              |
| `Refactor`    | Code restructuring, no behavior change |
| `Enhancement` | Improvements to existing features      |
| `Doc`         | Documentation updates                  |
| `Chore`       | Build system or workflow changes       |
| `Bench`       | Benchmark updates                      |
| `Lint`        | Formatting / linting only              |
| `CI`          | CI/CD pipeline changes                 |
| `Test`        | Test additions or fixes                |

**Rules**:

- Must start with `[Type]` or `[Type][Scope]` — brackets are required.
- Type must be one of the valid types above (exact casing).
- Description must follow immediately after the closing `]`, separated by a space.
- Description must not be empty and should be ≤ 72 chars total for the subject line.
- No trailing period on the subject line.

**Valid examples**:

```
[Feat] Add multi-head attention forward op
[BugFix] Fix index out of bounds in reduction kernel
[Enhancement][MHA] Improve forward op performance on Hopper
[Chore][CI] Update CUDA version to 12.9
[Doc] Update README.md
```

**Invalid examples**:

```
feat: add something          # wrong format, no brackets
[feat] add something         # wrong casing
[Feat]add something          # missing space after ]
Add something                # no type prefix
[Unknown] do stuff           # unrecognized type
```

______________________________________________________________________

## Workflow

### Phase 0: Validate environment

```bash
git status
git diff --cached --stat
```

- Confirm there are staged changes. If nothing is staged, stop and ask the user what to stage.
- Confirm the Python environment is active (venv, conda, etc.) and dependencies are installed.

### Phase 1: Run pre-commit

```bash
pre-commit run --all-files
```

- If any hook fails, fix the issues and re-stage before proceeding.
- Do NOT skip hooks with `--no-verify`.
- Common fixes:
  - Formatting: `ruff format .` then re-stage
  - Lint: `ruff check --fix .` then re-stage
  - Spelling: fix the flagged words in `docs/spelling_wordlist.txt` or the source file

### Phase 2: Validate commit message

Before running `git commit`, verify the message matches the convention above.

Use the installed `commit-msg` hook (see `scripts/hooks/commit-msg`) — it runs automatically on `git commit`.

If you are constructing the message programmatically, validate it yourself:

```python
import re

# Keep this in sync with VALID_TYPES in scripts/hooks/commit-msg
VALID_TYPES = r"Feat|BugFix|Refactor|Enhancement|Doc|Chore|Bench|Lint|CI|Test"
pattern = re.compile(rf"^\[({VALID_TYPES})\](\[[\w\-]+\])? .+")
assert pattern.match(msg), f"Invalid commit message: {msg}"
```

### Phase 3: Commit

```bash
git commit -m "[Type][Scope] Short description"
```

- The `commit-msg` hook will reject non-compliant messages automatically.
- If the hook rejects the message, fix the format and retry.

______________________________________________________________________

## Checklist

- [ ] `git diff --cached --stat` shows the intended changes
- [ ] `pre-commit run --all-files` passes with no failures
- [ ] Commit message matches `[Type]` or `[Type][Scope]` format
- [ ] Type is one of the valid types (exact casing)
- [ ] Subject line ≤ 72 characters, no trailing period

______________________________________________________________________

## Installing the hooks (one-time setup)

```bash
pre-commit install                          # installs pre-commit hook
pre-commit install --hook-type commit-msg   # installs commit-msg hook
```

Both must be installed for full validation.
