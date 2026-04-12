# Pre-Create Checks

## 1. PR title pre-flight validation

**HARD GATE.** Validate the PR title locally against the same source of truth that CI uses, **before** calling `gh pr create`. This guarantees the `validate-pr-title` required check cannot fail.

```bash
source .claude/conventions/types.sh
TITLE="[{{Type}}] {{Description}}"   # substitute your actual title
if [[ ! "$TITLE" =~ $COMMIT_MSG_PATTERN ]]; then
  echo "BLOCKED: title does not match CI pattern: $TITLE"
  echo "Pattern: $COMMIT_MSG_PATTERN"
fi
```

**If validation fails:** fix the title and re-validate. Do NOT proceed to `gh pr create`.

## 2. Op/kernel readiness

**Skip entirely** if PR does not touch kernel/op code.

If PR introduces or modifies a kernel/op:

1. Read [op-readiness-checklist.md](op-readiness-checklist.md).
1. Verify every item → `PASS`, `FAIL (reason)`, or `SKIP (reason)`.

**HARD GATE:** Any `[REQUIRED]` FAIL blocks the PR. `[RECOMMENDED]` may be skipped with reason. In `## Structural Readiness`, list only FAILs and SKIPs. All pass → "All checks passed."

## 3. Test node delta

**Skip entirely** if PR does not modify files under `tests/`.

If PR adds or modifies test files:

```bash
git fetch upstream main --quiet
python scripts/test_node_delta.py --base upstream/main
```

The script auto-detects changed test files via `git diff`. If auto-detect fails (e.g. in a worktree), pass files explicitly:

```bash
python scripts/test_node_delta.py --base upstream/main tests/ops/test_<name>.py
```

**Interpreting output:**

- **No growth on existing files** → nothing to report.
- **Growth on existing files** → include script output and one-line justification in PR body under `## Test node delta`.
- **New test files only** (delta shows "all N nodes from new files") → no delta report needed.

**SOFT GATE:** Does not block PR creation, but missing justification will be flagged during review (per [`.claude/domain-rules/testing-budget.md`](../../.claude/domain-rules/testing-budget.md)).
