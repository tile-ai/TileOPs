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

## 2. Test node delta

**Skip entirely** if the PR does not modify files under `tests/`.

```bash
git fetch upstream main --quiet
python scripts/test_node_delta.py --base upstream/main
```

The script auto-detects changed test files via `git diff`. Where auto-detect fails — in a worktree, for instance — pass them explicitly:

```bash
python scripts/test_node_delta.py --base upstream/main tests/ops/test_<name>.py
```

What to do with the output is in [`docs/design/testing.md` § Test node growth detection](../../docs/design/testing.md#test-node-growth-detection).

**SOFT GATE:** does not block PR creation. Missing justification is flagged during review, per [`.claude/domain-rules/testing-budget.md`](../../.claude/domain-rules/testing-budget.md).

## 3. PR body

Two checks CI does not make. Run them against the body you are about to pass to `gh pr create`.

```bash
BODY_FILE=/tmp/pr-body.md   # the file you will pass as --body-file
ok=1

grep -q '## Summary' "$BODY_FILE" || { echo "BLOCKED: PR body needs a ## Summary section"; ok=0; }
if grep -q '\\n' "$BODY_FILE"; then
  echo 'BLOCKED: PR body contains a literal \n — write actual newlines'
  ok=0
fi
[ "$ok" = 1 ] || exit 1
```

**HARD GATE.** A `BLOCKED` line means do not call `gh pr create`; fix the body and re-run.

A literal `\n` reaches the body when it is passed as a JSON string rather than as a file. Pass
`--body-file`, or a quoted heredoc, and it cannot happen.

Labels need no check here: [`auto-label.yml`](../../.github/workflows/auto-label.yml) derives them
from the title once the PR exists.
