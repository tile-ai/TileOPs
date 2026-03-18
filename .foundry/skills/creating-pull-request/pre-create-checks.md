# Pre-Create Checks: Structural Readiness (TileOPs)

## Structural readiness check (new ops only)

If this PR does **not** touch kernel/op code, skip this check entirely.

If this PR introduces or modifies a kernel/op:

1. Read [op-readiness-checklist.md](op-readiness-checklist.md) for the full checklist.
1. Verify the code against every item. Record each as `PASS`, `FAIL (reason)`, or `SKIP (reason)`.

**HARD GATE:** Any `[REQUIRED]` item that is `FAIL` must be fixed before proceeding. `[RECOMMENDED]` items may be skipped with a reason. In the PR body `## Structural Readiness`, only list failures and skips. If all pass, write "All checks passed."
