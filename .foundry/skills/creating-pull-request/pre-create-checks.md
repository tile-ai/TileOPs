# Pre-Create Checks

**Skip entirely** if PR does not touch kernel/op code.

If PR introduces or modifies a kernel/op:

1. Read [op-readiness-checklist.md](op-readiness-checklist.md).
1. Verify every item → `PASS`, `FAIL (reason)`, or `SKIP (reason)`.

**HARD GATE:** Any `[REQUIRED]` FAIL blocks the PR. `[RECOMMENDED]` may be skipped with reason. In `## Structural Readiness`, list only FAILs and SKIPs. All pass → "All checks passed."
