# Additional PR Body Sections (TileOPs)

These sections extend the standard foundry PR body template.

## Structural Readiness

- **When required**: PR adds new ops or modifies existing kernel/op code
- **Position**: after `## Test plan`, before `## Benchmark`
- **Content**: agent-generated from the op-readiness checklist. Only list FAIL and SKIP items. If all checks pass, write "All checks passed."
- **Do not edit manually** — the pre-create check (Step 2) populates this.

```markdown
## Structural Readiness

<!-- Required for new ops or kernel/op changes. Agent-generated — do not edit manually. -->

All checks passed.
```

## Benchmark (enhanced)

- **When required**: PR adds new ops or modifies existing kernel/op code. This is a lightweight performance profile, not a nightly regression suite.
- **Format**: see [templates/benchmark-template.md](templates/benchmark-template.md) for the required format.
- Overrides the standard foundry "required when PR involves performance changes" rule — in TileOPs, it is specifically required for kernel/op code changes.

## Draft-first workflow note

Copilot reviews drafts automatically; Gemini must be triggered separately (see `lifecycle-pull-request` Phase 2b). Once CI passes and both bot reviews are addressed, the `lifecycle-pull-request` skill marks the PR ready, triggering human reviewer notifications.
