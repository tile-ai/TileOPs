# PR Body Sections

Extensions to the standard foundry PR body template.

## Structural Readiness

- **When**: PR adds/modifies kernel or op code
- **Position**: after `## Test plan`, before `## Benchmark`
- **Content**: auto-generated from op-readiness checklist — list FAILs and SKIPs only; all pass → "All checks passed."
- Do not edit manually

## Benchmark

- **When**: PR adds/modifies kernel or op code (overrides foundry default of "performance changes only")
- **Format**: [templates/benchmark-template.md](templates/benchmark-template.md)

## Draft-first workflow

Copilot reviews drafts automatically. Gemini must be triggered (`/gemini review`). After CI + bot reviews pass, `lifecycle-pull-request` marks PR ready for human review.
