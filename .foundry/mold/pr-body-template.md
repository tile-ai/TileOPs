<!--
KEEP IT SHORT. Reviewers read PRs to learn WHAT changed and HOW to verify it.
- Summary: 3–5 bullets, one line each. No prose paragraphs.
- Do NOT recount the development process (reviewer rounds, follow-up issues filed,
  prior commits reverted, design rationale that already lives in the linked issue).
- Do NOT restate the linked issue's motivation; "Closes #N" already links it.
- One short paragraph of cross-cutting context is fine when bullets can't carry it.
- If a section doesn't apply, delete the header — never leave empty sections.
-->

Closes #\{issue-number}

## Summary

- {what was added/fixed/changed}
- {what was removed/replaced}

## Test plan

- [x] pre-commit passed
- [x] pytest passed

<!-- Delete inapplicable optional sections entirely. Never leave empty headers. -->

## Structural Readiness

<!-- Required for kernel/op changes. Agent-generated — do not edit. -->

All checks passed.

## Benchmark

<!-- Required for kernel/op changes. Format: benchmark-template.md -->

## Regression

<!-- Recommended for bugfix/refactor -->
