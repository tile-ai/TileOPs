# Issue Body Sections (TileOPs)

These sections extend or override the standard foundry issue body template.

## Template

```markdown
## Description

### Symptom / Motivation

{What is observed or what motivates this change}

### Root Cause Analysis

{Why it happens — file paths, logic errors, missing features, etc. Use "N/A" for feature requests}

### Related Files

{Key files, functions, or configurations involved}

## Goal

{Concrete objective of this issue}

## Plan

<!-- type: {proposal | fixed} -->

{Ordered steps to achieve the goal}

## Constraints

{Scope limits, API stability, performance budgets, etc. Leave empty if none}

## Acceptance Criteria

- [ ] Modified files pass unit tests
{Additional checkboxes as needed}
```

## Rules

- **All five top-level sections are required** (`Description`, `Goal`, `Plan`, `Constraints`, `Acceptance Criteria`).
- `Description` must contain all three subsections (`Symptom / Motivation`, `Root Cause Analysis`, `Related Files`).
- `Goal` must not be empty.
- `Plan` must contain at least one step (line starting with `- ` or `1.`).
- `Plan` must include a `<!-- type: -->` comment: `proposal` (steps are suggestions) or `fixed` (steps are prescribed).
- `Acceptance Criteria` must contain at least one checkbox (`- [ ]`). The default criterion `"Modified files pass unit tests"` is always included.
- `Constraints` may be empty but the header must be present.

## Smart defaults

When creating an issue from a brief description without explicit arguments:

1. **Goal**: extracted from description
1. **Plan**: `proposal` type, steps inferred from description
1. **Constraints**: empty
1. **Acceptance Criteria**: `"Modified files pass unit tests"` (always included)
