# Issue Body Sections

## Required sections

All five top-level sections must be present. `Constraints` header required even if empty.

```markdown
## Description

### Symptom / Motivation
{what is observed or motivates the change}

### Root Cause Analysis
{file paths, logic errors, missing features — "N/A" for feature requests}

### Related Files
{key files, functions, or configs}

## Goal
{concrete objective — must not be empty}

## Plan
<!-- type: {proposal | fixed} -->
1. {at least one step}

## Constraints
{scope limits, API stability, perf budgets — or empty}

## Acceptance Criteria
- [ ] Modified files pass unit tests
- [ ] {additional criteria as needed}
```

## Validation rules

| Section             | Rule                                                         |
| ------------------- | ------------------------------------------------------------ |
| Description         | All three subsections present and non-empty                  |
| Goal                | Non-empty                                                    |
| Plan                | ≥1 step (`- ` or `1.`) + `<!-- type: -->` comment            |
| Acceptance Criteria | ≥1 checkbox; always include "Modified files pass unit tests" |

## Defaults (when creating from brief description)

- Goal → extract from description
- Plan → `proposal` type, infer steps
- Constraints → empty
- Acceptance Criteria → "Modified files pass unit tests"
