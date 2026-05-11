# Issue Body Sections

Structural contract for the issue body: the template and per-section rules below are what the pipeline parses. Constraints authoring policy (work-shape declarations, defaults) lives in [docs/design/trust-model.md §Issue-authoring](../../docs/design/trust-model.md#issue-authoring-declaring-scope).

## 1. Template

Copy verbatim. Replace each `{...}`. Keep all five top-level sections.

```markdown
## Description

### Symptom / Motivation
{what is observed or motivates the change}

### Root Cause Analysis
{file paths, logic errors, missing features — "N/A" for feature requests}

### Related Files
{key files, functions, or configs}

## Goal
{concrete objective}

## Plan
<!-- type: {proposal | fixed} -->
1. {at least one step}

## Constraints
- {behavioral / compatibility / perf constraint, bullet form}

## Acceptance Criteria
- [ ] Modified files pass unit tests
- [ ] {additional criteria as needed}
```

## 2. Per-section rules

| Section             | Required form                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| Description         | Three subsections (`Symptom / Motivation`, `Root Cause Analysis`, `Related Files`), each non-empty |
| Goal                | Non-empty single-paragraph objective                                                               |
| Plan                | `<!-- type: proposal \| fixed -->` comment plus at least one step (`- ` or `1.`)                   |
| Constraints         | At least one list item (`- `)                                                                      |
| Acceptance Criteria | At least one checkbox, including `- [ ] Modified files pass unit tests`                            |

## 3. Cross-references

- Constraints authoring policy: [docs/design/trust-model.md §Issue-authoring](../../docs/design/trust-model.md#issue-authoring-declaring-scope)
- Reviewer-side criteria: [.claude/review-checklists/pre-review.md](../../.claude/review-checklists/pre-review.md)
