# Issue Body Sections

The template and rules here keep the trust model enforceable as a semantic review lens, not as a directory-level syntactic block. Constraints are authored to declare the work's actual cross-stage shape so the pipeline and the reviewer judge the same diff against the same intent.

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

## 3. Constraints declares cross-stage shape

The pipeline reads Constraints bullets to learn which stages the PR is allowed to touch. The author declares the shape; the reviewer judges semantic correctness against `docs/design/trust-model.md`. Together this preserves the trust model — same-agent fabrication of oracle + impl is caught at review, while honest cross-stage work proceeds without syntactic blocks.

| Work shape                 | Constraints bullet                                                                                              | Result                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Joint change across stages | Behavioral / compatibility / perf bullets only                                                                  | Pipeline permits any stage; reviewer applies the lens |
| Single stage               | `Implementation-only PR.` (or `Test-only PR.`, etc.)                                                            | Pipeline confines the diff to that stage              |
| Multiple stages, declared  | One bullet per stage: `Implementation-only PR for kernel widening.` + `Test-only PR for parametrize expansion.` | Pipeline permits the named stages' union              |

Rules:

- A new behavior branch with no pre-existing test coverage uses the joint form. The reviewer's new-path-coverage criterion requires the test to land in the same PR.
- Constraints body is always bulleted. Prose-only Constraints is unparsable input and Phase A treats it as a block.
- `docs/design/trust-model.md` is cited only as a positive reference. Pairing the citation with "separate PR" / "own PR" / "standalone PR" in one bullet declares that stage forbidden — use only when that is genuinely intended.

## 4. Defaults when drafting from a brief description

| Field               | Default                                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Goal                | Extracted from the description                                                                                                 |
| Plan                | `<!-- type: proposal -->`; steps inferred                                                                                      |
| Constraints         | One bullet stating the behavioral or compatibility expectation. `<stage>-only PR` only when the work is genuinely single-stage |
| Acceptance Criteria | `- [ ] Modified files pass unit tests`                                                                                         |

## 5. Cross-references

- Reviewer-side criteria: [.claude/review-checklists/pre-review.md](../../.claude/review-checklists/pre-review.md)
- Trust-model semantics: [docs/design/trust-model.md](../../docs/design/trust-model.md)
