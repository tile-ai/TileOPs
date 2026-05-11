# Issue Body Sections

## Required sections

All five top-level sections must be present. Constraints content MUST be list items, not prose — the foundry write-scope gate fails closed on a prose-only Constraints section.

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
- {behavioral / compatibility constraints, bullet form}

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
| Constraints         | ≥1 list item; prose-only body fails the write-scope gate     |
| Acceptance Criteria | ≥1 checkbox; always include "Modified files pass unit tests" |

## Write-scope declaration in Constraints

The write-scope gate (`foundry/scripts/check-write-scope.sh`) parses Constraints list items for three signal classes. Pick the one that matches the work — do not default to `<stage>-only PR` for joint impl+test changes.

| Intent                               | Constraints bullet form                                                                                                                          | Gate effect                                                                                                                                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Joint change across stages (default) | Bullets state behavioral / compatibility / perf constraints. No `<stage>-only PR` phrasing. No `trust-model.md` anchor + "separate PR" phrasing. | `declared=[]`, gate inactive. Dev may touch any stage. Reviewer judges via review-lens (`docs/design/trust-model.md`, `.claude/review-checklists/pre-review.md`). |
| Genuinely single-stage               | `Implementation-only PR.` (or `Test-only PR.`, etc.)                                                                                             | `declared=[<stage>]`. Dev's diff must stay in that stage.                                                                                                         |
| Genuinely multi-stage by enumeration | Each stage on its own bullet: `Implementation-only PR for kernel widening.` + `Test-only PR for parametrize expansion.`                          | `declared=[impl, test]`. Dev free within that union.                                                                                                              |

### Anti-patterns

- `Implementation-only PR.` on a PR that adds a new behavior branch with no pre-existing test coverage. Forces the dev into an unsatisfiable state: kernel-only diff fails reviewer's new-path-coverage criterion; joint diff fails the write-scope gate. Use the joint-change form instead.
- Constraints body containing only prose (no `- ` bullets). Parses as zero list items → `policy=unclear_block` → Phase A FATAL.
- Citing `docs/design/trust-model.md` together with "separate PR" / "own PR" / "standalone PR" in the same bullet. That combination marks the named stage as `forbidden`, even when the rest of the issue allows it.

## Cross-references

- Review-side criteria: [.claude/review-checklists/pre-review.md](../../.claude/review-checklists/pre-review.md)
- Trust-model semantics (review lens): [docs/design/trust-model.md](../../docs/design/trust-model.md)
- Gate implementation: `foundry/scripts/check-write-scope.sh` + `foundry/conventions/scope-map.json`

## Defaults (when creating from brief description)

- Goal → extract from description
- Plan → `proposal` type, infer steps
- Constraints → one bullet stating the behavioral / compatibility expectation. Do not insert `<stage>-only PR` unless the work genuinely requires it.
- Acceptance Criteria → "Modified files pass unit tests"
