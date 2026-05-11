Common rules for every checklist in this folder. Load before any title-specific checklist.

## Load order

1. `docs/design/trust-model.md` — stage contracts (`manifest → test → implementation → benchmark`).
1. The title-specific checklist.

## Review lens

Apply trust-model rules as a review lens. Surface cross-layer diffs as comments with a concrete citation; the author supplies rationale; judge on content. Provenance labels (`automated` / `needs-review` / `nightshift`) mark origin only — rule semantics are identical across labels.

## Cross-layer review criteria

| Criterion         | Pass                                                                              | Fail                                                                |
| ----------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Oracle origin     | `ref_program` resolves to PyTorch / NumPy / closed-form / IEEE-754                | Agent-fabricated literal as expected value                          |
| Coverage set      | Unchanged or strictly expanded; deletions target code paths the same diff removes | dtype / shape / sign cells removed with the code path still present |
| New-path coverage | Every added behavior branch is exercised by at least one test                     | New branch lands with no test reaching it                           |

Cite the failing criterion by name.

## Comment quality

| Rule               | Required form                                                                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Concrete pointer   | `file:line`, entry path, field name, parametrize axis, ref URL, test name, or the offending diff line                                                   |
| Decidable language | Concrete claim with the pointer above. Vague forms — "looks reasonable", "may want to verify", "consider revising", "could be clearer" — fail this rule |
| Reviewer restraint | Suggestions stay within fixes for flagged items                                                                                                         |

## Scope

Checklists are the floor. Add PR-specific checks when the diff warrants them.
