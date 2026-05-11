Common rules for every checklist in this folder. Load before any title-specific checklist.

## Load order

1. `docs/design/trust-model.md` — defines `manifest → test → implementation → benchmark` boundaries.
1. The title-specific checklist.

## Trust-model is a review lens, not an auto-reject

Cross-layer diffs do not auto-reject on directory layout. Surface them as review comments with a concrete citation; the author replies with rationale; judge on content.

Provenance labels `automated`, `needs-review`, `nightshift` mark origin only. Rule semantics are identical across labels.

## Cross-layer review criteria

| Criterion         | Pass                                                                             | Fail                                                                         |
| ----------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Oracle origin     | `ref_program` resolves to PyTorch / NumPy / closed-form / IEEE-754               | Agent-fabricated literal as expected value                                   |
| Coverage set      | Unchanged or strictly expanded; deletions target code paths that no longer exist | dtype / shape / sign cells removed without a corresponding code-path removal |
| New-path coverage | Every added behavior branch is exercised by at least one test                    | New branch lands with no test reaching it                                    |

Cite the failing criterion by name in the comment.

## Comment quality

| Rule               | Required form                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Concrete pointer   | `file:line`, entry path, field name, parametrize axis, ref URL, test name, or the offending diff line              |
| No hedging         | Reject "looks reasonable", "may want to verify", "consider revising", "could be clearer"                           |
| Reviewer restraint | Verify against existing reference / spec / convention. Do not propose new content beyond what fixes a flagged item |

## Scope

Checklists are the floor. Add PR-specific checks when warranted.
