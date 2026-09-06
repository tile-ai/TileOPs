Run before approving any PR. Apply each item if its scope matches the diff.

## Tests

The author reports the test node delta per [testing-budget.md](../domain-rules/testing-budget.md).
Two thresholds read off it: above 10% growth the author justifies and reviewer silence is not
approval; above 25% an unresolved blocker means the PR is not clean.

- [ ] **Per-case verdict.** Triage every added or modified test case as `keep` / `shrink` /
  `delete`. Raise a comment only on blockers; a clean test PR stays clean.

  | Verdict                        | When                                                                         | Blocks? |
  | ------------------------------ | ---------------------------------------------------------------------------- | ------- |
  | `keep — guards <path/dtype>`   | distinct code path or dtype (per `docs/design/testing.md §Test case policy`) | no      |
  | `shrink — fold to <axis>`      | Cartesian expansion; fold to "boundary + one representative interior point"  | yes     |
  | `delete — duplicate of <node>` | same-failure-mode duplicate of a kept case                                   | yes     |

  A case the reviewer cannot classify with confidence counts as untriaged, which blocks: ask the
  author for the rationale. Every blocker resolves — shrink, delete, or downgrade to `keep` with a
  stated reason — before the PR is clean.

- [ ] **Per-case purpose stated.** Above the 10% trigger, each new case or parametrize cell serves
  exactly one of: dtype correctness, kernel-branch shape coverage, feature coverage, regression.
  The PR body names which, with `file:line`.

- [ ] **No Cartesian-product expansion.** A parametrize stack whose growth is the product of two or
  more axes' cardinalities needs a per-cell rationale. Crossing axes is allowed only when each cell
  maps to a distinct code path the author can name; otherwise the stack is a performance sweep, not
  a unit test.

- [ ] **Numerical floor.** Re-run the delta yourself against a fresh `upstream/main` rather than
  trusting the reported number. Request changes with the full node-ID list if existing-file growth
  exceeds 25% AND any case carries an unresolved `shrink` / `delete` blocker or is untriaged.
  Absence of a comment is not itself a blocker — silent `keep` is the default.

- [ ] **Critical-path floor.** Never remove the last test on an output-distinguishing input: tile
  boundary, vectorization alignment, degenerate dimension (size = 1), or a dispatch branch carrying
  observable behavior. Tests with no output-distinguishing input are removable.

- [ ] **No AC defense.** Reject "AC-N required this matrix" — acceptance-criteria text does not bind
  the merged suite.

## Staged-rollout markers

- [ ] Every `xfail`, `skip`, or loosened assertion the diff introduces carries the
  `FIXME(staged-rollout)` block from [code-style.md](../rules/code-style.md), naming the broken
  invariant and the condition that removes it.
