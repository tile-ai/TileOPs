Run before approving any PR. Apply each item if its scope matches the diff. If any applicable check fails, REQUEST_CHANGES until the developer pushes a triage commit.

## Tests

- [ ] **Per-case verdict.** Triage every added/modified test case as `keep` / `shrink` / `delete`. Inline only on blockers; clean test PRs stay clean (`criteria.md §3`).

  | Verdict                        | When                                                                         | Inline?     |
  | ------------------------------ | ---------------------------------------------------------------------------- | ----------- |
  | `keep — guards <path/dtype>`   | distinct code path or dtype (per `docs/design/testing.md §Test case policy`) | no          |
  | `shrink — fold to <axis>`      | Cartesian expansion; fold to "boundary + one representative interior point"  | **blocker** |
  | `delete — duplicate of <node>` | same-failure-mode duplicate of a kept case                                   | **blocker** |

  Cases the reviewer cannot classify with confidence count as untriaged → **blocker inline** asking the developer for rationale. Every blocker must be resolved (shrink / delete, or downgrade to `keep` with rationale) before APPROVE.

- [ ] **Numerical floor.** Run `python scripts/test_node_delta.py --base upstream/main` (prereq: `upstream` points to tile-ai/TileOPs and is fresh — `git fetch upstream` first). REQUEST_CHANGES with the full node-ID list if existing-file growth > 25% AND any case carries an unresolved `shrink`/`delete` blocker or is untriaged. Absence of an inline is not itself a blocker — silent `keep` is the default.

- [ ] **Critical-path floor.** Never remove the last guarding case for tile boundary, vectorization alignment, degenerate dimension (size = 1), or dispatch branch.

- [ ] **No AC defense.** Reject "AC-N required this matrix" — AC text does not bind the merged suite.

## Authoring discipline

- [ ] **PR body.** Conforms to `.foundry/mold/pr-body-template.md`; records final state only — what the PR does (Summary, scoped to the merged diff) + verification facts (test plan, pre-commit, structural readiness, test node delta). Strip dev-process narration: per-round fix history, tally IDs (`T001–T0NN`), "Driven by review iteration", reviewer-by-reviewer changelogs, abandoned approaches. Those belong in commit history / review threads. REQUEST_CHANGES if found.

- [ ] **Replies.** Outcome only — `Done in <sha>.`, `Won't-fix: <one-line reason>.` No commit-by-commit narration, "what I tried", thread/tally IDs, root-cause essays, or design restatements. Process detail ages out and clutters the thread. REQUEST_CHANGES asking the developer to edit the comment to a one-liner if found.

## Review process

- [ ] **Batch-once.** If only cleanup-class issues (keep / shrink / delete / rename / dedupe) remain with no correctness blockers, surface every such item from the full diff in this single pass. Don't defer — either include now or demote to advisory (no longer gates APPROVE).

- [ ] **Re-run on triage.** Re-run every applicable check above on the developer's triage commit before approving.

## Scope-specific

- [ ] **Skill edits (`.claude/skills/**`).** Require a tightening pass before APPROVE — condense wording without changing what the skill instructs. Verify: semantics preserved, every step has exactly one valid execution path, no example included unless load-bearing, retained examples reference durable concepts rather than implementation details that age out.
