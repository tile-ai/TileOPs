Run before approving any PR. Apply each item below if its scope matches the diff. If any applicable check fails, request changes; do not approve until the developer pushes a triage commit.

- [ ] **Per-case verdict (blocker default).** Triage every test case added or modified in this PR as one of `keep` / `shrink` / `delete`. Inline comments are required only for blocker verdicts; clean test PRs stay clean (no per-case inline noise) — consistent with `criteria.md §3` (`Clean — no issues.`).

  - `keep — guards <distinct code path or dtype>` (no action required) — per `docs/design/testing.md §Test case policy`. **Do not** post inline.
  - `shrink — fold to <axis>` (**BLOCKER, inline required**) — Cartesian expansion; fold to "boundary + one representative interior point".
  - `delete — duplicate of <node ID>` (**BLOCKER, inline required**) — same-failure-mode duplicate of a kept case.

  Cases the reviewer cannot classify with confidence count as untriaged → **BLOCKER, inline required** asking the developer for the rationale. Every blocker must be resolved (case shrunk / deleted, or verdict downgraded to `keep` with rationale) before APPROVE.

- [ ] **Numerical floor.** Run `python scripts/test_node_delta.py --base upstream/main`. If existing-file growth > 25% AND any case carries an unresolved blocker verdict (`shrink` / `delete`) or remains untriaged, REQUEST_CHANGES with the full list of affected node IDs in the summary. (Absence of an inline comment is not a blocker on its own — silent `keep` is the default.)

- [ ] Reject "AC-N required this matrix" as a defense — AC text does not bind the merged suite.

- [ ] Critical-path floor: never remove the last guarding case for tile boundary, vectorization alignment, degenerate dimension (size = 1), or dispatch branch.

- [ ] **PR body discipline.** Body must conform to `.foundry/mold/pr-body-template.md` and record only the final state: what the PR does (Summary, scoped to the merged diff) + verification facts (test plan, pre-commit, structural readiness, test node delta). Strip dev-process narration — per-round fix history, tally IDs (`T001–T0NN`), "Driven by review iteration", reviewer-by-reviewer changelogs, abandoned approaches. Those belong in commit history / review threads. If found in the body, REQUEST_CHANGES.

- [ ] **Batch-once.** If the only remaining issues are cleanup-class (keep / shrink / delete / rename / dedupe) with no correctness blockers left, surface every such item from the full diff in this single review pass. Do not defer to a later round. If you find yourself wanting to "leave it for next time", either include it now or demote it to advisory (no longer gates APPROVE).

- [ ] **Skill edits (`.claude/skills/**`) — tightening pass before APPROVE.** Require the developer to condense the skill's wording without changing what it instructs. Verify: semantics preserved, every step has exactly one valid execution path, no example included unless it is load-bearing, and any retained example references durable concepts rather than implementation details that age out.

- [ ] On the triage commit, re-run every check above before approving.
