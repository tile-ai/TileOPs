Run before approving any PR that adds or modifies tests. If any check fails, request changes; do not approve until the developer pushes a triage commit.

- [ ] **Per-case verdict (blocker default).** For every test case added or modified in this PR, post an inline comment on the test definition with one of:

  - `keep — guards <distinct code path or dtype>` (no action required) — per `docs/design/testing.md §Test case policy`.
  - `shrink — fold to <axis>` (**BLOCKER**) — Cartesian expansion; fold to "boundary + one representative interior point".
  - `delete — duplicate of <node ID>` (**BLOCKER**) — same-failure-mode duplicate of a kept case.

  Cases left without a verdict comment count as untriaged and are also **BLOCKER**. Every blocker must be resolved (case shrunk / deleted, or verdict downgraded to `keep` with rationale) before APPROVE.

- [ ] **Numerical floor.** Run `python scripts/test_node_delta.py --base upstream/main`. If existing-file growth > 25% AND any added case lacks a verdict comment, REQUEST_CHANGES with the full list of unjustified node IDs in the summary.

- [ ] Reject "AC-N required this matrix" as a defense — AC text does not bind the merged suite.

- [ ] Critical-path floor: never remove the last guarding case for tile boundary, vectorization alignment, degenerate dimension (size = 1), or dispatch branch.

- [ ] **PR body discipline.** Body must conform to `.foundry/mold/pr-body-template.md` and record only the final state: what the PR does (Summary, scoped to the merged diff) + verification facts (test plan, pre-commit, structural readiness, test node delta). Strip dev-process narration — per-round fix history, tally IDs (`T001–T0NN`), "Driven by review iteration", reviewer-by-reviewer changelogs, abandoned approaches. Those belong in commit history / review threads. If found in the body, REQUEST_CHANGES.

- [ ] **Batch-once.** If the only remaining issues are cleanup-class (keep / shrink / delete / rename / dedupe) with no correctness blockers left, surface every such item from the full diff in this single review pass. Do not defer to a later round. If you find yourself wanting to "leave it for next time", either include it now or demote it to advisory (no longer gates APPROVE).

- [ ] On the triage commit, re-run every check above before approving.
