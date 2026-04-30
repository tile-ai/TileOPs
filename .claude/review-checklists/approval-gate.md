Run before approving any PR that adds or modifies tests. If any check fails, request changes; do not approve until the developer pushes a triage commit.

- [ ] Classify every new or changed test case as **keep** / **shrink** / **delete**:
  - **keep** — guards a distinct code path or dtype per `docs/design/testing.md §Test case policy`.
  - **shrink** — Cartesian expansion; fold to "boundary + one representative interior point".
  - **delete** — same-failure-mode duplicate of a kept case.
- [ ] For each `shrink` / `delete`, post a review comment with the node ID and the kept case it duplicates or the axis to fold.
- [ ] Reject "AC-N required this matrix" as a defense — AC text does not bind the merged suite.
- [ ] Critical-path floor: never remove the last guarding case for tile boundary, vectorization alignment, degenerate dimension (size = 1), or dispatch branch.
- [ ] PR body conforms to `.foundry/mold/pr-body-template.md` and reflects the final state of the diff (no stale process narration). If not, request changes and have the developer update it before approving.
- [ ] **Batch-once.** If the only remaining issues are cleanup-class (keep / shrink / delete / rename / dedupe) with no correctness blockers left, surface every such item from the full diff in this single review pass. Do not defer to a later round. If you find yourself wanting to "leave it for next time", either include it now or demote it to advisory (no longer gates APPROVE).
- [ ] On the triage commit, re-run every check above before approving.
