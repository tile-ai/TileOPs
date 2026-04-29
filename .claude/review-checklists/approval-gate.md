# Approval gate

Run before approving any PR that adds or modifies tests. If any check fails, request changes; do not approve until the developer pushes a triage commit.

- [ ] Classify every new or changed test case as **keep** / **shrink** / **delete**:
  - **keep** — guards a distinct code path or dtype per `docs/design/testing.md §Test case policy`.
  - **shrink** — Cartesian expansion; fold to "boundary + one representative interior point".
  - **delete** — same-failure-mode duplicate of a kept case.
- [ ] For each `shrink` / `delete`, post a review comment with the node ID and the kept case it duplicates or the axis to fold.
- [ ] Reject "AC-N required this matrix" as a defense — AC text does not bind the merged suite.
- [ ] Critical-path floor: never remove the last guarding case for tile boundary, vectorization alignment, degenerate dimension (size = 1), or dispatch branch.
- [ ] On the triage commit, re-run every check above before approving.
