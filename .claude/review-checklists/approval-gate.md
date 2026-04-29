# Approval gate

Last step before a reviewer gives approve on a PR that adds or modifies tests. This is a mechanism, not a style suggestion: if a check fails, withhold approve until the developer pushes a triage commit.

## Why this exists

Issue acceptance criteria may mandate exhaustive matrices (e.g. full dtype × shape combinations) so a developer agent cannot claim "done" by hitting a lucky subset. That trust-model role ends at approval. The suite checked into `main` follows `docs/design/testing.md §Test case policy`, not the AC text — "AC-N required this matrix" is not a defense.

## Reviewer action

For every new or changed test case in the PR, classify:

- **keep** — guards a distinct code path or dtype per `docs/design/testing.md §Test case policy`.
- **shrink** — Cartesian expansion to fold to "boundary + one representative interior point".
- **delete** — same-failure-mode duplicate of a kept case.

If anything is `shrink` or `delete`:

1. Post review comments naming the node IDs to fold or drop, and the kept case each duplicates.
1. Request changes. **Do not approve** until the developer pushes a triage commit addressing every cited case.
1. Re-run this gate on the triage commit.

## Critical-path floor

Triage must not delete the last guarding case for a critical code path identified in `docs/design/testing.md §Test case policy`:

- tile boundary (shape not divisible by tile size)
- vectorization alignment (shape not aligned to vector width)
- degenerate dimension (size = 1)
- dispatch branch (different shape ranges → different kernel variants)

When in doubt, keep one.
