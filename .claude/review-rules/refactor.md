# Review rule: refactor

Applies to PRs prefixed `[Refactor]` — restructures code without changing externally observable behavior.

## Must check

- **Behavior preservation**: existing tests still pass without being weakened. Any `xfail` / `skip` / loosened assertion needs a `FIXME(staged-rollout)` block per `.claude/rules/code-style.md`.
- **Scope discipline**: refactor PRs do one thing. Mixing refactor + new behavior makes review and revert hard — push back.
- **Public interface stability**: if the refactor changes a signature, callers must be updated in the same PR. If the op has a manifest entry, signature must still match it.
- **Import + naming hygiene**: `.claude/rules/code-style.md` (relative imports inside package, abbreviation casing, underscore-separated norm filenames).

## Sub-type: `[Refactor][Manifest]`

Treat as a manifest-spec change. Load `.claude/domain-rules/manifest-spec.md`. Code under `tileops/ops/` and `tileops/kernels/` must NOT be touched in the same PR (trust model).

## Sub-type: `[Refactor][Ops]` / `[Refactor][<Family>]`

Code-side migration. Often co-occurs with a `status: spec-only → implemented` flip — if so, also load `manifest-flip.md`.

## Don't gate on

- Stylistic preferences not encoded in `.claude/rules/`.
- Adding tests beyond what the refactored code already covered (refactors aren't an excuse to demand new coverage).

## Hard rejects

- Silently weakens a test to make refactored code pass.
- Refactor PR also adds a new feature or fixes a separate bug.
