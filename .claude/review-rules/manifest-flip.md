# Review rule: manifest-flip

Applies to PRs that flip a manifest entry's `status: spec-only → implemented` (or back). Often arrives bundled with a `[Refactor][Ops]` / `[Refactor][<Family>]` op-migration PR.

## Must check

- **Conformance is real, not asserted**: the op's public interface (constructor signature, forward signature, dtype handling, shape rules) actually matches the manifest entry. Diff the code against the manifest entry field-by-field.
- **Tests pass on the implementation**: the spec tests for this op are no longer xfail/skip; they execute and pass. Look for sneaky `pytest.skip` left from the spec-only era.
- **`FIXME(staged-rollout)` cleanup**: any markers tied to "implementation lands later" must be removed when the implementation lands. Grep `FIXME(staged-rollout)` in the touched files.
- **Benchmark exists or rationale given**: `[Refactor][Ops]` flips usually expect a working benchmark. If not included, PR should say why (e.g. follow-up).
- **Single-direction flip**: `implemented → spec-only` only when implementation is removed/broken; reviewer should challenge this.

## Don't gate on

- Performance numbers vs roofline unless the PR claims a perf goal.
- Cosmetic changes to the manifest entry beyond the status field — those belong in a `[Refactor][Manifest]` PR.

## Hard rejects

- Status flipped to `implemented` while spec tests are still xfail/skipped or have weakened assertions.
- Flip PR also rewrites the manifest entry's signature/shape rules — that's reverse-engineering spec from code (`.claude/rules/manifest-trust-model.md`). Demand split into two PRs.
