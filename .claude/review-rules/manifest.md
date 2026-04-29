# Review rule: manifest

Applies to PRs prefixed `[Maintain]` (in practice almost always `[Maintain][Manifest]`) — any change to `tileops/manifest/`. Covers three sub-cases:

1. **Adding new entries** — new op spec, lands as `status: spec-only`.
1. **Restructuring entries** — splitting yaml files, realigning to PyTorch reference, refining `roofline`/`shape_rules`. May arrive as `[Refactor][Manifest]`; same review standard applies.
1. **Status flip** — `spec-only → implemented` (or back). Often bundled with a `[Refactor][Ops]` PR; when it is, this rule loads alongside `refactor.md`.

Load `.claude/domain-rules/manifest-spec.md` before reviewing.

## Must check

- **PyTorch parity**: when the op name maps to `torch.nn.*` / `torch.nn.functional.*`, the signature MUST match PyTorch's public API exactly — parameter names, order, defaults. No invented parameters. (`.claude/rules/manifest-trust-model.md`)
- **Reference URL**: the entry's `source` / reference points to authoritative docs (PyTorch, paper, vendor). Reviewer should be able to verify shape/dtype rules from that link alone.
- **Required structural fields present**: `signature`, `shape_rules`, `dtype_combos`, `roofline`, `kernel_map` (where applicable), `static_dims` (where applicable).
- **Validator passes**: `scripts/validate_manifest.py` must be green. If the PR disables a validator check, that's a red flag.
- **Status correctness for the sub-case**:
  - New entries land as `status: spec-only`.
  - Flipping to `implemented` requires the implementation to actually conform: spec tests pass without xfail/skip, op signature matches the manifest entry field-by-field, and any `FIXME(staged-rollout)` markers tied to this op are removed.
  - Flipping back to `spec-only` only when implementation is removed or broken — challenge any other rationale.

## Don't gate on

- Op family organization (which yaml file an entry lives in) unless it conflicts with existing conventions.
- Roofline numerical precision — first-pass author estimates are acceptable; refinement is a follow-up.
- Cosmetic changes to a manifest entry that accompany a status flip (those should be in a separate PR, but a one-line edit isn't worth blocking).

## Hard rejects

- Same PR also modifies `tileops/ops/` or `tileops/kernels/`, except when the only manifest delta is a `status` flip that matches a code-side migration in the same PR.
- Removes `roofline.vars`, `shape_rules`, or `params` to silence the validator (`.claude/rules/manifest-trust-model.md`).
- Signature differs from PyTorch reference for a PyTorch-named op.
- Status flipped to `implemented` while spec tests are still xfail / skipped or have weakened assertions.
- Manifest entry rewritten to "match" what the code does — that inverts the trust model. Spec is authoritative.
