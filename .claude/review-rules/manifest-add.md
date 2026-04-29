# Review rule: manifest-add

Applies to PRs prefixed `[Maintain][Manifest]` — adds new entries to `tileops/manifest/`. Pure spec authoring; no implementation in the same PR.

Load `.claude/domain-rules/manifest-spec.md` before reviewing.

## Must check

- **PyTorch parity**: when the op name maps to `torch.nn.*` / `torch.nn.functional.*`, the signature MUST match PyTorch's public API exactly — parameter names, order, defaults. No invented parameters. (`.claude/rules/manifest-trust-model.md`)
- **Reference URL**: the entry's `source` / reference points to authoritative docs (PyTorch, paper, vendor). Reviewer should be able to verify shape/dtype rules from that link alone.
- **Required structural fields present**: `signature`, `shape_rules`, `dtype_combos`, `roofline`, `kernel_map` (where applicable), `static_dims` (where applicable).
- **Status correctness**: new entries land as `status: spec-only`. Flipping to `implemented` is a separate PR (`manifest-flip.md`).
- **Validator passes**: `scripts/validate_manifest.py` must be green. If the PR disables a validator check, that's a red flag.

## Don't gate on

- Op family organization (which yaml file an entry lives in) unless it conflicts with existing conventions.
- Roofline numerical precision — first-pass author estimates are acceptable; refinement is a follow-up.

## Hard rejects

- Same PR also modifies `tileops/ops/` or `tileops/kernels/`.
- Removes `roofline.vars`, `shape_rules`, or `params` to silence the validator (`.claude/rules/manifest-trust-model.md`).
- Signature differs from PyTorch reference for a PyTorch-named op.
