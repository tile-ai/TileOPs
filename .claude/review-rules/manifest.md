# Review rule: manifest

For `[Maintain]` (almost always `[Maintain][Manifest]`) — any `tileops/manifest/` change. Also load this rule for `[Refactor][Manifest]` and for `[Refactor][Ops]` PRs that flip `status`.

Sub-cases:

1. Add new entry → lands as `status: spec-only`.
1. Restructure entry → split yaml, realign to PyTorch reference, refine `roofline` / `shape_rules`.
1. Status flip → `spec-only → implemented` (or back).

Load `.claude/domain-rules/manifest-spec.md` before reviewing.

## Checklist

- [ ] [REQ] PyTorch parity: when op name maps to `torch.nn.*` / `torch.nn.functional.*`, signature matches PyTorch's public API exactly — names, order, defaults; no invented parameters (`.claude/rules/manifest-trust-model.md`)
- [ ] [REQ] `source` field points to authoritative docs (PyTorch, paper, vendor); shape/dtype rules verifiable from that link alone
- [ ] [REQ] Required fields present: `signature`, `shape_rules`, `dtype_combos`, `roofline`; `kernel_map` and `static_dims` where applicable
- [ ] [REQ] `scripts/validate_manifest.py` is green; no validator check disabled
- [ ] [REQ] New entries land as `status: spec-only`
- [ ] [REQ] Flip to `implemented`: spec tests pass without xfail/skip; op signature matches the entry field-by-field; related `FIXME(staged-rollout)` markers removed
- [ ] [REQ] Flip back to `spec-only` only when implementation is removed or broken
- [ ] [REQ] Same PR does NOT modify `tileops/ops/` or `tileops/kernels/`, except a pure `status` flip aligned with a code-side migration in the same PR
- [ ] [REQ] No removal of `roofline.vars`, `shape_rules`, or `params` to silence the validator
- [ ] [REQ] Manifest entry not rewritten to match what the code does — spec is authoritative
- [ ] [REC] Op family file placement consistent with existing conventions
- [ ] [REC] Roofline numerical precision — first-pass estimates OK; refinement is a follow-up
