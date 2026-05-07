- For the OWNS / boundary chart of the manifest stage, see [docs/design/trust-model.md](../../docs/design/trust-model.md) §Manifest.
- When the op name matches a PyTorch op (`torch.nn.*`, `torch.nn.functional.*`), the manifest signature must match PyTorch's public API. Do not invent parameters.
- Implementation does not conform to spec → set `status: spec-only`, fix code in follow-up PR. Never modify manifest to match code.
- Do not remove `roofline.vars`, `shape_rules`, or `params` to silence validator errors.

## Status flip carve-out

An implementation PR may edit the aligned op's manifest entry only at:

- `status` (any direction).
- `source.kernel_map` entries.
- `workloads` — **only** when the same PR flips `status: spec-only → implemented` on that op (promotion forces non-empty workloads to satisfy `test_every_op_has_at_least_two_workloads`).

Every other field — `signature`, `roofline.*`, `params`, output-dtype, shape rules, and any other `workloads` edit — needs a separate manifest-only PR with human review.

The carve-out narrows the prohibition; it does not relax the trust boundary.
