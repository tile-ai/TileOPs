- For the OWNS / boundary chart of the manifest stage, see [docs/design/trust-model.md](../../docs/design/trust-model.md) §Manifest.
- When the op name matches a PyTorch op (`torch.nn.*`, `torch.nn.functional.*`), the manifest signature must match PyTorch's public API. Do not invent parameters.
- Implementation does not conform to spec → set `status: spec-only`, fix code in follow-up PR. Never modify manifest to match code.
- Do not remove `roofline.vars`, `shape_rules`, or `params` to silence validator errors.

## Status flip carve-out

An implementation PR (the one that aligns op code with the manifest) MAY touch the manifest entry of the op being aligned, but ONLY within the metadata fields enumerated below. Contractual fields stay frozen and require a separate manifest-only PR with human review.

**Allowed in an implementation PR** (metadata only — no human re-review of the spec needed):

- Flip `status: spec-only` ↔ `status: implemented`.
- Add or remove entries in `source.kernel_map` (the dispatch registration table).
- Add entries to `workloads`, ONLY when this PR also flips `status` from `spec-only` to `implemented` for the same op. A `spec-only` entry has empty `workloads` by definition, and `test_every_op_has_at_least_two_workloads` requires ≥2 entries on `implemented` ops, so promotion forces this addition. Other status transitions (including `implemented` → `spec-only`) and ops already at `implemented` MUST NOT change `workloads` in an implementation PR; route those edits through a separate manifest-only PR.

Except for the `workloads` allowance above, these edits are permitted ONLY on op entries whose `signature`, `workloads`, `roofline`, and `params` blocks are byte-identical between the PR's base and head.

**Not allowed in an implementation PR** (contractual — require a manifest-only PR with human review):

- Any change to `signature` (parameter names, order, default values, `dtype`, `ref_api`, `ref_dtype`).
- Any change to `workloads` (shape entries, axis names, named-tile sets), except the `spec-only` → `implemented` promotion case enumerated in the "Allowed" list above.
- Any change to `roofline.*` (`vars`, formulas, `flops`, `bytes`, `peak_*`, per-consumer fields).
- Any change to `params` (declared static or dynamic parameters, defaults).
- Any change to output-dtype rules or shape rules (e.g. `output_dtype`, `shape_rules`).

This carve-out narrows the prohibition; it does not relax the trust boundary. If an implementation PR needs any change in the "not allowed" list, stop, open a manifest-only PR for that change first, and resume the implementation PR against the merged manifest.
