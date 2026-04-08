## Boundary

- **OWNS**: `tileops/ops_manifest.yaml`
- **MUST NOT modify**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `benchmarks/`
- **MAY READ**: PyTorch public API (to match signatures)
- Manifest changes require human review in a separate PR.

→ [trust-model.md §Manifest](../../docs/trust-model.md#manifest)

______________________________________________________________________

- `inputs`, `outputs`, `params` are ordered dicts. Key order = function signature position. Do not reorder.
- Params include all PyTorch-supported parameters, even if the current kernel only supports the default.
- `dtype` syntax: `|` for alternatives, `same_as(ref)` for dependent types. `same_as(ref)` is a dtype constraint (identity) — means "must be the exact same dtype as `ref` at runtime". Does not contribute independent axes to the Cartesian product in `dtype_combos`.
- `dtype_combos` when supported set is a strict subset of the Cartesian product. Omit when all combinations are valid.
- Every output shape must be fully specified via `shape`, `same_as(ref)`, and/or `shape_rules`.
- `shape` present = fixed rank. Names become roofline variables. `shape` absent = arbitrary rank, use `params` + `shape_rules`.
- `same_as(ref)` = identical shape to referenced tensor. Shared dimension names across tensors = sizes must match.
- `shape_rules` are Python expressions for shape relationships. `shape`, `same_as(ref)`, and `shape_rules` fully specify output shape derivation.
- `status: implemented` = all validator levels apply. `status: spec-only` = L0 only. Default is `implemented`.
- Roofline `vars` maps variable names to Python expressions over tensor shapes and params. Required for arbitrary-rank ops.
- Op signatures must match PyTorch's public API (parameter names, parameter set, semantics). Do not invent parameters.
- No `Optional[Tensor]` in manifest. Ops with conditional inputs are split into variant entries linked by `variant_of`.
  - `variant_of` is one level only. Variant → primary. Primary must not have `variant_of`.
  - Variants share `source.kernel` and `source.op`. Each has its own `signature`, `workloads`, `roofline`.
- Tensor layout defaults to contiguous row-major. When an op requires non-default layout (e.g., `channels_last`), add `layout` field to the tensor declaration. `shape` dimension names reflect actual memory order.
- Never modify manifest to match non-conforming code. If code doesn't match spec: set `status: spec-only` and add a comment explaining the discrepancy, then fix implementation in a follow-up PR. Never remove params, vars, or shape_rules to silence validator errors.
