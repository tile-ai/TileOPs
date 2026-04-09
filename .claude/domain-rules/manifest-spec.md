## Boundary

- **OWNS**: `tileops/ops_manifest.yaml`
- **MUST NOT modify**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `benchmarks/`
- **MAY READ**: PyTorch public API (to match signatures)
- Manifest changes require human review in a separate PR.

→ [trust-model.md §Manifest](../../docs/trust-model.md#manifest)

______________________________________________________________________

- Manifest keys are PascalCase Op class names: `{PascalCaseName}{Direction}Op` (e.g., `RMSNormFwdOp`). The key must exactly match the Python class name (`cls.__name__`). No abbreviation rules — the manifest author determines the name.
- `ref_api` (required) — declares the external API the signature follows (e.g., `torch.nn.functional.rms_norm`). Set to `"none"` if no direct external counterpart exists. Informational only; does not affect validation.
- `inputs`, `outputs`, `params` are ordered dicts. Key order = function signature position. Do not reorder.
- Params include all PyTorch-supported parameters, even if the current kernel only supports the default.
- `dtype` syntax: `|` for alternatives, `same_as(ref)` = same dtype as ref.
- `same_as(ref)` is a dtype identity constraint: the tensor must have the exact same dtype as `ref` at runtime. `same_as`-bound tensors do not contribute independent axes to the Cartesian product in `dtype_combos`.
- `dtype_combos` when supported set is a strict subset of the Cartesian product. Omit when all combinations are valid.
- Every output tensor's shape must be fully specified via `shape` and/or `shape_rules`. Inputs may omit `shape` (→ arbitrary rank). `same_as` is dtype-only — do not use it for shape.
- `shape` present = fixed rank. Names become roofline variables. `shape` absent = arbitrary rank, use `params` + `shape_rules`.
- Shared dimension names across tensors = sizes must match.
- `shape_rules` are Python expressions for shape relationships. `shape` and `shape_rules` fully specify output shape derivation.
- `status: implemented` = all validator levels apply. `status: spec-only` = L0 only. Default is `spec-only`.
- Roofline `vars` maps variable names to Python expressions over tensor shapes and params. Required for arbitrary-rank ops.
- Op signatures must match PyTorch's public API (parameter names, parameter set, semantics). Do not invent parameters.
- No `Optional[Tensor]` in manifest. Ops with conditional inputs are split into variant entries linked by `variant_of`.
  - `variant_of` is one level only. Variant → primary. Primary must not have `variant_of`.
  - Variants share `source.kernel` and `source.op`. Each has its own `signature`, `workloads`, `roofline`.
- Tensor layout defaults to contiguous row-major. When an op requires non-default layout (e.g., `channels_last`), add `layout` field to the tensor declaration. `shape` dimension names reflect actual memory order.
- `source.kernel_map` declares the static dispatch key → Kernel class mapping. Keys are stable snake_case identifiers; they MUST NOT be derived from `cls.__name__`. Values are Kernel class names matching `cls.__name__`. Required when `status: implemented`, optional when `status: spec-only`.
- The manifest `kernel_map` MUST match the Op's `default_kernel_map` exactly — same keys, same class names. Runtime conditionals (e.g., architecture-dependent dispatch) are not represented in the manifest.
- Never modify manifest to match non-conforming code. If code doesn't match spec: set `status: spec-only` and add a comment explaining the discrepancy, then fix implementation in a follow-up PR. Never remove params, vars, or shape_rules to silence validator errors.
