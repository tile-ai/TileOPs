## Boundary

- **OWNS**: `tileops/manifest/`
- **MUST NOT modify**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `benchmarks/`
- **MAY READ**: PyTorch public API (to match signatures)
- Manifest changes require human review in a separate PR.

→ [trust-model.md §Manifest](../../docs/design/trust-model.md#manifest)

______________________________________________________________________

- Manifest keys are PascalCase Op class names: `{PascalCaseName}{Direction}Op` (e.g., `RMSNormFwdOp`). The key must exactly match the Python class name (`cls.__name__`). No abbreviation rules — the manifest author determines the name.
- `ref_api` (required) — declares the external API the signature follows (e.g., `torch.nn.functional.rms_norm`). Set to `"none"` if no direct external counterpart exists. Informational only; does not affect validation.
- `inputs`, `outputs`, `params` are ordered dicts. Key order = function signature position. Do not reorder.
- Params include all PyTorch-supported parameters, even if the current kernel only supports the default. Params default to `__init__` kwargs (architecture-decided, fixed for the Op instance's lifetime); in rare cases a param belongs in `forward()` when PyTorch's reference API requires it or when the value is per-batch — justify the exception in the op's introducing issue.
- `dtype` syntax: `|` for alternatives. `same_as(ref)` is a dtype-only identity constraint: the tensor must have the exact same dtype as `ref` at runtime, does not contribute an independent axis to the Cartesian product in `dtype_combos`, and must not be used for shape.
- `dtype_combos` when supported set is a strict subset of the Cartesian product. Omit when all combinations are valid.
- Every output tensor's shape must be fully specified via `shape` and/or `shape_rules`. Inputs may omit `shape` (→ arbitrary rank).
- `shape` present = fixed rank. Names become roofline variables. `shape` absent = arbitrary rank, use `params` + `shape_rules`.
- Shared dimension names across tensors = sizes must match.
- `shape_rules` are Python expressions for shape relationships. `shape` and `shape_rules` fully specify output shape derivation.
- For reduction ops whose `dim` accepts an integer or sequence (`list[int]` / `tuple[int, ...]`): encode the contract as `shape_rules` in the order **validate range → normalize negatives → enforce uniqueness**. (a) Range validity: `"dim is None or all(-x.ndim <= d < x.ndim for d in ([dim] if isinstance(dim, int) else dim))"` — do NOT silently wrap out-of-range indices with `% x.ndim`; PyTorch rejects them. (b) Downstream shape_rules and roofline.vars apply `% x.ndim` only after range has been validated, using `{d % x.ndim for d in dim}` to deduplicate. (c) Uniqueness: `"isinstance(dim, (int, type(None))) or len({d % x.ndim for d in dim}) == len(dim)"`. Empty sequence equals full reduction for ops accepting `dim=None`; ops that do not accept `None` (e.g. logsumexp) additionally declare `"isinstance(dim, int) or len(dim) > 0"`.
- `status` is required. `status: implemented` = all validator levels apply. `status: spec-only` = L0 only.
- Roofline `vars` maps variable names to Python expressions over tensor shapes and params. Required for arbitrary-rank ops.
- Op signatures must match PyTorch's public API (parameter names, parameter set, semantics). Do not invent parameters.
- No `Optional[Tensor]` in manifest. Ops with conditional inputs split into variant entries linked by `variant_of`, which is single-level (variant → primary, no chaining). Variants share `source.kernel` and `source.op`; each has its own `signature`, `workloads`, `roofline`.
- Tensor layout defaults to contiguous row-major. When an op requires non-default layout (e.g., `channels_last`), add `layout` field to the tensor declaration. `shape` dimension names reflect actual memory order.
- `source.kernel_map` is the Op→Kernel dispatch registration table (`dispatch_key: KernelClassName`). It declares which Kernels an Op uses so agents know what to implement. Required when `status: implemented`, optional when `status: spec-only`. Does not describe dispatch strategy.
- Never modify manifest to match non-conforming code. If code doesn't match spec: set `status: spec-only` and add a comment explaining the discrepancy, then fix implementation in a follow-up PR. Never remove params, vars, or shape_rules to silence validator errors.
