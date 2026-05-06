## Boundary

- **OWNS**: `tileops/manifest/`
- **MUST NOT WRITE**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `benchmarks/`
- Manifest changes require human review in a separate PR.

→ [trust-model.md §Manifest](../../docs/design/trust-model.md#manifest)

______________________________________________________________________

- Manifest key must equal the corresponding Op `cls.__name__` exactly. Class-naming convention is in `ops-design.md`.

- `ref_api` (required) — declares the external API the signature follows (e.g., `torch.nn.functional.rms_norm`). Set to `"none"` if no direct external counterpart exists. Informational only; does not affect validation.

- `inputs`, `outputs`, `params` are ordered dicts. Key order = function signature position. Do not reorder.

- Params include all PyTorch-supported parameters, even if the current kernel only supports the default. Params default to `__init__` kwargs (architecture-decided, fixed for the Op instance's lifetime); in rare cases a param belongs in `forward()` when PyTorch's reference API requires it or when the value is per-batch — justify the exception in the op's introducing issue.

- `dtype` syntax: `|` for alternatives. `same_as(ref)` is a dtype-only identity constraint: the tensor must have the exact same dtype as `ref` at runtime, does not contribute an independent axis to the Cartesian product in `dtype_combos`, and must not be used for shape.

- `dtype_combos` when supported set is a strict subset of the Cartesian product. Omit when all combinations are valid.

- Every output tensor's shape must be fully specified via `shape` and/or `shape_rules`. Inputs may omit `shape` (→ arbitrary rank).

- `shape` present = fixed rank. Names become roofline variables. `shape` absent = arbitrary rank, use `params` + `shape_rules`.

- Shared dimension names across tensors = sizes must match.

- `shape_rules` are Python expressions for shape relationships. `shape` and `shape_rules` fully specify output shape derivation.

- Reduction-dim validation: do NOT silently wrap out-of-range indices with `% x.ndim`. Canonical predicates are registered in `tileops.manifest.shape_rules.HELPERS` and referenced via the `helper:` URI prefix; new reduction ops MUST use the registry, inline string expressions are a transitional fallback only.

- `status` is required. `status: implemented` = all validator levels apply. `status: spec-only` = L0 only.

- Roofline `vars` maps variable names to Python expressions over tensor shapes and params. Required for arbitrary-rank ops.

- Op signatures must match PyTorch's public API (parameter names, parameter set, semantics). Do not invent parameters.

- No `Optional[Tensor]` in manifest. Ops with conditional inputs split into variant entries linked by `variant_of`, which is single-level (variant → primary, no chaining). Variants share `source.kernel` and `source.op`; each has its own `signature`, `workloads`, `roofline`.

- Tensor layout defaults to contiguous row-major. When an op requires non-default layout (e.g., `channels_last`), add `layout` field to the tensor declaration. `shape` dimension names reflect actual memory order.

- `source.kernel_map` is the Op→Kernel dispatch registration table (`dispatch_key: KernelClassName`). It declares which Kernels an Op uses so agents know what to implement. Required when `status: implemented`, optional when `status: spec-only`. Does not describe dispatch strategy.

- Never modify manifest to match non-conforming code. If code doesn't match spec: set `status: spec-only` and fix implementation in a follow-up PR. Never remove params, vars, or shape_rules to silence validator errors.

- **Manifest comment policy.** Manifest YAML carries technical content the DSL cannot express structurally (schema clarifications, edge cases, conventions, file-level headers). It does **not** carry development-process metadata — anything bound to a specific issue, PR, commit, or development round. Test: would this comment still be meaningful if every issue / PR had different numbers and every milestone was renamed? Yes → keep. No → move to commit message, PR description, or follow-up issue.

  Discovery scan (flags candidates, not a hard gate): `grep -rnE '#[0-9]{3,}|[Ff]ollow.?up|AC-[0-9]+' tileops/manifest/*.yaml`
