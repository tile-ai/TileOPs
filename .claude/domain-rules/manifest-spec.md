## Boundary

- **OWNS**: `tileops/manifest/`
- **MUST NOT WRITE**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `benchmarks/`
- Manifest changes require human review in a separate PR.

→ [trust-model.md §Manifest](../../docs/design/trust-model.md#manifest)

______________________________________________________________________

- Manifest key must equal the Op `cls.__name__` exactly.

- `ref_api` (required): the external API the signature mirrors (e.g. `torch.nn.functional.rms_norm`); `"none"` if none. Informational, not validated.

- `inputs`, `outputs`, `params` are ordered dicts — key order is signature position. Don't reorder.

- Op signatures must match PyTorch's public API (parameter names, set, semantics). Don't invent parameters.

- Params include all PyTorch-supported parameters even if the kernel only honors the default. Default to `__init__` kwargs (architecture-decided, lifetime-fixed); a param belongs in `forward()` only when the reference API requires it or the value is per-batch — justify in the introducing issue.

- `dtype` syntax: `|` for alternatives. `same_as(ref)` is dtype-only identity (matches `ref` at runtime, no extra axis in `dtype_combos`, never used for shape).

- `dtype_combos` only when the supported set is a strict subset of the Cartesian product. Omit when all combinations are valid.

- Every output tensor's shape is fully specified by `shape` and/or `shape_rules`. Inputs may omit `shape` (= arbitrary rank).

- `shape` present → fixed rank, names become roofline variables. `shape` absent → arbitrary rank; use `params` + `shape_rules`.

- Shared dimension names across tensors → sizes must match.

- `shape_rules` are Python expressions describing shape relationships.

- Reduction-dim validation: never silently wrap out-of-range indices with `% x.ndim`. Use the canonical predicates / extractors in `tileops.manifest.shape_rules` (callable by bare name from any rule body); inline string expressions are a transitional fallback only.

- Roofline `vars` maps variable names to Python expressions over tensor shapes and params. Required for arbitrary-rank ops.

- `status` is required: `implemented` or `spec-only`.

- No `Optional[Tensor]` in manifest. Conditional inputs split into variant entries linked by `variant_of` (single-level, no chaining). Variants share `source.kernel` and `source.op`; each carries its own `signature`, `workloads`, `roofline`.

- Tensor layout defaults to contiguous row-major. Non-default needs an explicit `layout` field; `shape` dim names reflect memory order.

- `source.kernel_map` is the Op→Kernel dispatch registration table (`dispatch_key: KernelClassName`). It declares what an Op uses, not how dispatch picks.

- Never modify manifest to match non-conforming code. Code drift → `status: spec-only` and fix code in a follow-up PR. Never remove `params`, roofline `vars`, or `shape_rules` to silence validator errors.

- **Manifest comment policy.** Comments may carry technical content the DSL can't express (schema clarifications, edge cases, conventions, file headers). They MUST NOT carry development-process metadata bound to a specific issue / PR / commit / round. Test: would the comment still be meaningful if every issue and PR were renumbered? Yes → keep. No → move to commit message, PR description, or follow-up issue.

  Discovery scan: `grep -rnE '#[0-9]{3,}|[Ff]ollow.?up|AC-[0-9]+' tileops/manifest/*.yaml`
