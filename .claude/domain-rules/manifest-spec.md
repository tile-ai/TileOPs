## Boundary

- **OWNS**: `tileops/manifest/`
- **MUST NOT WRITE**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `benchmarks/`
- Manifest changes require human review in a separate PR.

→ [trust-model.md §Manifest](../../docs/design/trust-model.md#manifest)

______________________________________________________________________

- Manifest key must equal the Op `cls.__name__` exactly. Class-naming convention: see [ops-design.md](ops-design.md).

- `ref_api` (required): the external API the signature mirrors (e.g. `torch.nn.functional.rms_norm`); `"none"` if none. Validator enforces presence + string type only; semantics not checked.

- `inputs`, `outputs`, `params` are ordered dicts — key order is signature position. Don't reorder.

- Op signatures must match PyTorch's public API (names, set, semantics); include every supported parameter even if the kernel only honors the default. Default to `__init__` kwargs (lifetime-fixed); use `forward()` only when the reference API requires it or the value is per-batch — justify in the introducing issue.

- `dtype` syntax: `|` for alternatives. `same_as(ref)` is dtype-only identity (matches `ref` at runtime, no extra axis in `dtype_combos`, never used for shape).

- `dtype_combos` only when the supported set is a strict subset of the Cartesian product. Omit when all combinations are valid.

- Output shapes are fully specified by `shape` and/or `shape_rules`. `shape` present → fixed rank, names become roofline variables; `shape` absent on inputs → arbitrary rank, use `params` + `shape_rules`. Shared dim names across tensors → sizes must match.

- `shape_rules` are Python expressions describing shape relationships. For reduction-dim validation, use the canonical predicates / extractors in `tileops.manifest.shape_rules` (callable by bare name from any rule body); never silently wrap out-of-range indices with `% x.ndim`. Inline string expressions are a transitional fallback only.

- **Reduction `dim` authoring contract.** When `dim` accepts an integer or a sequence (`list[int]` / `tuple[int, ...]`), declare three `shape_rules` in this order:

  1. **Range validity.** Every axis in `[-x.ndim, x.ndim)`. For ops accepting `None`: `"dim is None or all(-x.ndim <= d < x.ndim for d in ([dim] if isinstance(dim, int) else dim))"`. Drop the `dim is None or` prefix when the op does not accept `None`.
  1. **Normalize negatives.** Downstream rules apply `% x.ndim` only after step 1, producing the canonical axis set `{d % x.ndim for d in dim}`.
  1. **Uniqueness (sequence only).** `"isinstance(dim, (int, type(None))) or len({d % x.ndim for d in dim}) == len(dim)"`.

  Empty-sequence semantics is per-op:

  - Ops accepting `dim=None` (`sum`, `mean`, `amax`, `amin`, `var`, `std`, `var_mean`, `all`, `any`, `count_nonzero`, `linalg.vector_norm` variants): empty sequence ≡ full reduction; formulas use `set(range(x.ndim))` as fallback.
  - Ops without `dim=None` (e.g. `logsumexp`): empty sequence is invalid; declare `"isinstance(dim, int) or len(dim) > 0"`.

- Roofline `vars` maps variable names to Python expressions over tensor shapes and params. Required for arbitrary-rank ops.

- `status` is required: `implemented` or `spec-only`.

- No `Optional[Tensor]` in manifest. Conditional inputs split into variant entries linked by `variant_of` (single-level, no chaining). Variants share `source.kernel` and `source.op`; each carries its own `signature`, `workloads`, `roofline`.

- Tensor layout defaults to contiguous row-major. Non-default needs an explicit `layout` field; `shape` dim names reflect memory order.

- `source.kernel_map` is the Op→Kernel dispatch registration table (`dispatch_key: KernelClassName`). It declares what an Op uses, not how dispatch picks.

- Never modify manifest to match non-conforming code. Code drift → `status: spec-only` and fix code in a follow-up PR. Never remove `params`, roofline `vars`, or `shape_rules` to silence validator errors.

- **Manifest comment policy.** Comments may carry technical content the DSL can't express (schema clarifications, edge cases, conventions, file headers); they MUST NOT carry process metadata bound to a specific issue, PR, commit, or round. Keep only if meaningful after every issue/PR is renumbered; otherwise move to commit message, PR description, or follow-up issue.

  Discovery scan: `grep -rnE '#[0-9]{3,}|[Ff]ollow.?up|AC-[0-9]+' tileops/manifest/*.yaml`
