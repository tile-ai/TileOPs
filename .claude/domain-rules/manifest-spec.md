→ [trust-model.md §Manifest](../../docs/design/trust-model.md#manifest)

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

- `torch_compile_fullgraph`: literal `true` only; omit for no promise; invalid on `spec-only`. Declare only ops with a registered cold `fullgraph=True` compile test. Semantics: [manifest.md](../../docs/design/manifest.md#torch_compile_fullgraph).

- A tensor input the op reads may declare `optional: true` under `signature.inputs`. "Not passed" means bound to `None`; presence is a fact kernel dispatch may read. Params express optionality with `default`. A caller-supplied `out=` buffer is not an optional input.

- An optional input's name may appear only in its own `dtype` / `shape` declaration, in a bare `X is None` / `X is not None` test, or in a use guarded by `X is None` earlier in the same expression. `shape_rules` take `X is None or <condition>`; a roofline presence test goes in a `vars` entry, which `flops` / `bytes` then read, and a formula needing the tensor's own shape uses `roofline: {func: ...}`. Symbols first bound in `X`'s `shape` may not appear in a required input's or output's `shape`. Per-position table: [manifest.md](../../docs/design/manifest.md#optional-inputs).

- Every optional input needs a workload row that passes it and one that omits it, counted per input rather than per combination. Param values and kernel shape ranges are out of scope.

- Merging a signature does not merge the performance account: workload rows stay split by presence, and roofline counts the optional inputs the call actually passed rather than assuming all of them.

- Output names and count are fixed per entry; an op whose return changes with a switch is two entries. Entries that stay separate are independent, sharing only their `source.op` path; no field links them.

- The validator parses `shape_rules` and checks where names appear. It does not evaluate them, does not enumerate ways to call the op, and does not stop a call that passes half of a co-occurring group — the op's `forward` raises for that, naming the group.

- Tensor layout defaults to contiguous row-major. Non-default needs an explicit `layout` field; `shape` dim names reflect memory order.

- `source.kernel_map`, `source.test` and `source.bench` are discoverability pointers: they name where the current implementation, test and benchmark live, and are retargeted whenever those move. `source.kernel` and `source.op` are contract — they say where the op is *defined*, written as the path inside the distribution because the manifest ships in the wheel and must not name a path absent there; on disk they resolve under `src/`.

- `source.kernel_map` is the Op→Kernel dispatch registration table (`dispatch_key: KernelClassName`). It declares what an Op uses, not how dispatch picks.

- Never modify manifest to match non-conforming code. Code drift → `status: spec-only` and fix code in a follow-up PR. Never remove `params`, roofline `vars`, or `shape_rules` to silence validator errors.

- `ref_api` is the spec oracle for the manifest signature, not an Op-layer dispatch target at forward time.

- **Manifest comment policy.** Comments may carry technical content the DSL can't express (schema clarifications, edge cases, conventions, file headers); they MUST NOT carry process metadata bound to a specific issue, PR, commit, or round. Keep only if meaningful after every issue/PR is renumbered; otherwise move to commit message, PR description, or follow-up issue.

  Discovery scan: `grep -rnE '#[0-9]{3,}|[Ff]ollow.?up|AC-[0-9]+' src/tileops/manifest/*.yaml`
