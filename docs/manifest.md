# Op Manifest Specification

[`ops_manifest.yaml`](../tileops/ops_manifest.yaml) is the **source of truth** for op interfaces, benchmark workloads, and roofline metadata.

## Trust Model

```mermaid
flowchart LR
    H["Human reviewer"] -->|writes / approves| M["ops_manifest.yaml"]
    M -->|reads spec from| A["Agent (codegen)"]
    A -->|produces| C["Op code, tests, benchmarks"]
    M -->|validates against| V["Validator (CI)"]
    C -->|checked by| V
```

- **Human reviewer** — only actor that modifies the manifest. All changes require PR review.
- **Agent** — generates Ops, tests, benchmarks from the manifest. Reads only, never modifies.
- **Validator** — [`scripts/validate_manifest.py`](../scripts/validate_manifest.py) in CI. Enforces manifest ↔ code consistency.

**Invariants:**

1. The manifest is the sole source of truth for op interfaces.
1. Validation is derived from the manifest, not from the generating agent.
1. `workloads` define benchmark shapes/dtypes, not unit-test coverage.
1. `signature.params` ⊆ Op's `__init__()` + `forward()` param names. `forward()` params must match manifest inputs in order. CI enforces this.
1. Benchmarks must use declared workloads via `load_workloads`. No hardcoded shapes.

## Rules

**R1. Ordered dict.** `inputs`, `outputs`, `params` are keyed by name. Key order = function signature position. Reordering is a breaking change.

> YAML does not guarantee order. This project relies on Python 3.7+ dict insertion-order via `yaml.safe_load()`. All consumers MUST use an order-preserving parser.

**R2. Full interface.** Params include all PyTorch-supported parameters, even if the kernel only supports the default.

**R2a. Param placement — default rule.** Params are `__init__` kwargs by default (architecture-decided, fixed for the Op instance's lifetime). In rare cases a param belongs in `forward()` when PyTorch's reference API requires it or when the value is per-batch. The exception is justified in the op's introducing issue; the manifest schema does not encode the distinction.

**R3. `dtype` syntax.** `|` for alternatives. `same_as(ref)` is a dtype-only identity constraint: the tensor must have the exact same dtype as `ref` at runtime, does not contribute an independent axis to the Cartesian product in R4, and must not be used for shape.

**R4. `dtype_combos`.** Enumerates supported cross-tensor dtype combinations.

- **Present:** exhaustive. Only listed combinations are valid.
- **Absent:** all Cartesian-product combinations are assumed valid.

Use when the supported set is a strict subset (e.g., mixed-precision GEMM). Omit when all combos are valid (e.g., all inputs `same_as(x)`).

```yaml
dtype_combos:
  - {x: float16, weight: float16}
  - {x: float16, weight: float8_e4m3}
  - {x: bfloat16, weight: bfloat16}
```

**R5. Explicit shape.** Every output tensor's shape must be fully specified via `shape` and/or `shape_rules`. Input tensors may omit `shape` (→ arbitrary rank per R7).

**R6. `shape` = fixed rank.** Declares exact dimensions (e.g., `"[M, K]"`). No ellipsis or wildcards. Roofline variable binding is defined in [roofline.md](roofline.md).

**R7. No `shape` = arbitrary rank.** Constraints go in `params` + `shape_rules`. Optionally, `static_dims` declares values the user commits to at Op construction time (R20).

**R8. No shape aliasing.** Each tensor declares its own shape. Use shared dimension names (R9) or `shape_rules` (R11) to express shape relationships.

**R9. Shared dimension names = equality.** `K` in two tensors means their sizes must match.

**R10. `constraints`.** Restricts dimensions: `"64 | 128 | 256"` (enumerated) or `"power_of_2"`, `"divisible_by(k)"`, `"even"`, `"positive"` (predicates). Requires `shape`.

**R11. `shape_rules`.** Python expressions for shape relationships. Required when `shape` alone cannot fully specify output shape.

**R13. Status gating.** `status: spec-only` → L0 only. `status: implemented` → all levels. `--check-op <name>` forces L0-L4 on a targeted entry (includes its variants).

**R14. Roofline metadata.** See [roofline.md](roofline.md). That document is the source of truth for roofline modes, variable binding, formula syntax, consumers, and codegen behavior.

**R15. PyTorch API alignment.** Op signatures match PyTorch's public API (names, parameter set, semantics). Do not invent parameters.

**R16. No Optional[Tensor].** Fixed tensor inputs per entry. Conditional inputs split into variants via `variant_of`, which is single-level (variant → primary, no chaining). Variants share `source.kernel` and `source.op`; each has its own `signature`, `workloads`, `roofline`.

**R19. Tensor layout.** Default: contiguous row-major (no `layout` field). Non-default: add `layout` field, `shape` names reflect memory order.

**R20. `static_dims`.** For arbitrary-rank ops (no `shape` declaration), `static_dims` declares values the user commits to at Op construction time. Each entry maps an `__init__` keyword name to a single-axis shape expression `<tensor>.shape[<const_or_param>]`. See [`static_dims`](#static_dims) for full semantics, rules, and examples.

## `static_dims`

`static_dims` declares what becomes statically known at the moment the user constructs the Op instance. It is **per-op**, not per-family.

```yaml
static_dims:
  N: "x.shape[dim]"
```

### Semantics

The shape expression is a **forward-time validation rule**, not an init-time derivation — no tensor exists at `__init__`. Two time points, one contract:

- `__init__` is the **commitment point**. The user-supplied value is stored on `self`. The expression is NOT evaluated here.
- `forward` is the **validation point**. The expression is evaluated against the actual tensor and must match the committed value.

```python
# __init__ — commitment point. No tensor; expression not evaluated.
def __init__(self, *, N: int, dtype: torch.dtype, dim: int = -1, ...):
    self.N = N
    self.dtype = dtype
    self.dim = dim
    # ...

# forward — validation point. Expression evaluated against the actual tensor.
def forward(self, x: torch.Tensor):
    if x.shape[self.dim] != self.N:
        raise ValueError(
            f"static_dim mismatch: expected x.shape[{self.dim}] == {self.N}, "
            f"got {x.shape[self.dim]}"
        )
    # ... rest of forward
```

### Rules

- Every `static_dims` entry's key is a required `__init__` keyword parameter. **No defaults**; the user must supply every committed value at ctor.
- The expression MUST be a **single-axis reference** of the form `<tensor>.shape[<const_or_param>]`. Multi-axis forms (e.g., `product(x.shape[i] for i in ...)`, comprehensions, arithmetic over shape) are forbidden.
- Referenced tensor names must be in `signature.inputs`. Referenced axis names (when not integer literals) must be in `signature.params`.
- Key order determines the order those kwargs appear in the generated `__init__`, consistent with R1.
- `static_dims` is only for arbitrary-rank ops. Fixed-rank ops get dimensions from `shape` (R6).

### Evaluation context

Shared with `shape_rules`: all `signature.inputs` tensor names (with `.shape` accessor) and all `signature.params` names.

### Multi-input example — LinearFwdOp

The expression may reference any tensor in `signature.inputs`, not just the primary one. For `torch.nn.functional.linear(input, weight, bias)` with arbitrary-rank `input`:

```yaml
LinearFwdOp:
  signature:
    inputs:
      input:  {dtype: "float16 | bfloat16"}
      weight: {dtype: "same_as(input)"}
      bias:   {dtype: "same_as(input)"}
    outputs:
      output: {dtype: "same_as(input)"}
    static_dims:
      in_features:  "input.shape[-1]"
      out_features: "weight.shape[0]"
    shape_rules:
      - "weight.shape == (out_features, in_features)"
      - "bias.shape == (out_features,)"
      - "output.shape == input.shape[:-1] + (out_features,)"
```

`out_features` is intrinsically a property of `weight`, not `input` — there is no equivalent expression in terms of `input.shape`. Binding to `weight.shape[0]` is the only faithful declaration.

### Generated `__init__` kwarg block order

The signature has three blocks in this order:

1. `static_dims` entries — in manifest key order
1. `dtype` (single parameter, unless the op has explicit multi-dtype axes)
1. `params` entries — in manifest key order

All parameters are keyword-only (`*`-separated). Block order determines the visible signature for documentation and introspection; callers always use kwargs.

### Empty `static_dims`

Empty (`static_dims: {}` or absent) is legal. Typical case: PyTorch-aligned reductions that accept `dim=None`, where the reduction extent depends on the entire input shape and is not a user-provided hyperparameter:

```yaml
SumFwdOp:
  signature:
    inputs:  {x: {dtype: "..."}}
    outputs: {y: {dtype: "same_as(x)"}}
    params:
      dim:     {type: "int | list[int] | tuple[int, ...] | None", default: -1}
      keepdim: {type: bool, default: false}
    # static_dims absent — equivalent to static_dims: {}
    shape_rules: [...]
```

The generated `__init__` has no shape kwargs:

```python
def __init__(self, *, dtype, dim=-1, keepdim=False, ...):
    # ...
```

**When `static_dims` is empty, the Op author MUST override `_cache_key`.** The Op base class's default `_cache_key` falls back to the full input shape tuple when no axes are committed — correct, but pathological under dynamic shapes: every distinct input shape produces a new kernel compile. A typical full-reduce override:

```python
class SumFwdOp(Op):
    def _cache_key(self, x_shape):
        return (
            math.prod(x_shape),
        )  # all full-reductions with same numel share a kernel
```

The base class emits a once-per-type runtime warning if the default `_cache_key` is invoked with empty `static_dims` and no subclass override, to catch missing overrides early. See [ops-design.md § Implementing an Op](ops-design.md#implementing-an-op) for the `_cache_key` interface.

## Manifest Key Format

Each entry in `ops:` is keyed by the **Python class name** of the Op — PascalCase with a mandatory direction suffix and `Op` suffix:

```
{PascalCaseName}{Direction}Op
```

- **PascalCaseName** — the op's descriptive name in PascalCase (e.g., `RMSNorm`, `BatchNorm`, `Softmax`). No abbreviation rules are enforced; the manifest author determines the name.
- **Direction** — a mandatory suffix indicating the computation direction: `Fwd` or `Bwd`.
- **Op** — the literal suffix `Op`.

Examples: `RMSNormFwdOp`, `BatchNormFwdOp`, `SoftmaxFwdOp`, `LinearFwdOp`.

The validator enforces `assert cls.__name__ == manifest_key` — the manifest key must exactly match the Op class name. There is no heuristic resolution or snake_case-to-PascalCase conversion.

## Entry Structure

| Field       | Required | Description                                                   |
| ----------- | -------- | ------------------------------------------------------------- |
| `family`    | yes      | Op family. See [below](#family).                              |
| `ref_api`   | yes      | External API reference, or `"none"` if no direct counterpart. |
| `status`    | yes      | `spec-only` or `implemented`.                                 |
| `signature` | yes      | Op interface. See [Signature](#signature).                    |
| `workloads` | yes      | Benchmark shapes/dtypes.                                      |
| `roofline`  | yes      | Performance model.                                            |
| `source`    | yes      | Implementation paths.                                         |

### `family`

Closed set: `elementwise`, `reduction`, `normalization`, `convolution`, `gemm`, `quantize`, `sampling`, `attention`, `moe`, `linear_attention`, `ssm`, `scan`.

### `ref_api`

Fully qualified external API name (typically PyTorch), or `"none"` if no direct counterpart. Required but informational — validator checks presence only, does not affect signature validation or code generation.

```yaml
RMSNormFwdOp:
  ref_api: "torch.nn.functional.rms_norm"
NSAFwdOp:
  ref_api: "none"
```

### Signature

```yaml
signature:
  inputs:       # tensor name → {dtype, shape?, constraints?}
  outputs:      # tensor name → {dtype, shape?, constraints?}
  params:       # param name → {type, default?}
  static_dims:  # kwarg → "<tensor>.shape[<axis>]" — arbitrary-rank only (R20)
  shape_rules:  # Python expressions for shape inference
  dtype_combos: # valid cross-tensor dtype combinations
```

**Tensor fields:**

| Field         | Required | Description                                                |
| ------------- | -------- | ---------------------------------------------------------- |
| `dtype`       | yes      | `\|` for alternatives, `same_as(ref)` = same dtype as ref. |
| `shape`       | no       | Dimension names (e.g., `"[M, K]"`). Present = fixed rank.  |
| `constraints` | no       | Dimension restrictions (requires `shape`).                 |
| `layout`      | no       | Memory format when non-default (R19).                      |

**Param fields:** `type` (string: `int`, `float`, `bool`, `"list[int]"`) + optional `default`.

#### Shape Decision Tree

```
Fixed rank, expressible with dimension names?
├─ YES → shape: "[D1, D2, ...]"                           [R6]
│   Relationships beyond shared names?
│   └─ YES → add shape_rules                              [R11]
└─ NO (arbitrary rank)
   ├─ write shape_rules                                   [R11]
   └─ Values committed at Op construction time?
      └─ YES → add static_dims                            [R20]
```

#### Optional Inputs

Manifest does not support `Optional[Tensor]` (R16). Split into variant entries with fixed signatures, linked by `variant_of`.

**Decision tree:**

```
Op has Optional[Tensor] inputs?
├─ NO → single entry
└─ YES
   ├─ 1 optional → 2 entries (primary + variant)
   ├─ 2 optionals, always together → 2 entries
   ├─ 2 optionals, independent → up to 4 entries
   └─ 3+ → decompose the op first
```

**Naming:** Variants follow the same PascalCase key format, with a descriptive suffix inserted before `{Direction}Op` (e.g., `MoEFusedMoeCbFwdOp` where `Cb` is the variant suffix).

### Workloads

Shape keys use `<tensor_name>_shape`. Op-specific parameters can be added per entry.

```yaml
- {x_shape: [2048, 4096], dtypes: [float16, bfloat16], label: "llama-3.1-8b"}
```

`workloads` are for benchmark parametrization only, not unit-test coverage.

### Roofline

Roofline metadata is required on every manifest entry. Its modes,
variable binding rules, formula syntax, consumers, and codegen behavior
are defined in [roofline.md](roofline.md).

### Source

| Field                   | Required | Description                                                            |
| ----------------------- | -------- | ---------------------------------------------------------------------- |
| `kernel`                | yes      | Kernel file path(s).                                                   |
| `kernel_map`            | \*       | Dispatch key → Kernel class name. Required when `status: implemented`. |
| `op`                    | yes      | Op class file path.                                                    |
| `test`                  | yes      | Test file path.                                                        |
| `bench`                 | yes      | Benchmark file path.                                                   |
| `bench_manifest_driven` | no       | `true` = L4 is a hard CI error. Migration flag.                        |

#### kernel_map

Op→Kernel dispatch registration table. Declares which Kernels an Op uses so agents know what to implement. Does not describe dispatch strategy (runtime concern). Format: `dispatch_key: KernelClassName`. See [ops-design-reference.md § Kernel Dispatch](ops-design-reference.md#kernel-dispatch-kernel_map).

```yaml
# Single-kernel op
source:
  kernel: tileops/kernels/norm/rms_norm.py
  kernel_map:
    rms_norm: RmsNormKernel
  op: tileops/ops/norm/rms_norm.py

# Multi-kernel op
source:
  kernel:
    - tileops/kernels/attention/gqa_bwd.py
  kernel_map:
    mha_bwd_preprocess_kernel: FlashAttnBwdPreprocessKernel
    mha_bwd_kernel: MHABwdKernel
    mha_bwd_postprocess_kernel: FlashAttnBwdPostprocessKernel
  op: tileops/ops/attention/mha.py
```

- Optional when `status: spec-only`. Required when `status: implemented`.

## Entry Examples

**Fixed rank — GEMM** \[R6, R9\]:

```yaml
inputs:
  a: {dtype: "float16 | bfloat16", shape: "[M, K]"}
  b: {dtype: "same_as(a)", shape: "[K, N]"}
outputs:
  c: {dtype: "same_as(a)", shape: "[M, N]"}
```

**Fixed rank + constraints — FFT** \[R6, R10\]:

```yaml
inputs:
  x: {dtype: "complex64", shape: "[M, N]", constraints: {N: "power_of_2"}}
outputs:
  y: {dtype: "same_as(x)", shape: "[M, N]"}
```

**Arbitrary rank — RMSNorm** \[R7, R11, R20\]:

```yaml
inputs:
  x: {dtype: "float16 | bfloat16"}
  weight: {dtype: "same_as(x)"}
outputs:
  y: {dtype: "same_as(x)"}
params:
  dim: {type: int, default: -1}
  eps: {type: float, default: 1e-6}
static_dims:
  N: "x.shape[dim]"
shape_rules:
  - "y.shape == x.shape"
  - "weight.shape == (x.shape[dim],)"
```

**Arbitrary rank — Reduce** \[R7, R11\]:

```yaml
inputs:
  x: {dtype: "float16 | bfloat16"}
outputs:
  y: {dtype: "same_as(x)"}
params:
  dim: {type: "int | list[int] | tuple[int, ...] | None", default: -1}
  keepdim: {type: bool, default: false}
shape_rules:
  # R11a step 1 — range validity: every axis must be in [-ndim, ndim)
  - "dim is None or all(-x.ndim <= d < x.ndim for d in ([dim] if isinstance(dim, int) else dim))"
  # R11a step 2 — reduce_axes is a SET of normalized axis indices
  - "y.ndim == x.ndim if keepdim else x.ndim - len({dim % x.ndim} if isinstance(dim, int) else {d % x.ndim for d in dim} if isinstance(dim, (list, tuple)) and len(dim) > 0 else set(range(x.ndim)))"
  - "y.shape[i] == (1 if i in ({dim % x.ndim} if isinstance(dim, int) else {d % x.ndim for d in dim} if isinstance(dim, (list, tuple)) and len(dim) > 0 else set(range(x.ndim))) and keepdim else x.shape[i])"
  # R11a step 3 — sequence dims must be unique after normalization
  - "isinstance(dim, (int, type(None))) or len({d % x.ndim for d in dim}) == len(dim)"
```

All reduction ops include `dim` + `keepdim`. **Exception:** softmax/log_softmax preserve input shape (no `keepdim`); use `shape_rules` to express `y.shape == x.shape`. count_nonzero has no `keepdim` (per R15).

**R11a. `dim` contract for reduction ops.** When `dim` accepts an integer or a sequence (`list[int]` / `tuple[int, ...]`), the manifest expresses the contract as three `shape_rules`, in this order:

1. **Range validity.** Every axis must satisfy `-x.ndim <= d < x.ndim`. Out-of-range indices are invalid in PyTorch; the manifest must not silently wrap them with `% x.ndim`. Declare:
   `"dim is None or all(-x.ndim <= d < x.ndim for d in ([dim] if isinstance(dim, int) else dim))"` (for ops accepting `None`), or drop the `dim is None or` prefix for ops that do not.
1. **Normalize negatives.** Downstream expressions apply `% x.ndim` only after step 1 has validated range, producing a canonical non-negative axis set `{d % x.ndim for d in dim}`.
1. **Uniqueness (sequence only).** After normalization, entries must be pairwise distinct. PyTorch rejects duplicates. Declare:
   `"isinstance(dim, (int, type(None))) or len({d % x.ndim for d in dim}) == len(dim)"`.

**Empty-sequence semantics is per-op:**

- Ops accepting `dim=None` (`sum`, `mean`, `amax`, `amin`, `var`, `std`, `var_mean`, `all`, `any`, `count_nonzero`, `linalg.vector_norm` variants): empty sequence is equivalent to `dim=None` (full reduction). Formulas use `set(range(x.ndim))` as the fallback axis set.
- Ops that do **not** accept `dim=None` (e.g. `logsumexp`): empty sequence is invalid; declare `"isinstance(dim, int) or len(dim) > 0"`.

These rules are `shape_rules` (Python expressions) rather than a new manifest field — they reuse the existing vocabulary and are enforceable by future codegen or by the op's forward-time validation.

**Full entry — RMSNorm:**

```yaml
ops:
  RMSNormFwdOp:
    family: normalization
    ref_api: "torch.nn.functional.rms_norm"
    status: implemented

    signature:
      inputs:
        x: {dtype: "float16 | bfloat16"}
        weight: {dtype: "same_as(x)"}
      outputs:
        y: {dtype: "same_as(x)"}
      params:
        dim: {type: int, default: -1}
        eps: {type: float, default: 1e-6}
      static_dims:
        N: "x.shape[dim]"
      shape_rules:
        - "y.shape == x.shape"
        - "weight.shape == (x.shape[dim],)"

    workloads:
      - {x_shape: [2048, 4096], dtypes: [float16, bfloat16], label: "llama-3.1-8b-prefill"}
      - {x_shape: [1, 4096], dtypes: [bfloat16], label: "llama-3.1-8b-decode"}

    roofline:
      vars:
        M: "product(x.shape[:dim])"
        N: "x.shape[dim]"
      flops: "4 * M * N"
      bytes: "(2 * M * N + N) * elem_bytes"

    source:
      kernel: tileops/kernels/norm/rms_norm.py
      op: tileops/ops/norm/rms_norm.py
      test: tests/ops/test_rms_norm.py
      bench: benchmarks/ops/bench_rms_norm.py
```

## Benchmark Pattern

Benchmarks must use manifest-driven workloads. See [testing.md](testing.md)
for benchmark structure and [roofline.md](roofline.md) for roofline
consumption.

### Workload entry schema

Each entry under `workloads:` is a mapping. Three keys — `x_shape`,
`dtypes`, `label` — are reserved by the schema; any other key becomes
an **op-call parameter** forwarded to the op's `__init__`.

| Key             | Required | Meaning                                                                                                  |
| --------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `x_shape`       | yes      | Input tensor shape (list of ints).                                                                       |
| `dtypes`        | yes      | List of dtype strings (`["float16", "bfloat16"]`).                                                       |
| `label`         | no       | Human-readable id used in the pytest param id and report tables.                                         |
| *any other key* | no       | Op param value (`dim`, `keepdim`, `correction`, …). Overrides the manifest's `signature.params` default. |

Example — parametrizing a reduction workload over a non-last `dim`:

```yaml
workloads:
  - {x_shape: [2048, 4096], dtypes: [bfloat16], dim: -1, label: "reduce-last"}
  - {x_shape: [2048, 4096], dtypes: [bfloat16], dim:  0, label: "reduce-first"}
```

## Manifest Validation

[`scripts/validate_manifest.py`](../scripts/validate_manifest.py) runs five levels:

| Level | Check     | Description                                                          |
| ----- | --------- | -------------------------------------------------------------------- |
| L0    | Schema    | Required fields exist, correct types                                 |
| L1    | Signature | Params ⊆ `__init__()` ∪ `forward()` names; `forward()` order matches |
| L2    | Shape     | `shape_rules` are valid Python expressions                           |
| L3    | Dtype     | dtype strings are valid torch types or `same_as()` refs              |
| L4    | Benchmark | Bench file uses manifest-driven workloads                            |

`spec-only` ops → L0 only. `implemented` ops → all levels. `--check-op <name>` forces L0-L4 on a targeted entry + its variants.

```bash
python scripts/validate_manifest.py
python scripts/validate_manifest.py --check-op SoftmaxFwdOp
```

## Exclusions

The manifest does NOT describe: multi-kernel execution ordering, accumulator dtypes, persistent state, tile sizes, or autotuning config.
