# Op Manifest Specification

The [`tileops/manifest/`](../../tileops/manifest/) package is the **source of truth** for op interfaces, benchmark workloads, and roofline metadata.

## Layout

One or more YAML files per family (single file by default; large families may shard). Each file is a flat top-level mapping `op_name → entry`. The `tileops.manifest` package merges all files at load; duplicate op names across files are an error.

- **Add or edit an op**: edit the family file matching the op's `family` field. Use `ruamel.yaml` for round-trip edits.
- **Read programmatically**: `from tileops.manifest import load_manifest, load_workloads, manifest_files`. `load_manifest()` returns the merged `ops` dict.
- **Read for inspection**: `yaml.safe_load` the relevant family file. No aggregate file on disk.

## Trust Model

```mermaid
flowchart LR
    H["Human reviewer"] -->|writes / approves| M["tileops/manifest/"]
    M -->|reads spec from| A["Agent (codegen)"]
    A -->|produces| C["Op code, tests, benchmarks"]
    M -->|validates against| V["Validator (CI)"]
    C -->|checked by| V
```

- **Human reviewer** — only actor that modifies the manifest. All changes require PR review.
- **Agent** — generates Ops, tests, benchmarks from the manifest. Reads only, never modifies.
- **Validator** — [`scripts/validate_manifest.py`](../../scripts/validate_manifest.py) in CI. Enforces manifest ↔ code consistency.

**Invariants:**

1. The manifest is the sole source of truth for op interfaces.
1. Validation is derived from the manifest, not from the generating agent.
1. `workloads` define benchmark shapes/dtypes, not unit-test coverage.
1. `signature.params` ⊆ Op's `__init__()` + `forward()` param names. `forward()` params must match manifest inputs in order. CI enforces this.
1. Benchmarks must use declared workloads via `load_workloads`. No hardcoded shapes.

## Rules

**R1. Ordered dict.** `inputs`, `outputs`, `params` are keyed by name. Key order = function signature position. Reordering is a breaking change. Consumers MUST use an order-preserving parser (Python 3.7+ `yaml.safe_load` qualifies).

**R2. Full interface.** Params include all PyTorch-supported parameters, even if the kernel only supports the default.

**R3. Param placement.** Default: `__init__` kwarg (architecture-decided, lifetime-fixed). Use `forward()` only when the reference API requires it or the value is per-batch; justify in the introducing issue. The manifest schema does not encode the distinction.

**R4. `dtype` syntax.** `|` for alternatives. `same_as(ref)` is a dtype-only identity constraint: the tensor must have the exact same dtype as `ref` at runtime, does not contribute an independent axis to the Cartesian product in R6, and must not be used for shape.

**R5. `promote_int_to_float(ref)`.** Output-dtype construct for ops whose integral inputs PyTorch promotes to a floating result (e.g. `torch.reciprocal`). Resolves to `float32` when `ref`'s runtime dtype is integral (`uint8` / `int8` / `int16` / `int32` / `int64`); otherwise resolves to `same_as(ref)`. May appear inside `|` unions on the output side (e.g. `"promote_int_to_float(input) | float64"`). `ref` MUST name a `signature.inputs` tensor; references to outputs or to the tensor itself are rejected. The construct is allowed only inside `signature.outputs[*].dtype`; it MUST NOT appear on input tensors, in `signature.dtype_combos` rows, or in `workloads[*].dtypes` (those positions require concrete `torch.*` dtypes or `same_as(ref)`). The validator expands the resolved dtype set when checking parity with `_validate_dtypes` and `dtype_combos`.

Worked example — `torch.reciprocal` accepts integral inputs and returns `float32`, while floating inputs round-trip:

```yaml
ReciprocalFwdOp:
  ref_api: "torch.reciprocal"
  signature:
    inputs:
      input: {dtype: "float16 | bfloat16 | float32 | int8 | int16 | int32 | int64 | uint8"}
    outputs:
      # int8/int16/int32/int64/uint8 -> float32; float16/bfloat16/float32 unchanged.
      output: {dtype: "promote_int_to_float(input)"}
```

The op-layer implementation must mirror this contract: integer inputs are cast to `float32` before the float kernel runs, and `output_dtype` is `float32` for those constructions.

**R6. `dtype_combos`.** Enumerates supported cross-tensor dtype combinations.

- **Present:** exhaustive. Only listed combinations are valid.
- **Absent:** all Cartesian-product combinations are assumed valid.

Use when the supported set is a strict subset (e.g., mixed-precision GEMM). Omit when all combos are valid (e.g., all inputs `same_as(x)`).

```yaml
dtype_combos:
  - {x: float16, weight: float16}
  - {x: float16, weight: float8_e4m3}
  - {x: bfloat16, weight: bfloat16}
```

**R7. Explicit shape.** Every output tensor's shape must be fully specified via `shape` and/or `shape_rules`. Input tensors may omit `shape` (→ arbitrary rank per R9).

**R8. `shape` = fixed rank.** Declares exact dimensions (e.g., `"[M, K]"`). No ellipsis or wildcards. Roofline variable binding is defined in [roofline.md](roofline.md).

**R9. No `shape` = arbitrary rank.** Constraints go in `params` + `shape_rules`. Optionally, `static_dims` declares values the user commits to at Op construction time (R20).

**R10. No shape aliasing.** Each tensor declares its own shape. Use shared dimension names (R11) or `shape_rules` (R13) to express shape relationships.

**R11. Shared dimension names = equality.** `K` in two tensors means their sizes must match.

**R12. `constraints`.** Restricts dimensions: `"64 | 128 | 256"` (enumerated) or `"power_of_2"`, `"divisible_by(k)"`, `"even"`, `"positive"` (predicates). Requires `shape`.

**R13. `shape_rules`.** Python expressions for shape relationships. Required when `shape` alone cannot fully specify output shape.

**R14. Reduction `dim` semantics.** Expressed via `shape_rules` (range validity, normalize-then-check, uniqueness), reusing the existing vocabulary rather than a dedicated manifest field. Authoring boilerplate and per-op empty-sequence semantics: see [domain-rules/manifest-spec.md](../../.claude/domain-rules/manifest-spec.md).

**R15. Status gating.** `status: spec-only` → L0 only. `status: implemented` → all levels. `--check-op <name>` forces L0-L4 on a targeted entry (includes its variants).

**R16. Roofline metadata.** See [roofline.md](roofline.md). That document is the source of truth for roofline modes, variable binding, formula syntax, consumers, and codegen behavior.

**R17. PyTorch API alignment.** Op signatures match PyTorch's public API (names, parameter set, semantics). Do not invent parameters.

**R18. No Optional[Tensor].** Fixed tensor inputs per entry. Conditional inputs split into variants via `variant_of`, which is single-level (variant → primary, no chaining). Variants share `source.kernel` and `source.op`; each has its own `signature`, `workloads`, `roofline`.

**R19. Tensor layout.** Default: contiguous row-major (no `layout` field). Non-default: add `layout` field, `shape` names reflect memory order.

**R20. `static_dims`.** For arbitrary-rank ops (no `shape` declaration), `static_dims` declares values the user commits to at Op construction time. Each entry maps an `__init__` keyword name to a single-axis shape expression `<tensor>.shape[<const_or_param>]`. See [`static_dims`](#static_dims) for full semantics, rules, and examples.

## `static_dims`

`static_dims` declares what becomes statically known at the moment the user constructs the Op instance. It is **per-op**, not per-family.

```yaml
static_dims:
  N: "x.shape[dim]"
```

### Semantics

The shape expression is a **forward-time validation rule**, not an init-time derivation. Two time points, one contract:

- `__init__` — **commitment point**. User-supplied value stored on `self`. Expression NOT evaluated (no tensor yet).
- `forward` — **validation point**. Expression evaluated against the actual tensor; must equal the committed value.

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
- `static_dims` is only for arbitrary-rank ops. Fixed-rank ops get dimensions from `shape` (R8).

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

Three blocks in order:

1. `static_dims` — manifest key order
1. `dtype` — single parameter unless the op has explicit multi-dtype axes
1. `params` — manifest key order

All parameters are keyword-only (`*`-separated); callers always use kwargs.

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

**When `static_dims` is empty, the Op author MUST override `_cache_key`.** The default falls back to the full input shape tuple — correct but pathological under dynamic shapes (every distinct input shape recompiles). Typical full-reduce override:

```python
class SumFwdOp(Op):
    def _cache_key(self, x_shape):
        return (
            math.prod(x_shape),
        )  # all full-reductions with same numel share a kernel
```

The base class emits a once-per-type runtime warning when the default `_cache_key` is invoked with empty `static_dims` and no subclass override. See [ops-design.md § Implementing an Op](ops-design.md#implementing-an-op).

## Manifest Key Format

Each top-level entry is keyed by the **Python class name** of the Op — PascalCase with a mandatory direction suffix and `Op` suffix:

```
{PascalCaseName}{Direction}Op
```

- **PascalCaseName** — descriptive name in PascalCase (`RMSNorm`, `BatchNorm`, `Softmax`). Author chooses; no abbreviation rules.
- **Direction** — mandatory: `Fwd` or `Bwd`.
- **Op** — literal suffix.

Examples: `RMSNormFwdOp`, `BatchNormFwdOp`, `SoftmaxFwdOp`, `LinearFwdOp`.

Validator enforces `cls.__name__ == manifest_key` exactly — no heuristic resolution or case conversion.

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

Fully qualified external API name (typically PyTorch), or `"none"`. Informational — validator checks presence only.

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

| Field         | Required | Description                                                                                                                              |
| ------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `dtype`       | yes      | `\|` for alternatives, `same_as(ref)` = same dtype as ref, `promote_int_to_float(ref)` = `float32` for integral ref else `same_as(ref)`. |
| `shape`       | no       | Dimension names (e.g., `"[M, K]"`). Present = fixed rank.                                                                                |
| `constraints` | no       | Dimension restrictions (requires `shape`).                                                                                               |
| `layout`      | no       | Memory format when non-default (R19).                                                                                                    |

**Param fields:** `type` (string: `int`, `float`, `bool`, `"list[int]"`) + optional `default`.

#### Shape Decision Tree

```
Fixed rank, expressible with dimension names?
├─ YES → shape: "[D1, D2, ...]"                           [R8]
│   Relationships beyond shared names?
│   └─ YES → add shape_rules                              [R13]
└─ NO (arbitrary rank)
   ├─ write shape_rules                                   [R13]
   └─ Values committed at Op construction time?
      └─ YES → add static_dims                            [R20]
```

#### Optional Inputs

Manifest does not support `Optional[Tensor]` (R18). Split into variant entries with fixed signatures, linked by `variant_of`.

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

> **Conditional dispatch.** The flat `dispatch_key: KernelClassName` form covers static (single or multi-key) dispatch. Conditional dispatch (Hopper vs non-Hopper, op-state branches, best-of-N selection) is described under design proposal [manifest-kernel-dispatch.md](manifest-kernel-dispatch.md) and not yet implemented in the validator.

## Entry Examples

**Fixed rank — GEMM** \[R8, R11\]:

```yaml
inputs:
  a: {dtype: "float16 | bfloat16", shape: "[M, K]"}
  b: {dtype: "same_as(a)", shape: "[K, N]"}
outputs:
  c: {dtype: "same_as(a)", shape: "[M, N]"}
```

**Fixed rank + constraints — FFT** \[R8, R12\]:

```yaml
inputs:
  x: {dtype: "complex64", shape: "[M, N]", constraints: {N: "power_of_2"}}
outputs:
  y: {dtype: "same_as(x)", shape: "[M, N]"}
```

**Arbitrary rank — RMSNorm** \[R9, R13, R20\]:

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

**Arbitrary rank — Reduce** \[R9, R13\]:

```yaml
inputs:
  x: {dtype: "float16 | bfloat16"}
outputs:
  y: {dtype: "same_as(x)"}
params:
  dim: {type: "int | list[int] | tuple[int, ...] | None", default: -1}
  keepdim: {type: bool, default: false}
shape_rules:
  - "dim is None or all(-x.ndim <= d < x.ndim for d in ([dim] if isinstance(dim, int) else dim))"
  - "y.ndim == x.ndim if keepdim else x.ndim - len({dim % x.ndim} if isinstance(dim, int) else {d % x.ndim for d in dim} if isinstance(dim, (list, tuple)) and len(dim) > 0 else set(range(x.ndim)))"
  - "y.shape[i] == (1 if i in ({dim % x.ndim} if isinstance(dim, int) else {d % x.ndim for d in dim} if isinstance(dim, (list, tuple)) and len(dim) > 0 else set(range(x.ndim))) and keepdim else x.shape[i])"
  - "isinstance(dim, (int, type(None))) or len({d % x.ndim for d in dim}) == len(dim)"
```

All reduction ops include `dim` + `keepdim`. **Exception:** softmax/log_softmax preserve input shape (no `keepdim`); use `shape_rules` to express `y.shape == x.shape`. count_nonzero has no `keepdim` (per R17). Authoring contract for `dim`: see R14 → [domain-rules/manifest-spec.md](../../.claude/domain-rules/manifest-spec.md).

**Full entry — RMSNorm:**

```yaml
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

Each entry under `workloads:` is a mapping. Shape keys take the form
`<tensor_name>_shape` (e.g. `x_shape`, `q_shape`, `kv_shape`) — one per
tensor input in the workload. `dtypes` and `label` are also reserved.
Any other key becomes an **op-call parameter** forwarded to the op's
`__init__`.

| Key                   | Required | Meaning                                                                                                  |
| --------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `<tensor_name>_shape` | yes      | Shape for a tensor input (list of ints). Include one key per tensor input the workload exercises.        |
| `dtypes`              | yes      | List of dtype strings (`["float16", "bfloat16"]`).                                                       |
| `label`               | no       | Human-readable id used in the pytest param id and report tables.                                         |
| *any other key*       | no       | Op param value (`dim`, `keepdim`, `correction`, …). Overrides the manifest's `signature.params` default. |

Example — parametrizing a reduction workload over a non-last `dim`:

```yaml
workloads:
  - {x_shape: [2048, 4096], dtypes: [bfloat16], dim: -1, label: "reduce-last"}
  - {x_shape: [2048, 4096], dtypes: [bfloat16], dim:  0, label: "reduce-first"}
```

## Manifest Validation

[`scripts/validate_manifest.py`](../../scripts/validate_manifest.py) runs five levels:

| Level | Check     | Description                                                                                                                 |
| ----- | --------- | --------------------------------------------------------------------------------------------------------------------------- |
| L0    | Schema    | Required fields exist, correct types                                                                                        |
| L1    | Signature | Params ⊆ `__init__()` ∪ `forward()` names; `forward()` order matches                                                        |
| L2    | Shape     | `shape_rules` are valid Python expressions                                                                                  |
| L3    | Dtype     | dtype strings are valid torch types, `same_as()` refs, or `promote_int_to_float()` refs                                     |
| L4    | Benchmark | Bench file imports/calls `load_workloads` and `eval_roofline` (directly or via `workloads_to_params` / `ManifestBenchmark`) |

`spec-only` ops → L0 only. `implemented` ops → all levels. `--check-op <name>` forces L0-L4 on a targeted entry + its variants.

```bash
python scripts/validate_manifest.py
python scripts/validate_manifest.py --check-op SoftmaxFwdOp
```

## Exclusions

The manifest does NOT describe: multi-kernel execution ordering, accumulator dtypes, persistent state, tile sizes, or autotuning config.
