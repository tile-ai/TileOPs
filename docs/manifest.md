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

**R3. `dtype` syntax.** `|` for alternatives, `same_as(ref)` to indicate the dtype is the same as `ref`.

**R3a. `same_as(ref)` identity constraint.** `same_as(ref)` is dtype-only: the tensor must have the exact same dtype as `ref` at runtime. `same_as`-bound tensors do not contribute independent axes to the Cartesian product in R4. Do not use `same_as` for shape.

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

**R5. Explicit shape.** Every output tensor's shape must be fully specified via `shape` and/or `shape_rules`. Input tensors may omit `shape` (→ arbitrary rank per R7). `same_as` is dtype-only — do not use it for shape.

**R6. `shape` = fixed rank.** Declares exact dimensions (e.g., `"[M, K]"`). Names become roofline variables. No ellipsis or wildcards.

**R7. No `shape` = arbitrary rank.** Constraints go in `params` + `shape_rules`.

**R8. No shape aliasing.** Each tensor declares its own shape. Use shared dimension names (R9) or `shape_rules` (R11) to express shape relationships.

**R9. Shared dimension names = equality.** `K` in two tensors means their sizes must match.

**R10. `constraints`.** Restricts dimensions: `"64 | 128 | 256"` (enumerated) or `"power_of_2"`, `"divisible_by(k)"`, `"even"`, `"positive"` (predicates). Requires `shape`.

**R11. `shape_rules`.** Python expressions for shape relationships. Required when `shape` alone cannot fully specify output shape.

**R12. Shape derivation.** `shape` + `shape_rules` fully specify output shape derivation. Manifest and implementation must be consistent.

**R13. Status gating.** `status: spec-only` → L0 only. `status: implemented` → all levels. `--check-op <name>` forces L0-L4 on a targeted entry (includes its variants).

**R14. Roofline variable binding.** See [Roofline](#roofline).

**R15. PyTorch API alignment.** Op signatures match PyTorch's public API (names, parameter set, semantics). Do not invent parameters.

**R16. No Optional[Tensor].** Fixed tensor inputs per entry. Conditional inputs → split into variants via `variant_of`.

> **R17.** `variant_of` is one level only. Variant → primary. No chaining.
>
> **R18.** Variants share `source.kernel` and `source.op`. Each has its own `signature`, `workloads`, `roofline`.

**R19. Tensor layout.** Default: contiguous row-major (no `layout` field). Non-default: add `layout` field, `shape` names reflect memory order.

```yaml
x: {dtype: "float16", shape: "[N, H, W, C]", layout: "channels_last"}
```

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
   └─ write shape_rules                                   [R11]
```

#### Optional Inputs

Manifest does not support `Optional[Tensor]` (R16). Split into variant entries with fixed signatures, linked by `variant_of` (R17-R18).

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

| Mode          | Format                   | When                         |
| ------------- | ------------------------ | ---------------------------- |
| Inline        | `flops`/`bytes` only     | Fixed-rank ops with `shape`. |
| Inline + vars | `vars` + `flops`/`bytes` | Arbitrary-rank ops.          |
| Func          | `func: "module.path"`    | Complex formulas.            |

**Variable binding (R14):**

- `elem_bytes` — byte size of first input's dtype (built-in).
- `shape` dimension names auto-bind as roofline variables.
- `vars` — explicit mappings for arbitrary-rank ops. Evaluation context: tensor shapes, params, `product()`, arithmetic, `range()`, comprehensions.

```yaml
# Fixed rank — shape names auto-bind
roofline:
  flops: "2 * M * N * K"
  bytes: "(M * K + K * N + M * N) * elem_bytes"

# Arbitrary rank — explicit vars
roofline:
  vars:
    M: "product(x.shape[:dim])"
    N: "x.shape[dim]"
  flops: "4 * M * N"
  bytes: "(2 * M * N + N) * elem_bytes"
```

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

Op→Kernel dispatch registration table. Declares which Kernels an Op uses so agents know what to implement. Does not describe dispatch strategy (runtime concern). Format: `dispatch_key: KernelClassName`. See [ops-design.md § Kernel Dispatch](ops-design.md#kernel-dispatch-kernel_map).

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
    - tileops/kernels/flash_attn/bwd.py
  kernel_map:
    mha_bwd_preprocess_kernel: FlashAttnBwdPreprocessKernel
    mha_bwd_kernel: MhaBwdKernel
    mha_bwd_postprocess_kernel: FlashAttnBwdPostprocessKernel
  op: tileops/ops/mha.py
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

**Arbitrary rank — RMSNorm** \[R7, R11\]:

```yaml
inputs:
  x: {dtype: "float16 | bfloat16"}
  weight: {dtype: "same_as(x)"}
outputs:
  y: {dtype: "same_as(x)"}
params:
  dim: {type: int, default: -1}
  eps: {type: float, default: 1e-6}
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
  dim: {type: "int | list[int]"}
  keepdim: {type: bool, default: false}
shape_rules:
  - "y.ndim == x.ndim if keepdim else x.ndim - len([dim] if isinstance(dim, int) else dim)"
  - "y.shape[i] == (1 if i in ([dim] if isinstance(dim, int) else dim) and keepdim else x.shape[i])"
```

All reduction ops include `dim` + `keepdim`. **Exception:** softmax/log_softmax preserve input shape (no `keepdim`); use `shape_rules` to express `y.shape == x.shape`. count_nonzero has no `keepdim` (per R15).

**Full entry — RMSNorm:**

```yaml
ops:
  RMSNormFwdOp:
    family: norm
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

Benchmarks must use manifest-driven workloads:

```python
from tileops.manifest import eval_roofline, load_workloads

_OP_NAME = "RMSNormFwdOp"


def _manifest_params():
    params = []
    for w in load_workloads(_OP_NAME):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype, True, id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype, tune", _manifest_params())
def test_rms_norm_bench(m, n, dtype, tune): ...


# Roofline
flops, mem_bytes = eval_roofline(_OP_NAME, M=m, N=n, elem_bytes=elem_bytes)
```

## Manifest Validation

[`scripts/validate_manifest.py`](../scripts/validate_manifest.py) runs five levels:

| Level | Check     | Description                                                          |
| ----- | --------- | -------------------------------------------------------------------- |
| L0    | Schema    | Required fields exist, correct types                                 |
| L1    | Signature | Params ⊆ `__init__()` ∪ `forward()` names; `forward()` order matches |
| L2    | Shape     | `shape_rules` are valid Python expressions                           |
| L3    | Dtype     | dtype strings are valid torch types or `same_as()` refs              |
| L4    | Benchmark | Bench file imports `load_workloads` / `eval_roofline`                |

`spec-only` ops → L0 only. `implemented` ops → all levels. `--check-op <name>` forces L0-L4 on a targeted entry + its variants.

```bash
python scripts/validate_manifest.py
python scripts/validate_manifest.py --check-op SoftmaxFwdOp
```

## Exclusions

The manifest does NOT describe: multi-kernel execution ordering, accumulator dtypes, persistent state, tile sizes, or autotuning config.
