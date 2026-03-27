# Op Manifest Specification

`ops_manifest.yaml` is the central op registry and the agent entry point. Every operator has a declarative spec here before code is written. The spec drives code generation, test validation, performance evaluation, and documentation — but does not affect runtime. The user-facing API remains plain Python classes.

```
ops_manifest.yaml (spec)
       │
       ├──→ Agent reads spec, generates minimal Python code
       ├──→ Test generator reads workloads + tolerance
       ├──→ Benchmark reads workloads
       ├──→ Roofline reads flops/bytes formulas
       └──→ Docs generator reads signatures
```

## Fields

Each manifest entry lives under the top-level `ops:` key. Structure:

| Field       | Required | Description                                                                                                   |
| ----------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `family`    | yes      | Op family for grouping (e.g., `norm`, `attention`, `gemm`). Machine-readable — not derived from source paths. |
| `signature` | yes      | Op interface: inputs, outputs, params, shape_rules. See [Signature](#signature).                              |
| `workloads` | yes      | Representative shapes/dtypes for benchmarks. See [Workloads](#workloads).                                     |
| `roofline`  | yes      | FLOPs and bytes formulas for performance evaluation. See [Roofline](#roofline).                               |
| `source`    | yes      | Pointers to implementation files. See [Source](#source).                                                      |

### Signature

Declares the op's interface. Contains the following fields:

| Field          | Type | Required | Description                                                                                                                      |
| -------------- | ---- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `inputs`       | list | yes      | Input tensors, in positional order.                                                                                              |
| `outputs`      | list | yes      | Output tensors, in positional order.                                                                                             |
| `params`       | list | no       | Scalar / config parameters (e.g., `dim`, `eps`).                                                                                 |
| `shape_rules`  | list | no       | Inter-tensor shape relationships (list of Python expression strings).                                                            |
| `dtype_combos` | list | no       | Explicit list of valid dtype combinations across tensors. When present, overrides per-tensor `dtype` for combination validation. |

Each tensor entry (`inputs` / `outputs`) has:

| Field         | Type   | Required | Description                                                                                            |
| ------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------ |
| `name`        | string | yes      | Tensor identifier, used in `shape_rules` and `same_as(ref)` references.                                |
| `dtype`       | string | yes      | Data type. `\|` for alternatives, `same_as(ref)` for dependent types.                                  |
| `shape`       | string | no       | Dimension names (e.g., `"[M, K]"`) or `"same_as(ref)"`. Present = fixed rank; absent = arbitrary rank. |
| `constraints` | map    | no       | Dimension restrictions. Only valid when `shape` is present.                                            |

Each param entry has:

| Field     | Type   | Required | Description                                                                      |
| --------- | ------ | -------- | -------------------------------------------------------------------------------- |
| `name`    | string | yes      | Parameter identifier.                                                            |
| `type`    | string | yes      | Python type (e.g., `int`, `float`, `bool`, `"list[int]"`, `"int \| list[int]"`). |
| `default` | any    | no       | Default value. Omit if the parameter is required.                                |

#### Rules

**Structure**

1. **Signature uses list, not dict.** Function signatures have positional semantics — list preserves parameter order without relying on YAML mapping order (which the spec does not guarantee).
1. **Params declare the full interface.** If an op mathematically supports a parameter (e.g., `dim` for norm), it belongs in the manifest even if the current kernel only supports the default value.

**dtype**

3. **`dtype`** uses `|` for alternatives, `same_as(ref)` for dependent types. Concrete entries may list dtypes explicitly.
1. **`dtype_combos`** — when not all dtype combinations across tensors are valid, enumerate the legal combinations explicitly. Each entry is a map of tensor name to dtype. When present, `dtype_combos` is the source of truth; per-tensor `dtype` fields are still written for documentation but are not used for combination validation.

```yaml
# Only 3 of the 4 possible combinations are supported
dtype_combos:
  - {x: float16, weight: float16}
  - {x: float16, weight: float8_e4m3}
  - {x: bfloat16, weight: bfloat16}
```

**Shape**

4. **Every output tensor must have an explicit shape declaration.** There is no implicit default. Use one of the mechanisms below.
1. **`shape` present = fixed rank.** The value is a list of dimension names (e.g., `"[M, K]"`), declaring the exact number of dimensions. Each name becomes a variable available in `roofline` expressions and `constraints`. There is no intermediate form — do not use ellipsis or wildcards to partially constrain rank.
1. **`shape` absent = arbitrary rank.** The tensor accepts any number of dimensions. Constraints on specific axes are expressed through `params` (e.g., `dim`) and `shape_rules`.
1. **`same_as(ref)`** — shorthand for "identical shape (rank and sizes) to the referenced tensor". Valid for both fixed-rank and arbitrary-rank tensors. When the referenced tensor has no `shape` (arbitrary rank), `same_as(ref)` means the output inherits whatever rank and sizes the reference has at runtime.
1. **Shared dimension names imply equality.** If two tensors both use `K` in their `shape`, the sizes must match. This is the primary mechanism for expressing inter-tensor shape relationships in fixed-rank ops.
1. **`constraints`** — an optional map on a tensor with `shape`, restricting specific dimensions beyond "any positive integer". Two forms:
   - Enumerated values: `"64 | 128 | 256"`
   - Predicate: `"power_of_2"`, `"divisible_by(k)"`, `"even"`, `"positive"`
1. **`shape_rules`** — a list of Python expression strings for inter-tensor shape relationships that `shape` fields and `same_as` cannot express (e.g., `"weight.shape == (x.shape[dim],)"`). When `shape_rules` is absent or insufficient, tools fall back to calling `Op.infer_shape()` via `source.op`. The `Op` base class defines this interface; all subclasses must implement it.

#### Shape decision tree

When declaring shape for an output tensor, follow this flow:

```
Is the output shape identical to an input?
├─ YES → shape: "same_as(ref)"                              (Rule 7)
└─ NO
   Does the output have a fixed rank expressible with dimension names?
   ├─ YES → shape: "[D1, D2, ...]"                          (Rule 5)
   │   Need inter-tensor relationships beyond shared names?
   │   └─ YES → add shape_rules                             (Rule 10)
   └─ NO (arbitrary rank, output shape depends on params)
      Can the relationship be expressed as inline expressions?
      ├─ YES → add shape_rules                              (Rule 10)
      └─ NO  → omit shape_rules; tools call Op.infer_shape()
```

#### Examples

**Fixed rank — GEMM** (Rules 1, 5, 8):

```yaml
inputs:
  - name: a
    dtype: "float16 | bfloat16"
    shape: "[M, K]"
  - name: b
    dtype: "same_as(a)"
    shape: "[K, N]"
outputs:
  - name: c
    dtype: "same_as(a)"
    shape: "[M, N]"
```

`K` appears in both `a` and `b` — sizes must match. Output dimensions `M`, `N` are derived from inputs.

**Fixed rank with constraints — FFT** (Rules 5, 7, 9):

```yaml
inputs:
  - name: x
    dtype: "complex64"
    shape: "[M, N]"
    constraints:
      N: "power_of_2"
outputs:
  - name: y
    dtype: "same_as(x)"
    shape: "same_as(x)"
```

**Arbitrary rank, output same as input — RMSNorm** (Rules 6, 7, 10):

```yaml
inputs:
  - name: x
    dtype: "float16 | bfloat16"
  - name: weight
    dtype: "same_as(x)"
outputs:
  - name: y
    dtype: "same_as(x)"
    shape: "same_as(x)"
params:
  - name: dim
    type: int
    default: -1
  - name: eps
    type: float
    default: 1e-6
shape_rules:
  - "weight.shape == (x.shape[dim],)"
```

No `shape` on `x` — any rank accepted. `dim` selects the normalization axis. Output shape mirrors input via `same_as(x)`. The only non-trivial relationship (`weight` is 1-D matching the target axis) goes in `shape_rules`.

**Arbitrary rank, output shape depends on params — Reduce** (Rule 6):

```yaml
inputs:
  - name: x
    dtype: "float16 | bfloat16"
outputs:
  - name: y
    dtype: "same_as(x)"
params:
  - name: dim
    type: "int | list[int]"
  - name: keepdim
    type: bool
    default: false
```

Output rank depends on `dim` and `keepdim` — cannot be expressed as inline expressions. No `shape_rules`; tools call `SumOp.infer_shape()` via `source.op`.

**Arbitrary rank, shape transformation — Transpose** (Rule 6):

```yaml
inputs:
  - name: x
    dtype: "float16 | bfloat16"
outputs:
  - name: y
    dtype: "same_as(x)"
params:
  - name: dims
    type: "list[int]"
```

Output has the same rank as input but permuted dimensions — no inline expression can capture this. Tools call `TransposeOp.infer_shape()` via `source.op`.

**Full entry — RMSNorm** (all sections):

```yaml
ops:
  rmsnorm_fwd:
    family: norm

    signature:
      inputs:
        - name: x
          dtype: "float16 | bfloat16"
        - name: weight
          dtype: "same_as(x)"
      outputs:
        - name: y
          dtype: "same_as(x)"
          shape: "same_as(x)"
      params:
        - name: dim
          type: int
          default: -1
        - name: eps
          type: float
          default: 1e-6
      shape_rules:
        - "weight.shape == (x.shape[dim],)"

    workloads:
      - x_shape: [2048, 4096]
        dtypes: [float16, bfloat16]
        label: "llama-3.1-8b-prefill"
      - x_shape: [1, 4096]
        dtypes: [bfloat16]
        label: "llama-3.1-8b-decode"

    roofline:
      flops: "4 * M * N"
      bytes: "2 * (M * N + N + M * N)"

    source:
      kernel: tileops/kernels/norm/rms_norm.py
      op: tileops/ops/norm/rms_norm.py
      test: tests/ops/test_rms_norm.py
      bench: benchmarks/ops/bench_rms_norm.py
```

### Workloads

Representative shape/dtype combinations for benchmarking:

| Field     | Required | Description                                                                     |
| --------- | -------- | ------------------------------------------------------------------------------- |
| `x_shape` | yes      | Input shape for the workload. Drives benchmark execution and code generation.   |
| `dtypes`  | yes      | List of dtypes to test.                                                         |
| `label`   | no       | Human-readable tag for reports. Auto-generated from shape + dtype when omitted. |

Op-specific parameters (e.g., `dim` for norm, `causal` for attention) can be added per workload entry. Shapes are chosen by the op developer based on target model architectures.

### Roofline

Two modes — inline expression for simple ops, Python function reference for complex ops:

| Mode     | Format                           | When to use                                                                    |
| -------- | -------------------------------- | ------------------------------------------------------------------------------ |
| Inline   | `flops: "expr"`, `bytes: "expr"` | FLOPs and bytes can be expressed as simple arithmetic on dimension names.      |
| Function | `func: "module.path"`            | Complex ops where formulas depend on multiple parameters or conditional logic. |

Referenced functions live in `tileops/perf/formulas.py` and return `{"flops": int, "bytes": int}`.

The field is `bytes` (total bytes moved), not `memory` — maps directly to `bytes_moved` in the roofline formula `memory_time = bytes_moved / hbm_bandwidth`.

### Source

Pointers to implementation files for navigation and CI validation:

| Field    | Required | Description                 |
| -------- | -------- | --------------------------- |
| `kernel` | yes      | Kernel implementation path. |
| `op`     | yes      | Op class path.              |
| `test`   | yes      | Test file path.             |
| `bench`  | yes      | Benchmark file path.        |

## What Is NOT in the Manifest

- **Reference implementations** — stay in test files as PyTorch code.
- **Kernel implementation details** — tile sizes, memory strategies, `num_per_thread` are implementation choices.
- **Autotuning configuration** — handled by the kernel layer.
