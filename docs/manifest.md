# Op Manifest Specification

`ops_manifest.yaml` is the central op registry and the agent entry point. The manifest is the **source of truth** — runtime code (including `Op.infer_shape()`) is generated from it, not the other way around.

```
ops_manifest.yaml (spec)
       │
       ├──→ Agent generates Op code + Op.infer_shape() from signature + shape_rules
       ├──→ Test generator reads workloads + tolerance
       ├──→ Benchmark reads workloads
       ├──→ Roofline reads flops/bytes formulas
       └──→ Docs generator reads signatures
```

## Fields

Each entry lives under the top-level `ops:` key:

| Field       | Required | Description                                                                      |
| ----------- | -------- | -------------------------------------------------------------------------------- |
| `family`    | yes      | Op family for grouping (e.g., `norm`, `attention`). Not derived from file paths. |
| `signature` | yes      | Op interface. See [Signature](#signature).                                       |
| `workloads` | yes      | Benchmark shapes/dtypes. See [Workloads](#workloads).                            |
| `roofline`  | yes      | Performance model. See [Roofline](#roofline).                                    |
| `source`    | yes      | Implementation file paths. See [Source](#source).                                |

### Signature

| Field          | Type | Required | Description                                                                            |
| -------------- | ---- | -------- | -------------------------------------------------------------------------------------- |
| `inputs`       | list | yes      | Input tensors, positional order.                                                       |
| `outputs`      | list | yes      | Output tensors, positional order.                                                      |
| `params`       | list | no       | Scalar / config parameters.                                                            |
| `shape_rules`  | list | no       | Python expressions for shape inference. Agent generates `Op.infer_shape()` from these. |
| `dtype_combos` | list | no       | Valid dtype combinations. Overrides per-tensor `dtype` when present.                   |

**Tensor fields** (`inputs` / `outputs`):

| Field         | Type   | Required | Description                                                                       |
| ------------- | ------ | -------- | --------------------------------------------------------------------------------- |
| `name`        | string | yes      | Identifier. Referenced in `shape_rules` and `same_as(ref)`.                       |
| `dtype`       | string | yes      | `\|` for alternatives, `same_as(ref)` for dependent types.                        |
| `shape`       | string | no       | Dimension names or `same_as(ref)`. Present = fixed rank, absent = arbitrary rank. |
| `constraints` | map    | no       | Dimension restrictions. Requires `shape`.                                         |

**Param fields**:

| Field     | Type   | Required | Description                                          |
| --------- | ------ | -------- | ---------------------------------------------------- |
| `name`    | string | yes      | Identifier.                                          |
| `type`    | string | yes      | Python type (`int`, `float`, `bool`, `"list[int]"`). |
| `default` | any    | no       | Default value. Omit if required.                     |

#### Rules

**R1. List, not dict.** Signature fields use lists — preserves positional order (YAML mapping order is unspecified).

**R2. Full interface.** Params include all mathematically supported parameters, even if the current kernel only supports the default.

**R3. `dtype` syntax.** `|` for alternatives, `same_as(ref)` for dependent types.

**R4. `dtype_combos`.** Enumerates valid cross-tensor dtype combinations. Source of truth when present; per-tensor `dtype` remains for documentation.

```yaml
dtype_combos:
  - {x: float16, weight: float16}
  - {x: float16, weight: float8_e4m3}
  - {x: bfloat16, weight: bfloat16}
```

**R5. Output shape completeness.** Every output's shape must be fully specified via `shape`, `same_as(ref)`, and/or `shape_rules`. No fallback — the manifest must be sufficient for generating `Op.infer_shape()`.

**R6. `shape` present = fixed rank.** Declares exact dimensions (e.g., `"[M, K]"`). Names become variables in `roofline` and `constraints`. No ellipsis or wildcards.

**R7. `shape` absent = arbitrary rank.** Any number of dimensions. Axis constraints go in `params` + `shape_rules`.

**R8. `same_as(ref)`.** Output has identical shape to the referenced tensor. Works for both fixed and arbitrary rank.

**R9. Shared dimension names = equality.** `K` in two tensors means their sizes must match.

**R10. `constraints`.** Restricts dimensions on tensors with `shape`. Values: `"64 | 128 | 256"` (enumerated) or `"power_of_2"`, `"divisible_by(k)"`, `"even"`, `"positive"` (predicates).

**R11. `shape_rules`.** Python expressions describing shape relationships. Agent generates `Op.infer_shape()` from these. Required when `shape` + `same_as` cannot fully specify output shape.

**R12. Manifest → `infer_shape()`.** Agent generates `Op.infer_shape()` from `shape`, `same_as(ref)`, and `shape_rules`. Manifest and code must be consistent.

#### Shape decision tree

Agent flow: (1) declare output shape in manifest, (2) generate `Op.infer_shape()` from declaration.

**Step 1 — Declare:**

```
Output shape identical to an input?
├─ YES → shape: "same_as(ref)"                            [R8]
└─ NO
   Fixed rank, expressible with dimension names?
   ├─ YES → shape: "[D1, D2, ...]"                        [R6]
   │   Inter-tensor relationships beyond shared names?
   │   └─ YES → add shape_rules                           [R11]
   └─ NO (arbitrary rank, depends on params)
      └─ write shape_rules                                [R11]
```

Every leaf is a complete spec — no "omit and fallback" path.

**Step 2 — Generate `Op.infer_shape()`:**

| Declaration                             | Generated logic                                     |
| --------------------------------------- | --------------------------------------------------- |
| `shape: "same_as(ref)"`                 | `return ref.shape`                                  |
| `shape: "[D1, D2, ...]"` + shared names | Return shape with matched dimensions from inputs    |
| `shape_rules`                           | Translate expressions into Python shape computation |

#### Examples

**Fixed rank — GEMM** \[R1, R6, R9\]:

```yaml
# Shared K implies a.shape[1] == b.shape[0]
inputs:
  - {name: a, dtype: "float16 | bfloat16", shape: "[M, K]"}
  - {name: b, dtype: "same_as(a)", shape: "[K, N]"}
outputs:
  - {name: c, dtype: "same_as(a)", shape: "[M, N]"}
```

**Fixed rank + constraints — FFT** \[R6, R8, R10\]:

```yaml
inputs:
  - {name: x, dtype: "complex64", shape: "[M, N]", constraints: {N: "power_of_2"}}
outputs:
  - {name: y, dtype: "same_as(x)", shape: "same_as(x)"}
```

**Arbitrary rank + same_as — RMSNorm** \[R7, R8, R11\]:

```yaml
# No shape on x → any rank. dim selects axis. weight is 1-D along that axis.
inputs:
  - {name: x, dtype: "float16 | bfloat16"}
  - {name: weight, dtype: "same_as(x)"}
outputs:
  - {name: y, dtype: "same_as(x)", shape: "same_as(x)"}
params:
  - {name: dim, type: int, default: -1}
  - {name: eps, type: float, default: 1e-6}
shape_rules:
  - "weight.shape == (x.shape[dim],)"
```

**Arbitrary rank + shape_rules — Reduce** \[R7, R11\]:

```yaml
# Output rank depends on dim and keepdim — shape_rules fully describe the logic.
inputs:
  - {name: x, dtype: "float16 | bfloat16"}
outputs:
  - {name: y, dtype: "same_as(x)"}
params:
  - {name: dim, type: "int | list[int]"}
  - {name: keepdim, type: bool, default: false}
shape_rules:
  - "y.ndim == x.ndim if keepdim else x.ndim - len(dim)"
  - "y.shape[i] == 1 if i in dim and keepdim else x.shape[i]"
```

**Arbitrary rank + shape_rules — Transpose** \[R7, R11\]:

```yaml
# Same rank, permuted dimensions.
inputs:
  - {name: x, dtype: "float16 | bfloat16"}
outputs:
  - {name: y, dtype: "same_as(x)"}
params:
  - {name: dims, type: "list[int]"}
shape_rules:
  - "y.ndim == x.ndim"
  - "y.shape[i] == x.shape[dims[i]]"
```

**Full entry — RMSNorm:**

```yaml
ops:
  rmsnorm_fwd:
    family: norm

    signature:
      inputs:
        - {name: x, dtype: "float16 | bfloat16"}
        - {name: weight, dtype: "same_as(x)"}
      outputs:
        - {name: y, dtype: "same_as(x)", shape: "same_as(x)"}
      params:
        - {name: dim, type: int, default: -1}
        - {name: eps, type: float, default: 1e-6}
      shape_rules:
        - "weight.shape == (x.shape[dim],)"

    workloads:
      - {x_shape: [2048, 4096], dtypes: [float16, bfloat16], label: "llama-3.1-8b-prefill"}
      - {x_shape: [1, 4096], dtypes: [bfloat16], label: "llama-3.1-8b-decode"}

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

| Field     | Required | Description                                                       |
| --------- | -------- | ----------------------------------------------------------------- |
| `x_shape` | yes      | Input shape. Drives benchmark execution and code generation.      |
| `dtypes`  | yes      | List of dtypes to test.                                           |
| `label`   | no       | Human-readable tag. Auto-generated from shape + dtype if omitted. |

Op-specific parameters (e.g., `causal` for attention) can be added per entry. Shapes target real model architectures.

### Roofline

| Mode     | Format                           | When to use                                                |
| -------- | -------------------------------- | ---------------------------------------------------------- |
| Inline   | `flops: "expr"`, `bytes: "expr"` | Simple arithmetic on dimension names.                      |
| Function | `func: "module.path"`            | Complex formulas with multiple parameters or conditionals. |

Functions live in `tileops/perf/formulas.py`, return `{"flops": int, "bytes": int}`. The field is `bytes` (total bytes moved), not `memory`.

### Source

| Field    | Required | Description                 |
| -------- | -------- | --------------------------- |
| `kernel` | yes      | Kernel implementation path. |
| `op`     | yes      | Op class path.              |
| `test`   | yes      | Test file path.             |
| `bench`  | yes      | Benchmark file path.        |

## What Is NOT in the Manifest

- **Reference implementations** — stay in test files as PyTorch code.
- **Kernel implementation details** — tile sizes, memory strategies, `num_per_thread`.
- **Autotuning configuration** — handled by the kernel layer.
