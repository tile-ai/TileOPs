# Op Manifest Specification

`ops_manifest.yaml` is the **source of truth** for op interfaces. Code, tests, benchmarks, and docs are generated from it.

Design rationale: `TileOpsGov/docs/manifest-design-rationale.md`

## Trust Model

1. The manifest is the sole source of truth for op interfaces. Changes require human review.
1. Programmatic validation is derived from the manifest, not from the generating agent.
1. Test coverage (shapes, dtypes, workloads) is determined by the manifest, not by the agent.
1. `Op.forward()` signature must match the manifest. CI enforces this.
1. Benchmarks must use declared workloads. No hardcoded shapes.

## Entry Structure

Each entry lives under `ops:`:

| Field       | Required | Description                                           |
| ----------- | -------- | ----------------------------------------------------- |
| `family`    | yes      | Op family (e.g., `norm`, `attention`).                |
| `signature` | yes      | Op interface. See [Signature](#signature).            |
| `workloads` | yes      | Benchmark shapes/dtypes. See [Workloads](#workloads). |
| `roofline`  | yes      | Performance model. See [Roofline](#roofline).         |
| `source`    | yes      | Implementation paths. See [Source](#source).          |

## Signature

```yaml
signature:
  inputs:       # dict — tensor name → {dtype, shape?, constraints?}
  outputs:      # dict — tensor name → {dtype, shape?, constraints?}
  params:       # list — [{name, type, default?}]
  shape_rules:  # list — Python expressions for shape inference
  dtype_combos: # list — valid cross-tensor dtype combinations
```

**Tensor fields** (`inputs`/`outputs`): key = tensor name, value = dict with:

| Field         | Required | Description                                                |
| ------------- | -------- | ---------------------------------------------------------- |
| `dtype`       | yes      | `\|` for alternatives, `same_as(ref)` for dependent types. |
| `shape`       | no       | Dimension names or `same_as(ref)`. Present = fixed rank.   |
| `constraints` | no       | Dimension restrictions (requires `shape`).                 |

**Param fields**: `name` (string), `type` (string: `int`, `float`, `bool`, `"list[int]"`), `default` (optional).

### Rules

- **R1.** `inputs`/`outputs` are dicts keyed by tensor name. `params` is a list.
- **R2.** Params include all supported parameters, even if the kernel only supports the default.
- **R3.** dtype syntax: `|` for alternatives, `same_as(ref)` for dependent types.
- **R4.** `dtype_combos` enumerates valid cross-tensor combinations. Source of truth when present.
- **R5.** Every output shape must be fully specified via `shape`, `same_as(ref)`, and/or `shape_rules`.
- **R6.** `shape` present = fixed rank. Dimension names become variables in `roofline`/`constraints`.
- **R7.** `shape` absent = arbitrary rank. Use `params` + `shape_rules` for axis constraints.
- **R8.** `same_as(ref)` = identical shape to referenced tensor.
- **R9.** Shared dimension names = equality constraint (`K` in two tensors means sizes match).
- **R10.** `constraints`: `"64 | 128 | 256"` (enumerated), `"power_of_2"`, `"divisible_by(k)"`.
- **R11.** `shape_rules`: Python expressions for shape relationships. Required when `shape` + `same_as` are insufficient.

### Examples

**Fixed rank — GEMM:**

```yaml
inputs:
  a: {dtype: "float16 | bfloat16", shape: "[M, K]"}
  b: {dtype: "same_as(a)", shape: "[K, N]"}
outputs:
  c: {dtype: "same_as(a)", shape: "[M, N]"}
```

**Arbitrary rank — RMSNorm:**

```yaml
inputs:
  x: {dtype: "float16 | bfloat16"}
  weight: {dtype: "same_as(x)"}
outputs:
  y: {dtype: "same_as(x)", shape: "same_as(x)"}
params:
  - {name: dim, type: int, default: -1}
  - {name: eps, type: float, default: 1e-6}
shape_rules:
  - "weight.shape == (x.shape[dim],)"
```

## Workloads

Each entry is a dict. Shape keys use `<tensor_name>_shape` convention.

| Field    | Required | Description             |
| -------- | -------- | ----------------------- |
| `dtypes` | yes      | List of dtypes to test. |
| `label`  | no       | Human-readable tag.     |

```yaml
# Single primary input:
- {x_shape: [2048, 4096], dtypes: [float16, bfloat16], label: "llama-3.1-8b"}
# Multiple inputs:
- {q_shape: [1, 1, 32, 128], kv_shape: [1, 8192, 8, 128], dtypes: [float16], label: "gqa-decode"}
```

Op-specific parameters (e.g., `groups`, `is_causal`) can be added per entry.

## Roofline

| Mode   | Format                           | When                                  |
| ------ | -------------------------------- | ------------------------------------- |
| Inline | `flops: "expr"`, `bytes: "expr"` | Arithmetic on dimension variables.    |
| Func   | `func: "module.path"`            | Complex formulas needing Python code. |

Inline expressions are evaluated by `_safe_eval()` — only numeric constants, arithmetic (`+−*/÷**%`), and `log2`/`ceil`/`floor`. Func-mode functions live in `tileops/perf/formulas.py`, accept kwargs, return `{"flops": int, "bytes": int}`.

Roofline variable bindings are the consumer's responsibility, not the manifest's.

## Source

| Field    | Required | Type           | Description          |
| -------- | -------- | -------------- | -------------------- |
| `kernel` | yes      | string or list | Kernel file path(s). |
| `op`     | yes      | string         | Op class file path.  |
| `test`   | yes      | string         | Test file path.      |
| `bench`  | yes      | string         | Benchmark file path. |

## Tier Model

Ops must have **all-required tensors** and **fixed output count** to enter the manifest. Ops with `Optional[Tensor]` or conditional outputs must be split first.

| Tier | Roofline                            | Families                                      |
| ---- | ----------------------------------- | --------------------------------------------- |
| 1    | Inline arithmetic expression        | norm, reduction, elementwise, RoPE, FFT, GEMM |
| 2    | `func` reference to Python function | attention, conv, MoE, varlen attention        |

## Exclusions

The manifest does NOT describe: kernel dispatch logic, multi-kernel pipelines, accumulator dtypes, tensor layout, non-tensor persistent state (`__init__` LUTs/caches), tile sizes, or autotuning config.
