# Op Manifest Specification

`ops_manifest.yaml` is the **source of truth** for op interfaces, benchmark workloads, and benchmark roofline metadata.

## Trust Model

1. The manifest is the sole source of truth for op interfaces. Changes require human review.
1. Programmatic validation is derived from the manifest, not from the generating agent.
1. `workloads` define benchmark shapes and dtypes for nightly/performance coverage, not unit-test coverage.
1. `Op.forward()` signature must match the manifest. CI enforces this.
1. Benchmarks must use declared workloads. No hardcoded shapes.

## Entry Structure

Each entry lives under `ops:`:

| Field       | Required | Description                                                                        |
| ----------- | -------- | ---------------------------------------------------------------------------------- |
| `family`    | yes      | Op family (e.g., `norm`, `attention`).                                             |
| `signature` | yes      | Op interface. See [Signature](#signature).                                         |
| `workloads` | yes      | Benchmark shapes/dtypes for nightly/performance runs. See [Workloads](#workloads). |
| `roofline`  | yes      | Performance model. See [Roofline](#roofline).                                      |
| `source`    | yes      | Implementation paths. See [Source](#source).                                       |

## Signature

```yaml
signature:
  inputs:       # dict — tensor name → {dtype, shape?, constraints?}
  outputs:      # dict — tensor name → {dtype, shape?, constraints?}
  params:       # dict — param name → {type, default?}
  shape_rules:  # list — Python expressions for shape inference
  dtype_combos: # list — valid cross-tensor dtype combinations
```

**Tensor fields** (`inputs`/`outputs`): key = tensor name, value = dict with:

| Field         | Required | Description                                                |
| ------------- | -------- | ---------------------------------------------------------- |
| `dtype`       | yes      | `\|` for alternatives, `same_as(ref)` for dependent types. |
| `shape`       | no       | Dimension names or `same_as(ref)`. Present = fixed rank.   |
| `constraints` | no       | Dimension restrictions (requires `shape`).                 |

**Param fields**: key = parameter name, value = dict with `type` (string: `int`, `float`, `bool`, `"list[int]"`) and optional `default`.

### Rules

**R1. Dict, not list.** Signature `inputs`, `outputs`, and `params` all use dicts keyed by name.

**R2. Full interface.** Params include all mathematically supported parameters, even if the current kernel only supports the default.

**R3. `dtype` syntax.** `|` for alternatives, `same_as(ref)` for dependent types.

**R4. `dtype_combos`.** Enumerates valid cross-tensor dtype combinations. Source of truth when present; per-tensor `dtype` remains for documentation.

```yaml
dtype_combos:
  - {x: float16, weight: float16}
  - {x: float16, weight: float8_e4m3}
  - {x: bfloat16, weight: bfloat16}
```

**R5. Output shape completeness.** Every output's shape must be fully specified via `shape`, `same_as(ref)`, and/or `shape_rules`. The manifest must be sufficient for generating `Op.infer_shape()`.

**R6. `shape` present = fixed rank.** Declares exact dimensions (e.g., `"[M, K]"`). Names become variables in `roofline` and `constraints`. No ellipsis or wildcards.

**R7. `shape` absent = arbitrary rank.** Any number of dimensions. Axis constraints go in `params` + `shape_rules`.

**R8. `same_as(ref)`.** Output has identical shape to the referenced tensor. Works for both fixed and arbitrary rank.

**R9. Shared dimension names = equality.** `K` in two tensors means their sizes must match.

**R10. `constraints`.** Restricts dimensions on tensors with `shape`. Values: `"64 | 128 | 256"` (enumerated) or `"power_of_2"`, `"divisible_by(k)"`, `"even"`, `"positive"` (predicates).

**R11. `shape_rules`.** Python expressions describing shape relationships. Agent generates `Op.infer_shape()` from these. Required when `shape` + `same_as` cannot fully specify output shape.

**R12. Manifest → `infer_shape()`.** Agent generates `Op.infer_shape()` from `shape`, `same_as(ref)`, and `shape_rules`. Manifest and code must be consistent.

### Shape Decision Tree

Step 1 — declare output shape in the manifest. Step 2 — generate `Op.infer_shape()` from that declaration.

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

### Examples

**Fixed rank — GEMM** \[R1, R6, R9\]:

```yaml
# Shared K implies a.shape[1] == b.shape[0]
inputs:
  a: {dtype: "float16 | bfloat16", shape: "[M, K]"}
  b: {dtype: "same_as(a)", shape: "[K, N]"}
outputs:
  c: {dtype: "same_as(a)", shape: "[M, N]"}
```

**Fixed rank + constraints — FFT** \[R6, R8, R10\]:

```yaml
inputs:
  x: {dtype: "complex64", shape: "[M, N]", constraints: {N: "power_of_2"}}
outputs:
  y: {dtype: "same_as(x)", shape: "same_as(x)"}
```

**Arbitrary rank + same_as — RMSNorm** \[R7, R8, R11\]:

```yaml
# No shape on x → any rank. dim selects axis. weight is 1-D along that axis.
inputs:
  x: {dtype: "float16 | bfloat16"}
  weight: {dtype: "same_as(x)"}
outputs:
  y: {dtype: "same_as(x)", shape: "same_as(x)"}
params:
  dim: {type: int, default: -1}
  eps: {type: float, default: 1e-6}
shape_rules:
  - "weight.shape == (x.shape[dim],)"
```

**Arbitrary rank + shape_rules — Reduce** \[R7, R11\]:

```yaml
# Output rank depends on dim and keepdim — shape_rules fully describe the logic.
inputs:
  x: {dtype: "float16 | bfloat16"}
outputs:
  y: {dtype: "same_as(x)"}
params:
  dim: {type: "int | list[int]"}
  keepdim: {type: bool, default: false}
shape_rules:
  - "y.ndim == x.ndim if keepdim else x.ndim - len(dim)"
  - "y.shape[i] == 1 if i in dim and keepdim else x.shape[i]"
```

**Full entry — RMSNorm:**

```yaml
ops:
  rmsnorm_fwd:
    family: norm

    signature:
      inputs:
        x: {dtype: "float16 | bfloat16"}
        weight: {dtype: "same_as(x)"}
      outputs:
        y: {dtype: "same_as(x)", shape: "same_as(x)"}
      params:
        dim: {type: int, default: -1}
        eps: {type: float, default: 1e-6}
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

## Workloads

Each entry is a dict. Shape keys use `<tensor_name>_shape` convention.

| Field    | Required | Description                  |
| -------- | -------- | ---------------------------- |
| `dtypes` | yes      | List of dtypes to benchmark. |
| `label`  | no       | Human-readable tag.          |

```yaml
# Single primary input:
- {x_shape: [2048, 4096], dtypes: [float16, bfloat16], label: "llama-3.1-8b"}
# Multiple inputs:
- {q_shape: [1, 1, 32, 128], kv_shape: [1, 8192, 8, 128], dtypes: [float16], label: "gqa-decode"}
```

Op-specific parameters (e.g., `groups`, `is_causal`) can be added per entry.

`workloads` are for benchmark scheduling and benchmark parametrization only. They do not prescribe unit-test cases or UT branch coverage; unit tests remain developer-owned and should target implementation-critical branches directly.

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

## Benchmark Pattern

Benchmark files must use manifest-driven workloads via `load_workloads` and `eval_roofline` from `tileops.manifest`. This ensures benchmarks always match the declared interface.

### Required imports

```python
from tileops.manifest import eval_roofline, load_workloads
```

### Workload parametrization

Use `load_workloads(op_name)` to generate pytest parameters from the manifest:

```python
_OP_NAME = "rmsnorm_fwd"


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
```

### Roofline evaluation

Use `eval_roofline(op_name, **variables)` to compute analytical flops and bytes from the manifest's roofline expressions:

```python
flops, mem_bytes = eval_roofline(_OP_NAME, M=m, N=n, elem_bytes=elem_bytes)
```

The variable names must match those used in the manifest's `roofline.flops` and `roofline.bytes` expressions.

### Hardcoded shapes are not allowed

Benchmark files must not hardcode workload shapes. The L4 check in `scripts/validate_manifest.py` enforces that every benchmark file listed in the manifest's `source.bench` imports `load_workloads`.

## Manifest Validation

`scripts/validate_manifest.py` runs five check levels against every entry in `ops_manifest.yaml`:

| Level | Check             | Description                                                                                                 |
| ----- | ----------------- | ----------------------------------------------------------------------------------------------------------- |
| L0    | YAML schema       | Required fields exist and have correct types                                                                |
| L1    | Signature         | `Op.forward()` params match manifest inputs, plus any manifest-declared runtime params it accepts, in order |
| L2    | Shape rules       | `shape_rules` entries are valid Python expressions                                                          |
| L3    | Dtype conformance | dtype strings are valid torch types or `same_as()` refs                                                     |
| L4    | Benchmark file    | Bench file imports and calls `load_workloads` / `eval_roofline` with this op name                           |

**Spec-only ops** (`status: spec-only`) receive L0 only; L1-L4 are skipped. Implemented ops are expected to pass all checks in CI.

For L4, strict CI enforcement is enabled per entry by setting `source.bench_manifest_driven: true`. This makes the migration state explicit in the manifest instead of inferring it from benchmark code.

The validator runs in CI as part of the preflight workflow. Run locally:

```bash
python scripts/validate_manifest.py          # normal
python scripts/validate_manifest.py --verbose # per-op progress
```

## Exclusions

The manifest does NOT describe: kernel dispatch logic, multi-kernel pipelines, accumulator dtypes, tensor layout, non-tensor persistent state (`__init__` LUTs/caches), tile sizes, or autotuning config.
