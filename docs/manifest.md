# Op Manifest Specification

`ops_manifest.yaml` is the central op registry, the agent entry point, and the **source of truth** for op interfaces. Runtime code (including `Op.infer_shape()`) is generated from the manifest, not the other way around.

```
ops_manifest.yaml (spec)
       │
       ├──→ Agent generates Op code + Op.infer_shape()
       ├──→ Test generator reads workloads + tolerance
       ├──→ Benchmark reads workloads
       ├──→ Roofline reads flops/bytes formulas
       └──→ Docs generator reads signatures
```

## Trust Model

The manifest separates specification from implementation. Op interfaces are declared in the manifest and reviewed by humans; code is generated from the manifest and validated programmatically.

**Motivation.** When the same agent produces both kernel code and test code, test inputs and tolerances may inadvertently align with implementation behavior rather than mathematical correctness. A human-reviewed spec with automated validation reduces this coupling.

```
Human reviews manifest (source of truth)
  │
  ├──→ Parser derives programmatic checks (CI enforces)
  │     ├─ Op.forward() signature matches manifest
  │     ├─ Tests cover all declared dtypes/shapes
  │     ├─ Benchmarks use declared workloads
  │     ├─ infer_shape() is consistent with shape_rules
  │     └─ constraints are respected
  │
  └──→ Agent generates code within spec constraints
        ├─ Manifest changes require human approval
        ├─ Generated code must pass parser validation
        └─ CI enforces validation
```

**Invariants:**

1. The manifest is the sole source of truth for op interfaces. Changes require human review.
1. Programmatic validation is derived from the manifest, not from the generating agent.
1. Test coverage (shapes, dtypes, workloads) is determined by the manifest, not by the agent.
1. Reference implementations are provided independently of kernel implementations. The spec defines *what*, the kernel implements *how*, the reference independently verifies *whether*.

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

| Field          | Type | Required | Description                                                               |
| -------------- | ---- | -------- | ------------------------------------------------------------------------- |
| `inputs`       | dict | yes      | Input tensors keyed by name.                                              |
| `outputs`      | dict | yes      | Output tensors keyed by name.                                             |
| `params`       | list | no       | Scalar / config parameters.                                               |
| `shape_rules`  | list | no       | Python expressions for shape inference. Used to generate `infer_shape()`. |
| `dtype_combos` | list | no       | Valid dtype combinations. Overrides per-tensor `dtype` when present.      |

**Tensor fields** (`inputs` / `outputs`):

Each tensor is a key-value pair where the key is the tensor name (referenced in `shape_rules`, `same_as(ref)`, and roofline expressions) and the value is a dict with:

| Field         | Type   | Required | Description                                                                       |
| ------------- | ------ | -------- | --------------------------------------------------------------------------------- |
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

**R1. Dict, not list.** Signature `inputs` and `outputs` use dicts keyed by tensor name. This avoids redundant `name` fields inside each entry and aligns with how tensors are referenced elsewhere (roofline expressions, shape_rules, dtype_combos). `params` remains a list to preserve positional order.

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

#### Shape Decision Tree

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

#### Examples

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
  - {name: dim, type: int, default: -1}
  - {name: eps, type: float, default: 1e-6}
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
  x: {dtype: "float16 | bfloat16"}
outputs:
  y: {dtype: "same_as(x)"}
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
        x: {dtype: "float16 | bfloat16"}
        weight: {dtype: "same_as(x)"}
      outputs:
        y: {dtype: "same_as(x)", shape: "same_as(x)"}
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

Each workload entry is a dict. Only `dtypes` is universal; all other keys are op-specific.

| Field    | Required | Description                                                       |
| -------- | -------- | ----------------------------------------------------------------- |
| `dtypes` | yes      | List of dtypes to test.                                           |
| `label`  | no       | Human-readable tag. Auto-generated from shape + dtype if omitted. |

Shape keys correspond to signature input tensor names with a `_shape` suffix. For ops with a single primary input, `x_shape` is conventional. For ops with multiple independently-shaped inputs, use per-tensor keys:

```yaml
# Single primary input (norm, elementwise, reduction):
- {x_shape: [2048, 4096], dtypes: [float16, bfloat16], label: "llama-3.1-8b"}

# Multiple inputs with independent shapes (attention):
- {q_shape: [1, 1, 32, 128], kv_shape: [1, 8192, 8, 128], dtypes: [float16], label: "gqa-decode"}
```

Op-specific scalar parameters (e.g., `groups` for GroupNorm, `is_causal` for attention) can be added per entry. Shapes target real model architectures.

### Roofline

| Mode     | Format                           | When to use                                                |
| -------- | -------------------------------- | ---------------------------------------------------------- |
| Inline   | `flops: "expr"`, `bytes: "expr"` | Simple arithmetic on dimension names.                      |
| Function | `func: "module.path"`            | Complex formulas with multiple parameters or conditionals. |

Functions live in `tileops/perf/formulas.py`, return `{"flops": int, "bytes": int}`. The field is `bytes` (total bytes moved), not `memory`.

### Source

| Field    | Required | Type           | Description                                           |
| -------- | -------- | -------------- | ----------------------------------------------------- |
| `kernel` | yes      | string or list | Kernel implementation path(s). List for multi-kernel. |
| `op`     | yes      | string         | Op class path.                                        |
| `test`   | yes      | string         | Test file path.                                       |
| `bench`  | yes      | string         | Benchmark file path.                                  |

## Design Principles

### YAML is data, not a DSL

The manifest is a **data structure** — key-value pairs, lists, dicts. It must not encode logic. Specifically:

- No conditionals, loops, or branching in YAML values.
- Roofline expressions (`"4 * M * N"`) are arithmetic on variables, not code. They are evaluated by `_safe_eval()` which only permits numeric constants, basic arithmetic, and whitelisted functions (`log2`, `ceil`, `floor`).
- `shape_rules` are declarative assertions about shape relationships, not imperative shape computation. They describe *what* must hold, not *how* to compute it.
- Variable bindings between roofline formulas and workload shapes are the responsibility of the consumer (benchmark code), not the manifest. The manifest declares formulas; consumers supply variable values.

When a proposed manifest extension requires encoding logic (conditionals, dispatch rules, evaluation order), the answer is not "extend the schema" — it is "move that concern to code" (Tier 2 `func` mode) or "redesign the Op interface."

### Op interface requirements for manifest entry

Every Op in the manifest must have a **clean, static interface**:

1. **All tensor parameters are required.** No `Optional[Tensor]` in `forward()`.
1. **Output count is fixed.** No conditional returns based on constructor parameters.
1. **One Op, one concern.** If an Op serves two usage patterns (e.g., training without state vs. inference with state), split it into two Op classes that share the same Kernel.

Ops that violate these rules must be refactored before entering the manifest. This is not a manifest limitation — it is an Op design hygiene requirement. Optional tensors and conditional outputs make interfaces ambiguous for all consumers (tests, benchmarks, code generators, documentation), not just the manifest.

**Examples of required refactoring:**

| Current interface                                                 | Problem                                   | Refactored                                                                              |
| ----------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------- |
| `GLAFwdOp(q, k, v, g, initial_state=None) → (o, Optional[state])` | Optional input, conditional output        | `GLAFwdOp(q, k, v, g) → o` + `GLAFwdStatefulOp(q, k, v, g, state) → (o, new_state)`     |
| `FusedTopKOp(gating, correction_bias=None) → (weights, ids)`      | Optional input                            | `FusedTopKOp(gating) → (w, ids)` + `FusedTopKWithCorrectionOp(gating, bias) → (w, ids)` |
| `DeltaNetFwdOp(...) → (o, S, Aw, Au, w, u)`                       | Exposes backward intermediates as outputs | `DeltaNetFwdOp(...) → o` (intermediates saved internally)                               |

Both split variants share the same underlying Kernel — the split is at the Op wrapper level, with zero code duplication.

### Tier model

Ops are classified into two tiers based on roofline complexity. Both tiers share the same signature, workloads, and source schema — they differ only in how roofline is expressed.

**Decision flow:**

```
Does forward() have all-required tensors and fixed output count?
├─ NO → Refactor the Op interface first (see above)
└─ YES
    Can roofline be expressed as a single arithmetic expression per metric?
    ├─ YES → Tier 1 (inline roofline)
    └─ NO  → Tier 2 (func roofline)
```

**Tier 1 — inline roofline.** Simple arithmetic on dimension variables. Covers ops where FLOPs and bytes are a direct function of tensor dimensions.

```yaml
roofline:
  flops: "4 * M * N"
  bytes: "(2 * M * N + N) * elem_bytes"
```

Families: norm, reduction, elementwise, RoPE, FFT, GEMM.

**Tier 2 — func roofline.** References a Python function for complex formulas where inline expressions are insufficient (e.g., causal masks halve FLOPs, conv stride/padding affect output size, variable-length sequences).

```yaml
roofline:
  func: "tileops.perf.formulas.mha_fwd_roofline"
```

Functions live in `tileops/perf/formulas.py`, accept workload parameters as kwargs, and return `{"flops": int, "bytes": int}`.

Families: attention (fwd/bwd), conv, MoE composites, variable-length attention.

**Classification of current Op families:**

| Family                    | Tier   | Count | Notes                                                                            |
| ------------------------- | ------ | ----- | -------------------------------------------------------------------------------- |
| Norm                      | 1      | ~10   | Currently in manifest                                                            |
| Reduction                 | 1      | ~18   | `output_dtype` may differ (e.g., argmax → int64); declared directly in signature |
| Elementwise               | 1      | ~60   | Highly uniform; batch-generate manifest entries                                  |
| RoPE                      | 1      | ~5    | LUT state is internal to `__init__`, not in `forward()` signature                |
| FFT                       | 1      | ~1    | Same — twiddle factors are `__init__` state                                      |
| GEMM                      | 1      | ~2    | Simple roofline: `2 * M * N * K`                                                 |
| Conv                      | 2      | ~3    | Roofline depends on stride, padding, kernel size                                 |
| Dense Attention (fwd)     | 2      | ~3    | Causal mask affects FLOPs                                                        |
| Dense Attention (bwd)     | 2      | ~2    | Multi-kernel pipeline is internal; external signature is fixed                   |
| Variable-length Attention | 2      | ~4    | `cu_seqlens` is a required tensor input, not optional                            |
| Linear Attention          | 1 or 2 | ~8    | After interface refactoring (split stateful variants)                            |
| MoE (composites)          | 2      | ~4    | External interface is simple; internal pipeline is implementation detail         |
| MoE (primitives)          | 1      | ~6    | FusedTopK, Permute, Unpermute, GroupedGemm                                       |

### What the manifest does NOT describe

The following are **implementation details** that belong in code, not in the manifest:

- **Kernel dispatch logic** — which kernel class is selected based on shape, dtype, or hardware (e.g., Hopper vs. Ampere). This is the Op's `default_kernel_map` responsibility.
- **Multi-kernel pipelines** — an Op that internally chains prep → main → post kernels. The manifest describes the Op's external interface, not its internal orchestration.
- **Accumulator dtypes** — intermediate float32 accumulators used inside kernels. These do not appear in `forward()` signatures.
- **Tensor layout** — BSHD vs. BHSD, packed vs. padded. Layout is a kernel-level concern. If two layouts require different interfaces, they are different Ops.
- **Non-tensor persistent state** — lookup tables, RNG state, frequency caches. These are constructed in `__init__` and do not appear in `forward()`.
- **Kernel implementation details** — tile sizes, memory strategies, `num_per_thread`.
- **Autotuning configuration** — handled by the kernel layer.
