# Roofline

This document describes the `roofline` field in `ops_manifest.yaml`: what it is, how to author one, and who consumes it.

## 1. Performance Model

### 1.1 Baseline Selection

Kernel performance is measured against hardware Speed-of-Light (SOL), not against PyTorch or vendor baselines. The `roofline` field supplies the per-op inputs this model needs (§2).

### 1.2 Metric Definition

```
memory_time  = bytes_moved / hbm_bandwidth
compute_time = total_flops / peak_flops
sol_time     = max(memory_time, compute_time)
efficiency   = sol_time / actual_time
```

Inputs:

- `bytes_moved`, `total_flops` — manifest `roofline` (§2).
- `hbm_bandwidth`, `peak_flops` — GPU profile (§5.1).
- `actual_time` — benchmark output (§5.2).

Bound type is whichever term dominates `sol_time` (memory-bound if `memory_time > compute_time`, else compute-bound). It depends on shape, not on the op; the roofline tool computes it per-workload and the manifest does not declare it.

## 2. Field Specification

### 2.1 Output Contract

Per workload, the `roofline` field yields `(flops: int, bytes: int)`. Consumers read these integers via `tileops.manifest.eval_roofline()` (§4.1).

### 2.2 Formula Modes

An entry uses one of two modes:

| Mode   | Form                      | When                              |
| ------ | ------------------------- | --------------------------------- |
| Inline | `vars?` + `flops`/`bytes` | Formula fits a Python expression. |
| Func   | `func: "module.path"`     | Formula needs real Python logic.  |

**Inline.** Roofline variables come from `shape` dim names where possible. Anything `shape` cannot supply — arbitrary-rank dims, slice products, shape-derived quantities — is declared in `vars`. `flops` and `bytes` are Python expressions over all resolved variables + `elem_bytes` + approved helpers (§3.3). `elem_bytes` is the byte size of the first input's dtype.

**Func.** Point at `tileops.perf.formulas.<name>` returning `{"flops": int, "bytes": int}`. Use when inline arithmetic is insufficient (conditionals, shape traversal, data-dependent logic). See §3.4.

```yaml
# Inline — shape dim names cover all variables
roofline:
  flops: "2 * M * N * K"
  bytes: "(M * K + K * N + M * N) * elem_bytes"

# Inline — shape cannot supply the variables; vars fills in
roofline:
  vars:
    M: "product(x.shape[:dim])"
    N: "x.shape[dim]"
  flops: "4 * M * N"
  bytes: "(2 * M * N + N) * elem_bytes"

# Func — complex formulas
roofline:
  func: "tileops.perf.formulas.my_op_roofline"
```

## 3. Authoring Rules

### 3.1 Two-Layer Expression Model

Roofline metadata has two expression layers. Agents must keep them
separate.

**Variable resolution** derives named roofline variables from runtime
inputs. Two sources contribute: `shape` dim names auto-bind when `shape`
is declared; any remaining variables are declared in `vars:`. This layer
belongs to manifest analysis only. The `vars:` evaluation context
contains all `signature.inputs` tensor names with a `.shape` accessor,
all `signature.params` names, and `elem_bytes`. It may use shape access,
slicing, `product()`, `range()`, and small comprehensions:

```yaml
roofline:
  vars:
    M: "product(x.shape[:dim])"
    N: "x.shape[dim]"
```

The result of variable resolution is ordinary data:

```python
{"M": 1024, "N": 4096, "elem_bytes": 2}
```

The `roofline.vars` evaluator exposes `product`, `isinstance`, `len`,
`set`, `tuple`, `list`, `range`, `int`, `float`, `bool`, `min`, `max`,
`sum`, `abs`, `log2`, `ceil`, and `floor`. Expressions run with
`__builtins__` stripped.

When `roofline.vars` uses a reduction `dim`, it follows the manifest
`shape_rules` contract: validate range first, then normalize negative
axes with `% x.ndim`, then reject duplicate axes for sequence dims. A
roofline expression must not silently normalize an invalid axis.

**Arithmetic formulas** compute final `flops` and `bytes` from resolved
variables only:

```yaml
roofline:
  flops: "4 * M * N"
  bytes: "(2 * M * N + N) * elem_bytes"
```

Arithmetic formulas must not contain tensor access, shape slicing,
comprehensions, attributes, or arbitrary calls. Generated Op code may
consume only this arithmetic layer.

### 3.2 Inline Formula Language

`roofline.flops` and `roofline.bytes` are Python expression source
strings. They are not a custom DSL and must not be parsed into a separate
IR.

Validator checks are the syntax and execution gate:

1. Compile each formula with Python `compile(expr, filename, "eval")`.
1. Evaluate it in CI with validator-provided sample bindings.
1. Require the result to be finite, non-negative, and numeric.

Invalid syntax, unknown names, unsupported helper usage, non-numeric
results, and runtime errors are CI failures.

### 3.3 Shared Namespace

The expression namespace is fixed and shared by validator evaluation and
generated Op code. Formula strings may reference only:

- resolved roofline variable names
- `elem_bytes`
- approved math helpers: `ceil`, `floor`, `log2`

Agents must update validator and codegen together when adding or removing
helper names. A formula accepted by validator must run in generated Op
code without semantic changes.

### 3.4 Escape Hatch: `roofline.func`

If an op's roofline cannot be expressed as a small Python expression over
resolved variables and approved helpers, use:

```yaml
roofline:
  func: "tileops.perf.formulas.my_op_roofline"
```

Do not extend inline formulas with new mini-language features when the
formula is better represented as Python code in `tileops/perf/formulas.py`.

### 3.5 Runtime Evaluation Timing

Generated `eval_roofline()` follows the same timing rule as shape
inference: run it at the first moment all required variables are known,
cache the result, and do not recompute for identical inputs.

| Op category    | When variables are known                                           | `eval_roofline()` behavior                                            |
| -------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Fixed-rank     | `__init__` (all dimensions provided)                               | Called once during init; result may be stored on the Op.              |
| Arbitrary-rank | `__init__` for `static_dims`; `forward` for remaining dynamic dims | Called in `forward()` when dynamic vars are resolved; cache by input. |

Non-runtime consumers should use `tileops.manifest.eval_roofline()`
directly. Dynamic-rank Op methods require `self.*` variables that may not
exist until `forward()` has seen concrete inputs.

## 4. Consumers and Evaluator Boundaries

### 4.1 Consumers

`ops_manifest.yaml` is the source of truth for roofline metadata. Its
consumers are:

- **Validator / CI** — checks the `roofline` schema, validates inline
  formula syntax, evaluates formulas with sample bindings, and rejects
  formulas that cannot produce finite non-negative numeric results.
- **Benchmark layer** — `ManifestBenchmark` and bespoke benchmark modules
  use `tileops.manifest.resolve_roofline_vars()` and
  `tileops.manifest.eval_roofline()` to report FLOPs and bytes for
  concrete benchmark workloads.
- **Roofline analysis tools** — M5 tooling combines benchmark raw latency,
  manifest FLOPs/bytes, and GPU profile data to compute SOL efficiency
  and bound type.
- **Op codegen** — generated Op code derives runtime `eval_roofline()`
  behavior from manifest formulas after all required variables are known.

Tests and workloads are not roofline formula consumers. They may provide
shapes and dtypes that benchmark consumers use, but they must not define
or reinterpret roofline formulas.

### 4.2 Evaluator Surfaces

The project has two legitimate expression-evaluator surfaces, plus a
third that was considered and explicitly rejected. Future work must
respect these boundaries so roofline semantics do not grow a second
parser or a second DSL.

**Surface 1 — `tileops.manifest._safe_eval`.** Pre-existing manifest-side
safe evaluator used for manifest-level expression checks (e.g. validating
manifest fields that contain expressions). Its scope is narrow and
already fixed; it must not be extended to interpret roofline formulas.

**Surface 2 — `roofline.vars` evaluator.** Used by
`resolve_roofline_vars()` to derive named roofline variables from runtime
inputs. It is deliberately more capable than inline arithmetic because
it must handle shape-derived logic: tensor shape access, slicing,
`product(...)`, `range(...)`, small comprehensions, and reduction-dim
normalization. Its namespace is the one listed under **Variable
resolution** above.

**Surface 3 (rejected) — Op-local `eval_roofline()` evaluator.** Earlier
drafts proposed a base-class AST-based safe evaluator that would
re-interpret `roofline.flops` / `roofline.bytes` at Op runtime
(class-level `_roofline_vars` / `_flops_expr` / `_bytes_expr`, AST
lowering in a base class). This is **not** how roofline is implemented.
Codegen does not introduce a third evaluator — it copies
validator-approved expression strings into generated Python (see
[§4.3 Codegen Contract](#43-codegen-contract)).

Boundary rules:

- `roofline.vars` may be complex (shape access, slicing, comprehensions).
  `roofline.flops` / `roofline.bytes` must remain pure Python arithmetic
  over resolved variables, `elem_bytes`, and approved helpers. They are
  never promoted into a second DSL.
- `tileops.manifest._safe_eval` and the `roofline.vars` evaluator must
  not both interpret the same class of expression. `_safe_eval` stays in
  its current manifest-field role; `roofline.vars` owns shape-derived
  variable resolution for roofline.
- If a roofline formula is too complex for simple arithmetic — needs
  conditionals, shape traversal, or helpers outside the approved set —
  move it to [`roofline.func`](#34-escape-hatch-rooflinefunc). Do not
  extend inline formulas into a mini-language.
- No Op-local parser or AST evaluator for roofline expressions. Op
  runtime consumes generated `eval_roofline()` that was produced by
  copying a validator-approved expression.

### 4.3 Codegen Contract

Codegen does not interpret, transform, or partially parse
`roofline.flops` / `roofline.bytes`. It copies validated expression
strings into generated Python `eval_roofline()` code.

Example generated shape:

```python
def eval_roofline(self) -> tuple[int, int]:
    M = self.M
    N = self.N
    elem_bytes = self.dtype.itemsize
    return (
        4 * M * N,
        (2 * M * N + N) * elem_bytes,
    )
```

Generated code must expose the same local names and helper functions used
by validator. It must not call an Op-local expression parser.

### 4.4 Benchmark Consumption

Benchmarks consume roofline metadata through `tileops.manifest`, not by
reimplementing formulas locally.

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


flops, mem_bytes = eval_roofline(_OP_NAME, M=m, N=n, elem_bytes=elem_bytes)
```

`ManifestBenchmark` evaluates `roofline.vars` against the concrete
workload shape and op params so non-last-axis and multi-axis reductions
use the same `M` / `N` bindings as the actual op call:

```python
from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params


@pytest.mark.parametrize(
    "shape, dtype, op_params", workloads_to_params("SumFwdOp", include_extra=True)
)
def test_sum_bench(shape, dtype, op_params):
    test = SumTest(shape, dtype)
    bm = ManifestBenchmark("SumFwdOp", test, op_params=op_params)
    ...
```

For entries driven by `workloads_to_params(..., include_extra=True)`,
workload keys other than `x_shape`, `dtypes`, and `label` are treated as
op-call params and are forwarded to `resolve_roofline_vars()`.

Manifest validation must ensure implemented benchmark files use the
manifest roofline helpers rather than hardcoded formulas.

## 5. Reference

### 5.1 GPU Profile

Hardware parameters use theoretical values with calibration factors from one-time microbenchmark measurements. YAML files store only `theoretical` and `calibration`; `effective = theoretical × calibration` is computed by `load_profile()`:

```yaml
# tileops/perf/profiles/h200.yaml
hbm:
  theoretical: 4800e9       # bytes/s, from spec sheet
  calibration: 0.94         # from microbench
tensor_core:
  fp16:
    theoretical: 989.5e12   # FLOPS, from spec sheet
    calibration: 0.75       # from microbench (cuBLAS peak)
```

Profiles are stored in `tileops/perf/profiles/`. Microbenchmarks for calibration live in `benchmarks/hardware/`.

### 5.2 Benchmark–Roofline Decoupling

Benchmark (M4) produces raw time (JSON/CSV). Roofline (M5) is a separate tool that reads raw time + manifest formulas + GPU profile to compute efficiency. This separation enables:

- Re-analyzing historical data when GPU profiles are updated
- Multiple consumers of raw benchmark data (roofline, regression detection, dashboards)
- Benchmark module has no third-party dependencies beyond the project itself
