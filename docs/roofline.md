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

Per workload, the `roofline` field yields `(flops: int, bytes: int)`. Consumers read these integers via `op.eval_roofline()` on an instantiated Op (§4.4).

### 2.2 Formula Modes

An entry uses one of two modes:

| Mode   | Form                      | When                              |
| ------ | ------------------------- | --------------------------------- |
| Inline | `vars?` + `flops`/`bytes` | Formula fits a Python expression. |
| Func   | `func: "module.path"`     | Formula needs real Python logic.  |

**Inline.** Roofline variables come from `shape` dim names where possible. Anything `shape` cannot supply — arbitrary-rank dims, slice products, shape-derived quantities — is declared in `vars`. `flops` and `bytes` are Python expressions over all resolved variables + `elem_bytes` + approved helpers (§4.4.4). `elem_bytes` is the byte size of the first input's dtype. **Ops whose `bytes` depend on multiple input dtypes (mixed-precision GEMM, Attention, etc.) cannot be expressed in inline mode** and must use `func`.

**Func.** Point at `tileops.perf.formulas.<name>`. The callable is human-authored and returns `(flops, bytes)`. **Recommended signature: `func(op)`** — matching the agent-generated `eval_roofline(self)` path, which is what codegen's emitted call assumes. A human author who prefers a different signature owns the resulting integration (e.g., a wrapper). Use `func` when inline arithmetic is insufficient (mixed-precision byte accounting, conditionals, shape traversal, data-dependent logic).

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

## 3. Consumers

`ops_manifest.yaml` is the source of truth for the `roofline` field. Four modules read it:

- **Schema validator / CI** — structural checks only (schema, mode exclusivity, `func` importability). Does **not** execute formulas or hold a helper whitelist. Spec: §4.1.
- **Benchmark layer** — instantiates an Op per workload and reads `(flops, bytes)` from `op.eval_roofline()`. Hardcoded formulas in benchmark files are a CI failure. Spec: §4.2.
- **Roofline tool (M5)** — reads `(flops, bytes)` from benchmark output (which was produced by `op.eval_roofline()`), combines with GPU profile (§5.1) to compute SOL efficiency and bound type. Spec: §4.3.
- **Op codegen** — generates each op's `eval_roofline()` method; is the authoritative gate for name and form correctness. Spec: §4.4.

Tests and workloads are not consumers: they may supply shapes and dtypes but must not define or reinterpret roofline formulas.

## 4. Consumer Specifications

### 4.1 Schema Validator / CI

Runs on every PR touching `ops_manifest.yaml`. Scope is structural.

In scope:

- Required fields per mode: inline must have `flops` and `bytes`; func must have `func`.
- Mode exclusivity: `flops`/`bytes`/`vars` and `func` must not coexist.
- Field types: `flops`/`bytes`/`func` are non-empty strings; `vars` is a mapping of str → non-empty str.
- `func` dotted path resolves at import time.

Out of scope:

- Name whitelist — a formula's names are checked by codegen (§4.4), which owns the binding table. Validator does not mirror it.
- Form checks (layer violations, forbidden AST nodes) — codegen refuses to emit invalid forms.
- Numeric checks (finite / non-negative / numeric) — tests exercise generated `eval_roofline()` on each workload.

Validator holds no callables, no sample bindings, no `__builtins__` sandbox. Adding a helper does not touch the validator.

### 4.2 Benchmark Layer

Contract:

- Instantiate the Op for each workload and call `op.eval_roofline()` to obtain `(flops, bytes)`. No manifest-level helper exists — roofline evaluation lives only inside each Op's generated method.
- `ManifestBenchmark` and `workloads_to_params(..., include_extra=True)` are the canonical harnesses; non-reserved workload keys forward as op-call params passed to the Op's `__init__`.
- A benchmark file that computes FLOPs or bytes locally is a CI failure.
- Benchmark output must record the `(flops, bytes)` returned by `op.eval_roofline()` so downstream consumers (M5) can read the numbers without re-instantiating ops.

### 4.3 Roofline Tool (M5)

Inputs:

- Benchmark output (JSON/CSV) produced by M4, carrying per-workload latency and the `(flops, bytes)` that benchmark obtained from `op.eval_roofline()`.
- GPU profile (§5.1).

Outputs: SOL efficiency, bound type, per-workload reports.

Does not interpret formula strings at all. M5 reads pre-computed numbers from the benchmark output; it never instantiates Ops or runs roofline expressions.

### 4.4 Op Codegen

Codegen runs for `status: implemented` entries only. `spec-only` entries — where either the implementation does not exist or the Op interface does not yet match the manifest — are skipped; codegen re-evaluates them once the status flips.

Codegen is the authoritative gate for name and form correctness. A formula referencing an unknown name or violating a layer's form constraints fails codegen; a manifest that fails codegen cannot land. Numeric correctness is exercised by tests, not codegen.

#### 4.4.1 Method Template

For each op, codegen emits an `eval_roofline()` method returning `(flops: int, bytes: int)`. The method signature is part of the shared Op interface defined in [ops-design-reference.md](ops-design-reference.md); this document specifies only how the body is generated from the manifest.

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

#### 4.4.2 Manifest Inputs

For each manifest entry, codegen reads one of:

- **Inline** — `vars` (optional), `flops`, `bytes`. All are Python expression source strings. Codegen emits the method body per §4.4.3.
- **Func** — `func` (dotted module path resolving to a human-authored callable). Codegen emits `return <func>(self)` as the method body. This presumes the **recommended** signature `func(op) -> tuple[int, int]` (returning `(flops, bytes)`), aligned with the agent-generated `eval_roofline(self)` shape. The recommendation is a reference for authors, not a gate — codegen does not introspect or validate the callable. If the author wants a different signature, they take responsibility for making the callable work with the emitted call (e.g., by writing a thin wrapper at the dotted path).

#### 4.4.3 Expression Layers

Inline mode has two layers. Codegen emits them as two sequential blocks in the method body.

- **vars layer** — shape-derived resolution. Allowed operations: tensor shape access, slicing, `product()`, `range()`, small comprehensions.
- **arithmetic layer** — `flops` and `bytes` over resolved variables + `elem_bytes` + approved helpers only. Forbidden: tensor access, shape slicing, comprehensions, attributes, arbitrary calls.

Codegen actions emit two sequential blocks:

**Block 1 — vars resolution.**

- Bind each `signature.inputs` tensor to a local: `x = self.x`. (Arbitrary-rank ops: `self.x` is bound in `forward()` before `eval_roofline()` runs.)
- Bind each `signature.params` name: `dim = self.dim`.
- Bind `elem_bytes` from whichever dtype source exists at the call site (§4.4.5): use `self.dtype.itemsize` when `eval_roofline()` runs at `__init__` (fixed-rank — no tensor yet), and `self.<first_input>.dtype.itemsize` when it runs in `forward()` (arbitrary-rank — the tensor is bound).
- If `vars:` is present, emit one assignment per entry in YAML declaration order: `<name> = <vars[name]>`, copying the expression string verbatim. Later entries may reference earlier locals.
- If `vars:` is absent and `shape` is fixed-rank, emit assignments from the `shape` declaration (tuple-unpack `self.x.shape`, or read `self.<dim>` if the Op stored dims at `__init__`).

**Block 2 — arithmetic.** Return `(<flops>, <bytes>)` with both expression strings copied verbatim. They reference only Block 1 locals + `elem_bytes` + arithmetic-layer helpers (§4.4.4).

Do **not** inline a vars expression into the arithmetic expression (e.g. `return (4 * product(x.shape[:dim]) * x.shape[dim], ...)`). That collapses the two layers and violates arithmetic-layer restrictions.

Reduction dim handling in the vars layer follows the manifest `shape_rules` contract: validate range → normalize `% x.ndim` → reject duplicate axes for sequence dims. A roofline expression must not silently normalize an invalid axis.

Example — arbitrary-rank (explicit `vars`):

```python
def eval_roofline(self) -> tuple[int, int]:
    # Block 1: vars layer
    x = self.x
    dim = self.dim
    elem_bytes = self.x.dtype.itemsize
    M = product(x.shape[:dim])
    N = x.shape[dim]
    # Block 2: arithmetic layer
    return (
        4 * M * N,
        (2 * M * N + N) * elem_bytes,
    )
```

Fixed-rank form: see the template in §4.4.1.

Any `Name` node not resolvable to a Block 1 local, `elem_bytes`, or an arithmetic-layer helper causes codegen to raise. Any forbidden AST node (`Attribute`, `Subscript`, `Comprehension`, `Lambda`, …) in the arithmetic layer also causes codegen to raise. These are the enforcement of the "authoritative gate" responsibility.

#### 4.4.4 Namespace

Codegen knows how to bind the following names when generating the method body. This is the single source of truth for what an inline formula may reference.

**vars layer**

| Bucket    | Names                                                                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Tensors   | All `signature.inputs` names, exposed with a `.shape` accessor                                                                               |
| Params    | All `signature.params` names                                                                                                                 |
| Constants | `elem_bytes`                                                                                                                                 |
| Helpers   | `product`, `isinstance`, `len`, `set`, `tuple`, `list`, `range`, `int`, `float`, `bool`, `min`, `max`, `sum`, `abs`, `log2`, `ceil`, `floor` |

**arithmetic layer**

| Bucket    | Names                             |
| --------- | --------------------------------- |
| Variables | Resolved vars from the vars layer |
| Constants | `elem_bytes`                      |
| Helpers   | `ceil`, `floor`, `log2`           |

Adding or removing a helper = edit codegen's binding table. No parallel update in validator or anywhere else is required. If a formula references a name not in this table, codegen fails; the manifest does not land.

#### 4.4.5 Runtime Timing

Generated `eval_roofline()` follows shape-inference timing: resolve variables at the first moment they are known, cache the result, do not recompute for identical inputs.

| Op category    | Variables known at                                                 | `eval_roofline()` behavior                                            |
| -------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Fixed-rank     | `__init__` (all dimensions provided)                               | Called once during init; result may be stored on the Op.              |
| Arbitrary-rank | `__init__` for `static_dims`; `forward` for remaining dynamic dims | Called in `forward()` when dynamic vars are resolved; cache by input. |

Non-runtime consumers must instantiate the Op (or read pre-computed `(flops, bytes)` from benchmark output). No manifest-level roofline evaluator exists; every value flows through `op.eval_roofline()`.

#### 4.4.6 Evaluator Surface Boundary

Roofline expressions live in exactly one place at runtime: the plain Python body that codegen emits into each op's `eval_roofline()`. No standalone roofline evaluator exists.

| Surface                           | Scope                 | Interprets roofline expressions? |
| --------------------------------- | --------------------- | -------------------------------- |
| `tileops.manifest._safe_eval`     | Manifest field checks | No                               |
| Op-local AST evaluator            | —                     | **REJECTED** — must not be built |
| Manifest-level roofline evaluator | —                     | **REJECTED** — must not be built |

Rules:

- `tileops.manifest._safe_eval` is scoped to manifest-field expression checks and must not be extended to interpret roofline expressions.
- No `tileops.manifest.eval_roofline()` / `resolve_roofline_vars()` helper that evaluates roofline expressions exists in the target design. Any consumer wanting `(flops, bytes)` either calls `op.eval_roofline()` on an Op instance or reads pre-computed values from benchmark output.
- Generated `eval_roofline()` must not parse, AST-analyze, or safe-eval its own formula strings. Codegen does the name/form check at generation time (§4.4.3 / §4.4.4) and then copies validated expressions into plain Python.
- If a formula is too complex for inline arithmetic (conditionals, shape traversal, data-dependent logic), switch the entry to `func` mode (§2.2). Do not extend inline formulas into a mini-language.

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

Benchmark (M4) produces per-workload records containing raw time and the `(flops, bytes)` from `op.eval_roofline()`. Roofline (M5) is a separate tool that reads those records + GPU profile to compute efficiency. This separation enables:

- Re-analyzing historical data when GPU profiles are updated
- Multiple consumers of raw benchmark data (roofline, regression detection, dashboards)
- Benchmark module has no third-party dependencies beyond the project itself
