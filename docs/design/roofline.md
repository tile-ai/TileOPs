# Roofline

This document describes the `roofline` field in `src/tileops/manifest/`: what it is, how to author one, and who consumes it.

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
- `hbm_bandwidth` — GPU profile (§5.1), `hbm` section, effective value.
- `peak_flops` — GPU profile section named by `op.compute_roof()` (§1.4), effective value.
- `actual_time` — benchmark output (§5.2): device-busy time plus any device copies excluded from it (`uncounted_copy_ms`).

Bound type is whichever term dominates `sol_time` (memory-bound if `memory_time > compute_time`, else compute-bound). It depends on shape, not on the op; the roofline tool computes it per-workload and the manifest does not declare it.

The metric is **algorithmic** SOL efficiency. Three statements delimit what a reading means:

1. `bytes_moved` is the algorithm's minimum traffic, not measured DRAM traffic: each distinct input storage the algorithm reads counts one read, each public output one write, and a `mutated` input counts both. An intermediate never counts, whatever stage produces it, and a declared input the algorithm does not read produces no traffic.
1. The metric is defined on a call that binds one storage per declared input, which is what every `workloads` row binds. An aliasing call — `add(x, x)` — is priced at two operands, above what it moves: the metric does not describe that call, and the formula is not wrong. Pricing it would require every multi-operand op to expose storage identity to its formula, which the oracle's meta tensors cannot carry.
1. `total_flops` follows the §1.3 counting convention, not per-instruction hardware cost; the metric does not certify an SFU-bound kernel as at its limit.
1. The compute roof is the unit an optimal implementation would use (§1.4), not the unit the current kernel runs on.

Rows the model cannot price honestly are handled three ways:

- **Blank, never guessed** — timed without CUPTI, `bytes` formula yields zero, or no GPU profile matches the device.
- **Labeled latency-bound** — `sol_time` and measured time both below the latency floor: launch overhead dominates and the model has no traction; regression detection still covers the row.
- **Reported as a formula error, never as a fast kernel** — the row implies a rate above a *theoretical* ceiling.

### 1.3 Convention

Per-element FLOP rule for elementwise ops:

- One basic arithmetic op (add, sub, mul, div, neg, abs, recip) counts as 1 FLOP.
- One transcendental call (`exp`, `log`, `log1p`, `erf`, `tanh`, `sin`, `cos`, `sqrt`, `rsqrt`, etc.) counts as 1 FLOP at the convention level. Hardware-specific cost models do not feed back into the manifest.
- One compare-and-select (`max`, `min`, `maximum`, `minimum`, single- or two-bound clamp, `relu`-style branch, `where`) counts as 1 FLOP per output element.
- Predicate-only outputs (`eq`, `gt`, etc.) count as 1 FLOP per element.

Composite ops sum their primitives — `sigmoid = neg + exp + add + recip = 4` FLOPs/elem; `silu = sigmoid + mul = 5` FLOPs/elem.

### 1.4 Compute Roof

`Op.compute_roof()` returns the GPU-profile key (§5.1) of the unit that prices the op's FLOPs — `"cuda_core.fp32"`, `"tensor_core.bf16"`, `"tensor_core.fp8"`, ….

- The key states the unit an **optimal** implementation would use, declared by the op author in code. It is never inferred from the running kernel — that would price a kernel on the wrong unit against the wrong ceiling and hide exactly the gap the metric exists to expose. Nor from the input dtype alone — an fp8-backend attention takes fp16/bf16 tensors.
- The `Op` base defaults to `"cuda_core.fp32"`, which covers every op whose arithmetic runs on CUDA cores in fp32 (elementwise, reductions, norms, scans). An op whose FLOPs are matmul contractions overrides it, normally with `tensor_core_roof(self.dtype)`; one whose unit depends on instance state (a backend switch, a quantized path) branches on that state.
- The declaration is valid whenever `eval_roofline()` is — after the op's dtype is bound.
- A wrong or missing override is caught by the nightly physics check (§4.3): a tensor-core kernel priced against the CUDA-core ceiling implies a FLOP rate above that ceiling's theoretical value, reported as a formula error on the next run.

## 2. Field Specification

### 2.1 Output Contract

Per workload, the `roofline` field yields `(flops: int, bytes: int)`. Consumers read these integers via `op.eval_roofline()` on an instantiated Op (§4.4). The compute roof is not part of the manifest field; the op declares it in code (§1.4).

### 2.2 Formula Modes

An entry uses one of two modes:

| Mode   | Form                      | When                              |
| ------ | ------------------------- | --------------------------------- |
| Inline | `vars?` + `flops`/`bytes` | Formula fits a Python expression. |
| Func   | `func: "module.path"`     | Formula needs real Python logic.  |

**Inline.** Roofline variables come from `shape` dim names where possible. Anything `shape` cannot supply — arbitrary-rank dims, slice products, shape-derived quantities — is declared in `vars`. `flops` and `bytes` are Python expressions over all resolved variables + `elem_bytes` + approved helpers (§4.4.4). `elem_bytes` is the byte size of the dtype the call bound; `out_elem_bytes` is the declared output's, so an entry whose write is not its read's dtype — a bool predicate, an integral input promoted to float — states that much inline. **An op whose `bytes` depends on more than those two dtypes (mixed-precision GEMM, Attention, a per-operand quantization) cannot be expressed in inline mode** and must use `func`.

**Func.** Point at `tileops.perf.formulas.<name>`. The callable is human-authored and returns `(flops, bytes)`. **Recommended signature: `func(op)`** — matching the agent-generated `eval_roofline(self)` path, which is what codegen's emitted call assumes. A human author who prefers a different signature owns the resulting integration (e.g., a wrapper). Use `func` when inline arithmetic is insufficient (mixed-precision byte accounting, conditionals, shape traversal, data-dependent logic).

A composite op additionally declares `composition`, naming the stages its cost is made of. It
coexists with either mode and is specified in [manifest.md § Roofline](manifest.md#roofline).

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

`src/tileops/manifest/` is the source of truth for the `roofline` field. Four modules read it:

- **Schema validator / CI** — structural checks only (schema, mode exclusivity, `func` importability). Does **not** execute formulas or hold a helper whitelist. Spec: §4.1.
- **Benchmark layer** — instantiates an Op per workload and reads `(flops, bytes)` from `op.eval_roofline()`. Hardcoded formulas in benchmark files are a CI failure. Spec: §4.2.
- **Roofline tool (M5)** — reads per-workload `(flops, bytes)`, the roof key, and timing from benchmark output, prices them against the GPU profile (§5.1), and emits SOL efficiency and verdicts. Spec: §4.3.
- **Op codegen** — generates an op's `eval_roofline()` method unless the op defines one itself (§4.4.1); is the authoritative gate for name and form correctness. Spec: §4.4.

Two auditors check the field's values rather than consume them: the structural oracle (§4.6) and the NCU bytes audit (§4.5).

Tests and workloads are not consumers: they may supply shapes and dtypes but must not define or reinterpret roofline formulas.

## 4. Consumer Specifications

### 4.1 Schema Validator / CI

Runs on every PR touching `src/tileops/manifest/`. Scope is structural.

Every roofline entry MUST satisfy:

- Required fields per mode: inline has `flops` and `bytes`; func has `func`.
- Mode exclusivity: `flops`/`bytes`/`vars` and `func` do not coexist.
- Field types: `flops`/`bytes`/`func` are non-empty strings; `vars` is a mapping of str → non-empty str.
- `read_bound_exception`, where present, is a mapping of `when` and `reason`, both non-empty strings. `when` joins names, negated names and comparisons of names against literals with `and` or `or`, over params, the workload keys stating what the call does, and `dtype`. Every clause, at every depth, must read the call, so none can settle the condition on its own — that would waive every call of the op (§4.5).
- `func` dotted path resolves at import time.

Out of the validator's scope:

- Name whitelist — a formula's names are checked by codegen (§4.4), which owns the binding table. Validator does not mirror it.
- Form checks (layer violations, forbidden AST nodes) — codegen refuses to emit invalid forms.
- Numeric checks (finite / non-negative / numeric) — tests exercise generated `eval_roofline()` on each workload.

Validator holds no callables, no sample bindings, no `__builtins__` sandbox. Adding a helper does not touch the validator.

### 4.2 Benchmark Layer

Contract:

- Instantiate the Op for each workload and call `op.eval_roofline()` to obtain `(flops, bytes)`. No manifest-level helper exists — roofline evaluation lives inside that method, whether codegen wrote it or the op did.
- `ManifestBenchmark(op, workload)` and `workloads_to_params(..., include_extra=True)` are the canonical consumers; non-reserved workload keys forward as op-call params passed to the Op's `__init__`.
- A benchmark file that computes FLOPs or bytes locally is a CI failure.
- Benchmark output must record the `(flops, bytes)` from `op.eval_roofline()` and the roof key from `op.compute_roof()` (§1.4), so M5 reads the numbers without re-instantiating ops.

### 4.3 Roofline Tool (M5)

Inputs:

- Benchmark output produced by M4, carrying per-workload timing, the `(flops, bytes)` from `op.eval_roofline()`, and the roof key from `op.compute_roof()`.
- GPU profile (§5.1), selected by matching the profile's `gpu` field against the measured device name; no match leaves every SOL reading blank.

Per-workload outputs: SOL efficiency, bound type, latency-bound labels, anomaly reports.

Does not interpret formula strings at all. M5 reads pre-computed numbers from the benchmark output; it never instantiates Ops or runs roofline expressions.

Verdict lines are rendering thresholds, not CI gates:

| Verdict    | Condition | Meaning                                                                                                                                                                                                           |
| ---------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| At ceiling | ≥ 80%     | Done. The HBM ceiling is an envelope over access mixes, and a kernel's own mix caps below it (a perfect 2R:1W kernel reaches ~90% of it, a perfect 1R:1W ~87%); the line sits below every mix's personal ceiling. |
| Anomaly    | > 105%    | Above the achievable ceiling: the formula or the calibration is wrong. Excluded from "at ceiling".                                                                                                                |

Physics check: every row's implied rates (`bytes / time`, `flops / time`) are compared against the *theoretical* ceilings of its roofs. A breach is physically impossible, so it is reported as a formula error that fails the run's health — a formula edit that inflates work is caught on the next nightly. This check is the standing guard on formula overestimation; equality-level validation belongs to the structural oracle (§4.6), read-side hardware-counter validation to the bytes audit (§4.5).

### 4.4 Op Codegen

Codegen runs for `status: implemented` entries only. `spec-only` entries — where either the implementation does not exist or the Op interface does not yet match the manifest — are skipped; codegen re-evaluates them once the status flips.

Codegen is the authoritative gate for name and form correctness. A formula referencing an unknown name or violating a layer's form constraints fails codegen; a manifest that fails codegen cannot land. Numeric correctness is exercised by tests, not codegen.

#### 4.4.1 Method Template

Codegen emits an `eval_roofline()` method returning `(flops: int, bytes: int)` for every op that does not define one. The method signature is part of the shared Op interface defined in [ops-design-reference.md](ops-design-reference.md); this document specifies only how the body is generated from the manifest.

An op that defines the method itself keeps it, and codegen installs nothing. That is for an op whose entry the vars layer cannot express, and its entry says which. Where a formula needs what the call bound rather than what the op was built with, the op states it in `_roofline_kwargs` and the formula reads it — the entry still runs. Everywhere else the entry is what runs, so changing it changes the number.

```python
def eval_roofline(self) -> tuple[int, int]:
    x = _resolve_tensor_binding(self, "x", "SomeFwdOp", optional=False)
    M = product(x.shape[:dim])
    N = x.shape[dim]
    elem_bytes = self.dtype.itemsize
    return (
        4 * M * N,
        (2 * M * N + N) * elem_bytes,
    )
```

A tensor resolves through `self.<name>` or `self.<name>_shape`, so an op binds what its entry reads. Exposing neither is the author's wiring and raises `ValueError`; exposing one and leaving it unset is a caller who has not run `forward()` and raises `RuntimeError`.

#### 4.4.2 Manifest Inputs

For each manifest entry, codegen reads one of:

- **Inline** — `vars` (optional), `flops`, `bytes`. All are Python expression source strings. Codegen emits the method body per §4.4.3.
- **Func** — `func` (dotted module path resolving to a human-authored callable). Codegen emits `return <func>(self)` as the method body. This presumes the **recommended** signature `func(op) -> tuple[int, int]` (returning `(flops, bytes)`), aligned with the agent-generated `eval_roofline(self)` shape. The recommendation is not a gate: codegen does not introspect the callable, and an author choosing another signature owns making it work with the emitted call (e.g. a thin wrapper at the dotted path).

#### 4.4.3 Expression Layers

Inline mode has two layers. Codegen emits them as two sequential blocks in the method body.

- **vars layer** — shape-derived resolution. Allowed operations: tensor shape access, slicing, `product()`, `range()`, small comprehensions.
- **arithmetic layer** — `flops` and `bytes` over resolved variables + `elem_bytes` + approved helpers only. Forbidden: tensor access, shape slicing, comprehensions, attributes, arbitrary calls.

**Block 1 — vars resolution.**

- Bind each `signature.inputs` tensor and `signature.params` name *referenced by the roofline expressions* to a local. Names declared in the manifest but unused by the roofline are not bound; ops are not required to expose them. Op-author conventions for exposing referenced names on the op instance are specified in [`.claude/domain-rules/ops-design.md`](../../.claude/domain-rules/ops-design.md).
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
| Constants | `elem_bytes`; `out_elem_bytes` where the entry declares exactly one output                                                                   |
| Helpers   | `product`, `isinstance`, `len`, `set`, `tuple`, `list`, `range`, `int`, `float`, `bool`, `min`, `max`, `sum`, `abs`, `log2`, `ceil`, `floor` |

**arithmetic layer**

| Bucket    | Names                             |
| --------- | --------------------------------- |
| Variables | Resolved vars from the vars layer |
| Constants | `elem_bytes`, `out_elem_bytes`    |
| Helpers   | `ceil`, `floor`, `log2`           |

`out_elem_bytes` resolves the declared output dtype through the manifest, so an op whose output dtype is not its input's — a bool predicate, an integral input promoted to float — states its write without a second source. Adding or removing a helper = edit codegen's binding table. No parallel update in validator or anywhere else is required. If a formula references a name not in this table, codegen fails; the manifest does not land.

#### 4.4.5 Runtime Timing

Generated `eval_roofline()` follows shape-inference timing: resolve variables at the first moment they are known, cache the result, do not recompute for identical inputs.

| Op category    | Variables known at                                                 | `eval_roofline()` behavior                                            |
| -------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Fixed-rank     | `__init__` (all dimensions provided)                               | Called once during init; result may be stored on the Op.              |
| Arbitrary-rank | `__init__` for `static_dims`; `forward` for remaining dynamic dims | Called in `forward()` when dynamic vars are resolved; cache by input. |

Non-runtime consumers must instantiate the Op (or read pre-computed `(flops, bytes)` from benchmark output). No manifest-level roofline evaluator exists; every value flows through `op.eval_roofline()`.

#### 4.4.6 Evaluator Surface Boundary

Roofline expressions live in one place at runtime: the plain Python body that codegen emits into each op's `eval_roofline()`, or the method an op defines for itself under §4.4.1. No standalone roofline evaluator exists, and neither surface parses a formula string.

| Surface                           | Scope | Interprets roofline expressions? |
| --------------------------------- | ----- | -------------------------------- |
| Op-local AST evaluator            | —     | **REJECTED** — must not be built |
| Manifest-level roofline evaluator | —     | **REJECTED** — must not be built |

Rules:

- No `tileops.manifest.eval_roofline()` / `resolve_roofline_vars()` helper that evaluates roofline expressions exists. Any consumer wanting `(flops, bytes)` either calls `op.eval_roofline()` on an Op instance or reads pre-computed values from benchmark output.
- Generated `eval_roofline()` must not parse, AST-analyze, or safe-eval its own formula strings. Codegen does the name/form check at generation time (§4.4.3 / §4.4.4) and then copies validated expressions into plain Python.
- If a formula is too complex for inline arithmetic (conditionals, shape traversal, data-dependent logic), switch the entry to `func` mode (§2.2). Do not extend inline formulas into a mini-language.

### 4.5 Bytes Audit (NCU)

`scripts/validate_roofline_bytes.py` compares an op's `bytes` formula against DRAM counters. Its verdict covers the read side only: writes still resident in L2 when the kernel ends fall outside the profiled range, so the write side is measured and reported but never judged.

The read-side bound is conditional, not a theorem. It holds while each kernel is replayed from cold caches, which inflates a multi-kernel op's reads rather than deflating them; a verdict states that premise alongside it.

It carries a second premise: that every conforming implementation must fetch what the formula charges. Some calls break it. Where an input's value decides nothing at some positions, a kernel may predicate those loads away and read less than the call binds, while the formula still charges the whole input — the positions are chosen at run time, and charging a fraction of them would be an expected value, not this call's traffic (§4.7).

An entry states such calls in `roofline.read_bound_exception`: a `when` over the call, and the `reason` the premise fails there. The audit evaluates `when` against the row it measured, and a shortfall inside the condition is EXEMPT — measured, reported, not a verdict on the formula. Rows outside it are judged as before, which is why the exception carries a condition rather than covering the op.

The condition's form is checked, its aptness is not: no check tells a property of the call from a value that matches today's rows. Review reads the `reason`, and an entry earns the exception from a measurement of the behaviour it names.

`(flops, bytes)` does not carry the read/write split, so an op sent here states its read half in `Op.eval_roofline_read_bytes()`. There is no fallback. Summing the call's input tensors is not the read half: an op that reads a subset of an input — a routed MoE reading the experts its routing selects — would be charged the whole of it, and a correct formula would fail. An op that declares nothing gets NO-VERDICT, which is not a pass.

| Verdict    | Meaning                                                                |
| ---------- | ---------------------------------------------------------------------- |
| FAIL       | Measured reads fall short of the declared read half.                   |
| WARN       | Measured reads far exceed it: multi-pass or replay cost.               |
| EXEMPT     | They fall short inside a `read_bound_exception`: reported, not judged. |
| SKIPPED    | Never run, or the formula declares no read at all.                     |
| ERROR      | The audit did not produce a usable measurement.                        |
| NO-VERDICT | No read half was declared.                                             |

The read half comes off `bytes` by subtracting the write half the contract settles, and pricing the outputs needs the shapes the call carried. An op keeps only what its own `eval_roofline` needs, so `tileops.ops.op_base.record_roofline_calls()` makes `Op.__call__` remember each input tensor's shape and dtype. The audit turns it on around the call it reads the declaration off, and it is off everywhere else: it costs about a microsecond per call, which a benchmark row would otherwise carry.

Workloads come from the manifest's own rows and cover the formula's branch signatures. Scaled-up shapes are not used: they can cross kernel-selection thresholds and audit an implementation the benchmark never runs.

Runs on demand. It needs GPU performance counters, which the driver restricts to admin by default and which a rootless container cannot obtain however privileged it is.

### 4.6 Structural Oracle (tests)

A CI test recomputes each audited `bytes` value from an independent path — the sizes of the tensors the workload actually binds (each distinct input storage once, each output once) — and requires equality with `eval_roofline()`. The formula and the oracle share only the minimum-traffic definition, so a coefficient slip, a missed output, a wrong `elem_bytes`, or a broadcast counted at the wrong shape breaks the equality.

Traffic that depends on tensor *content* is recounted the same way: the case constructs the selecting tensor itself, exactly as it constructs shapes, so content dependence is no reason to exempt an op. Coverage is golden workloads per op, not randomized sweeps.

Coverage is three levels and an op sits at exactly one:

| Level | Case                                                                                                                                                            | Shares with the formula                                                                                                                           |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| One   | A binder builds it from the signature, one workload row, the dtypes and the mutation marks. An op reaches this level by being recountable, not by being listed. | The minimum-traffic definition, the op's statement of its output extents, and the manifest's output-dtype resolution. Never the `roofline` block. |
| Two   | Hand-written, for a call the contract does not settle.                                                                                                          | Written beside the case.                                                                                                                          |
| Three | None: marked with what is missing, asserted against nothing.                                                                                                    | —                                                                                                                                                 |

A completeness test keeps the three total: an op added to the manifest is recounted by the binder or fails until it is placed.

### 4.7 Value-Determined Traffic

A few ops move an amount their inputs' values decide: a routed MoE reads the experts `topk_ids` names, a sparse attention the blocks its selection kept. Such a formula prices this call, not an average over calls of that shape, which imposes three rules.

- It reads the call's semantic inputs — the routing, the offsets, the block table — and never a quantity it computed for itself, which would make a recount an identity.
- Its workload row builds those inputs from a generator of its own, not the global stream: a draw added anywhere upstream would otherwise move the traffic and the efficiency the row reports.
- It states what decided the number in `Op.roofline_inputs()`, which the benchmark records beside the reading. Nothing judges it; it is what makes a moved number readable.

A recount builds the call's two kinds of tensor: bulk operands on the meta device, and the metadata whose values decide the traffic with those values. Where the row states them — a packed batch's lengths — the recount restates them and stays at level one. Where the values are drawn per call, the recount runs the workload that draws them, and the op sits at level two.

## 5. Reference

### 5.1 GPU Profile

Hardware parameters use theoretical values with calibration factors from one-time microbenchmark measurements. A bandwidth calibration is the **envelope** over the measured access mixes (copy, Triad, pure read, pure write): a ceiling some legitimate mix can exceed is not a ceiling, and readings above 100% must stay reserved for formula errors; each mix's own measured fraction is kept as data (`calibration_mixes`), so a future per-mix ceiling reads it instead of re-measuring. YAML files store only measured values; `effective = theoretical × calibration` is computed by `load_profile()`:

```yaml
# src/tileops/perf/profiles/<gpu>.yaml
hbm:
  theoretical: 4800e9       # bytes/s, from spec sheet
  calibration: 0.938        # microbench envelope over access mixes
tensor_core:
  fp16:
    theoretical: 989.5e12   # FLOPS, from spec sheet
    calibration: 0.75       # from microbench (cuBLAS peak)
```

Profiles are stored in `src/tileops/perf/profiles/`. Microbenchmarks for calibration live in `benchmarks/hardware/`.

### 5.2 Benchmark–Roofline Decoupling

Benchmark (M4) produces per-workload records containing raw time and the `(flops, bytes)` from `op.eval_roofline()`. Roofline (M5) is a separate tool that reads those records + GPU profile to compute efficiency. This separation enables:

- Re-analyzing historical data when GPU profiles are updated
- Multiple consumers of raw benchmark data (roofline, regression detection, dashboards)
- Benchmark module has no third-party dependencies beyond the project itself
