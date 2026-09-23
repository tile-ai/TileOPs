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
- `actual_time` — benchmark output (§5.2): device-busy time plus any device copies excluded from it.

Bound type is whichever term dominates `sol_time`; a tie is memory-bound. It depends on shape, not on the op; the roofline tool computes it per-workload and the manifest does not declare it.

The metric is **algorithmic** SOL efficiency. Four statements delimit what a reading means:

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
- The base default covers every op whose arithmetic runs on CUDA cores in fp32 (elementwise, reductions, norms, scans). An op whose FLOPs are matmul contractions overrides it; one whose unit depends on instance state (a backend switch, a quantized path) branches on that state.
- The declaration is valid whenever `eval_roofline()` is — after the op's dtype is bound.
- A wrong or missing override prices the op against the wrong ceiling. The physics check (§4.3) reports it once the implied rate breaches that ceiling, which a kernel far from its own roof does not reach.

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

**Func.** Point at `tileops.perf.formulas.<name>`. The callable is human-authored and returns `(flops, bytes)`. **Recommended signature: `func(op)`** — matching the agent-generated `eval_roofline(self)` path, which is what codegen's emitted call assumes. A human author who prefers a different signature owns the resulting integration (e.g., a wrapper). Use `func` when inline arithmetic is insufficient (mixed-precision byte accounting, shape traversal, data-dependent logic).

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

`src/tileops/manifest/` is the source of truth for the `roofline` field. Five modules read it:

- **Roofline analysis** — reads an entry once and answers two questions: every defect it carries, and whether it can be emitted. Owns the name and form rules. Spec: §4.4.
- **Schema validator / CI** — structural checks (schema, mode exclusivity, `func` importability), and it renders the analysis's defects. Does **not** execute formulas or hold a helper whitelist. Spec: §4.1.
- **Benchmark layer** — instantiates an Op per workload and reads `(flops, bytes)` from `op.eval_roofline()`. Hardcoded formulas in benchmark files are a CI failure. Spec: §4.2.
- **Roofline tool (M5)** — reads per-workload `(flops, bytes)`, the roof key, and timing from benchmark output, prices them against the GPU profile (§5.1), and emits SOL efficiency and verdicts. Spec: §4.3.
- **Op codegen** — emits the `eval_roofline()` method of every implemented entry (§4.4.1) from what the analysis decided. Judges nothing. Spec: §4.4.

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
- An implemented entry's formula analyses cleanly. The validator does not judge the formula: it renders what the analysis found, which for a `func` entry covers whether the dotted path resolves.

Rules the validator does not own:

- Name whitelist — a formula's names are checked by the analysis (§4.4), which owns the binding table. Validator does not mirror it; it renders what the analysis says.
- Form checks — the analysis refuses invalid forms. Validator does not mirror them either; it renders what the analysis found.
- Numeric checks (finite / non-negative / numeric) — outside the validator entirely; tests exercise generated `eval_roofline()` on each workload.

Validator holds no helper callables, no sample bindings, no `__builtins__` sandbox. Adding a helper does not touch the validator.

### 4.2 Benchmark Layer

Contract:

- Instantiate the Op for each workload and call `op.eval_roofline()` to obtain `(flops, bytes)`; there is nowhere else to get them (§4.4.6).
- Non-reserved workload keys forward as op-call params to the Op's constructor.
- A benchmark file that computes FLOPs or bytes locally is a CI failure.
- Benchmark output must record the `(flops, bytes)` from `op.eval_roofline()` and the roof key from `op.compute_roof()` (§1.4), so M5 reads the numbers without re-instantiating ops.

### 4.3 Roofline Tool (M5)

Inputs:

- Benchmark output produced by M4, carrying per-workload timing, the `(flops, bytes)` from `op.eval_roofline()`, and the roof key from `op.compute_roof()`.
- GPU profile (§5.1), selected by matching the profile's `gpu` field against the measured device name; no match leaves every SOL reading blank.

Per-workload outputs: SOL efficiency, bound type, latency-bound labels, anomaly reports.

M5 reads pre-computed numbers and never instantiates an Op (§4.4.6).

Verdict lines are rendering thresholds, not CI gates:

| Verdict    | Condition | Meaning                                                                                                                                        |
| ---------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| At ceiling | ≥ 80%     | Done. The HBM ceiling is an envelope over access mixes and a kernel's own mix caps below it; the line sits below every mix's personal ceiling. |
| Anomaly    | > 105%    | Above the achievable ceiling: the formula or the calibration is wrong. Excluded from "at ceiling".                                             |

Physics check: every row's implied rates (`bytes / time`, `flops / time`) are compared against the *theoretical* ceilings of its roofs. A breach is physically impossible, so it is reported as a formula error that fails the run's health — a formula edit that inflates work is caught on the next nightly. This check is the standing guard on formula overestimation; equality-level validation belongs to the structural oracle (§4.6), read-side hardware-counter validation to the bytes audit (§4.5).

### 4.4 Op Codegen

Analysis and emission both run for `status: implemented` entries only. `spec-only` entries — where either the implementation does not exist or the Op interface does not yet match the manifest — are skipped, and are re-read once the status flips.

The analysis is the authoritative gate for name and form correctness. A formula referencing an unknown name or violating a layer's form constraints fails it, and a manifest that fails it cannot land. Numeric correctness is exercised by tests.

Analysis and emission are separate: analysis reads the entry and decides, emission writes the method from those decisions and decides nothing. Three properties follow, and each is a rule:

- **Total.** Analysis accepts whatever YAML produced and never raises on it. A defect is a verdict, not an exception.
- **Lossless.** A fact is absent, malformed or valid, and the three are distinguished. Collapsing the first two makes a missing dependency indistinguishable from a satisfied one.
- **Accumulating.** A defect does not stop the pass. A judgment that cannot be reached for want of a fact is recorded as unreached, naming the fact.

Whether a defect stops emission follows the formula rather than the defect: a malformed `outputs` blocks a formula that reads `out_elem_bytes` and not one that never does. A name the formula reads resolves from one place only — declared twice, or shared with a helper, it would bind twice in the emitted body.

The two gates divide by question, not by field. §4.1 rules on whether the blocks are structurally
what the spec says; the analysis rules on whether the formula is legal. Neither withholds its answer because the
other has one, so a formula defect is reported however the rest of the entry reads: a precondition
wide enough to suppress the overlap also suppresses a defect that merely sits beside an unrelated
one. The exception is a signature too malformed to say what names the formula may use, where the
structural verdict is the only one there is.

A rejection is a verdict rather than an exception for a caller to classify.

An inline entry is decided from the entry alone — no op instance, no tensor library, no device — which is what lets the validator (§4.1) ask the question wherever the manifest can be read. A `func` entry additionally imports the module its path names, so what that module needs at import, deciding the entry needs too. A formula callable that pulls a runtime into an import therefore costs the manifest a check it could otherwise run anywhere.

#### 4.4.1 Generated Method

Every implemented manifest entry is served by a generated `eval_roofline()` returning `(flops: int, bytes: int)`. The method belongs to that entry: a subclass with its own entry receives its own evaluator rather than inheriting another entry's formula. The signature is part of the shared Op interface defined in [ops-design-reference.md](ops-design-reference.md).

A tensor the formula reads resolves through `self.<name>` or `self.<name>_shape`, so an op exposes what its entry reads and nothing more. Exposing neither is an authoring defect; exposing one and leaving it unset is a caller who has not run `forward()`. The two are distinct failures and are reported as such.

#### 4.4.2 Manifest Inputs

For each entry, codegen reads one of:

- **Inline** — `vars` (optional), `flops`, `bytes`, all Python expression source strings, emitted as the method body per §4.4.3.
- **Func** — `func`, a dotted path to `func(op) -> tuple[int, int]`, emitted as a call to it. The callable reads construction-bound and call-bound state off the op; call-bound state wins where both supply the same input.

#### 4.4.3 Expression Layers

Inline mode has two layers, emitted as two sequential blocks.

- **vars layer** — shape-derived resolution. Tensor shape access, slicing, `product()`, `range()`, small comprehensions. Entries resolve in declaration order and each may read the ones above it, so a formula names an intermediate once and builds on it.
- **arithmetic layer** — `flops` and `bytes` over the resolved variables, the element-size constants and approved helpers only. No tensor access, shape slicing, comprehensions, attributes or arbitrary calls.

The layers are what keeps a formula readable and recountable, so a vars expression must not be inlined into an arithmetic expression: that collapses the two and puts shape traversal where the arithmetic layer forbids it. A formula the arithmetic layer cannot carry switches to `func` mode (§2.2) rather than extending inline formulas into a mini-language.

Both expression strings are copied verbatim into plain Python. The analysis resolves every name against the namespace (§4.4.4) and checks the arithmetic layer's form; a formula failing either does not land (§4.1).

Reduction dim handling in the vars layer follows the manifest `shape_rules` contract: validate range, normalize against `ndim`, reject duplicate axes for sequence dims. A roofline expression must not silently normalize an invalid axis.

#### 4.4.4 Namespace

One binding table is the single source of truth for what an inline formula may reference, and a refusal states the allowed names. The buckets:

| Layer      | May reference                                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------------------------------------- |
| vars       | `signature.inputs` names by `.shape`, `signature.params` names, the element-size constants, shape and sequence helpers |
| arithmetic | variables resolved by the vars layer, the element-size constants, numeric helpers                                      |

Two element-size constants exist. `elem_bytes` prices the input dtype; `out_elem_bytes` resolves the declared output dtype through the manifest, so an op whose output dtype is not its input's — a bool predicate, an integral input promoted to float — states its write without a second source. It is available where the entry declares exactly one output.

A helper is added or removed in the binding table and nowhere else.

#### 4.4.5 Evaluation Timing

`eval_roofline()` is valid once the call has bound what the formula reads. The dtype is always call-bound, so no op can be priced before its first `forward()`; an arbitrary-rank op's dynamic dims are bound there too. The method recomputes on each call and holds no cache: a cached `(flops, bytes)` would outlive the shapes it was computed for.

A consumer that is not the op itself instantiates the Op or reads pre-computed `(flops, bytes)` from benchmark output.

#### 4.4.6 Evaluator Surface Boundary

Roofline expressions live in exactly one place at runtime: the plain Python body of each op's `eval_roofline()`. Two surfaces are rejected and must not be built — an op-local AST evaluator, and a manifest-level roofline evaluator that any consumer could call for `(flops, bytes)`.

Neither the generated body nor anything else parses, AST-analyzes or evaluates a formula string at run time. The name and form check happens once, before emission, which copies the checked expressions into plain Python.

### 4.5 Bytes Audit (NCU)

The bytes audit compares an op's `bytes` formula against DRAM counters. Its verdict covers the read side only: writes still resident in L2 when the kernel ends fall outside the profiled range, so the write side is measured and reported but never judged.

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

The read half comes off `bytes` by subtracting the write half the contract settles, and pricing the outputs needs the shapes the call carried. An op keeps only what its own `eval_roofline` needs, so the audit records each input's shape and dtype around the call it reads the declaration off. The recording is off everywhere else: it costs per call, which a benchmark row must not carry.

Workloads come from the manifest's own rows and cover the formula's branch signatures. Scaled-up shapes are not used: they can cross kernel-selection thresholds and audit an implementation the benchmark never runs.

Runs on demand: it needs GPU performance counters, which the driver restricts to admin.

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

Hardware parameters are theoretical values with calibration factors from one-time microbenchmark measurements: `effective = theoretical × calibration`. A profile stores the two inputs, never the derived value, and is selected by matching the device name.

A bandwidth calibration is the **envelope** over the measured access mixes (copy, Triad, pure read, pure write). A ceiling some legitimate mix can exceed is not a ceiling, and readings above 100% must stay reserved for formula errors. Each mix's own measured fraction is kept as data, so a future per-mix ceiling reads it rather than re-measuring.

### 5.2 Benchmark–Roofline Decoupling

Benchmark (M4) produces per-workload records containing raw time and the `(flops, bytes)` from `op.eval_roofline()`. Roofline (M5) is a separate tool that reads those records + GPU profile to compute efficiency. This separation enables:

- Re-analyzing historical data when GPU profiles are updated
- Multiple consumers of raw benchmark data (roofline, regression detection, dashboards)
- Benchmark module has no third-party dependencies beyond the project itself
