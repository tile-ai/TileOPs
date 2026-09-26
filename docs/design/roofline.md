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

1. `bytes_moved` is the algorithm's minimum traffic, not measured DRAM traffic: each distinct input storage the algorithm reads counts one read, each public output one write, a `mutated` input on a branch where it is written both, a `write_only` input one write, and a declared alias one storage. An intermediate never counts, whatever stage produces it, and any other input the algorithm does not read produces no traffic.
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

| Mode   | Form                      | When                            |
| ------ | ------------------------- | ------------------------------- |
| Inline | `flops`, optional `bytes` | The formula is an expression.   |
| Func   | `func: "module.path"`     | The formula needs Python logic. |

**Inline.** `flops` and `bytes` are expressions in the manifest expression language ([manifest.md](manifest.md#t-lang)) over `ix` — the signature's indices, construction parameters and `let`, with `present(t)` for presence; `forall` value lists are not in `ix` — plus `bytes(t)`, the byte size of tensor `t`, and the built-in primitives. A cost that varies with presence reads `present(...)`.

**Derived `bytes`.** An entry omitting `bytes` is charged each tensor read or written once: every tensor argument is its own storage, and a declared alias (`buffer`, `alias`) is one; an input not written counts a read, an output a write, a `mutated` input both, a `write_only` input a write; a tensor's size is `prod(shape) * bits(dtype) / 8`, packed dtypes by carrier. An entry whose traffic differs writes `bytes` and a test for it.

**Func.** `tileops.perf.formulas.<name>`, a module-level function `f(call) -> tuple[int, int]` over the checked call and nothing else: every signature parameter and `call.ix` — the same `ix` an inline formula reads, with the indices, dtype indices and `let`s the call's branch resolves — presence, each tensor's shape, dtype and `bytes(t)`, and the metadata tensors whose values decide traffic (§4.7). A formula introduces no signature dependency and never reads the op; a fact it needs that the call lacks means the call model is incomplete.

```yaml
roofline:
  flops: "2 * M * N * K"     # bytes derived from the signature
# or
roofline:
  func: "tileops.perf.formulas.gqa_fwd_roofline"
```

## 3. Consumers

`src/tileops/manifest/` is the source of truth for the `roofline` field. These modules read it:

- **Roofline analysis** — reads an entry once and answers two questions: every defect it carries, and whether it can be emitted. Owns the name and form rules. Spec: §4.4.
- **Schema validator / CI** — structural checks (schema, mode exclusivity, `func` importability), and it renders the analysis's defects. Does **not** execute formulas or hold a helper whitelist. Spec: §4.1.
- **Benchmark layer** — instantiates an Op per workload and reads `(flops, bytes)` from `op.eval_roofline()`. Hardcoded formulas in benchmark files are a CI failure. Spec: §4.2.
- **Roofline tool (M5)** — reads per-workload `(flops, bytes)`, the roof key, and timing from benchmark output, prices them against the GPU profile (§5.1), and emits SOL efficiency and verdicts. Spec: §4.3.
- **Op codegen** — emits the `eval_roofline()` method of every implemented entry (§4.4.1) from what the analysis decided. Judges nothing. Spec: §4.4.

Two auditors check the field's values rather than consume them: the structural oracle (§4.6) and the NCU bytes audit (§4.5).

Tests and workloads are not consumers: they may supply shapes and dtypes but must not define or reinterpret roofline formulas.

## 4. Consumer Specifications

### 4.1 Schema Validator / CI

Scope is structural.

Every roofline entry MUST satisfy:

- Required fields per mode: inline has `flops`; func has `func`.
- Mode exclusivity: `flops`/`bytes` and `func` do not coexist.
- Field types: `flops`/`bytes`/`func` are non-empty strings.
- Every entry's formula analyses cleanly. The validator does not judge the formula: it renders what the analysis found. Whether a `func` path resolves is checked for implemented entries.

Rules the validator does not own:

- Name whitelist — a formula's names are checked by the analysis (§4.4), which reads the primitive tables of [manifest.md](manifest.md#t-prims). Validator does not mirror it; it renders what the analysis says.
- Form checks — the analysis refuses invalid forms. Validator does not mirror them either; it renders what the analysis found.
- Numeric checks (finite / non-negative / numeric) — outside the validator entirely; tests exercise generated `eval_roofline()` on each workload.

The validator keeps no copy of the primitives: it calls their one shared implementation, so adding a primitive changes no validator code.

### 4.2 Benchmark Layer

Contract:

- Instantiate the Op for each workload and call `op.eval_roofline()` to obtain `(flops, bytes)`; there is nowhere else to get them (§4.4.5).
- Each workload row is instantiated as [manifest.md § Workloads](manifest.md#workloads) defines: its `signature.params` values construct the Op, its indices, `some` and `dtype_cases` build the call's tensors, and `label` names the case.
- A benchmark file that computes FLOPs or bytes locally is a CI failure.
- Benchmark output must record the `(flops, bytes)` from `op.eval_roofline()` and the roof key from `op.compute_roof()` (§1.4), so M5 reads the numbers without re-instantiating ops.

### 4.3 Roofline Tool (M5)

Inputs:

- Benchmark output produced by M4, carrying per-workload timing, the `(flops, bytes)` from `op.eval_roofline()`, and the roof key from `op.compute_roof()`.
- GPU profile (§5.1), selected by matching the profile's `gpu` field against the measured device name; no match leaves every SOL reading blank.

Per-workload outputs: SOL efficiency, bound type, latency-bound labels, anomaly reports.

M5 reads pre-computed numbers and never instantiates an Op (§4.4.5).

Verdict lines are rendering thresholds, not CI gates:

| Verdict    | Condition | Meaning                                                                                                                                        |
| ---------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| At ceiling | ≥ 80%     | Done. The HBM ceiling is an envelope over access mixes and a kernel's own mix caps below it; the line sits below every mix's personal ceiling. |
| Anomaly    | > 105%    | Above the achievable ceiling: the formula or the calibration is wrong. Excluded from "at ceiling".                                             |

Physics check: every row's implied rates (`bytes / time`, `flops / time`) are compared against the *theoretical* ceilings of its roofs. A breach is physically impossible, so it is reported as a formula error that fails the run's health — a formula edit that inflates work is caught on the next nightly. This check is the standing guard on formula overestimation; equality-level validation belongs to the structural oracle (§4.6), read-side hardware-counter validation to the bytes audit (§4.5).

### 4.4 Op Codegen

Analysis runs for every entry. Emission runs for `status: implemented` entries only.

An inline entry is decided from the entry alone — no op instance, no tensor library, no device — which is what lets the validator (§4.1) ask the question wherever the manifest can be read. A `func` path is imported only for an implemented entry.

The analysis is the authoritative gate for name and form correctness. A formula referencing an unknown name or violating a layer's form constraints fails it, and a manifest that fails it cannot land. Numeric correctness is exercised by tests.

Analysis and emission are separate: analysis reads the entry and decides, emission writes the method from those decisions and decides nothing. Three properties follow, and each is a rule:

- **Total.** Analysis accepts whatever YAML produced and never raises on it. A defect is a verdict, not an exception.
- **Lossless.** A fact is absent, malformed or valid, and the three are distinguished. Collapsing the first two makes a missing dependency indistinguishable from a satisfied one.
- **Accumulating.** A defect does not stop the pass. A judgment that cannot be reached for want of a fact is recorded as unreached, naming the fact.

#### 4.4.1 Generated Method

Every implemented manifest entry is served by a generated `eval_roofline()` returning `(flops: int, bytes: int)`. The method belongs to that entry: a subclass with its own entry receives its own evaluator rather than inheriting another entry's formula. It is emitted per discriminant point, like the call checks, and evaluates over the op's last completed call. The signature is part of the shared Op interface defined in [ops-design-reference.md](ops-design-reference.md).

#### 4.4.2 Manifest Inputs

Codegen reads the formula mode of §2.2: an inline body folded at each point, with `bytes` derived from the signature when omitted, or a call to `func` with the checked call.

#### 4.4.3 Namespace

The primitive tables of [manifest.md](manifest.md#t-prims) are the only list of names an inline formula may call (§2.2); a primitive is added there and in its one implementation, and nowhere else. A refusal states the allowed names. A name resolves from one place only.

#### 4.4.4 Evaluation Timing

`eval_roofline()` is valid once a call has completed; it prices `Op.last_call`, part of the `Op` base class interface in [ops-design-reference.md](ops-design-reference.md). The dtype is always call-bound, so no op can be priced before its first `forward()`; an arbitrary-rank op's dynamic dims are bound there too. The method recomputes on each call and holds no cache: a cached `(flops, bytes)` would outlive the shapes it was computed for. A call on meta tensors holds no values; a formula that reads metadata values raises `OpNotAvailableError` on it.

A consumer that is not the op itself instantiates the Op or reads pre-computed `(flops, bytes)` from benchmark output.

#### 4.4.5 Evaluator Surface Boundary

Roofline expressions live in exactly one place at runtime: the plain Python body of each op's `eval_roofline()`. Two surfaces are rejected and must not be built — an op-local AST evaluator, and a manifest-level roofline evaluator that any consumer could call for `(flops, bytes)`.

Neither the generated body nor anything else parses, AST-analyzes or evaluates a formula string at run time. The name and form check happens once, before emission, which copies the checked expressions into plain Python.

### 4.5 Bytes Audit (NCU)

The bytes audit compares an op's `bytes` formula against DRAM counters. Its verdict covers the read side only: writes still resident in L2 when the kernel ends fall outside the profiled range, so the write side is measured and reported but never judged.

The read-side bound is conditional, not a theorem. It holds while each kernel is replayed from cold caches, which inflates a multi-kernel op's reads rather than deflating them; a verdict states that premise alongside it.

It carries a second premise: that every conforming implementation must fetch what the formula charges. Some calls break it. Where an input's value decides nothing at some positions, a kernel may predicate those loads away and read less than the call binds, while the formula still charges the whole input — the positions are chosen at run time, and charging a fraction of them would be an expected value, not this call's traffic (§4.7).

Such calls are configured in the audit script, not the manifest: a `when` over the call, and the `reason` the premise fails there. The audit evaluates `when` against the row it measured, and a shortfall inside the condition is EXEMPT — measured, reported, not a verdict on the formula. Rows outside it are judged as before, which is why the exception carries a condition rather than covering the op.

The condition's form is checked, its aptness is not: no check tells a property of the call from a value that matches today's rows. Review reads the `reason`, and an entry earns the exception from a measurement of the behaviour it names.

`(flops, bytes)` does not carry the read/write split, so an op sent here states its read half in `Op.eval_roofline_read_bytes()`. There is no fallback. Summing the call's input tensors is not the read half: an op that reads a subset of an input — a routed MoE reading the experts its routing selects — would be charged the whole of it, and a correct formula would fail. An op that declares nothing gets NO-VERDICT, which is not a pass.

| Verdict    | Meaning                                                              |
| ---------- | -------------------------------------------------------------------- |
| FAIL       | Measured reads fall short of the declared read half.                 |
| WARN       | Measured reads far exceed it: multi-pass or replay cost.             |
| EXEMPT     | They fall short inside a configured exception: reported, not judged. |
| SKIPPED    | Never run, or the formula declares no read at all.                   |
| ERROR      | The audit did not produce a usable measurement.                      |
| NO-VERDICT | No read half was declared.                                           |

Workloads come from the manifest's own rows and cover the formula's branch signatures. Scaled-up shapes are not used: they can cross kernel-selection thresholds and audit an implementation the benchmark never runs.

### 4.6 Structural Oracle (tests)

A CI test recomputes each audited `bytes` value from an independent path — the sizes of the tensors the workload actually binds, counted by the effect rules of §2.2 without reading the `roofline` block — and requires equality with `eval_roofline()`. The formula and the oracle share only the minimum-traffic definition, so a coefficient slip, a missed output, a wrong `elem_bytes`, or a broadcast counted at the wrong shape breaks the equality.

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

- It reads the checked call's metadata tensors — the routing, the offsets, the block table — and never a quantity the op computed for itself, which would make a recount an identity.
- Its workload row builds those inputs from a generator of its own, not the global stream: a draw added anywhere upstream would otherwise move the traffic and the efficiency the row reports.
- It states what decided the number in `Op.roofline_inputs()`, which the benchmark records beside the reading. Nothing judges it; it is what makes a moved number readable.

Where the row states the deciding values — a packed batch's lengths — the recount restates them and stays at level one. Where the values are drawn per call, the recount runs the workload that draws them, and the op sits at level two.

## 5. Reference

### 5.1 GPU Profile

Hardware parameters are theoretical values with calibration factors from one-time microbenchmark measurements: `effective = theoretical × calibration`. A profile stores the two inputs, never the derived value, and is selected by matching the device name.

A bandwidth calibration is the **envelope** over the measured access mixes (copy, Triad, pure read, pure write). A ceiling some legitimate mix can exceed is not a ceiling, and readings above 100% must stay reserved for formula errors. Each mix's own measured fraction is kept as data, so a future per-mix ceiling reads it rather than re-measuring.

### 5.2 Benchmark–Roofline Decoupling

Benchmark (M4) produces per-workload records containing raw time and the `(flops, bytes)` from `op.eval_roofline()`. Roofline (M5) is a separate tool that reads those records + GPU profile to compute efficiency. This separation enables:

- Re-analyzing historical data when GPU profiles are updated
- Multiple consumers of raw benchmark data (roofline, regression detection, dashboards)
- Benchmark module has no third-party dependencies beyond the project itself
