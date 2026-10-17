# Op Interface Design — Reference

The contracts an op is built against: base-class attributes, family protocol variables, naming, parameter design, calling conventions, and what CI enforces.

## Slot Rules

The per-slot codegen rules live with their consumer:
[`op-slot-rules.md`](op-slot-rules.md).
This document holds the contracts those rules emit against.

## Family-Base Protocol (Appendix) <a id="base-class-protocol"></a>

Per-family protocol variables, declared by L2 bases and overridden by L3 ops.

| Variable   | Family    | Purpose                                                                                                          |
| ---------- | --------- | ---------------------------------------------------------------------------------------------------------------- |
| `_op_kind` | reduction | Kernel-dispatch op-kind string (`"sum"` / `"prod"` for `CumulativeOp`; `"sum"`, `"mean"`, … for `_ReduceOpBase`) |

**The scaffolding playbook does NOT emit these variables** — kernel-dispatch-convention-dependent (e.g., `VectorNormKernel` uses `{"l1", "l2", "inf"}`, `ReduceKernel` uses `{"sum", "mean", ...}`); Adding a new protocol variable requires updating the L2 base and all concrete ops.

### `Op` base class interface ([`src/tileops/ops/op_base.py`](../../src/tileops/ops/op_base.py))

Abstract interface: `forward()`, the op's computation and the one method every op writes. Construction: `Op.__init__(*, target=None)`, which a subclass calls once its manifest parameters are assigned; it stores the target, checks the parameters and installs the implementations. Methods generated from the manifest entry: the construction and call checks, `_infer_output_shapes`, `eval_roofline`, and, for an entry with a call-time tensor input and no composition, the compile boundary `_call_boundary` with the operators `compile_op_names` names. `__call__` enters `_call_boundary` where the op has one, and runs the same call steps around `forward` otherwise.

- `kernel_types` (class attribute) is the one declaration of an op's in-tree dispatch keys. Every instance installs all of them, together with the implementations a backend added through `register_kernel_type(op, key, kernel_type)` under keys the op does not already have. `interfaces` (class attribute) maps each place the op calls a kernel to its kernel interface.
- `delegate_types` (class attribute) is the one declaration of the sub-ops an op may hold: stage name to op class, in stage order. Default empty.
- `last_call` (property) is the `SignatureCall` of the op's last successfully completed call: its `indices`, tensors, effects, metadata tensors, and the checked calls its sub-ops completed during it, by stage. It raises `RuntimeError` before one completes. `eval_roofline` prices it.

#### Kernel caching and enumeration methods

Rationale and the interface / entry vocabulary: [ops-design.md § Kernel caching and enumeration](ops-design.md#kernel-caching-and-enumeration).

| Method                          | Purpose                                                                                                                                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kernel_for(interface, call)`   | The in-tree entry serving this call, resolved once per call spec. The only way an op's in-tree implementation reaches a kernel; a target serves the whole op     |
| `key_for(interface, call)`      | The key of the interface's implementation that serves the call. Introspection and tests; `kernel_for` asks it on a miss                                          |
| `built_entries(interface)`      | Read-only view of an interface's entries, whoever built them; empty before its first build. Introspection only, never dispatch                                   |
| `delegate_for(stage, identity)` | The sub-op held for a stage and identity, built once on a miss with the op's `target`, or the injected one. The only way an op holds a sub-op                    |
| `held_delegates()`              | The sub-ops `delegate_for` holds, in stage order. Derived; never overridden                                                                                      |
| `iter_kernels()`                | The TileOPs `Kernel` instances the entries hold, deduplicated: the op's interface entries and those of its held sub-ops; a target's kernels are not among them   |
| `kernel_config()`               | The configuration the op's kernels were built with, a target's included, or `None`                                                                               |
| `serving_target`                | The target serving the instance: `None` until a call selects it, `BUILTIN` for the in-tree implementation, else the target's name                                |
| `request_tune()`                | The one way into tuned mode: tunes the built kernels and held sub-ops, and every entry and sub-op built later is tuned as it is built; so are a target's kernels |

### `Kernel` base class attributes ([`src/tileops/kernels/kernel_base.py`](../../src/tileops/kernels/kernel_base.py))

Unlike `Op`, a `Kernel` **is** constructed for one element type — it compiles a dtype-specialized program, so `dtype` is a ctor argument here. The op supplies it from the tensors at `forward()`.

| Attribute                            | Type                    | Purpose                                                             |
| ------------------------------------ | ----------------------- | ------------------------------------------------------------------- |
| `dtype`                              | `Optional[torch.dtype]` | Element type this kernel is specialized for                         |
| `config`                             | `Dict[str, Any]`        | Tile configuration (block sizes, stages, etc.)                      |
| `autotune_configs`                   | `Optional[list[dict]]`  | Search space for autotuning; the untuned config is timed with it    |
| `supported_archs`                    | `Optional[list[int]]`   | Architectures of its one device type, e.g. SM `[80, 86, 89, 90]`    |
| `kernel`                             | `Callable`              | Compiled TileLang kernel function                                   |
| `autotune_accepts_random_int_inputs` | `bool`                  | Whether autotuning may generate the integer tensor inputs at random |
| `devices`                            | `frozenset[str]`        | Device types the implementation runs on; default `{"cuda"}`         |
| `general`                            | `bool`                  | Below every other implementation of its interface; at most one      |
| `preferred_over`                     | `frozenset[str]`        | Keys of the implementations it wins over, transitively              |

Abstract interface: `forward()`. Key methods: `init_config(config)`, `request_tune()`, `autotune(warmup, rep)`, and `tuning_candidates(configs, untuned)`, which every tuning loop times, so tuning never keeps a config it measured slower than the one the kernel runs untuned. Selection classmethods, each with a default ([ops-design.md § Kernel selection](ops-design.md#kernel-selection)): `refusal(call)` returns `None` for a call the implementation serves and the limit it hits otherwise, `unavailable(call)` reads `devices` and `supported_archs`, and `entry_for(call)` returns the build identity and the factory.

## Optional Hooks (Appendix)

Hooks family bases expose for op-specific semantics. The scaffolding playbook does NOT emit these.

A restriction on the accepted domain is a refinement of the signature, never a hook. A hook that compensates for what a kernel cannot do belongs to that kernel, not here: the op hands over the tensor its manifest declares.

| Hook              | Family    | Purpose                                                                                      |
| ----------------- | --------- | -------------------------------------------------------------------------------------------- |
| `_empty`          | reduction | The manifest's empty-`dim` mode of `reduced`: `"full"`, `"noop"` or `"reject"`               |
| `_identity`       | reduction | The result over an empty reduced extent                                                      |
| `_output`         | reduction | The output dtype of a logical reduction (`torch.bool`, or `torch.int64` for `count_nonzero`) |
| `_call`           | reduction | The call spec the `reduce` interface takes                                                   |
| `_output_dtype`   | reduction | The output dtype; the input's by default                                                     |
| `_scalar_forward` | reduction | The result on a 0-d input                                                                    |

## Naming Conventions (Appendix) <a id="naming-conventions"></a>

- **Op class:** `{PascalCaseName}{Direction}Op`. `Direction` ∈ {`Fwd`, `Bwd`}, mandatory. Manifest key must equal `cls.__name__`. Abbreviation casing: `RMSNormFwdOp`, `SSDRecurrentFwdOp` — fully uppercase per `.claude/rules/code-style.md`. Slot [S6](op-slot-rules.md#slot-s6).
- **Kernel class:** `{PascalCaseName}Kernel`, naming the algorithm and its variant, e.g. `BatchNormBwdSplitKernel`. No direction suffix is required: the interface it inherits carries the direction.
- **Kernel interface:** `{PascalCaseName}{Direction}Interface`, beside the call spec it names: in the family's `call_spec.py`, or in the kernel module of a family with one kernel file. Same direction-suffix rule as the op class: variant words precede the direction, e.g. `BatchNormTrainFwdInterface`.
- **`kernel_types` keys:** `snake_case`, decoupled from Kernel class names. Values must match the Kernel `cls.__name__`. The table does not describe dispatch strategy. Slot [S14](op-slot-rules.md#slot-s14).
- **Builder functions:** `snake_case`, e.g. `def rms_norm_fwd(M, N, dtype, ...): ...`.
- **Filenames:** all-lowercase with underscores. Multi-word abbreviations stay fully lowercase (`rms_norm.py`, `ssd_decode.py`; never `RMSNorm.py` or `Ssd_decode.py`). Norm-related names never contract (`rms_norm`, not `rmsnorm`).

## Codegen Details (Appendix) <a id="codegen"></a>

The manifest ([`src/tileops/manifest/`](../../src/tileops/manifest/)) is the sole source of truth. The call checks, dtype validation, shape inference and the fake derive from the signature; roofline codegen is defined in [roofline.md](roofline.md). What the validator holds the code to is [manifest.md § Validation](manifest.md#validation).

### Parameter design <a id="parameter-design"></a>

Two time points: `__init__` takes `signature.params`, construction-time tensors among them. At each call, parameters and presence seed inference, construction-time tensors and call-time inputs are unified together, and `let` derives the rest; generators determine indices only when a workload row is instantiated. An input dtype belongs to the call, never to construction: the tensors carry it, so restating it at construction only creates a second source that can disagree with the first. An output dtype the inputs do not determine is a dtype parameter ([manifest.md § Dtypes](manifest.md#dtypes)). Kernels are built at the first `forward` for a specialization, keyed opaquely by every input that changes what is built.

### Calling conventions

- **Kernel construction:** in `forward`, through `kernel_for` — never on a traced path; an op with a compile boundary runs `forward` inside its operator ([Compile Dispatch Boundary](ops-design.md#compile-dispatch-boundary)). See [Slot S16](op-slot-rules.md#slot-s16).
- **Signature check:** generated, runs on every call, and is the only place an op rejects a dtype.
- **Non-runtime consumers** (validator, graph compiler): call `_infer_output_shapes` with concrete shape tuples, and the input dtypes (keyword `dtypes`) where an output shape reads a dtype index, without constructing tensors. Roofline consumers use interfaces in [`roofline.md`](roofline.md).

## Development Path (Appendix) <a id="development-path"></a>

Pragmatic sequence:

1. **New op inherits L1 directly (T2).** When a family has 1-2 ops, the op owns its full `forward()`. Transitional state.
1. **Family accumulates ops.** When 2-3 ops share identical `forward()` flow, extract an L2 family base.
1. **L1-direct and L1→L2→L3 coexist.** L1-direct ops are candidates for future L2 extraction, not an alternative design.

Create an L2 family base when multiple ops share the same `forward()` control flow, the shared boilerplate is substantial, and per-op differences fit into class variables or hooks. Do NOT create one when only 1 op uses the pattern, ops share math but differ in flow, or a common base would need excessive `if/else`.

### Adding a new family base <a id="adding-a-new-family-base"></a>

1. Implement 2-3 concrete T2 ops to understand the pattern before abstracting.
1. Identify shared `forward()` steps.
1. Extract shared steps into the base; lift per-op differences into class variables or overridable hooks (see [Family-Base Protocol (Appendix)](#base-class-protocol) and [Optional Hooks (Appendix)](#optional-hooks-appendix)).
1. Migrate existing ops; verify tests pass unchanged.
1. Register any new protocol variables in the Family-Base Protocol table.
