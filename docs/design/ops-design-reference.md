# Op Interface Design — Reference

The contracts an op is built against: base-class attributes, family protocol variables, naming, parameter design, calling conventions, and what CI enforces.

## Slot Rules

The per-slot codegen rules live with their consumer:
[`op-slot-rules.md`](op-slot-rules.md).
This document holds the contracts those rules emit against.

## Family-Base Protocol (Appendix) <a id="base-class-protocol"></a>

Per-family protocol variables, declared by L2 bases and overridden by L3 ops.

| Variable      | Family      | Purpose                                                                                                          |
| ------------- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `_kernel_key` | reduction   | Kernel-map lookup key                                                                                            |
| `_kernel_cls` | reduction   | Kernel class reference                                                                                           |
| `_op_kind`    | reduction   | Kernel-dispatch op-kind string (`"sum"` / `"prod"` for `CumulativeOp`; `"sum"`, `"mean"`, … for `_ReduceOpBase`) |
| `_op_name`    | elementwise | `torch.library.custom_op` registration key                                                                       |
| `kernel_cls`  | elementwise | Kernel class reference                                                                                           |

**The scaffolding playbook does NOT emit these variables** — kernel-dispatch-convention-dependent (e.g., `VectorNormKernel` uses `{"l1", "l2", "inf"}`, `ReduceKernel` uses `{"sum", "mean", ...}`); Adding a new protocol variable requires updating the L2 base and all concrete ops.

### `Op` base class interface ([`src/tileops/ops/op_base.py`](../../src/tileops/ops/op_base.py))

Abstract interface: `forward()`. Methods generated from the manifest entry: the construction and call checks, `_infer_output_shapes`, `_validate_dtypes`, `eval_roofline`.

- `kernel_types` (class attribute) is the one declaration of an op's dispatch keys; `default_kernel_map` (property) is derived from it. Each op class created adds its keys to a set `op_base` holds, and a `kernel_map` override naming a key outside that set is refused.
- `last_call` (property) is the `SignatureCall` of the op's last successfully completed call: its `ix`, tensors, effects and metadata tensors. It raises `RuntimeError` before one completes. `eval_roofline` prices it.

#### Kernel caching and enumeration methods

Rationale and the role / entry vocabulary: [ops-design.md § Kernel caching and enumeration](ops-design.md#kernel-caching-and-enumeration).

| Method                           | Purpose                                                                                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kernel_for(role, inputs, call)` | The in-tree kernel serving this call, built once on a miss. The only way an op's in-tree implementation reaches a kernel. A target serves the whole op instead |
| `entry_for(role, call)`          | The in-tree identity and builder. The default selects among the op's candidates and asks the chosen class; an op with one implementation overrides it          |
| `built_kernels(name)`            | Read-only view of a name's entries, whoever built them; empty before its first build. Introspection only, never dispatch                                       |
| `kernel_delegates()`             | The ops whose kernels this op runs. Default `()`; a composite op overrides it                                                                                  |
| `iter_kernels()`                 | The TileOPs `Kernel` instances the entries hold, deduplicated: role entries, `self.kernel`, and delegates. What `autotune()` tunes                             |
| `settled_target`                 | What a call settled the op on: `None` before, `BUILTIN` for the in-tree implementation, else the target's name                                                 |
| `autotune()`                     | Puts the op in tuned mode: tunes built kernels, and sets `tune` so later in-tree builds tune too; a target is not passed `tune`                                |

### `Kernel` base class attributes ([`src/tileops/kernels/kernel_base.py`](../../src/tileops/kernels/kernel_base.py))

Unlike `Op`, a `Kernel` **is** constructed for one element type — it compiles a dtype-specialized program, so `dtype` is a ctor argument here. The op supplies it from the tensors at `forward()`.

| Attribute                            | Type                    | Purpose                                                             |
| ------------------------------------ | ----------------------- | ------------------------------------------------------------------- |
| `dtype`                              | `Optional[torch.dtype]` | Element type this kernel is specialized for                         |
| `config`                             | `Dict[str, Any]`        | Tile configuration (block sizes, stages, etc.)                      |
| `autotune_configs`                   | `Optional[list[dict]]`  | Search space for autotuning                                         |
| `supported_archs`                    | `Optional[list[int]]`   | GPU SM versions (e.g., `[80, 86, 89, 90]`)                          |
| `kernel`                             | `Callable`              | Compiled TileLang kernel function                                   |
| `autotune_accepts_random_int_inputs` | `bool`                  | Whether autotuning may generate the integer tensor inputs at random |

Abstract interface: `forward()`. Key methods: `init_config(config, tune)`, `autotune(warmup, rep)`.

## Optional Hooks (Appendix)

Hooks family bases expose for op-specific semantics. The scaffolding playbook does NOT emit these.

A restriction on the accepted domain is a refinement of the signature, never a hook. A hook that compensates for what a kernel cannot do belongs to that kernel, not here: the op hands over the tensor its manifest declares.

### `_cache_key` override (L1-level, not family-specific)

`Op._cache_key(self, *input_shapes) -> Hashable` defaults to the full input shapes. Override when the kernel's math permits coarser keying — e.g., RMSNorm only depends on the product `M` of the leading axes:

```python
class RMSNormFwdOp(Op):
    def _cache_key(self, x_shape):
        dim = normalize_axis(self.dim, len(x_shape))
        return (math.prod(s for i, s in enumerate(x_shape) if i != dim),)
```

## Naming Conventions (Appendix) <a id="naming-conventions"></a>

- **Op class:** `{PascalCaseName}{Direction}Op`. `Direction` ∈ {`Fwd`, `Bwd`}, mandatory. Manifest key must equal `cls.__name__`. Abbreviation casing: `RMSNormFwdOp`, `SSDDecodeFwdOp` — fully uppercase per `.claude/rules/code-style.md`. Slot [S6](op-slot-rules.md#slot-s6).
- **Kernel class:** `{PascalCaseName}{Direction}Kernel`. Same direction-suffix rule.
- **`kernel_map` keys:** `snake_case`, decoupled from Kernel class names. Values must match the Kernel `cls.__name__`. The table does not describe dispatch strategy. Slot [S14](op-slot-rules.md#slot-s14).
- **Builder functions:** `snake_case`, e.g. `def rms_norm_fwd(M, N, dtype, ...): ...`.
- **Filenames:** all-lowercase with underscores. Multi-word abbreviations stay fully lowercase (`rms_norm.py`, `ssd_decode.py`; never `RMSNorm.py` or `Ssd_decode.py`). Norm-related names never contract (`rms_norm`, not `rmsnorm`).

## Codegen Details (Appendix) <a id="codegen"></a>

The manifest ([`src/tileops/manifest/`](../../src/tileops/manifest/)) is the sole source of truth. The call checks, dtype validation, shape inference and the fake derive from the signature; roofline codegen is defined in [roofline.md](roofline.md). What the validator holds the code to is [manifest.md § Validation](manifest.md#validation).

### Parameter design <a id="parameter-design"></a>

Two time points: `__init__` takes `signature.params`, construction-time tensors among them. At each call, parameters and presence seed inference, construction-time tensors and call-time inputs are unified together, and `let` derives the rest; generators determine indices only when a workload row is instantiated. An input dtype belongs to the call, never to construction: the tensors carry it, so restating it at construction only creates a second source that can disagree with the first. An output dtype the inputs do not determine is a dtype parameter ([manifest.md § Dtypes](manifest.md#dtypes)). Kernels are built at the first `forward` for a specialization, keyed opaquely by every input that changes what is built.

### Calling conventions

- **Kernel construction:** in `_eager_forward`, through `kernel_for` — never in the traced `forward`, which is one call to the op's operator ([Compile Dispatch Boundary](ops-design.md#compile-dispatch-boundary)). See [Slot S16](op-slot-rules.md#slot-s16).
- **`_validate_dtypes`:** runs on every call, and is the only place an op rejects a dtype.
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
