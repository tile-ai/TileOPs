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

**The scaffolding playbook does NOT emit these variables** — kernel-dispatch-convention-dependent (e.g., `VectorNormKernel` uses `{"l1", "l2", "inf"}`, `ReduceKernel` uses `{"sum", "mean", ...}`); Adding a new protocol variable requires updating the L2 base, all concrete ops, and the manifest schema if applicable.

### `Op` base class attributes ([`src/tileops/ops/op_base.py`](../../src/tileops/ops/op_base.py))

| Attribute      | Type                                 | Purpose                                                                                      |
| -------------- | ------------------------------------ | -------------------------------------------------------------------------------------------- |
| `kernel`       | `Kernel`                             | Set only by an op that holds one kernel; an op that builds per specialization uses a role    |
| `kernel_map`   | `Optional[Dict[str, Kernel]]`        | Dispatched kernels keyed by name                                                             |
| `dtype`        | `Optional[torch.dtype]`              | Dtype of the most recent `forward()`; `None` before the first one                            |
| `device`       | `Optional[Union[torch.device, str]]` | Device (default `'cuda'`)                                                                    |
| `input_shapes` | `Optional[list[tuple]]`              | Expected input tensor shapes (for introspection and non-runtime consumers)                   |
| `tune`         | `bool`                               | Whether kernels this op builds tune themselves; read by a factory when it runs               |
| `_static_axes` | `frozenset[tuple[int, int]]`         | Static axes as `(input_index, axis)` pairs (default `frozenset()`); consumed by `_cache_key` |

Abstract interface: `default_kernel_map` (property), `forward()`. Manifest-driven methods (codegen-emitted by concrete ops): `_infer_output_shapes`, `_validate_dtypes`, `eval_roofline`.

#### Kernel caching and enumeration methods

Rationale and the role / entry vocabulary: [ops-design.md § Kernel caching and enumeration](ops-design.md#kernel-caching-and-enumeration).

| Method                                             | Purpose                                                                                                                                                                                                                                                                    |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_or_build_kernel(name, inputs, *, key, build)` | Return the kernel for this call, building it once on a miss. The only get-or-build in L1-L3. `key` and `build` are the in-tree recipe; `inputs` is what an external target's builder is described with, and an op that has not been wired to external targets yet omits it |
| `built_kernels(name)`                              | Read-only view of a name's entries; empty before its first build. Introspection only, never dispatch                                                                                                                                                                       |
| `kernel_delegates()`                               | The ops whose kernels this op runs. Default `()`; a composite op overrides it                                                                                                                                                                                              |
| `iter_kernels()`                                   | Every `Kernel` the op holds, deduplicated: entries and delegates                                                                                                                                                                                                           |
| `autotune()`                                       | Puts the op in tuned mode: tunes built kernels, and sets `tune` so later builds tune too                                                                                                                                                                                   |

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

| Hook              | Family    | Default                     | Override example                                      |
| ----------------- | --------- | --------------------------- | ----------------------------------------------------- |
| `_validate_dim()` | reduction | accept `int` or `list[int]` | `ArgmaxFwdOp._validate_dim` restricts to scalar `int` |

A hook that compensates for what a kernel cannot do belongs to that kernel, not here: the op hands over the tensor its manifest declares.

### `_cache_key` override (L1-level, not family-specific)

`Op._cache_key(self, *input_shapes) -> Hashable` defaults to projecting non-static axes via `self._static_axes`. Override when the kernel's math permits coarser keying — e.g., RMSNorm only depends on the non-static axis product `M`:

```python
class RMSNormFwdOp(Op):
    def _cache_key(self, x_shape):
        dim = self.dim % len(x_shape)
        return (math.prod(s for i, s in enumerate(x_shape) if i != dim),)
```

**When `_static_axes` is empty, override is mandatory** — the default keys by the full input shape (one kernel compile per distinct shape). The base emits a once-per-type `UserWarning` when invoked with empty `_static_axes` and no subclass override.

## Naming Conventions (Appendix) <a id="naming-conventions"></a>

- **Op class:** `{PascalCaseName}{Direction}Op`. `Direction` ∈ {`Fwd`, `Bwd`}, mandatory. Manifest key must equal `cls.__name__`. Abbreviation casing: `RMSNormFwdOp`, `SSDDecodeFwdOp` — fully uppercase per `.claude/rules/code-style.md`. Slot [S6](op-slot-rules.md#slot-s6).
- **Kernel class:** `{PascalCaseName}{Direction}Kernel`. Same direction-suffix rule.
- **`kernel_map` keys:** `snake_case`, decoupled from Kernel class names. Values must match the Kernel `cls.__name__`. The table does not describe dispatch strategy. Slot [S14](op-slot-rules.md#slot-s14).
- **Builder functions:** `snake_case`, e.g. `def rms_norm_fwd(M, N, dtype, ...): ...`.
- **Filenames:** all-lowercase with underscores. Multi-word abbreviations stay fully lowercase (`rms_norm.py`, `ssd_decode.py`; never `RMSNorm.py` or `Ssd_decode.py`). Norm-related names never contract (`rms_norm`, not `rmsnorm`).

## Codegen Details (Appendix) <a id="codegen"></a>

The manifest ([`src/tileops/manifest/`](../../src/tileops/manifest/)) is the sole source of truth. Dtype validation and shape inference derive from manifest; roofline codegen is defined in [roofline.md](roofline.md).

### Parameter design <a id="parameter-design"></a>

Three time points: (1) manifest — constraint structure; (2) `__init__` — user commits `static_dims` values; (3) `forward` — shapes concrete, commitments validated, dtype read from the tensors. See [manifest.md § `static_dims`](manifest.md#static_dims).

**Dtype belongs to time point 3, never to 2.** The tensors carry it, so requiring the caller to restate it at construction only creates a second source that can disagree with the first. Constructing an op therefore commits to shape structure and nothing about element type.

|                          | Fixed-rank op           | Arbitrary-rank op                                            |
| ------------------------ | ----------------------- | ------------------------------------------------------------ |
| Manifest has `shape`     | yes                     | no                                                           |
| `__init__` shape source  | `shape` dimension names | `static_dims`                                                |
| Undeclared dimensions    | none                    | derived from tensor at forward time                          |
| Kernel construction time | forward (first call)    | forward (first encounter)                                    |
| Forward keying           | dtype                   | opaque to L1; carries every input that changes what is built |

### Calling conventions

- **Fully static op:** `_infer_output_shapes` called once in `__init__`, result stored as an instance attribute.
- **Op with dynamic dims:** `_infer_output_shapes` called once dynamic dims resolve, and by the fake while tracing.
- **Kernel construction:** in `_eager_forward`, through `get_or_build_kernel` — never in the traced `forward`, which is one call to the op's operator ([Compile Dispatch Boundary](ops-design.md#compile-dispatch-boundary)). See [Slot S16](op-slot-rules.md#slot-s16).
- **`_validate_dtypes`:** runs on every call, and is the only place an op rejects a dtype.
- **Non-runtime consumers** (validator, graph compiler): call `_infer_output_shapes` with concrete shape tuples without constructing tensors. Roofline consumers use interfaces in [`roofline.md`](roofline.md).

### Inheritance in family-base hierarchies

| Scenario                                             | Codegen method defined at | Concrete op action    |
| ---------------------------------------------------- | ------------------------- | --------------------- |
| Family shares logic                                  | L2 family base            | Inherits, no override |
| Family member has variant logic (e.g., multi-output) | L3 concrete op            | Overrides             |
| Op inherits L1 directly (T2)                         | L3 concrete op            | Scaffold emits body   |

### Consistency enforcement

| Check                                                    | Mechanism                            |
| -------------------------------------------------------- | ------------------------------------ |
| Manifest schema and declared fields are well-formed      | Validator (CI), L0 checks            |
| `__init__` params match manifest `params`                | Validator signature check (L1)       |
| `static_dims` keys are `__init__` parameters             | Validator signature check (L1)       |
| `shape_rules` syntax is valid                            | Validator `shape_rules` parsing (L2) |
| `_infer_output_shapes` output satisfies `shape_rules`    | Validator infer-shape parity (L2)    |
| `dtype`/`dtype_combos` strings are valid                 | Validator dtype conformance (L3)     |
| `_validate_dtypes` matches `dtype_combos` / dtype unions | Validator dtype parity (L3)          |
| Empty `static_dims` without `_cache_key` override        | `Op` base class runtime warning      |

Checks beyond this table are tracked as separate issues, not as spec status.

**Parity check coverage.** The L2 / L3 parity checks compare the manifest spec against the concrete method the op class defines. When the class has not migrated to the codegen protocol, the validator emits a **warning** naming the missing method — the gap is surfaced, never silently passed. When the method exists, the parity check runs and any disagreement is a hard L2 / L3 error. Ops whose method genuinely cannot be invoked in a CPU-only validator context must declare `status: spec-only`; there is no parity opt-out, and demotion is only legitimate when the implementation truly does not conform.

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
