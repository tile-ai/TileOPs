# Op Interface Design — Reference

Detail sections referenced by [ops-design.md](ops-design.md). This file contains design rationale, rules, and protocol details that go beyond the execution guide.

## Parameter Design

### Arbitrary-Rank Ops and `static_dims`

Manifest does not declare `shape` (accepts arbitrary rank), but this does not mean all shape information must wait until forward.

Three time points with increasing information specificity:

1. **Manifest declaration** — most general. Describes constraint structure: "one dimension is known (the dimension to apply norm on), others are arbitrary."
1. **Op declaration (init)** — user commits to specific values. E.g., RMSNorm user commits `N=4096` but not batch size.
1. **Forward** — all information is concrete. Tensors are passed in, all shapes are known; committed values are validated against actual tensor shapes.

`static_dims` allows the manifest to declare which values the user commits to at ctor:

```yaml
# manifest
RMSNormFwdOp:
  signature:
    inputs:
      x: {dtype: "float16 | bfloat16"}
      weight: {dtype: "same_as(x)"}
    params:
      dim: {type: int, default: -1}
      eps: {type: float, default: 1e-6}
    static_dims:
      N: "x.shape[dim]"
```

See [manifest.md §R20](manifest.md#rules) for full syntax, single-axis constraint, evaluation context, two-time-point semantics, and empty-`static_dims` rules.

### Static vs Dynamic Comparison

|                          | Fixed-rank op            | Arbitrary-rank op                                                  |
| ------------------------ | ------------------------ | ------------------------------------------------------------------ |
| Manifest has `shape`     | yes                      | no                                                                 |
| `__init__` shape source  | `shape` dimension names  | `static_dims`                                                      |
| Undeclared dimensions    | none (all dims declared) | derived from tensor at forward time                                |
| Kernel construction time | init (all dims known)    | init (`static_dims` known) or forward (first encounter, cached)    |
| Forward cache keying     | N/A (single kernel)      | `_cache_key(*input_shapes)` — default non-static axes, overridable |

## Development Path

Agent-driven development follows a pragmatic sequence:

1. **New op inherits L1 directly.** When a family has only 1-2 ops, the op owns its full `forward()`. This is a transitional state, not a target architecture.
1. **Family accumulates ops.** When 2-3 ops in a family share identical `forward()` flow, extract an L2 family base via refactoring.
1. **L1-direct and L1→L2→L3 coexist.** This is a natural consequence of incremental development. L1-direct ops are candidates for future L2 extraction, not an alternative design.

Create an L2 family base when **multiple ops share the same `forward()` control flow**, the shared boilerplate is substantial, and per-op differences fit into class variables or hooks.

Do NOT create one when only 1 op uses the pattern, ops share math but differ in flow, or a common base would need excessive `if/else`.

## Codegen

The manifest ([`ops_manifest.yaml`](../tileops/ops_manifest.yaml)) is the **sole source of truth** for op interfaces. Op-layer runtime behavior — dtype validation and shape inference — MUST be derived from the manifest, not independently maintained. Roofline codegen and runtime behavior are defined in [roofline.md](roofline.md).

Agent reads the manifest and generates code (codegen). [Validator](../scripts/validate_manifest.py) (CI) enforces manifest schema and signature consistency; enforced checks are listed in [Consistency Enforcement](#consistency-enforcement) below.

### Calling Conventions

- **Fully static op:** `_infer_output_shapes` is called once in `__init__`, result stored as an instance attribute.
- **Op with dynamic dims:** `_infer_output_shapes` is called in `forward()` when dynamic dims are resolved. Kernel construction caches keyed by `_cache_key(*input_shapes)` (see [ops-design.md](ops-design.md#_cache_key)).
- **`_validate_dtypes`:** runs on every `forward()` call — dtype validity depends on the actual tensors passed, not cached.
- **Non-runtime consumers** (validator, graph compiler): can call `_infer_output_shapes` with concrete shapes without constructing tensors. `_validate_dtypes` requires actual dtypes (not shapes). Roofline consumers use the interfaces defined in [roofline.md](roofline.md).

### Inheritance in Family-Base Hierarchies

| Scenario                                             | Codegen method defined at | Concrete op action    |
| ---------------------------------------------------- | ------------------------- | --------------------- |
| Family shares logic                                  | L2 family base            | Inherits, no override |
| Family member has variant logic (e.g., multi-output) | L3 concrete op            | Overrides             |
| Op inherits L1 directly                              | L3 concrete op            | Agent generates       |

### Consistency Enforcement

| Check                                                    | Mechanism                          |
| -------------------------------------------------------- | ---------------------------------- |
| Manifest schema and declared fields are well-formed      | Validator (CI), L0 checks          |
| `__init__` params match manifest `params`                | Validator signature check (L1)     |
| `static_dims` keys are `__init__` parameters             | Validator signature check (L1)     |
| `shape_rules` syntax is valid                            | Validator shape_rules parsing (L2) |
| `_infer_output_shapes` output satisfies `shape_rules`    | Validator infer-shape parity (L2)  |
| `dtype`/`dtype_combos` strings are valid                 | Validator dtype conformance (L3)   |
| `_validate_dtypes` matches `dtype_combos` / dtype unions | Validator dtype parity (L3)        |
| Empty `static_dims` without `_cache_key` override        | Op base class runtime warning      |

Checks beyond this table are tracked as separate issues, not as spec status.

## Naming Conventions

#### Op Classes

```
{PascalCaseName}{Direction}Op
```

- **Direction** — mandatory: `Fwd` or `Bwd`. No exceptions.

The manifest key must exactly equal `cls.__name__`.

#### Kernel Classes

```
{PascalCaseName}{Direction}Kernel
```

- **Direction** — mandatory: `Fwd` or `Bwd`. No exceptions.

#### Kernel Dispatch (kernel_map)

A flat dict mapping dispatch keys to Kernel classes. The manifest declares this table; agents implement the listed Kernels.

- **Keys**: snake_case identifiers, decoupled from Kernel class names.
- **Values**: Kernel class names, must match `cls.__name__`.
- The table does not describe dispatch strategy. Strategy is a runtime concern.

#### Builder Functions

Kernel builder functions remain `snake_case`:

```python
def rms_norm_fwd(M, N, dtype, ...): ...
```

## Base Class Protocol

#### Op base class ([`tileops/ops/op_base.py`](../tileops/ops/op_base.py))

| Attribute        | Type                          | Purpose                                                                                      |
| ---------------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| `kernel`         | `Kernel`                      | Kernel instance used by `forward()`                                                          |
| `kernel_map`     | `Optional[Dict[str, Kernel]]` | Dispatched kernels keyed by name                                                             |
| `dtype`          | `Optional[torch.dtype]`       | Computation dtype                                                                            |
| `device`         | `Optional[str]`               | Device (default `'cuda'`)                                                                    |
| `_output_shapes` | `Optional[Dict[str, tuple]]`  | Inferred output shapes (populated at init or forward)                                        |
| `_static_axes`   | `frozenset[tuple[int, int]]`  | Static axes as `(input_index, axis)` pairs (default `frozenset()`); consumed by `_cache_key` |

Abstract interface: `default_kernel_map` (property), `forward()`.

Manifest-driven methods (generated by agent):

- `_infer_output_shapes(...)` → output name → shape
- `_validate_dtypes(...)` → raise on invalid dtypes
- `eval_roofline()` → see [roofline.md](roofline.md)

Base-class provided methods (optional override):

- `_cache_key(*input_shapes) → Hashable` — default returns tuple of non-static-axis sizes across all input shapes, using `self._static_axes` to determine which axes are committed. Override to project the shape onto whatever the kernel actually depends on. See [ops-design.md § `_cache_key`](ops-design.md#_cache_key). When `_static_axes` is empty and no override is provided, the base class emits a once-per-type runtime warning.

#### Kernel base class ([`tileops/kernels/kernel_base.py`](../tileops/kernels/kernel_base.py))

| Attribute          | Type                    | Purpose                                        |
| ------------------ | ----------------------- | ---------------------------------------------- |
| `dtype`            | `Optional[torch.dtype]` | Data type                                      |
| `config`           | `Dict[str, Any]`        | Tile configuration (block sizes, stages, etc.) |
| `autotune_configs` | `Optional[list[dict]]`  | Search space for autotuning                    |
| `supported_archs`  | `Optional[list[int]]`   | GPU SM versions (e.g., `[80, 86, 89, 90]`)     |
| `kernel`           | `Callable`              | Compiled TileLang kernel function              |

Abstract interface: `forward()`.
Key methods: `init_config(config, tune)`, `autotune(warmup, rep)`.

#### Family-base protocol variables

| Variable                  | Family          | Purpose                                                   |
| ------------------------- | --------------- | --------------------------------------------------------- |
| `_kernel_key`             | norm, reduction | Kernel-map lookup key                                     |
| `_kernel_cls`             | norm, reduction | Kernel class reference                                    |
| `_op_kind`                | reduction       | Reduction kind string (`"sum"`, `"mean"`, `"std"`, etc.)  |
| `_kernel_handles_padding` | reduction       | `True` → kernel uses masked loads, skip host-side padding |
| `_op_name`                | elementwise     | `torch.library.custom_op` registration key                |
| `kernel_cls`              | elementwise     | Kernel class reference                                    |

Adding a new protocol variable requires updating: (1) the base class, (2) all concrete ops, (3) the manifest schema if applicable.

## Conventions in Code

| Convention                             | Enforced By                                          |
| -------------------------------------- | ---------------------------------------------------- |
| Non-contiguous → `.contiguous()`       | Per-family base `forward()` or per-op implementation |
| Alignment padding                      | Per-family base `forward()` or kernel masked loads   |
| CUDA device check                      | Per-family base `forward()` or per-op implementation |
| `torch.library.custom_op` registration | Per-op module or shared registration utility         |
| Docstring format (Google style)        | Linter / CI check                                    |

## Adding a New Family Base

1. **Implement 2-3 concrete ops inheriting Op directly** — understand the pattern before abstracting
1. **Identify shared steps** — which parts of `forward()` are identical?
1. **Extract the base class** — shared steps into base, per-op differences as hooks
1. **Migrate existing ops** — verify tests pass unchanged
1. **Register the pattern** — update the hierarchy section of this document
