# Op Interface Design

## Op and Kernel

Every operator is split into two classes: **Op** and **Kernel**.

- **Op** — host-side. Validates inputs, prepares memory layout, dispatches to Kernel, assembles output.
- **Kernel** — device-side. Owns the TileLang program, tile configuration, JIT compilation.

The two layers are independently modifiable — changing a Kernel's tile strategy does not require changing the Op, and vice versa.

## Op Class Hierarchy

```
Op                          ← L1: thin base, shared by all ops
  └── FamilyBase            ← L2: family-specific forward() flow
        └── ConcreteOp      ← L3: pure declaration, no logic override
```

- **L1 (Op)** — abstract base. Provides `__call__`, `dispatch_kernel()`, `autotune()`. Thin — only infrastructure all ops share.
- **L2 (FamilyBase)** — per-family shared `forward()` pipeline. One per family.
- **L3 (ConcreteOp)** — leaf class. Declares kernel class, op kind, dimension wiring. No logic.

New ops start by inheriting L1 directly. When a family accumulates 2-3 ops with identical `forward()` flow, extract an L2 base via refactoring. L1-direct and L1→L2→L3 coexist as a natural consequence of incremental development.

See [Development Path](ops-design-reference.md#development-path) for when to create an L2 family base.

## Execution Timing

Kernel construction and shape inference follow one principle: **do it at the first moment all required information is known, do it once, cache the result. Same inputs never trigger recomputation.**

| Op category    | When all info is known                                             | Behavior                                                                                                                                     |
| -------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Fixed-rank     | `__init__` (all dimensions provided)                               | Everything runs once at init.                                                                                                                |
| Arbitrary-rank | `__init__` for `static_dims` values; `forward` for everything else | Values committed in `static_dims` are stored at init; remaining shape information triggers kernel/shape on first encounter, cached by value. |

This applies uniformly to kernel construction and `_infer_output_shapes`.
Roofline timing and formula semantics are defined in [roofline.md](roofline.md).
Cache key granularity is determined by the Op author via `_cache_key` (see [`_cache_key`](#_cache_key) below) — the default is correct for any op; an override buys efficiency when the kernel's math permits coarser keying.

**Exception:** `_validate_dtypes` runs on every `forward()` call — dtype validity depends on the actual tensors passed, not just their shapes.

## Implementing an Op

A complete Op implements `__init__`, `forward`, `default_kernel_map`, three codegen methods, and optionally overrides `_cache_key`:

```python
class MyFwdOp(Op):
    def __init__(self, *, M: int, N: int, dtype: torch.dtype):
        self.M, self.N, self.dtype = M, N, dtype
        self.dispatch_kernel()
        self.kernel = self.kernel_map["my_kernel"](M, N, dtype)
        self._output_shapes = self._infer_output_shapes(x_shape=(M, N))

    @property
    def default_kernel_map(self):
        return {"my_kernel": MyFwdKernel}

    def _infer_output_shapes(self, x_shape: tuple) -> Dict[str, tuple]:
        return {"y": x_shape}

    def _validate_dtypes(self, x: torch.Tensor) -> None:
        if x.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_dtypes(x)
        x = x.contiguous().reshape(self.M, self.N)
        return self.kernel(x)
```

Each method is explained below.

### Parameter Sourcing from Manifest

The manifest ([`ops_manifest.yaml`](../tileops/ops_manifest.yaml)) is the sole source of truth. Every `__init__` and `forward` parameter must trace back to a manifest declaration:

| Manifest source                      | Goes to    | Examples                |
| ------------------------------------ | ---------- | ----------------------- |
| `signature.inputs` (tensors)         | `forward`  | `x`, `weight`           |
| `signature.params` (non-tensor)      | `__init__` | `dim`, `eps`, `keepdim` |
| per-tensor `dtype` fields            | `__init__` | `dtype` (see below)     |
| `shape` dimension names (fixed-rank) | `__init__` | `M`, `K`, `N`           |
| `static_dims` (arbitrary-rank)       | `__init__` | `N`                     |

**dtype parameter derivation:** When all tensors share the same dtype via `same_as(x)`, a single `dtype` parameter covers all of them. When `dtype_combos` declares multiple independent dtype axes, the agent generates a named parameter for each independent axis.

Information not declared in the manifest MUST NOT appear in `__init__`. No exceptions.

See [manifest.md](manifest.md) for the full manifest specification and [Parameter Design](ops-design-reference.md#parameter-design) for fixed-rank vs arbitrary-rank details.

### `__init__`

`__init__` uses **keyword-only arguments** (Python `*` syntax). Parameter names come directly from manifest dimension names and param names.

**Fixed-rank op** — all dimensions from manifest `shape`:

```python
class GemmFwdOp(Op):
    def __init__(self, *, M: int, K: int, N: int, dtype: torch.dtype):
        self.M, self.K, self.N, self.dtype = M, K, N, dtype
        self.dispatch_kernel()
        self.kernel = self.kernel_map["gemm"](M, K, N, dtype)
        self._output_shapes = self._infer_output_shapes(a_shape=(M, K), b_shape=(K, N))
```

**Arbitrary-rank op** — `static_dims` values + `params`:

```python
class RMSNormFwdOp(RowNormOp):
    def __init__(self, *, N: int, dtype: torch.dtype, dim: int = -1, eps: float = 1e-6):
        self.N, self.dtype, self.dim, self.eps = N, dtype, dim, eps
        self.dispatch_kernel()
        # kernel construction deferred to forward — non-static axes unknown at init
        self._kernel_cache = {}
```

**Kwarg block order.** The `__init__` signature has three blocks in this order:

1. `static_dims` entries — in manifest key order
1. `dtype` (single parameter, unless the op has multi-dtype axes)
1. `params` entries — in manifest key order

All keyword-only. Callers always use kwargs; block order fixes the visible signature for documentation and introspection.

> [!NOTE]
> `def __init__(self, *, ...)` — the `*` forces all parameters to be keyword-only. Callers must write `MyOp(N=4096, dtype=torch.float16, dim=-1)`; positional arguments are rejected.

### `forward`

`forward` receives tensors declared in manifest `signature.inputs`. It validates dtypes, derives any undeclared dimensions from tensor shapes, and calls the kernel.

```python
def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    self._validate_dtypes(x, weight)
    orig_shape = x.shape
    # validate static_dims commitment, then derive M
    dim = self.dim % x.ndim
    if x.shape[dim] != self.N:
        raise ValueError(
            f"static_dim mismatch: expected x.shape[{dim}] == {self.N}, got {x.shape[dim]}"
        )
    self.M = math.prod(s for i, s in enumerate(x.shape) if i != dim)
    M = self.M
    # kernel cached by _cache_key (overridden to return (M,) — see _cache_key section)
    key = self._cache_key(x.shape)
    if key not in self._kernel_cache:
        self._kernel_cache[key] = self.kernel_map["rms_norm"](
            M, self.N, self.dtype, eps=self.eps
        )
    kernel = self._kernel_cache[key]
    # move target dim to last, reshape to 2D, compute, restore layout
    x = x.movedim(dim, -1).contiguous().reshape(M, self.N)
    y = kernel(x, weight)
    return y.reshape(*orig_shape[:dim], *orig_shape[dim + 1 :], self.N).movedim(-1, dim)
```

### `_infer_output_shapes` (codegen)

Generated from manifest `shape_rules`. Accepts shape tuples, returns output name → shape mapping.

```yaml
# manifest
shape_rules:
  - "y.shape == x.shape"
  - "weight.shape == (x.shape[dim],)"
```

```python
# generated
def _infer_output_shapes(self, x_shape: tuple, weight_shape: tuple) -> Dict[str, tuple]:
    return {"y": x_shape}
```

Follows the [Execution Timing](#execution-timing) principle: called at init (fixed-rank) or at forward (arbitrary-rank). For arbitrary-rank ops, shape inference depends on the full input shape (not just M), so it runs per unique input shape — not per unique M like kernel construction.

### `_validate_dtypes` (codegen)

Generated from manifest `dtype` fields and `dtype_combos`.

```yaml
# manifest
inputs:
  x: {dtype: "float16 | bfloat16"}
  weight: {dtype: "same_as(x)"}
```

```python
# generated
def _validate_dtypes(self, x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"x.dtype must be float16 or bfloat16, got {x.dtype}")
    if weight.dtype != x.dtype:
        raise ValueError(f"weight.dtype must match x.dtype, got {weight.dtype}")
```

Supersedes `SUPPORTED_DTYPES` as a standalone class variable.

### `eval_roofline` (codegen)

Generated from the manifest `roofline` section. The source of truth for
formula syntax, variable binding, runtime timing, validator behavior, and
codegen lowering is [roofline.md](roofline.md). Do not define an Op-local
roofline expression parser in `Op` or family bases.

### `default_kernel_map`

Property that returns `{dispatch_key: KernelClass}`. The manifest declares this table; agents implement the listed Kernels.

```python
# single-kernel op
@property
def default_kernel_map(self):
    return {"rms_norm": RMSNormFwdKernel}


# multi-kernel pipeline
@property
def default_kernel_map(self):
    return {
        "mha_bwd_preprocess": FlashAttnBwdPreprocessKernel,
        "mha_bwd": MHABwdKernel,
        "mha_bwd_postprocess": FlashAttnBwdPostprocessKernel,
    }
```

### `_cache_key`

For arbitrary-rank ops, kernels are compiled lazily at forward time and cached. The Op base class provides a default `_cache_key(self, *input_shapes) -> Hashable` that returns a tuple of non-static-axis sizes across all inputs — correct for any op, but potentially over-fragmenting (one compile per distinct input shape when `static_dims` is empty).

Override `_cache_key` when the kernel's math permits coarser keying:

```python
class RMSNormFwdOp(Op):
    def _cache_key(self, x_shape):
        # kernel treats input as (M, N); M is the only non-static axis
        dim = self.dim % len(x_shape)
        return (math.prod(s for i, s in enumerate(x_shape) if i != dim),)
```

**When `static_dims` is empty, the override is mandatory.** Typical case: PyTorch-aligned reductions that accept `dim=None`. The default would cache by the full input shape tuple, producing one kernel compile per distinct shape — pathological under dynamic shapes. Override to project the shape onto whatever the kernel actually depends on:

```python
class SumFwdOp(Op):
    def _cache_key(self, x_shape):
        # every full-reduction with the same numel shares a kernel
        return (math.prod(x_shape),)
```

The base class emits a once-per-type runtime warning if the default `_cache_key` is invoked with empty `static_dims` and no subclass override, surfacing missing overrides during development.

Override for **finer** granularity is rare but supported (e.g., kernel math depending on parity or divisibility of a non-static axis).

## Implementing a Kernel

| Interface             | Required | Description                                               |
| --------------------- | -------- | --------------------------------------------------------- |
| `__init__(self, ...)` | yes      | Receives shape params and dtype. Builds TileLang program. |
| `forward(self, ...)`  | yes      | Launches the compiled kernel. Called by Op's `forward()`. |
| `kernel`              | yes      | Attribute. The TileLang program builder (JIT-compiled).   |
| `default_config`      | no       | Property. Default tile configuration dict.                |
| `autotune_configs`    | no       | Class variable. Search space for autotuning.              |
| `supported_archs`     | no       | Class variable. List of supported GPU SM versions.        |

```python
class MyFwdKernel(Kernel):
    supported_archs = [80, 86, 89, 90]

    def __init__(self, M, N, dtype, *, config=None, tune=False):
        super().__init__()
        self.M, self.N, self.dtype = M, N, dtype
        self.kernel = self._build_program(M, N, dtype)
        self.init_config(config, tune)

    def _build_program(self, M, N, dtype):
        # Returns a TileLang program (JIT-compiled)
        ...

    @property
    def default_config(self):
        return {"block_M": 128, "block_N": 128, "num_stages": 2}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, **self.config)
```

## Naming Conventions

- Op: `{PascalCaseName}{Direction}Op` (e.g., `RMSNormFwdOp`). Direction suffix is mandatory, no exceptions.
- Kernel: `{PascalCaseName}{Direction}Kernel`. Direction suffix is mandatory, no exceptions.
- `kernel_map` keys: `snake_case`, decoupled from class names.
- Builder functions: `snake_case` (e.g., `rms_norm_fwd(M, N, dtype, ...)`).
- Manifest key must exactly equal `cls.__name__`.

See [Naming Conventions](ops-design-reference.md#naming-conventions) for full rules.

## Further Reference

- [Parameter Design](ops-design-reference.md#parameter-design) — `static_dims` spec, static vs dynamic comparison
- [Development Path](ops-design-reference.md#development-path) — when and how to extract an L2 family base
- [Codegen details](ops-design-reference.md#codegen) — calling conventions, inheritance rules, consistency enforcement
- [Base Class Protocol](ops-design-reference.md#base-class-protocol) — Op and Kernel base class attributes
- [Naming Conventions](ops-design-reference.md#naming-conventions) — full naming rules
- [Adding a New Family Base](ops-design-reference.md#adding-a-new-family-base) — step-by-step process
