# Op Interface Design

## Architecture

### Op/Kernel Boundary

Adding a new operator requires implementing two classes: an **Op** and a **Kernel**. They have distinct responsibilities and interfaces.

**Op** — host-side orchestration. Receives user-facing tensors, validates inputs, prepares data layout, dispatches to Kernel, and assembles output.

| Interface                | Required | Description                                                                                |
| ------------------------ | -------- | ------------------------------------------------------------------------------------------ |
| `__init__(self, *, ...)` | yes      | Keyword-only params from manifest. Dispatches kernel, runs shape inference.                |
| `forward(self, ...)`     | yes      | Validate → reshape → call kernel → reshape output. User-facing entry point.                |
| `default_kernel_map`     | yes      | Property. Returns `{dispatch_key: KernelClass}`.                                           |
| `_infer_output_shapes()` | yes      | Codegen from manifest `shape_rules`. See [Shape Inference](#shape-inference-codegen).      |
| `_validate_dtypes()`     | yes      | Codegen from manifest `dtype`. See [Dtype Validation](#dtype-validation-codegen).          |
| `eval_roofline()`        | yes      | Codegen from manifest `roofline`. See [Roofline Evaluation](#roofline-evaluation-codegen). |

```python
class MyFwdOp(Op):
    def __init__(self, *, M: int, N: int, dtype: torch.dtype, ...):
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

> [!NOTE]
> `def __init__(self, *, ...)` — the `*` forces all parameters to be keyword-only. Callers must write `MyFwdOp(M=2048, N=4096, dtype=torch.float16)`; positional arguments are rejected. This is deliberate: parameter names come from manifest dimension names (single letters like `M`, `K`, `N`), keyword-only eliminates ordering ambiguity.
>
> **TODO:** When implementation catches up, add a corresponding rule to `.claude/rules/code-style.md`.

**Kernel** — device-side computation. Owns the TileLang program, tile configuration, and JIT compilation.

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

The two layers are independently modifiable — changing a Kernel's tile strategy does not require changing the Op, and vice versa.

### Op Class Hierarchy

```
Op                          ← L1: thin base, shared by all ops
  └── FamilyBase            ← L2: family-specific forward() flow
        └── ConcreteOp      ← L3: pure declaration, no logic override
```

- **Op (L1)** — abstract base class. Defines the contract: `default_kernel_map`, `forward()`, kernel dispatch, manifest-driven validation and shape inference. Thin by design — only infrastructure that ALL ops share.
- **FamilyBase (L2)** — intermediate base per op family. Owns the shared `forward()` pipeline. One per family.
- **ConcreteOp (L3)** — leaf class. Pure declaration: kernel class, op kind, dimension wiring. No logic override.

This three-layer structure is a design decision. Without L2 family bases, shared forward() logic gets duplicated across every concrete op, creating maintenance problems at scale.

In L1→L2→L3 hierarchies, a concrete Op (L3) is a pure declaration:

- Which kernel (`_kernel_cls`, `_kernel_key`)
- Which op kind (`_op_kind`)
- Dimension wiring (keyword params → kernel constructor)

Shared mechanics — validation, reshape, padding, shape inference, dtype validation, kernel dispatch, trimming — are inherited from L2.

Ops that still inherit L1 directly own their full `forward()`. As their family matures, shared logic migrates to an L2 base, and concrete ops converge toward declarations.

## Development Path

Agent-driven development follows a pragmatic sequence:

1. **New op inherits L1 directly.** When a family has only 1-2 ops, the op owns its full `forward()`. This is a transitional state, not a target architecture.
1. **Family accumulates ops.** When 2-3 ops in a family share identical `forward()` flow, extract an L2 family base via refactoring.
1. **L1-direct and L1→L2→L3 coexist.** This is a natural consequence of incremental development. L1-direct ops are candidates for future L2 extraction, not an alternative design.

Create an L2 family base when **multiple ops share the same `forward()` control flow**, the shared boilerplate is substantial, and per-op differences fit into class variables or hooks.

Do NOT create one when only 1 op uses the pattern, ops share math but differ in flow, or a common base would need excessive `if/else`.

## Manifest-Driven Op Interface

The manifest (`ops_manifest.yaml`) is the **sole source of truth** for op interfaces. Op-layer runtime behavior — dtype validation, shape inference, roofline evaluation — MUST be derived from the manifest, not independently maintained.

Agent reads the manifest and generates code (codegen). Validator (CI) checks consistency between generated code and manifest. This prevents drift between spec and implementation.

### Keyword-Only `__init__` with Manifest Dimension Names

Op `__init__` parameters use **keyword-only arguments** named after the manifest's dimension names and param names.

**Fixed-rank op** (manifest declares `shape`):

```yaml
# manifest
inputs:
  a: {dtype: "float16 | bfloat16", shape: "[M, K]"}
  b: {dtype: "same_as(a)", shape: "[K, N]"}
params:
  # (none for GEMM)
```

```python
# generated Op __init__
class GemmFwdOp(Op):
    def __init__(self, *, M: int, K: int, N: int, dtype: torch.dtype, ...):
        self.M = M
        self.K = K
        self.N = N
        self.dtype = dtype
```

**Arbitrary-rank op** (manifest uses `roofline.vars` to derive dimensions):

```yaml
# manifest
inputs:
  x: {dtype: "float16 | bfloat16"}
  weight: {dtype: "same_as(x)"}
params:
  dim: {type: int, default: -1}
  eps: {type: float, default: 1e-6}
roofline:
  vars:
    M: "product(x.shape[:dim])"
    N: "x.shape[dim]"
```

```python
# generated Op __init__
class RMSNormFwdOp(RowNormOp):
    def __init__(self, *, N: int, dtype: torch.dtype,
                 M: int = None, dim: int = -1, eps: float = 1e-6, ...):
        self.N = N      # kernel-relevant, required
        self.M = M      # batch-relevant, optional (dynamic when None)
        self.dim = dim
        self.eps = eps
```

**Naming rule:** keyword names are taken directly from the manifest — `shape` dimension names (`M`, `K`, `N`), `roofline.vars` keys, and `params` keys. No additional mapping.

### Static and Dynamic Dimensions

Static `__init__` is the **default path**. All shape dimensions provided at construction time enables shape inference, kernel construction, and roofline evaluation to complete once at init. This aligns with TileLang's JIT compilation model where kernels are shape-specialized.

Dynamic dimensions are supported as a controlled extension for cases where shapes vary across calls (e.g., variable batch size in serving).

**Convention:** provided value = static, `None` = dynamic.

| Phase        | All static                                   | Has dynamic dimensions                                                                                 |
| ------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `__init__`   | Shape inference complete, kernel constructed | Static dimensions resolved, dynamic deferred                                                           |
| `forward()`  | Validate input shapes → call kernel          | Resolve dynamic dims from input tensors → complete shape inference → kernel cache lookup → call kernel |
| Kernel cache | Single kernel instance                       | Cached by dynamic dimension values                                                                     |

```python
# Fully static (default, recommended) — everything resolved at init
op = GemmFwdOp(M=2048, K=4096, N=1024, dtype=torch.float16)

# Dynamic M — resolved at forward time, kernel cached by M
op = RMSNormFwdOp(N=4096, dtype=torch.float16)
```

### Shape Inference (Codegen)

Agent generates an `_infer_output_shapes()` instance method on each Op from the manifest's `shape_rules`. The method accepts shape tuples and params, returns output name → shape mapping.

```yaml
# manifest shape_rules
shape_rules:
  - "y.shape == x.shape"
  - "weight.shape == (x.shape[dim],)"
```

```python
# generated method
def _infer_output_shapes(
    self, x_shape: tuple, weight_shape: tuple, dim: int = -1
) -> Dict[str, tuple]:
    return {"y": x_shape}
```

**Calling convention:**

- **Fully static op:** called once in `__init__`, result stored in `self._output_shapes`.
- **Op with dynamic dims:** called in `forward()` when dynamic dims are resolved. Result cached by dynamic dimension values.
- **Non-runtime consumers** (validator, graph compiler): can call with concrete shapes without constructing tensors.

**Inheritance in family-base hierarchies:**

| Scenario                                             | `_infer_output_shapes` defined at | Concrete op action    |
| ---------------------------------------------------- | --------------------------------- | --------------------- |
| Family shares shape logic                            | L2 family base                    | Inherits, no override |
| Family member has variant logic (e.g., multi-output) | L3 concrete op                    | Overrides             |
| Op inherits L1 directly                              | L3 concrete op                    | Agent generates       |

### Dtype Validation (Codegen)

Agent generates a `_validate_dtypes()` instance method from the manifest's `dtype` fields and `dtype_combos`.

```yaml
# manifest
inputs:
  x: {dtype: "float16 | bfloat16"}
  weight: {dtype: "same_as(x)"}
```

```python
# generated method
def _validate_dtypes(self, x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"x.dtype must be float16 or bfloat16, got {x.dtype}")
    if weight.dtype != x.dtype:
        raise ValueError(f"weight.dtype must match x.dtype, got {weight.dtype}")
```

`SUPPORTED_DTYPES` as a standalone class variable is superseded by this approach. Dtype constraints live in the manifest; Op code is generated from them.

### Roofline Evaluation (Codegen)

Agent generates roofline metadata as class-level declarations from the manifest's `roofline` section. Evaluation uses the same `self.*` attributes populated by `__init__`.

```yaml
# manifest
roofline:
  vars:
    M: "product(x.shape[:dim])"
    N: "x.shape[dim]"
  flops: "4 * M * N"
  bytes: "(2 * M * N + N) * elem_bytes"
```

```python
# generated class-level declarations
_roofline_vars = ["M", "N"]
_flops_expr = "4 * M * N"
_bytes_expr = "(2 * M * N + N) * elem_bytes"
```

```python
# base class provides eval_roofline()
def eval_roofline(self) -> Tuple[int, int]:
    ctx = {name: getattr(self, name) for name in self._roofline_vars}
    ctx["elem_bytes"] = torch.finfo(self.dtype).bits // 8
    return eval(self._flops_expr, ctx), eval(self._bytes_expr, ctx)
```

Roofline variable names, `__init__` keyword names, and `shape` dimension names all share the same namespace — defined once in the manifest, consumed uniformly.

### Consistency Enforcement

| Check                                                     | Mechanism                             | When     |
| --------------------------------------------------------- | ------------------------------------- | -------- |
| Generated code matches manifest                           | Validator (CI), extended L1/L2 checks | Every PR |
| `__init__` keywords match manifest dims + params          | Validator signature check             | Every PR |
| `_infer_output_shapes` consistent with `shape_rules`      | Validator shape check                 | Every PR |
| `_validate_dtypes` consistent with `dtype`/`dtype_combos` | Validator dtype check                 | Every PR |

## Reference

### Naming Conventions

#### Op Classes

```
{PascalCaseName}{Direction}Op
```

- **Direction** — mandatory: `Fwd` or `Bwd`.
- Exception: elementwise ops omit the direction suffix (`ReluOp`, `AddOp`).

The manifest key must exactly equal `cls.__name__`.

#### Kernel Classes

```
{PascalCaseName}{Direction}Kernel
```

- Exception: elementwise kernels omit the direction suffix (`ReluKernel`, `AddKernel`).

#### Kernel Dispatch (kernel_map)

A flat dict mapping dispatch keys to Kernel classes. The manifest declares this table; agents implement the listed Kernels.

```python
# Single-kernel op (family-base pattern)
@property
def default_kernel_map(self) -> Dict[str, Kernel]:
    return {self._kernel_key: self._kernel_cls}


# Multi-kernel pipeline (direct-inheritance pattern)
@property
def default_kernel_map(self) -> Dict[str, Kernel]:
    return {
        "mha_bwd_preprocess_kernel": FlashAttnBwdPreprocessKernel,
        "mha_bwd_kernel": MHABwdKernel,
        "mha_bwd_postprocess_kernel": FlashAttnBwdPostprocessKernel,
    }
```

- **Keys**: snake_case identifiers, decoupled from Kernel class names.
- **Values**: Kernel class names, must match `cls.__name__`.
- The table does not describe dispatch strategy. Strategy is a runtime concern.

#### Builder Functions

Kernel builder functions remain `snake_case`:

```python
def rms_norm_fwd(M, N, dtype, ...): ...
```

### Base Class Protocol

#### Op base class (`tileops/ops/op_base.py`)

| Attribute        | Type                          | Purpose                                               |
| ---------------- | ----------------------------- | ----------------------------------------------------- |
| `kernel`         | `Kernel`                      | Kernel instance used by `forward()`                   |
| `kernel_map`     | `Optional[Dict[str, Kernel]]` | Dispatched kernels keyed by name                      |
| `dtype`          | `Optional[torch.dtype]`       | Computation dtype                                     |
| `device`         | `Optional[str]`               | Device (default `'cuda'`)                             |
| `_output_shapes` | `Optional[Dict[str, tuple]]`  | Inferred output shapes (populated at init or forward) |

Abstract interface: `default_kernel_map` (property), `forward()`.

Manifest-driven methods (generated by agent):

- `_infer_output_shapes(...)` → output name → shape
- `_validate_dtypes(...)` → raise on invalid dtypes
- `eval_roofline()` → (flops, bytes)

#### Kernel base class (`tileops/kernels/kernel_base.py`)

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

### Conventions in Code, Not Documentation

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
