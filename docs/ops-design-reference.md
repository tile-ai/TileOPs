# Op Interface Design — Reference

Slot-keyed rule dictionary consumed on demand by [ops-design.md](ops-design.md) and the `op-scaffold` skill. Each `### Slot S{N}` entry states the authoritative **Rule**, its manifest **Derivation**, a concrete **Example** modelled on [`tileops/ops/reduction/cumsum.py`](../tileops/ops/reduction/cumsum.py), and **Common mistakes**. Non-slot content lives in the appendices. Slot IDs S8–S11 are intentionally absent (reserved during iteration for T1 thin-wrapper slots later declared out of scope).

## Slot Rules

### Slot S1: <a id="slot-s1"></a> Module docstring

- **Rule.** File begins with a triple-quoted docstring: one-line `<ClassName>: <purpose>` summary followed by a short behaviour paragraph.
- **Derivation.** Class name from S6; purpose templated from `signature` semantics.
- **Example.**
  ```python
  """Cumulative sum operator (L2 Op layer).

  Provides:
    - CumsumFwdOp: y = cumsum(x, dim=-1)
  """
  ```
- **Common mistakes.** Referencing tile sizes or kernel-internals in the module docstring; omitting the one-line purpose.

### Slot S2: <a id="slot-s2"></a> Import — `Kernel` base class

- **Rule.** Import `Kernel` whenever `kernel_map` typing is annotated.
- **Derivation.** Fixed import path.
- **Example.**
  ```python
  from tileops.kernels.kernel_base import Kernel
  ```
- **Common mistakes.** Aliasing the import; re-exporting `Kernel`.

### Slot S3: <a id="slot-s3"></a> Import — concrete `Kernel` class

- **Rule.** One absolute import from `tileops.kernels.*` per Kernel class listed in the manifest `kernel_map`.
- **Derivation.** Manifest `kernel_map` values.
- **Example.**
  ```python
  from tileops.kernels.reduction.cumulative import CumulativeKernel
  ```
- **Common mistakes.** Relative cross-package import; importing a kernel not in `kernel_map`.

### Slot S4: <a id="slot-s4"></a> Import — `Op` base class

- **Rule.** Relative import of the L1 `Op` base class.
- **Derivation.** Fixed: `from ..op_base import Op` (or `from .op_base import Op` for ops directly under `tileops/ops/`).
- **Example.**
  ```python
  from ..op_base import Op
  ```
- **Common mistakes.** Absolute `tileops.ops.op_base` import — violates the relative-import rule in `.claude/rules/code-style.md`.

### Slot S5: <a id="slot-s5"></a> `__all__`

- **Rule.** `__all__` contains exactly the concrete op class name (S6).
- **Derivation.** `[<ClassName>]`.
- **Example.**
  ```python
  __all__ = ["CumsumFwdOp"]
  ```
- **Common mistakes.** Re-exporting the `Kernel` class; omitting `__all__`.

### Slot S6: <a id="slot-s6"></a> Class name

- **Rule.** `{PascalCaseName}{Direction}Op`, `Direction` ∈ {`Fwd`, `Bwd`}, no exceptions. Manifest entry key must equal `cls.__name__` verbatim.
- **Derivation.** Manifest entry key.
- **Example.**
  ```python
  class CumsumFwdOp(Op):
  ```
- **Common mistakes.** Direction suffix missing; abbreviation mis-casing (see [Naming Conventions (Appendix)](#naming-conventions-appendix)).

### Slot S7: <a id="slot-s7"></a> Class docstring

- **Rule.** One-sentence summary; `Args:` block enumerating every `__init__` kwarg (S12) with type and short description; optional `Example:` block.
- **Derivation.** `Args` block from manifest `signature.params` + `static_dims` + `dtype`.
- **Example.**
  ```python
  class CumsumFwdOp(Op):
      """Cumulative sum operator: y = cumsum(x, dim=-1).

      Output has the same shape and dtype as input.

      Args:
          M: Number of rows (product of all dims except last).
          N: Hidden dimension (last dim).
          dtype: Data type (float32, float16, or bfloat16).
          kernel_map: Optional override for kernel dispatch.
          tune: Whether to autotune (default False).
      """
  ```
- **Common mistakes.** Args out of sync with `__init__`; listing tensor inputs in `Args` (they belong to `forward`).

### Slot S12: <a id="slot-s12"></a> `__init__` signature

- **Rule.** Keyword-only via `*`. Kwarg block order: (1) `static_dims` entries in manifest key order, no defaults; (2) `dtype`; (3) `signature.params` entries in manifest key order; (4) `kernel_map` and `tune` last.
- **Derivation.** Manifest `static_dims` + `dtype` + `signature.params`.
- **Example.**
  ```python
  def __init__(
      self,
      *,
      M: int,
      N: int,
      dtype: torch.dtype,
      kernel_map: Optional[Dict[str, Kernel]] = None,
      tune: bool = False,
  ):
  ```
- **Common mistakes.** Missing `*` (positional accepted); `static_dims` kwargs with defaults; params/static_dims block order inverted; kwargs not backed by a manifest source.

### Slot S13: <a id="slot-s13"></a> `__init__` body

- **Rule.** Body sequence: (a) `self.<name> = <name>` per kwarg; (b) `self.dispatch_kernel(kernel_map)`; (c) `self.kernel = self.kernel_map[<key>](...)`; (d) `self._infer_output_shapes(...)` when shapes are fully known at init.
- **Derivation.** Each `self.*` assignment mirrors one S12 kwarg. Kernel-build positional args follow the kernel class's ctor (kernel author's API).
- **Example.**
  ```python
  self.M = M
  self.N = N
  self.dtype = dtype
  self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
  self.dispatch_kernel(kernel_map)
  self.kernel = self.kernel_map["cumulative_fwd"](M, N, "sum", dtype, tune=tune)
  ```
- **Common mistakes.** `_infer_output_shapes` called before `dispatch_kernel`; hard-coding the kernel class instead of routing through `self.kernel_map`.

### Slot S14: <a id="slot-s14"></a> `default_kernel_map` property

- **Rule.** `@property` returning the manifest `kernel_map` dict literal with `snake_case` keys and Kernel-class values.
- **Derivation.** Manifest `kernel_map`, verbatim.
- **Example.**
  ```python
  @property
  def default_kernel_map(self) -> Dict[str, Kernel]:
      return {"cumulative_fwd": CumulativeKernel}
  ```
- **Common mistakes.** Class-level dict (not a property); keys that duplicate the class name instead of being dispatch strings.

### Slot S15: <a id="slot-s15"></a> `forward` signature

- **Rule.** Positional tensor parameters in manifest `signature.inputs` order; return annotation `torch.Tensor` or `Tuple[torch.Tensor, ...]` matching `signature.outputs`.
- **Derivation.** Manifest `signature.inputs` for names; `signature.outputs` for return annotation.
- **Example.**
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
  ```
- **Common mistakes.** Keyword-only tensor parameters; non-tensor kwargs in `forward` (they belong to `__init__`).

### Slot S16: <a id="slot-s16"></a> `forward` body

- **Rule.** Body sequence: (a) `self._validate_dtypes(...)`; (b) `.contiguous()` + reshape to the kernel layout; (c) validate each `static_dims` commitment against the actual tensor shape; (d) call the kernel; (e) restore the original shape.
- **Derivation.** Validation expressions come from each `static_dims` entry's `<tensor>.shape[<axis>]` RHS.
- **Example.**
  ```python
  if not x.is_cuda:
      raise ValueError("x must be a CUDA tensor")
  if x.dtype != self.dtype:
      raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
  if x.shape[-1] != self.N:
      raise ValueError(f"Expected last dim {self.N}, got {x.shape[-1]}")
  orig_shape = x.shape
  x = x.contiguous().reshape(-1, self.N)
  y = self.kernel(x)
  return y.reshape(orig_shape)
  ```
- **Common mistakes.** Skipping `_validate_dtypes`; reshape before `.contiguous()`; not restoring the original shape.

### Slot S17: <a id="slot-s17"></a> `_infer_output_shapes` method body

- **Rule.** Signature takes `<input>_shape: tuple` per manifest `signature.inputs`, returns `Dict[str, tuple]` keyed by output name. The L1 base raises `NotImplementedError` as a `FIXME(staged-rollout)` stub; each concrete op supplies a complete body. PR #1005's validator exercises the method with mock inputs at CI and reports disagreement with `shape_rules` as a hard L2 error.
- **Derivation.** Manifest `shape_rules` (see [manifest.md § Rules](manifest.md#rules)).
- **Example.**
  ```python
  def _infer_output_shapes(self, x_shape: tuple) -> Dict[str, tuple]:
      return {"y": x_shape}
  ```
- **Common mistakes.** Shape tuple disagreeing with `shape_rules` (hard L2 error); accepting/returning `torch.Tensor` instead of shape tuples; `parity_opt_out` used to silence a genuine disagreement.

### Slot S18: <a id="slot-s18"></a> `_validate_dtypes` method body

- **Rule.** Positional parameters match `signature.inputs`; raises `ValueError` on invalid dtype combinations. L1 stub raises `NotImplementedError` (FIXME staged-rollout). PR #1005's validator exhaustively probes `dtype_combos` / declared unions + out-of-union negatives and reports divergence as hard L3 error.
- **Derivation.** Manifest `dtype` (union) and `dtype_combos`.
- **Example.**
  ```python
  def _validate_dtypes(self, x: torch.Tensor) -> None:
      if x.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
          raise ValueError(f"x.dtype must be float32/float16/bfloat16, got {x.dtype}")
  ```
- **Common mistakes.** Accepting a dtype outside the declared union; rejecting a dtype listed in `dtype_combos`; ignoring `same_as(ref)` linkage between inputs.

### Slot S19: <a id="slot-s19"></a> `eval_roofline` method body

- **Rule.** Codegen emits a complete plain-Python body reading `self.*` attributes. Per [`roofline.md` §4.4.6](roofline.md#446-evaluator-surface-boundary) (Evaluator Surface Boundary) there is NO shared AST evaluator on L1 and NO class-level `_flops_expr` / `_bytes_expr` / `_roofline_vars` strings. L1 stub raises `NotImplementedError` (FIXME staged-rollout).
- **Derivation.** Manifest `roofline.vars`, `roofline.flops_expr`, `roofline.bytes_expr`; see [`roofline.md` §4.4](roofline.md#44-op-codegen).
- **Example.**
  ```python
  def eval_roofline(self) -> tuple[int, int]:
      flops = 4 * self.M * self.N
      bytes_ = (2 * self.M * self.N + self.N) * self.dtype.itemsize
      return flops, bytes_
  ```
- **Common mistakes.** Class-level `_flops_expr` / `_bytes_expr` / `_roofline_vars` strings (prohibited by §4.4.6); any `ast.parse` or shared `_safe_eval` path; returning `float` or `numpy` types (contract is `tuple[int, int]`).

### Slot S20: <a id="slot-s20"></a> Package `__init__.py` registration

- **Rule.** `tileops/ops/{family}/__init__.py` gains one `from .<module> import <ClassName>` line plus a matching `<ClassName>` entry in `__all__`, placed under the family's grouping comment block.
- **Derivation.** Class name (S6) and the op's source filename.
- **Example.**
  ```python
  # --- CumulativeKernel ops ---
  from .cumsum import CumsumFwdOp
  ```
- **Common mistakes.** Import outside its family grouping comment; missing `__all__` entry (silently breaks `import *`).

### Slot S21: <a id="slot-s21"></a> `_static_axes` class attribute

- **Rule.** Each concrete op declares `_static_axes: frozenset[tuple[int, int]]` of `(input_index, axis)` pairs committed at constructor time. `input_index` is the positional index in `signature.inputs`; `axis` is non-negative. Empty frozenset is legal (means "no axes committed at ctor").
- **Derivation.** Manifest `static_dims`; for each entry `<kwarg>: <tensor>.shape[<axis>]`, emit `(input_index_of_<tensor>, <axis>)`. PyTorch-aligned reductions with `dim=None` → empty frozenset (see [manifest.md § Empty static_dims](manifest.md#empty-static_dims)).
- **Example.**
  ```python
  class CumsumFwdOp(Op):
      _static_axes = frozenset({(0, 1)})  # x.shape[1] is committed as N
  ```
- **Common mistakes.** Omitting `_static_axes` when `static_dims` is non-empty (default `frozenset()` silently disables static-axis projection in `_cache_key`); negative axis indices (must be non-negative per [`op_base.py`](../tileops/ops/op_base.py)); empty `_static_axes` without overriding `_cache_key` (emits a once-per-type `UserWarning` — see [Optional Hooks (Appendix)](#optional-hooks-appendix)).

## Family-Base Protocol (Appendix) <a id="base-class-protocol"></a>

Per-family protocol variables, declared by L2 bases and overridden by L3 ops.

| Variable                  | Family          | Purpose                                                   |
| ------------------------- | --------------- | --------------------------------------------------------- |
| `_kernel_key`             | norm, reduction | Kernel-map lookup key                                     |
| `_kernel_cls`             | norm, reduction | Kernel class reference                                    |
| `_op_kind`                | reduction       | Reduction kind string (`"sum"`, `"mean"`, `"std"`, etc.)  |
| `_kernel_handles_padding` | reduction       | `True` → kernel uses masked loads, skip host-side padding |
| `_op_name`                | elementwise     | `torch.library.custom_op` registration key                |
| `kernel_cls`              | elementwise     | Kernel class reference                                    |

**The `op-scaffold` skill does NOT emit these variables** — kernel-dispatch-convention-dependent (e.g., `VectorNormKernel` uses `{"l1", "l2", "inf"}`, `ReduceKernel` uses `{"sum", "mean", ...}`); filled in during family-specific refactoring (future skill). Adding a new protocol variable requires updating the L2 base, all concrete ops, and the manifest schema if applicable.

### `Op` base class attributes ([`tileops/ops/op_base.py`](../tileops/ops/op_base.py))

| Attribute        | Type                          | Purpose                                                                                      |
| ---------------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| `kernel`         | `Kernel`                      | Kernel instance used by `forward()`                                                          |
| `kernel_map`     | `Optional[Dict[str, Kernel]]` | Dispatched kernels keyed by name                                                             |
| `dtype`          | `Optional[torch.dtype]`       | Computation dtype                                                                            |
| `device`         | `Optional[str]`               | Device (default `'cuda'`)                                                                    |
| `_output_shapes` | `Optional[Dict[str, tuple]]`  | Inferred output shapes (populated at init or forward)                                        |
| `_static_axes`   | `frozenset[tuple[int, int]]`  | Static axes as `(input_index, axis)` pairs (default `frozenset()`); consumed by `_cache_key` |

Abstract interface: `default_kernel_map` (property), `forward()`. Manifest-driven methods (codegen-emitted by concrete ops): `_infer_output_shapes`, `_validate_dtypes`, `eval_roofline`.

### `Kernel` base class attributes ([`tileops/kernels/kernel_base.py`](../tileops/kernels/kernel_base.py))

| Attribute          | Type                    | Purpose                                        |
| ------------------ | ----------------------- | ---------------------------------------------- |
| `dtype`            | `Optional[torch.dtype]` | Data type                                      |
| `config`           | `Dict[str, Any]`        | Tile configuration (block sizes, stages, etc.) |
| `autotune_configs` | `Optional[list[dict]]`  | Search space for autotuning                    |
| `supported_archs`  | `Optional[list[int]]`   | GPU SM versions (e.g., `[80, 86, 89, 90]`)     |
| `kernel`           | `Callable`              | Compiled TileLang kernel function              |

Abstract interface: `forward()`. Key methods: `init_config(config, tune)`, `autotune(warmup, rep)`.

## Optional Hooks (Appendix)

Hooks family bases expose for op-specific semantics. The `op-scaffold` skill does NOT emit these.

| Hook              | Family    | Default                     | Override example                                                       |
| ----------------- | --------- | --------------------------- | ---------------------------------------------------------------------- |
| `_pad_value()`    | reduction | `0.0` (neutral for sum)     | `ArgmaxFwdOp._pad_value → -inf` (`tileops/ops/reduction/argmax.py:61`) |
| `_validate_dim()` | reduction | accept `int` or `list[int]` | `ArgmaxFwdOp._validate_dim` restricts to scalar `int`                  |
| `_pre_kernel()`   | reduction | identity                    | `AllFwdOp._pre_kernel` converts unsupported storage dtypes to fp32     |
| `_post_kernel()`  | reduction | identity                    | Convert kernel output dtype to the manifest-declared output dtype      |

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

- **Op class:** `{PascalCaseName}{Direction}Op`. `Direction` ∈ {`Fwd`, `Bwd`}, mandatory. Manifest key must equal `cls.__name__`. Abbreviation casing: `RMSNormFwdOp`, `SSDDecodeOp` — fully uppercase per `.claude/rules/code-style.md`. Slot [S6](#slot-s6).
- **Kernel class:** `{PascalCaseName}{Direction}Kernel`. Same direction-suffix rule.
- **`kernel_map` keys:** `snake_case`, decoupled from Kernel class names. Values must match the Kernel `cls.__name__`. The table does not describe dispatch strategy. Slot [S14](#slot-s14).
- **Builder functions:** `snake_case`, e.g. `def rms_norm_fwd(M, N, dtype, ...): ...`.
- **Filenames:** all-lowercase with underscores. Multi-word abbreviations stay fully lowercase (`rms_norm.py`, `ssd_decode.py`; never `RMSNorm.py` or `Ssd_decode.py`). Norm-related names never contract (`rms_norm`, not `rmsnorm`).

## Codegen Details (Appendix) <a id="codegen"></a>

The manifest ([`ops_manifest.yaml`](../tileops/ops_manifest.yaml)) is the sole source of truth. Dtype validation and shape inference derive from manifest; roofline codegen is defined in [roofline.md](roofline.md).

### Parameter design <a id="parameter-design"></a>

Three time points: (1) manifest — constraint structure; (2) `__init__` — user commits `static_dims` values; (3) `forward` — shapes concrete, commitments validated. See [manifest.md § `static_dims`](manifest.md#static_dims).

|                          | Fixed-rank op           | Arbitrary-rank op                                                  |
| ------------------------ | ----------------------- | ------------------------------------------------------------------ |
| Manifest has `shape`     | yes                     | no                                                                 |
| `__init__` shape source  | `shape` dimension names | `static_dims`                                                      |
| Undeclared dimensions    | none                    | derived from tensor at forward time                                |
| Kernel construction time | init (all dims known)   | init (`static_dims` known) or forward (first encounter, cached)    |
| Forward cache keying     | N/A (single kernel)     | `_cache_key(*input_shapes)` — default non-static axes, overridable |

### Calling conventions

- **Fully static op:** `_infer_output_shapes` called once in `__init__`, result stored as an instance attribute.
- **Op with dynamic dims:** `_infer_output_shapes` called in `forward()` once dynamic dims resolve; kernel construction cached by `_cache_key(*input_shapes)`.
- **`_validate_dtypes`:** runs on every `forward()` call.
- **Non-runtime consumers** (validator, graph compiler): call `_infer_output_shapes` with concrete shape tuples without constructing tensors. Roofline consumers use interfaces in [`roofline.md`](roofline.md).

### Inheritance in family-base hierarchies

| Scenario                                             | Codegen method defined at | Concrete op action    |
| ---------------------------------------------------- | ------------------------- | --------------------- |
| Family shares logic                                  | L2 family base            | Inherits, no override |
| Family member has variant logic (e.g., multi-output) | L3 concrete op            | Overrides             |
| Op inherits L1 directly (T2)                         | L3 concrete op            | Scaffold emits body   |

### Consistency enforcement

| Check                                                    | Mechanism                                   |
| -------------------------------------------------------- | ------------------------------------------- |
| Manifest schema and declared fields are well-formed      | Validator (CI), L0 checks                   |
| `__init__` params match manifest `params`                | Validator signature check (L1)              |
| `static_dims` keys are `__init__` parameters             | Validator signature check (L1)              |
| `shape_rules` syntax is valid                            | Validator `shape_rules` parsing (L2)        |
| `_infer_output_shapes` output satisfies `shape_rules`    | Validator infer-shape parity (L2; PR #1005) |
| `dtype`/`dtype_combos` strings are valid                 | Validator dtype conformance (L3)            |
| `_validate_dtypes` matches `dtype_combos` / dtype unions | Validator dtype parity (L3; PR #1005)       |
| Empty `static_dims` without `_cache_key` override        | `Op` base class runtime warning             |

Checks beyond this table are tracked as separate issues, not as spec status.

**Parity check coverage.** The L2 / L3 parity checks compare the manifest spec against the concrete method the op class defines. When the class has not migrated to the codegen protocol, the validator emits a **warning** naming the missing method — the gap is surfaced, never silently passed. When the method exists, the parity check runs and any disagreement is a hard L2 / L3 error. A manifest entry may declare `parity_opt_out: [shape_parity, dtype_parity]` (or `parity_opt_out: true` for both) to suppress the warning for documented GPU-only ops whose method cannot be invoked in a CPU-only validator context; see [manifest.md § Entry Structure](manifest.md#entry-structure). Do not use `parity_opt_out` to silence a genuine disagreement.

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
