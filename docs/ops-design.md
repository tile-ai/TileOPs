# Op Interface Design

Step-by-step playbook for scaffolding a new op from a manifest entry, plus short concepts and links to [`ops-design-reference.md`](ops-design-reference.md) for the authoritative per-slot rules.

## Concepts

Every operator is split into two classes — **Op** (host-side: validates inputs, dispatches to Kernel, assembles output) and **Kernel** (device-side: owns the TileLang program, tile configuration, JIT compilation). The two layers are independently modifiable — changing a Kernel's tile strategy does not require changing the Op.

### Class hierarchy

```
Op                          ← L1: thin base, shared by all ops
  └── FamilyBase            ← L2: family-specific forward() flow (optional)
        └── ConcreteOp      ← L3: leaf class emitted by the scaffold
```

- **L1 (`Op`, [`tileops/ops/op_base.py`](../tileops/ops/op_base.py)):** provides `__call__`, `dispatch_kernel()`, `autotune()`, the default `_cache_key()`, and `NotImplementedError` stubs for the three codegen methods (`_infer_output_shapes`, `_validate_dtypes`, `eval_roofline` — `FIXME(staged-rollout)`, per PR #1012).
- **L2 (`FamilyBase`):** per-family shared `forward()` pipeline (one per family). **Not produced by this playbook** — see [Family-Base Refactoring (Future Work)](#family-base-refactoring-future-work).
- **L3 (`ConcreteOp`):** this playbook's target. New ops start by inheriting L1 directly (T2 shape). Once 2-3 ops accumulate in a family with identical `forward()` flow, extract an L2 base via refactoring.

### Execution timing

**Do it at the first moment all required information is known, do it once, cache the result.**

| Op category    | When all info is known                                      | Behaviour                                                |
| -------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| Fixed-rank     | `__init__` (all dims provided)                              | `_infer_output_shapes` runs once at init.                |
| Arbitrary-rank | `__init__` for `static_dims`; `forward` for everything else | Kernel built on first encounter, cached by `_cache_key`. |

`_validate_dtypes` runs on every `forward()` call — dtype validity depends on the actual tensors passed, not just their shapes. Roofline timing and formula semantics are defined in [roofline.md](roofline.md). See [Parameter Design](ops-design-reference.md#parameter-design) for fixed-rank vs arbitrary-rank details and [Codegen Details](ops-design-reference.md#codegen) for calling conventions.

## Scaffolding an Op from a Manifest Entry

The scaffold emits a T2 (L1-direct) op file from one manifest entry. Each step has typed **Input** (manifest fields consumed), **Output** (the code fragment produced), **Validation** (concrete check), and a **Reference** link to the authoritative slot rule in [`ops-design-reference.md`](ops-design-reference.md). All examples are for `CumsumFwdOp` ([`tileops/ops_manifest.yaml`](../tileops/ops_manifest.yaml), [`tileops/ops/reduction/cumsum.py`](../tileops/ops/reduction/cumsum.py)).

### Step 1: File header + imports

**Input.** `source.kernel_map` values (Kernel classes to import).

**Output.**

```python
"""Cumulative sum operator (host-side Op layer).

Provides:
  - CumsumFwdOp: y = cumsum(x, dim=-1)
"""

import math
from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.cumulative import CumulativeKernel

from ..op_base import Op
```

**Validation.** Every concrete-Kernel import matches one `source.kernel_map` value verbatim. The `Kernel` base import and `..op_base` relative import are fixed.

**Reference.** [Slot S1](ops-design-reference.md#slot-s1), [S2](ops-design-reference.md#slot-s2), [S3](ops-design-reference.md#slot-s3), [S4](ops-design-reference.md#slot-s4).

### Step 2: Class declaration + docstring + `__all__`

**Input.** Manifest entry key (= class name); `signature.inputs`, `signature.params`, `static_dims`, per-tensor `dtype` (Args block content).

**Output.**

```python
__all__ = ["CumsumFwdOp"]


class CumsumFwdOp(Op):
    """Cumulative sum operator: y = cumsum(x, dim=-1).

    Output has the same shape and dtype as input.

    Args:
        N: Hidden dimension (size along the reduction axis), committed
            at ctor via ``static_dims: N: "x.shape[dim]"``.
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """
```

**Validation.** Class name ≡ manifest entry key, byte-exact (`CumsumFwdOp`). Every `Args:` entry appears as an `__init__` kwarg in Step 3; no extras.

**Reference.** [Slot S5](ops-design-reference.md#slot-s5), [S6](ops-design-reference.md#slot-s6), [S7](ops-design-reference.md#slot-s7).

### Step 3: `_static_axes` + `__init__` signature and body

**Input.** `static_dims` (literal-axis → class-level `_static_axes` frozenset; param-axis → empty class-level default, bind at `forward()` after `dim % x.ndim` normalization); `signature.params`; `dtype`.

**Output.**

```python
    # static_dims: N: "x.shape[dim]" — the axis is param-dependent
    # (may be negative like dim=-1), so the concrete (input_index,
    # axis) pair cannot be resolved until x.ndim is known. Leave the
    # class-level default empty; bind in forward() after normalizing
    # dim against x.ndim (Op base requires a non-negative axis).
    _static_axes: frozenset[tuple[int, int]] = frozenset()

    def __init__(
        self,
        *,
        N: int,
        dtype: torch.dtype,
        dim: int = -1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N = N
        self.dtype = dtype
        self.dim = dim
        self.tune = tune
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        # M is not a static_dim — deferred to forward() where x.ndim
        # is known and M is derived from the non-reduction axes.
        self._kernel_cache: Dict[tuple, Kernel] = {}
```

**Validation.** Every `__init__` kwarg has a manifest source (`static_dims` or `signature.params` or `dtype`); no extras except `kernel_map` / `tune`. In particular, `M` is NOT a ctor kwarg — `CumsumFwdOp.static_dims` declares only `N`, so `M` is derived at forward time. Keyword-only via `*`, no defaults on `static_dims` entries. `_static_axes` matches the manifest axis form (literal-int axis → populated class-level frozenset; param-dependent axis → empty class-level default, bound at forward after `dim % x.ndim` normalization).

**Reference.** [Slot S21](ops-design-reference.md#slot-s21), [S12](ops-design-reference.md#slot-s12), [S13](ops-design-reference.md#slot-s13).

### Step 4: `default_kernel_map` + `forward`

**Input.** Manifest `source.kernel_map`; `signature.inputs`; `static_dims` (for the forward-time commitment check); `shape_rules` (for `dim` range validation).

**Output.**

```python
    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"cumulative_fwd": CumulativeKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_dtypes(x)
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        # Validate `dim` against shape_rule `-x.ndim <= dim < x.ndim`
        # and normalize to a non-negative axis (Op._static_axes contract).
        if not -x.ndim <= self.dim < x.ndim:
            raise ValueError(
                f"dim {self.dim} out of range for x.ndim={x.ndim}")
        dim = self.dim % x.ndim
        # Validate the static_dims commitment: x.shape[dim] == N
        if x.shape[dim] != self.N:
            raise ValueError(
                f"static_dim mismatch: expected x.shape[{dim}] == {self.N}, "
                f"got {x.shape[dim]}")
        # Bind _static_axes now that the concrete axis is known.
        self._static_axes = frozenset({(0, dim)})
        # Derive M (product of non-reduction dims) and cache kernel by (M,).
        M = math.prod(s for i, s in enumerate(x.shape) if i != dim)
        self.M = M  # stored for eval_roofline
        key = (M,)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["cumulative_fwd"](
                M, self.N, "sum", self.dtype, tune=self.tune)
        kernel = self._kernel_cache[key]
        # Move reduction axis to last, reshape to (M, N), compute, restore.
        orig_shape = x.shape
        x2 = x.movedim(dim, -1).contiguous().reshape(M, self.N)
        y2 = kernel(x2)
        if self.N_padded != self.N:
            y2 = y2[:, : self.N]
        y = y2.reshape(*orig_shape[:dim], *orig_shape[dim + 1:], self.N)
        return y.movedim(-1, dim)
```

**Validation.** `default_kernel_map` keys / values match manifest `source.kernel_map` verbatim. `forward` calls `self._validate_dtypes(...)` first (not inline dtype comparisons — that is Step 5's job). Every `static_dims` commitment is validated against the actual tensor shape at the normalized axis before the kernel is called. `_static_axes` is bound from the normalized (non-negative) axis before the kernel cache lookup. Padding trim emitted iff the kernel operates on `align_up(N, DEFAULT_ALIGNMENT)` (`self.N_padded != self.N`).

**Reference.** [Slot S14](ops-design-reference.md#slot-s14), [S15](ops-design-reference.md#slot-s15), [S16](ops-design-reference.md#slot-s16).

### Step 5: `_infer_output_shapes` + `_validate_dtypes`

**Input.** Manifest `shape_rules` (for S17); per-tensor `dtype` and `dtype_combos` (for S18).

**Output.**

```python
class CumsumFwdOp(Op):
    ...

    def _infer_output_shapes(self, x_shape: tuple) -> Dict[str, tuple]:
        return {"y": x_shape}

    def _validate_dtypes(self, x: torch.Tensor) -> None:
        if x.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError(f"x.dtype must be float32/float16/bfloat16, got {x.dtype}")
```

**Validation.** `python scripts/validate_manifest.py` exercises both methods at CI (PR #1005). **L2 parity:** `_infer_output_shapes(mock_inputs)` must agree with `shape_rules`. **L3 parity:** `_validate_dtypes` must accept exactly the declared `dtype` union / `dtype_combos` and reject everything else — disagreement is a hard error. Opting out (for GPU-only ops whose methods cannot be invoked in a CPU-only validator context) requires the manifest entry to declare `parity_opt_out: [shape_parity, dtype_parity]`; do not use it to silence a real disagreement.

**Reference.** [Slot S17](ops-design-reference.md#slot-s17), [S18](ops-design-reference.md#slot-s18).

### Step 6: `eval_roofline`

**Input.** Manifest `roofline.vars`, `roofline.flops`, `roofline.bytes`.

**Output.**

```python
class CumsumFwdOp(Op):
    ...

    def eval_roofline(self) -> tuple[int, int]:
        flops = self.M * self.N
        bytes_ = 2 * self.M * self.N * self.dtype.itemsize
        return flops, bytes_
```

**Validation.** The body is **plain Python** reading `self.*` attributes. No class-level roofline expression strings, no `ast.parse`, no shared L1 evaluator — prohibited by [`roofline.md §4.4.6` Evaluator Surface Boundary](roofline.md#446-evaluator-surface-boundary). Return type is `tuple[int, int]`, not `float` or `numpy`. Expressions derive directly from `roofline.vars` bindings + `roofline.flops` + `roofline.bytes`; see [`roofline.md §4.4` Op Codegen](roofline.md#44-op-codegen).

**Reference.** [Slot S19](ops-design-reference.md#slot-s19).

### Step 7: Package registration

**Input.** The class name (Step 2) and the op's source filename.

**Output (append to `tileops/ops/reduction/__init__.py`):**

```python
# --- CumulativeKernel ops ---
from .cumsum import CumsumFwdOp
```

…with a matching entry added to the module's `__all__` list.

**Validation.** The import sits under its family's grouping comment block; a matching `__all__` entry is present (otherwise `from tileops.ops.reduction import *` silently drops the op).

**Reference.** [Slot S20](ops-design-reference.md#slot-s20).

### Slot coverage

| Step | Slots produced                                                                                                       |
| ---- | -------------------------------------------------------------------------------------------------------------------- |
| 1    | S1, S2, S3, S4                                                                                                       |
| 2    | S5, S6, S7                                                                                                           |
| 3    | S21, S12, S13                                                                                                        |
| 4    | S14, S15, S16                                                                                                        |
| 5    | S17, S18                                                                                                             |
| 6    | S19                                                                                                                  |
| 7    | S20                                                                                                                  |
| —    | S8-S11: reserved — intentionally skipped from slot iteration (T1 thin-wrapper slots, out of scope for this playbook) |

## Out of Scope

This playbook emits exactly the 17 slots above. The following are **not** produced by the scaffold — each needs separate treatment:

- **Family-specific protocol variables.** `_op_kind` (reduction), `_kernel_key`, `_kernel_cls` (norm + reduction T1 wrappers), `_kernel_handles_padding`, `_op_name`, `kernel_cls`. Kernel-dispatch-convention-dependent; cannot be mechanically derived from the manifest. See [Family-Base Protocol (Appendix)](ops-design-reference.md#base-class-protocol).
- **Optional hooks.** `_pad_value`, `_validate_dim`, `_pre_kernel`, `_post_kernel`. Op-specific business logic (e.g., `ArgmaxFwdOp._pad_value = -inf`). See [Optional Hooks (Appendix)](ops-design-reference.md#optional-hooks-appendix).
- **`_cache_key` override.** The default projection via `_static_axes` is correct but sometimes over-fragmenting. Override logic depends on what subset of the input shape the kernel actually depends on — kernel-math-specific.
- **Family-base (T1) subclassing.** See [Family-Base Refactoring (Future Work)](#family-base-refactoring-future-work).
- **Kernel implementations themselves.** The playbook's scope is the Op (host) layer. See [Implementing a Kernel](#implementing-a-kernel) for the kernel-side interface surface.

## Implementing a Kernel

Brief reference surface for the device-side class that a scaffolded Op depends on. Kernel implementation is not covered by the scaffold-op skill.

| Interface             | Required | Description                                                   |
| --------------------- | -------- | ------------------------------------------------------------- |
| `__init__(self, ...)` | yes      | Receives shape params and dtype; builds the TileLang program. |
| `forward(self, ...)`  | yes      | Launches the compiled kernel; called by Op's `forward()`.     |
| `kernel`              | yes      | Attribute. The TileLang program builder (JIT-compiled).       |
| `default_config`      | no       | Property. Default tile configuration dict.                    |
| `autotune_configs`    | no       | Class variable. Search space for autotuning.                  |
| `supported_archs`     | no       | Class variable. List of supported GPU SM versions.            |

See [Kernel base class attributes](ops-design-reference.md#base-class-protocol) for the full attribute table.

## Family-Base Refactoring (Future Work)

The scaffold emits T2 (L1-direct) ops only. Once a family accumulates 2-3 ops sharing an identical `forward()` flow, extract an L2 family base via refactoring; concrete ops then become T1 thin wrappers declaring family protocol variables (`_op_kind`, `_kernel_key`, `_kernel_cls`, …). This transformation is driven by a separate family-specific skill, not the scaffold-op. See [Development Path](ops-design-reference.md#development-path) for when to extract an L2 base and [Adding a New Family Base](ops-design-reference.md#adding-a-new-family-base) for the step-by-step process.

## Further Reference

- [Slot Rules](ops-design-reference.md#slot-rules) — full Rule / Derivation / Example / Common mistakes per slot
- [Codegen Details](ops-design-reference.md#codegen) — calling conventions, inheritance rules, consistency enforcement
- [Base Class Protocol](ops-design-reference.md#base-class-protocol) — `Op` and `Kernel` base class attributes
- [Naming Conventions](ops-design-reference.md#naming-conventions) — class / `kernel_map` / builder function rules
- [Parameter Design](ops-design-reference.md#parameter-design) — static vs dynamic op comparison
- [manifest.md](manifest.md) — manifest entry structure, `static_dims`, `shape_rules`, `roofline`, `parity_opt_out`
- [roofline.md](roofline.md) — roofline formula syntax, codegen, evaluator surface boundary
