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

- **L1 (`Op`):** shared host-side plumbing (dispatch, get-or-build kernel caching, kernel enumeration, autotune) plus the contracts for the three codegen methods (`_infer_output_shapes`, `_validate_dtypes`, `eval_roofline`).
- **L2 (`FamilyBase`):** per-family shared `forward()` pipeline (one per family). **Not produced by this playbook** — see [Family-Base Refactoring](#family-base-refactoring).
- **L3 (`ConcreteOp`):** this playbook's target. New ops start by inheriting L1 directly (T2 shape); see [Family-Base Refactoring](#family-base-refactoring) for when a family graduates to L2.

### Execution timing

**Do it at the first moment all required information is known, do it once, cache the result.**

| Op category    | When all info is known                                      | Behaviour                                 |
| -------------- | ----------------------------------------------------------- | ----------------------------------------- |
| Fixed-rank     | `__init__` (all dims provided)                              | `_infer_output_shapes` runs once at init. |
| Arbitrary-rank | `__init__` for `static_dims`; `forward` for everything else | `_infer_output_shapes` runs per shape.    |

**Dtype is not a constructor parameter when the inputs determine it.** An op reads it from the input tensors in `forward()`: a caller who passes fp16 tensors gets the fp16 kernel without having said so twice, and an op can no longer be constructed in a state that disagrees with the tensors it is about to be handed.

An output dtype is determined by the inputs when it is `same_as(...)`, `promote_int_to_float(...)`, one concrete dtype, or a union equal to some input's. When some output dtype is an independent choice — an op that generates a tensor from parameters alone, or an fp8 path whose output may be fp16 or bf16 — the tensors are not a second source and `dtype` stays a `signature.params` entry.

The kernel is dtype-specialized, so this makes kernel construction uniformly deferred to the first `forward()` — for fixed-rank and arbitrary-rank ops alike — keyed by every input that selects a specialization, dtype among them. `dispatch_kernel()` stays in `__init__`: resolving the kernel *class* needs no tensor. It also needs no device, and must not ask for one — see [Kernel selection](#kernel-selection).

### Kernel selection

**Construction reads no device property.** An op constructs where it is imported. The tensors arrive later, perhaps on a device the process has not touched, perhaps on hardware where the probe does not exist at all. Installing the kernel map resolves classes and nothing more; a target that cannot run the op is refused when a kernel is first selected, built or called — by the implementation, which owns the architectures it was written for.

**Choosing a slot is the op's; choosing among a slot's implementations is not.** Which slot serves a call follows from the call's user-visible semantics. Which implementation of that slot runs belongs with the implementations.

**An implementation states the region it serves, positively.** Never by excluding a sibling, never by architecture — its declared support already answers that.

**Order decides nothing.** Selection takes the implementation that applies; the one declared general runs where no specialised one does. Nothing applicable is an error, and two specialised implementations claiming one call is an ambiguity error rather than a silent preference. A replacement the caller supplies answers the same question as the class it replaces.

The rule is implementation choice within one slot. Choosing the slot sits above it, dtype specialization beside it; neither goes through it. See [S13](ops-design-reference.md#slot-s13).

`self.dtype` exists for `eval_roofline` / `total_memory` only. No execution path may read it: it records an earlier call, and the next dtype invalidates it.

`_validate_dtypes` runs on every `forward()` call — dtype validity depends on the actual tensors passed, not just their shapes. It is the only dtype gate; an op does not compare an incoming tensor against a dtype it was constructed with, because there is no such dtype. Roofline timing and formula semantics are defined in [roofline.md](roofline.md). See [Parameter Design](ops-design-reference.md#parameter-design) for fixed-rank vs arbitrary-rank details and [Codegen Details](ops-design-reference.md#codegen) for calling conventions.

### Kernel caching and enumeration

L1 owns get-or-build. An op names the **role** a kernel plays, the **key** identifying the specialization, and the factory that builds it. The factory runs on the first miss for that key and never again. An op MUST NOT carry a get-or-build of its own — no cache dict, no build guarded on a kernel attribute being unset. Holding what L1 returned in `self.kernel` is not one.

The key is opaque to L1 and must carry every input that can change what gets built. Naming the axes is the op's job, because only the op knows what its factory closes over.

The entry, not the kernel, is the unit built once. A specialization that must build several kernels together returns them as one immutable entry from one factory; kernels keyed independently of each other are separate roles.

`iter_kernels()` enumerates entries and delegates explicitly, never by reflecting over attributes. An op that runs a kernel another op built declares that op in `kernel_delegates()`, so a composite reaches its delegates' kernels without overriding `autotune()`. Reflection could only guess: a kernel nested deeper than the traversal descended, or held in an attribute whose type it did not recognise, was silently invisible. Declaring turns that silent omission into a missing declaration.

## Scaffolding an Op from a Manifest Entry

The scaffold emits a T2 (L1-direct) op file from one manifest entry. Each step has typed **Input** (manifest fields consumed), **Output** (the code fragment produced), **Validation** (concrete check), and a **Reference** link to the authoritative slot rule in [`slot-rules.md`](../../.claude/skills/scaffold-op/slot-rules.md). Examples scaffold the fictional `ExampleCumsumFwdOp` (cumulative-sum semantics) in T2 (L1-direct) form from an equally fictional manifest entry; nothing in them mirrors a shipped file.

### Step 1: File header + imports

**Input.** `source.kernel_map` values (Kernel classes to import).

**Output.**

```python
"""Cumulative sum operator (host-side Op layer).

Provides:
  - ExampleCumsumFwdOp: y = cumsum(x, dim=-1)
"""

import math
from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.example_cumsum import ExampleCumsumKernel

from ..op_base import Op
```

**Validation.** Every concrete-Kernel import matches one `source.kernel_map` value verbatim. The `Kernel` base import and `..op_base` relative import are fixed.

**Reference.** [Slot S1](../../.claude/skills/scaffold-op/slot-rules.md#slot-s1), [S2](../../.claude/skills/scaffold-op/slot-rules.md#slot-s2), [S3](../../.claude/skills/scaffold-op/slot-rules.md#slot-s3), [S4](../../.claude/skills/scaffold-op/slot-rules.md#slot-s4).

### Step 2: Class declaration + docstring + `__all__`

**Input.** Manifest entry key (= class name); `signature.inputs`, `signature.params`, `static_dims` (Args block content).

**Output.**

```python
__all__ = ["ExampleCumsumFwdOp"]


class ExampleCumsumFwdOp(Op):
    """Cumulative sum operator: y = cumsum(x, dim=-1).

    Output has the same shape and dtype as input.

    Args:
        N: Hidden dimension (size along the reduction axis), committed
            at ctor via ``static_dims: N: "x.shape[dim]"``.
        dim: Reduction dimension (default -1).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """
```

**Validation.** Class name ≡ manifest entry key, byte-exact (`ExampleCumsumFwdOp`). Every `Args:` entry appears as an `__init__` kwarg in Step 3; no extras.

**Reference.** [Slot S5](../../.claude/skills/scaffold-op/slot-rules.md#slot-s5), [S6](../../.claude/skills/scaffold-op/slot-rules.md#slot-s6), [S7](../../.claude/skills/scaffold-op/slot-rules.md#slot-s7).

### Step 3: `_static_axes` + `__init__` signature and body

**Input.** `static_dims` (literal-axis → class-level `_static_axes` frozenset; param-axis → empty class-level default, bind at `forward()` after `dim % x.ndim` normalization); `signature.params`.

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
        dim: int = -1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N = N
        self.dim = dim
        self.tune = tune
        # M is not a static_dim — deferred to forward() where x.ndim
        # is known and M is derived from the non-reduction axes.
        self.dispatch_kernel(kernel_map)
```

**Validation.** Every `__init__` kwarg has a manifest source (`static_dims` or `signature.params`); no extras except `kernel_map` / `tune`. `dtype` is not a kwarg — it is read from the input in `forward()`. In particular, `M` is NOT a ctor kwarg — `ExampleCumsumFwdOp.static_dims` declares only `N`, so `M` is derived at forward time. Keyword-only via `*`, no defaults on `static_dims` entries. `_static_axes` matches the manifest axis form (literal-int axis → populated class-level frozenset; param-dependent axis → empty class-level default, bound at forward after `dim % x.ndim` normalization).

**Reference.** [Slot S21](../../.claude/skills/scaffold-op/slot-rules.md#slot-s21), [S12](../../.claude/skills/scaffold-op/slot-rules.md#slot-s12), [S13](../../.claude/skills/scaffold-op/slot-rules.md#slot-s13).

### Step 4: `default_kernel_map` + `forward`

**Input.** Manifest `source.kernel_map`; `signature.inputs`; `static_dims` (for the forward-time commitment check); `shape_rules` (for `dim` range validation).

**Optional inputs.** An `optional: true` input takes a `None` default in `forward`, and presence is read from the call rather than settled at construction, so one instance serves both ways of calling the op. A `shape_rules` entry naming it applies to the calls that supply it, and `forward` is what enforces that. Where the presence changes what gets built, it belongs in the kernel cache key alongside the shapes.

**Output.**

```python
    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"example_cumsum_fwd": ExampleCumsumKernel}

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
        # Derive M (product of non-reduction dims); cache by shape and dtype.
        M = math.prod(s for i, s in enumerate(x.shape) if i != dim)
        self.M = M  # stored for eval_roofline
        self.dtype = x.dtype  # ditto; the op commits to no dtype before this
        kernel = self.get_or_build_kernel(
            "example_cumsum_fwd",
            (x,),                        # the tensors the kernel will be handed
            key=((M,), x.dtype),         # what the in-tree kernel specializes on
            build=lambda: self.kernel_map["example_cumsum_fwd"](
                M, self.N, "sum", x.dtype, tune=self.tune),
        )
        # Move reduction axis to last, reshape to (M, N), compute, restore.
        orig_shape = x.shape
        x2 = x.movedim(dim, -1).contiguous().reshape(M, self.N)
        y2 = kernel(x2)
        y = y2.reshape(*orig_shape[:dim], *orig_shape[dim + 1:], self.N)
        return y.movedim(-1, dim)
```

**Validation.**

- `default_kernel_map` keys / values match manifest `source.kernel_map` verbatim.
- `forward` calls `self._validate_dtypes(...)` first — not inline dtype comparisons, which are Step 5's job.
- Every `static_dims` commitment is checked against the tensor shape at the normalized axis, and `_static_axes` is bound from that (non-negative) axis. Both before the get-or-build call.
- The kernel comes from `self.get_or_build_kernel`, never a cache dict the op owns:
  - `key=` and `build=` are the in-tree recipe. The kernel is built from `x.dtype` and the key carries it, so a call with another dtype builds a second kernel rather than reusing the first.
  - `inputs=` is the tensors the kernel is handed, which is what an external target's builder is described with. A new op passes it; an op not yet migrated omits it and stays in-tree only.
- The op never trims kernel output: a kernel that pads internally returns the semantic shape.

**Reference.** [Slot S14](../../.claude/skills/scaffold-op/slot-rules.md#slot-s14), [S15](../../.claude/skills/scaffold-op/slot-rules.md#slot-s15), [S16](../../.claude/skills/scaffold-op/slot-rules.md#slot-s16).

### Step 5: `_infer_output_shapes` + `_validate_dtypes`

**Input.** Manifest `shape_rules` (for S17); per-tensor `dtype` and `dtype_combos` (for S18).

**Output.**

```python
class ExampleCumsumFwdOp(Op):
    ...

    def _infer_output_shapes(self, x_shape: tuple) -> Dict[str, tuple]:
        return {"y": x_shape}

    def _validate_dtypes(self, x: torch.Tensor) -> None:
        if x.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError(f"x.dtype must be float32/float16/bfloat16, got {x.dtype}")
```

**Validation.** `python scripts/validate_manifest.py` exercises both methods at CI on every op with `status: implemented`; `spec-only` entries skip L2/L3. **L2 parity:** `_infer_output_shapes(mock_inputs)` must agree with `shape_rules`. **L3 parity:** `_validate_dtypes` must accept exactly the declared `dtype` union / `dtype_combos` and reject everything else. Parity disagreements route to `strict_errors`; advisory mode (default) reports them as warnings, `--strict` / `MANIFEST_STRICT_BLOCKING=1` makes them blocking.

**Reference.** [Slot S17](../../.claude/skills/scaffold-op/slot-rules.md#slot-s17), [S18](../../.claude/skills/scaffold-op/slot-rules.md#slot-s18).

### Step 6: `eval_roofline`

**Input.** Manifest `roofline.vars`, `roofline.flops`, `roofline.bytes`.

**Output.**

```python
class ExampleCumsumFwdOp(Op):
    ...

    def eval_roofline(self) -> tuple[int, int]:
        flops = self.M * self.N
        bytes_ = 2 * self.M * self.N * self.dtype.itemsize
        return flops, bytes_
```

**Validation.** The body is **plain Python** reading `self.*` attributes. Those attributes — `self.M` and `self.dtype` here — are bound by `forward()`, so `eval_roofline` is callable only after at least one forward; there is no ctor-time dtype to read. No class-level roofline expression strings, no `ast.parse`, no shared L1 evaluator — prohibited by [`roofline.md §4.4.6` Evaluator Surface Boundary](roofline.md#446-evaluator-surface-boundary). Return type is `tuple[int, int]`, not `float` or `numpy`. Expressions derive directly from `roofline.vars` bindings + `roofline.flops` + `roofline.bytes`; see [`roofline.md §4.4` Op Codegen](roofline.md#44-op-codegen).

**Reference.** [Slot S19](../../.claude/skills/scaffold-op/slot-rules.md#slot-s19).

### Step 7: Package registration

**Input.** The class name (Step 2) and the op's source filename.

**Output (append to `src/tileops/ops/reduction/__init__.py`):**

```python
# --- ExampleCumsumKernel ops ---
from .example_cumsum import ExampleCumsumFwdOp
```

…with a matching entry added to the module's `__all__` list.

**Validation.** The import sits under its family's grouping comment block; a matching `__all__` entry is present (otherwise `from tileops.ops.reduction import *` silently drops the op).

**Reference.** [Slot S20](../../.claude/skills/scaffold-op/slot-rules.md#slot-s20).

### Slot coverage

| Step | Slots produced |
| ---- | -------------- |
| 1    | S1, S2, S3, S4 |
| 2    | S5, S6, S7     |
| 3    | S21, S12, S13  |
| 4    | S14, S15, S16  |
| 5    | S17, S18       |
| 6    | S19            |
| 7    | S20            |

## Out of Scope

This playbook emits exactly the 17 slots above. The following are **not** produced by the scaffold — each needs separate treatment:

- **Family-specific protocol variables.** `_op_kind` (reduction), `_kernel_key`, `_kernel_cls` (norm + reduction T1 wrappers), `_kernel_handles_padding`, `_op_name`, `kernel_cls`. Kernel-dispatch-convention-dependent; cannot be mechanically derived from the manifest. See [Family-Base Protocol (Appendix)](ops-design-reference.md#base-class-protocol).
- **Optional hooks.** `_pad_value`, `_validate_dim`, `_pre_kernel`, `_post_kernel`. Op-specific business logic (e.g., `ArgmaxFwdOp._pad_value = -inf`). See [Optional Hooks (Appendix)](ops-design-reference.md#optional-hooks-appendix).
- **`_cache_key` override.** The default projection via `_static_axes` is correct but sometimes over-fragmenting. Override logic depends on what subset of the input shape the kernel actually depends on — kernel-math-specific.
- **Family-base (T1) subclassing.** See [Family-Base Refactoring](#family-base-refactoring).
- **Kernel implementations themselves.** The playbook's scope is the Op (host) layer. See [Implementing a Kernel](#implementing-a-kernel) for the kernel-side interface surface.
- **`torch_compile_fullgraph` declaration.** Requires registered compile-test evidence. Semantics: [manifest.md](manifest.md#torch_compile_fullgraph).
- **Compile dispatch boundary.** See [Compile Dispatch Boundary](#compile-dispatch-boundary).

## Implementing a Kernel

Kernel implementation is not covered by the scaffold-op skill. The device-side interface a scaffolded Op depends on — required `__init__` / `forward` / `kernel`, optional `default_config` / `autotune_configs` / `supported_archs` — is specified in [Kernel base class attributes](ops-design-reference.md#base-class-protocol).

## Compile Dispatch Boundary

Contract for every op declaring `torch_compile_fullgraph` while resolving
kernels at call time.

**Invariant.** A dynamo-traced `forward` MUST NOT construct a `Kernel` or
enter a TileLang builder. Kernel-cache misses run TileLang JIT machinery
(`inspect`-based signature handling) that dynamo cannot trace; an eager
warm-up before `torch.compile` only hides the miss path and does not
satisfy the cold-call contract.

**Mechanism** (`src/tileops/ops/compile_boundary.py`; reference adopters:
`pool.py`, `norm/batch_norm.py`):

1. `Op.dispatch_kernel` registers every op in a weak instance registry at
   `__init__` time and stores `self._instance_key`.
1. The family defines one `torch.library.custom_op` per output arity. Its
   eager body resolves the instance from the registry and calls
   `self._eager_forward` — cache lookup, Kernel construction, and launch
   all run untraced. Its fake derives output shapes from
   `_infer_output_shapes` and dtypes from the manifest contract.
1. `forward` becomes a single dispatch call:
   `return _family_fwd(input, self._instance_key)`; the previous body is
   renamed `_eager_forward` unchanged.

**Constraints.**

- The instance key is a **string**: dynamo bakes string custom-op
  arguments as static constants, while an `int` key is generalized to an
  unhashable `SymInt` once a second instance compiles through the same
  frame. Stale-graph safety comes from dynamo's ID_MATCH guard holding a
  weak reference to the compiled callable: a dead instance forces
  recompilation, so a reused `id()` cannot resolve against a stale graph.
- The boundary covers forward-only compilation. Declaring
  `torch_compile_fullgraph` on an op whose compiled graph must
  backpropagate additionally requires registering an autograd formula for
  the dispatch custom op.
- An op that builds no kernel in `forward` — every kernel already in its
  cache — does not need the boundary; the invariant still applies to its
  `forward`.

## Family-Base Refactoring

The scaffold emits T2 (L1-direct) ops only; once a family accumulates 2-3 ops sharing an identical `forward()` flow, a separate family-specific refactoring (not scaffold-op) extracts an L2 base and rewrites the concrete ops as T1 thin wrappers — see [Development Path](ops-design-reference.md#development-path) for when to extract and [Adding a New Family Base](ops-design-reference.md#adding-a-new-family-base) for the process. Family bases MUST NOT normalize genuine per-op behavior differences.

## Further Reference

- [Slot Rules](../../.claude/skills/scaffold-op/slot-rules.md) — full Rule / Derivation / Example / Common mistakes per slot
- [Codegen Details](ops-design-reference.md#codegen) — calling conventions, inheritance rules, consistency enforcement
- [Base Class Protocol](ops-design-reference.md#base-class-protocol) — `Op` and `Kernel` base class attributes
- [Naming Conventions](ops-design-reference.md#naming-conventions) — class / `kernel_map` / builder function rules
- [Parameter Design](ops-design-reference.md#parameter-design) — static vs dynamic op comparison
- [manifest.md](manifest.md) — manifest entry structure, `static_dims`, `shape_rules`, `roofline`
- [roofline.md](roofline.md) — roofline formula syntax, codegen, evaluator surface boundary
