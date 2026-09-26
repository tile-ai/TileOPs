# Op Interface Design

What an Op's interface rests on, then the step-by-step playbook for scaffolding one from a manifest entry, then the contract an op takes on when it declares itself compilable. Per-slot rules are authoritative in [`ops-design-reference.md`](ops-design-reference.md); this file states the decisions those rules follow from.

## Concepts

Every operator is split into two classes — **Op** (host-side: validates inputs, dispatches to Kernel, assembles output) and **Kernel** (device-side: owns the TileLang program, tile configuration, JIT compilation). The two layers are independently modifiable — changing a Kernel's tile strategy does not require changing the Op.

### Class hierarchy

```
Op                          ← L1: thin base, shared by all ops
  └── FamilyBase            ← L2: family-specific forward() flow (optional)
        └── ConcreteOp      ← L3: leaf class emitted by the scaffold
```

- **L1 (`Op`):** shared host-side plumbing (dispatch, get-or-build kernel caching, kernel enumeration, autotune) plus the methods generated from the manifest signature: the call checks, `_infer_output_shapes`, `_validate_dtypes` and `eval_roofline`.
- **L2 (`FamilyBase`):** per-family shared `forward()` pipeline (one per family). **Not produced by this playbook** — see [Family-Base Refactoring](#family-base-refactoring).
- **L3 (`ConcreteOp`):** this playbook's target. New ops start by inheriting L1 directly (T2 shape); see [Family-Base Refactoring](#family-base-refactoring) for when a family graduates to L2.

### Execution timing

**Do it at the first moment all required information is known, do it once, cache the result.** What an op knows at construction is its `signature.params`; every index the call's tensors carry is solved per call.

**Shape is not a constructor parameter when the tensors carry it.** Only `signature.params` belongs in `__init__`, plus the code-owned execution-policy parameters of [manifest.md table 7](manifest.md#t-policy). A dimension declared nowhere is not construction information: it arrives with the call, and taking it twice lets an instance disagree with the tensors it is handed. What the kernel is compiled for goes in the memory key instead, so a second shape builds a second kernel.

**Dtype is not a constructor parameter when the inputs determine it.** An op reads it from the input tensors in `forward()`: a caller who passes fp16 tensors gets the fp16 kernel without having said so twice, and an op can no longer be constructed in a state that disagrees with the tensors it is about to be handed.

**An output dtype is a dtype expression of the signature** — a `DType` index solved from the inputs, a constant, a dtype primitive, or a dtype parameter the caller passes at construction ([manifest.md](manifest.md#dtypes)).

The kernel is dtype-specialized, so this makes kernel construction uniformly deferred to the first `forward()` — for fixed-rank and arbitrary-rank ops alike — keyed by every input that selects a specialization, dtype among them. `dispatch_kernel()` stays in `__init__`: resolving the kernel *class* needs no tensor. It also needs no device, and must not ask for one — see [Kernel selection](#kernel-selection).

The generated `_validate_dtypes` is the only dtype gate, and it runs on every `forward()` call: validity depends on the tensors passed, and an op has no constructed dtype to compare them against. `self.dtype` exists for `eval_roofline` / `total_memory` only: it records an earlier call, and the next dtype invalidates it, so no execution path may read it. Roofline timing and formula semantics are in [roofline.md](roofline.md); see [Parameter Design](ops-design-reference.md#parameter-design) for fixed-rank vs arbitrary-rank details and [Codegen Details](ops-design-reference.md#codegen) for calling conventions.

### Kernel selection

**Construction reads no device property.** An op constructs where it is imported. The tensors arrive later, perhaps on a device the process has not touched, perhaps on hardware where the probe does not exist at all. Installing the kernel map resolves classes and nothing more; a target that cannot run the op is refused when a kernel is first selected, built or called — by the implementation, which owns the architectures it was written for.

**Choosing a slot is the op's; choosing among a slot's implementations is not.** Which slot serves a call follows from the call's user-visible semantics. Which implementation of that slot runs belongs with the implementations.

**An implementation states the region it serves, positively.** Never by excluding a sibling, never by architecture — its declared support already answers that.

**Order decides nothing.** Selection takes the implementation that applies; the one declared general runs where no specialised one does. Nothing applicable is an error, and two specialised implementations claiming one call is an ambiguity error rather than a silent preference. A replacement the caller supplies answers the same question as the class it replaces.

**An implementation states how it is built.** `entry_for(call)` returns the identity two builds must share to be one entry, and the thunk that produces it. The identity is the construction arguments other than `tune`, which changes how fast a kernel runs but not what it computes, plus the device where the constructor could produce a different object on another one. An op names no candidate's constructor.

Three records, each with one owner: the **call** (a `CallSpec` subclass) is what the caller asked for plus the device it runs on, and carries every fact the family's `applies` / `refusal` / `entry_for` read; the **build identity** is the selected class's projection of it; the **role** is the memoization bucket, one per op and never the dispatch key.

`kernel_for` is the only way an op's in-tree implementation reaches a kernel. It asks `Op.entry_for(role, call)` for the identity and the builder, and supplies both to get-or-build. The default `entry_for` selects among the op's candidates and asks the chosen class. An op with one implementation and no call record overrides `entry_for` and states its own identity and builder there, rather than opening a second path to the cache.

The rule is implementation choice within one slot. Choosing the slot sits above it, dtype specialization beside it; neither goes through it. See [S13](op-slot-rules.md#slot-s13).

### Target boundary

**A target replaces the whole op.** A target that registers a builder for an op serves every call of it, and its kernel is called with the tensors its builder was described with. The op's own body is the in-tree implementation and does not run for a target.

**The op layer guarantees a target the manifest, and nothing more.** The generated checks run before the target is called. Every tensor is on the call device except those declaring `device: cpu`, every tensor declaring `contiguous: true` is contiguous, and the call, a caller-supplied output buffer included, meets the signature.

**A traced op is one graph node whichever target serves it.** An op on the [compile boundary](#compile-dispatch-boundary) chooses between the in-tree kernels and a target inside its operator.

**A composite needs no builder of its own.** An op that builds no kernel of its own runs its composition when the target registers no builder for it; each sub-op, given the composite's `target`, settles on a target itself.

**An op with no call-time tensor input is placed by the call-device rule** of [manifest.md § Call Semantics](manifest.md#call-semantics).

### Kernel caching and enumeration

L1 owns get-or-build. An op names the **role** a kernel plays, and `entry_for` names the **identity** of the specialization and the factory that builds it. The factory runs on the first miss for that identity and never again. An op MUST NOT carry a get-or-build of its own — no cache dict, no build guarded on a kernel attribute being unset. Holding what L1 returned in `self.kernel` is not one.

The identity is opaque to L1 and must carry every input that can change what gets built. Where the op selects among candidates, the selected class names those axes in its own `entry_for`, because only it knows what its constructor reads; where the op has one kernel, the op's `entry_for` names them.

The entry, not the kernel, is the unit built once. A specialization that must build several kernels together returns them as one immutable entry from one factory; kernels keyed independently of each other are separate roles.

`iter_kernels()` enumerates entries and delegates explicitly, never by reflecting over attributes. Reflection could only guess: a kernel nested deeper than the traversal went, or held in an attribute of an unrecognised type, was silently invisible. Declaring turns that silent omission into a missing declaration.

**Sub-ops follow the rule for kernels.** A sub-op's constructor arguments may come from the call, so an instance built at construction cannot show what a composite holds.

- A composite declares the sub-op classes it may hold in `delegate_types`: its composition is a fact of the class, checkable before any call.
- It holds every sub-op through `delegate_for`, once per identity, whether the sub-op is built at construction, built per call or injected. The sub-op inherits the composite's execution policy.
- `kernel_delegates()` is derived from what `delegate_for` holds, so enumeration is complete by construction and a composite never overrides `autotune()`.

`delegate_for` is eager, like `kernel_for`: a sub-op that depends on the call is built in `_eager_forward`, never on a traced path.

`built_kernels(role)` is the backend-neutral view: one entry per identity, whoever built it. `iter_kernels()`, and through it `autotune()` and `run_config()`, act on the TileOPs `Kernel` instances the entries hold. A target's builder is not passed `tune`, so a tuning request that cannot reach it warns instead of being dropped.

## Scaffolding an Op from a Manifest Entry

The scaffold emits a T2 (L1-direct) op file from one manifest entry. The call checks, `_infer_output_shapes`, `_validate_dtypes` and `eval_roofline` are generated from the entry and are not scaffolded. Each step has typed **Input** (manifest fields consumed), **Output** (the code fragment produced), **Validation** (concrete check), and a **Reference** link to the authoritative slot rule in [`op-slot-rules.md`](op-slot-rules.md). Examples scaffold the fictional `ExampleCumsumFwdOp` (cumulative-sum semantics) in T2 (L1-direct) form from an equally fictional manifest entry; nothing in them mirrors a shipped file.

### Step 1: File header + imports

**Input.** The Kernel classes the op dispatches to. The kernel map is owned by the code, not the manifest.

**Output.**

```python
"""Cumulative sum operator (host-side Op layer).

Provides:
  - ExampleCumsumFwdOp: y = cumsum(x, dim=-1)
"""

import math
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.example_cumsum import ExampleCumsumKernel
from tileops.manifest.primitives import normalize_axis

from ..op_base import Op
```

**Validation.** Every concrete-Kernel import matches one `kernel_types` value verbatim. The `Kernel` base import and `..op_base` relative import are fixed.

**Reference.** [Slot S1](op-slot-rules.md#slot-s1), [S2](op-slot-rules.md#slot-s2), [S3](op-slot-rules.md#slot-s3), [S4](op-slot-rules.md#slot-s4).

### Step 2: Class declaration + docstring + `__all__`

**Input.** Manifest entry key (= class name); `signature.inputs`, `signature.params` (Args block content).

**Output.**

```python
__all__ = ["ExampleCumsumFwdOp"]


class ExampleCumsumFwdOp(Op):
    """Cumulative sum operator: y = cumsum(x, dim=-1).

    Output has the same shape and dtype as input.

    Args:
        dim: Reduction dimension (default -1).
        target: Backend target to serve this op, or None to decide from the input device.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """
```

**Validation.** Class name ≡ manifest entry key, byte-exact (`ExampleCumsumFwdOp`). Every `Args:` entry appears as an `__init__` kwarg in Step 3; no extras.

**Reference.** [Slot S5](op-slot-rules.md#slot-s5), [S6](op-slot-rules.md#slot-s6), [S7](op-slot-rules.md#slot-s7).

### Step 3: `__init__` signature and body

**Input.** `signature.params`, and the execution-policy parameters of [manifest.md table 7](manifest.md#t-policy) the op takes.

**Output.**

```python
def __init__(
    self,
    dim: int = -1,
    *,
    target: Target = None,
    kernel_map: Optional[Dict[str, Kernel]] = None,
    tune: bool = False,
):
    self.dim = dim
    self.target = target
    self.tune = tune
    self.dispatch_kernel(kernel_map)
```

**Validation.** `__init__` matches `signature.params` item by item ([manifest.md](manifest.md#parameters)), followed by the table-7 execution-policy parameters it takes. `dtype` is not a kwarg — it is read from the input in `forward()`. A param declaring `kw_only: true` goes after `*`.

**Reference.** [Slot S12](op-slot-rules.md#slot-s12), [S13](op-slot-rules.md#slot-s13).

### Step 4: `kernel_types` + `forward`

**Input.** `signature.inputs`; the kernels of Step 1.

**Optional inputs.** An `optional: true` input takes a `None` default in `forward`, and presence is read from the call rather than settled at construction, so one instance serves both ways of calling the op. Where the presence changes what gets built, it belongs in the kernel cache key alongside the shapes.

**Output.**

```python
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"example_cumsum_fwd": ExampleCumsumKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The generated signature checks have run: dtype, shape, dim range.
        dim = normalize_axis(self.dim, x.ndim)
        M = math.prod(s for i, s in enumerate(x.shape) if i != dim)
        x = x.contiguous()          # handed over as the manifest declares it
        # The tensors the kernel will be handed, then what this call is.
        kernel = self.kernel_for(
            "example_cumsum_fwd", (x,), (tuple(x.shape), dim, x.dtype, x.device.index, M)
        )
        return kernel(x)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built from the whole shape and the axis it scans."""
        _shape, dim, dtype, device_index, m = call
        return call, lambda: self.kernel_map["example_cumsum_fwd"](
            m, _shape[dim], "sum", dtype, scan_axis=dim, tune=self.tune, device_index=device_index
        )
```

**Validation.**

- `forward` repeats no check the signature states. It checks no device kind: a kernel states which devices it runs on.
- The kernel comes from `self.kernel_for`, never a cache dict the op owns:
  - `entry_for` is the in-tree recipe. The kernel is built from `x.dtype` and the identity carries it, so a call with another dtype builds a second kernel rather than reusing the first.
- The op never trims kernel output, and never reshapes its input for the kernel: a kernel that pads or permutes internally takes and returns the shapes the manifest declares.

**Reference.** [Slot S14](op-slot-rules.md#slot-s14), [S15](op-slot-rules.md#slot-s15), [S16](op-slot-rules.md#slot-s16).

### Step 5: Generated methods

**Input.** The whole entry.

**Output.** Nothing to write. `_infer_output_shapes`, `_validate_dtypes`, the call checks around `forward` and `eval_roofline` are generated from the signature and the `roofline` field; see [manifest.md § Call Semantics](manifest.md#call-semantics) and [roofline.md §4.4](roofline.md#44-op-codegen).

**Validation.** `python scripts/validate_manifest.py --strict` on every `status: implemented` entry.

**Reference.** [Slot S17](op-slot-rules.md#slot-s17), [S18](op-slot-rules.md#slot-s18), [S19](op-slot-rules.md#slot-s19).

**Compute roof.** `Op.compute_roof()` names the GPU-profile unit that prices the FLOPs `eval_roofline()` counts; the base default `"cuda_core.fp32"` covers CUDA-core fp32 arithmetic. An op whose FLOPs are matmul contractions overrides it — normally `return tensor_core_roof(self.dtype)`, branching on instance state (a backend switch) where the contraction dtype differs from the input dtype. Contract and rationale: [`roofline.md §1.4`](roofline.md#14-compute-roof).

### Step 6: Package registration

**Input.** The class name (Step 2) and the op's source filename.

**Output.** Two files, both with a matching `__all__` entry.

Implementation package, `src/tileops/ops/reduction/__init__.py`:

```python
# --- ExampleCumsumKernel ops ---
from .example_cumsum import ExampleCumsumFwdOp
```

Public path, `src/tileops/reduction.py` — this is the one callers import from:

```python
from .ops.reduction import ExampleCumsumFwdOp
```

**Validation.** The implementation import sits under its family's grouping comment block, and both files carry a matching `__all__` entry — miss the second and the op is unreachable from `tileops.reduction`.

**Reference.** [Slot S20](op-slot-rules.md#slot-s20).

### Slot coverage

| Step | Slots produced |
| ---- | -------------- |
| 1    | S1, S2, S3, S4 |
| 2    | S5, S6, S7     |
| 3    | S12, S13       |
| 4    | S14, S15, S16  |
| 5    | S17, S18, S19  |
| 6    | S20            |

## Out of Scope

This playbook emits exactly the 16 slots above. The following are **not** produced by the scaffold — each needs separate treatment:

- **Family-specific protocol variables.** `_op_kind` (reduction), `_kernel_key`, `_kernel_cls` (norm + reduction T1 wrappers), `_op_name`, `kernel_cls`. Kernel-dispatch-convention-dependent; cannot be mechanically derived from the manifest. See [Family-Base Protocol (Appendix)](ops-design-reference.md#base-class-protocol).
- **Family-base (T1) subclassing.** See [Family-Base Refactoring](#family-base-refactoring).
- **Kernel implementations themselves.** The playbook's scope is the Op (host) layer. See [Implementing a Kernel](#implementing-a-kernel) for the kernel-side interface surface.
- **`fullgraph` compile registration.** Declaring a compile boundary is the class's claim that it supports `fullgraph=True`; its cold compile test, registered in `tests/compile_contract.py`, is the evidence, and the registered set equals the implemented classes declaring a boundary.
- **Compile dispatch boundary.** See [Compile Dispatch Boundary](#compile-dispatch-boundary).

## Implementing a Kernel

Kernel implementation is not covered by this playbook. The device-side interface a scaffolded Op depends on — required `__init__` / `forward` / `kernel`, optional `default_config` / `autotune_configs` / `supported_archs` — is specified in [Kernel base class attributes](ops-design-reference.md#base-class-protocol).

## Compile Dispatch Boundary

Contract for every op registered for `fullgraph=True` compilation while resolving kernels at call time.

**Invariant.** A dynamo-traced `forward` MUST NOT construct a `Kernel` or enter a TileLang builder. Kernel-cache misses run TileLang JIT machinery that dynamo cannot trace; an eager warm-up before `torch.compile` only hides the miss path and does not satisfy the cold-call contract.

**Decisions.**

- A class declaring a compile boundary claims `fullgraph=True` support: a parametric class declares `compile_boundary = True`, a legacy one keeps its `OperatorSpec`s. The manifest records nothing; the registered compile tests are the evidence, and their set equals the implemented classes declaring a boundary.
- The operators are generated from the manifest entry, one `torch.library.custom_op` per effect branch ([manifest.md § Effects](manifest.md#effects)), so no op writes registration code and a schema cannot drift from its entry. The operator is what makes the graph node this op's, and it stays the same node when a target serves the op.
- `forward` only chooses which operator to call. The operator's eager body runs the generated checks once, then the in-tree kernels (`_eager_forward`) or the target; its fake comes from the signature.
- An op's operators write exactly the inputs the manifest marks `mutated`; the validator holds them equal.
- The operator's name is derived from the family and the class; an op does not choose it.
- The boundary covers forward-only compilation. An op whose compiled graph must backpropagate also needs an autograd formula for its operator.
- An op with no tensor input has no node to own and registers no boundary. An op that builds no kernel in `forward` does not need the boundary; the invariant still applies to it.

## Family-Base Refactoring

The scaffold emits T2 (L1-direct) ops only; once a family accumulates 2-3 ops sharing an identical `forward()` flow, a separate family-specific refactoring, outside this playbook, extracts an L2 base and rewrites the concrete ops as T1 thin wrappers — see [Development Path](ops-design-reference.md#development-path) for when to extract and [Adding a New Family Base](ops-design-reference.md#adding-a-new-family-base) for the process. Family bases MUST NOT normalize genuine per-op behavior differences.

## Further Reference

- [Slot Rules](op-slot-rules.md) — full Rule / Derivation / Example / Common mistakes per slot
- [Codegen Details](ops-design-reference.md#codegen) — calling conventions, consistency enforcement
- [Base Class Protocol](ops-design-reference.md#base-class-protocol) — `Op` and `Kernel` base class attributes
- [Naming Conventions](ops-design-reference.md#naming-conventions) — class / `kernel_map` / builder function rules
- [Parameter Design](ops-design-reference.md#parameter-design) — construction time versus call time
- [manifest.md](manifest.md) — manifest entry structure, signature, workloads, call semantics
- [roofline.md](roofline.md) — roofline formula syntax, codegen, evaluator surface boundary
