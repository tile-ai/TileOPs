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

- **L1 (`Op`):** shared host-side plumbing (construction through `Op.__init__`, the steps of a call, target selection, get-or-build entry caching, kernel enumeration, tuning requests) plus the methods generated from the manifest signature: the construction and call checks, `_infer_output_shapes`, `eval_roofline` and the compile boundary.
- **L2 (`FamilyBase`):** per-family shared `forward()` pipeline (one per family). **Not produced by this playbook** — see [Family-Base Refactoring](#family-base-refactoring).
- **L3 (`ConcreteOp`):** this playbook's target. New ops start by inheriting L1 directly (T2 shape); see [Family-Base Refactoring](#family-base-refactoring) for when a family graduates to L2.

### Execution timing

**Do it at the first moment all required information is known, do it once, cache the result.** What an op knows at construction is its `signature.params`; every index the call's tensors carry is solved per call.

**Shape is not a constructor parameter when the tensors carry it.** Only `signature.params` belongs in `__init__`, plus the code-owned execution-policy parameters of [manifest.md table 7](manifest.md#t-policy). A dimension declared nowhere is not construction information: it arrives with the call, and taking it twice lets an instance disagree with the tensors it is handed. What the kernel is compiled for goes in the memory key instead, so a second shape builds a second kernel.

**Dtype is not a constructor parameter when the inputs determine it.** An op reads it from the input tensors in `forward()`: a caller who passes fp16 tensors gets the fp16 kernel without having said so twice, and an op can no longer be constructed in a state that disagrees with the tensors it is about to be handed.

**An output dtype is a dtype expression of the signature** — a `DType` index solved from the inputs, a constant, a dtype primitive, or a dtype parameter the caller passes at construction ([manifest.md](manifest.md#dtypes)).

The kernel is dtype-specialized, so this makes kernel construction uniformly deferred to the first `forward()` — for fixed-rank and arbitrary-rank ops alike — keyed by every input that selects a specialization, dtype among them. `Op.__init__` installs the implementations: resolving the kernel *class* needs no tensor. It also needs no device, and must not ask for one — see [Kernel selection](#kernel-selection).

The generated signature check is the only dtype gate, and it runs on every eager call: validity depends on the tensors passed, and an op has no constructed dtype to compare them against. An op without a compile boundary runs no check while traced (see [Compile Dispatch Boundary](#compile-dispatch-boundary)). Roofline timing and formula semantics are in [roofline.md](roofline.md); see [Parameter Design](ops-design-reference.md#parameter-design) for fixed-rank vs arbitrary-rank details and [Codegen Details](ops-design-reference.md#codegen) for calling conventions.

### Kernel selection

**Construction reads no device property.** Installing the kernel types resolves classes only; a device that cannot run the op is refused when a kernel is first selected, built or called. Why: the tensors arrive later, perhaps on a device the process has not touched.

**A kernel interface is a place an op calls a kernel.** Its class publishes the call contract: the call spec, the tensors handed, what returns, and what `entry_for(call)` owes. An op opens one only where semantics or that contract changes. Why: a backend implements an interface from the contract alone.

**An implementation is a kernel class that inherits an interface, registered under a key.** It is one complete algorithm; tile sizes, split counts and fusion among fixed stages are its plan. Why: it is the unit a backend replaces and selection chooses.

**A call spec holds the immutable facts of one call.** Device facts derive from its device on a miss. A construction parameter that decides which implementation serves the call travels in the call spec, and every op installs all of its `kernel_types`. Why: equal call specs denote one call, so one resolved entry serves every recurrence, and `refusal` is the one place that selects a key.

**Availability filters before selection.** An implementation is available where its `devices` and `supported_archs` allow. Why: where a kernel runs is a fact of the implementation, not of the calls it serves.

**A device type's facts come from one reader.** TileOPs reads CUDA's `arch`, `calibration`, `sm_count` and `smem_budget`; a backend registers another device type's with `register_device_facts`, and an unregistered type reports none. An implementation declaring `supported_archs` runs on one device type. Why: selection and the construction check read the same facts, and an architecture number compares only within one device type.

**`refusal(call)` states the calls an implementation serves.** It returns `None` for a call the implementation serves and names the limit it hits otherwise; an override that narrows a base class ends with `super().refusal(call)`. An op checks no implementation's limits, and the signature holds only what the algorithm requires ([manifest.md § Refinements](manifest.md#refinements)). Why: each implementation answers for itself.

**Precedence picks among the available implementations that serve the call.** `general` is below every other, and `preferred_over` names the implementations one wins over, transitively and acyclically; no winner is an error, several an ambiguity. Why: a declared relation composes implementations unaware of each other, where order or numeric priority cannot.

**An entry is what `entry_for` builds, shared by build identity.** A hit is one lookup by interface and call spec; a miss selects, then builds or reuses. Tuning acts on the entry. Why: a hit costs one lookup.

**Adding a kernel takes two hooks.** Register the implementation, then state `refusal`, adding `preferred_over` only where it overlaps another non-general implementation. Undeclared, an implementation is available on the CUDA devices of every architecture, serves every call and has no precedence. Why: a single-implementation op only inherits its interface.

Choosing the interface sits above selection, dtype specialization beside it. See [S13](op-slot-rules.md#slot-s13).

### Target boundary

**A target replaces the whole op.** A target that registers a builder for an op serves every call of it, and its kernel is called with the tensors its builder was described with. The op's own body is the in-tree implementation and does not run for a target.

**A backend adds an implementation under a new key.** `register_kernel_type(op, key, kernel_type)` adds it beside the in-tree `kernel_types`, under a key the op does not already have; every instance constructed afterwards installs it under each kernel interface it inherits, and selects it by the same `refusal`, `general` and `preferred_over`. A call it does not serve stays with the in-tree implementations. Why: each key has one implementation, so what selects a key is declared in one place.

**The call device is decided once.** The generated check places the call. Selection reads that device's facts; the build and tuning run with it current, for an in-tree entry and a target alike. Why: a build may compile or allocate for the current device, and the kernel is keyed by the device it was built for.

**The op layer guarantees a target the manifest, and nothing more.** The generated checks run before the target is called. Every tensor is on the call device except those declaring `device: cpu`, every tensor declaring `contiguous: true` is contiguous, and the call, a caller-supplied output buffer included, meets the signature.

**A traced op is one graph node whichever target serves it.** An op on the [compile boundary](#compile-dispatch-boundary) chooses between the in-tree kernels and a target inside its operator.

**A composite needs no builder of its own.** An op that builds no kernel of its own runs its composition when the target registers no builder for it; each sub-op, given the composite's `target`, settles on a target itself.

**An op with no call-time tensor input is placed by the call-device rule** of [manifest.md § Call Semantics](manifest.md#call-semantics).

### Kernel caching and enumeration

L1 owns get-or-build. An op names the **kernel interface** a kernel serves, and the selected implementation's `entry_for` names the **identity** of the specialization and the factory that builds it. The factory runs on the first miss for that identity and never again. An op MUST NOT carry a get-or-build of its own — no cache dict, no build guarded on a kernel attribute being unset. `forward` holds what `kernel_for` returned in a local and asks again on every call; a hit is one lookup.

The identity is opaque to L1 and must carry every input that can change what gets built. The selected implementation names those axes in its own `entry_for`, because only it knows what its constructor reads.

The entry, not the kernel, is the unit built once. A specialization that must build several kernels together returns them as one immutable entry from one factory; kernels keyed independently of each other are separate interfaces.

`iter_kernels()` enumerates entries and delegates explicitly, never by reflecting over attributes. Reflection could only guess: a kernel nested deeper than the traversal went, or held in an attribute of an unrecognised type, was silently invisible. Declaring turns that silent omission into a missing declaration.

**Sub-ops follow the rule for kernels.** A sub-op's constructor arguments may come from the call, so an instance built at construction cannot show what a composite holds.

- A composite declares the sub-op classes it may hold in `delegate_types`: its composition is a fact of the class, checkable before any call.
- It holds every sub-op through `delegate_for`, once per identity, whether the sub-op is built at construction, built per call or injected. The sub-op inherits the composite's `target`.
- One sub-op instance is held under one `(stage, identity)`; holding it under a second is an error.
- A completed call of a sub-op not held through `delegate_for` fails the composite's call, because its stages would lose that call.
- `held_delegates()` is derived from what `delegate_for` holds, so enumeration is complete by construction and a composite never overrides `request_tune()`.

`delegate_for` is eager, like `kernel_for`: a traced `forward` reaches no miss.

`built_entries(interface)` is the backend-neutral view: one entry per identity, whoever built it. `iter_kernels()` yields the TileOPs `Kernel` instances the entries hold. `request_tune()` and `kernel_config()` act on those and on the kernels a target built.

`request_tune()` is the one way into tuned mode, called on a constructed op; no op or kernel constructor takes a tuning flag. It tunes the kernels already built and the sub-ops already held, and every entry and sub-op built while the op is in tuned mode is tuned as it is built. Tuning reaches a target through the kernel it built, never its builder: a kernel defining `request_tune()` receives it once per device on every request, and one that does not makes the request warn.

## Scaffolding an Op from a Manifest Entry

The scaffold emits a T2 (L1-direct) op file from one manifest entry. The construction and call checks, `_infer_output_shapes`, `eval_roofline` and the compile boundary are generated from the entry and are not scaffolded. Each step has typed **Input** (manifest fields consumed), **Output** (the code fragment produced), **Validation** (concrete check), and a **Reference** link to the authoritative slot rule in [`op-slot-rules.md`](op-slot-rules.md). Examples scaffold the fictional `ExampleCumsumFwdOp` (cumulative-sum semantics) in T2 (L1-direct) form from an equally fictional manifest entry; nothing in them mirrors a shipped file.

### Step 1: File header + imports

**Input.** The Kernel classes the op dispatches to, and the family's kernel interfaces. `kernel_types` is owned by the code, not the manifest.

**Output.**

```python
"""Cumulative sum operator (host-side Op layer).

Provides:
  - ExampleCumsumFwdOp: y = cumsum(x, dim=-1)
"""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.reduction.call_spec import (
    ExampleCumsumCall,
    ExampleCumsumFwdInterface,
)
from tileops.kernels.reduction.example_cumsum import ExampleCumsumKernel
from tileops.manifest.primitives import normalize_axis
from tileops.ops.op_base import Op
```

**Validation.** Every concrete-Kernel import matches one `kernel_types` value verbatim, and every kernel interface one `interfaces` value. The `Kernel` and `KernelInterface` base imports and the `tileops.ops.op_base` import are fixed.

**Reference.** [Slot S1](op-slot-rules.md#slot-s1), [S2](op-slot-rules.md#slot-s2), [S3](op-slot-rules.md#slot-s3), [S4](op-slot-rules.md#slot-s4).

### Step 2: Class declaration + docstring + `__all__`

**Input.** Manifest entry key (= class name); what the op computes.

**Output.**

```python
__all__ = ["ExampleCumsumFwdOp"]


class ExampleCumsumFwdOp(Op):
    """Cumulative sum operator: y = cumsum(x, dim=-1).

    Output has the same shape and dtype as input.
    """
```

**Validation.** Class name ≡ manifest entry key, byte-exact (`ExampleCumsumFwdOp`). The class docstring has no `Args:` block: construction parameters are documented on `__init__` (Step 3).

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
):
    """Build the op.

    Args:
        dim: Reduction dimension (default -1).
        target: Backend target to serve this op, or None to decide from the input device.
    """
    self.dim = dim
    super().__init__(target=target)
```

**Validation.** Every `__init__` kwarg has an `Args:` entry in its docstring; no extras. `__init__` matches `signature.params` item by item ([manifest.md](manifest.md#parameters)), followed by the table-7 execution-policy parameters it takes. `dtype` is not a kwarg — it is read from the input in `forward()`. A param declaring `kw_only: true` goes after `*`. `super().__init__(target=target)` runs once every manifest parameter is assigned, because it checks them; derived attributes and sub-ops are built after it.

**Reference.** [Slot S12](op-slot-rules.md#slot-s12), [S13](op-slot-rules.md#slot-s13).

### Step 4: `kernel_types` + `interfaces` + `forward`

**Input.** `signature.inputs`; the kernels and kernel interfaces of Step 1.

**Optional inputs.** An `optional: true` input takes a `None` default in `forward`, and presence is read from the call rather than settled at construction, so one instance serves both ways of calling the op. Where the presence changes what gets built, it belongs in the kernel cache key alongside the shapes.

**Output.**

```python
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"example_cumsum_fwd": ExampleCumsumKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "example_cumsum_fwd": ExampleCumsumFwdInterface
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The generated signature checks have run: dtype, shape, dim range.
        dim = normalize_axis(self.dim, x.ndim)
        x = x.contiguous()          # handed over as the manifest declares it
        call = ExampleCumsumCall(device=x.device, shape=tuple(x.shape), dim=dim, dtype=x.dtype)
        return self.kernel_for("example_cumsum_fwd", call)(x)
```

**Validation.**

- `forward` repeats no check the signature states. It checks no device kind: a kernel states which devices it runs on.
- The kernel comes from `self.kernel_for`, never a cache dict the op owns:
  - The call spec carries `x.dtype`, so a call with another dtype resolves a second entry rather than reusing the first. The implementation's own `entry_for` names what it is built from; the op defines none.
- The op never trims kernel output, and never reshapes its input for the kernel: a kernel that pads or permutes internally takes and returns the shapes the manifest declares.

**Reference.** [Slot S14](op-slot-rules.md#slot-s14), [S15](op-slot-rules.md#slot-s15), [S16](op-slot-rules.md#slot-s16).

### Step 5: Generated methods

**Input.** The whole entry.

**Output.** Nothing to write. `_infer_output_shapes`, the signature check, the call checks around `forward`, `eval_roofline` and the compile boundary are generated from the signature and the `roofline` field; see [manifest.md § Call Semantics](manifest.md#call-semantics) and [roofline.md §4.4](roofline.md#44-op-codegen).

**Validation.** `python scripts/validate_manifest.py`.

**Reference.** [Slot S17](op-slot-rules.md#slot-s17), [S18](op-slot-rules.md#slot-s18), [S19](op-slot-rules.md#slot-s19).

**Roof key.** `Op.roof_key()` names the GPU-profile unit that prices the FLOPs `eval_roofline()` counts; the base default `"cuda_core.fp32"` covers CUDA-core fp32 arithmetic. An op whose FLOPs are matmul contractions overrides it — normally `tensor_core_roof` of the contraction's input dtype read from `self.last_call`, branching on instance state (a backend switch) where the contraction dtype differs from the input dtype. Contract and rationale: [`roofline.md §1.4`](roofline.md#14-compute-roof).

### Step 6: Package registration

**Input.** The class name (Step 2) and the op's source filename.

**Output.** Two files, both with a matching `__all__` entry.

Implementation package, `src/tileops/ops/reduction/__init__.py`:

```python
# --- ExampleCumsumKernel ops ---
from tileops.ops.reduction.example_cumsum import ExampleCumsumFwdOp
```

Public path, `src/tileops/reduction.py` — this is the one callers import from:

```python
from tileops.ops.reduction import ExampleCumsumFwdOp
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

- **Family-specific protocol variables and hooks.** `_op_kind`, and the hooks of [Optional Hooks (Appendix)](ops-design-reference.md#optional-hooks-appendix) (reduction). Kernel-dispatch-convention-dependent; cannot be mechanically derived from the manifest. See [Family-Base Protocol (Appendix)](ops-design-reference.md#base-class-protocol).
- **Family-base (T1) subclassing.** See [Family-Base Refactoring](#family-base-refactoring).
- **Kernel implementations themselves.** The playbook's scope is the Op (host) layer. See [Implementing a Kernel](#implementing-a-kernel) for the kernel-side interface surface.
- **`fullgraph` compile registration.** An op whose entry has a call-time tensor input and no composition has a compile boundary and must pass a cold `fullgraph=True` compile; its cold compile test, registered in `tests/compile_contract.py`, is the evidence, and the registered set equals the implemented entries that have a boundary.
- **Compile dispatch boundary.** See [Compile Dispatch Boundary](#compile-dispatch-boundary).

## Implementing a Kernel

Kernel implementation is not covered by this playbook. The device-side interface a scaffolded Op depends on — the kernel interface's `forward` and the classmethod `entry_for`, required `kernel`, optional `default_config` / `autotune_configs` / `supported_archs` — is specified in [Kernel base class attributes](ops-design-reference.md#base-class-protocol).

## Compile Dispatch Boundary

Contract for every op registered for `fullgraph=True` compilation while resolving kernels at call time.

**Invariant.** A dynamo-traced `forward` MUST NOT construct a `Kernel` or enter a TileLang builder. Kernel-cache misses run TileLang JIT machinery that dynamo cannot trace; an eager warm-up before `torch.compile` only hides the miss path and does not satisfy the cold-call contract.

**Decisions.**

- The manifest entry decides the boundary: a class whose entry has a call-time tensor input and no composition gets its operators when the class is defined, and its `compile_op_names` is non-empty exactly then, which the validator checks. Every such implemented op must pass a cold `fullgraph=True` compile; the registered compile tests are the evidence, and their set equals the implemented entries that have a boundary.
- The operators are generated from the manifest entry, one `torch.library.custom_op` per effect branch ([manifest.md § Effects](manifest.md#effects)), so no op writes registration code and a schema cannot drift from its entry. The operator is what makes the graph node this op's, and it stays the same node when a target serves the op.
- `forward` is the op's computation for every op, and a subclass writes nothing else to call it. `__call__` enters the generated `_call_boundary`, which chooses which operator to call; the operator's eager body runs the generated checks once, then `forward` on the in-tree kernels or the target's kernel. Its fake comes from the signature.
- An op's operators write exactly the inputs the manifest marks `mutated`; the validator holds them equal.
- The operator's name is derived from the family and the class; an op does not choose it.
- The boundary covers forward-only compilation. No operator carries an autograd formula, so a backward op's operator refuses an input that tracks history; a caller that needs to backpropagate wires its own `torch.autograd.Function` around the forward and backward ops.
- An op with no call-time tensor input has no node to own, and a composite's sub-ops own theirs; neither has a boundary. Eager, `__call__` runs the same call steps around `forward` for them; traced, it runs no check and calls the serving target's kernel or `forward`. The invariant still applies to them.

## Family-Base Refactoring

The scaffold emits T2 (L1-direct) ops only; once a family accumulates 2-3 ops sharing an identical `forward()` flow, a separate family-specific refactoring, outside this playbook, extracts an L2 base and rewrites the concrete ops as T1 thin wrappers — see [Development Path](ops-design-reference.md#development-path) for when to extract and [Adding a New Family Base](ops-design-reference.md#adding-a-new-family-base) for the process. Family bases MUST NOT normalize genuine per-op behavior differences.

## Further Reference

- [Slot Rules](op-slot-rules.md) — full Rule / Derivation / Example / Common mistakes per slot
- [Codegen Details](ops-design-reference.md#codegen) — calling conventions, consistency enforcement
- [Base Class Protocol](ops-design-reference.md#base-class-protocol) — `Op` and `Kernel` base class attributes
- [Naming Conventions](ops-design-reference.md#naming-conventions) — class / `kernel_types` key / builder function rules
- [Parameter Design](ops-design-reference.md#parameter-design) — construction time versus call time
- [manifest.md](manifest.md) — manifest entry structure, signature, workloads, call semantics
- [roofline.md](roofline.md) — roofline formula syntax, codegen, evaluator surface boundary
