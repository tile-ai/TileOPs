# Op Slot Rules

The authoritative rule for each slot of a T2 (L1-direct) op file: S1-S7 and S12-S20. Slots S8-S11
belong to T1 thin-wrapper subclasses and are not covered here. Each entry gives the rule, an
example, and the mistakes that rule prevents. [`ops-design.md § Scaffolding an Op from a Manifest Entry`](./ops-design.md#scaffolding-an-op-from-a-manifest-entry) walks the same slots in the order
you write them.

Examples use the fictional `ExampleCumsumFwdOp`; none mirrors a shipped file.

Contracts these rules emit against — base-class attributes, protocol variables, naming, parameter
design, calling conventions — live in
[`ops-design-reference.md`](./ops-design-reference.md).

### Slot S1: <a id="slot-s1"></a> Module docstring

- **Rule.** Open the file with a triple-quoted docstring: one-line module summary, then an optional
  `Provides:` block listing `<ClassName>: <one-line semantics>` per concrete op. Template the
  semantics from manifest `ref_api` and `signature`.
- **Example.**
  ```python
  """Cumulative sum operator (L2 Op layer).

  Provides:
    - ExampleCumsumFwdOp: y = cumsum(x, dim=-1)
  """
  ```
- **Common mistakes.** Naming tile sizes or kernel internals; omitting the one-line purpose.

### Slot S2: <a id="slot-s2"></a> Import — `Kernel` base class

- **Rule.** `from tileops.kernels.kernel_base import Kernel`, whenever `kernel_types` is annotated.
  Never alias it, never re-export it.

### Slot S3: <a id="slot-s3"></a> Import — concrete `Kernel` class

- **Rule.** One absolute `from tileops.kernels.* import <KernelClass>` per `kernel_types`
  value. Import nothing that `kernel_types` does not list.
- **Example.** `from tileops.kernels.reduction.example_cumsum import ExampleCumsumKernel`
- **Common mistakes.** A relative import.

### Slot S4: <a id="slot-s4"></a> Import — `Op` base class

- **Rule.** `from tileops.ops.op_base import Op`. Every import is absolute; ruff `TID252`
  enforces it.

### Slot S5: <a id="slot-s5"></a> `__all__`

- **Rule.** `__all__ = ["<ClassName>"]` — the concrete op from S6, and nothing else. Never
  re-export the Kernel class.

### Slot S6: <a id="slot-s6"></a> Class name

- **Rule.** `{PascalCaseName}{Direction}Op`, `Direction` ∈ {`Fwd`, `Bwd`}, no exceptions. The
  manifest entry key IS the class name, verbatim.
- **Common mistakes.** Missing direction suffix; mis-cased abbreviation (see
  [Naming Conventions](./ops-design-reference.md#naming-conventions)).

### Slot S7: <a id="slot-s7"></a> Class docstring

- **Rule.** One-sentence summary, then what a caller acts on: the semantics and any deviation
  from the reference. Optional `Example:` block. No `Args:` block: the S12 kwargs are documented
  in the `__init__` docstring, derived from manifest `signature.params` and the execution-policy
  parameters of S12.
- **Example.**
  ```python
  class ExampleCumsumFwdOp(Op):
      """Cumulative sum operator: y = cumsum(x, dim=-1).

      Output has the same shape and dtype as input.
      """
  ```
- **Common mistakes.** An `Args:` block here, which `scripts/lint/op_docstrings_lint.py` rejects;
  listing tensor inputs (they belong to `forward`); documenting a `dtype` kwarg — there is none,
  dtype comes from the input at `forward`.

### Slot S12: <a id="slot-s12"></a> `__init__` signature

- **Rule.** `signature.params` entries in manifest key order; then `*` and any param declaring
  `kw_only: true`, followed by the execution-policy parameters of
  [manifest.md table 7](./manifest.md#t-policy): `target` on every op, then the op's injected
  implementation objects where it takes them.
- **Example.**
  ```python
  def __init__(
      self,
      dim: int = -1,
      *,
      target: Target = None,
  ):
  ```
- **Common mistakes.** Parameters with no manifest source; taking an input dtype as `dtype` or
  `in_dtype` when the tensors carry it; making a param keyword-only that the manifest does not
  declare `kw_only`; a tuning flag, which no constructor takes (`request_tune()` puts a constructed
  op in tuned mode).

### Slot S13: <a id="slot-s13"></a> `__init__` body

- **Rule.** Sequence: (a) `self.<name> = <name>` per manifest parameter; (b)
  `super().__init__(target=target)`, which stores the target, checks the manifest parameters and
  installs the implementations, resolving each kernel *class* without a tensor; (c) any derived
  attribute or sub-op, after the check. **Construct no kernel and declare no cache here**: the kernel is specialized by what the call
  carries, and L1 owns get-or-build
  ([Kernel caching](./ops-design.md#kernel-caching-and-enumeration)).

- **Example (arbitrary-rank).**

  ```python
  self.dim = dim
  super().__init__(target=target)
  ```

- **Common mistakes.** Hard-coding the kernel class instead of declaring it in `kernel_types`;
  assigning `self.target` by hand; calling `super().__init__` before the manifest parameters it
  checks are assigned; storing `self.dtype` at ctor time; a private cache dict in place of
  `Op.kernel_for`.

### Slot S14: <a id="slot-s14"></a> `kernel_types`

- **Rule.** A class attribute declaring the op's dispatch keys: `snake_case` keys, Kernel-class
  values. It is the one declaration of the in-tree keys, and every instance installs all of it; a
  construction parameter that decides which implementation serves a call travels in the call spec,
  where the implementations' `refusal` reads it. The code owns it; the manifest does not list
  kernels. Beside it, `interfaces` maps each place the op calls a kernel to its
  kernel interface, and every key's class inherits one of them.
- **Example.**
  ```python
  kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
      "example_cumsum_fwd": ExampleCumsumKernel
  }
  interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
      "example_cumsum_fwd": ExampleCumsumFwdInterface
  }
  ```
- **Common mistakes.** Keys that echo the class name instead of being dispatch strings; a
  construction parameter that selects keys at construction instead of travelling in the call spec;
  an interface per shape regime instead of one interface whose implementations split the regimes.

### Slot S15: <a id="slot-s15"></a> `forward` signature

- **Rule.** The parameter list starts with the signature's call-time inputs not declaring
  `kw_only`, in `signature.inputs` order, optional inputs defaulting to `None`, followed by `out`
  when an output declares `buffer: out` ([manifest.md § Effects](./manifest.md#effects)), then,
  after `*`, each input declaring `kw_only: true`, defaulting to `None`
  ([manifest.md § Presence](./manifest.md#presence)). Code-defined execution parameters may follow.
  The return matches `signature.outputs`: one tensor, a tuple in declared order, `None` in a
  nullable position whose expression is false, and `None` when `outputs` is empty.
- **Common mistakes.** A keyword-only tensor parameter the manifest does not declare `kw_only`, or
  a `kw_only` input left positional; non-tensor contract parameters, which belong to `__init__`.

### Slot S16: <a id="slot-s16"></a> `forward` body

- **Rule.** The checks generated from the signature have run before the body. Sequence: (a)
  normalize parameter-dependent axes with the manifest's axis rule
  (`dim = normalize_axis(self.dim, x.ndim)`, which maps `0` and `-1` to the scalar axis at rank 0); (b) make contiguous
  each input the kernel needs contiguous, never a `mutated` input, which is written in place; (c)
  `self.kernel_for(<interface>, <call>)`; (d) call what it returned with the parameters the
  interface's abstract `forward` declares, in that order — output buffers included, `None` for an
  absent optional one.
  `forward` is the body for every op. An op with a compile boundary runs it inside the operator
  its `__call__` enters — that operator is generated and outside the scaffold's scope, see
  [Compile Dispatch Boundary](./ops-design.md#compile-dispatch-boundary).
- **Derivation.** The first argument is a key of `interfaces`; the call is that interface's
  call spec. The interface's abstract `forward` states what the kernel is handed, so the op passes
  those parameters and neither restates nor reorders them. A specialization that implies more than
  a dtype — a compute dtype differing from the semantic one, an output dtype no input supplies —
  makes the entry one frozen record rather than a bare kernel, and those fields never live in
  `self.*` ([Forward keying](./ops-design-reference.md#base-class-protocol)).
- **What the op does not do.** It states no device requirement — the kernel it fetched does that —
  and it does not reshape for the kernel: rank reduction, padding and their inverses belong to the
  kernel's own call wrapper, so a backend is handed the shapes the manifest declares.
- **Example (arbitrary-rank).**
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      dim = normalize_axis(self.dim, x.ndim)
      x = x.contiguous()
      call = ExampleCumsumCall(
          device=x.device, shape=tuple(x.shape), dim=dim, dtype=x.dtype
      )
      return self.kernel_for("example_cumsum_fwd", call)(x)
  ```
- **Common mistakes.** Building a kernel in a traced `forward`; holding the returned entry on
  `self` instead of asking `kernel_for` on every call; keying on shape alone, so a second
  dtype reuses the first dtype's kernel; calling the returned kernel with arguments the interface's
  `forward` does not declare, or out of its order; a `.is_cuda` check in the op; repeating a check
  the signature states; reshaping before the fetch; passing an already-built kernel where
  a factory is expected, which rebuilds on every call; fetching a kernel under two interfaces in one
  op where one entry holding both would do; an op-level `entry_for`, where the implementation
  states its own.

### Slot S17: <a id="slot-s17"></a> `_infer_output_shapes`

- **Rule.** Generated from the signature; the op file does not define it.
- **Common mistakes.** A hand-written override, which can disagree with the signature.

### Slot S18: <a id="slot-s18"></a> Signature check

- **Rule.** Generated from the signature, its dtype expressions and `dtype_combos` among it; it
  runs on every call and is the only place an op rejects a dtype. The op file does not define it.
- **Common mistakes.** A hand-written override, or an inline dtype check in `forward`.

### Slot S19: <a id="slot-s19"></a> `eval_roofline`

- **Rule.** Generated from the `roofline` field over the `indices` of the op's last call; see
  [`roofline.md` §4.4](./roofline.md#44-op-codegen). It is defined only after at least one
  `forward()`.
- **Common mistakes.** Class-level roofline expression strings parsed at runtime, any `ast.parse`
  or shared `_safe_eval` path — prohibited by
  [`roofline.md` §4.4.5](./roofline.md#445-evaluator-surface-boundary); returning `float` or
  `numpy` types when the contract is `tuple[int, int]`.

### Slot S20: <a id="slot-s20"></a> Package `__init__.py` registration

- **Rule.** Two imports, each with a matching `__all__` entry: one
  `from tileops.ops.{family}.<module> import <ClassName>` in `src/tileops/ops/{family}/__init__.py`, under the
  family's grouping comment, and one `from tileops.ops.{family} import <ClassName>` in
  `src/tileops/{family}.py`, which is the public path.
- **Example.**
  ```python
  # src/tileops/ops/reduction/__init__.py
  # --- ExampleCumsumKernel ops ---
  from tileops.ops.reduction.example_cumsum import ExampleCumsumFwdOp

  # src/tileops/reduction.py
  from tileops.ops.reduction import ExampleCumsumFwdOp
  ```
- **Common mistakes.** Import placed outside its grouping comment; missing `__all__` entry, which
  silently breaks `import *`; registering only the implementation package, which leaves the op
  unreachable from `tileops.{family}`.
