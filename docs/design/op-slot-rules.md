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

- **Rule.** `from tileops.kernels.kernel_base import Kernel`, whenever `kernel_map` is annotated.
  Never alias it, never re-export it.

### Slot S3: <a id="slot-s3"></a> Import — concrete `Kernel` class

- **Rule.** One absolute `from tileops.kernels.* import <KernelClass>` per `kernel_types`
  value. Import nothing that `kernel_types` does not list.
- **Example.** `from tileops.kernels.reduction.example_cumsum import ExampleCumsumKernel`
- **Common mistakes.** Relative cross-package import.

### Slot S4: <a id="slot-s4"></a> Import — `Op` base class

- **Rule.** `from ..op_base import Op`, or `from .op_base import Op` for ops directly under
  `src/tileops/ops/`. Absolute `tileops.ops.op_base` violates the relative-import rule in
  [`code-style.md`](../../.claude/rules/code-style.md).

### Slot S5: <a id="slot-s5"></a> `__all__`

- **Rule.** `__all__ = ["<ClassName>"]` — the concrete op from S6, and nothing else. Never
  re-export the Kernel class.

### Slot S6: <a id="slot-s6"></a> Class name

- **Rule.** `{PascalCaseName}{Direction}Op`, `Direction` ∈ {`Fwd`, `Bwd`}, no exceptions. The
  manifest entry key IS the class name, verbatim.
- **Common mistakes.** Missing direction suffix; mis-cased abbreviation (see
  [Naming Conventions](./ops-design-reference.md#naming-conventions)).

### Slot S7: <a id="slot-s7"></a> Class docstring

- **Rule.** One-sentence summary, then an `Args:` block covering every S12 kwarg with type and
  short description. Optional `Example:` block. Derive `Args` from manifest `signature.params` and
  the execution-policy parameters of S12.
- **Example.**
  ```python
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
- **Common mistakes.** `Args` out of sync with `__init__`; listing tensor inputs (they belong to
  `forward`); documenting a `dtype` kwarg — there is none, dtype comes from the input at `forward`.

### Slot S12: <a id="slot-s12"></a> `__init__` signature

- **Rule.** `signature.params` entries in manifest key order; then `*` and any param declaring
  `kw_only: true`, followed by the execution-policy parameters of
  [manifest.md table 7](./manifest.md#t-policy): `target`, `kernel_map` and `tune` on every op,
  then the op's injected implementation objects or `config` where it takes them.
- **Example.**
  ```python
  def __init__(
      self,
      dim: int = -1,
      *,
      target: Target = None,
      kernel_map: Optional[Dict[str, Kernel]] = None,
      tune: bool = False,
  ):
  ```
- **Common mistakes.** Parameters with no manifest source; taking an input dtype as `dtype` or
  `in_dtype` when the tensors carry it; making a param keyword-only that the manifest does not
  declare `kw_only`.

### Slot S13: <a id="slot-s13"></a> `__init__` body

- **Rule.** Sequence: (a) `self.<name> = <name>` per parameter, `target` among them; (b)
  `self.dispatch_kernel(kernel_map)`, which resolves the kernel *class* and needs no tensor.
  **Construct no kernel and declare no cache here**: the kernel is specialized by what the call
  carries, and L1 owns get-or-build
  ([Kernel caching](./ops-design.md#kernel-caching-and-enumeration)).

- **Example (arbitrary-rank).**

  ```python
  self.dim = dim
  self.target = target
  self.tune = tune
  self.dispatch_kernel(kernel_map)
  ```

- **Common mistakes.** Hard-coding the kernel class
  instead of routing through `self.kernel_map`; storing `self.dtype` at ctor time; a private cache
  dict in place of `Op.kernel_for`.

### Slot S14: <a id="slot-s14"></a> `kernel_types`

- **Rule.** A class attribute declaring the op's dispatch keys: `snake_case` keys, Kernel-class
  values. It is the one declaration of the keys; `default_kernel_map`, the instance's view, is
  derived from it — all of it, or the entries a construction parameter selects. The code owns it;
  the manifest does not list kernels.
- **Example.**
  ```python
  kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
      "example_cumsum_fwd": ExampleCumsumKernel
  }
  ```
- **Common mistakes.** Keys that echo the class name instead of being dispatch strings; a
  `default_kernel_map` naming a key `kernel_types` does not declare.

### Slot S15: <a id="slot-s15"></a> `forward` signature

- **Rule.** The parameter list starts with the signature's call-time inputs in `signature.inputs`
  order, optional inputs defaulting to `None`, followed by `out` when an output declares
  `buffer: out` ([manifest.md § Effects](./manifest.md#effects)). Code-defined execution parameters
  may follow. The return matches `signature.outputs`: one tensor, a tuple in declared order, `None`
  in a nullable position whose expression is false, and `None` when `outputs` is empty.
- **Common mistakes.** Keyword-only tensor parameters; non-tensor contract parameters, which belong
  to `__init__`.

### Slot S16: <a id="slot-s16"></a> `forward` body

- **Rule.** The checks generated from the signature have run before the body. Sequence: (a)
  normalize parameter-dependent axes with the manifest's axis rule
  (`dim = normalize_axis(self.dim, x.ndim)`, which maps `0` and `-1` to the scalar axis at rank 0); (b) make contiguous
  each input the kernel needs contiguous, never a `mutated` input, which is written in place; (c)
  `self.kernel_for(<role>, <tensors>, <call>)`, handing over every tensor the kernel reads or
  writes, output buffers included, and `None` for an absent optional one; (d) call the kernel.
  An op registered for `fullgraph=True` compilation keeps this body under the name `_eager_forward`,
  and its `forward` becomes one call to the operator it registers — that operator is outside the
  scaffold's scope, see
  [Compile Dispatch Boundary](./ops-design.md#compile-dispatch-boundary).
- **Derivation.** The role is the `kernel_map` dispatch key whose kernel the factory builds. A specialization that implies more than a dtype — a compute dtype differing from the
  semantic one, an output dtype no input supplies — makes the entry one frozen record rather than a
  bare kernel, and those fields never live in `self.*`
  ([Forward keying](./ops-design-reference.md#base-class-protocol)).
- **What the op does not do.** It states no device requirement — the kernel it fetched does that —
  and it does not reshape for the kernel: rank reduction, padding and their inverses belong to the
  kernel's own call wrapper, so a backend is handed the shapes the manifest declares.
- **Example (arbitrary-rank).**
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      dim = normalize_axis(self.dim, x.ndim)
      x = x.contiguous()
      kernel = self.kernel_for("example_cumsum_fwd", (x,), (tuple(x.shape), dim, x.dtype))
      return kernel(x)


  def entry_for(self, role: str, call: tuple) -> Entry:
      """One implementation, built per shape, axis and dtype."""
      shape, dim, dtype = call
      return call, lambda: self.kernel_map["example_cumsum_fwd"](
          shape[dim], "sum", dtype, tune=self.tune
      )
  ```
- **Common mistakes.** Building a kernel in a traced `forward`; keying on shape alone, so a second
  dtype reuses the first dtype's kernel; a `.is_cuda` check in the op; repeating a check the
  signature states; reshaping before the fetch; passing an already-built kernel where
  a factory is expected, which rebuilds on every call; fetching a kernel under two roles in one op
  where one entry holding both would do.

### Slot S17: <a id="slot-s17"></a> `_infer_output_shapes`

- **Rule.** Generated from the signature; the op file does not define it.
- **Common mistakes.** A hand-written override, which can disagree with the signature.

### Slot S18: <a id="slot-s18"></a> `_validate_dtypes`

- **Rule.** Generated from the signature's dtype expressions and `dtype_combos`; the op file does
  not define it.
- **Common mistakes.** A hand-written override, or an inline dtype check in `forward`.

### Slot S19: <a id="slot-s19"></a> `eval_roofline`

- **Rule.** Generated from the `roofline` field over the `ix` of the op's last call; see
  [`roofline.md` §4.4](./roofline.md#44-op-codegen). It is defined only after at least one
  `forward()`.
- **Common mistakes.** Class-level roofline expression strings parsed at runtime, any `ast.parse`
  or shared `_safe_eval` path — prohibited by
  [`roofline.md` §4.4.5](./roofline.md#445-evaluator-surface-boundary); returning `float` or
  `numpy` types when the contract is `tuple[int, int]`.

### Slot S20: <a id="slot-s20"></a> Package `__init__.py` registration

- **Rule.** Two imports, each with a matching `__all__` entry: one
  `from .<module> import <ClassName>` in `src/tileops/ops/{family}/__init__.py`, under the
  family's grouping comment, and one `from .ops.{family} import <ClassName>` in
  `src/tileops/{family}.py`, which is the public path.
- **Example.**
  ```python
  # src/tileops/ops/reduction/__init__.py
  # --- ExampleCumsumKernel ops ---
  from .example_cumsum import ExampleCumsumFwdOp

  # src/tileops/reduction.py
  from .ops.reduction import ExampleCumsumFwdOp
  ```
- **Common mistakes.** Import placed outside its grouping comment; missing `__all__` entry, which
  silently breaks `import *`; registering only the implementation package, which leaves the op
  unreachable from `tileops.{family}`.
