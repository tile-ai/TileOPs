# Op Slot Rules

The authoritative rule for each slot of a T2 (L1-direct) op file: S1-S7 and S12-S21. Slots S8-S11
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

- **Rule.** One absolute `from tileops.kernels.* import <KernelClass>` per manifest `kernel_map`
  value. Import nothing that `kernel_map` does not list.
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
  short description. Optional `Example:` block. Derive `Args` from manifest `signature.params` +
  `static_dims`.
- **Example.**
  ```python
  class ExampleCumsumFwdOp(Op):
      """Cumulative sum operator: y = cumsum(x, dim=-1).

      Output has the same shape and dtype as input.

      Args:
          M: Number of rows (product of all dims except the reduction axis).
          N: Hidden dimension (size along the reduction axis).
          dim: Reduction dimension (default -1).
          kernel_map: Optional override for kernel dispatch.
          tune: Whether to autotune (default False).
      """
  ```
- **Common mistakes.** `Args` out of sync with `__init__`; listing tensor inputs (they belong to
  `forward`); documenting a `dtype` kwarg — there is none, dtype comes from the input at `forward`.

### Slot S12: <a id="slot-s12"></a> `__init__` signature

- **Rule.** Block order: (1) `static_dims` entries in manifest key order, no defaults;
  (2) `signature.params` entries in manifest key order; then `*` and (3) any param declaring
  `kw_only: true`, followed by `target`, `kernel_map`, `tune`. Give `dtype` a parameter only when
  the inputs do not determine every output dtype — see
  [Parameter design](./ops-design-reference.md#parameter-design).
- **Example.**
  ```python
  def __init__(
      self,
      N: int,
      dim: int = -1,
      *,
      target: Target = None,
      kernel_map: Optional[Dict[str, Kernel]] = None,
      tune: bool = False,
  ):
  ```
- **Common mistakes.** Parameters with no manifest source; accepting `dtype`, `in_dtype` or
  `out_dtype`; making a param keyword-only that the manifest does not declare `kw_only`.

### Slot S13: <a id="slot-s13"></a> `__init__` body

- **Rule.** Sequence: (a) `self.<name> = <name>` per parameter, `target` among them; (b)
  `self.dispatch_kernel(kernel_map)`, which resolves the kernel *class* and needs no tensor.
  **Construct no kernel and declare no cache here**: the kernel is dtype-specialized and no dtype
  exists until a call arrives, and L1 owns get-or-build
  ([Kernel caching](./ops-design.md#kernel-caching-and-enumeration)).
  A fully-static op — every `signature.inputs` axis is a manifest `shape` dim or a ctor-resolvable
  `static_dims` key — may precompute `self._infer_output_shapes(<input>_shape=(...))` for callers
  that need the output shape before the first call; anything else defers it.
- **Example (arbitrary-rank).**
  ```python
  self.N = N
  self.dim = dim
  self.target = target
  self.tune = tune
  self.dispatch_kernel(kernel_map)
  ```
- **Common mistakes.** `_infer_output_shapes` before `dispatch_kernel`; hard-coding the kernel class
  instead of routing through `self.kernel_map`; storing `self.dtype` at ctor time; a private cache
  dict in place of `Op.get_or_build_kernel`.

### Slot S14: <a id="slot-s14"></a> `default_kernel_map` property

- **Rule.** A `@property` returning the manifest `kernel_map` verbatim: `snake_case` dispatch keys,
  Kernel-class values.
- **Example.**
  ```python
  @property
  def default_kernel_map(self) -> Dict[str, Kernel]:
      return {"example_cumsum_fwd": ExampleCumsumKernel}
  ```
- **Common mistakes.** A class-level dict instead of a property; keys that echo the class name
  instead of being dispatch strings.

### Slot S15: <a id="slot-s15"></a> `forward` signature

- **Rule.** Positional tensor parameters in manifest `signature.inputs` order; return annotation
  `torch.Tensor` or `Tuple[torch.Tensor, ...]` matching `signature.outputs` —
  `def forward(self, x: torch.Tensor) -> torch.Tensor:`. An `optional: true` input defaults to
  `None`.
- **Common mistakes.** Keyword-only tensor parameters; non-tensor kwargs, which belong to
  `__init__`.

### Slot S16: <a id="slot-s16"></a> `forward` body

- **Rule.** Sequence: (a) `self._validate_dtypes(...)`; (b) validate `shape_rules` and normalise parameter-dependent
  axes via modulo (`dim = self.dim % x.ndim`); (c) validate each `static_dims` commitment
  (`x.shape[<resolved_axis>] == self.<kwarg>`); (d) bind `self._static_axes` for arbitrary-rank
  ops; (e) `.contiguous()` every input; (f)
  `self.get_or_build_kernel(<name>, <inputs>, key=<key>, build=<factory>)`, handing over one slot
  per `signature.inputs` entry — `None` for an absent optional one; (g) call the kernel.
  An op that declares `torch_compile_fullgraph` keeps this body under the name `_eager_forward`,
  and its `forward` becomes one call to the operator it registers — that operator is outside the
  scaffold's scope, see
  [Compile Dispatch Boundary](./ops-design.md#compile-dispatch-boundary).
- **Derivation.** Validation expressions come from each `static_dims` entry's
  `<tensor>.shape[<axis>]` RHS; the role is the `kernel_map` dispatch key whose kernel the factory
  builds. A specialization that implies more than a dtype — a compute dtype differing from the
  semantic one, an output dtype no input supplies — makes the entry one frozen record rather than a
  bare kernel, and those fields never live in `self.*`
  ([Forward keying](./ops-design-reference.md#base-class-protocol)).
- **What the op does not do.** It states no device requirement — the kernel it fetched does that —
  and it does not reshape for the kernel: rank reduction, padding and their inverses belong to the
  kernel's own call wrapper, so a backend is handed the shapes the manifest declares.
- **Example (arbitrary-rank).**
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      self._validate_dtypes(x)
      if not -x.ndim <= self.dim < x.ndim:
          raise ValueError(f"dim {self.dim} out of range for x.ndim={x.ndim}")
      dim = self.dim % x.ndim
      if x.shape[dim] != self.N:
          raise ValueError(f"expected x.shape[{dim}] == {self.N}, got {x.shape[dim]}")
      self._static_axes = frozenset({(0, dim)})
      self.dtype = x.dtype
      x = x.contiguous()
      kernel = self.get_or_build_kernel(
          "example_cumsum_fwd",
          (x,),
          key=(self._cache_key(x.shape), x.dtype),
          build=lambda: self.kernel_map["example_cumsum_fwd"](
              self.N, "sum", x.dtype, tune=self.tune
          ),
      )
      return kernel(x)
  ```
- **Common mistakes.** Building a kernel in a traced `forward`; keying on shape alone, so a second
  dtype reuses the first dtype's kernel; a `.is_cuda` check in the op; reshaping before the fetch;
  binding `self._static_axes` before the axis is non-negative; passing an already-built kernel where
  a factory is expected, which rebuilds on every call; fetching a kernel at two sites in one op.

### Slot S17: <a id="slot-s17"></a> `_infer_output_shapes` method body

- **Rule.** Take one `<input>_shape: tuple` per manifest `signature.inputs`; return `Dict[str, tuple]` keyed by output name. Derive from manifest `shape_rules` (see
  [manifest.md § Rules](./manifest.md#rules)). The L1 base raises
  `NotImplementedError`; every op the manifest calls `implemented` supplies a body, which the
  validator's C9 check requires. CI exercises the method with mock inputs and reports disagreement with `shape_rules` as a hard L2
  error.
- **Example.**
  ```python
  def _infer_output_shapes(self, x_shape: tuple) -> Dict[str, tuple]:
      return {"y": x_shape}
  ```
- **Common mistakes.** Accepting or returning `torch.Tensor` instead of shape tuples; demoting an
  op to `status: spec-only` to silence a genuine disagreement — legitimate only when the
  implementation truly does not conform.

### Slot S18: <a id="slot-s18"></a> `_validate_dtypes` method body

- **Rule.** Positional parameters match `signature.inputs`; raise `ValueError` on an invalid dtype
  combination. Derive from manifest `dtype` (union) and `dtype_combos`. L1 stub raises
  `NotImplementedError`; check C6 requires the override. The validator probes `dtype_combos`, declared
  unions and out-of-union negatives exhaustively; divergence is a hard L3 error.
- **Example.**
  ```python
  def _validate_dtypes(self, x: torch.Tensor) -> None:
      if x.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
          raise ValueError(f"x.dtype must be float32/float16/bfloat16, got {x.dtype}")
  ```
- **Common mistakes.** Accepting a dtype outside the declared union; rejecting one listed in
  `dtype_combos`; ignoring `same_as(ref)` linkage between inputs.

### Slot S19: <a id="slot-s19"></a> `eval_roofline` method body

- **Rule.** Codegen emits a complete plain-Python body over `self.*` attributes that `forward()`
  binds, `self.dtype` among them — so `eval_roofline` is defined only after at least one
  `forward()`. Derive from manifest `roofline.vars` / `.flops` / `.bytes`; see
  [`roofline.md` §4.4](./roofline.md#44-op-codegen). L1 stub raises
  `NotImplementedError`; check C6 requires the override.
- **Example.**
  ```python
  def eval_roofline(self) -> tuple[int, int]:
      flops = 4 * self.M * self.N
      bytes_ = (2 * self.M * self.N + self.N) * self.dtype.itemsize
      return flops, bytes_
  ```
- **Common mistakes.** Class-level roofline expression strings parsed at runtime (`_flops_str`,
  `_bytes_str`, `_roofline_vars`), any `ast.parse` or shared `_safe_eval` path — all prohibited by
  [`roofline.md` §4.4.6](./roofline.md#446-evaluator-surface-boundary), which
  rules out a shared evaluator on L1; returning `float` or `numpy` types when the contract is
  `tuple[int, int]`; assuming `self.dtype` is set on a freshly-constructed op.

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

### Slot S21: <a id="slot-s21"></a> `_static_axes` class attribute

- **Rule.** `frozenset[tuple[int, int]]` of `(input_index, axis)` pairs, `input_index` indexing
  `signature.inputs` and `axis` non-negative — `Op` indexes `*input_shapes` non-negatively. Per
  manifest `static_dims` entry `<kwarg>: <tensor>.shape[<axis>]`:

  - `<axis>` is a non-negative literal → class-level
    `_static_axes = frozenset({(input_index_of_<tensor>, <axis>)})`.
  - `<axis>` is a ctor param or a negative literal → class-level `frozenset()`, then assign in
    `forward()` after the `static_dims` check and after `dim % x.ndim`, or override `_cache_key`
    and project inline instead.
  - No `static_dims` (a reduction taking `dim=None`) → `frozenset()`, and override `_cache_key`
    unless a once-per-type `UserWarning` is acceptable. See
    [manifest.md § Empty static_dims](./manifest.md#empty-static_dims).

- **Example.**

  ```python
  class ExampleCumsumFwdOp(Op):
      # static_dims: N: "x.shape[dim]" — dim is a ctor param and may be
      # negative, so the pair is resolved in forward().
      _static_axes: frozenset[tuple[int, int]] = frozenset()
  ```

- **Common mistakes.** A literal pair when the axis is a ctor param, which is the wrong axis under
  arbitrary rank; binding it in `__init__`, where `x.ndim` is unknown; storing a negative axis.
