# Scaffold Slot Rules

The 17 slots `scaffold-op` emits: S1-S7, S12-S21. Examples scaffold the fictional
`ExampleCumsumFwdOp`; none mirrors a shipped file.

Contracts these rules emit against — base-class attributes, protocol variables, naming, parameter
design, calling conventions — live in
[`docs/design/ops-design-reference.md`](../../../docs/design/ops-design-reference.md).

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
  [`code-style.md`](../../rules/code-style.md).

### Slot S5: <a id="slot-s5"></a> `__all__`

- **Rule.** `__all__ = ["<ClassName>"]` — the concrete op from S6, and nothing else. Never
  re-export the Kernel class.

### Slot S6: <a id="slot-s6"></a> Class name

- **Rule.** `{PascalCaseName}{Direction}Op`, `Direction` ∈ {`Fwd`, `Bwd`}, no exceptions. The
  manifest entry key IS the class name, verbatim.
- **Common mistakes.** Missing direction suffix; mis-cased abbreviation (see
  [Naming Conventions](../../../docs/design/ops-design-reference.md#naming-conventions-appendix)).

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

- **Rule.** Keyword-only via `*`. Block order: (1) `static_dims` entries in manifest key order, no
  defaults; (2) `signature.params` entries in manifest key order; (3) `kernel_map`, `tune`. Give
  `dtype` a kwarg only when the inputs do not determine every output dtype — see
  [Parameter design](../../../docs/design/ops-design-reference.md#parameter-design).
- **Example.**
  ```python
  def __init__(
      self,
      *,
      M: int,
      N: int,
      dim: int = -1,
      kernel_map: Optional[Dict[str, Kernel]] = None,
      tune: bool = False,
  ):
  ```
- **Common mistakes.** Kwargs with no manifest source; accepting `dtype`, `in_dtype` or
  `out_dtype`.

### Slot S13: <a id="slot-s13"></a> `__init__` body

- **Rule.** Sequence: (a) `self.<name> = <name>` per kwarg; (b) `self.dispatch_kernel(kernel_map)`.
  **Construct no kernel here**, whatever the op shape: the kernel is dtype-specialized and no dtype
  exists until `forward()` receives a tensor. `dispatch_kernel` resolves the kernel *class*, which
  needs no tensor and reads no device property; an architecture that cannot run the op is refused
  when a kernel is first selected or built. **Declare no kernel cache here either** — L1 owns
  get-or-build and creates the role's entry table on first use, per
  [Kernel caching and enumeration](../../../docs/design/ops-design.md#kernel-caching-and-enumeration).
  - **Fully-static op** (every non-static axis committed at ctor): may precompute
    `self._infer_output_shapes(<input>_shape=(...))` when a caller needs output shapes before
    `forward()`. Shape inference is dtype-independent.
  - **Arbitrary-rank op** (at least one axis unknown until forward): defer `_infer_output_shapes`
    to `forward()`, once per unique input shape.
- **Derivation.** Each `self.*` assignment mirrors one S12 kwarg. "Fully-static" iff every
  `signature.inputs` shape axis is a manifest `shape` dim name or a ctor-resolvable `static_dims`
  key; the distinction governs when shape inference runs, not when the kernel is built.
- **Example (arbitrary-rank).**
  ```python
  self.N = N
  self.dim = dim
  self.tune = tune
  # M unknown at init (only N committed via static_dims), and no dtype
  # is known at all; the kernel is built in forward() from both.
  self.dispatch_kernel(kernel_map)
  ```
- **Common mistakes.** `_infer_output_shapes` before `dispatch_kernel`; hard-coding the kernel
  class instead of routing through `self.kernel_map`; storing `self.dtype` at ctor time; declaring a
  private cache dict instead of calling `Op.get_or_build_kernel` in `forward()`.

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
  `def forward(self, x: torch.Tensor) -> torch.Tensor:`.
- **Common mistakes.** Keyword-only tensor parameters; non-tensor kwargs, which belong to
  `__init__`.

### Slot S16: <a id="slot-s16"></a> `forward` body

- **Rule.** Sequence: (a) `self._validate_dtypes(...)`; (b) validate `shape_rules` and normalise
  parameter-dependent axes via modulo (`dim = self.dim % x.ndim`); (c) validate each `static_dims`
  commitment (`x.shape[<resolved_axis>] == self.<kwarg>`); (d) for arbitrary-rank ops bind
  `self._static_axes`, then — whatever the op shape — call
  `self.get_or_build_kernel(<name>, <inputs>, key=<key>, build=<factory>)`; (e)
  `.contiguous()` then reshape to the kernel's 2D layout; (f) call the kernel; (g) restore the
  original shape.
- **Derivation.** Validation expressions come from each `static_dims` entry's
  `<tensor>.shape[<axis>]` RHS. Axis normalisation mirrors the param evaluation in `static_dims` +
  `shape_rules`. The role is the `kernel_map` dispatch key whose kernel the factory builds; the key
  names every input the factory closes over. A kernel that pads
  internally returns the semantic shape, so the op does not trim.
- **Example (arbitrary-rank).**
  ```python
  self._validate_dtypes(x)
  if not x.is_cuda:
      raise ValueError("x must be a CUDA tensor")
  if not -x.ndim <= self.dim < x.ndim:
      raise ValueError(f"dim {self.dim} out of range for x.ndim={x.ndim}")
  dim = self.dim % x.ndim
  if x.shape[dim] != self.N:
      raise ValueError(
          f"static_dim mismatch: expected x.shape[{dim}] == {self.N}, "
          f"got {x.shape[dim]}"
      )
  self._static_axes = frozenset({(0, dim)})
  M = math.prod(s for i, s in enumerate(x.shape) if i != dim)
  self.M = M
  # default _cache_key projects non-static axes; override for coarser
  # keying when kernel math permits (see Optional Hooks appendix).
  self.dtype = x.dtype
  kernel = self.get_or_build_kernel(
      "example_cumsum_fwd",
      (x,),
      key=(self._cache_key(x.shape), x.dtype),
      build=lambda: self.kernel_map["example_cumsum_fwd"](
          M, self.N, "sum", x.dtype, tune=self.tune
      ),
  )
  orig_shape = x.shape
  x2 = x.movedim(dim, -1).contiguous().reshape(M, self.N)
  y2 = kernel(x2)
  y = y2.reshape(*orig_shape[:dim], *orig_shape[dim + 1 :], self.N)
  return y.movedim(-1, dim)
  ```
- **Entry.** The entry is the kernel when dtype is the whole story; when a specialization implies
  more — a compute dtype differing from the semantic one, an output dtype no input supplies — it is
  one frozen record holding them together. Those fields MUST NOT live in `self.*` attributes
  written while building the kernel: a second dtype leaves them describing the first.
- **Common mistakes.** Keying on shape alone — a second dtype then silently reuses the first
  dtype's kernel; reshaping before `.contiguous()`; hard-coding `x.shape[-1]` instead of the
  normalised `x.shape[self.dim % x.ndim]`; binding `self._static_axes` before the axis is
  non-negative; constructing the kernel outside the factory, or passing an already-built kernel
  where a factory is expected — the build then runs on every call, hit or miss; trimming padded
  kernel output in the op; not restoring the original shape.

### Slot S17: <a id="slot-s17"></a> `_infer_output_shapes` method body

- **Rule.** Take one `<input>_shape: tuple` per manifest `signature.inputs`; return `Dict[str, tuple]` keyed by output name. Derive from manifest `shape_rules` (see
  [manifest.md § Rules](../../../docs/design/manifest.md#rules)). The L1 base raises
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
  [`roofline.md` §4.4](../../../docs/design/roofline.md#44-op-codegen). L1 stub raises
  `NotImplementedError`; check C7 requires the override.
- **Example.**
  ```python
  def eval_roofline(self) -> tuple[int, int]:
      flops = 4 * self.M * self.N
      bytes_ = (2 * self.M * self.N + self.N) * self.dtype.itemsize
      return flops, bytes_
  ```
- **Common mistakes.** Class-level roofline expression strings parsed at runtime (`_flops_str`,
  `_bytes_str`, `_roofline_vars`), any `ast.parse` or shared `_safe_eval` path — all prohibited by
  [`roofline.md` §4.4.6](../../../docs/design/roofline.md#446-evaluator-surface-boundary), which
  rules out a shared evaluator on L1; returning `float` or `numpy` types when the contract is
  `tuple[int, int]`; assuming `self.dtype` is set on a freshly-constructed op.

### Slot S20: <a id="slot-s20"></a> Package `__init__.py` registration

- **Rule.** Add one `from .<module> import <ClassName>` to `src/tileops/ops/{family}/__init__.py`,
  under the family's grouping comment, plus a matching `__all__` entry.
- **Example.**
  ```python
  # --- ExampleCumsumKernel ops ---
  from .example_cumsum import ExampleCumsumFwdOp
  ```
- **Common mistakes.** Import placed outside its grouping comment; missing `__all__` entry, which
  silently breaks `import *`.

### Slot S21: <a id="slot-s21"></a> `_static_axes` class attribute

- **Rule.** Declare `_static_axes: frozenset[tuple[int, int]]` of `(input_index, axis)` pairs,
  where `input_index` indexes `signature.inputs` and `axis` is **non-negative**. Commit at one of
  two points:

  - **Ctor time**, as a class-level literal, when every axis resolves to a non-negative integer
    without knowing runtime rank (manifest declares `static_dims: M: "x.shape[0]"`).
  - **`forward()` time**, with an empty class-level default, when an axis depends on runtime rank —
    most often a ctor param that may be negative (`static_dims: N: "x.shape[dim]"`, `dim` defaulting
    to `-1`). Normalise the axis (`dim % x.ndim`), then assign
    `self._static_axes = frozenset({(i, <resolved_axis>)})`. Alternatively override `_cache_key`
    and project the shape inline, never populating `_static_axes`.

  An empty frozenset is a legal class-level default: no axes committed yet. Never store a negative
  axis — the `Op` base indexes `*input_shapes` non-negatively.

- **Derivation.** Per manifest `static_dims` entry `<kwarg>: <tensor>.shape[<axis>]`:

  - `<axis>` resolvable to a non-negative literal at class-definition time → class-level
    `_static_axes = frozenset({(input_index_of_<tensor>, <axis>)})`.
  - `<axis>` a ctor param, or a negative literal whose normalised value depends on runtime rank →
    class-level `frozenset()`, then assign in `forward()` after the `static_dims` check, or
    override `_cache_key`.
  - PyTorch-aligned reduction with `dim=None` → empty frozenset (see
    [manifest.md § Empty static_dims](../../../docs/design/manifest.md#empty-static_dims)).

- **Example.**

  ```python
  class ExampleCumsumFwdOp(Op):
      # static_dims: N: "x.shape[dim]" — axis is parameter-dependent
      # (and dim may be negative), so the concrete (input_index, axis)
      # pair is resolved at forward() time after dim % x.ndim
      # normalization. Class-level default is empty.
      _static_axes: frozenset[tuple[int, int]] = frozenset()
  ```

- **Common mistakes.** Omitting `_static_axes` when `static_dims` is non-empty — `Op`'s empty
  default then silently disables static-axis projection in `_cache_key`; emitting a literal
  `(input_index, axis)` when `axis` is a ctor param, which yields a wrong axis under arbitrary
  rank; binding `self._static_axes` in `__init__`, where `x.ndim` is unknown so a negative `dim`
  cannot be normalised; storing a negative axis; leaving `_static_axes` empty without overriding
  `_cache_key`, which emits a once-per-type `UserWarning` (see
  [Optional Hooks](../../../docs/design/ops-design-reference.md#optional-hooks-appendix)).
