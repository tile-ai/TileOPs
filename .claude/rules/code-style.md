Rules a reader has to apply by hand. The deprecated `T.Buffer` annotation, a dtype-first
`T.reinterpret`, a literal cast to a narrow float, and a file-level `noqa` are checked by
`scripts/lint/tilelang_idioms_lint.py`; that file states why each one is wrong.

- Every `src/tileops/kernels/*` subpackage MUST have an `__init__.py` with explicit `__all__` and `from .module import Symbol` re-exports.

- Intra-package imports: relative (`from .op import Op`). Cross-package: absolute (`tileops.foo.bar`).

- Each TileLang kernel is one `@T.prim_func` whose body opens `with T.Kernel(...)`; sub-routines use `@T.macro`, never nested `prim_func`.

- Promote overflow-prone fp16/bf16 math (cubic, division, `exp`, softmax accumulators) to fp32; cast back to storage dtype at the boundary.

- Decorate each `_<op>_kernel` builder (the `@tilelang.jit`-wrapping `Callable`) with `@functools.lru_cache(maxsize=<N>)`; every parameter must be hashable. `maxsize=32` needs no comment; `64` and `None` state above the decorator what makes the config space that wide or bounded.

- Tag code degraded by something outside its own scope — a contract stub that cannot be made abstract until every op migrates, a benchmark that must skip a manifest workload no kernel can run — with `FIXME(staged-rollout)`. Scan: `grep -rn 'FIXME(staged-rollout)'`.

  ```python
  # FIXME(staged-rollout): <one-line summary of what's degraded>
  #
  # Broken invariant: <what contract is currently violated>
  # Why: <which process constraint requires this temporary state>
  # Cleanup: <concrete condition that triggers removal of this marker>
  ```

- PascalCase abbreviations stay fully uppercase: `RMSNormKernel`, `SSDDecodeFwdOp`, `FusedAddRMSNormFwdOp`.

- Filenames: lowercase with underscores, abbreviations included (`rms_norm.py`, `ssd_decode.py`). Never contract a norm name (`rms_norm`, not `rmsnorm`).

- An `optional: true` input defaults to `None` in `forward`, and presence is read from the call, not settled at construction. Where the presence changes what the kernel build produces, it goes in that kernel's cache key.

- Docstrings: Google style. One-line summary, blank line, then optional `Args:` / `Returns:` / `Raises:` / `Example:`. Internal helpers may use a single-line summary. Never mix Sphinx (`:param:`, `#:`) or NumPy headers in one file; comment an attribute with `#` above its assignment.

- The docs site renders the class, `__init__` and `forward` docstrings of the `src/tileops/ops/` classes `docs/api/*.md` collects, and nothing else. A guarantee a caller acts on — an accuracy bound, a deviation from torch — goes in the op's class docstring; a kernel comment does not reach them.

- Expand domain abbreviations on first use in a docstring: `State Space Model (SSM)`, `State-Space Dual (SSD)`. Later uses may abbreviate.
