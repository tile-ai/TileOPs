- Every `tileops/kernels/*` subpackage MUST have an `__init__.py` with explicit `__all__` and `from .module import Symbol` re-exports.

- Intra-package imports: relative (`from .op import Op`). Cross-package: absolute (`tileops.foo.bar`).

- No file-level lint suppressions (`# ruff: noqa`, `# flake8: noqa`). Use targeted inline `# noqa: XXXX` only.

- TIR parameter type: `T.Tensor(shape, dtype)`, never the deprecated `T.Buffer`.

- Reinterpret cast: `T.reinterpret(value, dtype)` (value first), never the deprecated dtype-first form.

- Each TileLang kernel is one `@T.prim_func` whose body opens `with T.Kernel(...)`; sub-routines use `@T.macro`, never nested `prim_func`.

- No narrow-type literal casts (`T.cast(1.0, "float16")`). Reference `x.dtype`, or compute in a wider intermediate and cast at the boundary.

- Promote overflow-prone fp16/bf16 math (cubic, division, `exp`, softmax accumulators) to fp32; cast back to storage dtype at the boundary.

- Decorate each `_<op>_kernel` builder (the `@tilelang.jit`-wrapping `Callable`) with `@functools.lru_cache(maxsize=<N>)`; every parameter must be hashable. Default `maxsize=32`; use `64` only when the distinct-config working set demands it; `maxsize=None` only for intrinsically bounded config spaces. Document any non-default choice at the call site.

- Tag tests degraded by a process constraint (e.g. trust model splitting manifest and code PRs) with `FIXME(staged-rollout)`. Cleanup line names the invariant to restore — never a PR number. Scan: `grep -rn 'FIXME(staged-rollout)'`.

  ```python
  # FIXME(staged-rollout): <one-line summary of what's degraded>
  #
  # Broken invariant: <what contract is currently violated>
  # Why: <which process constraint requires this temporary state>
  # Cleanup: <concrete condition that triggers removal of this marker>
  ```

- PascalCase abbreviations stay fully uppercase: `RMSNormKernel`, `SSDDecodeOp`, `FusedAddRMSNormFwdOp`.

- Filenames: all-lowercase with underscores. Multi-letter abbreviations stay lowercase (`rms_norm.py`, `ssd_decode.py`); never capitalize a single letter. Never contract norm names (`rms_norm`, not `rmsnorm`).

- Docstrings: Google style. One-line summary, blank line, then optional `Args:` / `Returns:` / `Raises:` / `Example:`. Internal helpers may use a single-line summary. Never mix Sphinx (`:param:`) or NumPy headers in one file.

- Expand domain abbreviations on first use in a docstring: `State Space Model (SSM)`, `State-Space Dual (SSD)`. Later uses may abbreviate.

- Shipped source (code, docstrings, manifest YAML) must not reference issue/PR numbers, AC labels, round numbers, reviewer names, or `Follow-up: #N`. See [domain-rules/manifest-spec.md](../domain-rules/manifest-spec.md).

  Discovery scan: `grep -rnE '(^|[^[:alnum:]])#[0-9]{3,}|AC-[0-9]+|round-[0-9]+ review|[Ff]ollow-up:[[:space:]]*#' --exclude-dir=manifest tileops/ tests/ benchmarks/ scripts/`
