- Every `tileops/kernels/*` subpackage MUST have an `__init__.py` with explicit `__all__` and `from .module import Symbol` re-exports.

- Relative imports for intra-package references (`from .op import Op`); absolute `tileops.*` imports for cross-package references.

- No file-level lint suppressions (`# ruff: noqa`, `# flake8: noqa`). Targeted inline `# noqa: XXXX` only when needed.

- Use `T.Tensor(shape, dtype)` for TIR function parameters (not deprecated `T.Buffer`).

- Use `T.reinterpret(value, dtype)` (value first; not deprecated dtype-first form).

- Each TileLang kernel is one `@T.prim_func` whose body opens `with T.Kernel(...)`. Reusable sub-routines factor out as `@T.macro`, never nested `prim_func`.

- No narrow-type literal casts. Hardcoding storage dtype in numeric literals (e.g. `T.cast(1.0, "float16")`) blocks dtype reuse — reference `x.dtype` or compute in a wider intermediate and cast at the boundary.

- Promote overflow-prone fp16/bf16 math to fp32 (cubic, division, `exp`, softmax accumulators); cast back to storage dtype at the boundary.

- Cache kernel builders with `@functools.lru_cache(maxsize=32)`. The `_<op>_kernel(static_params) -> Callable` builder wrapping `@tilelang.jit` decorates with `functools.lru_cache(maxsize=32)`; every parameter must be hashable.

- Tests intentionally degraded by a process constraint (e.g. trust model splitting manifest and code PRs) get a `FIXME(staged-rollout)` marker. Cleanup describes the invariant to restore — never references a specific PR. Scan: `grep -rn 'FIXME(staged-rollout)'`.

  ```python
  # FIXME(staged-rollout): <one-line summary of what's degraded>
  #
  # Broken invariant: <what contract is currently violated>
  # Why: <which process constraint requires this temporary state>
  # Cleanup: <concrete condition that triggers removal of this marker>
  ```

- **Abbreviation casing in PascalCase**: standard abbreviations stay fully uppercase — `RMSNormKernel`, `SSDDecodeOp`, `FusedAddRMSNormFwdOp`.

- **Filenames**: all-lowercase with underscores; multi-letter abbreviations stay lowercase — `rms_norm.py`, `ssd_decode.py`. Don't capitalize a single letter. Norm filenames specifically must be underscore-separated, never contracted (`rms_norm`, never `rmsnorm`).

- **Docstrings**: Google style. One-sentence summary, blank line, then optional `Args:` / `Returns:` / `Raises:` / `Example:`. Internal helpers may have a single-line summary. Never mix Sphinx (`:param x:`) or NumPy headers in the same file.

- **Expand abbreviations on first use in docstrings**: e.g. `State Space Model (SSM)`, `State-Space Dual (SSD)`. Subsequent uses may abbreviate.

- **No development-process metadata in shipped source.** Code, docstrings, and manifest YAML must not reference issue/PR numbers, AC labels, round numbers, reviewer names, or `Follow-up: #N` pointers — that context belongs in the commit message, PR description, and the follow-up issue. See [domain-rules/manifest-spec.md](../domain-rules/manifest-spec.md).

  Discovery scan: `grep -rnE '(^|[^[:alnum:]])#[0-9]{3,}|AC-[0-9]+|round-[0-9]+ review|[Ff]ollow-up:[[:space:]]*#' --exclude-dir=manifest tileops/ tests/ benchmarks/ scripts/`
