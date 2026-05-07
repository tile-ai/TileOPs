- Every `tileops/kernels/*` subpackage MUST have an `__init__.py` with explicit `__all__` and `from .module import Symbol` re-exports.

- Relative imports for intra-package references (`from .op import Op`); absolute `tileops.*` imports for cross-package references.

- No file-level lint suppressions (`# ruff: noqa`, `# flake8: noqa`). Targeted inline `# noqa: XXXX` only when needed.

- Use `T.Tensor(shape, dtype)` for TIR function parameters (not deprecated `T.Buffer`).

- Use `T.reinterpret(value, dtype)` (value first; not deprecated dtype-first form).

- Tests intentionally degraded by a process constraint (e.g. trust model splitting manifest and code PRs) get a `FIXME(staged-rollout)` marker. Cleanup describes the invariant to restore — never references a specific PR. Scan: `grep -rn 'FIXME(staged-rollout)'`.

  ```python
  # FIXME(staged-rollout): <one-line summary of what's degraded>
  #
  # Broken invariant: <what contract is currently violated>
  # Why: <which process constraint requires this temporary state>
  # Cleanup: <concrete condition that triggers removal of this marker>
  ```

- **Abbreviation casing in PascalCase**: standard abbreviations stay fully uppercase — `RMSNormKernel`, `SSDDecodeOp`, `FusedAddRMSNormFwdOp`.

- **Filenames**: all-lowercase with underscores; multi-letter abbreviations stay lowercase — `rms_norm.py`, `ssd_decode.py`. Don't capitalize a single letter.

- **Norm filenames**: underscore-separated — `rms_norm`, `layer_norm`, `batch_norm`, `fused_add_rms_norm`. Never contracted (`rmsnorm` etc.).

- **Docstrings**: Google style. One-sentence summary, blank line, then optional `Args:` / `Returns:` / `Raises:` / `Example:`. Internal helpers may have a single-line summary. Never mix Sphinx (`:param x:`) or NumPy headers in the same file.

- **Expand abbreviations on first use in docstrings**: e.g. `State Space Model (SSM)`, `State-Space Dual (SSD)`. Subsequent uses may abbreviate.

- **No development-process metadata in shipped source.** Code, docstrings, and manifest YAML must not reference issue/PR numbers, AC labels, round numbers, reviewer names, or `Follow-up: #N` pointers — that context belongs in the commit message, PR description, and the follow-up issue. See [domain-rules/manifest-spec.md](../domain-rules/manifest-spec.md).

  Discovery scan: `grep -rnE '(^|[^[:alnum:]])#[0-9]{3,}|AC-[0-9]+|round-[0-9]+ review|[Ff]ollow-up:[[:space:]]*#' --exclude-dir=manifest tileops/ tests/ benchmarks/ scripts/`
