- Every `tileops/kernels/*` subpackage must have an `__init__.py` with explicit `__all__` and `from .module import Symbol` re-exports.

- Relative imports for intra-package references (e.g. `from .op import Op`); absolute `tileops.*` imports for cross-package references.

- Do not use file-level lint suppressions (`# ruff: noqa`, `# flake8: noqa`). Use targeted inline `# noqa: XXXX` only when genuinely needed.

- Use `T.Tensor(shape, dtype)` for TIR function parameters, not the deprecated `T.Buffer(shape, dtype)`.

- Use `T.reinterpret(value, dtype)` (value first), not the deprecated `T.reinterpret(dtype, value)`.

- When a PR intentionally degrades a test (xfail, skip, weakened assertion) due to a process constraint (e.g. trust model requiring separate manifest and code PRs), mark it with `FIXME(staged-rollout)` using this template:

  ```python
  # FIXME(staged-rollout): <one-line summary of what's degraded>
  #
  # Broken invariant: <what contract is currently violated>
  # Why: <which process constraint requires this temporary state>
  # Cleanup: <concrete condition that triggers removal of this marker>
  ```

  Cleanup must describe the invariant to restore, not reference a specific PR. Scan with `grep -rn 'FIXME(staged-rollout)'`.

- **Abbreviation casing in PascalCase symbols**: Standard abbreviations must be fully uppercase — `RMS`, not `Rms`; `SSD`, not `Ssd`; `SSM`, not `Ssm`. Examples: `RMSNormKernel`, `SSDDecodeOp`, `FusedAddRMSNormFwdOp`.

- **Abbreviation casing in filenames**: Filenames use all-lowercase with underscores. Multi-word abbreviations keep all letters lowercase — `rms_norm.py`, `ssd_decode.py`. Do not capitalize a single letter (e.g. `Ssd_decode.py` is wrong).

- **Expand abbreviations on first use in docstrings**: When SSM, SSD, or other domain abbreviations first appear in a module or class docstring, write the full form followed by the abbreviation in parentheses. Subsequent uses in the same file can use the abbreviation alone.

  - SSM → State Space Model (SSM)
  - SSD → State-Space Dual (SSD)

- **Underscore-separated naming for norm files**: All norm-related filenames use underscore separation — `rms_norm`, `layer_norm`, `batch_norm`, `fused_add_rms_norm`. Do not contract (e.g. `rmsnorm`, `layernorm`, `batchnorm`).
