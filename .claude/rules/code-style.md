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

- **Docstring style: Google**. All Python docstrings (modules, classes, public functions / methods) follow Google style: one-sentence summary on the opening line, blank line, then optional sections `Args:`, `Returns:`, `Raises:`, `Example:`. Internal helpers may use a single-line summary docstring. Do NOT mix Sphinx-style (`:param x:`) or NumPy-style (separate header lines for each param) into the same file.

- **Expand abbreviations on first use in docstrings**: When SSM, SSD, or other domain abbreviations first appear in a module or class docstring, write the full form followed by the abbreviation in parentheses. Subsequent uses in the same file can use the abbreviation alone.

  - SSM → State Space Model (SSM)
  - SSD → State-Space Dual (SSD)

- **Underscore-separated naming for norm files**: All norm-related filenames use underscore separation — `rms_norm`, `layer_norm`, `batch_norm`, `fused_add_rms_norm`. Do not contract (e.g. `rmsnorm`, `layernorm`, `batchnorm`).

- **No development-process metadata in shipped source.** Code, YAML manifests, and docstrings must NOT reference: issue numbers (`#1170`), PR numbers (`PR #1169`), acceptance criteria identifiers (`AC-4`), pipeline rounds (`round-5`), reviewer names, or "follow-up" pointers. Merge code is read by future maintainers with no context on this PR's lifecycle — issue numbers go stale, AC labels mean nothing outside one run, reviewer-driven rationale rots. The right homes for that metadata:

  - **Long-form rationale** → commit message body
  - **Cross-cutting context, audit links** → PR description
  - **Gap detail / tracker** → the follow-up issue itself (don't sprinkle pointers to it across N manifest entries)
  - **Why a `status: spec-only` entry stays spec-only** → a short *technical* reason on that line is acceptable (`# kernel only supports float`); never an issue/PR number

  Specific patterns that must NOT appear in shipped source:

  ```
  # Follow-up: #N
  # Tracked in #N
  # AC-X of issue #N
  # Per round-N review / Per reviewer X
  # Added in PR #N / Reverted in commit abc1234
  ```

  Same rule for docstrings — docstrings describe what the code does, not which issue authorized it. Auditing "which issue caused this change" is `git blame` + commit message territory.

  Scan (case-insensitive; word-boundary on `#N` to avoid false-positives on inline numeric constants like `# 1024 elements`): `grep -rniE '(^|[[:space:]])#[0-9]{3,}\b|AC-[0-9]+|round-[0-9]+ review|follow-up:[[:space:]]*#' tileops/ tests/ benchmarks/ scripts/` should return nothing. The regex is a heuristic — a hit is a prompt to inspect, not an automatic blocker.
