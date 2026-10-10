Rules a reader has to apply by hand. A rule that can be checked mechanically lives in
`scripts/lint/`, the ruff configuration or `.pre-commit-config.yaml`, which state why each
one holds.

- Every `src/tileops/kernels/*` subpackage MUST have an `__init__.py` with explicit `__all__` and `from tileops.kernels.<subpackage>.<module> import Symbol` re-exports.

- Reach a C++/CUDA source under `src/tileops/csrc/` through `tileops._csrc.csrc_path("<file>")`.

- Each TileLang kernel is one `@T.prim_func` whose body opens `with T.Kernel(...)`; sub-routines use `@T.macro`, never nested `prim_func`.

- A value only one kernel reads — a tile size, a thread count, a register budget, a barrier id, a mask value — is a local of the function that reads it, with its reason beside it. A module-level constant is reserved for a value several functions in the module must agree on, and its comment states that reason.

- Promote overflow-prone fp16/bf16 math (cubic, division, `exp`, softmax accumulators) to fp32; cast back to storage dtype at the boundary.

- Memoize each `_<op>_kernel` builder with `functools.lru_cache`; every parameter must be hashable.

- Tag code degraded by something outside its own scope — a contract stub that cannot be made abstract until every op migrates, a benchmark that must skip a manifest workload no kernel can run — with `FIXME(staged-rollout)`. Scan: `grep -rn 'FIXME(staged-rollout)'`.

  ```python
  # FIXME(staged-rollout): <one-line summary of what's degraded>
  #
  # Broken invariant: <what contract is currently violated>
  # Why: <which process constraint requires this temporary state>
  # Cleanup: <concrete condition that triggers removal of this marker>
  ```

- Abbreviation spellings have one source of truth, `ABBREVIATIONS` in `scripts/lint/op_naming_lint.py`; manifest entry names and classes follow it.

- Filenames are lowercase with underscores and spell an abbreviation out as a word (`rms_norm.py`, not `rmsnorm.py`).

- Docstrings: Google style. One-line summary, blank line, then optional `Args:` / `Returns:` / `Raises:` / `Example:`. Never mix Sphinx or NumPy headers in one file.

- A guarantee a caller acts on — an accuracy bound, a deviation from torch — goes in the op's class docstring, which the docs site renders; a kernel comment does not reach the caller.

- Expand domain abbreviations on first use in a docstring: `State Space Model (SSM)`, `State-Space Dual (SSD)`. Later uses may abbreviate.
