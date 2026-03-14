# Op Structural Compliance Checklist

Items marked **[REQUIRED]** must pass — block the PR if they fail.
Items marked **[RECOMMENDED]** should pass — note in PR body if skipped with reason.

## Correctness & Safety

- [ ] **[REQUIRED]** `Op.forward` validates input dtype against `SUPPORTED_DTYPES` and raises `ValueError` for unsupported types — never let invalid dtypes reach the backend
- [ ] **[REQUIRED]** `Op.forward` validates input shape/numel matches the compiled kernel — prevents OOB GPU memory access
- [ ] **[REQUIRED]** No hardcoded narrow-type constants in kernels (e.g. `T.cast(1.0, "float16")`) — use `x.dtype` or an explicit wide intermediate type
- [ ] **[REQUIRED]** fp16/bf16 intermediate math that can overflow (cubic terms, division, exp) is promoted to fp32
- [ ] **[REQUIRED]** Runtime validation uses `ValueError`/`TypeError`, never `assert` (stripped under `python -O`)
- [ ] **[REQUIRED]** Output dtype matches PyTorch reference semantics (e.g. comparison ops → `bool`, not float 0/1)

## Kernel Structure

- [ ] **[REQUIRED]** `with T.Kernel()` is inside `@T.macro`, never directly in `@T.prim_func`
- [ ] **[REQUIRED]** `Kernel.forward` accepts only GPU tensors and only calls `self.kernel(config...)(tensors)` — no format conversion, batching, or dtype cast
- [ ] **[RECOMMENDED]** `_<op_name>_kernel(static_params) -> Callable` closure function exists
- [ ] **[RECOMMENDED]** `@tilelang.jit(out_idx=[...])` wraps a config-parameterised inner function
- [ ] **[RECOMMENDED]** No Python builtins (`float()`, `int()`, `math.cos()`) on TileLang IR nodes
- [ ] **[RECOMMENDED]** Tile-ops (`T.clear`, `T.copy`, `T.gemm`) are at `T.Kernel` scope, not inside `T.Parallel`

## Op Structure

- [ ] **[REQUIRED]** `Op.forward` owns all pre/post-processing; delegates GPU work to `self.kernel(...)` only
- [ ] **[REQUIRED]** `@torch.library.custom_op` + `.register_fake` wrapper exists for torch.compile compatibility
- [ ] **[RECOMMENDED]** `default_config` and `autotune_configs` properties are defined
- [ ] **[RECOMMENDED]** `supported_archs` class attribute is set
- [ ] **[RECOMMENDED]** `accum_dtype` is hardcoded in kernel — never a property, config key, or parameter
- [ ] **[RECOMMENDED]** `__init__` signature ends with `kernel_map=None, tune=False`; `dispatch_kernel(kernel_map)` called before kernel use

## Delivery

- [ ] **[REQUIRED]** Unit tests in `tests/ops/` with reference comparison (FP16 atol=1e-3, BF16 atol=1.6e-2)
- [ ] **[REQUIRED]** Tests cover unsupported-dtype rejection paths (expect `ValueError`)
- [ ] **[REQUIRED]** Dtype support matrix documented in PR body
- [ ] **[REQUIRED]** No issue references (`#123`, `TODO: see #456`) in source or test files — issues track goals in GitHub, not in code
- [ ] **[RECOMMENDED]** Benchmark class in `benchmarks/`
- [ ] **[RECOMMENDED]** `__init__.py` exports are synchronized (`__all__` + explicit re-exports)
