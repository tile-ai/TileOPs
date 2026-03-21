# Op Structural Readiness Checklist

`[REQUIRED]` → block PR on failure. `[RECOMMENDED]` → note skip reason in PR body.

## Correctness & Safety

- [ ] [REQ] `Op.forward` validates input shape/numel before kernel launch
- [ ] [REQ] `Kernel.__init__` validates dtype against `SUPPORTED_DTYPES`, raises `ValueError` (template kernels inherit; independent kernels add explicitly)
- [ ] [REQ] User-provided scalar params are validated at the Op/API boundary against the effective kernel dtype before codegen; do not rely on TIR/lowering failures or kernel-local hotfixes to reject invalid values
- [ ] [REQ] No hardcoded narrow-type constants (`T.cast(1.0, "float16")`) — use `x.dtype` or wide intermediate
- [ ] [REQ] fp16/bf16 math that can overflow (cubic, div, exp) promoted to fp32
- [ ] [REQ] Runtime validation uses `ValueError`/`TypeError`, never `assert`
- [ ] [REQ] Output dtype matches PyTorch semantics (comparison → `bool`, not float)

## Kernel Structure

- [ ] [REQ] `with T.Kernel()` inside `@T.prim_func`; reusable sub-routines as `@T.macro`
- [ ] [REQ] `Kernel.forward` accepts only GPU tensors, calls only `self.kernel(config...)(tensors)`
- [ ] [REQ] Builder function (`_<op>_kernel`) decorated with `@functools.lru_cache(maxsize=32)`; all params hashable
- [ ] [REC] `_<op>_kernel(static_params) -> Callable` closure exists
- [ ] [REC] `@tilelang.jit(out_idx=[...])` wraps config-parameterised inner function
- [ ] [REC] No Python builtins (`float()`, `int()`, `math.cos()`) on IR nodes
- [ ] [REC] Tile-ops (`T.clear`, `T.copy`, `T.gemm`) at `T.Kernel` scope, not inside `T.Parallel`

## Op Structure

- [ ] [REQ] `Op.forward` owns pre/post-processing; delegates GPU work to `self.kernel(...)` only
- [ ] [REQ] `@torch.library.custom_op` + `.register_fake` for torch.compile
- [ ] [REC] `default_config` and `autotune_configs` properties on Kernel
- [ ] [REC] `supported_archs` class attribute on Kernel
- [ ] [REC] `accum_dtype` hardcoded — never property/config/parameter
- [ ] [REC] Template ops: `__init__(…, kernel_map=None, tune=False)` + `dispatch_kernel(kernel_map)`. Independent ops: `Kernel.__init__(…, config=None, tune=False)`

## Benchmark

- [ ] [REQ] `benchmarks/ops/bench_<op>.py` exists, inherits `BenchmarkBase`
- [ ] [REQ] `calculate_flops()` and `calculate_memory()` return non-None
- [ ] [REQ] ≥3 shapes × all `SUPPORTED_DTYPES`; include non-pow2 if supported
- [ ] [REQ] `BenchmarkReport.record(op, ...)` uses Op object as first argument, not a string
- [ ] [REQ] At least one baseline recorded; if external baseline is conditional, torch fallback in `else` branch
- [ ] [REQ] Metrics: latency (ms), bandwidth (TB/s), TFLOPs + issue-specific metrics
- [ ] [REQ] PR body `## Benchmark`: environment + table + command + **Takeaways**
- [ ] [REC] Benchmark with default parameter values; multiple values recommended

## Delivery

- [ ] [REQ] Unit tests in `tests/ops/` with reference comparison (FP16 atol=1e-3, BF16 atol=1.6e-2)
- [ ] [REQ] Tests cover unsupported-dtype rejection (`ValueError`)
- [ ] [REQ] Dtype support matrix in PR body
- [ ] [REQ] No issue references (`#123`, `TODO: see #456`) in source/test files
- [ ] [REC] `__init__.py` exports synchronized (`__all__` + re-exports)
