# Op Structural Compliance Checklist

## T.prim_func

- [ ] `_<op_name>_kernel(static_params) -> Callable` closure function exists
- [ ] `@tilelang.jit(out_idx=[...])` wraps a config-parameterised inner function
- [ ] `with T.Kernel()` is inside `@T.macro`, never directly in `@T.prim_func`
- [ ] No Python builtins (`float()`, `int()`, `math.cos()`) on TileLang IR nodes — use `T.cast`, `T.cos`, etc.
- [ ] Tile-ops (`T.clear`, `T.copy`, `T.gemm`) are at `T.Kernel` scope, not inside `T.Parallel`

## Kernel class

- [ ] `Kernel.forward` accepts only GPU tensors — no format conversion, batching, or dtype cast
- [ ] `Kernel.forward` body is only `return self.kernel(config...)(tensors)`
- [ ] `default_config` and `autotune_configs` properties are defined
- [ ] `supported_archs` class attribute is set
- [ ] `accum_dtype` is hardcoded in kernel — never a property, config key, or parameter
- [ ] `@torch.library.custom_op` + `.register_fake` wrapper exists

## Op class

- [ ] `Op.forward` owns all pre/post-processing (format conversion, dtype cast, batching, reshape)
- [ ] `Op.forward` delegates GPU work to `self.kernel(...)` only
- [ ] `accum_dtype` is not stored on `Op` and not a parameter of `Op.__init__`
- [ ] `kernel_map` is the last `__init__` parameter; `dispatch_kernel(kernel_map)` called before kernel use
- [ ] `__init__.py` exports are synchronized (`__all__` + explicit re-exports)

## Delivery

- [ ] Unit tests in `tests/ops/` with reference comparison (FP16 atol=1e-3, BF16 atol=1.6e-2)
- [ ] Benchmark class in `benchmarks/`
- [ ] Dtype support matrix documented in PR body
