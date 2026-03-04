---
name: check-kernel-format
description: Verify that a kernel and op conform to format requirements after delivery or tuning.
allowed-tools: Read, Grep, Glob
---

# Check Kernel Format

> **When to run**: after kernel+op delivery via `migrating-new-op` and after tuning via
> `tune`. If any item fails, fix it before opening a PR.

First, read `docs/kernel-op-conventions.md` for correct code patterns and rationale.

______________________________________________________________________

## 1. `T.prim_func` Structure (conventions §1)

- [ ] `_<op_name>_kernel(static_params) -> Callable` closure exists
- [ ] `@tilelang.jit(out_idx=[...])` wraps the config-parameterised inner function
- [ ] `with T.Kernel()` blocks are inside `@T.macro`, **not** directly in `@T.prim_func`
- [ ] No Python builtins applied to TileLang IR nodes — use `T.cast`, `T.cos`, etc.
- [ ] Tile-ops (`T.clear`, `T.copy`, `T.gemm`) are at `T.Kernel` scope, not inside `T.Parallel`

## 2. `Kernel` Class Structure (conventions §2)

- [ ] `Kernel.forward` takes only GPU tensor args — no format conversion or batching loop
- [ ] `Kernel.forward` body is `return self.kernel(config...)(tensors)` and nothing else
- [ ] `default_config`, `autotune_configs`, and `supported_archs` are defined
- [ ] `accum_dtype` is hardcoded in the kernel — not a property, config key, or parameter
- [ ] `@torch.library.custom_op` + `.register_fake` wrapper exists

## 3. `Op` Class Structure (conventions §3)

- [ ] `Op.forward` owns all pre/post-processing (format conversion, dtype cast, batching)
- [ ] `Op.forward` delegates GPU computation to `self.kernel(...)`
- [ ] `accum_dtype` is not stored on `Op` and not a parameter of `Op.__init__`
- [ ] `kernel_map: Optional[Dict[str, Kernel]] = None` is the last `__init__` parameter
- [ ] `dispatch_kernel(kernel_map)` is called before any kernel is instantiated
