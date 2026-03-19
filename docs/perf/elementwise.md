# Elementwise Kernel Performance Checklist

> These are empirically-backed heuristics, not hard requirements. A violation is not necessarily wrong — but should be **deliberate and documented**. Flag unchecked items as "needs justification", not "needs fix". See [elementwise-evidence.md](elementwise-evidence.md) for reasoning and data.

## Strategy (§1)

- [ ] Unary-like kernel → `register_copy`
- [ ] Binary broadcast → `explicit_parallel`
- [ ] Binary same-shape → auto-selects `register_copy`

## Vectorization (§2)

- [ ] `register_copy` path uses `T.alloc_fragment` + `T.copy`
- [ ] Bool inputs packed as `uint8` in Op layer

## Config (§3)

- [ ] `explicit_parallel` npt: fp16/bf16=4, fp32=4, fp8=16
- [ ] `register_copy` npt: fp16/bf16=8, fp32=4, fp8=16
- [ ] `autotune_configs` defined for all template kernels (Unary, Binary, FusedGated): threads∈{128,256,512} × npt∈{2,4,8}; fp8: npt∈{16,32}
- [ ] All kernel classes (template and custom) cache `_compiled_fn` in `init_config()`
- [ ] Serialization-fallback `autotune()` override for template kernels with closure-based `op_func`

## Code Quality (§4–§6)

- [ ] `op_func` uses TileLang intrinsics over manual comparison chains
- [ ] Results written in-place to input register fragment
