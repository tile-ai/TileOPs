---
name: kernel-debug
description: Debugging patterns and pitfalls for TileLang GPU kernels in TileOPs — correctness triage, JIT cache, pipeline pitfalls
---

# TileOps / TileLang Kernel Debugging Guide

> Debugging patterns and pitfalls when writing and tuning custom GPU kernels with TileLang, based on hands-on experience in TileOps.

______________________________________________________________________

## 1. Correctness Failure Triage

### 1.1 Check if the error is config-specific or shape-specific

**First question**: does the error reproduce across ALL configs (block sizes, pipeline stages) for the same shape?

| Error pattern                                 | Likely cause                                                  |
| --------------------------------------------- | ------------------------------------------------------------- |
| Same max_err for all configs, same shape      | Shape-level bug (alignment, index formula, boundary handling) |
| Error only for specific (block_n, num_stages) | Code path bug in that branch                                  |
| Error only when `tune=True`                   | Autotune corrupts state or selects a broken config            |
| Intermittent / changes across runs            | Race condition or uninitialized memory                        |

**How to test**: Run the kernel directly with several explicit configs for the failing shape:

```python
configs = [(1, 32, 1), (4, 32, 2), (8, 32, 3), ...]
for block_n, rt, ns in configs:
    out = wrapped_kernel(n, k, dtype, block_n, rt, ns, a, b)
    err = (out - ref).abs().max().item()
    print(f"block_n={block_n} ns={ns} max_err={err:.4f}")
```

### 1.2 Distinguish numerical precision from implementation bugs

- **fp16 matmul**: expected max_err ≈ `1e-2` for large K (PyTorch uses Tensor Cores, our kernel uses fp32 accum — different rounding paths)
- **Suspicious values**: exact powers of 2 (0.125, 0.25, 0.5) usually indicate a **systematic offset** from a wrong index or missing reduction term, not rounding
- **max_err > 0.1 with `atol=1e-3`**: almost certainly an implementation bug, not numerical

### 1.3 Random seed dependence in pytest vs inline scripts

**Pitfall**: `torch.manual_seed(1235)` at session start does NOT mean inline scripts reproduce the same tensor values as pytest. pytest advances the RNG through all preceding test cases before reaching the failing one.

**Consequence**: an inline diagnostic script may show different max_err from what pytest actually observes. Do not use inline diagnostics to confirm "the tests now pass" — always run pytest.

**Rule**: use inline scripts only to scan across configs; use pytest for the authoritative pass/fail verdict.

______________________________________________________________________

## 2. TileLang-Specific Pitfalls

### 2.1 Sentinel values for pipeline depth

**Bad pattern**: using `num_stages=0` as "disable pipeline" sentinel.

```python
# WRONG: ambiguous, semantically invalid for T.Pipelined
if num_stages > 0:
    for bk in T.Pipelined(..., num_stages=num_stages):
        ...
else:
    for bk in T.serial(...):
        ...
```

**Problem**: `T.Pipelined` requires `num_stages >= 1`. Using 0 as a sentinel mixes concerns (pipeline depth vs. code path selection), and if `@tilelang.jit` traces `@T.prim_func` in a way that doesn't treat the `if` as purely compile-time, both branches may interfere.

**Correct pattern**: always `num_stages >= 1`, let the pipeline handle the "no overlap" case with `num_stages=1`:

```python
# CORRECT: num_stages=1 = sequential (no overlap), >=2 = actual pipeline
b_shared = T.alloc_shared((block_n, block_k), dtype)
for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
    T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
    ...
```

**Autotune configs**:

```python
# num_stages=1 is the baseline (shmem, no pipeline overlap)
# num_stages=2,3 are the actual candidates for latency hiding
for ns in [1, 2, 3]:
    ...
```

### 2.2 TMA vs cp.async for T.copy

`T.copy` on SM90 defaults to TMA, which requires `mbarrier` PTX and layout inference that TileLang cannot always infer for manually-indexed shared memory in non-WGMMA kernels.

**Symptom**: compilation error mentioning `mbarrier` or undefined symbol.

**Fix**: add `disable_tma=True` to use `cp.async` instead:

```python
T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
```

Note: `cp.async` still hides HBM latency when used inside `T.Pipelined`, just without TMA's extra features.

### 2.3 Thread sync API

TileLang uses `T.sync_threads()` (with underscore), **not** `T.syncthreads()`.

```python
# WRONG
T.syncthreads()  # AttributeError: module has no attribute 'syncthreads'

# CORRECT
T.sync_threads()
```

### 2.4 JIT cache staleness

`@tilelang.jit` caches compiled kernels on disk. When the kernel signature or structure changes (e.g., adding a new parameter), the old cached binary may be reused if the cache key doesn't capture the change.

**Symptom**: correctness failure that disappears after clearing the cache or in a fresh environment.

**When to clear**: after any structural change to `_gemv_func` or `_gemv_main` (new parameters, changed branching logic).

**Cache location**: typically `~/.tilelang/cache/` or `$TILELANG_CACHE_DIR`.

______________________________________________________________________

## 3. Debugging Workflow (SOP)

```
1. Run test; record failing test ID and max_err
2. Run same shape with multiple explicit configs → shape-level bug vs config-level?
3. Inspect index formulas for boundary alignment (ceildiv, mod, OOB risk)
4. Check for semantically invalid parameter values (num_stages=0, etc.)
5. Clear JIT cache; re-run → stale cache eliminated?
6. If numerical: compare accumulation dtypes (fp16 vs fp32 vs tf32)
7. If systematic offset: inspect reduction scope — missing thread, double-counted element?
8. Add targeted prints (out[0:10], ref[0:10]) to find the pattern
```

______________________________________________________________________

## 4. Case Studies

### Case A: `num_stages=0` caused persistent correctness failures (2026-02-27)

**Shape**: `(n=18432, k=7168, fp16, tune=True)`
**Error**: `max_err=0.125`–`0.25` across ALL configs, reproducible
**Initial wrong hypothesis**: config-specific bug in the autotune winner (block_n=4)
**Actual root cause**: `num_stages=0` is semantically invalid for a pipeline parameter; the Python `if num_stages > 0: ... else:` inside `@T.prim_func` created ambiguity in TileLang's tracing, potentially including both branches or causing JIT cache conflicts when the signature changed from O1 (no `num_stages` param) to O3 (with `num_stages` param)

**Fix**: Removed `else` branch entirely. Always use `T.Pipelined` with `num_stages >= 1`. Register-only fallback eliminated from autotune.

**Diagnostic that cracked it**:

```
block_n= 1 rt=32 ns=0  max_err=0.1250  [FAIL]
block_n= 4 rt=32 ns=0  max_err=0.1250  [FAIL]
block_n= 4 rt=32 ns=2  max_err=0.1250  [FAIL]
→ all configs fail with identical error → not config-specific → design flaw
```

**Lesson**: identical max_err across all configs → the design concept is wrong, not the implementation detail. Fix the concept first.

______________________________________________________________________

## 5. References

- [TileLang Source — pipelined loops](https://github.com/tile-ai/tilelang)
- [CUDA cp.async Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [TVM TIR IfThenElse vs Python conditionals](https://tvm.apache.org/docs/reference/langref/relay_expr.html)
