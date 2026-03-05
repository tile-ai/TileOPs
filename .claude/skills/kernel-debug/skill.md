---
name: kernel-debug
description: Debugging SOP for TileLang GPU kernels in TileOPs — correctness triage workflow, common failure signals, and when to check the wiki KB
---

# TileOps / TileLang Kernel Debugging Guide

> Step-by-step process for diagnosing correctness failures in TileLang kernels.
> TileLang-specific facts (API pitfalls, error semantics, JIT cache behaviour) live in the wiki Knowledge Base — this skill references them rather than duplicating them.

______________________________________________________________________

## 1. Correctness Failure Triage

### 1.1 Check if the error is config-specific or shape-specific

**First question**: does the error reproduce across ALL configs (block sizes, pipeline stages) for the same shape?

| Error pattern                                   | Likely cause                                                  |
| ----------------------------------------------- | ------------------------------------------------------------- |
| Same `max_err` for all configs, same shape      | Shape-level bug (alignment, index formula, boundary handling) |
| Error only for specific `(block_n, num_stages)` | Code-path bug in that branch                                  |
| Error only when `tune=True`                     | Autotune corrupts state or selects a broken config            |
| Intermittent / changes across runs              | Race condition or uninitialized memory                        |

For a full interpretation table and sweep recipe → [Validation Rules — Error Pattern Interpretation](https://github.com/tile-ai/TileOPs/wiki/TileLang-Validation-Rules#error-pattern-interpretation)

**How to test**: run the kernel directly with several explicit configs for the failing shape:

```python
configs = [(1, 32, 1), (4, 32, 2), (8, 32, 3), ...]
for block_n, rt, ns in configs:
    out = wrapped_kernel(n, k, dtype, block_n, rt, ns, a, b)
    err = (out - ref).abs().max().item()
    print(f"block_n={block_n} ns={ns} max_err={err:.4f}")
```

### 1.2 Distinguish numerical precision from implementation bugs

- **fp16 matmul**: expected `max_err ≈ 1e-2` for large K (different rounding paths)
- **Exact powers of 2** (0.125, 0.25, 0.5): systematic index offset — not rounding
- **`max_err > 0.1` with `atol=1e-3`**: almost certainly an implementation bug

### 1.3 Random seed dependence in pytest vs inline scripts

`torch.manual_seed(1235)` at session start does NOT guarantee the same tensor values as pytest (pytest advances the RNG through all preceding tests). Use inline scripts to scan configs; use pytest for the authoritative pass/fail verdict.

______________________________________________________________________

## 2. TileLang-Specific Pitfalls (Quick Reference)

These are the most common TileLang API mistakes encountered in TileOps. Full details in the wiki.

| Pitfall                                                         | Fix                                                     | Wiki reference                                                                                                                 |
| --------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `num_stages=0` as sentinel for `T.Pipelined`                    | Always use `num_stages >= 1`; use `1` for sequential    | [Anti-Patterns #25](https://github.com/tile-ai/TileOPs/wiki/TileLang-Anti-Patterns#25-num_stages0-as-sentinel-for-t-pipelined) |
| `T.copy` on SM90 uses TMA by default; fails for non-WGMMA shmem | Add `disable_tma=True`                                  | [Language Spec — Data Movement](https://github.com/tile-ai/TileOPs/wiki/TileLang-Language-Spec#data-movement)                  |
| `T.syncthreads()` — wrong name                                  | Use `T.sync_threads()` (underscore)                     | [Anti-Patterns #26](https://github.com/tile-ai/TileOPs/wiki/TileLang-Anti-Patterns#26-tsyncthreads--wrong-sync-api-name)       |
| JIT cache serves stale binary after structural change           | `tilelang.clear_cache()` or `rm -rf ~/.tilelang/cache/` | [Language Spec — JIT Cache](https://github.com/tile-ai/TileOPs/wiki/TileLang-Language-Spec#jit-compilation-cache)              |

______________________________________________________________________

## 3. Debugging Workflow (SOP)

```
1. Run test; record failing test ID and max_err
2. Run same shape with multiple explicit configs → shape-level bug vs config-level?
3. Inspect index formulas for boundary alignment (ceildiv, mod, OOB risk)
4. Check for semantically invalid parameter values (num_stages=0, etc.)
5. Clear JIT cache; re-run → stale cache eliminated?
6. If numerical: compare accumulation dtypes (fp16 vs fp32 vs tf32)
7. If systematic offset (power-of-2 error): inspect reduction scope — missing thread, double-counted element?
8. Add targeted prints (out[0:10], ref[0:10]) to find the pattern
```

______________________________________________________________________

## 4. Case Studies

| Case | Summary                                                                                                         | Detail                                                                                                      |
| ---- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| A    | `num_stages=0` sentinel → `max_err=0.125` across **all** configs; identical error = design flaw, not config bug | [Case Studies A](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies#case-a-num_stages0-sentinel) |

______________________________________________________________________

## 5. References

- [TileLang Anti-Patterns](https://github.com/tile-ai/TileOPs/wiki/TileLang-Anti-Patterns)
- [TileLang Language Spec](https://github.com/tile-ai/TileOPs/wiki/TileLang-Language-Spec)
- [TileLang Validation Rules](https://github.com/tile-ai/TileOPs/wiki/TileLang-Validation-Rules)
- [TileLang Case Studies](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies)
