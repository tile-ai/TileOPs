# Elementwise Performance Evidence

Reasoning and data behind [elementwise.md](elementwise.md). Test environment defined in [README.md](README.md).

______________________________________________________________________

## §1. Strategy Selection

| Category          | Strategy            | Key Number                        |
| ----------------- | ------------------- | --------------------------------- |
| Unary             | `register_copy`     | 3.53 vs 0.99 TB/s (relu 16M fp16) |
| Binary broadcast  | `explicit_parallel` | 3.65 TB/s (add 16M fp16)          |
| Binary same-shape | `register_copy`     | max/min 121–146% PyTorch          |
| FusedGated        | `explicit_parallel` | 3.38 vs 1.51 TB/s                 |

`register_copy` emits `uint4` 128-bit loads via `T.copy`. TVM auto-vectorizer fails on complex `op_func`, making `T.copy` necessary.

## §2. Vectorization

`T.copy` → 128-bit `uint4` loads (8×fp16 / 4×fp32). 2–3× BW gain.

**Bool**: TileLang cannot vectorize bool. Pack as `uint8` in Op layer → masked_fill: 0.81× → **1.55×** (fp16), 0.88× → **1.86×** (fp32).

## §3. Configuration

**npt**: `explicit_parallel` stride = `npt × elem_size`; larger npt → worse coalescing for small dtypes. `register_copy` uses cooperative `T.copy`; larger npt → bigger tile → better amortization.

| Strategy            | fp16/bf16 | fp32 | fp8 |
| ------------------- | --------- | ---- | --- |
| `explicit_parallel` | 4         | 4    | 16  |
| `register_copy`     | 8         | 4    | 16  |

fp16 npt=4 vs 8 under explicit_parallel: **+42%** (1.05 vs 0.74 TB/s). Thread count 256 vs 128: \<5% delta. Deployed as strategy-aware `_strategy_npt()` in `default_config` (#553).

## §4. IR Complexity

max/min: 5-node chain → `T.max` + `isnan`: 1.32 → **3.36 TB/s** (+155%). Prefer TileLang intrinsics.

## §5. Caching

`init_config()` stores `_compiled_fn`. Cold ~1.1s, disk-cached ~25ms, warm \<1ms.

## §6. Register Pressure

In-place write to input fragment, not a new `T.alloc_fragment` for output.

## §7. Profiling

- **Primary backend**: CUPTI — gives pure kernel time without host-launch overhead
- **Fallback**: CUDA event + median — only when CUPTI returns 0 (global singleton held by `ncu` or another process)
- **CUPTI filter fix**: tilelang excludes `vectorized_elementwise` kernels to strip `cache.zero_()`, but this also kills PyTorch baselines. Monkey-patch requires both `vectorized_elementwise` AND `FillFunctor` in kernel name to exclude
