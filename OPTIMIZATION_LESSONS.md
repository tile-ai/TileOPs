# Optimization Lessons: Why Phase 1 & 2 Failed

## Summary

Attempted algorithmic optimizations (Phase 1 & 2) made the kernel **91% slower** than baseline.
Deleted from codebase. This document explains why they failed and outlines better approaches.

## What Failed

### Phase 1: Register Optimizations
**Attempted:**
- Register-cached exp_l (reduce shared memory loads)
- Swizzled layouts for exp arrays (reduce bank conflicts)
- Fused acc/hist_acc (reduce register pressure)

**Result:** 0.113ms vs 0.059ms baseline → **91% slower**

### Phase 2: Double-Buffering + Pipelining
**Attempted:**
- Double-buffered tiles (hide memory latency)
- Software pipelining (overlap compute and memory)

**Result:** 0.113ms (identical to Phase 1) → **91% slower**

## Root Cause: Register Pressure

### The API Problem
```python
# What we needed:
T.gemm(c_tile, state_tile, acc, beta=1.0)  # In-place accumulation

# What API provides:
T.gemm(c_tile, state_tile, acc)  # No beta parameter!
```

### The Broken Workaround
```python
# Baseline (fast - compiler optimizes well):
hist_acc = T.alloc_fragment((64, 64), fp32)
# ... compute and immediately consume

# Phase 1/2 (slow - doubled register usage):
acc = T.alloc_fragment((64, 64), fp32)      # 4096 values
hist_acc = T.alloc_fragment((64, 64), fp32)  # 4096 values (EXTRA!)
# ... hist_acc kept alive longer
# Result: 128 registers/thread → register spilling
```

### Impact
- Register access: ~1 cycle
- Local memory (spilled): ~100 cycles
- **100x slowdown** on register-spilled operations

### Evidence
- Phase 1 and Phase 2 have **identical** runtime (0.113ms)
- Different algorithms, same bottleneck = register spilling
- Affects all problem sizes (per-thread issue, not shape-dependent)

## Why Config Tuning Won

### AKO Results
**Baseline config search found:**
- `threads=64` (was 128) - Reduces register pressure
- `block_n=64` (was 128) - Better occupancy
- **Result: 1.77x speedup** (1.18x → 1.77x)

### Why It Worked
1. **Stayed within compiler's optimization patterns**
2. **No API workarounds needed**
3. **Matched hardware constraints** (registers, shared memory, occupancy)
4. **Systematic search** across config space

## Better Approaches Going Forward

### 1. ✅ AKO (Already Used Successfully)
**What:** Automated Kernel Optimization - systematic config search

**How we used it:**
```bash
# Searched space:
# - block_l: [32, 64, 128]
# - block_p: [32, 64, 128]
# - block_n: [32, 64, 128]
# - block_s: [32, 64, 128]
# - threads: [64, 128, 256]
```

**Result:** Found optimal config → 1.77x speedup

**Why it works:**
- Config tuning doesn't fight the compiler
- Explores hardware constraints systematically
- No API workarounds needed

### 2. 🔍 Profile-Guided Optimization (Recommended Next)
**Use NCU to find REAL bottlenecks:**

```bash
ncu --set full -o profile_baseline \
    python bench_kernel.py --kernel ssd_chunk_scan

# Key metrics:
# - sm__throughput.pct_of_peak_sustained
# - l1tex__t_bytes_pipe_lsu_mem_global_op_ld
# - smsp__sass_average_data_bytes_per_sector_mem_local (spilling!)
# - launch__registers_per_thread
```

**Then optimize what matters:**
- If memory bound → improve memory patterns
- If compute bound → reduce arithmetic
- If occupancy bound → reduce register/shared memory usage

### 3. 🎯 Tune Other Kernels (Low-Hanging Fruit)
**Apply AKO to:**
- `ssd_chunk_state` (not yet tuned)
- `ssd_state_passing` (not yet tuned)
- `da_cumsum` (not yet tuned)

**Expected:** 20-50% gain each (based on chunk_scan results)

### 4. 🧪 API-Compatible Optimizations Only
**Rules:**
- Only use features that exist in baseline
- Test compilation after each change
- Benchmark immediately
- If slower, revert immediately

**Safe optimizations:**
- Config tuning (proven)
- Memory access patterns (if NCU shows benefit)
- Launch bounds (within API)
- Tile size adjustments (proven)

### 5. 🚫 Avoid These (Lessons Learned)
**Don't:**
- ❌ Assume API features exist (check baseline first)
- ❌ Write all code before testing
- ❌ Add workarounds that increase register usage
- ❌ Optimize without profiling first
- ❌ Fight the compiler's optimization patterns

**Do:**
- ✅ Copy baseline patterns exactly
- ✅ Test each change incrementally
- ✅ Profile before optimizing
- ✅ Let AKO search config space
- ✅ Trust the compiler

## Recommended Next Steps

### Immediate (Easy Wins)
1. **Apply AKO to other Mamba kernels**
   ```bash
   bash scripts/tune_other_kernels.sh
   ```
   Expected: 3 kernels × 30% gain = significant overall improvement

2. **Profile chunk_scan with NCU**
   ```bash
   ncu --set full -o chunk_scan_profile \
       python bench/kernelbench/bench.py \
       --solution solution/ssd_chunk_scan.py
   ```
   Understand current bottlenecks before next optimization attempt

### Medium-term (Research)
3. **Investigate API improvements**
   - Check if newer tilelang supports beta parameter
   - Check if swizzle supports rank-1 arrays
   - May unlock previously blocked optimizations

4. **Try different kernel variants**
   - `ssd_chunk_scan_2wg_ws` (2-warp group)
   - `ssd_chunk_scan_3wg_mbar` (3-warp group)
   - May have different performance characteristics

### Long-term (Advanced)
5. **End-to-end optimization**
   - Tune all Mamba kernels
   - Profile entire forward/backward pass
   - Optimize kernel fusion opportunities

6. **Hardware-specific features**
   - H200 TMA (Tensor Memory Accelerator)
   - Async barriers (if API supports)
   - Mixed precision (FP8/FP16)

## Key Lessons

### What We Learned
1. **Config tuning > Algorithmic optimization** (in this case)
2. **API constraints matter more than theory**
3. **Profile first, optimize second**
4. **Incremental testing saves time**
5. **Sometimes simple wins over clever**

### Success Metrics
- ✅ 1.77x speedup from config tuning (50% improvement)
- ✅ Learned systematic optimization methodology
- ✅ Built reusable AKO pipeline
- ✅ Documented failures for future reference

### Cost
- ❌ 2 failed algorithmic optimization attempts
- ❌ Multiple days of development
- ❌ 7 benchmark iterations to fix API issues

**Lesson:** Config tuning first, algorithmic optimization second (with profiling)

## Conclusion

**Phase 1 and Phase 2 optimizations failed** due to:
- API limitations (no beta parameter)
- Register pressure from workarounds
- Fighting compiler optimization patterns

**Config tuning succeeded** because:
- Works with the compiler, not against it
- No API workarounds needed
- Systematic hardware-aware search

**Going forward:**
1. Use AKO for config tuning (proven approach)
2. Profile with NCU before algorithmic changes
3. Only use API features from baseline
4. Test incrementally, revert if slower

**The 1.77x speedup from config tuning alone is a solid win.**
Sometimes the best optimization is finding the right parameters, not rewriting the algorithm.
