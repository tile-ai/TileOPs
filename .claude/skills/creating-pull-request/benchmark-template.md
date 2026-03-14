# Benchmark Section Template

PR benchmarks are lightweight performance profiles, not nightly regression suites.

## Rules

1. Benchmark every dtype in `SUPPORTED_DTYPES`
1. Op developer provides a set of benchmark shapes. Each op ≥3 shapes. Include non-pow2 if supported.
1. Baseline: **new ops** → PyTorch equivalent required; **op modifications** (strategy/optimization) → before/after required, PyTorch recommended
1. Required metrics: latency (ms), bandwidth (TB/s), TFLOPs. If issue specifies op-specific metrics, include those too.
1. Environment: GPU model, CUDA version, PyTorch version, TileLang version

## Benchmark code

```python
# benchmarks/ops/bench_<op>.py


class MyOpBenchmark(BenchmarkBase):
    def calculate_flops(self): ...  # None for pure data-movement ops
    def calculate_memory(self): ...  # must return non-None


# Profile both: BenchmarkReport.record(..., tag="tileops") / tag="baseline"
# Cover: SUPPORTED_DTYPES × shapes (≥3 per op)
```

## PR body format

`BenchmarkReport.dump()` auto-generates a unified table with speedup (baseline latency / tileops latency) when tileops + baseline pairs exist. Paste the generated table directly.

````markdown
## Benchmark

**Environment**: \<GPU>, CUDA \<ver>, PyTorch \<ver>, TileLang \<ver>

<!-- Paste the table from BenchmarkReport.dump() output.
     Speedup is auto-calculated. Add op-specific metric columns if issue requires them. -->

| params... | tileops lat(ms) | baseline lat(ms) | tflops | bandwidth(TB/s) | speedup |
|-----------|-----------------|-------------------|--------|-----------------|---------|
| ... | ... | ... | ... | ... | ...x |

**Takeaways:**
- \<wins: what's faster, by how much>
- \<losses: what's slower, why>
- \<patterns: dtype/shape trends>

**Benchmark command:**
\```bash
PYTHONPATH="$PWD" python -m pytest benchmarks/ops/bench_<op>.py -v
\```
````

## Formatting notes

- Speedup is auto-calculated by `BenchmarkReport.dump()` when tileops/baseline pairs exist
- Multiple ops → group by op with sub-headers
- Slower than baseline → brief reason (informational, not blocking)
- TFLOPs not meaningful (pure data movement) → `calculate_flops()` returns None, displayed as `—`
- **Takeaways** required — concise conclusions, not data repetition
