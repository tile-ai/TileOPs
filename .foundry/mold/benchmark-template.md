# Benchmark Template

## Rules

1. All dtypes in `SUPPORTED_DTYPES`
1. ≥3 shapes per op; include non-pow2 if supported
1. Shapes must map to real model geometry, not arbitrary flat numbers. Match the op's natural dimensionality (1D/2D/3D). Default to LLaMA-family sizes: hidden ∈ {4096, 5120, 8192}, intermediate ∈ {10240, 11008, 14336, 20480, 28672}, seq_len ∈ {2048, 4096}.
1. Baseline: new ops → PyTorch required; modifications → before/after required, PyTorch recommended
1. Metrics: latency (ms), bandwidth (TB/s), TFLOPs + issue-specific
1. Environment: GPU, CUDA version, PyTorch version, TileLang version

## Code skeleton

```python
# benchmarks/ops/bench_<op>.py


class MyOpBenchmark(BenchmarkBase):
    def calculate_flops(self): ...  # non-None
    def calculate_memory(self): ...  # non-None


# BenchmarkReport.record(..., tag="tileops") / tag="baseline"
# Cover: SUPPORTED_DTYPES × shapes (≥3)
```

## PR body format

````markdown
## Benchmark

**Environment**: <GPU>, CUDA <ver>, PyTorch <ver>, TileLang <ver>

| Op | Shape | dtype | … | Speedup |
|----|-------|-------|---|---------|

**Takeaways:**
- <wins>
- <losses + reason>
- <dtype/shape patterns>

**Benchmark command:**
```bash
PYTHONPATH="$PWD" python -m pytest benchmarks/ops/bench_<op>.py -v
```
````

## Notes

- Multiple ops → group with sub-headers
- Slower than baseline → brief reason (not blocking)
- TFLOPs not meaningful (pure data movement) → `—`
- **Takeaways** required — conclusions, not data repetition
