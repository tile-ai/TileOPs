# Benchmark Template

## Rules

1. All dtypes in `SUPPORTED_DTYPES`
1. ≥3 shapes per op; include non-pow2 if supported
1. Shapes must map to real model geometry, not arbitrary flat numbers. Match the op's natural dimensionality (1D/2D/3D). Default to LLaMA-family sizes: hidden ∈ {4096, 5120, 8192}, intermediate ∈ {10240, 11008, 14336, 20480, 28672}, seq_len ∈ {2048, 4096}.
1. Baseline: every benchmark **must** record at least one baseline. If an external baseline (FA3/fla) is conditional, add a `test.ref_program` / torch fallback in the `else` branch. No benchmark should silently skip baseline.
1. `BenchmarkReport.record()` first argument **must** be the Op object, never a string literal.
1. Metrics: latency (ms), bandwidth (TB/s), TFLOPs + issue-specific
1. Environment: GPU, CUDA version, PyTorch version, TileLang version

## Tag convention

| Tag          | Meaning                                                    |
| ------------ | ---------------------------------------------------------- |
| `"tileops"`  | TileOPs implementation (required, exactly once per config) |
| `"torch"`    | PyTorch reference implementation                           |
| `"FA3"`      | FlashAttention 3                                           |
| `"fla"`      | Functional Linear Attention                                |
| `"triton"`   | Triton kernel baseline                                     |
| `"baseline"` | Generic torch baseline (legacy, prefer specific name)      |

Tags starting with `"tileops"` (e.g. `"tileops-lut"`) are treated as TileOPs entries. All other tags are treated as baselines.

## Code skeleton

```python
# benchmarks/ops/bench_<op>.py


class MyOpBenchmark(BenchmarkBase):
    def calculate_flops(self): ...   # MUST return non-None
    def calculate_memory(self): ...  # MUST return non-None


def test_my_op_bench(...):
    op = MyOp(...)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")  # Op object, not string

    # Baseline (torch fallback mandatory)
    baseline_fn = _external_baseline(test)  # e.g. FA3/fla
    if baseline_fn is not None:
        result_bl = bm.profile(baseline_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="FA3")  # Update tag to match baseline
    else:
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")
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
