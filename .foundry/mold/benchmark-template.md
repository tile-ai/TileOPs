# Benchmark Template

## Bench-file rules

- All dtypes in `SUPPORTED_DTYPES`; ≥3 shapes per op, include non-pow2 if supported.
- Shapes must map to real model geometry. Default to LLaMA-family sizes: hidden ∈ {4096, 5120, 8192}, intermediate ∈ {10240, 11008, 14336, 20480, 28672}, seq_len ∈ {2048, 4096}.
- Every benchmark **must** record at least one baseline. External baselines (FA3 / fla / triton) may be conditional, but the `else` branch must fall back to `test.ref_program` / torch — never silently skip.
- `BenchmarkReport.record()` first arg must be the Op object, never a string literal.

## Tag convention

| Tag                      | Meaning                                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `tileops`                | TileOPs implementation (required, exactly once per config). Variants like `tileops-lut` also count as TileOPs entries. |
| `torch`                  | PyTorch reference                                                                                                      |
| `FA3` / `fla` / `triton` | External baselines                                                                                                     |

## Code skeleton

```python
# benchmarks/ops/bench_<op>.py

class MyOpBenchmark(BenchmarkBase):
    def calculate_flops(self): ...   # MUST return non-None
    def calculate_memory(self): ...  # MUST return non-None

def test_my_op_bench(...):
    op = MyOp(...)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
    baseline_fn = _external_baseline(test)
    if baseline_fn is not None:
        BenchmarkReport.record(op, locals(), bm.profile(baseline_fn, *inputs), tag="FA3")
    else:
        BenchmarkReport.record(op, locals(), bm.profile(test.ref_program, *inputs), tag="torch")
```

## PR body format

**One row per measurement** (op × shape × dtype). Don't aggregate multiple measurements into a single cell. Don't include the bench file path as a column — it's noise; name the bench at the section header instead.

```markdown
## Benchmark

**Environment**: <GPU>, CUDA <ver>, PyTorch <ver>, TileLang <ver>

| Op | Shape | dtype | TileOPs (ms) | Baseline (ms) | Speedup | TFLOPS | BW (TB/s) |
|----|-------|-------|--------------|---------------|---------|--------|-----------|

**Takeaways:** wins · losses with brief reason (not blocking) · dtype/shape patterns. Conclusions only — no data repetition.

**Command:** `PYTHONPATH="$PWD" python -m pytest benchmarks/ops/bench_<op>.py -v`
```

Multiple ops → group with sub-headers. TFLOPS not meaningful (pure data movement) → `—`.
