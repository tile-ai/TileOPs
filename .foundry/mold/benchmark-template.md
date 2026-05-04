<!--
INSTRUCTIONS FOR THE AGENT (do not copy into the PR body).

Filling in the template below:

- One row per measurement (op × shape × dtype). Don't aggregate multiple
  measurements into a single cell.
- Don't include the bench file path as a column — it's noise. If multiple
  bench files contribute, group with sub-headers per bench.
- TFLOPS not meaningful (pure data movement) → use "—".
- Takeaways = conclusions, not data repetition. Wins, losses with a brief
  reason (not blocking), dtype/shape patterns.

Bench-file authoring rules (apply when writing benchmarks/ops/*.py, NOT
copied into the PR body):

- All dtypes in `SUPPORTED_DTYPES`; ≥3 shapes per op, include non-pow2
  if supported.
- Shapes must map to real model geometry. Default LLaMA sizes:
  hidden ∈ {4096, 5120, 8192}, intermediate ∈ {10240, 11008, 14336,
  20480, 28672}, seq_len ∈ {2048, 4096}.
- Every benchmark must record at least one baseline. Tags: `tileops`
  (TileOPs implementation; required exactly once per config; variants
  like `tileops-lut` also count) and `torch` / `FA3` / `fla` / `triton`
  (baselines for comparison). External baselines may be conditional,
  but the else branch must fall back to torch — never silently skip.
- `BenchmarkReport.record()` first arg must be the Op object, never a
  string literal.
-->

## Benchmark

**Environment**: <GPU>, CUDA <ver>, PyTorch <ver>, TileLang <ver>

| Op  | Shape | dtype | TileOPs (ms) | Baseline (ms) | Speedup | TFLOPS | BW (TB/s) |
| --- | ----- | ----- | ------------ | ------------- | ------- | ------ | --------- |

**Takeaways:** \<wins · losses with brief reason · dtype/shape patterns>

**Command:** `PYTHONPATH="$PWD" python -m pytest benchmarks/ops/bench_<op>.py -v`
