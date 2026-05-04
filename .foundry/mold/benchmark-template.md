<!--
INSTRUCTIONS FOR THE AGENT (do not copy into the PR body).

Layout principle: one section per op, one table per section, TileOPs and
baseline side-by-side on the same row so readers can compare without
mentally joining two tables.

Filling in the template below:

- One row per measurement (shape × dtype) within an op's table.
- Baseline column header names the baseline (`torch (ms)`, `FA3 (ms)`,
  `triton (ms)`). Don't write a generic "Baseline". Multiple baselines →
  add more columns and more Speedup columns (`vs torch`, `vs FA3`).
- Speedup is always present — it's the first number readers look for.
  Format `4.96×`, two decimals, computed as baseline_ms / tileops_ms.
- Throughput column: show ONE — TFLOPS for compute-bound ops (matmul,
  attention), BW (TB/s) for memory-bound ops (reductions, elementwise,
  norms). Pure data movement → BW only. Don't list both.
- Show throughput for TileOPs only; baseline's absolute throughput is
  noise once Speedup is given.
- Drop the Shape column if the op only varies dtype. Never put autotune
  config (`block_m`, `threads`) in the table — implementation detail.
- Environment block goes once at the top, not per op.
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

**Environment**: \{GPU}, CUDA \{ver}, PyTorch \{ver}, TileLang \{ver}

### \{OpName}

| dtype | TileOPs (ms) | \{baseline} (ms) | Speedup | BW (TB/s) |
| ----- | ------------ | ---------------- | ------- | --------- |

<!-- Repeat one ### section per op. Add a Shape column on the left when
the op varies shape. Swap `BW (TB/s)` for `TFLOPS` on compute-bound ops. -->

**Takeaways:** {wins · losses with brief reason · dtype/shape patterns}

**Command:** `PYTHONPATH="$PWD" python -m pytest benchmarks/ops/bench_{op}.py -v`
