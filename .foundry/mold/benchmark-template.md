<!--
INSTRUCTIONS FOR THE AGENT (do not copy into the PR body).

This file is the PR-body table format only. Rules for writing the benchmark
itself — baselines, tags, shapes, dtypes, what may be asserted — are in
.claude/domain-rules/benchmark.md.

Layout principle: one section per op, one table per section, TileOPs and
baseline side-by-side on the same row so readers can compare without
mentally joining two tables.

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
- Drop the Shape column if the op only varies dtype; in that case put
  the fixed shape in the section header (e.g. `### {OpName} (4096, 4096)`)
  so the benchmark's scale stays visible. Never put autotune config
  (`block_m`, `threads`) in the table — implementation detail.
- **Preserve the original shape tuple — never flatten to a single
  element count.** Write `(4096, 4096)` for a 2D input, `(2, 4096, 128)`
  for a 3D one, and `(4194304,)` (or `(4M,)`) for a genuinely 1D op.
  A bare `4M` loses dimensionality: readers can't tell `(4M,)` from
  `(2K, 2K)` from `(64, 64K)`, yet those have very different access
  patterns. Use `K`/`M` only inside a tuple to keep large numbers
  readable, never to replace it.
- Environment block goes once at the top, not per op.
- Takeaways = conclusions, not data repetition. Wins, losses with a brief
  reason (not blocking), dtype/shape patterns.
-->

## Benchmark

**Environment**: \{GPU}, CUDA \{ver}, PyTorch \{ver}, TileLang \{ver}

### \{OpName}

| Shape | dtype | TileOPs (ms) | \{baseline} (ms) | Speedup | BW (TB/s) |
| ----- | ----- | ------------ | ---------------- | ------- | --------- |

<!-- Repeat one ### section per op. Drop the Shape column if the op only
varies dtype. Swap `BW (TB/s)` for `TFLOPS` on compute-bound ops.
Shape values are tuples: `(4096, 4096)`, `(2, 4096, 128)`, `(4M,)`. -->

**Takeaways:** {wins · losses with brief reason · dtype/shape patterns}

**Command:** `PYTHONPATH="$PWD" python -m pytest benchmarks/ops/bench_{op}.py -v`
