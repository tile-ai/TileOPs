- **MUST NOT**: import from `tests/`, or import ref/oracle functions from `workloads/`. Reads of any other file are unrestricted.
- **MUST NOT**: author `gen_inputs`. Import the op's workload from `workloads/`; if it has none, add it there instead of copying locally. Subclassing an imported workload to attach a baseline is fine.

Both checked by `benchmarks/tests/test_benchmark_boundaries.py`, which matches names
literally — it will not catch a draw helper under another name.

→ [trust-model.md §Benchmark](../../docs/design/trust-model.md#benchmark) | [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks)

______________________________________________________________________

- Every benchmark records ≥1 non-`tileops` baseline. If the external baseline is conditional, add a local torch fallback.
- Tag names: lowercase, hyphen-separated. Tags starting with `tileops` are TileOPs entries; everything else is a baseline.
- `calculate_flops()` / `calculate_memory()` return `None` to omit the metric.
- Benchmark shapes reflect real DNN workloads (LLaMA-family by default). Annotate shape constants with the model/scenario; never use arbitrary flat numbers (262K, 1M, 4M).
