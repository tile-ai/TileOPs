## Boundary

- **OWNS**: `benchmarks/`
- **MUST NOT modify**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `workloads/`, `tileops/manifest/`
- **MAY READ**: `workloads/`, `tileops/ops/`

→ [trust-model.md §Benchmark](../../docs/trust-model.md#benchmark) | [testing.md §Benchmarks](../../docs/testing.md#benchmarks)

______________________________________________________________________

- `BenchmarkReport.record()` first arg: Op object (preferred) or string name. Stay consistent per file.
- Every benchmark must record ≥1 non-`"tileops"` baseline. If external baseline is conditional, add a local torch fallback.
- Tag names: lowercase, hyphen-separated. `"torch"`, `"torch-cublas"`, `"torch-sdpa"`, `"fa3"`, `"fla"`, `"triton"`, etc. Tags starting with `"tileops"` = TileOPs entries; all others = baselines.
- `calculate_flops()` / `calculate_memory()`: return numeric or `None` (omits metric from report).
- Benchmark shapes must reflect real DNN workloads. Use LLaMA-family dimensions as defaults. Annotate shape constants with the model/scenario they represent. Do not use arbitrary flat numbers (262K, 1M, 4M).
