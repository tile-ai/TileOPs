## Boundary

- **OWNS**: `benchmarks/`
- **MUST NOT WRITE**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `workloads/`, `tileops/manifest/`
- **MUST NOT** (oracle-leakage rule): import oracle/ref functions from `tests/` or `workloads/`. Reads of any other file are unrestricted.

→ [trust-model.md §Benchmark](../../docs/design/trust-model.md#benchmark) | [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks)

______________________________________________________________________

- Every benchmark must record ≥1 non-`"tileops"` baseline. If external baseline is conditional, add a local torch fallback.
- Tag names: lowercase, hyphen-separated. Tags starting with `"tileops"` = TileOPs entries; all others = baselines.
- `calculate_flops()` / `calculate_memory()` return `None` to omit the metric from the report.
- Benchmark shapes must reflect real DNN workloads. Use LLaMA-family dimensions as defaults. Annotate shape constants with the model/scenario they represent. Do not use arbitrary flat numbers (262K, 1M, 4M).
