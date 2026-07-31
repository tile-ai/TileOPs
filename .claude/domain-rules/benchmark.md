## Boundary

- **OWNS**: `benchmarks/`
- **MUST NOT WRITE**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `workloads/`, `tileops/manifest/`
- **MUST NOT**: import from `tests/`, or import ref/oracle functions from `workloads/`. Reads of any other file are unrestricted.
- **MUST NOT**: author `gen_inputs` — import the op's workload from `workloads/`; if it has none, add it there in its own PR. Subclassing an imported workload to attach a baseline is fine; name the baseline `torch_baseline` / `<vendor>_baseline`, never `ref_program`.

Enforced by `benchmarks/tests/test_benchmark_import_boundaries.py`.

→ [trust-model.md §Benchmark](../../docs/design/trust-model.md#benchmark) | [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks)

______________________________________________________________________

- Every benchmark records ≥1 non-`tileops` baseline. If the external baseline is conditional, add a local torch fallback.
- Tag names: lowercase, hyphen-separated. Tags starting with `tileops` are TileOPs entries; everything else is a baseline.
- `calculate_flops()` / `calculate_memory()` return `None` to omit the metric.
- Benchmark shapes reflect real DNN workloads (LLaMA-family by default). Annotate shape constants with the model/scenario; never use arbitrary flat numbers (262K, 1M, 4M).
