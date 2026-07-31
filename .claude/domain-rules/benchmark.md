## Boundary

- **OWNS**: `benchmarks/`
- **MUST NOT WRITE**: `tileops/ops/`, `tileops/kernels/`, `tests/`, `workloads/`, `tileops/manifest/`
- **MUST NOT**: import from `tests/`, or import ref/oracle functions from `workloads/`. Reads of any other file are unrestricted.
- **MUST NOT**: author `gen_inputs`. Import the op's workload class from `workloads/`. If none exists, that is a test-stage gap — open a `workloads/`-only PR, do not copy the inputs locally. Subclassing an imported workload to attach a baseline method is allowed.
- Name benchmark-local baselines `torch_baseline` / `<vendor>_baseline`. `ref_program` names the test stage's correctness oracle and must not appear in `benchmarks/`.

Enforced by `benchmarks/tests/test_benchmark_import_boundaries.py`.

→ [trust-model.md §Benchmark](../../docs/design/trust-model.md#benchmark) | [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks)

______________________________________________________________________

- Every benchmark records ≥1 non-`tileops` baseline. If the external baseline is conditional, add a local torch fallback.
- Tag names: lowercase, hyphen-separated. Tags starting with `tileops` are TileOPs entries; everything else is a baseline.
- `calculate_flops()` / `calculate_memory()` return `None` to omit the metric.
- Benchmark shapes reflect real DNN workloads (LLaMA-family by default). Annotate shape constants with the model/scenario; never use arbitrary flat numbers (262K, 1M, 4M).
